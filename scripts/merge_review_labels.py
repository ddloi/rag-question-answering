"""Merge labels exported by scripts/review_tool.html into a blinded review sheet.

Only the five annotation columns change; question, prediction, context and chunk IDs are copied
unchanged so the payload hashes in review_key.json still verify.

    python -m scripts.merge_review_labels --sheet results/research_v1_test/human_review_author.csv \
        --labels review_labels.csv --key results/research_v1_test/review_key.json
"""
import argparse
import csv
import json
from pathlib import Path

from scripts.summarize_research_review import summarize

LABEL_FIELDS = ('faithfulness', 'citation_labels_json', 'rationale', 'reviewer_id', 'reviewer_kind')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames, list(reader)


def merge(sheet_rows, label_rows):
    by_id = {row['review_id']: row for row in label_rows if row.get('review_id')}
    unknown = set(by_id) - {row['review_id'] for row in sheet_rows}
    if unknown:
        raise ValueError(f'Labels for unknown review IDs: {sorted(unknown)}')
    merged = []
    for row in sheet_rows:
        label = by_id.get(row['review_id'])
        merged.append({**row, **{f: label.get(f, '') for f in LABEL_FIELDS}} if label else dict(row))
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sheet', type=Path, required=True, help='Blinded sheet to update in place')
    parser.add_argument('--labels', type=Path, required=True, help='review_labels.csv from review_tool.html')
    parser.add_argument('--key', type=Path, required=True, help='review_key.json of the same run')
    args = parser.parse_args()
    fields, sheet = read_csv(args.sheet)
    _, labels = read_csv(args.labels)
    merged = merge(sheet, labels)
    key = json.loads(args.key.read_text(encoding='utf8'))
    # Validate before writing: invalid labels or edited evidence never reach the sheet.
    result = summarize(merged, {r['review_id']: r for r in key['mapping']})
    with args.sheet.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    print(json.dumps({'status': result['status'], 'remaining': len(result['incomplete_review_ids'])}))


if __name__ == '__main__':
    main()
