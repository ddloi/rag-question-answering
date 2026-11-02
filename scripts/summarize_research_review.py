"""Validate blinded annotations; keep human and agent judgments separate."""
import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from experiments.research_data import json_write
from experiments.verify_dataset import digest


def summarize(records, mapping):
    summary = defaultdict(lambda: {'completed': 0, 'faithfulness': Counter(), 'citations': Counter()})
    incomplete, seen, completed_ids, human_ids = [], set(), set(), set()
    for row in records:
        review_id = row['review_id']
        if review_id not in mapping:
            raise ValueError(f'Unknown review ID: {review_id}')
        fields = ('faithfulness', 'citation_labels_json', 'rationale', 'reviewer_id', 'reviewer_kind')
        if not all(row.get(key, '').strip() for key in fields):
            incomplete.append(review_id); continue
        expected_hash = mapping[review_id].get('payload_sha256')
        if expected_hash:
            payload = {field: row[field] for field in ('question', 'prediction', 'context_used', 'source_chunk_ids')}
            if hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest() != expected_hash:
                raise ValueError('Review question/prediction/evidence was modified')
        kind = row['reviewer_kind']
        if kind not in ('human_independent', 'human_author', 'agent_self_review'):
            raise ValueError('Invalid reviewer_kind')
        if row['faithfulness'] not in ('supported', 'unsupported', 'unclear'):
            raise ValueError('Invalid faithfulness label')
        identity = (review_id, row['reviewer_id'])
        if identity in seen:
            raise ValueError('Duplicate annotation by the same reviewer')
        seen.add(identity)
        labels = json.loads(row['citation_labels_json'])
        expected = set(filter(None, row['source_chunk_ids'].split(';')))
        if not isinstance(labels, dict) or set(labels) != expected:
            raise ValueError(f'Citation labels must cover exactly the actual source chunks: {review_id}')
        if any(label not in ('supports_answer', 'does_not_support', 'unclear') for label in labels.values()):
            raise ValueError('Invalid citation label')
        group = summary[kind + '/' + mapping[review_id]['system']]
        group['completed'] += 1
        group['faithfulness'][row['faithfulness']] += 1
        group['citations'].update(labels.values())
        completed_ids.add(review_id)
        if kind == 'human_independent':
            human_ids.add(review_id)
    groups = {}
    for key, group in summary.items():
        assessed = group['citations']['supports_answer'] + group['citations']['does_not_support']
        faith_assessed = group['faithfulness']['supported'] + group['faithfulness']['unsupported']
        groups[key] = {**group, 'citation_precision_assessed': group['citations']['supports_answer'] / assessed if assessed else None,
                       'citation_unclear': group['citations']['unclear'],
                       'faithfulness_supported_fraction_all': group['faithfulness']['supported'] / group['completed'],
                       'faithfulness_supported_fraction_assessed': group['faithfulness']['supported'] / faith_assessed if faith_assessed else None,
                       'faithfulness_unclear': group['faithfulness']['unclear']}
    incomplete = sorted(set(incomplete) | (set(mapping) - completed_ids))
    return {'status': 'NOT RUN' if not groups else 'PARTIAL' if incomplete else 'COMPLETE',
            'independent_human_status': 'NOT RUN' if not human_ids else 'COMPLETE' if human_ids == set(mapping) else 'PARTIAL',
            'metric_unit': 'annotation and cited source; multiple reviewers are not additional independent QA samples',
            'groups': groups, 'incomplete_review_ids': incomplete,
            'note': 'Human-independent, human-author and agent-self-review counts are separate; no empty form is treated as a scored evaluation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--key', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.reviews.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    key = json.loads(args.key.read_text(encoding='utf8'))
    result = summarize(rows, {r['review_id']: r for r in key['mapping']})
    result.update(reviews_sha256=digest(args.reviews), key_sha256=digest(args.key),
                  aggregator_sha256=digest(Path(__file__)))
    json_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
