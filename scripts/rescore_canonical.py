"""Rescore an existing saved run against canonical answers; never rerun or overwrite predictions."""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.evaluate import retrieval_summary
from experiments.metrics import classify_error, normalize_answer, score_answer
from experiments.verify_dataset import digest, read_jsonl
from ragqa.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=RESULTS_DIR/'evaluation_v2_full')
    parser.add_argument('--output', type=Path, default=RESULTS_DIR/'canonical_rescore_v1')
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR/'verified_v2')
    args = parser.parse_args()
    source_manifest = json.loads((args.source/'manifest.json').read_text(encoding='utf8'))
    if source_manifest.get('status') != 'complete':
        raise ValueError('Source run is not complete')
    corpus = read_jsonl(args.data_dir/'chunk_metadata.jsonl')
    test = read_jsonl(args.data_dir/'test_qa.jsonl')
    expected_hashes = source_manifest['sha256']
    for path in [args.data_dir/'chunk_metadata.jsonl',args.data_dir/'test_qa.jsonl']:
        key = 'data/verified_v2/' + path.name
        if digest(path) != expected_hashes[key]:
            raise ValueError(f'Dataset checksum mismatch: {path}')
    predictions = read_jsonl(args.source/'predictions.jsonl')
    if len(test) != len(predictions):
        raise ValueError('Prediction count does not match test data')
    if args.output.exists():
        raise FileExistsError(f'Output exists: {args.output}')
    normalized_corpus = [normalize_answer(row['text']) for row in corpus]
    rescored = []
    alias_rows = []
    for record, prediction in zip(test, predictions):
        if record['question'] != prediction['question'] or record['answer'] != prediction['answer']:
            raise ValueError('Prediction ordering does not match test data')
        strict_record = {**record, 'answer_aliases': []}
        strict_prediction = {**prediction, 'answer_aliases': [record['answer']]}
        strict_prediction['systems'] = {name: {**arm, **score_answer(arm['answer'], strict_record)}
                                        for name,arm in prediction['systems'].items()}
        pattern = re.compile(r'(?<!\w)' + re.escape(normalize_answer(record['answer'])) + r'(?!\w)')
        corpus_has_answer = any(pattern.search(text) for text in normalized_corpus)
        rag = strict_prediction['systems']['rag_full']
        strict_prediction['corpus_has_canonical_answer'] = corpus_has_answer
        strict_prediction['error_category'] = classify_error(rag['answer'], strict_record,
            prediction['retrieval']['dense'][:source_manifest['top_k']], rag['context_used'], corpus_has_answer)
        rescored.append(strict_prediction)
        for alias in record.get('answer_aliases', []):
            if normalize_answer(alias) != normalize_answer(record['answer']):
                alias_rows.append({'question_id': record.get('source_question_id',''), 'question': record['question'],
                    'canonical_answer': record['answer'], 'candidate_alias': alias,
                    'decision': '', 'reviewer': '', 'reason': ''})
    retrieval = retrieval_summary(rescored)
    for values in retrieval.values():
        for key in list(values):
            if key.startswith('answer_alias_recall@'):
                values[key.replace('answer_alias_recall@', 'canonical_answer_recall@')] = values.pop(key)
    metrics = {'n': len(rescored), 'answer_policy': 'canonical_only',
        'note': 'Conservative exact canonical reference; may mark valid variants or spelling corrections incorrect.',
        'qa': {name: {metric: sum(r['systems'][name][metric] for r in rescored)/len(rescored)
                      for metric in ('em','f1')}
               for name in ('generator_only','retrieval_only','rag_full')},
        'retrieval': retrieval,
        'error_breakdown': dict(Counter(r['error_category'] for r in rescored))}
    args.output.mkdir(parents=True)
    (args.output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    # Store only changed scores, rather than duplicate 17 MB of contexts.
    with (args.output/'scores.jsonl').open('w',encoding='utf8') as f:
        for r in rescored:
            f.write(json.dumps({'id':r['id'],'question':r['question'],'answer':r['answer'],
                'systems': {name:{key:arm[key] for key in ('answer','em','f1')} for name,arm in r['systems'].items()},
                'error_category':r['error_category'],'corpus_has_canonical_answer':r['corpus_has_canonical_answer']},ensure_ascii=False)+'\n')
    with (args.output/'alias_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['question_id','question','canonical_answer','candidate_alias','decision','reviewer','reason'])
        writer.writeheader(); writer.writerows(alias_rows)
    manifest={'created_at':datetime.now(timezone.utc).isoformat(),'source_run':args.source.name,
        'prediction_sha256':digest(args.source/'predictions.jsonl'), 'test_sha256':digest(args.data_dir/'test_qa.jsonl'),
        'corpus_sha256':digest(args.data_dir/'chunk_metadata.jsonl'), 'answers':'canonical_only',
        'model_inference':'not rerun; scores derived from saved predictions', 'n':len(rescored),
        'candidate_aliases_to_review':len(alias_rows),
        'scoring_code_sha256': {path.relative_to(PROJECT_ROOT).as_posix():digest(path) for path in
            [Path(__file__).resolve(), Path(__file__).resolve().parents[1]/'experiments/metrics.py']}}
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
    print('Alias candidates for human review:',len(alias_rows))


if __name__=='__main__':
    main()
