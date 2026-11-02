"""Evaluate three trained systems and Hybrid under the frozen research protocol."""
import argparse
import csv
import gc
import gzip
import importlib.metadata
import json
import platform
import random
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ragqa.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, resolve_project_path
from .metrics import contains_answer, normalize_answer, reference_answers, score_answer
from .research_data import json_write
from .research_pipeline import dataset_manifest, ResearchRetriever, ResearchGenerator, ResearchExtractor
from .verify_dataset import digest, read_jsonl
from .research_artifacts import source_snapshot


def source_ids(passage):
    return set(passage.get('source_doc_ids', [passage['doc_id']]))


def retrieval_metrics(rows, k_values):
    summary = {}
    for method in ('dense', 'bm25', 'hybrid'):
        measures = Counter()
        for row in rows:
            record, ranking = row['record'], row['retrieval'][method]
            gold = set(record['gold_doc_ids'])
            for k in k_values:
                found = set().union(*(source_ids(p) for p in ranking[:k]))
                measures[f'doc_hit_rate@{k}'] += bool(found & gold)
                measures[f'doc_recall@{k}'] += len(found & gold) / len(gold) if gold else 0
                for policy in ('official', 'canonical'):
                    answers = reference_answers(record, policy)
                    measures[f'{policy}_answer_recall@{k}'] += any(contains_answer(p['text'], answers) for p in ranking[:k])
            measures['doc_mrr@30'] += next((1 / rank for rank, p in enumerate(ranking, 1) if source_ids(p) & gold), 0)
        summary[method] = {key: value / len(rows) for key, value in measures.items()}
    return summary


def error_category(answer, record, ranking, context, corpus_support, policy):
    if score_answer(answer, record, policy)['em']:
        return 'correct'
    answers = reference_answers(record, policy)
    # A visible heading can contain an answer absent from the passage body.
    # Prefer what the generator actually saw over corpus/body-only diagnostics.
    if contains_answer(context, answers):
        return 'generation_wrong_with_answer_string_in_context'
    if not corpus_support:
        return 'corpus_answer_string_missing'
    if not any(contains_answer(p['text'], answers) for p in ranking):
        return 'retrieval_answer_string_missing'
    return 'context_selection_or_truncation'


def bootstrap_difference(rows, policy, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(2000, len(rows)))
    result = {}
    for metric in ('em', 'f1'):
        delta = np.array([r['systems']['rag_hybrid']['scores'][policy][metric] -
                          r['systems']['rag_full']['scores'][policy][metric] for r in rows])
        means = delta[indices].mean(axis=1)
        result[metric] = {'hybrid_minus_dense': float(delta.mean()),
                         'paired_bootstrap_95_ci': [float(x) for x in np.quantile(means, [.025, .975])]}
    return {'unit': 'question', 'resamples': 2000, 'seed': seed,
            'scope': 'Question-level sampling uncertainty; not training-seed variance or unseen-document generalization.', **result}


def export_review(rows, out, size, seed):
    selection = sorted(random.Random(seed).sample(range(len(rows)), min(size, len(rows))))
    randomizer = random.Random(seed + 1)
    items, key = [], []
    for idx in selection:
        row = rows[idx]
        arms = ['rag_full', 'rag_hybrid']
        randomizer.shuffle(arms)
        for arm in arms:
            review_id = f'review_{len(items) + 1:03d}'
            output = row['systems'][arm]
            items.append({'review_id': review_id, 'question': row['record']['question'],
                'prediction': output['answer'], 'context_used': output['context_used'],
                'source_chunk_ids': ';'.join(p['chunk_id'] for p in output['used_passages']),
                'faithfulness': '', 'citation_labels_json': '', 'rationale': '',
                'reviewer_id': '', 'reviewer_kind': ''})
            immutable = {field: items[-1][field] for field in ('question', 'prediction', 'context_used', 'source_chunk_ids')}
            key.append({'review_id': review_id, 'question_id': row['record']['id'], 'system': arm,
                'payload_sha256': hashlib.sha256(json.dumps(immutable, sort_keys=True, ensure_ascii=False).encode('utf8')).hexdigest()})
    with (out / 'human_review_blinded.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(items[0])); writer.writeheader(); writer.writerows(items)
    json_write(out / 'review_key.json', {'seed': seed, 'selected_question_ids': [rows[i]['record']['id'] for i in selection], 'mapping': key})


def model_run(path, data_hash, expected_system=None):
    info = json.loads((path / 'manifest.json').read_text(encoding='utf8'))
    if expected_system is not None and info.get('system') != expected_system:
        raise ValueError(f'Wrong model system: expected {expected_system}, got {info.get("system")}')
    if info['status'] != 'complete' or info['dataset_manifest_sha256'] != data_hash:
        raise ValueError(f'Incomplete or incompatible model run: {path}')
    if digest(path / 'model/model.safetensors') != info['model_sha256']:
        raise ValueError(f'Model checksum mismatch: {path}')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR / 'research_v1')
    parser.add_argument('--closed-book', type=Path)
    parser.add_argument('--rag', type=Path)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--final', action='store_true', help='Freeze protocol and evaluate all held-out test rows')
    parser.add_argument('--allow-pilot', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.split == 'test' and (not args.final or args.samples is not None or args.allow_pilot):
        parser.error('Test requires --final, all rows, and completed non-pilot models')
    if args.samples is not None and args.samples < 1:
        parser.error('--samples must be positive')
    data_dir = resolve_project_path(args.data_dir)
    data_info = dataset_manifest(data_dir)
    config = data_info['config']; data_hash = digest(data_dir / 'manifest.json')
    model_root = PROJECT_ROOT / 'checkpoints' / config['version']
    closed_path = resolve_project_path(args.closed_book or model_root / 'closed_book')
    rag_path = resolve_project_path(args.rag or model_root / 'rag')
    closed_info, rag_info = model_run(closed_path, data_hash, 'closed_book'), model_run(rag_path, data_hash, 'rag')
    if closed_info['train_ids'] != rag_info['train_ids'] or closed_info['base_model'] != rag_info['base_model']:
        raise ValueError('Baseline and RAG must share training QA IDs and initialization')
    for key in ('config', 'precision', 'prompt_template', 'source_sha256'):
        if closed_info[key] != rag_info[key]:
            raise ValueError(f'Baseline/RAG training protocol mismatch: {key}')
    if (closed_info['pilot'] or rag_info['pilot']) and not args.allow_pilot:
        raise ValueError('Pilot models require --allow-pilot and a dev-only evaluation')
    out = resolve_project_path(args.run_dir or RESULTS_DIR / ('research_' + args.split + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'split': args.split,
        'final': args.final, 'pilot': closed_info['pilot'] or rag_info['pilot'], 'dataset_sha256': data_hash,
        'config': config, 'closed_book_model_sha256': closed_info['model_sha256'], 'rag_model_sha256': rag_info['model_sha256'],
        'source_sha256': {name: digest(PROJECT_ROOT / name) for name in ('experiments/research_evaluate.py',
            'experiments/research_pipeline.py', 'experiments/metrics.py', 'ragqa/rag_engine.py')},
        'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'sentence-transformers', 'datasets', 'faiss-cpu', 'rank-bm25')},
        'citation_definition': 'Application-provided actual used passages; not model-generated inline citations.',
        'source_snapshot_sha256': source_snapshot(out / 'source_snapshot.zip'),
        'error_definition': 'Lexical proxies; semantic error attribution requires the blinded review.'}
    json_write(out / 'manifest.json', manifest)
    try:
        import torch
        from transformers import set_seed
        set_seed(config['seed'])
        records = read_jsonl(data_dir / f'{args.split}_qa.jsonl')
        if args.samples:
            records = records[:args.samples]
        manifest['n'] = len(records)
        print('Building shared retrieval rankings...', flush=True)
        retriever = ResearchRetriever(data_dir)
        rankings = [retriever.all_rankings(r['question']) for r in records]
        del retriever
        gc.collect()
        documents = read_jsonl(data_dir / 'documents.jsonl')
        normalized_docs = {d['doc_id']: normalize_answer(d['text']) for d in documents}
        joined = '\n'.join(normalized_docs.values())
        import re
        def support(record, policy):
            patterns = [re.compile(r'(?<!\w)' + re.escape(normalize_answer(a)) + r'(?!\w)')
                        for a in reference_answers(record, policy) if normalize_answer(a)]
            if any(p.search(normalized_docs[d]) for d in record['gold_doc_ids'] for p in patterns):
                return True
            return any(p.search(joined) for p in patterns)
        corpus_support = [{p: support(r, p) for p in config['evaluation']['policies']} for r in records]
        del documents, normalized_docs, joined
        print('Generating closed-book baseline...', flush=True)
        generator = ResearchGenerator(closed_path / 'model', config['generation'])
        closed = [generator.answer(r['question']) for r in records]
        del generator; gc.collect(); torch.cuda.empty_cache()
        print('Running extractive baseline...', flush=True)
        extractor = ResearchExtractor(data_info['models']['extractive'])
        extracted = [extractor.answer(r['question'], ranks['dense'][:config['retrieval']['top_k']])
                     for r, ranks in zip(records, rankings)]
        del extractor; gc.collect(); torch.cuda.empty_cache()
        print('Generating Dense RAG, Hybrid RAG, and context ablation...', flush=True)
        generator = ResearchGenerator(rag_path / 'model', config['generation'])
        rows = []
        with gzip.open(out / 'predictions.jsonl.gz', 'wt', encoding='utf8') as stream:
            for i, (record, ranks) in enumerate(zip(records, rankings)):
                systems = {'generator_only': closed[i], 'retrieval_only': extracted[i],
                    'rag_full': generator.answer(record['question'], ranks['dense'][:config['retrieval']['context_top_n']]),
                    'rag_hybrid': generator.answer(record['question'], ranks['hybrid'][:config['retrieval']['context_top_n']]),
                    'rag_checkpoint_no_context': generator.answer(record['question'])}
                for output in systems.values():
                    output['scores'] = {p: score_answer(output['answer'], record, p) for p in config['evaluation']['policies']}
                errors = {arm: {p: error_category(systems[arm]['answer'], record,
                    ranks[method][:config['retrieval']['top_k']], systems[arm]['context_used'], corpus_support[i][p], p)
                    for p in config['evaluation']['policies']} for arm, method in [('rag_full', 'dense'), ('rag_hybrid', 'hybrid')]}
                row = {'record': record, 'retrieval': ranks, 'systems': systems, 'corpus_answer_support': corpus_support[i], 'errors': errors}
                rows.append(row); stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
                print(f"[{i + 1}/{len(records)}] dense EM={systems['rag_full']['scores']['canonical']['em']} hybrid EM={systems['rag_hybrid']['scores']['canonical']['em']}", flush=True)
        metrics = {'n': len(rows), 'qa': {arm: {policy: {metric: sum(r['systems'][arm]['scores'][policy][metric] for r in rows) / len(rows)
                    for metric in ('em', 'f1')} for policy in config['evaluation']['policies']} for arm in rows[0]['systems']},
            'retrieval': retrieval_metrics(rows, config['evaluation']['retrieval_k']),
            'error_breakdown': {arm: {policy: dict(Counter(r['errors'][arm][policy] for r in rows))
                for policy in config['evaluation']['policies']} for arm in ('rag_full', 'rag_hybrid')},
            'paired_bootstrap': {p: bootstrap_difference(rows, p, config['seed']) for p in config['evaluation']['policies']}}
        json_write(out / 'metrics.json', metrics)
        export_review(rows, out, config['evaluation']['review_sample'], config['seed'])
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                        predictions_sha256=digest(out / 'predictions.jsonl.gz'))
        print(json.dumps(metrics['qa'], indent=2), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        json_write(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
