"""Dev-only ablation of RAG context size (top-k, token budget) and a cross-encoder reranker.

Reuses the frozen research_v1 dataset, Dense retriever and RAG T5 checkpoint without retraining.
The held-out test split stays locked: results here are dev diagnostics, not new test claims.

    python -m experiments.research_ablation                 # full dev run (200 QA)
    python -m experiments.research_ablation --samples 5     # smoke run
"""
import argparse
import gzip
import importlib.metadata
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ragqa.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR, resolve_project_path
from .metrics import contains_answer, normalize_answer, reference_answers, score_answer
from .research_artifacts import source_snapshot
from .research_data import json_write
from .verify_dataset import digest, read_jsonl

RERANKER = {'name': 'cross-encoder/ms-marco-MiniLM-L6-v2',
            'revision': '233902d25c440f23af6f7d6e94d2946bac0bee0a',
            'local_dir': 'checkpoints/rerankers/ms-marco-MiniLM-L6-v2'}
BASELINE = 'dense_top2_512'
# (name, ranking, passages given to T5, T5 input token budget)
CONFIGS = [
    ('dense_top1_512', 'dense', 1, 512),
    ('dense_top2_512', 'dense', 2, 512),   # identical to the frozen research_v1 rag_full arm
    ('dense_top3_512', 'dense', 3, 512),
    ('dense_top2_1024', 'dense', 2, 1024),
    ('dense_top3_1024', 'dense', 3, 1024),
    ('rerank_top1_512', 'rerank', 1, 512),
    ('rerank_top2_512', 'rerank', 2, 512),
    ('rerank_top3_512', 'rerank', 3, 512),
]
RETRIEVAL_K = (1, 2, 3, 5, 10)
POLICIES = ('official', 'canonical')


def rerank(ranking, scores):
    """Order by cross-encoder score; ties keep the Dense order. Keeps the Dense cosine for reference."""
    order = sorted(range(len(ranking)), key=lambda i: (-scores[i], ranking[i]['rank']))
    return [{**ranking[i], 'dense_rank': ranking[i]['rank'], 'dense_score': ranking[i]['score'],
             'rank': new_rank, 'score': float(scores[i])} for new_rank, i in enumerate(order, 1)]


def passage_text(passage):
    return f"{passage['title']}\n{passage['text']}"


def visible_fraction(selected, used):
    """Share of selected passage characters that survived the T5 token budget."""
    total = sum(len(p['text']) for p in selected)
    return sum(len(p['text']) for p in used) / total if total else 0.0


def retrieval_summary(records, rankings, k_values=RETRIEVAL_K):
    summary = Counter()
    for record, ranking in zip(records, rankings):
        gold = set(record['gold_doc_ids'])
        for k in k_values:
            top = ranking[:k]
            found = set().union(*(set(p.get('source_doc_ids', [p['doc_id']])) for p in top))
            summary[f'doc_hit_rate@{k}'] += bool(found & gold)
            for policy in POLICIES:
                answers = reference_answers(record, policy)
                summary[f'{policy}_answer_recall@{k}'] += any(contains_answer(p['text'], answers) for p in top)
    keys = [f'{metric}@{k}' for metric in ('doc_hit_rate', *(f'{p}_answer_recall' for p in POLICIES))
            for k in k_values]
    return {key: summary[key] / len(records) for key in keys}


def error_category(answer, record, ranking, context, corpus_support, policy):
    """Same lexical categories as research_evaluate; ranking is the top-5 of the arm's own ranking."""
    if score_answer(answer, record, policy)['em']:
        return 'correct'
    answers = reference_answers(record, policy)
    if contains_answer(context, answers):
        return 'generation_wrong_with_answer_string_in_context'
    if not corpus_support:
        return 'corpus_answer_string_missing'
    if not any(contains_answer(p['text'], answers) for p in ranking):
        return 'retrieval_answer_string_missing'
    return 'context_selection_or_truncation'


def paired_bootstrap(candidate, baseline, seed, resamples=2000):
    """95% CI of mean(candidate - baseline) over questions."""
    import numpy as np
    delta = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    return {'delta': float(delta.mean()), 'ci95': [float(x) for x in np.quantile(means, [.025, .975])],
            'wins': int((delta > 0).sum()), 'losses': int((delta < 0).sum())}


def corpus_support_fn(documents):
    normalized = {d['doc_id']: normalize_answer(d['text']) for d in documents}
    joined = '\n'.join(normalized.values())

    def support(record, policy):
        patterns = [re.compile(r'(?<!\w)' + re.escape(normalize_answer(a)) + r'(?!\w)')
                    for a in reference_answers(record, policy) if normalize_answer(a)]
        if any(p.search(normalized[d]) for d in record['gold_doc_ids'] if d in normalized for p in patterns):
            return True
        return any(p.search(joined) for p in patterns)
    return support


def summarize(rows, seed):
    names = list(rows[0]['arms'])
    qa = {name: {policy: {metric: sum(r['arms'][name]['scores'][policy][metric] for r in rows) / len(rows)
                          for metric in ('em', 'f1')} for policy in POLICIES} for name in names}
    context = {name: {'mean_input_tokens': sum(r['arms'][name]['input_tokens'] for r in rows) / len(rows),
                      'mean_visible_fraction': sum(r['arms'][name]['visible_fraction'] for r in rows) / len(rows)}
               for name in names}
    errors = {name: {policy: dict(Counter(r['arms'][name]['errors'][policy] for r in rows)) for policy in POLICIES}
              for name in names}
    versus = {name: {policy: {metric: paired_bootstrap(
                [r['arms'][name]['scores'][policy][metric] for r in rows],
                [r['arms'][BASELINE]['scores'][policy][metric] for r in rows], seed)
              for metric in ('em', 'f1')} for policy in POLICIES} for name in names if name != BASELINE}
    return {'n': len(rows), 'baseline': BASELINE, 'qa': qa, 'context': context,
            'error_breakdown': errors, f'paired_bootstrap_vs_{BASELINE}': versus}


def frozen_agreement(rows, data_hash, rag_sha):
    """Check that the baseline arm reproduces the frozen dev rag_full answers."""
    for path in sorted(RESULTS_DIR.glob('research_*/manifest.json')):
        try:
            info = json.loads(path.read_text(encoding='utf8'))
        except (OSError, ValueError):
            continue
        if (info.get('status') == 'complete' and info.get('split') == 'dev' and not info.get('pilot')
                and info.get('dataset_sha256') == data_hash and info.get('rag_model_sha256') == rag_sha):
            with gzip.open(path.parent / 'predictions.jsonl.gz', 'rt', encoding='utf8') as stream:
                frozen = {}
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        frozen[row['record']['id']] = row['systems']['rag_full']['answer']
            same = sum(frozen.get(r['id']) == r['arms'][BASELINE]['answer'] for r in rows)
            return {'run': path.parent.name, 'compared': len(rows), 'identical_answers': same}
    return None


def write_report(out, metrics, retrieval, agreement):
    lines = ['# Ablation top-k và reranker trên dev (research_v1)', '',
             f"n = {metrics['n']} câu dev. Mốc so sánh: `{BASELINE}` (đúng cấu hình rag_full đã chốt). "
             'Không train lại; tập test vẫn khóa.', '',
             '## EM/F1 của T5 RAG', '',
             '| Cấu hình | Official EM | Official F1 | Canonical EM | Canonical F1 | Token vào | % passage T5 thấy | ΔEM official [CI 95%] |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    boot = metrics[f'paired_bootstrap_vs_{BASELINE}']
    for name, values in metrics['qa'].items():
        o, c, ctx = values['official'], values['canonical'], metrics['context'][name]
        delta = '—' if name == BASELINE else (
            '{:+.1f} [{:+.1f}; {:+.1f}]'.format(100 * boot[name]['official']['em']['delta'],
                                                *(100 * x for x in boot[name]['official']['em']['ci95'])))
        lines.append(f"| {name} | {100*o['em']:.2f}% | {100*o['f1']:.2f}% | {100*c['em']:.2f}% | {100*c['f1']:.2f}% "
                     f"| {ctx['mean_input_tokens']:.0f} | {100*ctx['mean_visible_fraction']:.0f}% | {delta} |")
    lines += ['', '## Retrieval: Dense so với Dense + reranker (pool 30)', '',
              '| Chỉ số | Dense | Rerank |', '| --- | ---: | ---: |']
    for key in retrieval['dense']:
        lines.append(f"| {key} | {100*retrieval['dense'][key]:.1f}% | {100*retrieval['rerank'][key]:.1f}% |")
    lines += ['', '## Phân rã lỗi (official)', '']
    categories = sorted({k for v in metrics['error_breakdown'].values() for k in v['official']})
    lines += ['| Cấu hình | ' + ' | '.join(categories) + ' |', '| --- |' + ' ---: |' * len(categories)]
    for name, values in metrics['error_breakdown'].items():
        lines.append(f'| {name} | ' + ' | '.join(str(values['official'].get(k, 0)) for k in categories) + ' |')
    if agreement:
        lines += ['', f"Kiểm tra tái lập: `{BASELINE}` cho {agreement['identical_answers']}/{agreement['compared']} "
                  f"đáp án trùng với run đã chốt `{agreement['run']}`."]
    lines += ['', 'Giới hạn: kết quả dev, một seed, không phải bằng chứng test độc lập. '
              'CI là paired bootstrap 2.000 lần theo câu hỏi. Phân rã lỗi là proxy từ vựng.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


def main():
    from .research_pipeline import dataset_manifest, ResearchRetriever, ResearchGenerator
    from .research_evaluate import model_run
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR / 'research_v1')
    parser.add_argument('--rag', type=Path)
    parser.add_argument('--reranker', type=Path, default=Path(RERANKER['local_dir']))
    parser.add_argument('--samples', type=int)
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.samples is not None and args.samples < 1:
        parser.error('--samples must be positive')
    data_dir = resolve_project_path(args.data_dir)
    info = dataset_manifest(data_dir)
    config, data_hash = info['config'], digest(data_dir / 'manifest.json')
    rag_path = resolve_project_path(args.rag or PROJECT_ROOT / 'checkpoints' / config['version'] / 'rag')
    rag_info = model_run(rag_path, data_hash, 'rag')
    if rag_info['pilot']:
        raise ValueError('Ablation requires the completed non-pilot RAG checkpoint')
    reranker_dir = resolve_project_path(args.reranker)
    if not (reranker_dir / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Missing reranker at {reranker_dir}; download {RERANKER["name"]}@{RERANKER["revision"]}')
    pilot = args.samples is not None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = resolve_project_path(args.run_dir or RESULTS_DIR / f"research_v1_ablation_dev{'_pilot' if pilot else ''}_{stamp}")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(), 'split': 'dev',
                'pilot': pilot, 'dataset_sha256': data_hash, 'rag_model_sha256': rag_info['model_sha256'],
                'configs': [dict(zip(('name', 'ranking', 'passages', 'max_input_tokens'), c)) for c in CONFIGS],
                'baseline': BASELINE, 'reranker': {**RERANKER, 'model_sha256': digest(reranker_dir / 'model.safetensors'),
                                                   'pool': config['retrieval']['pool'], 'input': 'title + newline + chunk text'},
                'generation': config['generation'], 'seed': config['seed'],
                'source_sha256': {name: digest(PROJECT_ROOT / name) for name in (
                    'experiments/research_ablation.py', 'experiments/research_pipeline.py', 'experiments/metrics.py')},
                'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in
                    ('torch', 'transformers', 'sentence-transformers', 'faiss-cpu')},
                'source_snapshot_sha256': source_snapshot(out / 'source_snapshot.zip'),
                'scope': 'Dev diagnostics only; the locked test split is not evaluated.'}
    json_write(out / 'manifest.json', manifest)
    try:
        import gc
        import torch
        from sentence_transformers import CrossEncoder
        from transformers import set_seed
        set_seed(config['seed'])
        records = read_jsonl(data_dir / 'dev_qa.jsonl')
        if args.samples:
            records = records[:args.samples]
        manifest['n'] = len(records)
        support = corpus_support_fn(read_jsonl(data_dir / 'documents.jsonl'))
        corpus = [{p: support(r, p) for p in POLICIES} for r in records]
        print('Dense retrieval...', flush=True)
        retriever = ResearchRetriever(data_dir, with_sparse=False)
        dense = retriever.dense_many([r['question'] for r in records], config['retrieval']['pool'])
        del retriever
        print('Cross-encoder reranking...', flush=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        encoder = CrossEncoder(str(reranker_dir), device=device, max_length=512)
        reranked = [rerank(ranking, encoder.predict([(r['question'], passage_text(p)) for p in ranking],
                                                    batch_size=32, show_progress_bar=False).tolist())
                    for r, ranking in zip(records, dense)]
        del encoder; gc.collect(); torch.cuda.empty_cache()
        rankings = {'dense': dense, 'rerank': reranked}
        retrieval = {name: retrieval_summary(records, ranks) for name, ranks in rankings.items()}
        print('Generating with the frozen RAG T5...', flush=True)
        generator = ResearchGenerator(rag_path / 'model', config['generation'])
        rows = []
        with gzip.open(out / 'predictions.jsonl.gz', 'wt', encoding='utf8') as stream:
            for i, record in enumerate(records):
                arms = {}
                for name, ranking_name, passages, budget in CONFIGS:
                    ranking = rankings[ranking_name][i]
                    selected = ranking[:passages]
                    generator.config = {**config['generation'], 'max_input_tokens': budget}
                    output = generator.answer(record['question'], selected)
                    output['visible_fraction'] = visible_fraction(selected, output['used_passages'])
                    output['scores'] = {p: score_answer(output['answer'], record, p) for p in POLICIES}
                    output['errors'] = {p: error_category(output['answer'], record,
                        ranking[:config['retrieval']['top_k']], output['context_used'], corpus[i][p], p) for p in POLICIES}
                    arms[name] = output
                row = {'id': record['id'], 'question': record['question'], 'arms': arms,
                       'rerank_top5': [{k: p[k] for k in ('chunk_id', 'rank', 'dense_rank', 'score')} for p in reranked[i][:5]]}
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
                print(f"[{i + 1}/{len(records)}] " + ' '.join(
                    f"{n.split('_')[0][0]}{n.split('_')[1][3:]}/{n.split('_')[2]}={a['scores']['official']['em']}"
                    for n, a in arms.items()), flush=True)
        metrics = {**summarize(rows, config['seed']), 'retrieval': retrieval}
        agreement = frozen_agreement(rows, data_hash, rag_info['model_sha256'])
        metrics['frozen_baseline_agreement'] = agreement
        json_write(out / 'metrics.json', metrics)
        write_report(out, metrics, retrieval, agreement)
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                        predictions_sha256=digest(out / 'predictions.jsonl.gz'))
        print(json.dumps({name: values['official'] for name, values in metrics['qa'].items()}, indent=2), flush=True)
        print(f'Report: {out / "REPORT.md"}', flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        json_write(out / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
