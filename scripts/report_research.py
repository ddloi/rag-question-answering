"""Recompute scores from saved predictions and export traceable report cases."""
import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from experiments.metrics import score_answer
from experiments.research_data import json_write
from experiments.research_evaluate import retrieval_metrics
from experiments.verify_dataset import digest
from ragqa.paths import resolve_project_path

NAMES = {'generator_only': 'T5 closed-book (fine-tune riêng)', 'retrieval_only': 'Dense + DistilBERT extractive',
         'rag_full': 'Dense + T5 RAG', 'rag_hybrid': 'Hybrid RRF + cùng T5 RAG',
         'rag_checkpoint_no_context': 'Ablation: checkpoint RAG bỏ context'}


def recompute(rows):
    result = {}
    if not rows or len({r['record']['id'] for r in rows}) != len(rows):
        raise ValueError('Empty predictions or duplicate QA IDs')
    for arm in rows[0]['systems']:
        result[arm] = {}
        for policy in ('official', 'canonical'):
            scores = [score_answer(r['systems'][arm]['answer'], r['record'], policy) for r in rows]
            result[arm][policy] = {metric: sum(s[metric] for s in scores) / len(scores) for metric in ('em', 'f1')}
    return result


def check_numbers(actual, expected, path='metrics'):
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError(f'Metric keys differ: {path}')
        for key in actual:
            check_numbers(actual[key], expected[key], path + '/' + key)
    elif not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f'Metric mismatch: {path}')


def case_records(rows):
    cases = []
    for row in rows:
        record = row['record']
        dense = row['systems']['rag_full']
        hybrid = row['systems']['rag_hybrid']
        direction = ('hybrid_rescue' if hybrid['scores']['canonical']['em'] > dense['scores']['canonical']['em'] else
                     'hybrid_regression' if hybrid['scores']['canonical']['em'] < dense['scores']['canonical']['em'] else 'same_em')
        cases.append({'question_id': record['id'], 'question': record['question'], 'canonical': record['answer'],
            **{arm: row['systems'][arm]['answer'] for arm in NAMES},
            'dense_error_proxy': row['errors']['rag_full']['canonical'], 'hybrid_error_proxy': row['errors']['rag_hybrid']['canonical'],
            'hybrid_direction': direction,
            'dense_canonical_em': dense['scores']['canonical']['em'], 'dense_canonical_f1': dense['scores']['canonical']['f1'],
            'dense_official_em': dense['scores']['official']['em'], 'dense_official_f1': dense['scores']['official']['f1'],
            'answer_policy_disagreement': dense['scores']['canonical']['em'] != dense['scores']['official']['em'],
            'dense_used_chunks': ';'.join(p['chunk_id'] for p in dense['used_passages']),
            'hybrid_used_chunks': ';'.join(p['chunk_id'] for p in hybrid['used_passages'])})
    return cases


def select_examples(cases, per_category=2):
    selected, counts = [], Counter()
    for row in cases:
        categories = [row['dense_error_proxy'], row['hybrid_direction']]
        if row.get('answer_policy_disagreement'):
            categories.append('official_canonical_disagreement')
        for category in categories:
            if category == 'same_em' or counts[category] >= per_category:
                continue
            selected.append({'category': category, **row})
            counts[category] += 1
    return selected


def cell(text):
    return str(text).replace('|', '\\|').replace('\n', ' ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    out = resolve_project_path(args.run_dir)
    if (out / 'REPORT.md').exists():
        raise FileExistsError('Preserving existing report')
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf8'))
    metrics = json.loads((out / 'metrics.json').read_text(encoding='utf8'))
    if manifest['status'] != 'complete' or digest(out / 'predictions.jsonl.gz') != manifest['predictions_sha256']:
        raise ValueError('Incomplete run or altered predictions')
    with gzip.open(out / 'predictions.jsonl.gz', 'rt', encoding='utf8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != manifest['n'] or len(rows) != metrics['n']:
        raise ValueError('Prediction count mismatch')
    qa = recompute(rows)
    retrieval = retrieval_metrics(rows, manifest['config']['evaluation']['retrieval_k'])
    check_numbers(qa, metrics['qa']); check_numbers(retrieval, metrics['retrieval'])
    cases = case_records(rows)
    with (out / 'cases.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0])); writer.writeheader(); writer.writerows(cases)
    examples = select_examples(cases)
    json_write(out / 'case_examples.json', {'selection': 'First two per available lexical error class and Hybrid direction, in frozen split order; illustrative, not random inference.', 'cases': examples})
    json_write(out / 'score_recomputation.json', {'status': 'PASS', 'n': len(rows), 'tolerance': 1e-12,
        'predictions_sha256': manifest['predictions_sha256'], 'qa_and_retrieval_recomputed': True,
        'report_script_sha256': digest(Path(__file__))})
    lines = [f"# Kết quả research_v1 · {manifest['split']} · {len(rows)} QA", '',
        'Số liệu được tính lại từ dự đoán đã lưu và đối chiếu với metrics.json (sai số tối đa 1e-12).', '',
        f"Dataset manifest SHA256: `{manifest['dataset_sha256']}`.",
        f"Closed-book model: `{manifest['closed_book_model_sha256']}`; RAG model: `{manifest['rag_model_sha256']}`.", '',
        '| Hệ | Official EM | Official F1 | Canonical EM | Canonical F1 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for arm, name in NAMES.items():
        scores = qa[arm]
        values = [scores[p][m] * 100 for p in ('official', 'canonical') for m in ('em', 'f1')]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{v:.2f}%' for v in values) + ' |')
    lines += ['', '## Retrieval', '', '| Phương pháp | Doc hit@5 | Doc recall@5 | Official answer recall@5 | Canonical answer recall@5 | MRR@30 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for method, scores in retrieval.items():
        keys = ('doc_hit_rate@5', 'doc_recall@5', 'official_answer_recall@5', 'canonical_answer_recall@5', 'doc_mrr@30')
        lines.append('| ' + method + ' | ' + ' | '.join(f'{scores[k]*100:.2f}%' for k in keys) + ' |')
    lines += ['', 'Doc hit khác doc recall khi một câu có nhiều gold sources. Answer recall chỉ dò chuỗi; nguồn qrels là document-level. Đủ k=1/3/5/10 trong metrics.json.', '', '## Hybrid so với Dense', '']
    for policy, interval in metrics['paired_bootstrap'].items():
        for metric in ('em', 'f1'):
            value = interval[metric]
            lo, hi = value['paired_bootstrap_95_ci']
            lines.append(f"- {policy} {metric.upper()}: chênh {value['hybrid_minus_dense']*100:+.2f} điểm phần trăm; paired bootstrap 95% CI [{lo*100:+.2f}, {hi*100:+.2f}].")
    lines += ['', 'CI theo câu hỏi; chưa thể hiện phương sai nhiều training seed. Không diễn giải Hybrid luôn tốt hơn.', '', '## Phân rã lỗi lexical', '', '| Nhánh | Nhóm | Số câu |', '| --- | --- | ---: |']
    for arm in ('rag_full', 'rag_hybrid'):
        for category, count in metrics['error_breakdown'][arm]['canonical'].items():
            lines.append(f'| {arm} | {category} | {count} |')
    lines += ['', '## Ví dụ phục vụ báo cáo', '', 'Các nhóm dưới là proxy, cần đọc context để xác nhận ngữ nghĩa. Mỗi nhóm lấy tối đa hai câu đầu theo thứ tự split; không đại diện tỷ lệ toàn bộ dữ liệu.', '', '| Nhóm | Question ID | Câu hỏi | Canonical | Dense RAG | Hybrid RAG |', '| --- | --- | --- | --- | --- | --- |']
    for row in examples:
        lines.append('| ' + ' | '.join(cell(row[k]) for k in ('category','question_id','question','canonical','rag_full','rag_hybrid')) + ' |')
    lines += ['', 'Tra question_id trong predictions.jsonl.gz để xem passage, source IDs và context thực dùng. cases.csv chứa toàn bộ câu để lọc đúng/sai/Hybrid tốt hơn hoặc kém hơn.', '',
        '## Giới hạn và phần cần bổ sung', '',
        '- Corpus nhỏ, cùng miền; shared documents có thể xuất hiện giữa các split, pretraining contamination chưa kiểm soát.',
        '- MiniLM cố định; T5 context nối chuỗi, không tái lập end-to-end RAG-Sequence/RAG-Token.',
        '- Một seed, hai T5 cùng QA và budget, extractive dùng checkpoint SQuAD sẵn có.',
        '- Faithfulness/citation người chấm: xem phiếu và summary riêng. Phiếu trống nghĩa NOT RUN; điểm EM/F1 không thay được nhãn ngữ nghĩa.',
        '- Chạy Colab thực tế: không được suy ra từ việc notebook qua kiểm tra cú pháp.',
        '- Split dev dùng phát triển và ví dụ demo; chỉ test đã khóa protocol mới là held-out evaluation.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(json.dumps({'status': 'PASS', 'n': len(rows), 'report': str(out / 'REPORT.md'), 'examples': len(examples)}, indent=2))


if __name__ == '__main__':
    main()
