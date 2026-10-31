"""Bounded, read-only research audit; never train, mutate data, or overwrite a run."""
import argparse
import csv
import json
import re
import string
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from experiments.metrics import contains_answer, score_answer
from experiments.verify_dataset import digest, read_jsonl
from ragqa.paths import DATA_DIR, PROJECT_ROOT, RESULTS_DIR


OFFICIAL_EVALUATOR = 'https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py'


def text_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def triviaqa_style_normalize(text):
    """Diagnostic implementation: punctuation becomes spaces, as in TriviaQA v1.0.

    Canonical-only scoring below is NOT the full official alias evaluation.
    """
    punctuation = set(string.punctuation + '‘’´`')
    text = text.lower().replace('_', ' ')
    text = ''.join(' ' if c in punctuation else c for c in text)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def triviaqa_style_score(prediction, canonical):
    pred = triviaqa_style_normalize(prediction)
    gold = triviaqa_style_normalize(canonical)
    common = sum((Counter(pred.split()) & Counter(gold.split())).values())
    denominator = len(pred.split()) + len(gold.split())
    return {'em': int(pred == gold), 'f1': 2 * common / denominator if common else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-tokenizer', action='store_true')
    parser.add_argument('--check-index', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Preserving previous audit: {args.output}')
    raw = json.loads((DATA_DIR / 'raw_documents.json').read_text(encoding='utf8'))
    documents = json.loads((DATA_DIR / 'verified_v2/raw_documents.json').read_text(encoding='utf8'))
    train = read_jsonl(DATA_DIR / 'training_pairs.jsonl')
    test = read_jsonl(DATA_DIR / 'verified_v2/test_qa.jsonl')
    corpus = read_jsonl(DATA_DIR / 'verified_v2/chunk_metadata.jsonl')
    official = {r['doc_id']: r for r in read_jsonl(DATA_DIR / 'provenance/triviaqa_official.jsonl')}
    predictions_path = RESULTS_DIR / 'evaluation_v2_full/predictions.jsonl'
    predictions = read_jsonl(predictions_path)
    manifest = json.loads((predictions_path.parent / 'manifest.json').read_text(encoding='utf8'))
    by_question = {text_key(d['qa'][0]['question']): d for d in documents}
    original_by_question = {text_key(d['qa'][0]['question']): d for d in raw}
    train_questions = {text_key(r['question']) for r in train}
    shared_questions = [r['question'] for r in test if text_key(r['question']) in train_questions]
    training_titles = {text_key(by_question[text_key(r['question'])]['title']) for r in train}
    shared_titles = [r['question'] for r in test
                     if text_key(by_question[text_key(r['question'])]['title']) in training_titles]

    training_checks = []
    for i, row in enumerate(train):
        doc_id = original_by_question[text_key(row['question'])]['id']
        aliases = [row['answer'], *official[doc_id]['answer']['aliases']]
        training_checks.append({'row': i, 'question': row['question'], 'answer': row['answer'],
            'doc_id': doc_id, 'canonical_in_context': bool(contains_answer(row['context'], [row['answer']])),
            'any_official_alias_in_context': bool(contains_answer(row['context'], aliases))})
    tokenizer_check = {'status': 'NOT RUN'}
    if args.check_tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(PROJECT_ROOT / 'generator_finetuned/tokenizer.json'))
        tokenizer.enable_truncation(max_length=384)
        for row, check in zip(train, training_checks):
            prefix = f"question: {row['question']} context: "
            encoded = tokenizer.encode(prefix + row['context'])
            end = max((end for _, end in encoded.offsets), default=0)
            visible = row['context'][:max(0, end - len(prefix))]
            check['canonical_after_384_token_limit'] = bool(contains_answer(visible, [row['answer']]))
        tokenizer_check = {'status': 'PASS', 'method': 'Serialized tokenizer backend, training-format input, max_length=384',
            'missing_canonical_after_truncation': sum(not r['canonical_after_384_token_limit'] for r in training_checks),
            'additional_lost_canonical': sum(r['canonical_in_context'] and not r['canonical_after_384_token_limit'] for r in training_checks)}

    comparison, changes = {}, []
    for name in ('generator_only', 'retrieval_only', 'rag_full'):
        current, alternate = [], []
        for row in predictions:
            prediction = row['systems'][name]['answer']
            original = score_answer(prediction, {'answer': row['answer']})
            diagnostic = triviaqa_style_score(prediction, row['answer'])
            current.append(original); alternate.append(diagnostic)
            if original != diagnostic:
                changes.append({'id': row['id'], 'system': name, 'question': row['question'],
                    'prediction': prediction, 'canonical': row['answer'], 'current': original,
                    'triviaqa_style_canonical_only': diagnostic})
        comparison[name] = {'current_custom_canonical': {m: sum(r[m] for r in current) / len(current) for m in ('em', 'f1')},
            'triviaqa_style_canonical_only': {m: sum(r[m] for r in alternate) / len(alternate) for m in ('em', 'f1')}}

    historical_hashes = {p: digest(PROJECT_ROOT / p) == expected for p, expected in manifest['sha256'].items()}
    index_check = {'status': 'NOT RUN'}
    if args.check_index:
        import faiss
        import numpy as np
        index = faiss.read_index(str(DATA_DIR / 'verified_v2/faiss.index'))
        vectors = index.reconstruct_n(0, index.ntotal)
        norms = np.linalg.norm(vectors, axis=1)
        finite = bool(np.isfinite(vectors).all())
        normalized = bool(np.allclose(norms, 1.0, atol=1e-3, rtol=0))
        index_check = {'status': 'PASS' if finite and normalized and index.ntotal == len(corpus) else 'FAIL',
            'type': type(index).__name__, 'count': index.ntotal, 'dimension': index.d,
            'finite': finite, 'unit_norm_atol': 1e-3, 'unit_norms_pass': normalized,
            'norm_min': float(norms.min()), 'norm_max': float(norms.max())}
    count = len(predictions)
    expected_order = len(test) == count and all(t['question'] == p['question'] for t, p in zip(test, predictions))
    review = json.loads((predictions_path.parent / 'review_sample_agent.json').read_text(encoding='utf8'))
    summary = {'created_at': datetime.now(timezone.utc).isoformat(), 'reviewer_kind': 'implementer_self_review',
        'purpose': 'Research readiness audit; no new model training/inference and no data changes.',
        'counts': {'qa_documents': len(raw), 'training_input_rows': len(train), 'heldout_question_rows': len(test),
            'verified_chunks': len(corpus), 'unique_verified_titles': len({text_key(d['title']) for d in documents})},
        'split': {'exact_normalized_question_overlap': len(shared_questions),
            'test_questions_sharing_source_title_with_training_input': len(shared_titles),
            'interpretation': 'Page overlap does not prove answer leakage, but cannot support unseen-document generalization.'},
        'training_context': {'rows_without_canonical_string': sum(not r['canonical_in_context'] for r in training_checks),
            'rows_without_any_official_alias_string': sum(not r['any_official_alias_in_context'] for r in training_checks),
            'note': 'Lexical diagnostics; semantic support requires manual review. 680 includes the configured inner 10% validation split.',
            'tokenizer': tokenizer_check},
        'duplicates': {'extra_normalized_identical_chunks': len(corpus) - len({text_key(r['text']) for r in corpus}),
            'queries_with_identical_dense_top_two_texts': sum(text_key(r['retrieval']['dense'][0]['text']) == text_key(r['retrieval']['dense'][1]['text']) for r in predictions)},
        'saved_run': {'status': manifest['status'], 'n': count, 'test_prediction_order_matches': expected_order,
            'all_three_systems_present': all(set(r['systems']) == {'generator_only', 'retrieval_only', 'rag_full'} for r in predictions),
            'max_saved_rag_input_tokens': max(r['systems']['rag_full']['input_tokens'] for r in predictions),
            'current_files_match_historical_manifest': historical_hashes},
        'normalization_diagnostic': {'reference': OFFICIAL_EVALUATOR,
            'note': 'Alternate normalization only; canonical references are NOT full official TriviaQA alias evaluation.',
            'qa': comparison, 'changed_system_question_pairs': len(changes)},
        'review_sample': review, 'index': index_check,
        'not_run': ['Colab execution', 'Independent human annotation', 'Fresh untouched holdout evaluation',
                    'New checkpoint training', 'Full Hybrid+anchor answer evaluation on a fresh holdout'],
        'audit_code_sha256': digest(Path(__file__))}
    args.output.mkdir(parents=True)
    (args.output / 'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    (args.output / 'normalization_differences.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    with (args.output / 'training_context_audit.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training_checks[0]))
        writer.writeheader(); writer.writerows(training_checks)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
