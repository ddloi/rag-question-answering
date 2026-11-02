"""Failure-oriented checks for the supplemental frozen replay evidence gate."""
import copy
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_research_replay import (ARMS, METHOD_FILES, ReplayValidationError,
                                             compare_split, _load_run)


def fixture(split='dev'):
    n = {'dev': 200, 'test': 300}[split]
    lock = {'arms': list(ARMS), 'test_n': 300, 'dataset_manifest_sha256': 'a' * 64,
            'source_snapshot_sha256': 'b' * 64,
            'config': {'dataset': {'dev': 200, 'test': 300}},
            'models_sha256': {'closed_book': 'c' * 64, 'rag': 'd' * 64},
            'source_sha256': {name: 'e' * 64 for name in METHOD_FILES}}
    manifest = {'status': 'complete', 'split': split, 'pilot': False, 'final': split == 'test',
                'n': n, 'config': copy.deepcopy(lock['config']),
                'dataset_sha256': lock['dataset_manifest_sha256'],
                'closed_book_model_sha256': lock['models_sha256']['closed_book'],
                'rag_model_sha256': lock['models_sha256']['rag'],
                'source_sha256': copy.deepcopy(lock['source_sha256']),
                'source_snapshot_sha256': lock['source_snapshot_sha256'],
                'predictions_sha256': 'f' * 64, 'python': 'fixture', 'packages': {}}
    rows = []
    for i in range(n):
        passage = {'chunk_id': f'chunk-{i}', 'text': 'An answer.', 'score': .5}
        systems = {arm: {'answer': 'Answer', 'context_used': '', 'used_passages': []} for arm in ARMS}
        for arm in ('retrieval_only', 'rag_full', 'rag_hybrid'):
            systems[arm].update(context_used='An answer.', used_passages=[copy.deepcopy(passage)])
        rows.append({'record': {'id': f'q-{i}', 'question': f'Question {i}?', 'answer': 'Answer'},
                     'systems': systems,
                     'retrieval': {m: [copy.deepcopy(passage)] for m in ('dense', 'bm25', 'hybrid')}})
    metrics = {'n': n, 'qa': {arm: {policy: {'em': .5, 'f1': .6}
                                 for policy in ('official', 'canonical')} for arm in ARMS}}
    return {'split': split, 'original_rows': rows, 'replay_rows': copy.deepcopy(rows),
            'original_manifest': manifest, 'replay_manifest': copy.deepcopy(manifest),
            'original_metrics': metrics, 'replay_metrics': copy.deepcopy(metrics), 'lock': lock}


class ReplayComparisonTests(unittest.TestCase):
    def test_exact_valid_evidence_passes_both_expected_split_sizes(self):
        for split, expected in (('dev', 200), ('test', 300)):
            with self.subTest(split=split):
                result = compare_split(**fixture(split))
                self.assertEqual(result['status'], 'PASS_EXACT_ANSWERS')
                self.assertEqual(result['n'], expected)
                self.assertEqual(result['evidence_diagnostics']['status'], 'MATCH')

    def test_duplicate_id_is_not_silently_collapsed(self):
        values = fixture()
        values['replay_rows'].append(copy.deepcopy(values['replay_rows'][0]))
        with self.assertRaisesRegex(ReplayValidationError, 'duplicate QA ID'):
            compare_split(**values)

    def test_missing_row_fails_before_answer_pass(self):
        values = fixture()
        values['replay_rows'].pop()
        with self.assertRaisesRegex(ReplayValidationError, 'expected 200 rows'):
            compare_split(**values)

    def test_equal_row_count_with_wrong_id_fails(self):
        values = fixture()
        values['replay_rows'][0]['record']['id'] = 'unexpected-q'
        with self.assertRaisesRegex(ReplayValidationError, 'QA ID mismatch'):
            compare_split(**values)

    def test_same_id_with_changed_question_is_invalid_input_provenance(self):
        values = fixture()
        values['replay_rows'][0]['record']['question'] = 'Another question?'
        with self.assertRaisesRegex(ReplayValidationError, 'input/reference records differ'):
            compare_split(**values)

    def test_wrong_model_hash_cannot_pass_even_if_all_answers_match(self):
        values = fixture()
        values['replay_manifest']['rag_model_sha256'] = '0' * 64
        with self.assertRaisesRegex(ReplayValidationError, 'rag model identity mismatch'):
            compare_split(**values)

    def test_incomplete_or_wrong_count_manifest_cannot_pass(self):
        for field, value, message in (('status', 'running', 'not complete'), ('n', 199, 'n must be 200')):
            with self.subTest(field=field):
                values = fixture()
                values['replay_manifest'][field] = value
                with self.assertRaisesRegex(ReplayValidationError, message):
                    compare_split(**values)

    def test_missing_arm_is_a_schema_failure(self):
        values = fixture()
        del values['replay_rows'][0]['systems']['generator_only']
        with self.assertRaisesRegex(ReplayValidationError, 'exactly the five'):
            compare_split(**values)

    def test_nonfinite_metrics_are_invalid_evidence(self):
        values = fixture()
        values['replay_metrics']['qa']['rag_full']['official']['em'] = float('nan')
        with self.assertRaisesRegex(ReplayValidationError, 'finite'):
            compare_split(**values)

    def test_metric_count_mismatch_is_invalid_evidence(self):
        values = fixture()
        values['replay_metrics']['n'] = 400
        with self.assertRaisesRegex(ReplayValidationError, 'metrics: n must be 200'):
            compare_split(**values)

    def test_actual_answer_change_needs_review_even_if_normalization_matches(self):
        values = fixture()
        values['replay_rows'][0]['systems']['rag_full']['answer'] = 'answer'
        result = compare_split(**values)
        self.assertEqual(result['status'], 'REVIEW_REQUIRED')
        self.assertEqual(result['raw_answer_mismatch_ids']['rag_full'], ['q-0'])
        self.assertEqual(result['normalized_answer_mismatch_ids']['rag_full'], [])

    def test_finite_metric_difference_needs_review(self):
        values = fixture()
        values['replay_metrics']['qa']['rag_full']['official']['em'] += 1e-5
        result = compare_split(**values)
        self.assertEqual(result['status'], 'REVIEW_REQUIRED')
        self.assertFalse(result['qa_metrics_equal_within_tolerance'])

    def test_context_change_is_reported_separately_from_exact_answers(self):
        values = fixture()
        values['replay_rows'][0]['systems']['rag_full']['context_used'] = 'Different context.'
        result = compare_split(**values)
        self.assertEqual(result['status'], 'PASS_EXACT_ANSWERS')
        self.assertEqual(result['evidence_diagnostics']['status'], 'REVIEW_REQUIRED')
        self.assertEqual(result['evidence_diagnostics']['context_mismatch_ids']['rag_full'], ['q-0'])

    def test_float_retrieval_score_rounding_is_not_an_evidence_content_mismatch(self):
        values = fixture()
        values['replay_rows'][0]['retrieval']['dense'][0]['score'] += 1e-7
        values['replay_rows'][0]['systems']['rag_full']['used_passages'][0]['score'] += 1e-7
        result = compare_split(**values)
        self.assertEqual(result['evidence_diagnostics']['status'], 'MATCH')

    def test_other_snapshot_archive_can_be_valid_when_locked_method_hashes_match(self):
        values = fixture()
        values['original_manifest']['source_snapshot_sha256'] = '1' * 64
        values['replay_manifest']['source_snapshot_sha256'] = '2' * 64
        result = compare_split(**values)
        self.assertEqual(result['status'], 'PASS_EXACT_ANSWERS')
        self.assertFalse(result['evidence_diagnostics']['source_snapshot_archive_equal'])

    def test_changed_locked_method_is_invalid_even_with_matching_answers(self):
        values = fixture()
        values['replay_manifest']['source_sha256'][METHOD_FILES[0]] = '0' * 64
        with self.assertRaisesRegex(ReplayValidationError, 'frozen method mismatch'):
            compare_split(**values)

    def test_malformed_lock_returns_a_useful_validation_error(self):
        values = fixture()
        values['lock']['models_sha256'] = []
        with self.assertRaisesRegex(ReplayValidationError, 'Lock model hashes are missing'):
            compare_split(**values)

    def test_run_loader_rejects_prediction_bytes_that_disagree_with_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = fixture()
            with gzip.open(root / 'predictions.jsonl.gz', 'wt', encoding='utf8') as stream:
                for row in values['original_rows']:
                    stream.write(json.dumps(row) + '\n')
            (root / 'source_snapshot.zip').write_bytes(b'archive fixture')
            manifest = values['original_manifest']
            manifest['source_snapshot_sha256'] = hashlib.sha256(b'archive fixture').hexdigest()
            (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
            (root / 'metrics.json').write_text(json.dumps(values['original_metrics']), encoding='utf8')
            with self.assertRaisesRegex(ReplayValidationError, 'predictions.jsonl.gz checksum mismatch'):
                _load_run(root)


if __name__ == '__main__':
    unittest.main()
