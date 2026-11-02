import unittest
import json
import tempfile
from pathlib import Path

from experiments.research_pipeline import prompt, context_text
from experiments.research_evaluate import retrieval_metrics, error_category, model_run


class ResearchProtocolTests(unittest.TestCase):
    def test_rag_checkpoint_cannot_be_mislabeled_as_closed_book(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'manifest.json').write_text(json.dumps({'system': 'rag'}), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'Wrong model system'):
                model_run(path, 'hash', 'closed_book')

    def test_same_template_is_used_with_and_without_context(self):
        self.assertEqual(prompt('Which city?'), 'question: Which city? context: ')
        self.assertEqual(prompt('Which city?', 'Paris'), prompt('Which city?') + 'Paris')
        self.assertEqual(context_text([]), '')

    def test_recall_distinguishes_one_hit_from_covering_all_relevant_documents(self):
        record = {'answer': 'Paris', 'official_answer_aliases': ['paris'], 'gold_doc_ids': ['a', 'b']}
        ranking = [{'doc_id': 'a', 'source_doc_ids': ['a'], 'text': 'Paris.'}]
        row = {'record': record, 'retrieval': {m: ranking for m in ('dense', 'bm25', 'hybrid')}}
        result = retrieval_metrics([row], (1,))['dense']
        self.assertEqual(result['doc_hit_rate@1'], 1)
        self.assertEqual(result['doc_recall@1'], 0.5)
        self.assertEqual(result['canonical_answer_recall@1'], 1)

    def test_deduplicated_passage_retains_all_relevant_source_links(self):
        record = {'answer': 'Paris', 'official_answer_aliases': ['paris'], 'gold_doc_ids': ['b']}
        ranking = [{'doc_id': 'a', 'source_doc_ids': ['a', 'b'], 'text': 'Paris.'}]
        row = {'record': record, 'retrieval': {m: ranking for m in ('dense', 'bm25', 'hybrid')}}
        self.assertEqual(retrieval_metrics([row], (1,))['dense']['doc_recall@1'], 1)

    def test_ranked_evidence_absent_from_visible_context_is_not_generation_failure(self):
        record = {'answer': 'Paris', 'official_answer_aliases': ['paris']}
        result = error_category('London', record, [{'text': 'Paris.'}], 'Truncated.', True, 'canonical')
        self.assertEqual(result, 'context_selection_or_truncation')

    def test_visible_title_evidence_takes_precedence_over_body_only_diagnostics(self):
        record = {'answer': 'Paris', 'official_answer_aliases': ['paris']}
        result = error_category('London', record, [{'text': 'A city in France.'}],
                                'Source [1] | Paris\nA city in France.', False, 'canonical')
        self.assertEqual(result, 'generation_wrong_with_answer_string_in_context')

    def test_policy_does_not_silently_add_canonical_to_official_aliases(self):
        record = {'answer': 'Canonical', 'official_answer_aliases': ['Alias']}
        result = error_category('Canonical', record, [], '', True, 'official')
        self.assertEqual(result, 'retrieval_answer_string_missing')


if __name__ == '__main__':
    unittest.main()
