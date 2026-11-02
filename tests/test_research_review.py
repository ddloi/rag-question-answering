import unittest
from scripts.summarize_research_review import summarize


class ResearchReviewTests(unittest.TestCase):
    def setUp(self):
        self.key = {'r1': {'system': 'rag_full'}}
        self.row = {'review_id': 'r1', 'source_chunk_ids': 'c1;c2', 'faithfulness': '',
                    'citation_labels_json': '', 'rationale': '', 'reviewer_id': '', 'reviewer_kind': ''}

    def test_empty_forms_are_not_claimed_as_an_evaluation(self):
        self.assertEqual(summarize([self.row], self.key)['status'], 'NOT RUN')

    def test_agent_labels_remain_separate_and_irrelevant_citations_count(self):
        row = {**self.row, 'faithfulness': 'supported', 'citation_labels_json':
               '{"c1":"supports_answer","c2":"does_not_support"}', 'rationale': 'Only c1 supports it.',
               'reviewer_id': 'assistant', 'reviewer_kind': 'agent_self_review'}
        result = summarize([row], self.key)
        self.assertEqual(result['groups']['agent_self_review/rag_full']['citation_precision_assessed'], 0.5)
        self.assertFalse(any(key.startswith('human') for key in result['groups']))
        self.assertEqual(result['independent_human_status'], 'NOT RUN')

    def test_missing_selected_questions_cannot_be_called_complete(self):
        key = {**self.key, 'r2': {'system': 'rag_hybrid'}}
        row = {**self.row, 'faithfulness': 'supported', 'citation_labels_json':
               '{"c1":"supports_answer","c2":"supports_answer"}', 'rationale': 'Both support it.',
               'reviewer_id': 'p1', 'reviewer_kind': 'human_author'}
        self.assertEqual(summarize([row], key)['status'], 'PARTIAL')

    def test_invented_citation_ids_are_rejected(self):
        row = {**self.row, 'faithfulness': 'supported', 'citation_labels_json': '{"invented":"supports_answer"}',
               'rationale': 'reason', 'reviewer_id': 'p1', 'reviewer_kind': 'human_author'}
        with self.assertRaises(ValueError):
            summarize([row], self.key)

    def test_unclear_judgments_remain_in_reported_coverage(self):
        key = {**self.key, 'r2': {'system': 'rag_full'}}
        row = {**self.row, 'faithfulness': 'supported', 'citation_labels_json':
               '{"c1":"supports_answer","c2":"unclear"}', 'rationale': 'Evidence is incomplete.',
               'reviewer_id': 'p1', 'reviewer_kind': 'human_author'}
        group = summarize([row, {**row, 'review_id': 'r2', 'faithfulness': 'unclear'}], key)['groups']['human_author/rag_full']
        self.assertEqual(group['faithfulness_supported_fraction_all'], .5)
        self.assertEqual(group['faithfulness_supported_fraction_assessed'], 1)
        self.assertEqual(group['faithfulness_unclear'], 1)
        self.assertEqual(group['citation_unclear'], 2)


if __name__ == '__main__':
    unittest.main()
