"""Pure helpers of the dev ablation/reranker run; no models are loaded."""
import unittest

from experiments.research_ablation import (CONFIGS, BASELINE, error_category, paired_bootstrap,
                                           rerank, retrieval_summary, visible_fraction)


def passage(chunk_id, rank, text, doc='d1'):
    return {'chunk_id': chunk_id, 'doc_id': doc, 'title': 'T', 'text': text, 'rank': rank, 'score': 1 / rank}


RECORD = {'id': 'q1', 'question': 'Which planet?', 'answer': 'Neptune',
          'official_answer_aliases': ['Neptune', 'Planet Neptune'], 'gold_doc_ids': ['d2']}


class AblationTests(unittest.TestCase):
    def test_baseline_matches_frozen_rag_full_protocol(self):
        self.assertIn((BASELINE, 'dense', 2, 512), CONFIGS)

    def test_rerank_orders_by_score_and_keeps_dense_rank(self):
        ranking = [passage('a', 1, 'x'), passage('b', 2, 'y'), passage('c', 3, 'z')]
        result = rerank(ranking, [0.1, 5.0, 0.1])
        self.assertEqual([p['chunk_id'] for p in result], ['b', 'a', 'c'])  # tie keeps Dense order
        self.assertEqual([p['rank'] for p in result], [1, 2, 3])
        self.assertEqual(result[0]['dense_rank'], 2)
        self.assertEqual(result[0]['dense_score'], 0.5)

    def test_visible_fraction_measures_truncation(self):
        selected = [passage('a', 1, 'abcd'), passage('b', 2, 'efgh')]
        used = [{'text': 'abcd'}, {'text': 'ef'}]
        self.assertAlmostEqual(visible_fraction(selected, used), 0.75)
        self.assertEqual(visible_fraction([], []), 0.0)

    def test_retrieval_summary_uses_policy_answers_and_gold_docs(self):
        ranking = [passage('a', 1, 'nothing here'), passage('b', 2, 'Planet Neptune is blue', doc='d2')]
        summary = retrieval_summary([RECORD], [ranking], k_values=(1, 2))
        self.assertEqual(summary['doc_hit_rate@1'], 0)
        self.assertEqual(summary['doc_hit_rate@2'], 1)
        self.assertEqual(summary['official_answer_recall@2'], 1)
        self.assertEqual(list(summary)[:2], ['doc_hit_rate@1', 'doc_hit_rate@2'])

    def test_error_category_separates_retrieval_and_generation(self):
        ranking = [passage('a', 1, 'Neptune is far')]
        self.assertEqual(error_category('Neptune', RECORD, ranking, '', True, 'official'), 'correct')
        self.assertEqual(error_category('Mars', RECORD, ranking, 'Neptune is far', True, 'official'),
                         'generation_wrong_with_answer_string_in_context')
        self.assertEqual(error_category('Mars', RECORD, ranking, 'cut off', True, 'official'),
                         'context_selection_or_truncation')
        self.assertEqual(error_category('Mars', RECORD, [passage('b', 1, 'none')], '', True, 'official'),
                         'retrieval_answer_string_missing')
        self.assertEqual(error_category('Mars', RECORD, ranking, '', False, 'official'),
                         'corpus_answer_string_missing')

    def test_paired_bootstrap_is_deterministic_and_counts_changes(self):
        result = paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0], seed=42, resamples=200)
        self.assertAlmostEqual(result['delta'], 0.5)
        self.assertEqual((result['wins'], result['losses']), (2, 0))
        self.assertEqual(result, paired_bootstrap([1, 1, 0, 1], [0, 1, 0, 0], seed=42, resamples=200))
        self.assertLessEqual(result['ci95'][0], result['delta'])


if __name__ == '__main__':
    unittest.main()
