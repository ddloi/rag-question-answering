"""Regression examples where punctuation changed scientific conclusions."""
import unittest

from experiments import metrics, legacy_metrics


class TriviaQAMetricsTests(unittest.TestCase):
    def test_extracted_apostrophes_are_equivalent_to_reference(self):
        examples = [("Eugene O ' Neill", 'Eugene O’Neill'),
                    ("A Hard Day ' s Night", "A Hard Day's Night"),
                    ('Twenty-One', 'twenty one'), ('The_Winter’s_Tale', 'Winter s Tale')]
        for prediction, answer in examples:
            with self.subTest(prediction=prediction):
                self.assertEqual(metrics.score_answer(prediction, {'answer': answer}), {'em': 1, 'f1': 1.0})

    def test_historical_scoring_is_preserved_explicitly(self):
        record = {'answer': 'Eugene O’Neill'}
        self.assertEqual(legacy_metrics.score_answer("Eugene O ' Neill", record)['em'], 0)
        self.assertEqual(metrics.score_answer("Eugene O ' Neill", record)['em'], 1)

    def test_official_policy_uses_only_the_supplied_normalized_alias_set(self):
        record = {'answer': 'Canonical', 'official_answer_aliases': ['an alias'], 'answer_aliases': ['unreviewed']}
        self.assertEqual(metrics.score_answer('Alias', record, 'official')['em'], 1)
        self.assertEqual(metrics.score_answer('Canonical', record, 'official')['em'], 0)
        self.assertEqual(metrics.score_answer('unreviewed', record, 'official')['em'], 0)
        self.assertEqual(metrics.score_answer('Canonical', record, 'canonical')['em'], 1)

    def test_missing_official_or_curated_labels_fail_explicitly(self):
        for policy in ('official', 'curated'):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                metrics.score_answer('answer', {'answer': 'answer'}, policy)

    def test_f1_empty_tokens_matches_official_zero_convention(self):
        self.assertEqual(metrics.f1_score('', ''), 0)
        self.assertEqual(metrics.f1_score('The', 'a'), 0)


if __name__ == '__main__':
    unittest.main()
