import unittest
from scripts.report_research import check_numbers, select_examples


class ResearchReportingTests(unittest.TestCase):
    def test_altered_scores_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            check_numbers({'qa': {'em': .5}}, {'qa': {'em': .6}})

    def test_rounding_tolerance_is_bounded(self):
        check_numbers({'em': 1/3}, {'em': .3333333333333333})

    def test_examples_cover_error_classes_and_negative_hybrid_results(self):
        rows = [{'question_id':str(i), 'dense_error_proxy':kind, 'hybrid_direction':direction}
                for i,(kind,direction) in enumerate([('correct','same_em'), ('retrieval','hybrid_rescue'),
                                                    ('generation','hybrid_regression')])]
        groups = {r['category'] for r in select_examples(rows)}
        self.assertEqual(groups, {'correct','retrieval','generation','hybrid_rescue','hybrid_regression'})


if __name__ == '__main__':
    unittest.main()
