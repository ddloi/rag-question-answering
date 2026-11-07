"""Merging exported labels must never touch the blinded evidence columns."""
import unittest

from scripts.merge_review_labels import merge

SHEET = [{'review_id': 'review_001', 'question': 'Q', 'prediction': 'P', 'context_used': 'Source [1] | T\nx',
          'source_chunk_ids': 'c1', 'faithfulness': '', 'citation_labels_json': '', 'rationale': '',
          'reviewer_id': 'author_1', 'reviewer_kind': 'human_author'},
         {'review_id': 'review_002', 'question': 'Q2', 'prediction': 'P2', 'context_used': 'y',
          'source_chunk_ids': 'c2', 'faithfulness': '', 'citation_labels_json': '', 'rationale': '',
          'reviewer_id': 'author_1', 'reviewer_kind': 'human_author'}]


class MergeTests(unittest.TestCase):
    def test_only_label_columns_change(self):
        labels = [{'review_id': 'review_001', 'faithfulness': 'supported', 'question': 'EDITED',
                   'citation_labels_json': '{"c1": "supports_answer"}', 'rationale': 'r',
                   'reviewer_id': 'author_1', 'reviewer_kind': 'human_author'}]
        merged = merge(SHEET, labels)
        self.assertEqual(merged[0]['question'], 'Q')
        self.assertEqual(merged[0]['faithfulness'], 'supported')
        self.assertEqual(merged[1], SHEET[1])

    def test_unknown_review_id_is_rejected(self):
        with self.assertRaises(ValueError):
            merge(SHEET, [{'review_id': 'review_999'}])


if __name__ == '__main__':
    unittest.main()
