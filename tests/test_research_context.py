"""Offsets fixture checks citation accounting at a truncation boundary, without ML."""
import unittest
from experiments.research_pipeline import encode_context, prompt


class Offsets:
    def __init__(self, end):
        self.end = end
    def __getitem__(self, index):
        return self
    def tolist(self):
        return [[0, self.end], [0, 0]]


class OffsetTokenizer:
    def __call__(self, text, **options):
        return {'offset_mapping': Offsets(min(options['max_length'], len(text))), 'input_ids': 'unchanged-fixture'}


class ResearchContextTests(unittest.TestCase):
    def test_visible_text_is_clipped_and_fully_hidden_source_is_omitted(self):
        passages = [{'chunk_id': 'a', 'title': 'A', 'text': 'Paris is a city.'},
                    {'chunk_id': 'b', 'title': 'B', 'text': 'London is a city.'}]
        end = len(prompt('Q?')) + len('Source [1] | A\nParis')
        inputs, context, used = encode_context(OffsetTokenizer(), 'Q?', passages, end)
        self.assertEqual(inputs['input_ids'], 'unchanged-fixture')
        self.assertEqual(context, 'Source [1] | A\nParis')
        self.assertEqual([p['chunk_id'] for p in used], ['a'])
        self.assertEqual(used[0]['text'], 'Paris')

    def test_no_context_has_no_citations(self):
        _, context, used = encode_context(OffsetTokenizer(), 'Q?', [], 512)
        self.assertEqual((context, used), ('', []))

    def test_partly_visible_title_is_recorded_without_claiming_body_was_seen(self):
        passages = [{'chunk_id': 'a', 'title': 'Paris', 'text': 'City in France.'}]
        heading = 'Source [1] | Par'
        inputs, context, used = encode_context(OffsetTokenizer(), 'Q?', passages, len(prompt('Q?')) + len(heading))
        self.assertEqual(context, heading)
        self.assertEqual(inputs['input_ids'], 'unchanged-fixture')
        self.assertEqual(used[0]['visible_heading'], heading)
        self.assertEqual(used[0]['text'], '')
        self.assertTrue(used[0]['title_only'])


if __name__ == '__main__':
    unittest.main()
