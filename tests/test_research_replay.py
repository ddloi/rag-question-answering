import json
import tempfile
import unittest
from pathlib import Path
from hashlib import sha256

from experiments.research_artifacts import records_sha256
from experiments.research_data import collect_source, question_key
from experiments.verify_dataset import read_jsonl


class ReplayTests(unittest.TestCase):
    def test_record_hash_ignores_serialization_and_newlines_but_preserves_text(self):
        rows = [{'id': 'x', 'text': 'Two words.'}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.jsonl'
            path.write_bytes((json.dumps(rows[0], indent=None) + '\r\n').encode())
            self.assertEqual(records_sha256(rows), records_sha256(read_jsonl(path)))
        self.assertEqual(records_sha256(rows), records_sha256([{'text': 'Two words.', 'id': 'x'}]))
        self.assertNotEqual(records_sha256(rows), records_sha256([{'id': 'x', 'text': 'Two  words.'}]))

    def test_cached_legacy_question_is_rejected_without_network(self):
        row = {'question_id': 'new-id', 'question': 'Question already used?'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf8')
            key = sha256(question_key(row['question']).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'legacy question'):
                collect_source({}, 'validation', 1, set(), set(), path, {key})

    def test_valid_complete_cache_is_reused_without_network(self):
        row = {'question_id': 'new-id', 'question': 'An unseen question?'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'source.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf8')
            self.assertEqual(collect_source({}, 'validation', 1, set(), set(), path), [row])


if __name__ == '__main__':
    unittest.main()
