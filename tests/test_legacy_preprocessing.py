"""Guard legacy workflows against hangs, mixed splits and global RNG changes."""
import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import ablation, split_train_test
from experiments.prepare_chunks import chunk_text


class LegacyPreprocessingTests(unittest.TestCase):
    def test_chunking_rejects_non_advancing_or_negative_windows(self):
        for size, overlap in [(0, 0), (-1, 0), (2, 2), (2, 3), (2, -1)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_text('one two three four', size, overlap)

    def test_shared_chunker_preserves_overlap_and_final_window(self):
        self.assertIs(ablation.chunk_text, chunk_text)
        self.assertEqual(chunk_text('a b c d e f', 3, 1), ['a b c', 'c d e', 'e f'])
        self.assertEqual(chunk_text('', 3, 0), [])

    def test_ablation_never_falls_back_to_unsplit_pairs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'qa_pairs.jsonl').write_text('{"question":"train example"}\n')
            with patch.object(ablation, 'DATA_DIR', path), self.assertRaisesRegex(FileNotFoundError, 'test split'):
                ablation.main()

    def test_ablation_rejects_empty_test_before_loading_models(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / 'test_qa.jsonl').write_text('\n')
            with patch.object(ablation, 'DATA_DIR', path), self.assertRaisesRegex(ValueError, 'empty'):
                ablation.main()

    def test_split_is_repeatable_and_preserves_global_random_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            records = [{'question': str(i), 'answer': 'a', 'gold_doc_id': 'doc'} for i in range(20)]
            corpus = [{'doc_id': 'doc', 'text': 'a'}]
            for name, rows in [('qa_pairs.jsonl', records), ('corpus_chunks.jsonl', corpus)]:
                (path / name).write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf8')
            state = random.getstate()
            with patch.object(split_train_test, 'DATA_DIR', path), contextlib.redirect_stdout(io.StringIO()):
                split_train_test.main()
                first = (path / 'test_qa.jsonl').read_bytes()
                split_train_test.main()
            self.assertEqual(first, (path / 'test_qa.jsonl').read_bytes())
            self.assertEqual(state, random.getstate())
            random.Random(42).shuffle(records)
            self.assertEqual([json.loads(line) for line in first.splitlines()], records[:3])


if __name__ == '__main__':
    unittest.main()
