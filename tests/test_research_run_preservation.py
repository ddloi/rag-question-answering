import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import research_train


class RunPreservationTests(unittest.TestCase):
    def test_completed_run_cannot_be_overwritten_by_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'manifest.json'
            content = '{"status":"complete"}\n'
            manifest.write_text(content, encoding='utf8')
            with patch('sys.argv', ['train', '--system', 'rag', '--resume', '--output', folder]), \
                 patch.object(research_train, 'dataset_manifest', return_value={'config': {'version': 'fixture'}}):
                with self.assertRaisesRegex(ValueError, 'immutable'):
                    research_train.main()
            self.assertEqual(manifest.read_text(encoding='utf8'), content)


if __name__ == '__main__':
    unittest.main()
