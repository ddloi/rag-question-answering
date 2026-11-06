import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('numpy', 'rank_bm25', 'sklearn', 'gensim')),
                     'Classical baseline tests require requirements/ml.txt')
class BaselineConfigTests(unittest.TestCase):
    def setUp(self):
        from experiments import baseline_retrieval
        self.baseline = baseline_retrieval

    def test_invalid_config_is_not_silently_replaced_with_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'hyperparams.json'
            path.write_text('{invalid', encoding='utf8')
            with patch.object(self.baseline, 'CONFIG_PATH', path):
                with self.assertRaisesRegex(ValueError, 'Cannot read baseline configuration'):
                    self.baseline.load_config()

    def test_local_config_takes_priority_over_versioned_defaults(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'hyperparams.json'
            config = {'random_seed': 17}
            path.write_text(json.dumps(config), encoding='utf8')
            with patch.object(self.baseline, 'CONFIG_PATH', path):
                self.assertEqual(self.baseline.load_config(), config)


if __name__ == '__main__':
    unittest.main()
