"""Path regressions: modules must not depend on the terminal's working folder."""

import os
import tempfile
import unittest
from pathlib import Path

from ragqa.paths import (
    DATA_DIR,
    MODEL_DIR,
    PROJECT_ROOT,
    RAW_DOCUMENTS_PATH,
    RESULTS_DIR,
    resolve_project_path,
)


class ProjectPathTests(unittest.TestCase):
    def test_relative_config_path_uses_project_root(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                resolved = resolve_project_path("data/test_qa.jsonl")
            finally:
                os.chdir(previous)
        self.assertEqual(resolved, DATA_DIR / "test_qa.jsonl")

    def test_absolute_config_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            absolute = Path(elsewhere) / "custom.jsonl"
            self.assertEqual(resolve_project_path(absolute), absolute)

    def test_artifact_locations_share_one_root(self):
        self.assertEqual(DATA_DIR, PROJECT_ROOT / "data")
        self.assertEqual(MODEL_DIR, PROJECT_ROOT / "generator_finetuned")
        self.assertEqual(RAW_DOCUMENTS_PATH, DATA_DIR / "raw_documents.json")
        self.assertEqual(RESULTS_DIR, PROJECT_ROOT / "results")
        self.assertTrue(all(p.is_absolute() for p in (DATA_DIR, MODEL_DIR, RAW_DOCUMENTS_PATH, RESULTS_DIR)))


if __name__ == "__main__":
    unittest.main()
