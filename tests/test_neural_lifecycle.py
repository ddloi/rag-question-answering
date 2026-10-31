"""Lifecycle checks use no ML dependencies, downloads or GPU."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragqa.rag_engine import RAGNeuralEngine


class NeuralLifecycleTests(unittest.TestCase):
    def test_index_without_checkpoint_is_not_available(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "faiss.index").touch()
            (root / "chunk_metadata.jsonl").touch()
            engine = RAGNeuralEngine(root, root / "model")
            self.assertFalse(engine.is_available())

    def test_simultaneous_load_requests_start_one_worker(self):
        engine = RAGNeuralEngine()
        with patch.object(engine, "is_available", return_value=True), \
             patch("ragqa.rag_engine.threading.Thread") as worker:
            engine.load_in_background()
            engine.load_in_background()
            self.assertTrue(engine.is_loading)
            worker.assert_called_once()
            worker.return_value.start.assert_called_once()

    def test_failed_thread_start_releases_loading_state(self):
        engine = RAGNeuralEngine()
        with patch.object(engine, "is_available", return_value=True), \
             patch("ragqa.rag_engine.threading.Thread") as worker:
            worker.return_value.start.side_effect = RuntimeError("thread unavailable")
            with self.assertRaises(RuntimeError):
                engine.load_in_background()
        self.assertFalse(engine.is_loading)

    def test_failed_load_clears_partial_components(self):
        engine = RAGNeuralEngine()
        engine.embed_model = object()
        with patch.object(engine, "is_available", return_value=False), \
             self.assertLogs("ragqa.rag_engine", level="ERROR"):
            engine.load_sync()
        self.assertFalse(engine.is_loading)
        self.assertFalse(engine.is_loaded)
        self.assertIsNone(engine.embed_model)
        self.assertTrue(engine.load_error)

    def test_busy_model_returns_promptly_and_failure_releases_lock(self):
        engine = RAGNeuralEngine()
        engine.is_loaded = True
        with engine._inference_lock:
            self.assertFalse(engine.execute_rag("question")["available"])
        with patch.object(engine, "_execute_loaded", side_effect=RuntimeError("failed")):
            with self.assertRaises(RuntimeError):
                engine.execute_rag("question")
        self.assertFalse(engine._inference_lock.locked())
