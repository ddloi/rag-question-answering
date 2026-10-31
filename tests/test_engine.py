"""Regression checks for the local BM25 and RAG demo. Run: python -m unittest discover -s tests -v"""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from ragqa.engine import DATA_DIR, DeterministicPipelineEngine


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        rows = [
            {"chunk_id": 0, "doc_id": "long", "title": "Long source", "text": "Apple is a fruit. " + "Other words for length. " * 20},
            {"chunk_id": 1, "doc_id": "short", "title": "Short source", "text": "Apple is a fruit."},
            {"chunk_id": 2, "doc_id": "repeat", "title": "Repeated term", "text": "Banana banana banana. Banana is yellow."},
            {"chunk_id": 3, "doc_id": "other", "title": "Other", "text": "A pear grows on a tree."},
        ]
        metadata = Path(cls.temp_dir.name) / "chunk_metadata.jsonl"
        metadata.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        cls.engine = DeterministicPipelineEngine(Path(cls.temp_dir.name))

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_length_normalization_changes_real_ranking(self):
        no_norm = self.engine.execute_pipeline("apple", b=0)["result"]["top_chunks"]
        normalized = self.engine.execute_pipeline("apple", b=1)["result"]["top_chunks"]
        self.assertEqual(no_norm[0]["doc_id"], "long")
        self.assertEqual(normalized[0]["doc_id"], "short")
        self.assertNotEqual(no_norm[0]["score"], normalized[0]["score"])

    def test_tf_saturation_changes_real_score(self):
        low = self.engine.execute_pipeline("banana", k1=0.5)["result"]["top_chunks"][0]["score"]
        high = self.engine.execute_pipeline("banana", k1=3.0)["result"]["top_chunks"][0]["score"]
        self.assertNotEqual(low, high)

    def test_displayed_breakdown_matches_ranking_formula(self):
        result = self.engine.execute_pipeline("apple fruit apple", k1=2.3, b=0.4)
        for chunk in result["result"]["top_chunks"]:
            self.assertAlmostEqual(chunk["score"], sum(term["term_score"] for term in chunk["score_breakdown"]), places=3)
            doc_idx = chunk["chunk_id"]
            expected = sum(
                self.engine.calculate_term_idf(term)
                * self.engine.term_counts[doc_idx].get(term, 0)
                * 3.3
                / (self.engine.term_counts[doc_idx].get(term, 0) + 2.3 * (0.6 + 0.4 * chunk["doc_len"] / self.engine.avgdl))
                for term in ("apple", "fruit")
            )
            self.assertAlmostEqual(chunk["score"], expected, places=3)
        self.assertEqual(result["stages"][0]["active_terms"], ["apple", "fruit"])

    def test_unknown_and_punctuation_queries_return_no_source(self):
        for query in ("zzzxqvnonsense", "???"):
            with self.subTest(query=query):
                result = self.engine.execute_pipeline(query)
                self.assertEqual(result["result"]["top_chunks"], [])
                self.assertIsNone(result["result"]["grounded_source_chunk"])
                self.assertEqual(result["stages"][2]["grounding_ratio"], 0)
                self.assertEqual(len(result["stages"]), 4)

    def test_snippet_is_verbatim_and_output_is_deterministic(self):
        first = self.engine.execute_pipeline("apple fruit", top_k=1)
        second = self.engine.execute_pipeline("apple fruit", top_k=1)
        self.assertEqual(first["result"], second["result"])
        self.assertIn(first["result"]["top_snippet"], first["result"]["top_chunks"][0]["text"])
        self.assertTrue(math.isfinite(first["telemetry"]["total_pipeline_latency_ms"]))

    def test_invalid_direct_engine_parameters(self):
        for arguments in ({"query": " "}, {"query": "x" * 2001}, {"top_k": 0}, {"top_k": -1}, {"top_k": True}, {"k1": float("nan")}, {"k1": 0}, {"b": 2}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.engine.execute_pipeline(**({"query": "apple"} | arguments))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original_dir = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:
                os.chdir(elsewhere)
                from ragqa.web import app
            finally:
                os.chdir(original_dir)
        cls.client = TestClient(app)

    def test_info_and_samples_use_real_local_corpus(self):
        info = self.client.get("/api/pipeline/info")
        self.assertEqual(info.status_code, 200)
        self.assertGreater(info.json()["total_chunks"], 0)
        self.assertIn("generator_enabled", info.json())
        self.assertTrue(DATA_DIR.is_absolute())
        samples = self.client.get("/api/pipeline/samples")
        self.assertEqual(samples.status_code, 200)
        self.assertGreater(len(samples.json()["samples"]), 0)

    def test_sample_query_executes_and_assets_are_served(self):
        question = self.client.get("/api/pipeline/samples").json()["samples"][0]["question"]
        response = self.client.post("/api/pipeline/execute", json={"query": question, "run_rag": False})
        self.assertEqual(response.status_code, 200)
        result = response.json()["result"]
        self.assertGreater(len(result["top_chunks"]), 0)
        self.assertIn(result["top_snippet"], result["top_chunks"][0]["text"])
        for path in ("/", "/static/app.js", "/static/style.css"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_api_rejects_out_of_range_parameters(self):
        for arguments in ({"query": " "}, {"query": "x" * 2001}, {"top_k": -1}, {"top_k": 11}, {"top_k": 1.5}, {"top_k": True}, {"k1": 0}, {"k1": 4}, {"b": -0.1}, {"b": 1.1}):
            with self.subTest(arguments=arguments):
                response = self.client.post("/api/pipeline/execute", json={"query": "apple"} | arguments)
                self.assertEqual(response.status_code, 422)

    def test_rag_execution_returns_structured_comparison(self):
        response = self.client.post("/api/pipeline/execute", json={"query": "Who directed the movie Jaws?", "run_rag": True})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rag", data)

    def test_rag_low_relevance_gating_prevents_hallucination(self):
        response = self.client.post("/api/pipeline/execute", json={"query": "???", "run_rag": True})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        if data.get("rag") and data["rag"].get("available"):
            self.assertFalse(data["rag"]["is_grounded"])
            self.assertLess(data["rag"]["top_score"], 0.25)
            self.assertIn("Không tìm thấy", data["rag"]["answer"])


if __name__ == "__main__":
    unittest.main()
