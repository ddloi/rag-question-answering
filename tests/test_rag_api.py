"""Optional neural failures must never discard the working BM25 result."""

import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from ragqa import web


class NeuralApiTests(unittest.TestCase):
    def setUp(self):
        self.neural = Mock(is_loaded=False, is_loading=False, load_error=None, device_name="CPU")
        self.neural.is_available.return_value = True
        self.neural.execute_rag.return_value = {"available": False, "is_loading": True}
        self.patch = patch.object(web, "rag_neural_engine", self.neural)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = TestClient(web.app)

    def test_cold_query_does_not_block_on_model_loading(self):
        response = self.client.post('/api/pipeline/execute', json={"query": "Neptune"})
        self.assertEqual(response.status_code, 200)
        self.neural.load_sync.assert_not_called()
        self.neural.load_in_background.assert_called_once()
        self.assertTrue(response.json()['result']['top_chunks'])

    def test_neural_failure_preserves_bm25_and_full_timing(self):
        self.neural.is_loaded = True
        self.neural.execute_rag.side_effect = RuntimeError('simulated GPU failure')
        with self.assertLogs(level='ERROR'):
            response = self.client.post('/api/pipeline/execute', json={"query": "Neptune"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['result']['top_chunks'])
        self.assertFalse(data['rag']['available'])
        self.assertGreaterEqual(data['request_latency_ms'], data['telemetry']['total_pipeline_latency_ms'])

    def test_failed_load_is_not_retried_by_every_question(self):
        self.neural.load_error = 'missing dependency'
        response = self.client.post('/api/pipeline/execute', json={"query": "Neptune"})
        self.assertEqual(response.status_code, 200)
        self.neural.load_sync.assert_not_called()
        self.neural.load_in_background.assert_not_called()

    def test_explicit_load_is_nonblocking_and_allows_retry(self):
        self.neural.load_error = 'previous failure'
        response = self.client.post('/api/rag/load')
        self.assertEqual(response.status_code, 202)
        self.neural.load_sync.assert_not_called()
        self.neural.load_in_background.assert_called_once_with()

    def test_info_reports_loading_failure_honestly(self):
        self.neural.load_error = 'missing dependency'
        data = self.client.get('/api/pipeline/info').json()
        self.assertEqual(data['rag_status'], 'error')
        self.assertFalse(data['generator_enabled'])
        self.assertEqual(data['rag_error'], 'missing dependency')

    def test_bm25_only_does_not_touch_neural_execution(self):
        data = self.client.post('/api/pipeline/execute', json={"query": "Neptune", "run_rag": False}).json()
        self.neural.load_in_background.assert_not_called()
        self.neural.execute_rag.assert_not_called()
        self.assertIsNone(data['rag'])

    def test_reference_works_beyond_visible_sample_list(self):
        response = self.client.post('/api/pipeline/execute', json={
            "query": "Proteus and Nereid are moons of which planet?", "run_rag": False})
        self.assertEqual(response.json()['reference']['ground_truth'], 'Neptune')
        self.neural.execute_rag.assert_not_called()

    def test_demo_samples_cover_the_four_explained_cases(self):
        samples = self.client.get('/api/pipeline/samples').json()['samples']
        self.assertEqual({sample['category'] for sample in samples},
                         {'correct', 'generation', 'missing', 'context'})
        self.assertTrue(all(sample['case_note'] and sample['ground_truth'] for sample in samples))

    def test_missing_keywords_warn_but_still_generate(self):
        self.neural.is_loaded = True
        self.neural.execute_rag.return_value = {'available': True, 'answer': 'example'}
        for question in ('What is zzzunknownthing?', 'Who invented zzzunknownperson?'):
            with self.subTest(question=question):
                result = self.client.post('/api/pipeline/execute', json={'query': question}).json()
                self.assertTrue(result['rag']['available'])
                self.assertTrue(result['rag']['missing_terms'])
        self.assertEqual(self.neural.execute_rag.call_count, 2)

    def test_retrieval_mode_is_explicit_and_invalid_values_are_rejected(self):
        self.client.post('/api/pipeline/execute', json={'query': 'Neptune', 'retrieval_mode': 'dense'})
        self.neural.execute_rag.assert_called_once_with(query='Neptune', top_k=5)
        self.neural.execute_rag.reset_mock()
        self.client.post('/api/pipeline/execute', json={'query': 'Neptune', 'retrieval_mode': 'hybrid'})
        options = self.neural.execute_rag.call_args.kwargs
        self.assertTrue(options['sparse_candidates'])
        self.assertIn('anchor_term', options)
        self.assertEqual(self.client.post('/api/pipeline/execute', json={
            'query': 'Neptune', 'retrieval_mode': 'invalid'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
