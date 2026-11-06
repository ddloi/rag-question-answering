import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from fastapi import FastAPI
from ragqa import research_web

app = FastAPI()
app.include_router(research_web.router)


class ResearchWebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_page_has_three_separate_systems(self):
        response = self.client.get('/research')
        self.assertEqual(response.status_code, 200)
        for arm in ('generator_only', 'retrieval_only', 'rag'):
            self.assertIn(f'id="answer-{arm}"', response.text)

    def test_incomplete_models_cannot_be_loaded(self):
        with patch.object(research_web, 'available', return_value=False):
            self.assertEqual(self.client.post('/api/research/load').status_code, 409)

    def test_inference_does_not_accept_reference_answers_from_client(self):
        with patch.dict(research_web._state, status='ready'), patch.object(research_web, '_runtime') as runtime:
            runtime.answer.return_value = {'systems': {}}
            response = self.client.post('/api/research/execute', json={
                'query': 'Who?', 'retrieval_mode': 'dense', 'answer': 'Injected label'})
            self.assertEqual(response.status_code, 200)
            runtime.answer.assert_called_once_with('Who?', 'dense')

    def test_simultaneous_request_is_rejected(self):
        with patch.dict(research_web._state, status='ready'), patch.object(research_web, '_runtime', Mock()):
            research_web._lock.acquire()
            try:
                self.assertEqual(self.client.post('/api/research/execute', json={'query': 'Who?'}).status_code, 409)
            finally:
                research_web._lock.release()

    def test_invalid_method_and_blank_question_are_rejected(self):
        for payload in ({'query': '  '}, {'query': 'Who?', 'retrieval_mode': 'oracle'}):
            self.assertEqual(self.client.post('/api/research/execute', json=payload).status_code, 422)

    def test_missing_optional_examples_do_not_break_info(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(research_web, 'latest_dev_run', return_value=Path(folder)), \
                 self.assertLogs(level='WARNING'):
                response = self.client.get('/api/research/info')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['cases'], [])
        self.assertIsNone(response.json()['case_run'])

    def test_corrupt_optional_examples_do_not_break_info(self):
        corrupt_payload = bytearray(gzip.compress(b'{}'))
        corrupt_payload[10] = 0xff
        fixtures = [b'not gzip', gzip.compress(b'{broken json'), gzip.compress(b'{}'),
                    gzip.compress(b'{}')[:-3], bytes(corrupt_payload)]
        for contents in fixtures:
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as folder:
                path = Path(folder)
                (path / 'predictions.jsonl.gz').write_bytes(contents)
                with patch.object(research_web, 'latest_dev_run', return_value=path), \
                     self.assertLogs(level='WARNING'):
                    response = self.client.get('/api/research/info')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['cases'], [])


if __name__ == '__main__':
    unittest.main()
