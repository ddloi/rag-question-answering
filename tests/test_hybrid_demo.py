"""Hybrid retrieval proof without ML libraries, model downloads or GPU."""
import unittest
from unittest.mock import Mock

from ragqa.rag_engine import RAGNeuralEngine
from ragqa.web import rare_query_anchor


class FakeVector:
    def __matmul__(self, other):
        return 0.4


class HybridDemoTests(unittest.TestCase):
    def setUp(self):
        self.engine = RAGNeuralEngine()
        self.engine.is_loaded = True
        self.engine.metadata = [
            dict(chunk_id=10, doc_id='general', title='General', text='General science.'),
            dict(chunk_id=11, doc_id='plural', title='Plural', text='Quarks and physics.'),
            dict(chunk_id=12, doc_id='specific', title='Specific', text='A quark is a particle.'),
        ]
        self.engine._chunk_positions = {10: 0, 11: 1, 12: 2}
        embedding = Mock()
        embedding.astype.return_value = [[1.0]]
        self.engine.embed_model = Mock()
        self.engine.embed_model.encode.return_value = embedding
        self.engine.index = Mock()
        self.engine.index.search.return_value = ([[0.9, 0.8]], [[0, 1]])
        self.engine.index.reconstruct.return_value = FakeVector()
        self.sparse = [dict(self.engine.metadata[1], score=99),
                       dict(self.engine.metadata[2], score=88)]

    def test_default_dense_order_is_preserved(self):
        result = self.engine.retrieve('quark science', 2)
        self.assertEqual([r['chunk_id'] for r in result], [10, 11])
        self.engine.index.reconstruct.assert_not_called()

    def test_hybrid_recovers_sparse_evidence_and_keeps_real_cosine(self):
        result = self.engine.retrieve('quark science', 3, self.sparse, 'quark')
        self.assertEqual(result[0]['chunk_id'], 12)
        self.assertEqual(result[0]['score'], 0.4)
        self.assertTrue(result[0]['matched_anchor'])
        self.assertFalse(next(r for r in result if r['chunk_id'] == 11)['matched_anchor'])
        self.assertEqual(len({r['chunk_id'] for r in result}), 3)
        self.assertEqual([r['rank'] for r in result], [1, 2, 3])

    def test_rrf_uses_both_rankings_without_an_anchor(self):
        result = self.engine.retrieve('science', 2, self.sparse)
        self.assertEqual(result[0]['chunk_id'], 11)
        self.assertAlmostEqual(result[0]['fusion_score'], 1 / 62 + 1 / 61, places=6)

    def test_anchor_requires_a_distinctly_rare_existing_query_term(self):
        def term(name, df, idf):
            return dict(term=name, df=df, idf=idf, active=True)
        rows = [term('quark', 4, 8), term('science', 150, 4), term('unknown', 0, 10)]
        self.assertEqual(rare_query_anchor({'term_analysis': rows}, 10000), 'quark')
        rows[0]['idf'] = 4.2
        self.assertIsNone(rare_query_anchor({'term_analysis': rows}, 10000))


if __name__ == '__main__':
    unittest.main()
