import unittest
from experiments.research_data import build_records, validate_disjoint


def source(qid, question, answer, texts):
    return {'question_id': qid, 'question': question, 'source_split': 'train',
            'answer': {'value': answer, 'normalized_aliases': [answer.lower()]},
            'entity_pages': {'title': [f'Page {i}' for i in range(len(texts))], 'wiki_context': texts}}


class ResearchDataTests(unittest.TestCase):
    def test_train_rejects_unsupported_context_and_test_is_preserved(self):
        sources = {'train': [source('a', 'Capital?', 'Paris', ['France is a country.']),
                             source('b', 'Which city?', 'Paris', ['Paris is in France.'])],
                   'dev': [], 'test': [source('c', 'Unknown city?', 'Missing', ['No answer here.'])]}
        docs, chunks, splits, accepted, rejected = build_records(sources, 20, 4, 100)
        self.assertEqual([r['id'] for r in accepted], ['b'])
        self.assertEqual([r['id'] for r in rejected], ['a'])
        self.assertEqual(len(splits['test']), 1)
        self.assertEqual(len(chunks), 3)

    def test_corpus_keeps_all_pages_even_without_test_answer(self):
        sources = {'train': [], 'dev': [], 'test': [source('c', 'Question?', 'Paris', ['London.', 'Paris.'])]}
        docs, chunks, splits, _, _ = build_records(sources, 20, 4, 100)
        self.assertEqual(len(docs), 2)
        self.assertEqual(len(splits['test'][0]['gold_doc_ids']), 2)

    def test_identical_pages_and_chunks_do_not_duplicate_vectors(self):
        sources = {'train': [source('a', 'City?', 'Paris', ['Paris France.', 'Paris France.'])], 'dev': [], 'test': []}
        docs, chunks, _, accepted, _ = build_records(sources, 20, 4, 100)
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(docs[0]['source_titles']), 2)

    def test_overlapping_questions_fail_instead_of_leaking(self):
        with self.assertRaises(ValueError):
            validate_disjoint({'train': [{'id': 'a', 'question': 'Who?'}],
                               'test': [{'id': 'b', 'question': 'who?'}]})


if __name__ == '__main__':
    unittest.main()
