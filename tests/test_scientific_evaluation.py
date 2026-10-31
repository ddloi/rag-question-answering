"""Regression proof for the scientific defects found during rubric review."""
import unittest
from experiments.metrics import contains_answer, classify_error, score_answer, select_answer_policy, token_support_fraction
from experiments.evaluate import retrieval_summary
from experiments.verify_dataset import repair_document


class ScientificEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.record={'answer':'United States','answer_aliases':['USA'],'gold_doc_id':'gold'}

    def test_official_alias_is_accepted_by_em_and_f1(self):
        self.assertEqual(score_answer('USA',self.record),{'em':1,'f1':1.0})

    def test_canonical_policy_does_not_accept_unreviewed_alias_or_mutate_record(self):
        record = {'answer': 'The Winter’s Tale', 'answer_aliases': ['Mopsa']}
        strict = select_answer_policy(record, 'canonical')
        self.assertEqual(score_answer('Mopsa', strict)['em'], 0)
        self.assertEqual(record['answer_aliases'], ['Mopsa'])
        self.assertEqual(score_answer('Mopsa', select_answer_policy(record, 'official-aliases'))['em'], 1)

    def test_answer_evidence_uses_word_boundaries(self):
        self.assertFalse(contains_answer('Russia',['US']))
        self.assertTrue(contains_answer('in the USA.',['USA']))
        self.assertTrue(contains_answer('Interior angles add up to 720°.', ['720']))

    def test_rank_three_evidence_is_context_selection_failure(self):
        retrieved=[{'doc_id':'x','text':'nothing'},{'doc_id':'y','text':'also nothing'},
                   {'doc_id':'gold','text':'The answer is USA.'}]
        self.assertEqual(classify_error('wrong',self.record,retrieved,'nothing also nothing',True),
                         'context_selection_or_truncation')

    def test_missing_corpus_evidence_is_not_generator_failure(self):
        self.assertEqual(classify_error('wrong',self.record,[],'',False),'corpus_evidence_missing')

    def test_wrong_answer_with_visible_evidence_is_generator_failure(self):
        self.assertEqual(classify_error('wrong',self.record,[{'text':'USA'}],'USA',True),
                         'generation_wrong_with_lexical_evidence')

    def test_lexical_support_is_not_semantic_faithfulness(self):
        self.assertEqual(token_support_fraction('Paris','Paris is not the capital of Germany.'),1)

    def test_retrieval_summary_is_measured_and_respects_rank(self):
        ranked=[{'doc_id':'x','text':'nothing'},{'doc_id':'gold','text':'USA'}]
        row={**self.record,'retrieval':{m:ranked for m in ['dense','bm25','hybrid_rrf']}}
        metrics=retrieval_summary([row],(1,2))
        self.assertEqual(metrics['dense']['doc_recall@1'],0)
        self.assertEqual(metrics['dense']['doc_recall@2'],1)
        self.assertEqual(metrics['hybrid_rrf']['doc_mrr@30'],.5)

    def test_repair_recovers_full_page_without_changing_id_or_qa(self):
        doc={'id':'gold','title':'Example','text':'intro','qa':[{'answer':'USA'}]}
        source={'answer':{'value':'United States','aliases':['USA']},
                'entity_pages':{'title':['Example'],'wiki_context':['intro and the USA evidence']}}
        repaired,aliases,policy=repair_document(doc,source)
        self.assertEqual(policy,'full_original_page')
        self.assertEqual(repaired['id'],doc['id'])
        self.assertEqual(repaired['qa'],doc['qa'])
        self.assertTrue(contains_answer(repaired['text'],aliases))

    def test_unresolved_source_is_preserved_and_flagged(self):
        doc={'id':'gold','title':'Example','text':'intro'}
        source={'answer':{'value':'USA','aliases':[]},'entity_pages':{'title':['Example'],'wiki_context':['unrelated']}}
        repaired,_,policy=repair_document(doc,source)
        self.assertEqual(repaired,doc)
        self.assertEqual(policy,'unresolved_in_official_pages')
