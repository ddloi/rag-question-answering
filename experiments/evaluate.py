"""Reproducible three-arm QA evaluation with per-question evidence and measured retrieval."""
import argparse
import csv
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ragqa.paths import DATA_DIR, MODEL_DIR, PROJECT_ROOT, RESULTS_DIR, resolve_project_path
from .metrics import (answers_for, classify_error, contains_answer, exact_match_score as exact_match_score,
                      f1_score as f1_score, normalize_answer, score_answer, select_answer_policy, token_support_fraction)
from .verify_dataset import digest, read_jsonl


def faithfulness_score(prediction, context):
    """Compatibility name: lexical token overlap only, not semantic faithfulness."""
    return token_support_fraction(prediction, context)


def citation_correctness_score(gold_answer, gold_doc_id, retrieved_passages):
    """Compatibility: source/evidence proxies, not entailment or citation precision."""
    return {"doc_match": int(any(p['doc_id'] == gold_doc_id for p in retrieved_passages)),
            "evidence_match": int(any(contains_answer(p['text'], [gold_answer]) for p in retrieved_passages))}


def retrieve_all(rag, bm25, question, pool=30):
    import numpy as np
    dense = rag.retrieve(question, pool)
    scores = bm25.get_scores(question.lower().split())
    sparse_ids = [int(i) for i in np.argsort(-scores, kind='stable') if scores[i] > 0][:pool]
    sparse = [{**rag.metadata[i], 'score': float(scores[i])} for i in sparse_ids]
    index_by_chunk = {p['chunk_id']: i for i,p in enumerate(rag.metadata)}
    dense_ids = [index_by_chunk[p['chunk_id']] for p in dense]
    fusion = {}
    for ranked in [sparse_ids, dense_ids]:
        for rank, i in enumerate(ranked, 1): fusion[i] = fusion.get(i, 0.) + 1 / (60 + rank)
    hybrid = [{**rag.metadata[i], 'score': float(score)} for i,score in sorted(fusion.items(), key=lambda pair:(-pair[1], pair[0]))[:pool]]
    return {'dense':dense, 'bm25':sparse, 'hybrid_rrf':hybrid}


def retrieval_summary(rows, k_values=(3,5,10)):
    output={}
    for method in ['dense','bm25','hybrid_rrf']:
        scores={}
        for k in k_values:
            scores[f'doc_recall@{k}']=sum(any(p['doc_id']==r['gold_doc_id'] for p in r['retrieval'][method][:k]) for r in rows)/len(rows)
            scores[f'answer_alias_recall@{k}']=sum(any(contains_answer(p['text'],r['answer_aliases']) for p in r['retrieval'][method][:k]) for r in rows)/len(rows)
        scores['doc_mrr@30']=sum(next((1/(i+1) for i,p in enumerate(r['retrieval'][method]) if p['doc_id']==r['gold_doc_id']),0.) for r in rows)/len(rows)
        output[method]=scores
    return output


def evaluate_end_to_end(rag, qa_records, writer=None, top_k=5):
    from rank_bm25 import BM25Okapi
    bm25=BM25Okapi([p['text'].lower().split() for p in rag.metadata],k1=1.5,b=.75)
    rows=[]
    corpus_normalized=[' '+normalize_answer(p['text'])+' ' for p in rag.metadata]
    for i,record in enumerate(qa_records):
        aliases=answers_for(record)
        corpus_support=any(' '+normalize_answer(a)+' ' in text for a in aliases if normalize_answer(a) for text in corpus_normalized)
        started=time.perf_counter()
        retrieval=retrieve_all(rag,bm25,record['question'])
        dense=retrieval['dense'][:top_k]
        rag_result=rag.generate_with_evidence(record['question'],dense)
        extractive=rag.answer_retrieval_only(record['question'],dense)
        generator_only=rag.answer_generator_only(record['question'])
        systems={}
        for name,result in [('rag_full',rag_result),('retrieval_only',extractive),('generator_only',{'answer':generator_only,'context_used':'','used_passages':[]})]:
            systems[name]={**result, **score_answer(result['answer'],record)}
            if name!='generator_only':
                systems[name]['token_support_fraction']=token_support_fraction(result['answer'],result['context_used'])
                systems[name]['used_context_has_gold_alias']=contains_answer(result['context_used'],aliases)
                systems[name]['source_doc_hit_in_used_passages']=any(p['doc_id']==record['gold_doc_id'] for p in result['used_passages'])
        row={'id':i,'question':record['question'],'answer':record['answer'],'answer_aliases':aliases,
             'gold_doc_id':record['gold_doc_id'],'corpus_has_answer_alias':corpus_support,
             'retrieval':retrieval,'systems':systems,
             'error_category':classify_error(rag_result['answer'],record,dense,rag_result['context_used'],corpus_support),
             'latency_ms':round((time.perf_counter()-started)*1000,2)}
        rows.append(row)
        if writer:
            writer.write(json.dumps(row,ensure_ascii=False)+'\n'); writer.flush()
        print(f"[{i+1}/{len(qa_records)}] RAG EM={systems['rag_full']['em']} category={row['error_category']}",flush=True)
    metrics={name:{metric:sum(r['systems'][name][metric] for r in rows)/len(rows) for metric in ['em','f1']} for name in ['generator_only','retrieval_only','rag_full']}
    proxies={name:{metric:sum(r['systems'][name][metric] for r in rows)/len(rows) for metric in ['token_support_fraction','used_context_has_gold_alias','source_doc_hit_in_used_passages']} for name in ['retrieval_only','rag_full']}
    return rows,{'n':len(rows),'qa':metrics,'retrieval':retrieval_summary(rows),'evidence_proxies':proxies,'error_breakdown':dict(Counter(r['error_category'] for r in rows))}


def export_review(rows,path,seed=42,size=20):
    """Freeze sample selection independently of scores; leave judgments blank."""
    selected=sorted(random.Random(seed).sample(range(len(rows)),min(size,len(rows))))
    fields=['id','question','prediction','gold_answer','used_context','used_doc_ids','faithfulness_label','citation_label','rationale','reviewer_kind']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for i in selected:
            r=rows[i]; arm=r['systems']['rag_full']
            writer.writerow({'id':r['id'],'question':r['question'],'prediction':arm['answer'],'gold_answer':r['answer'],
                             'used_context':arm['context_used'],'used_doc_ids':';'.join(p['doc_id'] for p in arm['used_passages'])})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DATA_DIR/'verified_v2')
    parser.add_argument('--samples',type=int)
    parser.add_argument('--k',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--answer-policy', choices=('canonical', 'official-aliases'), default='canonical',
                        help='canonical is conservative; official aliases require human review')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.samples is not None and args.samples<1: parser.error('--samples must be positive')
    if args.k<1 or args.k>30: parser.error('--k must be from 1 to 30')
    data_dir=resolve_project_path(args.data_dir)
    records=read_jsonl(data_dir/'test_qa.jsonl')
    records = [select_answer_policy(record, args.answer_policy) for record in records]
    if args.samples: records=records[:args.samples]
    if not records: parser.error('Empty test split')
    run_dir=resolve_project_path(args.run_dir) if args.run_dir else RESULTS_DIR/('evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run_dir.mkdir(parents=True,exist_ok=False)
    from transformers import set_seed
    set_seed(args.seed)
    from .rag_pipeline import RAGSystem, GENERATION_OPTIONS
    try:
        dataset_location = data_dir.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        dataset_location = '<external dataset>'
    def artifact_label(path):
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return '<external dataset>/' + path.name
    metadata={'status':'running','created_at':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'data_dir':dataset_location,
              'samples':len(records),'top_k':args.k,'answer_policy':args.answer_policy,'generation':GENERATION_OPTIONS,'context_top_n':2,'max_input_tokens':512,
              'neural_relevance_gate':False,'generator_only':'same fine-tuned checkpoint, context removed; not separately trained closed-book',
              'citation_note':'Used passages are evidence candidates, not generated inline citations. Proxies are NOT semantic faithfulness.',
              'manual_review':'sample seed 42, 20 rows selected before review; judgments are separate',
              'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','sentence-transformers','faiss-cpu','rank-bm25']},
              'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),
              'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT_ROOT,text=True).strip()),
              'sha256':{artifact_label(p):digest(p) for p in [data_dir/'test_qa.jsonl',data_dir/'chunk_metadata.jsonl',data_dir/'faiss.index',MODEL_DIR/'model.safetensors',DATA_DIR/'training_pairs.jsonl',PROJECT_ROOT/'experiments/rag_pipeline.py',PROJECT_ROOT/'experiments/evaluate.py',PROJECT_ROOT/'experiments/metrics.py',PROJECT_ROOT/'ragqa/rag_engine.py']}}
    def save_manifest(): (run_dir/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    save_manifest()
    try:
        rag=RAGSystem(data_dir=data_dir,top_k=args.k)
        metadata['device']=rag.device_name
        with (run_dir/'predictions.jsonl').open('w',encoding='utf-8') as writer:
            rows,summary=evaluate_end_to_end(rag,records,writer,args.k)
        (run_dir/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        export_review(rows,run_dir/'review_sample.csv')
        metadata['status']='complete'; metadata['finished_at']=datetime.now(timezone.utc).isoformat()
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
        print('Saved:',run_dir,flush=True)
    except Exception as exc:
        metadata['status']='failed'; metadata['error']=str(exc)
        raise
    finally: save_manifest()


if __name__=='__main__': main()
