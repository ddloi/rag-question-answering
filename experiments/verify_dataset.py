"""Create a frozen evidence-repaired corpus alongside the original, never drop QA rows.

Official TriviaQA aliases and complete evidence pages are recovered at a pinned
revision. If no alias occurs in the original excerpt, use the complete original
page, or the first supplied evidence page containing an official alias. This is
an explicitly answer-informed construction of a controlled corpus, not a
full-Wikipedia retrieval benchmark. No hyperparameters are tuned on test scores.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ragqa.paths import DATA_DIR, RAW_DOCUMENTS_PATH
from .metrics import answers_for, contains_answer, normalize_answer
from .prepare_chunks import chunk_text, clean_text


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def repair_document(doc, official):
    aliases = list(dict.fromkeys([official['answer']['value'], *official['answer']['aliases']]))
    pages = official['entity_pages']
    text, title, policy = doc['text'], doc['title'], 'original_excerpt'
    # Freeze the conservative expansion policy separately from unit-aware audit.
    def selection_match(value):
        normalized = ' ' + normalize_answer(value) + ' '
        return any(normalize_answer(a) and ' ' + normalize_answer(a) + ' ' in normalized for a in aliases)
    if not selection_match(text):
        for i, candidate in enumerate(pages['wiki_context']):
            if selection_match(candidate):
                text, title = candidate, pages['title'][i]
                policy = 'full_original_page' if i == 0 else 'alternate_official_evidence_page'
                break
        else:
            policy = ('original_excerpt_unit_boundary_verified' if contains_answer(text, aliases)
                      else 'unresolved_in_official_pages')
    return {**doc, 'text': text, 'title': title}, aliases, policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'verified_v2')
    parser.add_argument('--build-index', action='store_true')
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(f"Preserving frozen dataset: {out}")
    official_path = DATA_DIR / 'provenance/triviaqa_official.jsonl'
    official = {r['doc_id']: r for r in read_jsonl(official_path)}
    raw = json.loads(RAW_DOCUMENTS_PATH.read_text(encoding='utf-8'))
    if set(official) != {d['id'] for d in raw}:
        raise ValueError('Official recovery does not cover the original document IDs')
    out.mkdir(parents=True)
    repaired, audit, aliases_by_doc = [], [], {}
    for doc in raw:
        updated, aliases, policy = repair_document(doc, official[doc['id']])
        repaired.append(updated)
        aliases_by_doc[doc['id']] = aliases
        audit.append({'doc_id':doc['id'], 'question':doc['qa'][0]['question'],
                      'answer':doc['qa'][0]['answer'], 'answer_aliases':aliases,
                      'original_title':doc['title'], 'new_title':updated['title'], 'policy':policy,
                      'source_question_id':official[doc['id']]['source_id'],
                      'old_chars':len(doc['text']), 'new_chars':len(updated['text']),
                      'original_has_alias':contains_answer(doc['text'],aliases),
                      'repaired_has_alias':contains_answer(updated['text'],aliases)})
    chunks=[]
    for doc in repaired:
        for text in chunk_text(clean_text(doc['text']),200,40):
            chunks.append({'chunk_id':len(chunks),'doc_id':doc['id'],'title':doc['title'],'text':text})
    by_doc={}
    for c in chunks: by_doc.setdefault(c['doc_id'],[]).append(c['text'])
    test=[]
    old_test=read_jsonl(DATA_DIR/'test_qa.jsonl')
    training=read_jsonl(DATA_DIR/'training_pairs.jsonl')
    train_keys={normalize_answer(r['question']) for r in training}
    if any(normalize_answer(r['question']) in train_keys for r in old_test):
        raise ValueError('Train/test question overlap')
    for row in old_test:
        aliases=aliases_by_doc[row['gold_doc_id']]
        test.append({**row,'answer_aliases':aliases,
                     'source_question_id':official[row['gold_doc_id']]['source_id'],
                     'gold_source_has_alias':any(contains_answer(t,aliases) for t in by_doc[row['gold_doc_id']])})
    write_jsonl(out/'corpus_chunks.jsonl',chunks)
    write_jsonl(out/'chunk_metadata.jsonl',chunks)
    write_jsonl(out/'test_qa.jsonl',test)
    write_jsonl(out/'source_audit.jsonl',audit)
    (out/'raw_documents.json').write_text(json.dumps(repaired,ensure_ascii=False),encoding='utf-8')
    manifest={'dataset_version':'verified_v2','created_at':datetime.now(timezone.utc).isoformat(),
              'source_dataset':'mandarjoshi/trivia_qa','source_config':'rc.wikipedia','source_split':'validation',
              'source_revision':next(iter(official.values()))['revision'],
              'policy':__doc__, 'chunk_size':200,'overlap':40,'counts':{'documents':len(raw),'chunks':len(chunks),'test':len(test),
              'test_with_gold_alias_in_source':sum(r['gold_source_has_alias'] for r in test)},
              'repair_policies':dict(Counter(r['policy'] for r in audit)),
              'question_overlap_train_test':0,
              'checkpoint_note':'Existing checkpoint trained on ORIGINAL training_pairs; no retraining or test-label training.',
              'input_sha256':{str(p.relative_to(DATA_DIR)):digest(p) for p in [RAW_DOCUMENTS_PATH,official_path,DATA_DIR/'test_qa.jsonl',DATA_DIR/'training_pairs.jsonl']}}
    if args.build_index:
        import torch,faiss
        from sentence_transformers import SentenceTransformer
        name='sentence-transformers/all-MiniLM-L6-v2'
        model=SentenceTransformer(name,device='cuda' if torch.cuda.is_available() else 'cpu',local_files_only=True)
        vectors=model.encode([c['text'] for c in chunks],batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=True).astype('float32')
        index=faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
        faiss.write_index(index,str(out/'faiss.index'))
        manifest['embedding_model']=name
        manifest['embedding_dimension']=int(vectors.shape[1])
    manifest['output_sha256']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['counts','repair_policies']},indent=2),flush=True)


if __name__=='__main__':
    main()
