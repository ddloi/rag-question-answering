"""Build a demo-only corpus with source-documented extra documents.

These two pages were selected after examining test questions; demo_v3 must not
be used as an independent test corpus or compared with verified_v2 metrics.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from experiments.prepare_chunks import chunk_text, clean_text
from experiments.verify_dataset import digest, read_jsonl, write_jsonl
from ragqa.paths import DATA_DIR


def main():
    import faiss
    from sentence_transformers import SentenceTransformer
    source = DATA_DIR / 'verified_v2'
    output = DATA_DIR / 'demo_v3'
    if output.exists():
        raise FileExistsError(f'Preserving existing demo corpus: {output}')
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
    for filename in ('chunk_metadata.jsonl','faiss.index','test_qa.jsonl'):
        if digest(source/filename) != manifest['output_sha256'][filename]:
            raise ValueError(f'Frozen source checksum mismatch: {filename}')
    original = read_jsonl(source/'chunk_metadata.jsonl')
    index = faiss.read_index(str(source/'faiss.index'))
    if len(original) != index.ntotal:
        raise ValueError('Metadata/index count mismatch')
    sources = []
    added = []
    for filename, doc_id in [('knucklebones.json','demo_knucklebones'),('winters_tale.json','demo_winters_tale')]:
        path = DATA_DIR/'demo_sources'/filename
        page = json.loads(path.read_text(encoding='utf8'))
        sources.append({k:page[k] for k in ('title','page_id','revision','source_url','retrieved_utc')})
        sources[-1]['sha256'] = digest(path)
        for chunk in chunk_text(clean_text(page['extract']),200,40):
            added.append({'chunk_id':len(original)+len(added), 'doc_id':doc_id,
                          'title':page['title'], 'text':page['title'] + '. ' + chunk,
                          'source_url':page['source_url']})
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True)
    vectors = model.encode([r['text'] for r in added],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
    if vectors.shape[1] != index.d:
        raise ValueError('Embedding dimension differs from the original index')
    index.add(vectors)
    output.mkdir(parents=True)
    write_jsonl(output/'chunk_metadata.jsonl',original+added)
    shutil.copyfile(source/'test_qa.jsonl',output/'test_qa.jsonl')
    faiss.write_index(index,str(output/'faiss.index'))
    info={'dataset_version':'demo_v3','created_at':datetime.now(timezone.utc).isoformat(),
          'purpose':'Demo only: extra pages selected after seeing test questions; not a held-out evaluation corpus.',
          'base_dataset':'verified_v2','base_manifest_sha256':digest(source/'manifest.json'),
          'base_chunks':len(original),'additional_chunks':len(added),'sources':sources,
          'output_sha256':{p.name:digest(p) for p in output.iterdir() if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'base':len(original),'added':len(added),'total':index.ntotal,'sources':sources},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
