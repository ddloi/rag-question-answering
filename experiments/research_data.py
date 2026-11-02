"""Build an immutable, answer-independent controlled TriviaQA corpus and QA splits."""
import argparse
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ragqa.paths import DATA_DIR, PROJECT_ROOT, resolve_project_path
from .metrics import contains_answer
from .prepare_chunks import chunk_text, clean_text
from .verify_dataset import digest, read_jsonl, write_jsonl
from .research_artifacts import records_sha256


def question_key(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def load_config(path=None):
    path = resolve_project_path(path or 'configs/research_v1.json')
    return json.loads(path.read_text(encoding='utf8')), path


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def stable_id(prefix, text):
    return prefix + hashlib.sha256(text.encode('utf8')).hexdigest()[:24]


def validate_disjoint(splits):
    ids, questions = set(), set()
    for name, rows in splits.items():
        for row in rows:
            qkey = question_key(row['question'])
            if row['id'] in ids or qkey in questions:
                raise ValueError(f'Overlapping question in {name}: {row["id"]}')
            ids.add(row['id']); questions.add(qkey)


def collect_source(config, split, count, excluded_ids, excluded_questions, destination, excluded_hashes=()):
    """Resume append-only raw recovery; selection never inspects answer correctness."""
    rows = read_jsonl(destination) if destination.exists() else []
    if len(rows) > count:
        raise ValueError('Cached source has more rows than configured')
    seen_ids = {r['question_id'] for r in rows}
    seen_questions = {question_key(r['question']) for r in rows}
    if len(seen_ids) != len(rows) or len(seen_questions) != len(rows):
        raise ValueError('Duplicate questions in source cache')
    if seen_ids & excluded_ids or seen_questions & excluded_questions:
        raise ValueError('Source cache conflicts with excluded questions')
    if any(hashlib.sha256(key.encode()).hexdigest() in excluded_hashes for key in seen_questions):
        raise ValueError('Cached validation source includes a legacy question')
    if len(rows) == count:
        return rows
    from datasets import load_dataset
    data = config['dataset']
    source = load_dataset(data['name'], data['config'], split=split, streaming=True, revision=data['revision'])
    with destination.open('a', encoding='utf8') as stream:
        for offset, ex in enumerate(source):
            qid, key = ex['question_id'], question_key(ex['question'])
            if qid in excluded_ids or qid in seen_ids or key in excluded_questions or key in seen_questions:
                continue
            if hashlib.sha256(key.encode()).hexdigest() in excluded_hashes:
                continue
            if not key or not ex['answer']['value'].strip() or not any(ex['entity_pages']['wiki_context']):
                continue
            row = {k: ex[k] for k in ('question_id', 'question', 'answer', 'entity_pages')}
            row.update(source_split=split, source_offset=offset, source_revision=data['revision'])
            stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
            rows.append(row); seen_ids.add(qid); seen_questions.add(key)
            if len(rows) % 100 == 0:
                print(f'SOURCE {split}: {len(rows)}/{count}', flush=True)
            if len(rows) == count:
                return rows
    raise ValueError(f'Insufficient eligible questions in {split}: {len(rows)}/{count}')


def build_records(sources, chunk_words, overlap_words, max_chunks):
    if not 0 <= overlap_words < chunk_words:
        raise ValueError('Require 0 <= overlap < chunk_words')
    documents, questions = {}, {}
    for split, rows in sources.items():
        questions[split] = []
        for row in rows:
            source_docs = []
            pages = row['entity_pages']
            for title, raw_text in zip(pages['title'], pages['wiki_context']):
                text = clean_text(raw_text)
                if not text:
                    continue
                doc_id = stable_id('doc_', text)
                if doc_id not in documents:
                    documents[doc_id] = {'doc_id': doc_id, 'title': title, 'text': text, 'source_titles': [title]}
                elif title not in documents[doc_id]['source_titles']:
                    documents[doc_id]['source_titles'].append(title)
                source_docs.append(doc_id)
            aliases = list(dict.fromkeys(row['answer']['normalized_aliases']))
            if not aliases:
                raise ValueError(f'Missing official normalized aliases: {row["question_id"]}')
            questions[split].append({'id': row['question_id'], 'question': row['question'],
                'answer': row['answer']['value'], 'official_answer_aliases': aliases,
                'gold_doc_ids': sorted(set(source_docs)), 'source_split': row['source_split']})
    validate_disjoint(questions)
    chunks, by_text = [], {}
    for doc in documents.values():
        for text in chunk_text(doc['text'], chunk_words, overlap_words):
            chunk_id = stable_id('chunk_', text)
            if chunk_id in by_text:
                existing = by_text[chunk_id]
                if doc['doc_id'] not in existing['source_doc_ids']:
                    existing['source_doc_ids'].append(doc['doc_id'])
                continue
            chunk = {'chunk_id': chunk_id, 'doc_id': doc['doc_id'], 'source_doc_ids': [doc['doc_id']],
                     'title': doc['title'], 'text': text}
            by_text[chunk_id] = chunk; chunks.append(chunk)
            if len(chunks) > max_chunks:
                raise ValueError('Corpus exceeds configured resource guard; no silent truncation')
    by_doc = defaultdict(list)
    for chunk in chunks:
        for doc_id in chunk['source_doc_ids']:
            by_doc[doc_id].append(chunk)
    accepted, rejected = [], []
    for record in questions['train']:
        candidates = {c['chunk_id']: c for d in record['gold_doc_ids'] for c in by_doc[d]}
        positive = next((c for c in candidates.values() if contains_answer(c['text'], [record['answer']])), None)
        if positive is None:
            rejected.append({**record, 'reason': 'canonical_answer_not_found_in_supplied_evidence'})
        else:
            accepted.append({**record, 'positive_chunk_id': positive['chunk_id']})
    return list(documents.values()), chunks, questions, accepted, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'research_v1')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--build-index', action='store_true')
    parser.add_argument('--reference-manifest', type=Path, help='Replay the frozen config and model revisions into a new output directory')
    args = parser.parse_args()
    reference = None
    if args.reference_manifest:
        if args.config:
            parser.error('Use either --config or --reference-manifest')
        reference = json.loads(resolve_project_path(args.reference_manifest).read_text(encoding='utf8'))
        config = reference['config']
        config_path = None
        config_hash = reference['config_sha256']
    else:
        config, config_path = load_config(args.config)
        config_hash = digest(config_path)
    out = resolve_project_path(args.output)
    manifest_path = out / 'manifest.json'
    if out.exists() and not args.resume:
        raise FileExistsError(f'Preserving existing dataset; use --resume: {out}')
    out.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        if manifest['config_sha256'] != config_hash:
            raise ValueError('Configuration changed; create a new dataset version')
        if manifest['status'] == 'complete':
            raise ValueError('Dataset is frozen and complete; no rebuild in place')
    else:
        if reference is None:
            from huggingface_hub import HfApi
            api = HfApi()
            model_revisions = {key: {'name': name, 'revision': api.model_info(name).sha} for key, name in config['models'].items()}
        else:
            model_revisions = reference['models']
        manifest = {'status': 'building', 'created_at': datetime.now(timezone.utc).isoformat(),
            'version': config['version'], 'config': config, 'config_sha256': config_hash,
            'selection': 'First eligible source rows; validation assignment by seeded question-id hash; no answer-based test selection.',
            'corpus_policy': 'All supplied entity pages; no evidence page selected by test answer presence.',
            'models': model_revisions}
        if reference is not None:
            manifest['reference_manifest_sha256'] = digest(resolve_project_path(args.reference_manifest))
        json_write(manifest_path, manifest)
    try:
        data = config['dataset']
        train_source = collect_source(config, 'train', data['train_candidates'], set(), set(), out / 'source_train.jsonl')
        excluded_ids = {r['question_id'] for r in train_source}
        excluded_questions = {question_key(r['question']) for r in train_source}
        excluded_hashes = set()
        if data['exclude_legacy_from_validation']:
            legacy_path = DATA_DIR / 'provenance/legacy_exclusions.json'
            legacy = json.loads(legacy_path.read_text(encoding='utf8'))
            if legacy['revision'] != data['revision']:
                raise ValueError('Legacy exclusion revision differs from the dataset')
            excluded_ids.update(legacy['question_ids'])
            excluded_hashes.update(legacy['normalized_question_sha256'])
            manifest['legacy_exclusions_sha256'] = digest(legacy_path)
        validation = collect_source(config, 'validation', data['dev'] + data['test'],
                                    excluded_ids, excluded_questions, out / 'source_validation.jsonl', excluded_hashes)
        validation.sort(key=lambda row: hashlib.sha256(f"{config['seed']}:{row['question_id']}".encode()).hexdigest())
        sources = {'train': train_source, 'dev': validation[:data['dev']], 'test': validation[data['dev']:]}
        docs, chunks, splits, accepted, rejected = build_records(sources, data['chunk_words'], data['overlap_words'], data['max_chunks'])
        write_jsonl(out / 'documents.jsonl', docs)
        write_jsonl(out / 'chunk_metadata.jsonl', chunks)
        for split, rows in splits.items():
            write_jsonl(out / f'{split}_qa.jsonl', rows)
        write_jsonl(out / 'training_accepted.jsonl', accepted)
        write_jsonl(out / 'training_rejected.jsonl', rejected)
        manifest.update(counts={'documents': len(docs), 'chunks': len(chunks),
            **{k: len(v) for k, v in splits.items()}, 'training_accepted': len(accepted), 'training_rejected': len(rejected)},
            exact_question_overlap=0, training_positive_rule='Canonical lexical evidence; no first-chunk fallback; same accepted QA for both T5 models.',
            builder_sha256=digest(Path(__file__)), status='corpus_ready')
        json_write(manifest_path, manifest)
        if args.build_index:
            import faiss
            from sentence_transformers import SentenceTransformer
            model_info = manifest['models']['embedding']
            model = SentenceTransformer(model_info['name'], revision=model_info['revision'])
            vectors = model.encode([c['text'] for c in chunks], batch_size=32, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=True).astype('float32')
            index = faiss.IndexFlatIP(vectors.shape[1]); index.add(vectors)
            faiss.write_index(index, str(out / 'faiss.index'))
            manifest.update(embedding_dimension=int(index.d), embedding_max_tokens=model.max_seq_length,
                            status='complete', finished_at=datetime.now(timezone.utc).isoformat())
        manifest['output_sha256'] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        logical_files = {'chunk_metadata.jsonl': chunks, 'train_qa.jsonl': splits['train'],
            'dev_qa.jsonl': splits['dev'], 'test_qa.jsonl': splits['test'], 'training_accepted.jsonl': accepted}
        manifest['records_sha256'] = {name: records_sha256(rows) for name, rows in logical_files.items()}
        if reference is not None:
            reference_hashes = reference.get('records_sha256')
            if reference_hashes is None:
                # Older frozen manifests predate portable record hashing. Keep them immutable.
                descriptor_path = resolve_project_path(args.reference_manifest).parent / 'replay_reference.json'
                descriptor = json.loads(descriptor_path.read_text(encoding='utf8'))
                if descriptor['dataset_manifest_sha256'] != digest(resolve_project_path(args.reference_manifest)):
                    raise ValueError('Replay descriptor does not match the frozen manifest')
                reference_hashes = descriptor['records_sha256']
            manifest['reference_text_checksums_match'] = {name: actual == reference_hashes[name]
                for name, actual in manifest['records_sha256'].items()}
            if not all(manifest['reference_text_checksums_match'].values()):
                raise ValueError('Replay changed frozen question/corpus text; do not compare as the same dataset')
        json_write(manifest_path, manifest)
        print(json.dumps({'status': manifest['status'], 'counts': manifest['counts']}, indent=2), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        json_write(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
