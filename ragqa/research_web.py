"""Research demo adapter: the same artifacts and inference classes as evaluation."""
import gzip
import json
import logging
import threading
import zlib
from pathlib import Path
from typing import Literal
from functools import lru_cache
import hashlib
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, resolve_project_path

router = APIRouter()
DATA = resolve_project_path(os.environ.get('RAG_RESEARCH_DATA_DIR', str(DATA_DIR / 'research_v1')))
MODELS = PROJECT_ROOT / 'checkpoints/research_v1'
STATIC = Path(__file__).parent / 'static'
_state = {'status': 'idle', 'error': None}
_runtime = None
_lock = threading.Lock()


def available():
    try:
        dataset = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))
        if dataset['status'] != 'complete':
            return False
        for system in ('closed_book', 'rag'):
            model = MODELS / system
            info = json.loads((model / 'manifest.json').read_text(encoding='utf8'))
            if info['status'] != 'complete' or info['pilot'] or not (model / 'model/model.safetensors').is_file():
                return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def latest_dev_run():
    try:
        expected_data = hashlib.sha256((DATA / 'manifest.json').read_bytes()).hexdigest()
        model_hashes = [json.loads((MODELS / system / 'manifest.json').read_text(encoding='utf8')).get('model_sha256')
                        for system in ('closed_book', 'rag')]
    except (OSError, ValueError):
        return None
    candidates = []
    for path in RESULTS_DIR.glob('research_*/manifest.json'):
        try:
            info = json.loads(path.read_text(encoding='utf8'))
            if (info['status'] == 'complete' and info['split'] == 'dev' and not info['pilot'] and info['n'] == 200
                    and info['dataset_sha256'] == expected_data
                    and info['closed_book_model_sha256'] == model_hashes[0] and info['rag_model_sha256'] == model_hashes[1]):
                candidates.append((info['finished_at'], path.parent))
        except (OSError, ValueError, KeyError):
            continue
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def dev_cases():
    path = latest_dev_run()
    if path is None:
        return [], None
    try:
        return cached_dev_cases(path)
    except (OSError, EOFError, UnicodeError, ValueError, KeyError, TypeError, zlib.error):
        logging.warning('Cannot read optional dev examples from %s', path, exc_info=True)
        return [], None


@lru_cache(maxsize=2)
def cached_dev_cases(path):
    with gzip.open(path / 'predictions.jsonl.gz', 'rt', encoding='utf8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    cases, seen = [], set()
    for row in rows:
        category = row['errors']['rag_full']['canonical']
        dense = row['systems']['rag_full']['scores']['canonical']['em']
        hybrid = row['systems']['rag_hybrid']['scores']['canonical']['em']
        categories = [category]
        if hybrid > dense:
            categories.append('hybrid_rescue')
        if hybrid < dense:
            categories.append('hybrid_regression')
        if row['systems']['rag_full']['scores']['official']['em'] != dense:
            categories.append('official_canonical_disagreement')
        for kind in categories:
            if kind in seen:
                continue
            seen.add(kind)
            cases.append({'id': row['record']['id'], 'question': row['record']['question'],
                          'category': kind, 'snapshot_answer': row['systems']['rag_full']['answer']})
    return cases, path.name


class ResearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)
    retrieval_mode: Literal['dense', 'hybrid'] = 'dense'


class ResearchRuntime:
    def __init__(self):
        from experiments.research_pipeline import dataset_manifest, ResearchRetriever, ResearchGenerator, ResearchExtractor
        from experiments.research_evaluate import model_run
        from experiments.verify_dataset import digest, read_jsonl
        self.info = dataset_manifest(DATA)
        self.data_hash = digest(DATA / 'manifest.json')
        left, right = [model_run(MODELS / system, self.data_hash, system) for system in ('closed_book', 'rag')]
        for key in ('train_ids', 'base_model', 'config', 'precision', 'prompt_template', 'source_sha256'):
            if left[key] != right[key]:
                raise ValueError(f'Training pair mismatch: {key}')
        self.config = self.info['config']
        self.retriever = ResearchRetriever(DATA)
        self.closed = ResearchGenerator(MODELS / 'closed_book/model', self.config['generation'])
        self.rag = ResearchGenerator(MODELS / 'rag/model', self.config['generation'])
        self.extractor = ResearchExtractor(self.info['models']['extractive'])
        # Demo examples and labels are dev-only. Labels are attached after inference.
        self.references = {r['question'].strip().casefold(): r for r in read_jsonl(DATA / 'dev_qa.jsonl')}
        audit_path = RESULTS_DIR / 'research_v1_label_audit_dev.json'
        self.label_audit = json.loads(audit_path.read_text(encoding='utf8')) if audit_path.exists() else {}

    def answer(self, query, mode):
        from experiments.metrics import score_answer
        rankings = self.retriever.all_rankings(query)
        options = self.config['retrieval']
        outputs = {
            'generator_only': self.closed.answer(query),
            'retrieval_only': self.extractor.answer(query, rankings['dense'][:options['top_k']]),
            'rag': self.rag.answer(query, rankings[mode][:options['context_top_n']]),
        }
        reference = self.references.get(query.casefold())
        for output in outputs.values():
            output['scores'] = ({p: score_answer(output['answer'], reference, p) for p in ('official', 'canonical')}
                                if reference else None)
        reference_result = ({'answer': reference['answer'], 'official_answer_aliases': reference['official_answer_aliases'],
                             'split': 'dev'} if reference else None)
        if (reference and self.label_audit.get('question_id') == reference['id']
                and self.label_audit.get('dataset_manifest_sha256') == self.data_hash):
            reference_result['audit_note'] = ('Nhãn nguồn có cờ rà soát; xem casebook và nguồn đối chiếu. '
                                             'Điểm vẫn giữ nhãn frozen, chưa dùng đề xuất để chấm lại.')
            reference_result['audit_source'] = self.label_audit['primary_source']
        return {'dataset_version': 'research_v1', 'dataset_sha256': self.data_hash,
                'query': query, 'retrieval_mode': mode, 'systems': outputs,
                'retrieval': {m: r[:10] for m, r in rankings.items()},
                'reference': reference_result}


def load_runtime():
    global _runtime
    try:
        runtime = ResearchRuntime()
        _runtime = runtime
        _state.update(status='ready', error=None)
    except Exception:
        logging.exception('Research model loading failed')
        _state.update(status='error', error='Chưa nạp được bộ model nghiên cứu. Kiểm tra artifact và log máy chủ rồi thử lại.')


@router.get('/research')
def research_page():
    return FileResponse(STATIC / 'research.html')


@router.get('/api/research/info')
def info():
    counts = {}
    training_qa = None
    try:
        counts = json.loads((DATA / 'manifest.json').read_text(encoding='utf8'))['counts']
    except (OSError, ValueError, KeyError):
        pass
    try:
        model = json.loads((MODELS / 'rag/manifest.json').read_text(encoding='utf8'))
        if not model['pilot']:
            training_qa = len(model['train_ids'])
    except (OSError, ValueError, KeyError):
        pass
    cases, run = dev_cases()
    return {**_state, 'available': available(), 'dataset_version': 'research_v1', 'counts': counts,
            'cases': cases, 'case_run': run, 'training_qa': training_qa,
            'device': _runtime.rag.device if _runtime else None}


@router.post('/api/research/load', status_code=202)
def load():
    with _lock:
        if not available():
            raise HTTPException(409, 'Hai model research_v1 chưa hoàn tất hoặc thiếu artifact. Xem tiến độ nghiên cứu.')
        if _state['status'] not in ('loading', 'ready'):
            _state.update(status='loading', error=None)
            threading.Thread(target=load_runtime, daemon=True).start()
    return dict(_state)


@router.post('/api/research/execute')
def execute(req: ResearchRequest):
    if _state['status'] != 'ready' or _runtime is None:
        raise HTTPException(409, 'Cần nạp đủ ba hệ trước khi chạy.')
    # Serialize GPU inference; fail clearly instead of silently mixing requests.
    if not _lock.acquire(blocking=False):
        raise HTTPException(409, 'Đang xử lý câu hỏi khác. Chờ kết quả rồi thử lại.')
    try:
        return _runtime.answer(req.query, req.retrieval_mode)
    except Exception as exc:
        logging.exception('Research inference failed')
        raise HTTPException(500, 'Không chạy được câu hỏi. Kiểm tra log máy chủ và thử lại.') from exc
    finally:
        _lock.release()
