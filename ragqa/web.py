import json
import logging
import time
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .engine import DeterministicPipelineEngine
from .rag_engine import RAGNeuralEngine
from .paths import DATA_DIR

app = FastAPI(
    title="RAG Lab — Nghiên cứu & So sánh Hỏi-Đáp",
    description="So sánh trực tiếp BM25 Retrieval với Hệ thống RAG (FAISS Dense + T5 Generator) trên TriviaQA."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Hugging Face Space)
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def select_demo_data_dir():
    for version in ('demo_v3', 'verified_v2'):
        candidate = DATA_DIR / version
        if all((candidate / name).is_file() for name in
               ('manifest.json', 'faiss.index', 'chunk_metadata.jsonl', 'test_qa.jsonl')):
            return candidate
    return DATA_DIR


DEMO_DATA_DIR = select_demo_data_dir()
rag_neural_engine = RAGNeuralEngine(data_dir=DEMO_DATA_DIR)
print("[Server] Initializing Deterministic Pipeline Engine...")
engine = DeterministicPipelineEngine(DEMO_DATA_DIR)
source_urls = {row['chunk_id']: row['source_url'] for row in engine.metadata if row.get('source_url')}
print("[Server] Engine loaded and ready.")

# Neural models load only on request, so the lightweight demo and tests do
# not import torch, allocate GPU memory or access the model registry at startup.

# Load sample QA records from data/test_qa.jsonl
test_qa_path = DEMO_DATA_DIR / "test_qa.jsonl"
sample_records = []
if test_qa_path.exists():
    with open(test_qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample_records.append(json.loads(line))

class PipelineRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10, strict=True)
    k1: float = Field(default=1.5, ge=0.5, le=3.0, allow_inf_nan=False)
    b: float = Field(default=0.75, ge=0.0, le=1.0, allow_inf_nan=False)
    run_rag: bool = Field(default=True)
    retrieval_mode: Literal['dense', 'hybrid'] = 'hybrid'


def rare_query_anchor(normalization, total_chunks):
    """A corpus-based heuristic; no reference answers or question-specific rules."""
    terms = sorted((row for row in normalization['term_analysis']
                    if row['active'] and row['df'] > 0 and len(row['term']) >= 3),
                   key=lambda row: row['idf'], reverse=True)
    if (len(terms) >= 2 and terms[0]['df'] <= total_chunks * 0.01
            and terms[0]['idf'] >= 1.5 * terms[1]['idf']):
        return terms[0]['term']
    return None

@app.get("/")
def get_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/pipeline/info")
def get_info():
    rag_avail = rag_neural_engine.is_available()
    return {
        "status": "online",
        "dataset_version": DEMO_DATA_DIR.name if DEMO_DATA_DIR != DATA_DIR else "original",
        "demo_only_corpus": DEMO_DATA_DIR.name == 'demo_v3',
        "engine_type": "BM25 (Sparse) + FAISS (Dense) + T5-base Generator",
        "ai_elements": rag_neural_engine.is_loaded,
        "total_chunks": engine.total_chunks,
        "average_doc_length": round(engine.avgdl, 1),
        "vocabulary_size": len(engine.doc_freqs),
        "execution_mode": "Dual Comparison (BM25 vs FAISS + T5 RAG)",
        "confidence_type": "lexical_heuristic",
        "corpus_language": "en",
        "generator_enabled": rag_neural_engine.is_loaded,
        "rag_status": ("ready" if rag_neural_engine.is_loaded else
                       "loading" if rag_neural_engine.is_loading else
                       "error" if rag_neural_engine.load_error else
                       "idle" if rag_avail else "unavailable"),
        "rag_error": rag_neural_engine.load_error,
        "rag_device": rag_neural_engine.device_name if rag_neural_engine.is_loaded else ("Sẵn sàng tải" if rag_avail else "Không khả dụng"),
        "rag_available": rag_avail
    }

@app.get("/api/pipeline/samples")
def get_samples():
    samples = []
    cases = [
        ('Nullarbor Plain', 'correct', 'Lần đo trước: trả Australia, khớp đáp án.'),
        ('Melbourne Cup in 1930', 'correct', 'Lần đo trước: trả Phar Lap, khớp đáp án.'),
        ('mechanical shark', 'generation', 'Lần đo trước: có bằng chứng Bruce nhưng T5 trả Bruce Ramer.'),
        ('Et Tu Brute', 'generation', 'Lần đo trước: T5 trả Jesus dù câu hỏi nói về Julius Caesar.'),
        ('Fivestones', 'missing', 'Lần đo trước thiếu từ khóa; nay có thể kiểm tra nguồn bổ sung.'),
        ('Antigonus', 'missing', 'Lần đo trước thiếu tên nhân vật; nay có thể kiểm tra nguồn bổ sung.'),
        ('animal was Robin Hood', 'context', 'Lần đo trước: context không đủ rõ; T5 trả Animal thay vì Fox.'),
        ('first Beatles film', 'context', 'Lần đo trước: T5 trả Magical Mystery Tour thay vì A Hard Day’s Night.'),
    ]
    for needle, category, note in cases:
        r = next((item for item in sample_records if needle.casefold() in item.get('question', '').casefold()), None)
        if r is None:
            continue
        samples.append({
            "question": r.get("question", ""),
            "ground_truth": r.get("answer", ""),
            "doc_id": r.get("gold_doc_id", ""),
            "category": category, "case_note": note,
        })
    return {"samples": samples}

@app.post("/api/rag/load", status_code=202)
def load_rag_now():
    """Start or retry loading; clients poll info instead of holding a request."""
    if not rag_neural_engine.is_available():
        raise HTTPException(status_code=400, detail="Thiếu FAISS, metadata hoặc checkpoint T5 đã fine-tune. Xem hướng dẫn demo.")
    if not rag_neural_engine.is_loaded:
        rag_neural_engine.load_in_background()
    return {
        "status": "ready" if rag_neural_engine.is_loaded else "loading",
        "device": rag_neural_engine.device_name,
        "error": rag_neural_engine.load_error
    }

@app.post("/api/pipeline/execute")
def execute_pipeline(req: PipelineRequest):
    started = time.perf_counter()
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        res = engine.execute_pipeline(
            query=req.query,
            top_k=req.top_k,
            k1=req.k1,
            b=req.b
        )

        rag_res = None
        if req.run_rag:
            try:
                missing = [term for term in res['stages'][0]['active_terms']
                           if len(term) >= 4 and term not in engine.doc_freqs]
                if not rag_neural_engine.is_available():
                    rag_res = {"available": False, "error": "Thiếu FAISS, metadata hoặc checkpoint T5 đã fine-tune. BM25 vẫn hoạt động."}
                else:
                    if (not rag_neural_engine.is_loaded and not rag_neural_engine.is_loading
                            and not rag_neural_engine.load_error):
                        rag_neural_engine.load_in_background()
                    options = {}
                    if req.retrieval_mode == 'hybrid':
                        options['sparse_candidates'] = engine.stage2_mathematical_bm25_search(
                            res['stages'][0]['active_terms'], top_k=30, k1=req.k1, b=req.b)['ranked_chunks']
                        options['anchor_term'] = rare_query_anchor(res['stages'][0], engine.total_chunks)
                    rag_res = rag_neural_engine.execute_rag(query=req.query, top_k=req.top_k, **options)
                if missing and rag_res is not None:
                    rag_res['missing_terms'] = missing
            except Exception:
                logging.exception("Neural RAG failed; preserving BM25 result")
                rag_res = {"available": False, "error": "RAG gặp lỗi xử lý. Kết quả BM25 vẫn được giữ; thử lại hoặc tắt RAG."}

        res["rag"] = rag_res
        for chunk in res['result']['top_chunks']:
            if chunk.get('chunk_id') in source_urls:
                chunk['source_url'] = source_urls[chunk['chunk_id']]
        if rag_res:
            for chunk in rag_res.get('retrieved_chunks', []):
                if chunk.get('chunk_id') in source_urls:
                    chunk['source_url'] = source_urls[chunk['chunk_id']]
        # Reference labels are attached after inference and never enter the model prompt.
        reference = next((row for row in sample_records
                          if row.get("question", "").strip().casefold() == req.query.casefold()), None)
        res["reference"] = ({"ground_truth": reference.get("answer", ""),
                             "doc_id": reference.get("gold_doc_id", "")} if reference else None)
        res["request_latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return res
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logging.exception("Pipeline execution failed")
        raise HTTPException(status_code=500, detail="Không thể xử lý truy vấn. Vui lòng thử lại.") from exc

def main():
    """Run the local demo without loading the corpus twice."""
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)

if __name__ == "__main__":
    main()
