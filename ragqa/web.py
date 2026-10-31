import json
import logging
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .engine import DeterministicPipelineEngine
from .rag_engine import rag_neural_engine
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

print("[Server] Initializing Deterministic Pipeline Engine...")
engine = DeterministicPipelineEngine()
print("[Server] Engine loaded and ready.")

# Kick off neural RAG model loading in the background if files exist
if rag_neural_engine.is_available():
    print("[Server] Triggering background load for RAG Neural Engine (FAISS + T5)...")
    rag_neural_engine.load_in_background()

# Load sample QA records from data/test_qa.jsonl
test_qa_path = DATA_DIR / "test_qa.jsonl"
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

@app.get("/")
def get_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/pipeline/info")
def get_info():
    rag_avail = rag_neural_engine.is_available()
    return {
        "status": "online",
        "engine_type": "BM25 (Sparse) + FAISS (Dense) + T5-base Generator",
        "ai_elements": rag_neural_engine.is_loaded,
        "total_chunks": engine.total_chunks,
        "average_doc_length": round(engine.avgdl, 1),
        "vocabulary_size": len(engine.doc_freqs),
        "execution_mode": "Dual Comparison (BM25 vs FAISS + T5 RAG)",
        "confidence_type": "lexical_heuristic",
        "corpus_language": "en",
        "generator_enabled": rag_avail,
        "rag_status": "ready" if rag_neural_engine.is_loaded else ("loading" if rag_neural_engine.is_loading else "idle"),
        "rag_device": rag_neural_engine.device_name if rag_neural_engine.is_loaded else ("Sẵn sàng tải" if rag_avail else "Không khả dụng"),
        "rag_available": rag_avail
    }

@app.get("/api/pipeline/samples")
def get_samples():
    samples = []
    for r in sample_records[:12]:
        samples.append({
            "question": r.get("question", ""),
            "ground_truth": r.get("answer", ""),
            "doc_id": r.get("gold_doc_id", "")
        })
    return {"samples": samples}

@app.post("/api/rag/load")
def load_rag_now():
    """Explicitly load RAG neural engine synchronously."""
    if not rag_neural_engine.is_available():
        raise HTTPException(status_code=400, detail="Không tìm thấy index FAISS hoặc metadata trên ổ cứng.")
    if not rag_neural_engine.is_loaded:
        rag_neural_engine.load_sync()
    return {
        "status": "ready" if rag_neural_engine.is_loaded else "error",
        "device": rag_neural_engine.device_name,
        "error": rag_neural_engine.load_error
    }

@app.post("/api/pipeline/execute")
def execute_pipeline(req: PipelineRequest):
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
        if req.run_rag and rag_neural_engine.is_available():
            if not rag_neural_engine.is_loaded and not rag_neural_engine.is_loading:
                rag_neural_engine.load_sync()
            rag_res = rag_neural_engine.execute_rag(query=req.query, top_k=req.top_k)

        res["rag"] = rag_res
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
