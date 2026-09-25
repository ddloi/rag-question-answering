import os
import json
import uvicorn
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pipeline_engine import DeterministicPipelineEngine

app = FastAPI(
    title="Enterprise Pipeline Engine & Telemetry Studio",
    description="High-performance deterministic information retrieval without AI hallucination."
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

print("[Server] Initializing Deterministic Pipeline Engine...")
engine = DeterministicPipelineEngine()
print("[Server] Engine loaded and ready.")

# Load sample QA records from data/test_qa.jsonl
test_qa_path = Path("data") / "test_qa.jsonl"
sample_records = []
if test_qa_path.exists():
    with open(test_qa_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample_records.append(json.loads(line))

class PipelineRequest(BaseModel):
    query: str
    top_k: int = 5
    k1: float = 1.5
    b: float = 0.75

@app.get("/")
def get_index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/pipeline/info")
def get_info():
    return {
        "status": "online",
        "engine_type": "Deterministic Mathematical Engine (BM25 + Inverted Index)",
        "ai_elements": False,
        "total_chunks": engine.total_chunks,
        "average_doc_length": round(engine.avgdl, 1),
        "vocabulary_size": len(engine.doc_freqs),
        "execution_mode": "Pure Mathematical / Zero Hallucination",
        "latency_profile": "Ultra-Low (< 10ms)"
    }

@app.get("/api/pipeline/samples")
def get_samples():
    # Return 8 diverse samples
    samples = []
    for r in sample_records[:8]:
        samples.append({
            "question": r.get("question", ""),
            "ground_truth": r.get("answer", ""),
            "doc_id": r.get("doc_id", "")
        })
    return {"samples": samples}

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
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
