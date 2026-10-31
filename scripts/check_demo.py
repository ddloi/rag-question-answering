"""Read-only preflight. Run from RAG: python -m scripts.check_demo [--neural]."""
import argparse
import json
import sys
from pathlib import Path

from ragqa.paths import DATA_DIR, PROJECT_ROOT


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", action="store_true", help="Load cached models and run one real RAG query; no downloads")
    args = parser.parse_args()
    try:
        from ragqa.rag_engine import RAGNeuralEngine
        from fastapi.testclient import TestClient
        from ragqa.web import app, DEMO_DATA_DIR

        chunks = read_jsonl(DEMO_DATA_DIR / "chunk_metadata.jsonl")
        samples = read_jsonl(DEMO_DATA_DIR / "test_qa.jsonl")
        if not chunks or not samples:
            raise ValueError("Empty corpus or test questions")
        ids = [row["chunk_id"] for row in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk_id in metadata")
        documents = {row["doc_id"] for row in chunks}
        if any(row["gold_doc_id"] not in documents for row in samples):
            raise ValueError("Test question references a missing source document")
        training = DATA_DIR / "training_pairs.jsonl"
        if training.exists():
            train_questions = {row["question"].strip().casefold() for row in read_jsonl(training)}
            if train_questions & {row["question"].strip().casefold() for row in samples}:
                raise ValueError("Question overlap between training and test files")

        with TestClient(app) as client:
            for path in ("/", "/static/app.js", "/static/style.css", "/api/pipeline/info", "/api/pipeline/samples"):
                if client.get(path).status_code != 200:
                    raise RuntimeError(f"Endpoint failed: {path}")
            query = "Proteus and Nereid are moons of which planet?"
            response = client.post("/api/pipeline/execute", json={"query": query, "run_rag": False})
            response.raise_for_status()
            if not response.json()["result"]["top_chunks"]:
                raise RuntimeError("BM25 smoke query returned no source")
        print(f"PASS web/BM25: {len(chunks)} chunks, {len(samples)} test questions; root={PROJECT_ROOT}")
        neural = RAGNeuralEngine(data_dir=DEMO_DATA_DIR)
        print(f"Neural artifacts present: {neural.is_available()} (cached models/dependencies checked only with --neural)")
        if args.neural:
            neural.load_sync()
            if not neural.is_loaded:
                raise RuntimeError(neural.load_error)
            result = neural.execute_rag(query)
            if not result["available"] or not result.get("answer"):
                raise RuntimeError("RAG smoke query produced no answer")
            print(json.dumps({"device": neural.device_name, "answer": result["answer"],
                              "extractive_answer": result["extractive_answer"], "latency": result["latency"]}, ensure_ascii=False))
            print("PASS neural execution (not a dataset quality evaluation)")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
