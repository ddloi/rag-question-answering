"""
CẢI TIẾN 2: Ablation study — chunk_size và top_k
------------------------------------------------------
Đo Recall@k của retrieval khi thay đổi:
  - chunk_size: 100, 200, 300 từ/chunk
  - top_k: 3, 5, 10
  - retriever: dense-only vs hybrid (BM25+dense)

Recall@k là chỉ số RẺ để đo (không cần chạy generation), nên ablation
này chạy nhanh — phù hợp để thử nhiều cấu hình trên máy cá nhân.

Lưu ý: mỗi lần đổi chunk_size, phải rebuild lại chunk + index từ đầu
(vì ranh giới đoạn văn thay đổi hoàn toàn).
"""

import json
import re
import numpy as np
import faiss
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZES = [100, 200, 300]
TOP_KS = [3, 5, 10]
CHUNK_OVERLAP_RATIO = 0.2  # overlap = 20% chunk_size
RRF_K = 60
CANDIDATE_POOL = 30


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks_for_size(documents, chunk_size):
    overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)
    records = []
    chunk_id = 0
    for doc in documents:
        text = clean_text(doc["text"])
        for chunk in chunk_text(text, chunk_size, overlap):
            records.append({"chunk_id": chunk_id, "doc_id": doc["id"], "text": chunk})
            chunk_id += 1
    return records


def build_dense_index(chunks, embed_model):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(
        texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def dense_retrieve(index, embed_model, query, top_k):
    q_emb = embed_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    _, indices = index.search(q_emb, top_k)
    return [idx for idx in indices[0] if idx != -1]


def hybrid_retrieve(bm25, index, embed_model, query, top_k):
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ids = list(np.argsort(bm25_scores)[::-1][:CANDIDATE_POOL])
    dense_ids = dense_retrieve(index, embed_model, query, CANDIDATE_POOL)

    rrf_scores = {}
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [doc_id for doc_id, _ in ranked]


def recall_at_k(retrieved_doc_ids_per_query, gold_doc_ids):
    hits = sum(
        1 for retrieved, gold in zip(retrieved_doc_ids_per_query, gold_doc_ids)
        if gold in retrieved
    )
    return hits / len(gold_doc_ids)


def main():
    with open("raw_documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    qa_records = []
    with open(DATA_DIR / "qa_pairs.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            qa_records.append(json.loads(line))
    # Dùng subset nhỏ cho ablation để chạy nhanh (vd 100 câu)
    qa_records = qa_records[:100]

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")

    results_table = []

    for chunk_size in CHUNK_SIZES:
        print(f"\n=== chunk_size={chunk_size} ===")
        chunks = build_chunks_for_size(documents, chunk_size)
        doc_id_by_chunk_idx = [c["doc_id"] for c in chunks]

        dense_index = build_dense_index(chunks, embed_model)
        tokenized_corpus = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        for top_k in TOP_KS:
            dense_retrieved, hybrid_retrieved = [], []
            gold_ids = []

            for rec in qa_records:
                gold_ids.append(rec["gold_doc_id"])

                d_ids = dense_retrieve(dense_index, embed_model, rec["question"], top_k)
                dense_retrieved.append({doc_id_by_chunk_idx[i] for i in d_ids})

                h_ids = hybrid_retrieve(bm25, dense_index, embed_model, rec["question"], top_k)
                hybrid_retrieved.append({doc_id_by_chunk_idx[i] for i in h_ids})

            dense_recall = recall_at_k(dense_retrieved, gold_ids)
            hybrid_recall = recall_at_k(hybrid_retrieved, gold_ids)

            results_table.append({
                "chunk_size": chunk_size,
                "top_k": top_k,
                "recall_dense": round(dense_recall, 3),
                "recall_hybrid": round(hybrid_recall, 3),
            })
            print(f"top_k={top_k:2d}  Recall dense={dense_recall:.3f}  Recall hybrid={hybrid_recall:.3f}")

    df = pd.DataFrame(results_table)
    df.to_csv("ablation_results.csv", index=False)
    print("\n=== BẢNG TỔNG HỢP (đã lưu ablation_results.csv) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
