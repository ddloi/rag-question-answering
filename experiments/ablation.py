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

import sys
import json
from ragqa.paths import DATA_DIR, RAW_DOCUMENTS_PATH, RESULTS_DIR
from .prepare_chunks import chunk_text, clean_text

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_PATH = RESULTS_DIR / "ablation_results.csv"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZES = [100, 200, 300]
TOP_KS = [3, 5, 10]
CHUNK_OVERLAP_RATIO = 0.2  # overlap = 20% chunk_size
RRF_K = 60
CANDIDATE_POOL = 30


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
    import faiss
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
    import numpy as np
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


def load_test_questions():
    test_file = DATA_DIR / 'test_qa.jsonl'
    if not test_file.is_file():
        raise FileNotFoundError(f'Missing test split: {test_file}. Run python -m experiments split first.')
    with test_file.open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()][:100]
    if not records:
        raise ValueError(f'Test split is empty: {test_file}')
    return records


def main():
    qa_records = load_test_questions()
    import pandas as pd
    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    with open(RAW_DOCUMENTS_PATH, "r", encoding="utf-8") as f:
        documents = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)

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
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n=== BẢNG TỔNG HỢP (đã lưu {OUTPUT_PATH}) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
