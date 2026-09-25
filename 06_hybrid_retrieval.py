"""
CẢI TIẾN 1: Hybrid Sparse (BM25) + Dense Retrieval
------------------------------------------------------
Ý tưởng: BM25 giỏi bắt các từ khoá/tên riêng chính xác (exact match),
dense retrieval giỏi bắt ngữ nghĩa (semantic). Kết hợp 2 loại giúp bù
điểm yếu cho nhau.

Phương pháp kết hợp: Reciprocal Rank Fusion (RRF) — đơn giản, không cần
chuẩn hoá điểm số giữa 2 hệ thống (BM25 score và cosine score có thang
đo khác nhau, khó cộng trực tiếp). RRF chỉ dùng THỨ HẠNG, nên rất ổn định.

    RRF_score(doc) = sum over each ranker: 1 / (k + rank_in_that_ranker)

k thường chọn = 60 (giá trị chuẩn trong literature, không nhạy tham số).
"""

import json
import numpy as np
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
CANDIDATE_POOL = 30  # lấy top-30 từ mỗi hệ trước khi fuse, rồi cắt còn top_k cuối


def simple_tokenize(text: str):
    return text.lower().split()


class HybridRetriever:
    def __init__(self):
        self.metadata = []
        with open(DATA_DIR / "chunk_metadata.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        # --- BM25 (sparse) ---
        print("Đang xây BM25 index...")
        tokenized_corpus = [simple_tokenize(c["text"]) for c in self.metadata]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # --- Dense (đã có sẵn từ 02_build_index.py) ---
        print("Đang load FAISS index + embedding model...")
        self.index = faiss.read_index(str(DATA_DIR / "faiss.index"))
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")

    def _bm25_ranked_ids(self, query: str, pool_size: int):
        scores = self.bm25.get_scores(simple_tokenize(query))
        top_ids = np.argsort(scores)[::-1][:pool_size]
        return list(top_ids)

    def _dense_ranked_ids(self, query: str, pool_size: int):
        q_emb = self.embed_model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        _, indices = self.index.search(q_emb, pool_size)
        return [idx for idx in indices[0] if idx != -1]

    def retrieve(self, query: str, top_k: int = 5):
        bm25_ids = self._bm25_ranked_ids(query, CANDIDATE_POOL)
        dense_ids = self._dense_ranked_ids(query, CANDIDATE_POOL)

        rrf_scores = {}
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, doc_id in enumerate(dense_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (RRF_K + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for chunk_idx, score in ranked:
            chunk = self.metadata[chunk_idx]
            results.append({**chunk, "score": float(score)})
        return results


def compare_retrievers_demo():
    """So sánh nhanh dense-only vs hybrid trên vài câu hỏi mẫu."""
    hybrid = HybridRetriever()

    test_questions = [
        "Ví dụ câu hỏi 1",
        "Ví dụ câu hỏi 2",
    ]

    for q in test_questions:
        print(f"\nCâu hỏi: {q}")
        results = hybrid.retrieve(q, top_k=5)
        for r in results:
            print(f"  chunk_id={r['chunk_id']}  doc_id={r['doc_id']}  rrf_score={r['score']:.4f}")


if __name__ == "__main__":
    compare_retrievers_demo()
