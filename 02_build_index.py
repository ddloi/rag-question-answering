"""
BƯỚC 3: Embedding + xây FAISS index
-------------------------------------
Dùng sentence-transformers (all-MiniLM-L6-v2, nhẹ, chạy tốt trên CPU/GPU 6GB)
để mã hoá toàn bộ các chunk trong corpus, sau đó lập chỉ mục FAISS để
tìm kiếm nhanh theo similarity.

Chạy 1 lần duy nhất (offline) — không cần chạy lại mỗi lần hỏi.
"""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_DIR = Path("data")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 22M params, rất nhẹ
INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "chunk_metadata.jsonl"


def load_corpus():
    records = []
    with open(DATA_DIR / "corpus_chunks.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main():
    records = load_corpus()
    texts = [r["text"] for r in records]
    print(f"Đang mã hoá {len(texts)} chunks...")

    # GPU 6GB dư sức chạy model nhỏ này — device tự nhận "cuda" nếu có
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")

    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalize để dùng cosine similarity qua inner product
    )
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    # IndexFlatIP: exact search bằng inner product (= cosine vì đã normalize)
    # Với quy mô đồ án (vài chục nghìn chunk), FlatIP đủ nhanh, không cần
    # index xấp xỉ (IVF/HNSW) — giữ đơn giản, dễ giải thích trong báo cáo.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Đã lập chỉ mục {index.ntotal} vector, dim={dim}")
    print(f"Index lưu tại: {INDEX_PATH}")
    print(f"Metadata lưu tại: {METADATA_PATH}")


if __name__ == "__main__":
    main()
