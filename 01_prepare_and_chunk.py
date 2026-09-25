"""
BƯỚC 1-2: Chuẩn bị dữ liệu + Chunking
--------------------------------------
Input : bộ QA gốc (vd: Natural Questions subset, hoặc file JSON tự chuẩn bị)
Output: corpus_chunks.jsonl (mỗi dòng là 1 đoạn văn bản đã chia nhỏ)
        qa_pairs.jsonl       (câu hỏi + câu trả lời + id passage nguồn)

Định dạng dữ liệu đầu vào giả định (tự chỉnh nếu dữ liệu của bạn khác):
[
  {
    "id": "doc_001",
    "title": "...",
    "text": "toàn bộ nội dung bài viết dài...",
    "qa": [
       {"question": "...", "answer": "..."},
       ...
    ]
  },
  ...
]
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------
# CẤU HÌNH
# ---------------------------------------------------------------
INPUT_FILE = "raw_documents.json"     # đổi thành file dữ liệu thật của bạn
OUTPUT_DIR = Path("data")
CHUNK_SIZE = 200        # số từ mỗi chunk (không phải token, xấp xỉ)
CHUNK_OVERLAP = 40      # số từ overlap giữa 2 chunk liền kề

OUTPUT_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    """Loại bỏ khoảng trắng thừa, ký tự lạ."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Chia văn bản thành các đoạn theo số từ, có overlap.
    Đơn giản, dễ hiểu — đủ dùng cho quy mô đồ án.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # lùi lại để tạo overlap
    return chunks


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    corpus_records = []
    qa_records = []
    chunk_id_counter = 0

    for doc in documents:
        doc_id = doc["id"]
        text = clean_text(doc["text"])
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        # Lưu từng chunk kèm metadata để sau này truy vết lại nguồn
        first_chunk_id_of_doc = chunk_id_counter
        for chunk in chunks:
            corpus_records.append({
                "chunk_id": chunk_id_counter,
                "doc_id": doc_id,
                "title": doc.get("title", ""),
                "text": chunk,
            })
            chunk_id_counter += 1

        # Lưu QA pairs, giữ liên kết tới doc_id (không tới chunk cụ thể,
        # vì retrieval sẽ tự tìm chunk phù hợp — đây là điểm quan trọng
        # để đánh giá Recall@k sau này)
        for qa in doc.get("qa", []):
            qa_records.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "gold_doc_id": doc_id,
            })

    # Ghi ra file .jsonl (mỗi dòng 1 JSON object — chuẩn cho dữ liệu lớn)
    with open(OUTPUT_DIR / "corpus_chunks.jsonl", "w", encoding="utf-8") as f:
        for rec in corpus_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "qa_pairs.jsonl", "w", encoding="utf-8") as f:
        for rec in qa_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Đã tạo {len(corpus_records)} chunks từ {len(documents)} documents")
    print(f"Đã tạo {len(qa_records)} cặp câu hỏi-đáp")
    print(f"Ghi ra: {OUTPUT_DIR}/corpus_chunks.jsonl và {OUTPUT_DIR}/qa_pairs.jsonl")


if __name__ == "__main__":
    main()
