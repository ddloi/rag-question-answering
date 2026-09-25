"""
BƯỚC 1.5: Tách QA pairs thành train/test + tạo training_pairs.jsonl
------------------------------------------------------------------------
- test_qa.jsonl: KHÔNG dùng để fine-tune, chỉ dùng ở bước evaluate.
- training_pairs.jsonl: ghép câu hỏi + gold passage (chunk chứa câu trả lời)
  + câu trả lời, dùng để fine-tune generator.

Split cố định theo seed để đảm bảo khả năng tái lập (yêu cầu chung của đề bài).
"""

import json
import random
from pathlib import Path

DATA_DIR = Path("data")
SEED = 42
TEST_RATIO = 0.15

random.seed(SEED)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def find_gold_chunk(gold_doc_id: str, answer: str, chunks_by_doc: dict):
    """
    Tìm chunk chứa câu trả lời trong đúng document gốc.
    Nếu answer xuất hiện trong nhiều chunk, lấy chunk đầu tiên khớp.
    Nếu không tìm thấy chunk nào khớp chính xác, lấy chunk đầu tiên của
    document đó (fallback — vẫn hợp lý vì cùng nguồn).
    """
    candidates = chunks_by_doc.get(gold_doc_id, [])
    answer_lower = answer.lower().strip()
    for chunk in candidates:
        if answer_lower and answer_lower in chunk["text"].lower():
            return chunk
    return candidates[0] if candidates else None


def main():
    qa_records = load_jsonl(DATA_DIR / "qa_pairs.jsonl")
    corpus_records = load_jsonl(DATA_DIR / "corpus_chunks.jsonl")

    chunks_by_doc = {}
    for c in corpus_records:
        chunks_by_doc.setdefault(c["doc_id"], []).append(c)

    random.shuffle(qa_records)
    n_test = int(len(qa_records) * TEST_RATIO)
    test_records = qa_records[:n_test]
    train_records = qa_records[n_test:]

    # Ghi test_qa.jsonl (dùng nguyên cho evaluate — không cần gold chunk)
    with open(DATA_DIR / "test_qa.jsonl", "w", encoding="utf-8") as f:
        for rec in test_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Ghi training_pairs.jsonl (cần ghép context = gold chunk)
    training_pairs = []
    skipped = 0
    for rec in train_records:
        gold_chunk = find_gold_chunk(rec["gold_doc_id"], rec["answer"], chunks_by_doc)
        if gold_chunk is None:
            skipped += 1
            continue
        training_pairs.append({
            "question": rec["question"],
            "context": gold_chunk["text"],
            "answer": rec["answer"],
        })

    with open(DATA_DIR / "training_pairs.jsonl", "w", encoding="utf-8") as f:
        for rec in training_pairs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Train (fine-tune): {len(training_pairs)} mẫu (bỏ qua {skipped} vì không tìm được gold chunk)")
    print(f"Test (evaluate)  : {len(test_records)} mẫu")
    print(f"Seed dùng để split: {SEED}, tỉ lệ test: {TEST_RATIO}")


if __name__ == "__main__":
    main()
