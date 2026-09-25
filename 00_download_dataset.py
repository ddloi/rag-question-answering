"""
BƯỚC 0: Tải và chuẩn bị dataset subset
------------------------------------------
Đề bài cho phép: Natural Questions hoặc TriviaQA, dùng corpus subset có
kiểm soát để đảm bảo tái lập trên Colab/máy cá nhân.

Script này:
  1. Tải một subset nhỏ từ Hugging Face Datasets (không cần tải toàn bộ
     hàng chục GB của NQ/TriviaQA gốc).
  2. Convert sang định dạng raw_documents.json mà 01_prepare_and_chunk.py
     cần.
  3. In ra thống kê + license để bạn ghi vào báo cáo.

Chọn 1 trong 2 DATASET_CHOICE bên dưới.
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

# ---------------------------------------------------------------
# CẤU HÌNH — chọn dataset và kích thước subset
# ---------------------------------------------------------------
DATASET_CHOICE = "trivia_qa"   # "trivia_qa" hoặc "natural_questions"
N_SAMPLES = 800                 # số lượng câu hỏi lấy ra (subset nhỏ, đủ cho đồ án)
SEED = 42
OUTPUT_FILE = "raw_documents.json"

random.seed(SEED)


def prepare_trivia_qa():
    print("Đang tải TriviaQA (config: rc.wikipedia, split: validation, streaming)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation", streaming=True)

    documents = {}
    for ex in ds:
        if len(documents) >= N_SAMPLES:
            break
        question = ex["question"]
        answer = ex["answer"]["value"]

        titles = ex["entity_pages"]["title"]
        contexts = ex["entity_pages"]["wiki_context"]
        if not titles or not contexts:
            continue

        doc_id = f"doc_{len(documents)}"
        documents[doc_id] = {
            "id": doc_id,
            "title": titles[0],
            "text": contexts[0][:5000],
            "qa": [{"question": question, "answer": answer}],
        }
        if len(documents) % 100 == 0:
            print(f"Đã lấy {len(documents)}/{N_SAMPLES} documents...")

    return list(documents.values())


def prepare_natural_questions():
    """
    Natural Questions (simplified). License: CC BY-SA 3.0.
    Dùng bản 'sentence' hoặc bản rút gọn để nhẹ hơn bản gốc rất nặng.
    """
    print("Đang tải Natural Questions (split: validation, streaming để nhẹ)...")
    ds = load_dataset(
        "google-research-datasets/natural_questions", "default",
        split="validation", streaming=True,
    )

    documents = []
    count = 0
    for ex in ds:
        if count >= N_SAMPLES:
            break
        # Lấy câu trả lời ngắn đầu tiên nếu có
        short_answers = ex["annotations"]["short_answers"]
        if not short_answers or not short_answers[0]["text"]:
            continue

        question = ex["question"]["text"]
        answer = short_answers[0]["text"][0]
        # document_text đã tokenized dạng list từ, ghép lại
        tokens = ex["document"]["tokens"]["token"]
        text = " ".join(tokens[:1500])  # giới hạn độ dài

        documents.append({
            "id": f"doc_{count}",
            "title": ex["document"]["title"],
            "text": text,
            "qa": [{"question": question, "answer": answer}],
        })
        count += 1

    return documents


def main():
    if DATASET_CHOICE == "trivia_qa":
        documents = prepare_trivia_qa()
        license_note = (
            "TriviaQA: câu hỏi Apache 2.0; evidence từ Wikipedia CC BY-SA 3.0"
        )
    elif DATASET_CHOICE == "natural_questions":
        documents = prepare_natural_questions()
        license_note = "Natural Questions: CC BY-SA 3.0"
    else:
        raise ValueError("DATASET_CHOICE phải là 'trivia_qa' hoặc 'natural_questions'")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    total_qa = sum(len(d["qa"]) for d in documents)
    avg_len = sum(len(d["text"].split()) for d in documents) / max(len(documents), 1)

    print("\n=== THỐNG KÊ DATASET (ghi vào báo cáo) ===")
    print(f"Dataset: {DATASET_CHOICE}")
    print(f"License: {license_note}")
    print(f"Số documents: {len(documents)}")
    print(f"Số cặp QA: {total_qa}")
    print(f"Độ dài trung bình mỗi document: {avg_len:.0f} từ")
    print(f"Seed: {SEED}")
    print(f"Đã ghi ra: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
