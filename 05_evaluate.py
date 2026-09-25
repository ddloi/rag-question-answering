"""
BƯỚC 7: Evaluation
---------------------
- Recall@k của retrieval (đo riêng, không phụ thuộc generation)
- EM (Exact Match) và F1 cho câu trả lời cuối cùng
- Phân rã lỗi: lỗi do retrieval sai hay generation sai
"""

import json
import re
import string
from collections import Counter
from pathlib import Path
from rag_pipeline import RAGSystem  # đổi tên file 04_rag_pipeline.py -> rag_pipeline.py khi import

DATA_DIR = Path("data")


# ---------------- Các hàm chuẩn hoá & tính điểm (theo chuẩn SQuAD) ----------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------- Đánh giá Recall@k của retrieval ----------------
def evaluate_retrieval(rag: RAGSystem, qa_records: list, k: int = 5) -> float:
    hits = 0
    for rec in qa_records:
        retrieved = rag.retrieve(rec["question"], top_k=k)
        retrieved_doc_ids = {r["doc_id"] for r in retrieved}
        if rec["gold_doc_id"] in retrieved_doc_ids:
            hits += 1
    return hits / len(qa_records)


# ---------------- Đánh giá EM/F1 cho từng hệ thống + phân rã lỗi ----------------
def evaluate_end_to_end(rag: RAGSystem, qa_records: list):
    results = {"rag_full": [], "retrieval_only": [], "generator_only": []}
    error_breakdown = {"retrieval_fail": 0, "generation_fail": 0, "correct": 0}

    for rec in qa_records:
        question, gold_answer = rec["question"], rec["answer"]

        # Hệ RAG đầy đủ
        rag_result = rag.answer_rag_full(question)
        em = exact_match_score(rag_result["answer"], gold_answer)
        f1 = f1_score(rag_result["answer"], gold_answer)
        results["rag_full"].append({"em": em, "f1": f1})

        # Phân rã lỗi: retrieval có tìm đúng doc không?
        retrieved_doc_ids = {r["doc_id"] for r in rag_result["retrieved"]}
        retrieval_ok = rec["gold_doc_id"] in retrieved_doc_ids

        if em == 1:
            error_breakdown["correct"] += 1
        elif not retrieval_ok:
            error_breakdown["retrieval_fail"] += 1
        else:
            # retrieval đúng nhưng câu trả lời vẫn sai -> lỗi ở generation
            error_breakdown["generation_fail"] += 1

        # Hệ retrieval-only
        ro_result = rag.answer_retrieval_only(question)
        results["retrieval_only"].append({
            "em": exact_match_score(ro_result["answer"], gold_answer),
            "f1": f1_score(ro_result["answer"], gold_answer),
        })

        # Hệ generator-only
        go_answer = rag.answer_generator_only(question)
        results["generator_only"].append({
            "em": exact_match_score(go_answer, gold_answer),
            "f1": f1_score(go_answer, gold_answer),
        })

    return results, error_breakdown


def summarize(results: dict, n: int):
    print("\n=== KẾT QUẢ EM / F1 THEO TỪNG HỆ THỐNG ===")
    for system_name, records in results.items():
        avg_em = sum(r["em"] for r in records) / n
        avg_f1 = sum(r["f1"] for r in records) / n
        print(f"{system_name:20s}  EM={avg_em:.3f}  F1={avg_f1:.3f}")


def main():
    rag = RAGSystem()

    qa_records = []
    with open(DATA_DIR / "test_qa.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            qa_records.append(json.loads(line))

    recall_at_5 = evaluate_retrieval(rag, qa_records, k=5)
    print(f"Recall@5 (retrieval): {recall_at_5:.3f}")

    results, error_breakdown = evaluate_end_to_end(rag, qa_records)
    summarize(results, len(qa_records))

    print("\n=== PHÂN RÃ LỖI (RAG đầy đủ) ===")
    total = sum(error_breakdown.values())
    for k, v in error_breakdown.items():
        print(f"{k:20s}: {v}/{total} ({v/total*100:.1f}%)")


if __name__ == "__main__":
    main()
