"""
BƯỚC 7: Evaluation
---------------------
- Recall@k của retrieval (đo riêng, không phụ thuộc generation)
- EM (Exact Match) và F1 cho câu trả lời cuối cùng
- Phân rã lỗi: lỗi do retrieval sai hay generation sai
"""

import sys
import json
import re
import string
import importlib
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Filename starts with a digit so we use importlib instead of a plain import
_rag_module = importlib.import_module("04_rag_pipeline")
RAGSystem = _rag_module.RAGSystem

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


# ---------------- Đánh giá Faithfulness & Citation Correctness ----------------
def faithfulness_score(prediction: str, context: str) -> float:
    """
    Đo độ trung thực (Faithfulness / Context Grounding):
    Tỷ lệ token trong câu trả lời xuất hiện trong đoạn context bằng chứng.
    Tránh hiện tượng ảo giác (hallucination) của generator.
    """
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    ctx_norm = normalize_answer(context)
    ctx_token_set = set(ctx_norm.split())
    supported_tokens = sum(1 for tok in pred_tokens if tok in ctx_token_set)
    return supported_tokens / len(pred_tokens)


def citation_correctness_score(gold_answer: str, gold_doc_id: str, retrieved_passages: list) -> dict:
    """
    Đo độ chính xác của trích dẫn (Citation Correctness):
    - doc_match: Liệu tài liệu nguồn chuẩn (gold_doc_id) có nằm trong danh sách trích dẫn không.
    - evidence_match: Liệu có passage nào chứa trực tiếp chuỗi đáp án chuẩn (gold_answer) không.
    """
    doc_match = int(any(p.get("doc_id") == gold_doc_id for p in retrieved_passages))
    norm_gold = normalize_answer(gold_answer)
    evidence_match = 0
    if norm_gold:
        for p in retrieved_passages:
            if norm_gold in normalize_answer(p.get("text", "")):
                evidence_match = 1
                break
    return {"doc_match": doc_match, "evidence_match": evidence_match}


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
    faithfulness_scores = {"rag_full": [], "retrieval_only": []}
    citation_doc_scores = []
    citation_evidence_scores = []

    for rec in qa_records:
        question, gold_answer, gold_doc = rec["question"], rec["answer"], rec["gold_doc_id"]

        # Hệ RAG đầy đủ
        rag_result = rag.answer_rag_full(question)
        rag_pred = rag_result["answer"]
        em = exact_match_score(rag_pred, gold_answer)
        f1 = f1_score(rag_pred, gold_answer)
        results["rag_full"].append({"em": em, "f1": f1})

        # Context được dùng để sinh (top-2 passages)
        used_context = " ".join([r["text"] for r in rag_result["retrieved"][:2]])
        faithfulness_scores["rag_full"].append(faithfulness_score(rag_pred, used_context))

        # Citation correctness: kiểm tra trích dẫn cả cấp doc và cấp evidence
        cit_eval = citation_correctness_score(gold_answer, gold_doc, rag_result["retrieved"])
        citation_doc_scores.append(cit_eval["doc_match"])
        citation_evidence_scores.append(cit_eval["evidence_match"])

        # Phân rã lỗi: retrieval có tìm đúng doc không?
        retrieved_doc_ids = {r["doc_id"] for r in rag_result["retrieved"]}
        retrieval_ok = gold_doc in retrieved_doc_ids

        if em == 1:
            error_breakdown["correct"] += 1
        elif not retrieval_ok:
            error_breakdown["retrieval_fail"] += 1
        else:
            # retrieval đúng nhưng câu trả lời vẫn sai -> lỗi ở generation
            error_breakdown["generation_fail"] += 1

        # Hệ retrieval-only
        ro_result = rag.answer_retrieval_only(question)
        ro_pred = ro_result["answer"]
        results["retrieval_only"].append({
            "em": exact_match_score(ro_pred, gold_answer),
            "f1": f1_score(ro_pred, gold_answer),
        })
        ro_context = " ".join([r["text"] for r in ro_result["retrieved"][:3]])
        faithfulness_scores["retrieval_only"].append(faithfulness_score(ro_pred, ro_context))

        # Hệ generator-only
        go_answer = rag.answer_generator_only(question)
        results["generator_only"].append({
            "em": exact_match_score(go_answer, gold_answer),
            "f1": f1_score(go_answer, gold_answer),
        })

    metrics_extra = {
        "faithfulness": {
            k: sum(v) / len(v) if v else 0.0 for k, v in faithfulness_scores.items()
        },
        "citation_doc_accuracy": (
            sum(citation_doc_scores) / len(citation_doc_scores)
            if citation_doc_scores else 0.0
        ),
        "citation_evidence_accuracy": (
            sum(citation_evidence_scores) / len(citation_evidence_scores)
            if citation_evidence_scores else 0.0
        ),
    }

    return results, error_breakdown, metrics_extra


def summarize(results: dict, error_breakdown: dict, metrics_extra: dict, n: int):
    print("\n=== KẾT QUẢ EM / F1 THEO TỪNG HỆ THỐNG ===")
    for system_name, records in results.items():
        avg_em = sum(r["em"] for r in records) / n
        avg_f1 = sum(r["f1"] for r in records) / n
        print(f"{system_name:20s}  EM={avg_em:.3f}  F1={avg_f1:.3f}")

    print("\n=== ĐỘ TRUNG THỰC & TRÍCH DẪN (FAITHFULNESS & CITATION) ===")
    for sys_name, score in metrics_extra["faithfulness"].items():
        print(f"Faithfulness ({sys_name:15s}): {score:.3f}")
    print(f"Citation Accuracy (Document-level):  {metrics_extra['citation_doc_accuracy']:.3f}")
    print(f"Citation Precision (Evidence-level): {metrics_extra['citation_evidence_accuracy']:.3f}")

    print("\n=== PHÂN RÃ LỖI (RAG đầy đủ) ===")
    total = sum(error_breakdown.values())
    for k, v in error_breakdown.items():
        print(f"{k:20s}: {v}/{total} ({v/total*100:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline (Đề tài 20)")
    parser.add_argument("--samples", type=int, default=None, help="Số lượng mẫu test để đánh giá nhanh (mặc định: toàn bộ)")
    parser.add_argument("--k", type=int, default=5, help="Top-k cho Recall@k (mặc định: 5)")
    args = parser.parse_args()

    rag = RAGSystem()

    qa_records = []
    with open(DATA_DIR / "test_qa.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            qa_records.append(json.loads(line))

    if args.samples and args.samples < len(qa_records):
        print(f"Đang chạy đánh giá trên tập mẫu {args.samples}/{len(qa_records)} câu...")
        qa_records = qa_records[:args.samples]
    else:
        print(f"Đang chạy đánh giá trên toàn bộ {len(qa_records)} câu test...")

    recall_at_k = evaluate_retrieval(rag, qa_records, k=args.k)
    print(f"Recall@{args.k} (retrieval): {recall_at_k:.3f}")

    results, error_breakdown, metrics_extra = evaluate_end_to_end(rag, qa_records)
    summarize(results, error_breakdown, metrics_extra, len(qa_records))


if __name__ == "__main__":
    main()
