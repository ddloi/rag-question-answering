import torch
"""
Bước 5-6: Pipeline RAG đầy đủ - Retrieval + Generation
---------------------------------------------------------
Cài đặt 3 hệ thống theo đúng yêu cầu đề bài để so sánh:
  1. generator_only   : sinh câu trả lời KHÔNG có context (parametric knowledge)
  2. retrieval_only    : trích xuất câu trả lời trực tiếp từ passage (extractive)
  3. rag_full           : retrieval + generation (hệ chính)
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

DATA_DIR = Path("data")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_PATH = "generator_finetuned"     # model đã fine-tune ở bước 4
TOP_K = 5


class RAGSystem:
    def __init__(self):
        print("Đang load embedding model, FAISS index, generator...")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
        self.index = faiss.read_index(str(DATA_DIR / "faiss.index"))

        self.metadata = []
        with open(DATA_DIR / "chunk_metadata.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.metadata.append(json.loads(line))

        self.tokenizer = AutoTokenizer.from_pretrained(GENERATOR_PATH)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_PATH).to("cuda")

        # Dùng cho hệ generator_only (không context) — dùng model gốc chưa fine-tune
        self.qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            "distilbert-base-cased-distilled-squad"
        ).to("cuda")

    # ---------------- RETRIEVAL ----------------
    def retrieve(self, question: str, top_k: int = TOP_K):
        q_emb = self.embed_model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.metadata[idx]
            results.append({**chunk, "score": float(score)})
        return results

    # ---------------- GENERATION (RAG đầy đủ) ----------------
    def generate_answer(self, question: str, context: str) -> str:
        # FIX 1: Prompt rõ ràng hơn — yêu cầu model chỉ lấy answer từ context
        input_text = (
            f"answer the question from the context. "
            f"question: {question} "
            f"context: {context}"
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")
        output_ids = self.generator.generate(
            **inputs,
            max_new_tokens=32,          # giới hạn token sinh ra, tránh dài lê thê
            num_beams=8,                # tăng beam để tìm output tốt hơn
            early_stopping=True,
            no_repeat_ngram_size=3,     # tránh lặp cụm 3-gram
            length_penalty=0.8,         # thưởng câu ngắn gọn, tránh padding
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # ---------------- 3 HỆ THỐNG SO SÁNH ----------------
    def answer_generator_only(self, question: str) -> str:
        """Hệ 1: sinh câu trả lời không có context — parametric knowledge thuần."""
        # FIX 4: prompt rõ hơn, không có context
        input_text = f"answer the question without context. question: {question} answer:"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=256
        ).to("cuda")
        output_ids = self.generator.generate(
            **inputs,
            max_new_tokens=32,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def answer_retrieval_only(self, question: str) -> dict:
        """Hệ 2: retrieval + extractive QA (không sinh mới, chỉ trích xuất span)."""
        retrieved = self.retrieve(question, top_k=TOP_K)
        # FIX 2: thử lần lượt top-3 passage thay vì chỉ top-1
        best_answer = ""
        best_score = -1e9

        for candidate in retrieved[:3]:
            ctx = candidate["text"]
            if not ctx:
                continue
            inputs = self.qa_tokenizer(
                question, ctx,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to("cuda")
            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

            # Tìm cặp (start, end) có tổng logit cao nhất, với điều kiện end >= start
            n_tokens = start_logits.shape[0]
            best_span_score = -1e9
            best_start, best_end = 0, 0
            for s in range(n_tokens):
                for e in range(s, min(s + 20, n_tokens)):  # span tối đa 20 token
                    score = start_logits[s].item() + end_logits[e].item()
                    if score > best_span_score:
                        best_span_score = score
                        best_start, best_end = s, e

            # FIX 2b: chỉ nhận span nếu score đủ cao (threshold)
            if best_span_score > 2.0 and best_span_score > best_score:
                answer = self.qa_tokenizer.decode(
                    inputs["input_ids"][0][best_start: best_end + 1],
                    skip_special_tokens=True,
                ).strip()
                # Loại bỏ span toàn dấu câu hoặc rỗng
                if answer and len(answer) > 1:
                    best_answer = answer
                    best_score = best_span_score

        return {"answer": best_answer, "retrieved": retrieved}

    def answer_rag_full(self, question: str) -> dict:
        """Hệ 3: RAG đầy đủ — retrieval rồi generation dựa trên context truy hồi."""
        retrieved = self.retrieve(question, top_k=TOP_K)
        # FIX 3: dùng top-2 passage (đủ context, không quá dài gây nhiễu)
        context = " ".join([r["text"] for r in retrieved[:2]])
        answer = self.generate_answer(question, context)
        return {"answer": answer, "retrieved": retrieved}


def demo():
    rag = RAGSystem()
    question = "Ví dụ câu hỏi cần trả lời ở đây"

    print("\n--- Hệ 1: Generator only (không context) ---")
    print(rag.answer_generator_only(question))

    print("\n--- Hệ 2: Retrieval only (extractive) ---")
    result2 = rag.answer_retrieval_only(question)
    print(result2["answer"])

    print("\n--- Hệ 3: RAG đầy đủ ---")
    result3 = rag.answer_rag_full(question)
    print(result3["answer"])
    print("Passages đã truy hồi:", [r["chunk_id"] for r in result3["retrieved"]])


if __name__ == "__main__":
    demo()
