"""
File app.py dành cho Hugging Face Spaces (Chạy Web Demo RAG 24/7 trên Cloud).
Tự động load mô hình T5 và FAISS index từ Hugging Face Model Hub.
"""

import json
import faiss
import torch
import gradio as gr
from pathlib import Path
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering

# --- CẤU HÌNH REPO CỦA BẠN TRÊN HUGGING FACE ---
# Thay thế 'username/rag-t5-nlp-project' bằng repo_id thật của bạn
MODEL_REPO_ID = "YOUR_USERNAME/rag-t5-nlp-project"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

print("Đang khởi tạo các thành phần RAG từ Hugging Face Hub...")

# Tải FAISS index và chunk metadata từ Model Hub về cache của Space
faiss_file = hf_hub_download(repo_id=MODEL_REPO_ID, filename="faiss.index")
meta_file = hf_hub_download(repo_id=MODEL_REPO_ID, filename="chunk_metadata.jsonl")

# Load Index & Metadata
index = faiss.read_index(faiss_file)
metadata = []
with open(meta_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            metadata.append(json.loads(line))

# Load Embedding Model
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

# Load T5 Generator đã fine-tune trực tiếp từ Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
generator = AutoModelForSeq2SeqLM.from_pretrained(MODEL_REPO_ID)

# Load DistilBERT cho hệ Extractive baseline
qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

print(" Khởi tạo hoàn tất!")


def retrieve_chunks(query: str, k: int = TOP_K):
    q_emb = embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(q_emb, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx < len(metadata):
            chunk = metadata[idx]
            results.append({**chunk, "score": float(score)})
    return results


def answer_generator_only(question: str) -> str:
    """Hệ 1: Closed-book generator (không context)."""
    input_text = f"answer the question without context. question: {question} answer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
    outputs = generator.generate(**inputs, max_new_tokens=32, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def answer_retrieval_only(question: str, retrieved: list) -> str:
    """Hệ 2: Retrieval-only / extractive baseline (trích xuất span từ passage)."""
    best_answer = ""
    best_score = -1e9
    for candidate in retrieved[:3]:
        ctx = candidate["text"]
        inputs = qa_tokenizer(question, ctx, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        n_tokens = start_logits.shape[0]
        for s in range(n_tokens):
            for e in range(s, min(s + 20, n_tokens)):
                sc = start_logits[s].item() + end_logits[e].item()
                if sc > best_score:
                    best_score = sc
                    ans = qa_tokenizer.decode(inputs["input_ids"][0][s : e + 1], skip_special_tokens=True).strip()
                    if ans and len(ans) > 1:
                        best_answer = ans
    return best_answer if best_answer else "Không tìm thấy câu trả lời trích xuất phù hợp."


def answer_rag_full(question: str, retrieved: list) -> str:
    """Hệ 3: RAG hoàn chỉnh (Dense retrieval + T5 Generator)."""
    context = " ".join([r["text"] for r in retrieved[:2]])
    input_text = f"answer the question from the context. question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = generator.generate(**inputs, max_new_tokens=32, num_beams=6, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


def run_pipeline(question: str):
    if not question.strip():
        return "Vui lòng nhập câu hỏi!", "", "", ""

    # 1. Retrieval
    retrieved = retrieve_chunks(question, k=TOP_K)

    # 2. Chạy cả 3 hệ thống để so sánh
    ans_gen_only = answer_generator_only(question)
    ans_ret_only = answer_retrieval_only(question, retrieved)
    ans_rag = answer_rag_full(question, retrieved)

    # 3. Format context bằng chứng
    passages_md = ""
    for idx, r in enumerate(retrieved[:3]):
        passages_md += f"**[Passage {idx+1}] (Doc: `{r.get('doc_id')}`, Cosine Score: `{r.get('score'):.4f}`)**\n"
        passages_md += f"> {r.get('text', '')[:400]}...\n\n"

    return ans_rag, ans_ret_only, ans_gen_only, passages_md


# --- GIAO DIỆN GRADIO HIỆN ĐẠI ---
with gr.Blocks(title="Đề tài 20: Hệ thống RAG Hỏi-Đáp") as demo:
    gr.Markdown("# 🤖 Đề tài 20: Hệ thống Retrieval-Augmented Generation (RAG) cho Hỏi-Đáp")
    gr.Markdown(
        "Mô hình Seq2Seq T5 fine-tuned kết hợp FAISS Dense Retrieval theo bài báo *Lewis et al., NeurIPS 2020*. "
        "So sánh trực quan giữa 3 hệ thống theo đúng yêu cầu đề tài."
    )

    with gr.Row():
        with gr.Column(scale=4):
            query_input = gr.Textbox(
                label="Nhập câu hỏi (Tiếng Anh - TriviaQA/NQ):",
                placeholder="Ví dụ: Who directed the movie Jaws?",
                lines=2,
            )
            submit_btn = gr.Button("🔍 Chạy RAG Pipeline", variant="primary")
        with gr.Column(scale=1):
            gr.Examples(
                examples=[
                    ["Who directed the movie Jaws?"],
                    ["What is the capital city of Australia?"],
                    ["Which planet is closest to the Sun?"],
                ],
                inputs=query_input,
            )

    gr.Markdown("### 📊 Kết quả so sánh 3 hệ thống:")
    with gr.Row():
        out_rag = gr.Textbox(label="⭐ Hệ 3: RAG Đầy Đủ (Retrieval + Generator)", lines=3)
        out_ret = gr.Textbox(label="📌 Hệ 2: Retrieval-Only (Extractive Baseline)", lines=3)
        out_gen = gr.Textbox(label="📖 Hệ 1: Generator-Only (Không Retrieval)", lines=3)

    gr.Markdown("### 📑 Bằng chứng truy hồi từ FAISS Index (Top Passages):")
    out_evidence = gr.Markdown()

    submit_btn.click(
        fn=run_pipeline,
        inputs=query_input,
        outputs=[out_rag, out_ret, out_gen, out_evidence],
    )

if __name__ == "__main__":
    demo.launch()
