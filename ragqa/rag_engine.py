import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import DATA_DIR, MODEL_DIR

logger = logging.getLogger(__name__)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EXTRACTIVE_MODEL_NAME = "distilbert-base-cased-distilled-squad"
SIMILARITY_THRESHOLD = 0.25


class RAGNeuralEngine:
    """
    Neural RAG Engine using FAISS dense vector search and fine-tuned T5-base generator.
    Provides methods for:
      - Dense semantic retrieval via FAISS (all-MiniLM-L6-v2)
      - Grounded answer generation via fine-tuned T5-base
      - Parametric-only answer generation (generator-only, showing hallucinations)
      - Extractive QA answer extraction via DistilBERT
      - Relevance gating (threshold filtering to prevent false-confidence hallucination)
    """

    def __init__(self, data_dir: Path = DATA_DIR, model_dir: Path = MODEL_DIR):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.device = "cpu"
        self.device_name = "CPU"
        self.is_loaded = False
        self.is_loading = False
        self.load_error: Optional[str] = None
        self._lock = threading.Lock()

        # Neural components
        self.embed_model = None
        self.index = None
        self.metadata: List[Dict[str, Any]] = []
        self.tokenizer = None
        self.generator = None
        self.qa_tokenizer = None
        self.qa_model = None

    def is_available(self) -> bool:
        """Check if required files exist on disk."""
        faiss_file = self.data_dir / "faiss.index"
        meta_file = self.data_dir / "chunk_metadata.jsonl"
        return faiss_file.exists() and meta_file.exists()

    def load_sync(self):
        """Synchronously load embedding model, FAISS index, and T5 generator."""
        if self.is_loaded:
            return
        with self._lock:
            if self.is_loaded:
                return
            self.is_loading = True
            t0 = time.perf_counter()
            try:
                import torch
                import faiss
                from sentence_transformers import SentenceTransformer
                from transformers import (
                    AutoTokenizer,
                    AutoModelForSeq2SeqLM,
                    AutoModelForQuestionAnswering,
                )

                if torch.cuda.is_available():
                    self.device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    self.device_name = f"CUDA ({gpu_name})"
                else:
                    self.device = "cpu"
                    self.device_name = "CPU"

                logger.info(f"[RAGNeuralEngine] Loading on {self.device_name}...")

                # 1. Embedding Model
                self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=self.device)

                # 2. FAISS Index & Metadata
                faiss_path = str(self.data_dir / "faiss.index")
                self.index = faiss.read_index(faiss_path)

                self.metadata = []
                with open(self.data_dir / "chunk_metadata.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.metadata.append(json.loads(line))

                # 3. T5 Generator (fine-tuned if available, else t5-base)
                gen_path = str(self.model_dir) if self.model_dir.exists() else "t5-base"
                self.tokenizer = AutoTokenizer.from_pretrained(gen_path)
                self.generator = AutoModelForSeq2SeqLM.from_pretrained(gen_path).to(self.device)
                self.generator.eval()

                # 4. DistilBERT Extractive QA Model
                try:
                    self.qa_tokenizer = AutoTokenizer.from_pretrained(EXTRACTIVE_MODEL_NAME)
                    self.qa_model = AutoModelForQuestionAnswering.from_pretrained(EXTRACTIVE_MODEL_NAME).to(self.device)
                    self.qa_model.eval()
                except Exception as ext_err:
                    logger.warning(f"[RAGNeuralEngine] Could not load extractive model: {ext_err}")
                    self.qa_tokenizer = None
                    self.qa_model = None

                self.is_loaded = True
                self.load_error = None
                elapsed = time.perf_counter() - t0
                logger.info(f"[RAGNeuralEngine] Loaded successfully in {elapsed:.2f}s on {self.device_name}.")
            except Exception as exc:
                self.is_loaded = False
                self.load_error = str(exc)
                logger.exception(f"[RAGNeuralEngine] Failed to load: {exc}")
            finally:
                self.is_loading = False

    def load_in_background(self):
        """Kick off background loading thread."""
        if not self.is_loaded and not self.is_loading and self.is_available():
            thread = threading.Thread(target=self.load_sync, daemon=True, name="RAG-Model-Loader")
            thread.start()

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Dense retrieval using all-MiniLM-L6-v2 and FAISS IndexFlatIP."""
        if not self.is_loaded:
            raise RuntimeError("RAG model is not loaded yet.")

        q_emb = self.embed_model.encode(
            [question], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            results.append({
                "rank": rank,
                "chunk_id": chunk.get("chunk_id", idx),
                "doc_id": chunk.get("doc_id", "unknown"),
                "title": chunk.get("title", ""),
                "score": round(float(score), 4),
                "text": chunk.get("text", "")
            })
        return results

    def generate_answer(self, question: str, context: str) -> str:
        """Generate grounded answer with fine-tuned T5-base."""
        if not self.is_loaded:
            raise RuntimeError("RAG model is not loaded yet.")

        input_text = f"answer the question from the context. question: {question} context: {context}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        import torch
        with torch.no_grad():
            output_ids = self.generator.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def answer_generator_only(self, question: str) -> str:
        """Generate answer without context (pure parametric knowledge, showing hallucination)."""
        if not self.is_loaded:
            return ""

        input_text = f"answer the question without context. question: {question} answer:"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=256
        ).to(self.device)

        import torch
        with torch.no_grad():
            output_ids = self.generator.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=0.8,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def answer_extractive_only(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        """Extract answer span using DistilBERT on retrieved passages."""
        if not self.is_loaded or self.qa_model is None or not retrieved:
            return ""

        import torch
        best_answer = ""
        best_score = -1e9

        for candidate in retrieved[:3]:
            ctx = candidate.get("text", "")
            if not ctx:
                continue
            inputs = self.qa_tokenizer(
                question, ctx,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.qa_model(**inputs)

            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
            n_tokens = start_logits.shape[0]
            score_matrix = start_logits.unsqueeze(1) + end_logits.unsqueeze(0)
            mask = torch.triu(torch.ones(n_tokens, n_tokens, device=start_logits.device, dtype=torch.bool), diagonal=0)
            band_mask = torch.triu(torch.ones(n_tokens, n_tokens, device=start_logits.device, dtype=torch.bool), diagonal=20)
            valid_mask = mask & ~band_mask
            score_matrix[~valid_mask] = -1e9
            flat_idx = score_matrix.argmax()
            best_start = (flat_idx // n_tokens).item()
            best_end = (flat_idx % n_tokens).item()
            best_span_score = score_matrix[best_start, best_end].item()

            if best_span_score > 2.0 and best_span_score > best_score:
                ans = self.qa_tokenizer.decode(
                    inputs["input_ids"][0][best_start: best_end + 1],
                    skip_special_tokens=True,
                ).strip()
                if ans and len(ans) > 1:
                    best_answer = ans
                    best_score = best_span_score

        return best_answer

    def execute_rag(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute dense retrieval and generation, returning full comparison data."""
        if not self.is_loaded:
            return {
                "available": False,
                "is_loading": self.is_loading,
                "error": self.load_error or "Mô hình RAG đang khởi động hoặc chưa sẵn sàng."
            }

        t_start = time.perf_counter()

        # Step 1: FAISS Dense Retrieval
        t_ret_0 = time.perf_counter()
        retrieved = self.retrieve(query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_ret_0) * 1000

        top_score = retrieved[0]["score"] if retrieved else 0.0
        is_grounded = top_score >= SIMILARITY_THRESHOLD

        # Step 2: T5 Generation with context (top-2 passages)
        t_gen_0 = time.perf_counter()
        if is_grounded:
            context = " ".join([r["text"] for r in retrieved[:2]]) if retrieved else ""
            rag_answer = self.generate_answer(query, context) if context else "Không tìm thấy ngữ cảnh phù hợp."
        else:
            context = ""
            rag_answer = f"Không tìm thấy tài liệu liên quan trong kho tri thức (Độ tương đồng Cosine: {top_score:.3f} < {SIMILARITY_THRESHOLD})."
        generation_ms = (time.perf_counter() - t_gen_0) * 1000

        # Step 3: Generator-only (No context) for hallucination comparison
        gen_only_answer = self.answer_generator_only(query) if is_grounded else ""

        # Step 4: Extractive baseline
        extractive_ans = self.answer_extractive_only(query, retrieved) if is_grounded else ""

        total_ms = (time.perf_counter() - t_start) * 1000

        return {
            "available": True,
            "device": self.device,
            "device_name": self.device_name,
            "is_grounded": is_grounded,
            "top_score": top_score,
            "answer": rag_answer,
            "generator_only_answer": gen_only_answer,
            "extractive_answer": extractive_ans,
            "context_used": context,
            "retrieved_chunks": retrieved,
            "latency": {
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "total_ms": round(total_ms, 2)
            }
        }


# Global singleton instance
rag_neural_engine = RAGNeuralEngine()
