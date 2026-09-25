import os
import json
import time
import math
import re
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

DATA_DIR = Path("data")

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

def clean_tokens(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9_\-']+\b", text.lower())

class DeterministicPipelineEngine:
    """
    High-Performance Deterministic Information Retrieval Engine (No-AI).
    Provides transparent mathematical scoring, exact n-gram span extraction,
    and stage-by-stage telemetry.
    """
    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        metadata_file = self.data_dir / "chunk_metadata.jsonl"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        print(f"[DeterministicEngine] Loading corpus from {metadata_file}...")
        self.metadata = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metadata.append(json.loads(line))
                    
        self.total_chunks = len(self.metadata)
        
        t0 = time.perf_counter()
        self.tokenized_corpus = [clean_tokens(c["text"]) for c in self.metadata]
        self.doc_lens = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / max(1, self.total_chunks)
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        t_index = (time.perf_counter() - t0) * 1000
        print(f"[DeterministicEngine] BM25 Index built in {t_index:.2f} ms ({self.total_chunks} chunks).")
        self._build_df_table()

    def _build_df_table(self):
        self.doc_freqs = {}
        for doc in self.tokenized_corpus:
            seen = set(doc)
            for term in seen:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

    def calculate_term_idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self.total_chunks - df + 0.5) / (df + 0.5) + 1.0)

    def stage1_query_normalization(self, raw_query: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_tokens = clean_tokens(raw_query)
        keywords = [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]
        active_terms = keywords if keywords else raw_tokens
        
        term_analysis = []
        for term in active_terms:
            idf = self.calculate_term_idf(term)
            df = self.doc_freqs.get(term, 0)
            term_analysis.append({
                "term": term,
                "df": df,
                "idf": round(idf, 4),
                "is_stopword": term in STOPWORDS
            })
            
        term_analysis.sort(key=lambda x: x["idf"], reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        
        return {
            "stage_id": 1,
            "stage_name": "Query Normalization & Token Analysis",
            "latency_ms": round(latency_ms, 3),
            "raw_query": raw_query,
            "raw_tokens": raw_tokens,
            "active_terms": active_terms,
            "term_analysis": term_analysis
        }

    def stage2_mathematical_bm25_search(self, active_terms: List[str], top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if not active_terms:
            return {
                "stage_id": 2,
                "stage_name": "Inverted Index & BM25 Ranking",
                "latency_ms": 0.0,
                "ranked_chunks": []
            }

        scores = self.bm25.get_scores(active_terms)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        ranked_chunks = []
        for rank_idx, doc_idx in enumerate(top_indices):
            raw_score = float(scores[doc_idx])
            if raw_score <= 0:
                continue
                
            chunk_meta = self.metadata[doc_idx]
            doc_tokens = self.tokenized_corpus[doc_idx]
            doc_len = len(doc_tokens)
            
            tf_map = {}
            for t in doc_tokens:
                if t in active_terms:
                    tf_map[t] = tf_map.get(t, 0) + 1
                    
            breakdown = []
            for term in active_terms:
                tf = tf_map.get(term, 0)
                idf = self.calculate_term_idf(term)
                denom = tf + k1 * (1.0 - b + b * (doc_len / self.avgdl))
                term_score = idf * (tf * (k1 + 1.0)) / max(1e-9, denom)
                breakdown.append({
                    "term": term,
                    "tf": tf,
                    "idf": round(idf, 3),
                    "term_score": round(term_score, 4)
                })
                
            ranked_chunks.append({
                "rank": rank_idx + 1,
                "chunk_id": chunk_meta.get("chunk_id", doc_idx),
                "doc_id": chunk_meta.get("doc_id", "unknown"),
                "score": round(raw_score, 4),
                "doc_len": doc_len,
                "text": chunk_meta.get("text", ""),
                "score_breakdown": breakdown
            })
            
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "stage_id": 2,
            "stage_name": "Inverted Index & BM25 Ranking",
            "latency_ms": round(latency_ms, 3),
            "corpus_evaluated": self.total_chunks,
            "top_k": top_k,
            "ranked_chunks": ranked_chunks
        }

    def stage3_deterministic_span_extraction(self, query: str, active_terms: List[str], top_chunk: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if not top_chunk:
            return {
                "stage_id": 3,
                "stage_name": "Deterministic Span Extractor",
                "latency_ms": 0.0,
                "best_sentence": "Không tìm thấy đoạn văn phù hợp trong tập dữ liệu.",
                "confidence_score": 0.0,
                "sentence_index": -1,
                "matched_terms": []
            }

        text = top_chunk.get("text", "")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 10]
        if not sentences:
            sentences = [text]

        query_set = set(active_terms)
        best_sent = ""
        best_sent_idx = 0
        best_score = -1.0
        
        for idx, sent in enumerate(sentences):
            sent_tokens = clean_tokens(sent)
            if not sent_tokens:
                continue
            matched_terms = [t for t in sent_tokens if t in query_set]
            unique_matches = len(set(matched_terms))
            density = len(matched_terms) / max(1, len(sent_tokens))
            sent_score = (unique_matches * 2.0) + (density * 3.0) - (idx * 0.05)
            
            if sent_score > best_score:
                best_score = sent_score
                best_sent = sent
                best_sent_idx = idx

        matched_in_best = set(clean_tokens(best_sent)).intersection(query_set)
        grounding_ratio = len(matched_in_best) / max(1, len(query_set))
        confidence = min(1.0, round(grounding_ratio * 0.7 + (min(best_score, 5.0) / 5.0) * 0.3, 3))
        
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "stage_id": 3,
            "stage_name": "Deterministic Span Extractor",
            "latency_ms": round(latency_ms, 3),
            "best_sentence": best_sent,
            "sentence_index": best_sent_idx + 1,
            "total_sentences": len(sentences),
            "matched_terms": list(matched_in_best),
            "confidence_score": confidence,
            "grounding_ratio": round(grounding_ratio, 3)
        }

    def stage4_telemetry_audit(self, stages: List[Dict[str, Any]]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        total_latency = sum(s.get("latency_ms", 0.0) for s in stages)
        return {
            "stage_id": 4,
            "stage_name": "System Telemetry & Audit",
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            "total_pipeline_latency_ms": round(total_latency, 2),
            "ai_elements_used": False,
            "determinism_guarantee": "100% Mathematical Certainty",
            "hardware_mode": "CPU-Optimized Ultra-Low Memory / Instant",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def execute_pipeline(self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> Dict[str, Any]:
        s1 = self.stage1_query_normalization(raw_query=query)
        active_terms = s1["active_terms"]
        s2 = self.stage2_mathematical_bm25_search(active_terms=active_terms, top_k=top_k, k1=k1, b=b)
        top_chunk = s2["ranked_chunks"][0] if s2["ranked_chunks"] else None
        s3 = self.stage3_deterministic_span_extraction(query=query, active_terms=active_terms, top_chunk=top_chunk)
        s4 = self.stage4_telemetry_audit([s1, s2, s3])
        
        return {
            "query": query,
            "telemetry": s4,
            "stages": [s1, s2, s3, s4],
            "result": {
                "top_snippet": s3["best_sentence"],
                "confidence": s3["confidence_score"],
                "grounded_source_chunk": top_chunk["chunk_id"] if top_chunk else None,
                "grounded_doc_id": top_chunk["doc_id"] if top_chunk else None,
                "top_chunks": s2["ranked_chunks"]
            }
        }
