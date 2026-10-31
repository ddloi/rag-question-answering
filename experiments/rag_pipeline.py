"""Three-system adapter using the same tested inference as the web.
Generator-only is a no-context ablation of the same fine-tuned checkpoint.
"""
from ragqa.paths import DATA_DIR, MODEL_DIR
from ragqa.rag_engine import RAGNeuralEngine, EMBED_MODEL_NAME

TOP_K = 5
GENERATOR_PATH = str(MODEL_DIR)
GENERATION_OPTIONS = dict(max_new_tokens=32, num_beams=4, early_stopping=True,
                          no_repeat_ngram_size=3, length_penalty=0.8)


class RAGSystem(RAGNeuralEngine):
    def __init__(self, data_dir=DATA_DIR, model_dir=MODEL_DIR, top_k=TOP_K):
        super().__init__(data_dir, model_dir)
        self.top_k = top_k
        self.load_sync()
        if not self.is_loaded:
            raise RuntimeError(self.load_error)
        if self.qa_model is None:
            raise RuntimeError("Three-system evaluation requires the local DistilBERT cache.")

    def generate_with_evidence(self, question, retrieved):
        import torch
        selected = retrieved[:2]
        context = " ".join(row["text"] for row in selected)
        prefix = f"answer the question from the context. question: {question} context: "
        inputs = self.tokenizer(prefix + context, return_tensors="pt", truncation=True,
                                max_length=512, return_offsets_mapping=True)
        offsets = inputs.pop("offset_mapping")[0].tolist()
        visible_end = max((end for start, end in offsets), default=0)
        visible = context[:max(0, visible_end - len(prefix))]
        used, cursor = [], 0
        for passage in selected:
            length = max(0, min(len(passage["text"]), len(visible) - cursor))
            if length:
                used.append({**passage, "text": passage["text"][:length]})
            cursor += len(passage["text"]) + 1
        with torch.inference_mode():
            ids = self.generator.generate(**inputs.to(self.device), **GENERATION_OPTIONS)
        return {"answer": self.tokenizer.decode(ids[0], skip_special_tokens=True).strip(),
                "retrieved": retrieved, "context_used": visible, "used_passages": used,
                "input_tokens": int(inputs["input_ids"].shape[-1])}

    def answer_rag_full(self, question):
        return self.generate_with_evidence(question, self.retrieve(question, self.top_k))

    def answer_retrieval_only(self, question, retrieved=None):
        retrieved = self.retrieve(question, self.top_k) if retrieved is None else retrieved
        used = []
        for passage in retrieved[:3]:
            encoded = self.qa_tokenizer(question, passage["text"], max_length=512,
                                        truncation=True, return_offsets_mapping=True)
            ends = [end for (start, end), sid in zip(encoded["offset_mapping"], encoded.sequence_ids()) if sid == 1]
            used.append({**passage, "text": passage["text"][:max(ends, default=0)]})
        return {"answer": self.answer_extractive_only(question, retrieved), "retrieved": retrieved,
                "context_used": " ".join(p["text"] for p in used), "used_passages": used}


def demo():
    rag = RAGSystem()
    question = "Proteus and Nereid are moons of which planet?"
    print("Generator-only:", rag.answer_generator_only(question))
    print("Extractive:", rag.answer_retrieval_only(question)["answer"])
    print("RAG:", rag.answer_rag_full(question)["answer"])


if __name__ == "__main__":
    demo()
