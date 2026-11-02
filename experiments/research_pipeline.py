"""Explicit retrieval and generation stages for the versioned research protocol."""
import json
import time
from pathlib import Path

from ragqa.engine import clean_tokens
from ragqa.rag_engine import RAGNeuralEngine
from .verify_dataset import digest, read_jsonl


def prompt(question, context=''):
    return f'question: {question} context: {context}'


def context_text(passages):
    return '\n\n'.join(f"Source [{i}] | {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))


def encode_context(tokenizer, question, passages, max_tokens=512):
    context = context_text(passages)
    prefix = prompt(question)
    encoded = tokenizer(prefix + context, truncation=True, max_length=max_tokens,
                        return_offsets_mapping=True, return_tensors='pt')
    offsets = encoded.pop('offset_mapping')[0].tolist()
    end = max((end for _, end in offsets), default=0)
    visible = context[:max(0, end - len(prefix))]
    used, cursor = [], 0
    for i, passage in enumerate(passages, 1):
        heading = f"Source [{i}] | {passage['title']}\n"
        start = cursor + len(heading)
        length = max(0, min(len(passage['text']), len(visible) - start))
        heading_length = max(0, min(len(heading), len(visible) - cursor))
        if heading_length or length:
            used.append({**passage, 'citation_id': i, 'text': passage['text'][:length],
                         'visible_heading': heading[:heading_length], 'title_only': not bool(length)})
        cursor += len(heading) + len(passage['text']) + 2
    return encoded, visible, used


def dataset_manifest(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] != 'complete':
        raise ValueError('Research dataset/index is not complete')
    for name in ('chunk_metadata.jsonl', 'documents.jsonl', 'faiss.index', 'train_qa.jsonl',
                 'dev_qa.jsonl', 'test_qa.jsonl', 'training_accepted.jsonl', 'training_rejected.jsonl'):
        if digest(data_dir / name) != manifest['output_sha256'][name]:
            raise ValueError(f'Frozen dataset checksum mismatch: {name}')
    return manifest


class ResearchRetriever:
    def __init__(self, data_dir, with_sparse=True, device='cpu'):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.data_dir = Path(data_dir)
        self.manifest = dataset_manifest(self.data_dir)
        self.config = self.manifest['config']
        self.metadata = read_jsonl(self.data_dir / 'chunk_metadata.jsonl')
        self.index = faiss.read_index(str(self.data_dir / 'faiss.index'))
        info = self.manifest['models']['embedding']
        self.encoder = SentenceTransformer(info['name'], revision=info['revision'], device=device)
        if self.index.ntotal != len(self.metadata) or self.index.d != self.encoder.get_embedding_dimension():
            raise ValueError('Index/metadata/encoder mismatch')
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError('Expected the frozen inner-product index')
        if self.encoder.max_seq_length != self.manifest['embedding_max_tokens']:
            raise ValueError('Encoder token budget differs from the frozen index')
        self.bm25 = None
        if with_sparse:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi([clean_tokens(p['text']) for p in self.metadata])

    def dense_many(self, questions, top_k):
        vectors = self.encoder.encode(questions, batch_size=32, normalize_embeddings=True,
                                      convert_to_numpy=True).astype('float32')
        scores, indices = self.index.search(vectors, top_k)
        return [[{**self.metadata[int(idx)], 'rank': rank, 'score': float(score)}
                 for rank, (score, idx) in enumerate(zip(row_scores, row_indices), 1) if idx >= 0]
                for row_scores, row_indices in zip(scores, indices)]

    def all_rankings(self, question, pool=None):
        import numpy as np
        pool = pool or self.config['retrieval']['pool']
        dense = self.dense_many([question], pool)[0]
        if self.bm25 is None:
            raise ValueError('Sparse ranking was not initialized')
        scores = self.bm25.get_scores(clean_tokens(question))
        sparse_ids = [int(i) for i in np.argsort(-scores, kind='stable') if scores[i] > 0][:pool]
        sparse = [{**self.metadata[i], 'rank': rank, 'score': float(scores[i])}
                  for rank, i in enumerate(sparse_ids, 1)]
        candidates, fusion = {}, {}
        constant = self.config['retrieval']['rrf_constant']
        for ranking in (dense, sparse):
            for rank, row in enumerate(ranking, 1):
                candidates[row['chunk_id']] = row
                fusion[row['chunk_id']] = fusion.get(row['chunk_id'], 0) + 1 / (constant + rank)
        ids = sorted(fusion, key=lambda key: (-fusion[key], key))[:pool]
        hybrid = [{**candidates[key], 'rank': rank, 'score': fusion[key]}
                  for rank, key in enumerate(ids, 1)]
        return {'dense': dense, 'bm25': sparse, 'hybrid': hybrid}


class ResearchGenerator:
    def __init__(self, model_dir, generation_config, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True).to(self.device).eval()
        self.config = generation_config

    def answer(self, question, passages=()):
        import torch
        started = time.perf_counter()
        inputs, context, used = encode_context(self.tokenizer, question, passages, self.config['max_input_tokens'])
        options = {k: v for k, v in self.config.items() if k != 'max_input_tokens'}
        with torch.inference_mode():
            output = self.model.generate(**inputs.to(self.device), **options, early_stopping=True)
        return {'answer': self.tokenizer.decode(output[0], skip_special_tokens=True).strip(),
                'context_used': context, 'used_passages': used,
                'input_tokens': int(inputs['input_ids'].shape[-1]),
                'latency_ms': round((time.perf_counter() - started) * 1000, 2)}


class ResearchExtractor:
    def __init__(self, model_info, device=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
        self.adapter = RAGNeuralEngine()
        self.adapter.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.adapter.qa_tokenizer = AutoTokenizer.from_pretrained(model_info['name'], revision=model_info['revision'])
        self.adapter.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            model_info['name'], revision=model_info['revision']).to(self.adapter.device).eval()
        self.adapter.is_loaded = True

    def answer(self, question, passages):
        started = time.perf_counter()
        used = []
        for row in passages[:3]:
            encoding = self.adapter.qa_tokenizer(question, row['text'], truncation=True,
                max_length=512, return_offsets_mapping=True)
            end = max((end for (_, end), sid in zip(encoding['offset_mapping'], encoding.sequence_ids()) if sid == 1), default=0)
            used.append({**row, 'text': row['text'][:end]})
        return {'answer': self.adapter.answer_extractive_only(question, passages),
                'context_used': '\n\n'.join(p['text'] for p in used), 'used_passages': used,
                'latency_ms': round((time.perf_counter() - started) * 1000, 2)}
