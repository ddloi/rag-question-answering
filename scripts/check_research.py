"""Read-only preflight for the frozen research dataset and optional trained checkpoint."""
import argparse
import json
import math
from pathlib import Path

from ragqa.paths import DATA_DIR, resolve_project_path
from experiments.research_data import validate_disjoint, json_write
from experiments.research_pipeline import dataset_manifest, ResearchGenerator
from experiments.metrics import contains_answer
from experiments.verify_dataset import read_jsonl, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR / 'research_v1')
    parser.add_argument('--model-run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--download-base', action='store_true', help='Allow fetching pinned base weights for the parameter-update check on a clean machine')
    args = parser.parse_args()
    data_dir = resolve_project_path(args.data_dir)
    manifest = dataset_manifest(data_dir)
    splits = {s: read_jsonl(data_dir / f'{s}_qa.jsonl') for s in ('train', 'dev', 'test')}
    validate_disjoint(splits)
    chunks = read_jsonl(data_dir / 'chunk_metadata.jsonl')
    by_id = {r['chunk_id']: r for r in chunks}
    if len(by_id) != len(chunks):
        raise ValueError('Duplicate chunk IDs')
    for row in read_jsonl(data_dir / 'training_accepted.jsonl'):
        if not contains_answer(by_id[row['positive_chunk_id']]['text'], [row['answer']]):
            raise ValueError('Accepted training context lost its canonical evidence')
    import faiss
    import numpy as np
    index = faiss.read_index(str(data_dir / 'faiss.index'))
    if index.ntotal != len(chunks) or index.d != manifest['embedding_dimension']:
        raise ValueError('Index shape mismatch')
    # Check all vectors in bounded batches instead of duplicating the entire index in RAM.
    for start in range(0, index.ntotal, 4096):
        vectors = index.reconstruct_n(start, min(4096, index.ntotal - start))
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3, rtol=0):
            raise ValueError('Non-finite or non-unit embeddings')
    report = {'dataset': 'PASS', 'counts': manifest['counts'], 'all_vectors_finite_and_unit': True,
              'question_overlap': 0, 'checkpoint': 'NOT RUN'}
    documents_by_split = {name: set().union(*(set(r['gold_doc_ids']) for r in rows))
                          for name, rows in splits.items()}
    report['shared_document_counts'] = {f'{left}/{right}': len(documents_by_split[left] & documents_by_split[right])
        for left, right in (('train', 'dev'), ('train', 'test'), ('dev', 'test'))}
    report['generalization_scope'] = 'Unseen QA within a shared controlled corpus; not an unseen-document split.'
    if args.model_run:
        model_run = resolve_project_path(args.model_run)
        info = json.loads((model_run / 'manifest.json').read_text(encoding='utf8'))
        if info['status'] != 'complete' or info['dataset_manifest_sha256'] != digest(data_dir / 'manifest.json'):
            raise ValueError('Incomplete checkpoint or wrong dataset')
        if info['optimizer_steps'] < 1 or digest(model_run / 'model/model.safetensors') != info['model_sha256']:
            raise ValueError('Checkpoint hash/optimizer-step check failed')
        logs = read_jsonl(model_run / 'training_log.jsonl')
        for row in logs:
            for key in ('loss', 'grad_norm', 'eval_loss'):
                if key in row and not math.isfinite(float(row[key])):
                    raise ValueError(f'Non-finite training log: {key}')
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        base = hf_hub_download(info['base_model']['name'], 'model.safetensors',
                               revision=info['base_model']['revision'], local_files_only=not args.download_base)
        name = 'encoder.block.0.layer.0.SelfAttention.q.weight'
        with safe_open(base, framework='pt', device='cpu') as before, safe_open(
                str(model_run / 'model/model.safetensors'), framework='pt', device='cpu') as after:
            delta = (before.get_tensor(name).float() - after.get_tensor(name).float()).abs().max().item()
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError('Selected trainable parameter did not change from initialization')
        generator = ResearchGenerator(model_run / 'model', manifest['config']['generation'])
        model = generator.model
        pointers = [model.shared.weight.data_ptr(), model.encoder.embed_tokens.weight.data_ptr(),
                    model.decoder.embed_tokens.weight.data_ptr(), model.lm_head.weight.data_ptr()]
        if len(set(pointers)) != 1:
            raise ValueError('T5 tied weights were not restored')
        question = splits['dev'][0]['question']
        sample = generator.answer(question)
        report.update(checkpoint='PASS', pilot=info['pilot'], precision=info['precision'],
            selected_parameter_max_update=delta, tied_weights_restored=True,
            restored_inference_has_answer=bool(sample['answer']), restored_input_tokens=sample['input_tokens'],
            peak_training_gpu_bytes=info['peak_gpu_allocated_bytes'])
    if args.output:
        output = resolve_project_path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        json_write(output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
