"""Train a RAG T5 on retrieved (Dense + cross-encoder reranked) contexts instead of the gold passage only.

research_v1 trained the RAG T5 on one gold passage, but at inference it reads two retrieved passages,
often with distractors. This run keeps everything else identical to research_v1 (same 1,865 train QA
IDs, base model, hyperparameters, seed, prompt and token budget) and changes only the training context:

* use the top-2 reranked passages when they already show the answer within the token budget;
* otherwise put the gold passage together with the best retrieved distractor, in a seeded random order,
  so the target is always grounded in visible context;
* dev (for eval loss and checkpoint selection) uses the same reranked top-2 as inference, without gold.

    python -m experiments.research_train_retrieved                      # full run (~15 min on RTX 3050)
    python -m experiments.research_train_retrieved --max-steps 2 --train-limit 8 --dev-limit 4 --output <dir>
"""
import argparse
import importlib.metadata
import json
import math
import platform
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ragqa.paths import DATA_DIR, PROJECT_ROOT, resolve_project_path
from .metrics import contains_answer
from .research_ablation import RERANKER, passage_text, rerank
from .research_artifacts import source_snapshot
from .research_data import json_write
from .research_pipeline import context_text, dataset_manifest, encode_context, prompt
from .verify_dataset import digest, read_jsonl

CONTEXT_POLICY = ('Top-2 Dense+cross-encoder passages when the answer is visible within the token budget; '
                  'otherwise gold positive passage plus the best retrieved distractor in seeded random order. '
                  'Dev uses reranked top-2 without gold.')


def answer_visible(tokenizer, row, passages, max_tokens):
    _, visible, _ = encode_context(tokenizer, row['question'], passages, max_tokens)
    return contains_answer(visible, [row['answer']])


def build_training_context(tokenizer, row, reranked, gold, max_tokens, rng, top_n=2):
    """Return (passages, kind) or (None, 'rejected'). kind records how the context was formed."""
    retrieved = reranked[:top_n]
    if answer_visible(tokenizer, row, retrieved, max_tokens):
        return retrieved, 'retrieved'
    distractors = [p for p in reranked if p['chunk_id'] != gold['chunk_id']][:top_n - 1]
    orders = [[gold, *distractors], [*distractors, gold]]
    if rng.random() < 0.5:
        orders.reverse()
    for passages in orders:
        if answer_visible(tokenizer, row, passages, max_tokens):
            return passages, 'gold_plus_distractor'
    return None, 'rejected'


def retrieve_reranked(data_dir, questions, pool, reranker_dir):
    import gc
    import torch
    from sentence_transformers import CrossEncoder
    from .research_pipeline import ResearchRetriever
    retriever = ResearchRetriever(data_dir, with_sparse=False)
    dense = retriever.dense_many(questions, pool)
    del retriever
    encoder = CrossEncoder(str(reranker_dir), device='cuda' if torch.cuda.is_available() else 'cpu', max_length=512)
    reranked = [rerank(ranking, encoder.predict([(q, passage_text(p)) for p in ranking],
                                                batch_size=32, show_progress_bar=False).tolist())
                for q, ranking in zip(questions, dense)]
    del encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reranked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR / 'research_v1')
    parser.add_argument('--reference-run', type=Path, help='research_v1 RAG run whose train IDs are reused')
    parser.add_argument('--reranker', type=Path, default=Path(RERANKER['local_dir']))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-steps', type=int, default=-1, help='Positive values create a pilot run')
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--dev-limit', type=int)
    args = parser.parse_args()
    data_dir = resolve_project_path(args.data_dir)
    dataset_info = dataset_manifest(data_dir)
    config = dataset_info['config']
    data_hash = digest(data_dir / 'manifest.json')
    reference_dir = resolve_project_path(args.reference_run or PROJECT_ROOT / 'checkpoints' / config['version'] / 'rag')
    reference = json.loads((reference_dir / 'manifest.json').read_text(encoding='utf8'))
    if reference['status'] != 'complete' or reference['pilot'] or reference['dataset_manifest_sha256'] != data_hash:
        raise ValueError('The reference research_v1 RAG run must be complete, non-pilot and on this dataset')
    reranker_dir = resolve_project_path(args.reranker)
    if not (reranker_dir / 'model.safetensors').is_file():
        raise FileNotFoundError(f'Missing reranker at {reranker_dir}')
    pilot = args.max_steps > 0 or args.train_limit is not None or args.dev_limit is not None
    out = resolve_project_path(args.output or PROJECT_ROOT / 'checkpoints' / config['version'] /
                               (('pilot_' if pilot else '') + 'rag_retrieved'))
    if out.exists():
        raise FileExistsError(f'Preserving model run: {out}; choose a new --output')
    out.mkdir(parents=True)

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed)
    set_seed(config['seed'])
    info = dataset_info['models']['generator_base']
    if info != reference['base_model']:
        raise ValueError('Base model differs from the reference run')
    tokenizer = AutoTokenizer.from_pretrained(info['name'], revision=info['revision'])
    budget = config['generation']['max_input_tokens']
    top_n = config['retrieval']['context_top_n']
    metadata = {r['chunk_id']: r for r in read_jsonl(data_dir / 'chunk_metadata.jsonl')}
    wanted = set(reference['train_ids'])
    train = [r for r in read_jsonl(data_dir / 'training_accepted.jsonl') if r['id'] in wanted]
    dev = read_jsonl(data_dir / 'dev_qa.jsonl')
    if len(train) != len(wanted):
        raise ValueError('Some reference train IDs are missing from training_accepted.jsonl')
    if args.train_limit is not None:
        train = train[:args.train_limit]
    if args.dev_limit is not None:
        dev = dev[:args.dev_limit]

    print('Retrieving and reranking contexts for train/dev...', flush=True)
    pool = config['retrieval']['pool']
    ranked = retrieve_reranked(data_dir, [r['question'] for r in train + dev], pool, reranker_dir)
    train_ranked, dev_ranked = ranked[:len(train)], ranked[len(train):]
    rng = random.Random(config['seed'])
    usable, kinds, rejected, context_log = [], Counter(), [], []
    for row, reranked in zip(train, train_ranked):
        passages, kind = build_training_context(tokenizer, row, reranked, metadata[row['positive_chunk_id']],
                                                budget, rng, top_n)
        kinds[kind] += 1
        if passages is None:
            rejected.append(row['id'])
            continue
        usable.append({'question': row['question'], 'answer': row['answer'], 'context': context_text(passages)})
        context_log.append({'id': row['id'], 'kind': kind, 'chunk_ids': [p['chunk_id'] for p in passages]})
    dev_rows = [{'question': r['question'], 'answer': r['answer'], 'context': context_text(ranks[:top_n])}
                for r, ranks in zip(dev, dev_ranked)]
    with (out / 'training_contexts.jsonl').open('w', encoding='utf8') as stream:
        for item in context_log:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f'Training contexts: {dict(kinds)}', flush=True)
    if not usable or not dev_rows:
        raise ValueError('No usable train/dev rows')

    def preprocess(batch):
        encoded = tokenizer([prompt(q, c) for q, c in zip(batch['question'], batch['context'])],
                            max_length=budget, truncation=True)
        encoded['labels'] = tokenizer(text_target=batch['answer'])['input_ids']
        return encoded
    train_set = Dataset.from_list(usable)
    dev_set = Dataset.from_list(dev_rows)
    train_set = train_set.map(preprocess, batched=True, remove_columns=train_set.column_names)
    dev_set = dev_set.map(preprocess, batched=True, remove_columns=dev_set.column_names)
    del metadata

    # Training arguments mirror experiments/research_train.py so only the context policy differs.
    model = AutoModelForSeq2SeqLM.from_pretrained(info['name'], revision=info['revision'])
    model.config.use_cache = False
    options = config['training']
    precision = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp32'
    if not pilot and precision != reference['precision']:
        raise ValueError(f'Precision {precision} differs from the reference run ({reference["precision"]})')
    every = max(1, args.max_steps // 2) if pilot else None
    training_args = Seq2SeqTrainingArguments(output_dir=str(out),
        per_device_train_batch_size=options['micro_batch'], per_device_eval_batch_size=2,
        gradient_accumulation_steps=options['gradient_accumulation'],
        num_train_epochs=options['epochs'], learning_rate=options['learning_rate'],
        weight_decay=options['weight_decay'], bf16=precision == 'bf16', fp16=False,
        gradient_checkpointing=True, eval_strategy='steps' if pilot else 'epoch',
        save_strategy='steps' if pilot else 'epoch', eval_steps=every, save_steps=every or 500,
        logging_steps=1 if pilot else 10, save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, report_to='none',
        seed=config['seed'], data_seed=config['seed'], dataloader_num_workers=0,
        max_steps=args.max_steps, max_grad_norm=1.0)
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'system': 'rag', 'variant': 'retrieved_context', 'pilot': pilot, 'precision': precision,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'dataset_manifest_sha256': data_hash, 'config': config, 'base_model': info,
        'reference_run': {'path': reference_dir.relative_to(PROJECT_ROOT).as_posix()
                          if reference_dir.is_relative_to(PROJECT_ROOT) else str(reference_dir),
                          'model_sha256': reference['model_sha256']},
        'context_policy': CONTEXT_POLICY, 'context_counts': dict(kinds),
        'reranker': {**RERANKER, 'model_sha256': digest(reranker_dir / 'model.safetensors'), 'pool': pool},
        'train_ids': [r['id'] for r in train if r['id'] not in set(rejected)],
        'dev_ids': [r['id'] for r in dev], 'context_rejected_ids': rejected,
        'prompt_template': reference['prompt_template'],
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'datasets', 'accelerate', 'sentence-transformers')},
        'source_snapshot_sha256': source_snapshot(out / 'source_snapshot.zip'),
        'source_sha256': {name: digest(PROJECT_ROOT / name) for name in (
            'experiments/research_train_retrieved.py', 'experiments/research_pipeline.py',
            'experiments/research_ablation.py', 'experiments/metrics.py')}}
    json_write(out / 'manifest.json', manifest)

    class FiniteLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            for key in ('loss', 'eval_loss', 'grad_norm'):
                if key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f'Non-finite {key}')
            with (out / 'training_log.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps({'step': state.global_step, **logs}) + '\n')
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=dev_set,
        processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), callbacks=[FiniteLog()])
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        result = trainer.train()
        if not math.isfinite(result.training_loss):
            raise FloatingPointError('Training loss is not finite')
        target = out / 'model'
        trainer.save_model(str(target)); tokenizer.save_pretrained(str(target))
        trainer.save_state()
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
            training_metrics=result.metrics, optimizer_steps=trainer.state.global_step,
            best_validation_loss=trainer.state.best_metric, best_checkpoint=trainer.state.best_model_checkpoint,
            elapsed_seconds=time.perf_counter() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            model_sha256=digest(target / 'model.safetensors'))
        print(json.dumps({k: manifest[k] for k in ('status', 'pilot', 'optimizer_steps', 'elapsed_seconds',
                                                    'best_validation_loss', 'context_counts')}, indent=2), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        json_write(out / 'manifest.json', manifest)
        if not pilot:
            evidence = PROJECT_ROOT / 'results' / 'training' / config['version'] / out.name
            evidence.mkdir(parents=True, exist_ok=True)
            for filename in ('manifest.json', 'training_log.jsonl', 'trainer_state.json',
                             'source_snapshot.zip', 'training_contexts.jsonl'):
                if (out / filename).is_file():
                    shutil.copyfile(out / filename, evidence / filename)


if __name__ == '__main__':
    main()
