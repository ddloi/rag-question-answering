"""Train separate closed-book and RAG T5 models under one frozen protocol."""
import argparse
import json
import math
import importlib.metadata
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from ragqa.paths import DATA_DIR, PROJECT_ROOT, resolve_project_path
from .metrics import contains_answer
from .research_data import json_write
from .research_pipeline import dataset_manifest, prompt, context_text, ResearchRetriever
from .verify_dataset import digest, read_jsonl
from .research_artifacts import source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', choices=('closed_book', 'rag'), required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR / 'research_v1')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-steps', type=int, default=-1, help='Positive values create a pilot run')
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--dev-limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    data_dir = resolve_project_path(args.data_dir)
    dataset_info = dataset_manifest(data_dir)
    config = dataset_info['config']
    pilot = args.max_steps > 0 or args.train_limit is not None or args.dev_limit is not None
    suffix = ('pilot_' if pilot else '') + args.system
    out = resolve_project_path(args.output or PROJECT_ROOT / 'checkpoints' / config['version'] / suffix)
    if out.exists() and not args.resume:
        raise FileExistsError(f'Preserving model run: {out}; use --resume to restore training state')
    if args.resume and (out / 'manifest.json').exists():
        if json.loads((out / 'manifest.json').read_text(encoding='utf8'))['status'] == 'complete':
            raise ValueError('Completed model runs are immutable; create a new output for a new experiment')
    out.mkdir(parents=True, exist_ok=True)
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
        Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed)
    from transformers.trainer_utils import get_last_checkpoint
    set_seed(config['seed'])
    info = dataset_info['models']['generator_base']
    tokenizer = AutoTokenizer.from_pretrained(info['name'], revision=info['revision'])
    metadata = {r['chunk_id']: r for r in read_jsonl(data_dir / 'chunk_metadata.jsonl')}
    train = read_jsonl(data_dir / 'training_accepted.jsonl')
    dev = read_jsonl(data_dir / 'dev_qa.jsonl')
    if args.train_limit is not None:
        train = train[:args.train_limit]
    if args.dev_limit is not None:
        dev = dev[:args.dev_limit]
    rejected, usable = [], []
    # Check the RAG input even for closed-book, so both models receive identical QA IDs.
    for row in train:
        passage = metadata[row['positive_chunk_id']]
        context = context_text([passage])
        prefix = prompt(row['question'])
        encoded = tokenizer(prefix + context, max_length=config['generation']['max_input_tokens'],
                            truncation=True, return_offsets_mapping=True)
        end = max((end for _, end in encoded['offset_mapping']), default=0)
        visible = context[:max(0, end - len(prefix))]
        target = tokenizer(text_target=row['answer'])['input_ids']
        if not contains_answer(visible, [row['answer']]) or len(target) > config['generation']['max_new_tokens']:
            rejected.append(row['id']); continue
        usable.append({**row, 'context': context if args.system == 'rag' else ''})
    train = usable
    if not train or not dev:
        raise ValueError('No usable train/dev rows')
    if args.system == 'rag':
        retriever = ResearchRetriever(data_dir, with_sparse=False)
        rankings = retriever.dense_many([r['question'] for r in dev], config['retrieval']['top_k'])
        dev_contexts = [context_text(r[:config['retrieval']['context_top_n']]) for r in rankings]
        del retriever
    else:
        dev_contexts = [''] * len(dev)
    dev = [{**row, 'context': ctx} for row, ctx in zip(dev, dev_contexts)]
    def preprocess(batch):
        encoded = tokenizer([prompt(q, c) for q, c in zip(batch['question'], batch['context'])],
            max_length=config['generation']['max_input_tokens'], truncation=True)
        # Preserve complete targets; long train targets were explicitly rejected above.
        encoded['labels'] = tokenizer(text_target=batch['answer'])['input_ids']
        return encoded
    train_set = Dataset.from_list(train)
    dev_set = Dataset.from_list(dev)
    train_set = train_set.map(preprocess, batched=True, remove_columns=train_set.column_names)
    dev_set = dev_set.map(preprocess, batched=True, remove_columns=dev_set.column_names)
    del metadata
    model = AutoModelForSeq2SeqLM.from_pretrained(info['name'], revision=info['revision'])
    model.config.use_cache = False
    train_options = config['training']
    # Pilot FP16 overflowed on T5. Prefer BF16's wider range when supported;
    # otherwise keep FP32 (including typical Colab T4), rather than hide overflow.
    precision = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp32'
    every = max(1, args.max_steps // 2) if pilot else None
    training_args = Seq2SeqTrainingArguments(output_dir=str(out),
        per_device_train_batch_size=train_options['micro_batch'], per_device_eval_batch_size=2,
        gradient_accumulation_steps=train_options['gradient_accumulation'],
        num_train_epochs=train_options['epochs'], learning_rate=train_options['learning_rate'],
        weight_decay=train_options['weight_decay'], bf16=precision == 'bf16', fp16=False,
        gradient_checkpointing=True, eval_strategy='steps' if pilot else 'epoch',
        save_strategy='steps' if pilot else 'epoch', eval_steps=every, save_steps=every or 500,
        logging_steps=1 if pilot else 10, save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model='eval_loss', greater_is_better=False, report_to='none',
        seed=config['seed'], data_seed=config['seed'], dataloader_num_workers=0,
        max_steps=args.max_steps, max_grad_norm=1.0)
    training_fingerprint = digest(data_dir / 'training_accepted.jsonl')
    manifest = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
        'system': args.system, 'pilot': pilot, 'precision': precision,
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'dataset_manifest_sha256': digest(data_dir / 'manifest.json'),
        'config': config, 'base_model': info, 'training_source_sha256': training_fingerprint,
        'train_ids': [r['id'] for r in train], 'dev_ids': [r['id'] for r in dev],
        'token_budget_rejected_ids': rejected, 'prompt_template': 'question: {question} context: {context}',
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'datasets', 'accelerate')},
        'source_snapshot_sha256': source_snapshot(out / 'source_snapshot.zip'),
        'source_sha256': {name: digest(PROJECT_ROOT / name) for name in
                          ('experiments/research_train.py', 'experiments/research_pipeline.py', 'experiments/metrics.py')}}
    if args.resume:
        previous = json.loads((out / 'manifest.json').read_text(encoding='utf8'))
        for key in ('system', 'pilot', 'precision', 'dataset_manifest_sha256', 'train_ids', 'dev_ids', 'source_sha256'):
            if manifest[key] != previous[key]:
                raise ValueError(f'Resume mismatch: {key}')
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
        resume = get_last_checkpoint(str(out)) if args.resume else None
        if args.resume and resume is None:
            raise ValueError('No restorable Trainer checkpoint')
        result = trainer.train(resume_from_checkpoint=resume)
        if not math.isfinite(result.training_loss):
            raise FloatingPointError('Training loss is not finite')
        target = out / 'model'
        trainer.save_model(str(target)); tokenizer.save_pretrained(str(target))
        trainer.save_state()
        manifest.update(status='complete', finished_at=datetime.now(timezone.utc).isoformat(),
            training_metrics=result.metrics, optimizer_steps=trainer.state.global_step,
            best_validation_loss=trainer.state.best_metric, elapsed_seconds=time.perf_counter() - started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
            model_sha256=digest(target / 'model.safetensors'))
        print(json.dumps({k: manifest[k] for k in ('status', 'system', 'pilot', 'optimizer_steps', 'elapsed_seconds', 'best_validation_loss')}, indent=2), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        json_write(out / 'manifest.json', manifest)
        evidence = PROJECT_ROOT / 'results' / 'training' / config['version'] / out.name
        evidence.mkdir(parents=True, exist_ok=True)
        for filename in ('manifest.json', 'training_log.jsonl', 'trainer_state.json', 'source_snapshot.zip'):
            if (out / filename).is_file():
                shutil.copyfile(out / filename, evidence / filename)


if __name__ == '__main__':
    main()
