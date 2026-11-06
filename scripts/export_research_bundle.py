"""Export explicit source/data/model/evidence bundles without caches or credentials."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from ragqa.paths import PROJECT_ROOT, resolve_project_path
from experiments.verify_dataset import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('source', 'data', 'models', 'evidence'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/research_v1'))
    args = parser.parse_args()
    output = resolve_project_path(args.output)
    data_dir = resolve_project_path(args.data_dir)
    data_prefix = data_dir.relative_to(PROJECT_ROOT).as_posix()
    if output.exists():
        raise FileExistsError(output)
    patterns = {
        'source': ['app.py', 'README.md', 'ruff.toml', '.gitignore', '.gitattributes',
            'experiments/*.py', 'experiments/README.md', 'ragqa/*.py', 'ragqa/static/*', 'scripts/*.py', 'tests/*.py',
            'deployment/*.py', 'deployment/huggingface/*.py', 'deployment/huggingface/requirements.txt',
            'configs/*.json', 'configs/README.md', 'requirements/*.txt', 'requirements/README.md', 'docs/**/*.md',
            'notebooks/README.md', 'notebooks/research_reproduce_colab.ipynb',
            'notebooks/research_verify_frozen_colab.ipynb',
            'docs/assets/**/*', 'results/research_v1_label_audit_dev.json',
            'data/provenance/legacy_exclusions.json', 'data/research_v1/manifest.json', 'data/research_v1/replay_reference.json'],
        'data': [f'{data_prefix}/{name}' for name in ('manifest.json', 'replay_reference.json',
            'chunk_metadata.jsonl', 'documents.jsonl', 'faiss.index', '*_qa.jsonl', 'training_accepted.jsonl', 'training_rejected.jsonl')],
        'models': ['checkpoints/research_v1/closed_book/model/*', 'checkpoints/research_v1/rag/model/*'] +
            [f'checkpoints/research_v1/{system}/{name}' for system in ('closed_book', 'rag')
             for name in ('manifest.json', 'training_log.jsonl', 'trainer_state.json', 'source_snapshot.zip')],
        'evidence': ['results/research_*/*', 'results/research_*.json', 'results/research_*.zip', 'results/training/research_v1/**/*',
            'docs/RESEARCH*.md', 'docs/HOC_VA_VAN_HANH.md', 'docs/REVIEW_RUBRIC.md',
            'docs/CASES_RESEARCH_V1.md', 'docs/PHAN_BIEN_VA_TAP_TRINH_BAY.md', 'docs/assets/**/*',
            'docs/UI_VERIFICATION_RESEARCH_V1.md', 'docs/COLAB_VERIFICATION_RESEARCH_V1.md', 'docs/CLEANUP.md', 'docs/README.md',
            'docs/history/*.md',
            'notebooks/research_verify_frozen_colab.ipynb',
            f'{data_prefix}/manifest.json', f'{data_prefix}/replay_reference.json']}
    files = sorted({p for pattern in patterns[args.kind] for p in PROJECT_ROOT.glob(pattern) if p.is_file()})
    if not files:
        raise ValueError('No files to export')
    if args.kind == 'models':
        for system in ('closed_book', 'rag'):
            manifest = PROJECT_ROOT / 'checkpoints/research_v1' / system / 'manifest.json'
            if not manifest.exists() or json.loads(manifest.read_text(encoding='utf8'))['status'] != 'complete':
                raise ValueError('Both model runs must be complete before export')
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'kind': args.kind, 'data_dir': data_prefix, 'created_at': datetime.now(timezone.utc).isoformat(),
        'files_sha256': {p.relative_to(PROJECT_ROOT).as_posix(): digest(p) for p in files}}
    with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, 'RAG/' + path.relative_to(PROJECT_ROOT).as_posix())
        archive.writestr('bundle_manifest.json', json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({'path': str(output), 'files': len(files), 'bytes': output.stat().st_size, 'sha256': digest(output)}, indent=2))


if __name__ == '__main__':
    main()
