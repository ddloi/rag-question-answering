"""Paired comparison of two research_ablation runs (e.g. RAG T5 v1 vs retrieved-context T5) on the same QA.

    python -m scripts.compare_ablation_runs --base results/research_v1_ablation_dev \
        --candidate results/research_v1_ablation_dev_rag_retrieved --output <new.json>
"""
import argparse
import gzip
import json
from pathlib import Path

from experiments.research_ablation import paired_bootstrap
from experiments.research_data import json_write
from experiments.verify_dataset import digest

POLICIES = ('official', 'canonical')


def load(run):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf8'))
    if manifest['status'] != 'complete':
        raise ValueError(f'Incomplete run: {run}')
    with gzip.open(run / 'predictions.jsonl.gz', 'rt', encoding='utf8') as stream:
        rows = {row['id']: row for row in map(json.loads, filter(str.strip, stream))}
    return manifest, rows


def compare(base_rows, candidate_rows, seed):
    if set(base_rows) != set(candidate_rows):
        raise ValueError('Runs must cover the same question IDs')
    ids = sorted(base_rows)
    arms = [name for name in next(iter(base_rows.values()))['arms'] if name in candidate_rows[ids[0]]['arms']]
    result = {}
    for arm in arms:
        result[arm] = {}
        for policy in POLICIES:
            result[arm][policy] = {}
            for metric in ('em', 'f1'):
                base = [base_rows[i]['arms'][arm]['scores'][policy][metric] for i in ids]
                cand = [candidate_rows[i]['arms'][arm]['scores'][policy][metric] for i in ids]
                result[arm][policy][metric] = {'base': sum(base) / len(ids), 'candidate': sum(cand) / len(ids),
                                               **paired_bootstrap(cand, base, seed)}
    return {'n': len(ids), 'unit': 'question', 'comparison': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base_manifest, base_rows = load(args.base)
    cand_manifest, cand_rows = load(args.candidate)
    if base_manifest['dataset_sha256'] != cand_manifest['dataset_sha256']:
        raise ValueError('Runs use different datasets')
    result = {**compare(base_rows, cand_rows, base_manifest['seed']),
              'base': {'run': args.base.as_posix(), 'rag_model_sha256': base_manifest['rag_model_sha256'],
                       'predictions_sha256': digest(args.base / 'predictions.jsonl.gz')},
              'candidate': {'run': args.candidate.as_posix(), 'rag_model_sha256': cand_manifest['rag_model_sha256'],
                            'predictions_sha256': digest(args.candidate / 'predictions.jsonl.gz')},
              'scope': 'Same dev questions and retrieval configs; only the RAG T5 checkpoint differs.'}
    json_write(args.output, result)
    for arm, values in result['comparison'].items():
        em = values['official']['em']
        print(f"{arm:16s} EM {100*em['base']:.1f} -> {100*em['candidate']:.1f} "
              f"delta {100*em['delta']:+.1f} CI [{100*em['ci95'][0]:+.1f}; {100*em['ci95'][1]:+.1f}]")


if __name__ == '__main__':
    main()
