"""Audit frozen inference replay identity and answers; never retune the experiment."""
import argparse
import gzip
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.metrics import normalize_answer


ARMS = ('generator_only', 'retrieval_only', 'rag_full', 'rag_hybrid',
        'rag_checkpoint_no_context')
POLICIES = ('official', 'canonical')
METHOD_FILES = ('experiments/research_evaluate.py', 'experiments/research_pipeline.py',
                'experiments/metrics.py', 'ragqa/rag_engine.py')
COUNTS = {'dev': 200, 'test': 300}
TOLERANCE = 1e-12
SCOPE = ('Frozen inference replay of already evaluated QA splits; not a new independent '
         'holdout, retraining reproduction, or semantic faithfulness evaluation.')


class ReplayValidationError(ValueError):
    """The supplied evidence cannot support a valid replay comparison."""


def _require(condition, message):
    if not condition:
        raise ReplayValidationError(message)


def _sha(value, label):
    _require(isinstance(value, str) and len(value) == 64 and
             all(c in '0123456789abcdef' for c in value), f'{label}: invalid SHA256')
    return value


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _validate_lock(lock):
    _require(isinstance(lock, dict), 'Lock must be a JSON object')
    arms = lock.get('arms')
    _require(isinstance(arms, list) and all(isinstance(arm, str) for arm in arms) and
             set(arms) == set(ARMS) and len(arms) == len(ARMS),
             'Lock must name exactly the five frozen arms')
    _require(lock.get('test_n') == COUNTS['test'], 'Lock test_n must be 300')
    _sha(lock.get('dataset_manifest_sha256'), 'Lock dataset')
    _sha(lock.get('source_snapshot_sha256'), 'Lock source snapshot')
    _require(isinstance(lock.get('config'), dict), 'Lock config is missing')
    dataset = lock['config'].get('dataset')
    _require(isinstance(dataset, dict) and dataset.get('dev') == COUNTS['dev'] and
             dataset.get('test') == COUNTS['test'],
             'Lock must specify the frozen 200 dev / 300 test split sizes')
    _require(isinstance(lock.get('models_sha256'), dict), 'Lock model hashes are missing')
    _require(isinstance(lock.get('source_sha256'), dict), 'Lock method hashes are missing')
    for system in ('closed_book', 'rag'):
        _sha(lock.get('models_sha256', {}).get(system), f'Lock {system} model')
    for name in METHOD_FILES:
        _sha(lock.get('source_sha256', {}).get(name), f'Lock method {name}')


def _validate_manifest(manifest, lock, split, role):
    label = f'{split} {role} manifest'
    _require(isinstance(manifest, dict), f'{label}: must be an object')
    _require(manifest.get('status') == 'complete', f'{label}: status is not complete')
    _require(manifest.get('split') == split, f'{label}: wrong split')
    _require(manifest.get('pilot') is False, f'{label}: pilot must be false')
    _require(type(manifest.get('n')) is int and manifest['n'] == COUNTS[split],
             f'{label}: n must be {COUNTS[split]}')
    if split == 'test':
        _require(manifest.get('final') is True, f'{label}: test must be final')
    _require(manifest.get('config') == lock['config'], f'{label}: frozen config mismatch')
    _require(manifest.get('dataset_sha256') == lock['dataset_manifest_sha256'],
             f'{label}: dataset identity mismatch')
    for system in ('closed_book', 'rag'):
        _require(manifest.get(system + '_model_sha256') == lock['models_sha256'][system],
                 f'{label}: {system} model identity mismatch')
    # A snapshot ZIP can include other audit files or archive metadata. Only the
    # four method files are required to match the lock for every split/run.
    _require(isinstance(manifest.get('source_sha256'), dict), f'{label}: method hashes are missing')
    for name in METHOD_FILES:
        _require(manifest.get('source_sha256', {}).get(name) == lock['source_sha256'][name],
                 f'{label}: frozen method mismatch: {name}')
    _sha(manifest.get('predictions_sha256'), f'{label} predictions')
    _sha(manifest.get('source_snapshot_sha256'), f'{label} source snapshot')
    if split == 'test' and role == 'original':
        _require(manifest['source_snapshot_sha256'] == lock['source_snapshot_sha256'],
                 f'{label}: original test snapshot does not match the frozen lock')


def _indexed_rows(rows, split, role):
    label = f'{split} {role} predictions'
    _require(isinstance(rows, list), f'{label}: expected a list of JSONL records')
    indexed = {}
    for line, row in enumerate(rows, 1):
        _require(isinstance(row, dict) and isinstance(row.get('record'), dict),
                 f'{label} row {line}: missing record object')
        qid = row['record'].get('id')
        _require(isinstance(qid, str) and bool(qid.strip()), f'{label} row {line}: invalid QA ID')
        _require(qid not in indexed, f'{label}: duplicate QA ID {qid!r} at row {line}')
        systems = row.get('systems')
        _require(isinstance(systems, dict) and set(systems) == set(ARMS),
                 f'{label} {qid}: expected exactly the five frozen arms')
        for arm, output in systems.items():
            _require(isinstance(output, dict) and isinstance(output.get('answer'), str),
                     f'{label} {qid} {arm}: answer must be a string')
            _require(isinstance(output.get('context_used'), str) and
                     isinstance(output.get('used_passages'), list),
                     f'{label} {qid} {arm}: missing context/used_passages evidence')
            for passage in output['used_passages']:
                _require(isinstance(passage, dict) and isinstance(passage.get('chunk_id'), str),
                         f'{label} {qid} {arm}: malformed used passage')
            if arm in ('generator_only', 'rag_checkpoint_no_context'):
                _require(output['context_used'] == '' and output['used_passages'] == [],
                         f'{label} {qid} {arm}: a no-context arm contains evidence')
        retrieval = row.get('retrieval')
        _require(isinstance(retrieval, dict) and set(retrieval) == {'dense', 'bm25', 'hybrid'},
                 f'{label} {qid}: expected Dense/BM25/Hybrid rankings')
        for method, ranking in retrieval.items():
            _require(isinstance(ranking, list) and all(isinstance(p, dict) and
                     isinstance(p.get('chunk_id'), str) for p in ranking),
                     f'{label} {qid}: malformed {method} ranking')
        indexed[qid] = row
    _require(len(rows) == COUNTS[split], f'{label}: expected {COUNTS[split]} rows, got {len(rows)}')
    return indexed


def _qa_metrics(metrics, split, role):
    label = f'{split} {role} metrics'
    _require(isinstance(metrics, dict) and type(metrics.get('n')) is int and
             metrics['n'] == COUNTS[split], f'{label}: n must be {COUNTS[split]}')
    qa = metrics.get('qa')
    _require(isinstance(qa, dict) and set(qa) == set(ARMS), f'{label}: incorrect arm set')
    for arm, policies in qa.items():
        _require(isinstance(policies, dict) and set(policies) == set(POLICIES),
                 f'{label} {arm}: incorrect answer policies')
        for policy, values in policies.items():
            _require(isinstance(values, dict) and set(values) == {'em', 'f1'},
                     f'{label} {arm} {policy}: expected EM/F1')
            for metric, value in values.items():
                _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                         f'{label} {arm} {policy} {metric}: metric must be finite and in [0,1]')
    return qa


def _without_scores(passages):
    # Retrieval scores can differ by floating point rounding across devices.
    # IDs/order/text/visible evidence remain useful separate diagnostics.
    return [{key: value for key, value in p.items() if key != 'score'} for p in passages]


def compare_split(*, split, original_rows, replay_rows, original_manifest,
                  replay_manifest, original_metrics, replay_metrics, lock):
    """Pure comparison: invalid identity/schema raises; valid differences need review."""
    _require(split in COUNTS, f'Unsupported split: {split!r}')
    _validate_lock(lock)
    _validate_manifest(original_manifest, lock, split, 'original')
    _validate_manifest(replay_manifest, lock, split, 'replay')
    before = _indexed_rows(original_rows, split, 'original')
    after = _indexed_rows(replay_rows, split, 'replay')
    missing, extra = sorted(set(before) - set(after)), sorted(set(after) - set(before))
    _require(not missing and not extra, f'{split}: QA ID mismatch; missing={missing[:10]}, extra={extra[:10]}')
    ids = list(before)
    _require(all(before[qid]['record'] == after[qid]['record'] for qid in ids),
             f'{split}: QA input/reference records differ for matching IDs')
    old_qa = _qa_metrics(original_metrics, split, 'original')
    new_qa = _qa_metrics(replay_metrics, split, 'replay')
    raw = {arm: [qid for qid in ids if before[qid]['systems'][arm]['answer'] !=
                 after[qid]['systems'][arm]['answer']] for arm in ARMS}
    normalized = {arm: [qid for qid in ids if normalize_answer(before[qid]['systems'][arm]['answer']) !=
                        normalize_answer(after[qid]['systems'][arm]['answer'])] for arm in ARMS}
    deltas = {arm: {policy: {metric: new_qa[arm][policy][metric] - old_qa[arm][policy][metric]
                           for metric in ('em', 'f1')} for policy in POLICIES} for arm in ARMS}
    metric_match = all(abs(value) <= TOLERANCE for policies in deltas.values()
                       for metrics in policies.values() for value in metrics.values())
    ranking = {method: [qid for qid in ids if
               [p['chunk_id'] for p in before[qid]['retrieval'][method]] !=
               [p['chunk_id'] for p in after[qid]['retrieval'][method]]]
               for method in ('dense', 'bm25', 'hybrid')}
    retrieval_evidence = {method: [qid for qid in ids if
                          _without_scores(before[qid]['retrieval'][method]) !=
                          _without_scores(after[qid]['retrieval'][method])]
                          for method in ('dense', 'bm25', 'hybrid')}
    context = {arm: [qid for qid in ids if before[qid]['systems'][arm]['context_used'] !=
                    after[qid]['systems'][arm]['context_used']] for arm in ARMS}
    passages = {arm: [qid for qid in ids if
                     _without_scores(before[qid]['systems'][arm]['used_passages']) !=
                     _without_scores(after[qid]['systems'][arm]['used_passages'])] for arm in ARMS}
    evidence_equal = not any(values for group in (ranking, retrieval_evidence, context, passages)
                             for values in group.values())
    return {
        'status': 'PASS_EXACT_ANSWERS' if not any(raw.values()) and metric_match else 'REVIEW_REQUIRED',
        'n': len(ids), 'expected_n': COUNTS[split], 'unique_exact_ids': True,
        'identical_qa_records': True, 'complete_manifest_and_frozen_identity': True,
        'raw_answer_mismatch_ids': raw, 'normalized_answer_mismatch_ids': normalized,
        'qa_metric_deltas': deltas, 'qa_metrics_equal_within_tolerance': metric_match,
        'qa_metric_tolerance': TOLERANCE,
        'evidence_diagnostics': {
            'status': 'MATCH' if evidence_equal else 'REVIEW_REQUIRED',
            'scope': 'Separate from answer PASS; compares ranking order and evidence without floating point retrieval scores.',
            'ranking_mismatch_ids': ranking, 'retrieval_evidence_mismatch_ids': retrieval_evidence,
            'context_mismatch_ids': context, 'used_passage_mismatch_ids': passages,
            'source_snapshot_archive_equal': original_manifest['source_snapshot_sha256'] == replay_manifest['source_snapshot_sha256'],
            'source_snapshot_archive_equality_required': False,
        },
        'environments': {role: {key: manifest.get(key) for key in ('python', 'packages')}
                         for role, manifest in (('original', original_manifest), ('replay', replay_manifest))},
    }


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ReplayValidationError(f'Cannot read JSON {path}: {exc}') from exc


def _load_run(path):
    path = Path(path)
    manifest = _read_json(path / 'manifest.json')
    metrics = _read_json(path / 'metrics.json')
    checks = {'manifest_sha256': _digest(path / 'manifest.json'),
              'metrics_sha256': _digest(path / 'metrics.json')}
    for filename, field in (('predictions.jsonl.gz', 'predictions_sha256'),
                            ('source_snapshot.zip', 'source_snapshot_sha256')):
        try:
            observed = _digest(path / filename)
        except OSError as exc:
            raise ReplayValidationError(f'Missing run evidence {path / filename}: {exc}') from exc
        _require(observed == manifest.get(field), f'{path}: {filename} checksum mismatch')
        checks[field] = observed
    rows = []
    try:
        with gzip.open(path / 'predictions.jsonl.gz', 'rt', encoding='utf8') as stream:
            for line, text in enumerate(stream, 1):
                try:
                    rows.append(json.loads(text))
                except ValueError as exc:
                    raise ReplayValidationError(f'{path}: invalid prediction JSON at row {line}: {exc}') from exc
    except (OSError, EOFError, UnicodeError) as exc:
        raise ReplayValidationError(f'Cannot read prediction evidence {path}: {exc}') from exc
    return {'manifest': manifest, 'metrics': metrics, 'rows': rows,
            'file_checks': {'run_dir': str(path), **checks}}


def compare_replay(*, original_runs, replay_runs, lock_path, project_root):
    """Load immutable artifacts and validate their own digests before comparing."""
    lock_path, project_root = Path(lock_path), Path(project_root)
    lock = _read_json(lock_path)
    _validate_lock(lock)
    source_checks = {}
    for name in METHOD_FILES:
        try:
            observed = _digest(project_root / name)
        except OSError as exc:
            raise ReplayValidationError(f'Cannot read current frozen method {name}: {exc}') from exc
        source_checks[name] = {'expected': lock['source_sha256'][name], 'observed': observed,
                               'matches': observed == lock['source_sha256'][name]}
        _require(source_checks[name]['matches'], f'Current frozen method checksum mismatch: {name}')
    comparisons, file_checks = {}, {}
    for split in COUNTS:
        original, replay = _load_run(original_runs[split]), _load_run(replay_runs[split])
        comparisons[split] = compare_split(split=split, original_rows=original['rows'], replay_rows=replay['rows'],
            original_manifest=original['manifest'], replay_manifest=replay['manifest'],
            original_metrics=original['metrics'], replay_metrics=replay['metrics'], lock=lock)
        file_checks[split] = {'original': original['file_checks'], 'replay': replay['file_checks']}
    return {'status': 'PASS_EXACT_ANSWERS' if all(c['status'] == 'PASS_EXACT_ANSWERS'
                for c in comparisons.values()) else 'REVIEW_REQUIRED',
            'scope': SCOPE, 'created_at': datetime.now(timezone.utc).isoformat(),
            'validator_sha256': _digest(Path(__file__)),
            'lock': {'path': str(lock_path), 'sha256': _digest(lock_path)},
            'dataset_manifest_sha256': lock['dataset_manifest_sha256'],
            'models_sha256': lock['models_sha256'], 'current_method_source_checks': source_checks,
            'identity_scope': 'Current method files and run manifest input identities match the lock; each predictions/snapshot file matches its own manifest digest. Model weight and dataset byte preflights are separate evidence.',
            'artifact_checks': file_checks, 'comparisons': comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ('original', 'replay'):
        for split in COUNTS:
            parser.add_argument(f'--{role}-{split}', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='New JSON path; existing evidence is preserved')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f'Output already exists; choose a new evidence path: {args.output}')
    try:
        result = compare_replay(
            original_runs={s: getattr(args, 'original_' + s) for s in COUNTS},
            replay_runs={s: getattr(args, 'replay_' + s) for s in COUNTS},
            lock_path=args.lock, project_root=args.project_root)
        code = 0  # Valid REVIEW_REQUIRED evidence must still be saved/exported.
    except (ReplayValidationError, OSError) as exc:
        result = {'status': 'INVALID_PROVENANCE', 'scope': SCOPE, 'error': str(exc),
                  'created_at': datetime.now(timezone.utc).isoformat()}
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.exit(main())
