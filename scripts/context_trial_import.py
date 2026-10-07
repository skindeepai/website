"""Import and validate a portable canonical context-trial evidence bundle.

Only explicit selections replace published trials. Older unrelated measurements
remain intact; every imported attempt remains available in the local manifest.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'results/long-context/trial-selection.json'
RESULT = ROOT / 'results/long-context/result.json'
METRICS = {'elapsed_seconds': 'wall_s', 'completion_tokens': 'completion_tokens',
           'peak_context_tokens': 'peak_sent_ctx', 'lm_calls': 'lm_calls'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def safe_path(root, relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, f'Unsafe source path: {relative}')
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root.resolve()), f'Source outside bundle: {relative}')
    return resolved


def validate_bundle(bundle):
    """Check copies and cross-check completed results without the originating host."""
    manifest = read_json(bundle / 'manifest.json')
    require(manifest.get('schema') == 'context-trial-manifest.v1', 'Unsupported trial manifest')
    indexed = {}
    records = set()
    for row in manifest['rows']:
        ident = row['id']
        require(ident not in indexed, f'Duplicate trial ID: {ident}')
        require(row['source_record_id'] not in records, f'Duplicate source record: {ident}')
        require(row['source_record_id'].split('/')[-1] == ident, f'Trial identity mismatch: {ident}')
        require(row['status'] in ('completed', 'interrupted', 'incomplete'), f'Unknown lifecycle: {ident}')
        require(row.get('completed') == (row['status'] == 'completed'), f'Completion status mismatch: {ident}')
        indexed[ident] = row
        records.add(row['source_record_id'])
        fingerprint_hashes = {name: row['sources'].get(name, {}).get('sha256') for name in ('stream', 'expected')}
        fingerprint = json.dumps(fingerprint_hashes, indent=2, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False) + '\n'
        expected_fingerprint = hashlib.sha256(fingerprint.encode()).hexdigest() if all(fingerprint_hashes.values()) else None
        require(expected_fingerprint == row['task_fingerprint'], f'Task fingerprint mismatch: {ident}')
        sources = {}
        for name, source in row['sources'].items():
            if not source.get('copy_path'):
                continue
            path = safe_path(bundle, source['copy_path'])
            content = path.read_bytes()
            require(len(content) == source['bytes'], f'Source size mismatch: {ident}/{name}')
            require(hashlib.sha256(content).hexdigest() == source['sha256'],
                    f'Source hash mismatch: {ident}/{name}')
            sources[name] = content.decode('utf-8')
        if row['status'] != 'completed':
            continue
        require(row['task_fingerprint'] is not None, f'Completed trial lacks task identity: {ident}')
        require(all(name in sources for name in ('verifier', 'task', 'expected', 'summary')),
                f'Completed trial lacks portable evidence: {ident}')
        grade = json.loads(sources['verifier'])
        expected = json.loads(sources['expected'])
        metadata = tomllib.loads(sources['task'])['metadata']
        summary = next(csv.DictReader(io.StringIO(sources['summary']), delimiter='\t'))
        require(type(row['correct']) is int and type(row['asked']) is int,
                f'Non-integer score: {ident}')
        require(0 <= row['correct'] <= row['asked'] and row['asked'] > 0, f'Invalid score: {ident}')
        require(type(row['void']) is bool, f'Missing validity: {ident}')
        for field, source in [('correct', 'correct'), ('asked', 'n'), ('void', 'void')]:
            require(row[field] == grade[source], f'Grader {field} mismatch: {ident}')
        require(row['asked'] == len(expected), f'Question-count mismatch: {ident}')
        require(row['task_metadata'] == metadata, f'Task metadata mismatch: {ident}')
        family = ('retention' if metadata.get('n_surprise', 0) else 'narrative') if metadata['kind'] == 'sparse' else {'kvstream': 'key_value'}.get(metadata['kind'], metadata['kind'])
        require(row['task_family'] == family, f'Task family mismatch: {ident}')
        for field, source in [('seed', 'seed'), ('target_tokens', 'target_tokens'),
                              ('input_tokens', 'stream_tokens'), ('mode', 'mode'), ('density', 'density')]:
            require(row.get(field) == metadata.get(source), f'Task {field} mismatch: {ident}')
        require(row['arm'] == summary['arm'] and row['seed'] == int(summary['seed']),
                f'Summary identity mismatch: {ident}')
        require(row['input_tokens'] == int(summary['size']), f'Summary input size mismatch: {ident}')
        for field, source in METRICS.items():
            require(row[field] == float(summary[source]), f'Summary {field} mismatch: {ident}')
    return manifest, indexed


def project_result(data, config, indexed):
    """Apply explicit verified identities; missing or unfinished selections fail closed."""
    result = copy.deepcopy(data)
    rows = {r['id']: r for r in result['comparison']['rows']}
    used = set()
    for selection in config['selections']:
        ident = selection['trial_id']
        require(ident in indexed, f'Selected trial absent from manifest: {ident}')
        trial = indexed[ident]
        require(trial['status'] == 'completed', f'Selected trial is not completed: {ident}')
        require(trial['source_record_id'] == selection['source_record_id'], f'Source identity mismatch: {ident}')
        for field, expected in selection['expect'].items():
            require(trial.get(field) == expected, f'Selected {field} mismatch: {ident}')
        key = (selection['site_id'], selection['seed'])
        require(key not in used, f'Duplicate site selection: {key}')
        used.add(key)
        require(selection['site_id'] in rows, f'Unknown site row: {selection["site_id"]}')
        parent = rows[selection['site_id']]
        family = {'narrative': 'reading', 'key_value': 'key_value'}.get(trial['task_family'], trial['task_family'])
        require(parent['task_family'] == family, f'Site task family mismatch: {ident}')
        if 'trials' in parent:
            candidates = [t for t in parent['trials'] if t.get('seed') == selection['seed']]
            require(len(candidates) == 1, f'Ambiguous or absent site seed: {key}')
            target = candidates[0]
        else:
            require(parent.get('seed') == selection['seed'], f'Site seed mismatch: {key}')
            target = parent
        target.update(correct=trial['correct'], asked=trial['asked'], void=trial['void'],
                      right=str(trial['correct']), status='measured', seed=trial['seed'],
                      input_tokens=trial['input_tokens'], elapsed_seconds=trial['elapsed_seconds'],
                      generated_tokens=trial['completion_tokens'], peak_tokens=trial['peak_context_tokens'],
                      calls=trial['lm_calls'], time=f'{trial["elapsed_seconds"] / 60:.1f} min',
                      written=f'{round(trial["completion_tokens"] / 1000)}K tokens',
                      peak=f'{round(trial["peak_context_tokens"] / 1000)}K',
                      evidence_id=trial['source_record_id'], canonical_trial_id=ident,
                      task_fingerprint=trial['task_fingerprint'])
        target.pop('no_answer', None)
    result['canonical_trials'] = {'manifest': config['bundle'] + '/manifest.json',
                                  'selection': 'results/long-context/trial-selection.json',
                                  'source_revision': config.get('source_revision')}
    for site_id in {key[0] for key in used}:
        parent = rows[site_id]
        if 'trials' in parent:
            parent['right'] = ' and '.join(f'{t["correct"]}/{t["asked"]}' for t in parent['trials'])
            for field in ('time', 'written', 'peak'):
                parent[field] = ' and '.join(t[field] for t in parent['trials'])
    return result


def preserve_history(previous, incoming):
    for ident, old in previous.items():
        require(ident in incoming, f'Incoming snapshot drops an existing attempt: {ident}')
        new = incoming[ident]
        require(old['source_record_id'] == new['source_record_id'], f'Existing source identity changed: {ident}')
        if old.get('task_fingerprint'):
            require(old['task_fingerprint'] == new['task_fingerprint'], f'Existing task identity changed: {ident}')
        if old['status'] == 'completed':
            require(new['status'] == 'completed', f'Completed trial regressed: {ident}')
            require(old['sources']['verifier']['sha256'] == new['sources']['verifier']['sha256'],
                    f'Completed grader evidence changed: {ident}')


def load_published(data):
    config = read_json(CONFIG)
    bundle = safe_path(ROOT, config['bundle'])
    require(hashlib.sha256((bundle / 'manifest.json').read_bytes()).hexdigest() == config['manifest_sha256'],
            'Local manifest differs from the imported snapshot')
    manifest, indexed = validate_bundle(bundle)
    return project_result(data, config, indexed), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, help='Canonical export directory; copied into the site for offline builds')
    parser.add_argument('--source-revision', help='Optional immutable lab revision for external citations')
    parser.add_argument('--check', action='store_true', help='Validate the local snapshot and reject a stale generated result record')
    args = parser.parse_args()
    require(not (args.check and args.bundle), '--check uses the already imported snapshot')
    config = read_json(CONFIG)
    if args.source_revision:
        require(not args.check, '--source-revision cannot change a check')
        require(len(args.source_revision) >= 7 and all(c in '0123456789abcdef' for c in args.source_revision),
                'Source revision must be a Git commit hash')
        config['source_revision'] = args.source_revision
    destination = safe_path(ROOT, config['bundle'])
    source = args.bundle or destination
    manifest, indexed = validate_bundle(source)
    if args.bundle and (destination / 'manifest.json').exists():
        _, previous = validate_bundle(destination)
        preserve_history(previous, indexed)
    manifest_hash = hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest()
    if args.check:
        require(manifest_hash == config['manifest_sha256'], 'Local manifest differs from the imported snapshot')
    else:
        config['manifest_sha256'] = manifest_hash
    data = read_json(RESULT)
    generated = project_result(data, config, indexed)
    if args.check:
        require(generated == data, 'Published result record is stale; run scripts/context_trial_import.py')
        print(f'Validated {len(indexed)} portable trials and {len(config["selections"])} selected results.')
        return
    if args.bundle and source.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            stage = Path(temporary) / 'evidence'
            stage.mkdir()
            shutil.copyfile(source / 'manifest.json', stage / 'manifest.json')
            for row in manifest['rows']:
                for descriptor in row['sources'].values():
                    if descriptor.get('copy_path'):
                        target = safe_path(stage, descriptor['copy_path'])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(safe_path(source, descriptor['copy_path']), target)
            validate_bundle(stage)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(stage), destination)
    CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    RESULT.write_text(json.dumps(generated, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Imported {len(indexed)} trials; refreshed {len(config["selections"])} selected results.')


if __name__ == '__main__':
    main()
