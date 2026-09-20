"""Independent arithmetic, frozen-input, parity and trace audit (no inference)."""
import hashlib
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chat-label-suite/replay-100'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p, r, summary = [read(OUT / (name + '.json')) for name in ['protocol', 'records', 'result']]
    ps, parsed, pp = [read(OUT / (name + '.json')) for name in ['parser-result', 'parser-records', 'parser-protocol']]
    assert not (OUT / 'failure.json').exists()
    assert sha(OUT / 'source.py') == sha(ROOT / 'experiments/chat_label_suite.py') == p['source_sha256']
    assert summary['protocol_sha256'] == sha(OUT / 'protocol.json')
    assert all(sha(ROOT / f) == h for f, h in p['inputs'].items())
    assert pp['analysisSourceSha256'] == sha(ROOT / 'experiments/analyze_chat_label_suite.cjs')
    assert pp['parserSourceSha256'] == sha(ROOT / 'experiments/censor_decision_parsers.cjs')
    assert pp['sourceRecordsSha256'] == sha(OUT / 'records.json')
    assert pp['sourceResultSha256'] == sha(OUT / 'result.json')
    common = read(ROOT / 'results/chat-smoke/protocol.json')
    ids = [x['id'] for x in p['evaluation']]
    truth = {x['id']: x['expected'] for x in p['evaluation']}
    assert ids == common['splits']['evaluation'] and len(set(ids)) == 100
    assert Counter(truth.values()) == {0: 50, 1: 50}
    data_path = ROOT / 'experiments/.cache/toxicchat/toxic-chat_annotation_test.csv'
    assert sha(data_path) == p['dataset_sha256']
    with data_path.open(encoding='utf-8', newline='') as f:
        dataset = list(csv.DictReader(f))
    for rid, label in truth.items():
        assert rid.startswith('test:')
        source = dataset[int(rid.split(':')[1])]
        assert source['human_annotation'] == 'True' and int(source['toxicity']) == label
    assert len(r) == len({(x['id'], x['method'], x['allow']) for x in r}) == 1500
    expected_order = []
    for orders in [p['order'], p['distill_order']]:
        for rid, order in zip(ids, orders):
            expected_order.extend((rid, *pair) for pair in order)
    assert [(x['id'], x['method'], x['allow']) for x in r] == expected_order
    mlp = {x['id']: x for x in read(ROOT / 'results/chat-smoke-heads/predictions.json')['mlp']['evaluation']}
    tiny = {x['id']: x for x in read(ROOT / 'results/chat-smoke-specialist/predictions.json') if x['split'] == 'evaluation'}
    distill = {x['id']: x for x in read(ROOT / 'results/chat-smoke-adaptation/distill.json')['predictions'] if x['split'] == 'evaluation'}
    parity = 0
    for x in r:
        assert x['expected'] == truth[x['id']]
        assert math.isfinite(x['total_ms']) and x['total_ms'] > 0
        depth = x['qwen_depth']
        assert depth in [0, 6, 12, 18, 24]
        steps = len(x['generated_tokens']) if x['method'] == 'generated' else 1
        assert x['executed_qwen_layers'] == list(range(1, depth+1))*steps
        assert x['bert_layers'] == ([1, 2] if x['method'] in ['tiny', 'cascade'] else [])
        if x['method'] == 'generated':
            assert 1 <= steps <= 8 and depth == 24
        if x['method'] in ['full', 'distill', 'direct']:
            assert depth == 24
        elif x['method'] == 'fixed12':
            assert depth == 12
        elif x['method'] == 'learned':
            assert depth in [6, 12, 18, 24]
        elif x['method'] in ['tiny', 'cascade']:
            prob = x['tiny_block_probability']
            assert 0 <= prob <= 1
            if x['method'] == 'tiny':
                assert depth == 0 and x['prediction'] == int(prob >= .3)
            else:
                fallback = .05 < prob < .8
                assert x['fallback'] == fallback and depth == (24 if fallback else 0)
                if not fallback:
                    assert x['prediction'] == int(prob >= .8)
        if x['allow'] not in ['SAFE', 'unchanged'] or x['method'] in ['generated', 'direct']:
            continue
        method = x['method']
        if method in ['full', 'fixed12']:
            d = 24 if method == 'full' else 12
            probs = mlp[x['id']]['probabilities'][str(d)]
            assert x['prediction'] == int(probs[1] > probs[0]) and depth == d
        elif method == 'learned':
            ref = mlp[x['id']]['learned']
            assert (x['prediction'], depth) == (ref['prediction'], ref['depth'])
        elif method in ['tiny', 'cascade']:
            ref = tiny[x['id']]
            assert x['prediction'] == ref['specialist' if method == 'tiny' else 'cascade']
            assert depth == (24 if method == 'cascade' and ref['fallback'] else 0)
        elif method == 'distill':
            assert x['prediction'] == distill[x['id']]['heads']['24']['prediction'] and depth == 24
        else:
            raise AssertionError(method)
        parity += 1
    assert parity == 600
    for key, s in summary['summaries'].items():
        group = [x for x in r if x['method']+'_'+x['allow'] == key]
        assert len(group) == s['n'] == 100
        assert s['correct'] == sum(x['prediction'] == x['expected'] for x in group)
        assert s['invalid'] == sum(x['prediction'] is None for x in group)
        assert s['toxic_not_blocked'] == sum(x['expected'] == 1 and x['prediction'] != 1 for x in group)
        assert s['false_block'] == sum(x['expected'] == 0 and x['prediction'] == 1 for x in group)
        assert abs(s['mean_ms']-sum(x['total_ms'] for x in group)/100) < 1e-8
        assert s['qwen_exit_counts'] == dict(Counter(str(x['qwen_depth']) for x in group))
        assert s['mean_qwen_blocks'] == sum(len(x['executed_qwen_layers']) for x in group)/100
        assert s['mean_generated_tokens'] == sum(len(x['generated_tokens']) for x in group)/100
    assert len(parsed) == 600 and ps['numericChecks'] == 3900 and ps['numericPredictionsChanged'] == 0
    raw = {(x['id'], x['allow']): x for x in r if x['method'] == 'generated'}
    for name, values in ps['summaries'].items():
        for allow, s in values.items():
            group = [x for x in parsed if x['parser'] == name and x['allow'] == allow]
            assert len(group) == len({x['id'] for x in group}) == 100
            for x in group:
                source = raw[(x['id'], allow)]
                assert x['output'] == source['output'] and x['original_prediction'] == source['prediction']
                assert x['prediction'] == int(x['blocked'])
                assert x['expected'] == truth[x['id']]
                assert x['parseOk'] or x['blocked']
            assert s['correct'] == sum(x['prediction'] == x['expected'] for x in group)
            assert s['correct'] == s['parsedCorrect']+s['fallbackCorrect']
            assert s['parsed']+s['fallbackBlocked'] == 100
            assert s['toxicMissed']+s['safeBlocked']+s['correct'] == 100
            assert s['recoveredInvalid'] == sum(x['parseOk'] and x['original_prediction'] is None for x in group)
            assert s['recoveredCorrect'] == sum(x['parseOk'] and x['original_prediction'] is None and x['prediction'] == x['expected'] for x in group)
            assert s['parsed'] == sum(x['parseOk'] for x in group)
            assert s['parsedCorrect'] == sum(x['parseOk'] and x['prediction'] == x['expected'] for x in group)
            assert s['fallbackBlocked'] == sum(not x['parseOk'] for x in group)
            assert s['fallbackCorrect'] == sum(not x['parseOk'] and x['expected'] == 1 for x in group)
            assert s['fallbackFalseBlocks'] == sum(not x['parseOk'] and x['expected'] == 0 for x in group)
            assert s['toxicMissed'] == sum(x['expected'] == 1 and x['prediction'] == 0 for x in group)
            assert s['safeBlocked'] == sum(x['expected'] == 0 and x['prediction'] == 1 for x in group)
            exact = {x['id']: x for x in parsed if x['parser'] == 'exactWithFallback' and x['allow'] == allow}
            assert s['changesFromExactFallback'] == sum(x['prediction'] != exact[x['id']]['prediction'] for x in group)
            assert s['correctedFromExactFallback'] == sum(x['prediction'] == x['expected'] and exact[x['id']]['prediction'] != x['expected'] for x in group)
            assert s['addedErrorsFromExactFallback'] == sum(x['prediction'] != x['expected'] and exact[x['id']]['prediction'] == x['expected'] for x in group)
            assert s['modelMeanMs'] == summary['summaries']['generated_'+allow]['mean_ms']
    for method, comparison in ps['promptComparisons'].items():
        safe = [x for x in r if x['method'] == method and x['allow'] == 'SAFE']
        ok = {x['id']: x for x in r if x['method'] == method and x['allow'] == 'OK'}
        assert len(safe) == len(ok) == 100
        assert comparison['changed'] == sum(x['prediction'] != ok[x['id']]['prediction'] for x in safe)
        assert comparison['okCorrected'] == sum(x['prediction'] != x['expected'] and ok[x['id']]['prediction'] == x['expected'] for x in safe)
        assert comparison['okAddedErrors'] == sum(x['prediction'] == x['expected'] and ok[x['id']]['prediction'] != x['expected'] for x in safe)
        assert comparison['changedDepth'] == sum(x['qwen_depth'] != ok[x['id']]['qwen_depth'] for x in safe)
        assert all(x['input_tokens'] == ok[x['id']]['input_tokens'] for x in safe)
    # Independent output-only checks with no model work.
    for name in ['supplied', 'exactWithFallback', 'enhanced']:
        assert {x['allow'] for x in parsed if x['parser'] == name} == {'SAFE', 'OK'}
    full = {x['id']: x for x in r if x['method'] == 'full' and x['allow'] == 'SAFE'}
    fixed_ok = [x for x in r if x['method'] == 'fixed12' and x['allow'] == 'OK']
    added = [x for x in fixed_ok if x['prediction'] != x['expected'] and full[x['id']]['prediction'] == x['expected']]
    corrected = [x for x in fixed_ok if x['prediction'] == x['expected'] and full[x['id']]['prediction'] != x['expected']]
    full_comparison = dict(added_errors=len(added), corrected_errors=len(corrected),
        new_toxic_misses=sum(x['expected'] == 1 for x in added),
        new_false_blocks=sum(x['expected'] == 0 for x in added),
        added_error_ids=[x['id'] for x in added], corrected_error_ids=[x['id'] for x in corrected])
    report = dict(status='passed', calls=len(r), original_prediction_parity=parity,
        actual_layer_traces=len(r), parser_records=len(parsed), numeric_parser_identity_checks=3900,
        source_and_frozen_inputs_verified=True, order_verified=True,
        fixed12_ok_versus_full_safe=full_comparison,
        audit_source_sha256=sha(Path(__file__)),
        limits='Arithmetic and execution-artifact audit. Does not establish fresh-data validity, universal parser correctness, or repeatable latency.')
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
