"""Exploratory frozen-feature head/gate comparisons; no transformer timing claims."""
import os
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chat-smoke-heads'
CACHE = ROOT / 'experiments/.cache/toxicchat'
DEPTHS = [6, 12, 18, 24]


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def load_data(manifest):
    old = json.loads((ROOT / 'results/chat600/protocol.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    by_id = {}
    for source in ['train', 'test']:
        path = CACHE / f'toxic-chat_annotation_{source}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == old['data_sha256'][source]
        for i, row in enumerate(csv.DictReader(path.open(encoding='utf-8', newline=''))):
            by_id[f'{source}:{i}'] = int(row['toxicity'])
    cache = {s: torch.load(CACHE / f'{fingerprint}-{s}.pt', weights_only=True) for s in old['splits']}
    locations = {rid: (s, i) for s, ids in old['splits'].items() for i, rid in enumerate(ids)}
    result = {}
    for name, ids in manifest['splits'].items():
        result[name] = {'ids': ids, 'y': torch.tensor([by_id[rid] for rid in ids]),
                        'x': {d: torch.stack([cache[locations[rid][0]][d][locations[rid][1]] for rid in ids]) for d in DEPTHS}}
    return result


def measures(y, pred, depth, full=None):
    decided = pred >= 0
    toxic = y == 1
    m = {'n': len(y), 'correct': int((pred == y).sum()), 'missed_toxic': int((toxic & (pred == 0)).sum()),
         'false_block': int((~toxic & (pred == 1)).sum()), 'toxic': int(toxic.sum()),
         'review': int((~decided).sum()), 'review_toxic': int((~decided & toxic).sum()),
         'coverage': float(decided.float().mean()),
         'covered_accuracy': float((pred[decided] == y[decided]).float().mean()) if decided.any() else None,
         'mean_depth': float(depth.float().mean()), 'projected_blocks_skipped': float((24-depth).float().mean()/24),
         'exit_counts': {str(d): int((depth == d).sum()) for d in DEPTHS}}
    if full is not None:
        m.update({'added_errors': int(((full == y) & (pred != y) & decided).sum()),
                  'corrected_errors': int(((full != y) & (pred == y)).sum()),
                  'additional_missed_toxic': int((toxic & (full == 1) & (pred == 0)).sum())})
    return m


def train_heads(data, kind):
    torch.manual_seed(109)
    probs = {s: {} for s in data}
    arrays = {}
    y = data['train']['y']
    w = torch.bincount(y, minlength=2).float().reciprocal()
    w /= w.mean()
    for d in DEPTHS:
        x = data['train']['x'][d]
        mean, std = x.mean(0), x.std(0).clamp_min(.05)
        if kind == 'linear':
            model = torch.nn.Linear(x.shape[1], 2)
            lr, steps = .01, 300
        else:
            model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.GELU(),
                                        torch.nn.Dropout(.25), torch.nn.Linear(64, 2))
            lr, steps = .002, 200
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.1)
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model((x-mean)/std), y, weight=w)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            logits = model((data['tune']['x'][d]-mean)/std)
            temperature = min([.5, 1., 1.5, 2., 3., 4., 6., 8.],
                              key=lambda t: float(torch.nn.functional.cross_entropy(logits/t, data['tune']['y'])))
            for s in data:
                probs[s][d] = model((data[s]['x'][d]-mean)/std).div(temperature).softmax(1)
        for k, v in model.state_dict().items():
            arrays[f'{d}_{k}'] = v.numpy()
        arrays[f'{d}_mean'], arrays[f'{d}_std'] = mean.numpy(), std.numpy()
        arrays[f'{d}_temperature'] = np.array(temperature)
    np.savez_compressed(OUT / f'{kind}-heads.npz', **arrays)
    return probs


def gate_features(probs, d):
    p = probs[d]
    conf, pred = p.max(1)
    entropy = -(p * p.clamp_min(1e-9).log()).sum(1)
    if d == 6:
        prev_conf, agree, delta = torch.zeros_like(conf), torch.zeros_like(conf), torch.zeros_like(conf)
    else:
        prev = probs[d-6]
        prev_conf = prev.max(1).values
        agree = (prev.argmax(1) == pred).float()
        delta = p[:, 1]-prev[:, 1]
    return torch.stack([p[:, 1], conf, entropy, prev_conf, agree, delta], 1)


def fit_gates(data, probs, kind):
    torch.manual_seed(110)
    scores = {s: {} for s in data}
    arrays = {}
    y = data['tune']['y']
    full = probs['tune'][24].argmax(1)
    for d in DEPTHS[:-1]:
        x = gate_features(probs['tune'], d)
        mean, std = x.mean(0), x.std(0).clamp_min(.05)
        current = probs['tune'][d].argmax(1)
        # Two distinct targets: current error risk and recoverable error risk.
        target = torch.stack([(current != y).float(), ((current != y) & (full == y)).float()], 1)
        model = torch.nn.Linear(x.shape[1], 2)
        opt = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.1)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model((x-mean)/std), target)
            loss.backward()
            opt.step()
        with torch.inference_mode():
            for s in data:
                scores[s][d] = model((gate_features(probs[s], d)-mean)/std).sigmoid()
        for k, v in model.state_dict().items():
            arrays[f'{d}_{k}'] = v.numpy()
        arrays[f'{d}_mean'], arrays[f'{d}_std'] = mean.numpy(), std.numpy()
    np.savez_compressed(OUT / f'{kind}-gates.npz', **arrays)
    return scores


def decide(probs, policy, scores=None):
    pred = probs[24].argmax(1).clone()
    depth = torch.full_like(pred, 24)
    for d in DEPTHS[:-1]:
        p = probs[d]
        conf, label = p.max(1)
        accept = depth == 24
        if policy['type'] == 'learned':
            accept &= (scores[d][:, 0] <= policy['error_limit']) & (scores[d][:, 1] <= policy['benefit_limit'])
        else:
            threshold = torch.where(label == 0, policy['safe_threshold'], policy['block_threshold'])
            accept &= conf >= threshold
            if policy['agreement']:
                accept &= False if d == 6 else label == probs[d-6].argmax(1)
        pred[accept], depth[accept] = label[accept], d
    return pred, depth


def select_policy(data, probs, policies, scores=None):
    y = data['calibration']['y']
    full = probs['calibration'][24].argmax(1)
    candidates = []
    for policy in policies:
        pred, depth = decide(probs['calibration'], policy, scores['calibration'] if scores else None)
        m = measures(y, pred, depth, full)
        allowed = m['added_errors'] <= len(y)*.01 and m['additional_missed_toxic'] == 0
        candidates.append({'policy': policy, 'selection_metrics': m, 'eligible': allowed})
    eligible = [r for r in candidates if r['eligible']]
    chosen = min(eligible, key=lambda r: (r['selection_metrics']['mean_depth'], r['selection_metrics']['added_errors']))
    return chosen, candidates


def review_policy(probs, threshold):
    pred = probs[24].argmax(1).clone()
    pred[probs[24].max(1).values < threshold] = -1
    return pred, torch.full_like(pred, 24)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / 'results/chat-smoke/protocol.json'
    manifest = json.loads(manifest_path.read_text())
    data = load_data(manifest)
    assert all(k in data for k in ['train', 'tune', 'calibration', 'evaluation'])
    assert all(not set(data[a]['ids']) & set(data[b]['ids']) for a, b in itertools.combinations(data, 2))
    protocol = {'type': 'Exploratory reuse, not confirmatory evaluation',
                'shared_protocol_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'split_sizes': {s: len(v['ids']) for s, v in data.items()},
                'training': 'Train-only standardized heads. Linear 300 steps lr .01; MLP64 GELU dropout .25 200 steps lr .002. AdamW weight_decay .1, inverse-frequency class weights. Seed109.',
                'calibration': 'Temperature grid fit on tune; the same split trains logistic error/benefit gates. This reuse may overfit development; threshold selection and evaluation are disjoint.',
                'selection': 'Select minimum mean depth with <=1% added errors and zero additional toxic misses on calibration. Independent harms, no net-error cancellation. No inferential confidence guarantee.',
                'review': 'Full-depth abstention thresholds select covered accuracy >=95% and coverage >=20%; if none, REVIEW all. Reviewed cases never counted as correct.',
                'execution': 'Two CPU threads; cached hidden states. Report projected blocks only, no latency claim; thresholds/gate overhead and real stopped forwards unmeasured.'}
    write('protocol.json', protocol)
    results, predictions, sweeps = {}, {}, {}
    old = np.load(ROOT / 'results/chat600/heads.npz', allow_pickle=False)
    reference = {s: {} for s in data if s != 'train'}
    with torch.inference_mode():
        for s in reference:
            for d in DEPTHS:
                h = {k: torch.tensor(old[f'{d}_{k}']) for k in ['mean', 'std', 'weight', 'bias', 'temperature']}
                reference[s][d] = torch.nn.functional.linear((data[s]['x'][d]-h['mean'])/h['std'], h['weight'], h['bias']).div(h['temperature']).softmax(1)
    results['original_1400_linear'] = {
        f'fixed_{d}': {'note': 'Previous larger training set and previously tuned temperatures; context only, not matched training budget.',
                       'splits': {s: measures(data[s]['y'], reference[s][d].argmax(1), torch.full_like(data[s]['y'], d), reference[s][24].argmax(1)) for s in reference}}
        for d in DEPTHS}
    for kind in ['linear', 'mlp']:
        print(f'Train {kind}', flush=True)
        probs = train_heads(data, kind)
        scores = fit_gates(data, probs, kind)
        predictions[kind] = {s: [{'id': rid, 'label': int(data[s]['y'][i]),
                                 'probabilities': {str(d): [float(v) for v in probs[s][d][i]] for d in DEPTHS},
                                 'gate_scores': {str(d): [float(v) for v in scores[s][d][i]] for d in DEPTHS[:-1]}}
                                for i, rid in enumerate(data[s]['ids'])] for s in data if s != 'train'}
        methods = {}
        for d in DEPTHS:
            methods[f'fixed_{d}'] = {'splits': {s: measures(data[s]['y'], probs[s][d].argmax(1),
                                                          torch.full_like(data[s]['y'], d), probs[s][24].argmax(1))
                                               for s in data if s != 'train'}}
        grid = [.5, .7, .8, .9, .95, .975, .99, .995, 1.01]
        for mode in ['asymmetric', 'agreement', 'learned']:
            if mode == 'learned':
                policies = [{'type': 'learned', 'error_limit': a, 'benefit_limit': b}
                            for a in [-1, .01, .025, .05, .1, .15, .2] for b in [-1, .005, .01, .025, .05, .1]]
            else:
                policies = [{'type': mode, 'safe_threshold': a, 'block_threshold': b, 'agreement': mode == 'agreement'}
                            for a in grid for b in grid]
            chosen, candidates = select_policy(data, probs, policies, scores)
            sweeps[f'{kind}_{mode}'] = candidates
            methods[mode] = {'policy': chosen['policy'], 'splits': {}}
            for s in data:
                if s == 'train':
                    continue
                pred, depth = decide(probs[s], chosen['policy'], scores[s])
                methods[mode]['splits'][s] = measures(data[s]['y'], pred, depth, probs[s][24].argmax(1))
                for i, row in enumerate(predictions[kind][s]):
                    row[mode] = {'prediction': int(pred[i]), 'depth': int(depth[i])}
        review = []
        for t in grid:
            pred, depth = review_policy(probs['calibration'], t)
            m = measures(data['calibration']['y'], pred, depth)
            review.append({'threshold': t, 'metrics': m, 'eligible': m['coverage'] >= .2 and m['covered_accuracy'] >= .95})
        eligible = [r for r in review if r['eligible']]
        threshold = max(eligible, key=lambda r: r['metrics']['coverage'])['threshold'] if eligible else 1.01
        methods['review'] = {'policy': {'threshold': threshold, 'eligible_candidate_found': bool(eligible)}, 'splits': {}}
        sweeps[f'{kind}_review'] = review
        for s in data:
            if s == 'train':
                continue
            pred, depth = review_policy(probs[s], threshold)
            methods['review']['splits'][s] = measures(data[s]['y'], pred, depth, probs[s][24].argmax(1))
            for i, row in enumerate(predictions[kind][s]):
                row['review'] = {'prediction': int(pred[i]), 'depth': 24}
        results[kind] = methods
    write('result.json', results)
    write('predictions.json', predictions)
    write('selection-sweeps.json', sweeps)
    print(json.dumps({kind: {method: item['splits']['evaluation'] for method, item in methods.items()}
                      for kind, methods in results.items()}), flush=True)


if __name__ == '__main__':
    main()
