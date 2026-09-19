"""Reconstruct the adapter study's initial readouts; no transformer inference."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chat-smoke-adaptation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    target = OUT / 'initial-frozen-baseline.json'
    if target.exists():
        raise RuntimeError('Preserve completed baseline; use another output for a changed recipe.')
    protocol = json.loads((OUT / 'full-protocol.json').read_text())
    old = json.loads((ROOT / 'results/chat600/protocol.json').read_text())
    cache_id = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    ids = protocol['subset_ids']
    original_location = {rid: (split, i) for split, rows in old['splits'].items()
                         for i, rid in enumerate(rows)}
    labels = {}
    for source in ['train', 'test']:
        path = ROOT / f'experiments/.cache/toxicchat/toxic-chat_annotation_{source}.csv'
        assert digest(path) == old['data_sha256'][source]
        with path.open(encoding='utf-8', newline='') as handle:
            for i, row in enumerate(csv.DictReader(handle)):
                labels[f'{source}:{i}'] = int(row['toxicity'])
    cache_paths = {s: ROOT / f'experiments/.cache/toxicchat/{cache_id}-{s}.pt'
                   for s in old['splits']}
    cache = {s: torch.load(path, weights_only=True) for s, path in cache_paths.items()}
    y = {s: torch.tensor([labels[rid] for rid in rows]) for s, rows in ids.items()}
    weights = 1 / torch.bincount(y['train']).float()
    weights /= weights.mean()
    torch.manual_seed(73)
    arrays, summaries, predictions = {}, {s: {} for s in ids}, {s: [] for s in ids}
    for depth in [6, 12, 18, 24]:
        x = {s: torch.stack([cache[original_location[rid][0]][depth][original_location[rid][1]]
                             for rid in rows]) for s, rows in ids.items()}
        mean, std = x['train'].mean(0), x['train'].std(0).clamp_min(.05)
        model = torch.nn.Linear(896, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.1)
        for _ in range(200):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model((x['train'] - mean) / std),
                                                    y['train'], weight=weights)
            loss.backward()
            optimizer.step()
        arrays.update({f'{depth}_mean': mean.numpy(), f'{depth}_std': std.numpy(),
                       f'{depth}_weight': model.weight.detach().numpy(),
                       f'{depth}_bias': model.bias.detach().numpy()})
        with torch.inference_mode():
            for split in ids:
                logits = model((x[split] - mean) / std)
                probs = logits.softmax(1)
                pred = logits.argmax(1)
                target_y = y[split]
                tp = int(((pred == 1) & (target_y == 1)).sum())
                fn = int(((pred == 0) & (target_y == 1)).sum())
                fp = int(((pred == 1) & (target_y == 0)).sum())
                tn = int(((pred == 0) & (target_y == 0)).sum())
                summaries[split][str(depth)] = dict(n=len(pred), correct=tp+tn,
                    missed_toxic=fn, false_block=fp, toxic_recall=tp/(tp+fn),
                    specificity=tn/(tn+fp), balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)))
                for i, rid in enumerate(ids[split]):
                    if depth == 6:
                        predictions[split].append(dict(id=rid, label=int(target_y[i]), heads={}))
                    predictions[split][i]['heads'][str(depth)] = dict(prediction=int(pred[i]),
                        prob_block=float(probs[i, 1]))
    weight_path = OUT / 'initial-frozen-heads.npz'
    np.savez_compressed(weight_path, **arrays)
    result = dict(scope='Reconstructed matched initial readouts using original cached frozen Qwen features; exploratory reused data, no new transformer inference.',
        recipe='Seed73, train384, per-depth train-only mean/std clamp.05, 896-to-2 linear, 200AdamW steps lr.01 weight_decay.1, inverse-frequency class weights normalized to mean1. No temperature scaling.',
        runtime=dict(torch=torch.__version__, numpy=np.__version__, threads=1),
        script_sha256=digest(Path(__file__)), parent_protocol_sha256=digest(OUT/'full-protocol.json'),
        cached_feature_sha256={s: digest(path) for s, path in cache_paths.items()},
        portable_weights_sha256=digest(weight_path), metrics=summaries, predictions=predictions,
        limitation='Adapter initialization runs at4threads; this independent reconstruction uses1thread. Check initial_tune parity before attributing differences. It adds evaluation reporting after training began, without changing any trained candidate or selection.')
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(evaluation=summaries['evaluation'], initial_tune=summaries['tune'])))


if __name__ == '__main__':
    main()
