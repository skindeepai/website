"""Repair inference-mode labels in the sealed, outcome-free int8 diagnostic.

The original failed attempt is preserved. No model, split, head, threshold or
selection rule changes: tensors are cloned outside inference mode before fitting.
"""
import argparse
import chat_next_methods as study
from chat_next_methods import torch, np, ROOT, OUT, CACHE, read, sha, write


def plan():
    return {
        'reason': 'Original quant stage failed before producing outcomes: inference-mode target tensors cannot be saved for cross-entropy backward.',
        'change': 'Clone features and targets outside inference_mode before train-only linear fits; save collected features for reproducibility. No methodological change.',
        'source_sha256': {p: sha(ROOT/p) for p in [
            'experiments/chat_next_quant_repair.py', 'experiments/chat_next_methods.py',
            'results/chat-next-methods/protocol.json', 'results/chat-next-methods/quant-failure.json']}}


def run():
    study.guard('quantization.json')
    assert read('results/chat-next-methods/quant-repair-protocol.json') == plan()
    p, old, rows = study.load(); rt = study.Runtime(p)
    torch.backends.quantized.engine = 'x86'
    quant = torch.ao.quantization.quantize_dynamic(study.copy.deepcopy(rt.model), {torch.nn.Linear}, dtype=torch.qint8).eval()
    assert sum(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in quant.modules()) == 168
    qx = {}; fx = {}; ys = {}
    with torch.inference_mode():
        for split, ids in p['quantization_splits'].items():
            fx[split] = study.cached(old, ids)[24]
            ys[split] = torch.tensor([rows[rid]['label'] for rid in ids]); values = []
            for i, rid in enumerate(ids):
                values.append(rt.features(rows[rid], quant)[24][0])
                if (i+1) % 32 == 0: print(f'Quant repair {split}:{i+1}/{len(ids)}', flush=True)
            qx[split] = torch.stack(values)
    qx = {k: v.clone() for k, v in qx.items()}
    fx = {k: v.clone() for k, v in fx.items()}
    ys = {k: v.clone() for k, v in ys.items()}
    torch.save({'quant': qx, 'float': fx, 'labels': ys}, CACHE/'quant-repair-features.pt')
    pred = {}; weights = {}
    for kind, arrays in [('matched_float', fx), ('refit_quant', qx)]:
        torch.manual_seed(193); mean = arrays['train'].mean(0); std = arrays['train'].std(0).clamp_min(.05)
        head = torch.nn.Linear(896, 2); opt = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.1)
        for _ in range(200):
            opt.zero_grad(); torch.nn.functional.cross_entropy(head((arrays['train']-mean)/std), ys['train']).backward(); opt.step()
        with torch.inference_mode(): pred[kind] = head((arrays['development']-mean)/std).argmax(1)
        for k, v in [('mean', mean), ('std', std), ('weight', head.weight.detach()), ('bias', head.bias.detach())]: weights[f'{kind}_{k}'] = v.numpy()
    with torch.inference_mode():
        full = study.original(rt.heads, 24, fx['development']).argmax(1)
        qlog = study.original(rt.heads, 24, qx['development']); trainqlog = study.original(rt.heads, 24, qx['train'])
        margin = trainqlog[:, 1]-trainqlog[:, 0]
        thresholds = [-float('inf')]+sorted(set(float(v) for v in margin))+[float('inf')]
        threshold = max(thresholds, key=lambda t: int(((margin >= t).long() == ys['train']).sum()))
        pred['original_float'] = full; pred['unchanged_head_quant'] = qlog.argmax(1)
        pred['threshold_quant'] = ((qlog[:, 1]-qlog[:, 0]) >= threshold).long()
        drift = qx['development']-fx['development']; cos = torch.nn.functional.cosine_similarity(qx['development'], fx['development'], dim=1)
        metrics = {k: study.measures(ys['development'], v, full) for k, v in pred.items()}
    np.savez_compressed(OUT/'quant-refits.npz', **weights)
    records = [{'id': rid, 'label': int(ys['development'][i]), 'predictions': {k: int(v[i]) for k, v in pred.items()},
                'hidden_cosine': float(cos[i]), 'hidden_rms_delta': float(drift[i].square().mean().sqrt()),
                'float_logits': study.original(rt.heads, 24, fx['development'][i:i+1])[0].tolist(), 'quant_logits': qlog[i].tolist()}
               for i, rid in enumerate(p['quantization_splits']['development'])]
    write('quantization-records.json', records)
    write('quantization.json', {'metrics': metrics, 'protocol_sha256': sha(OUT/'protocol.json'),
        'repair_protocol_sha256': sha(OUT/'quant-repair-protocol.json'), 'feature_cache_sha256': sha(CACHE/'quant-repair-features.pt'),
        'weights_sha256': sha(OUT/'quant-refits.npz'), 'threshold': threshold if np.isfinite(threshold) else str(threshold),
        'mean_hidden_cosine': float(cos.mean()), 'mean_hidden_rms_delta': float(drift.square().mean().sqrt()),
        'scope': '128train/64olddevelopment; no fresh test or timing. Floatcachedbatch8 vs quantactualbatch1. Readout correction is not proof all backbone information is preserved.'})
    print(study.json.dumps(metrics), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if args.prepare:
        study.guard('quant-repair-protocol.json'); write('quant-repair-protocol.json', plan()); print('Repair sealed; no new outcomes.')
    else:
        try: run()
        except Exception as error:
            write('quant-repair-failure.json', {'error': str(error), 'type': type(error).__name__}); raise
