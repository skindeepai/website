"""Independent arithmetic/provenance review; no model inference or raw text output."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/quant-localize'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = read(OUT / 'protocol.json')
    result = read(OUT / 'result.json')
    rows = read(OUT / 'records.json')
    parent = read(ROOT / 'results/chat-next-methods/protocol.json')
    assert sha(ROOT / 'experiments/quant_localize.py') == protocol['source_sha256']
    assert sha(ROOT / 'results/chat-next-methods/protocol.json') == protocol['parent_sha256']
    dependencies = ['experiments/chat_next_methods.py', 'results/chat600/protocol.json', 'results/chat600/heads.npz']
    for name in dependencies:
        assert sha(ROOT / name) == parent['source_sha256'][name]
    expected_counts = {'all_default': 168, 'attention_only': 96, 'mlp_only': 72,
                       'first12_only': 84, 'last12_only': 84, 'all_per_channel': 168}
    checks = {}
    for method, count in expected_counts.items():
        records = [r for r in rows if r['method'] == method]
        assert [r['id'] for r in records] == protocol['ids']
        assert len(records) == 32
        summary = result['methods'][method]
        assert summary['quantized_linear_count'] == count == len(summary['quantized_names'])
        measures = dict(n=len(records),
                        correct=sum(r['prediction'] == r['label'] for r in records),
                        missed_toxic=sum(r['label'] == 1 and r['prediction'] == 0 for r in records),
                        false_block=sum(r['label'] == 0 and r['prediction'] == 1 for r in records),
                        toxic=sum(r['label'] == 1 for r in records),
                        added_errors=sum(r['float_prediction'] == r['label'] and r['prediction'] != r['label'] for r in records),
                        corrected_errors=sum(r['float_prediction'] != r['label'] and r['prediction'] == r['label'] for r in records),
                        additional_missed_toxic=sum(r['label'] == 1 and r['float_prediction'] == 1 and r['prediction'] == 0 for r in records))
        assert measures == summary['metrics']
        assert sum(r['float_prediction'] == r['label'] for r in records) == result['methods']['float']['correct']
        for record in records:
            assert record['prediction'] == max(range(2), key=lambda i: record['logits'][i])
            assert record['float_prediction'] == max(range(2), key=lambda i: record['float_logits'][i])
            assert set(record['layers']) == {str(i) for i in range(1, 26)}
        for layer, values in summary['layers'].items():
            for metric, reported in values.items():
                assert math.isclose(sum(r['layers'][layer][metric] for r in records) / 32, reported, abs_tol=1e-12)
        checks[method] = {'metrics_recomputed': True, 'state_means_recomputed': 25,
                          'changed_decisions': sum(r['float_prediction'] != r['prediction'] for r in records)}
    assert len(rows) == 192
    assert checks['attention_only']['changed_decisions'] == 4
    review = {'reviewer': 'Internal adversarial search agent; independent arithmetic and source inspection',
              'source_sha256': sha(Path(__file__)), 'checks': checks, 'prediction_records': len(rows),
              'artifact_sha256': {name: sha(OUT / name) for name in ['protocol.json', 'result.json', 'records.json']},
              'inherited_dependencies_verified': dependencies,
              'findings': ['Attention-only matches 28/32 correct but changes four decisions: two new errors and two corrections. One new error is a toxic miss.',
                           'State 25 is final normalized output, not a transformer layer. Captured vectors describe only the final input token.',
                           'All paths execute all 24 layers. These development-only outcomes contain no measured speed or storage benefit.',
                           'Selective conversion is a follow-up candidate; matching total accuracy is not preserved decisions or a no-quality-loss guarantee.'],
              'limits': 'Internal reproducible record/provenance audit, not an independent model rerun or external validation.'}
    (OUT / 'review.json').write_text(json.dumps(review, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('Quantization review passed: 192 records, six variants, 25 state means each.')


if __name__ == '__main__':
    main()
