"""Recompute visual-next evidence from retained numerical outputs, without models."""
import hashlib
import json
import math
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/visual-next'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def point_inside(point, sample):
    if sample['box_type'] == 'refusal':
        return False
    x, y = [point[i] * sample['image_size'][i] for i in range(2)]
    c = sample['box_coordinates']
    if sample['box_type'] == 'bbox':
        return c[0] <= x <= c[0] + c[2] and c[1] <= y <= c[1] + c[3]
    pairs = list(zip(c[::2], c[1::2]))
    angle = 0.0
    for a, b in zip(pairs, pairs[1:] + pairs[:1]):
        ax, ay, bx, by = a[0] - x, a[1] - y, b[0] - x, b[1] - y
        cross = ax * by - ay * bx
        dot = ax * bx + ay * by
        if abs(cross) < 1e-8 and dot <= 0:
            return True
        angle += math.atan2(cross, dot)
    return abs(angle) > math.pi


def region(probabilities, width, height):
    pending = {i for i, p in enumerate(probabilities) if p > .3 * max(probabilities)}
    components = []
    while pending:
        component = {min(pending)}
        pending -= component
        while True:
            additions = {b for a in component for b in [a - width, a + width, a - 1, a + 1] if b in pending and abs(a % width - b % width) + abs(a // width - b // width) == 1}
            if not additions:
                break
            component |= additions
            pending -= additions
        components.append(sorted(component))
    selected = max(components, key=lambda c: sum(probabilities[i] for i in c) / len(c))
    mass = sum(probabilities[i] for i in selected)
    return [sum((i % width + .5) / width * probabilities[i] for i in selected) / mass, sum((i // width + .5) / height * probabilities[i] for i in selected) / mass]


def verify_output(row):
    probs = row['probabilities']
    width, height = row['patch_grid']
    assert len(probs) == width * height <= 576
    assert abs(sum(probs) - 1) < 1e-5 and all(math.isfinite(x) and 0 <= x <= 1 for x in probs)
    peak = max(range(len(probs)), key=lambda i: probs[i])
    assert row['points']['max_patch'] == [(peak % width + .5) / width, (peak // width + .5) / height]
    assert max(abs(x - y) for x, y in zip(region(probs, width, height), row['points']['connected_region'])) < 1e-5
    assert row['peak_probability'] == max(probs)
    assert row['layers_executed'] == list(range(1, 29))
    assert row['elapsed_ms'] > 0


def main():
    protocol = read(OUT / 'protocol.json')
    predictions = read(OUT / 'predictions.json')
    result = read(OUT / 'result.json')
    decisions = read(OUT / 'decisions.json')
    samples = protocol['samples']
    assert len(predictions) == len(samples) == 50
    assert len({s['image_path'] for s in samples}) == 50
    assert sum(s['box_type'] == 'refusal' for s in samples) == 25
    assert result['protocol_sha256'] == sha(OUT / 'protocol.json')
    development = read(ROOT / 'results/screenspot/predictions.json')
    for method in ['peak_gate', 'relative_peak_gate']:
        scores = [r['peak_probability'] * (r['patch_grid'][0] * r['patch_grid'][1] if method == 'relative_peak_gate' else 1) for r in development]
        values = sorted(set(scores))
        cutoffs = [0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [577 if method == 'relative_peak_gate' else 2]
        permitted = []
        for cutoff in cutoffs:
            chosen = [r for score, r in zip(scores, development) if score >= cutoff]
            wrong = sum(not (r['bbox'][0] <= r['points']['connected_region'][0] <= r['bbox'][2] and r['bbox'][1] <= r['points']['connected_region'][1] <= r['bbox'][3]) for r in chosen)
            if wrong == 0:
                permitted.append((len(chosen), cutoff))
        coverage, cutoff = max(permitted, key=lambda x: (x[0], -x[1]))
        assert protocol['gates'][method] == {'cutoff': cutoff, 'development_accepted': coverage, 'development_wrong': 0}
    for source, checksum in protocol['source_sha256'].items():
        path = OUT / 'initial-runner.py' if source == 'experiments/visual_next.py' and (OUT / 'runtime-amendment.json').exists() else ROOT / source
        assert sha(path) == checksum
    if (OUT / 'runtime-amendment.json').exists():
        amendment = read(OUT / 'runtime-amendment.json')
        assert amendment['protocol_sha256'] == sha(OUT / 'protocol.json')
        source = OUT / 'four-thread-runner.py' if (OUT / 'runtime-amendment-12threads.json').exists() else ROOT / 'experiments/visual_next.py'
        assert amendment['amended_source_sha256'] == sha(source)
        assert sha(OUT / 'initial-runner.py') == protocol['source_sha256']['experiments/visual_next.py']
    if (OUT / 'runtime-amendment-12threads.json').exists():
        twelve = read(OUT / 'runtime-amendment-12threads.json')
        assert twelve['protocol_sha256'] == sha(OUT / 'protocol.json')
        assert twelve['amended_source_sha256'] == sha(ROOT / 'experiments/visual_next.py')
        assert twelve['previous_runtime_amendment_sha256'] == sha(OUT / 'runtime-amendment.json')
        assert twelve['previous_amended_source_sha256'] == sha(OUT / 'four-thread-runner.py')
        assert twelve['completed_ids_before_amendment'] == [r['id'] for r in predictions[:len(twelve['completed_ids_before_amendment'])]]
    for parity_path in sorted(OUT.glob('thread-parity*.json')):
        parity = read(parity_path)
        assert parity['status'] == 'passed'
        verify_output(parity['repeated_inference'])
        repeated = parity['repeated_inference']
        probability_delta = max(abs(a - b) for a, b in zip(predictions[0]['probabilities'], repeated['probabilities']))
        coordinate_delta = max(abs(a - b) for key in repeated['points'] for a, b in zip(predictions[0]['points'][key], repeated['points'][key]))
        assert probability_delta == parity['max_probability_delta'] < 1e-4
        assert coordinate_delta == parity['max_coordinate_delta'] < 1e-5
    for p, sample in zip(predictions, samples):
        assert p['id'] == sample['id']
        assert p['target_present'] == (sample['box_type'] != 'refusal')
        assert sha(ROOT / 'experiments/.cache/visual-next' / sample['image_path']) == sample['image_sha256']
        verify_output(p)
        for method, point in p['points'].items():
            assert p['hits'][method] == point_inside(point, sample)
        if 'crop' in p:
            assert p['id'] in protocol['crop_ids']
            crop = p['crop']
            verify_output(crop)
            left, top, right, bottom = crop['crop_xyxy']
            w, h = sample['image_size']
            assert right - left == w // 2 and bottom - top == h // 2
            assert left == max(0, min(w - w // 2, round(p['points']['connected_region'][0] * w - w // 2 / 2)))
            assert top == max(0, min(h - h // 2, round(p['points']['connected_region'][1] * h - h // 2 / 2)))
            mapped = [(left + crop['points']['connected_region'][0] * (right - left)) / w, (top + crop['points']['connected_region'][1] * (bottom - top)) / h]
            assert mapped == crop['global_point']
            assert crop['hit'] == point_inside(mapped, sample)
    for method, summary in result['methods'].items():
        rows = [r for r in decisions if r['method'] == method]
        assert len(rows) == 50
        for decision, p in zip(rows, predictions):
            assert decision['id'] == p['id']
            assert decision['target_present'] == p['target_present']
            location = method if method in ['max_patch', 'connected_region'] else 'connected_region'
            assert decision['point_correct'] == p['hits'][location]
            accepted = True
            if method in protocol['gates']:
                score = p['peak_probability'] * (len(p['probabilities']) if method == 'relative_peak_gate' else 1)
                accepted = score >= protocol['gates'][method]['cutoff']
            assert (decision['output'] == 'CLICK') == accepted
            assert decision['point'] == (p['points'][location] if accepted else None)
        click = [r for r in rows if r['output'] == 'CLICK']
        assert summary['accepted'] == len(click)
        assert summary['correct_clicks'] == sum(r['point_correct'] for r in click)
        assert summary['wrong_present_clicks'] == sum(r['target_present'] and not r['point_correct'] for r in click)
        assert summary['absent_clicks'] == sum(not r['target_present'] for r in click)
        assert summary['refused_absent'] == sum(not r['target_present'] and r['output'] == 'UNCERTAIN' for r in rows)
        assert summary['withheld_present'] == sum(r['target_present'] and r['output'] == 'UNCERTAIN' for r in rows)
        assert summary['withheld_correct_present'] == sum(r['target_present'] and r['point_correct'] and r['output'] == 'UNCERTAIN' for r in rows)
    crops = [p for p in predictions if 'crop' in p]
    assert len(crops) == result['crop_study']['n'] == 10
    assert result['crop_study']['before_correct'] == sum(p['hits']['connected_region'] for p in crops)
    assert result['crop_study']['after_correct'] == sum(p['crop']['hit'] for p in crops)
    if (OUT / 'published-runtime-replay.json').exists():
        replay = read(OUT / 'published-runtime-replay.json')
        replay_protocol = read(OUT / 'published-runtime-protocol.json')
        assert replay['status'] == 'passed' and replay['transformers'] == '4.50.3'
        assert replay['sample_id'] == samples[0]['id']
        assert replay['layers_executed'] == list(range(1, 29))
        delta = max(abs(a - b) for a, b in zip(replay['probabilities'], predictions[0]['probabilities']))
        assert delta == replay['max_probability_delta'] < 1e-4
        delta = max(abs(a - b) for key in replay['points'] for a, b in zip(replay['points'][key], predictions[0]['points'][key]))
        assert delta == replay['max_coordinate_delta'] < 1e-5
        wheel = ROOT / 'experiments/.cache/replay-wheels/transformers-4.50.3-py3-none-any.whl'
        assert sha(wheel) == replay_protocol['runtime_wheel_sha256']
        with zipfile.ZipFile(wheel) as archive:
            for file, checksum in replay['runtime_sources'].items():
                assert hashlib.sha256(archive.read('transformers/' + file)).hexdigest() == checksum
        for file, checksum in replay['source_sha256'].items():
            assert sha(ROOT / file) == checksum
    audit = {'status': 'passed', 'samples': 50, 'unique_images': 50, 'absent_author_labels': 25, 'crop_inferences': 10, 'method_decisions': len(decisions), 'checks': ['image and source hashes', 'probability distributions', 'independent region readout', 'exact layers1through28', 'rectangle and independent polygon hits', 'gold-independent crop geometry and mapping', 'gate decisions and all separate error/coverage totals', 'runtime amendment and parity artifact'], 'source_sha256': sha(Path(__file__)), 'artifact_sha256': {p.name: sha(p) for p in [OUT / 'protocol.json', OUT / 'predictions.json', OUT / 'decisions.json', OUT / 'result.json']}}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
