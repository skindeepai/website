"""Fresh OSWorld-G localisation/refusal stress test; no live actions or decoding.

Prepare seals all sample IDs, gates and crop decisions before model inference.
The old ScreenSpot sample is development only; no OSWorld labels tune a gate.
"""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/visual-next'
CACHE = ROOT / 'experiments/.cache/visual-next'
REV = 'daa6bd8e0e629f0917ad2984df930bf0bd967540'
BASE = 'https://raw.githubusercontent.com/xlang-ai/OSWorld-G/' + REV + '/'
MODEL = 'microsoft/GUI-Actor-2B-Qwen2-VL'
MODEL_REV = '8f87b366d004425a9823502553e2097c71116ece'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def download(url, path):
    if not path.exists():
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                path.write_bytes(data)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)


def gate_cutoff(rows, relative):
    def score(r):
        return r['peak_probability'] * (r['patch_grid'][0] * r['patch_grid'][1] if relative else 1)
    values = sorted({score(r) for r in rows})
    candidates = [0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [577 if relative else 2]
    permitted = []
    for cutoff in candidates:
        accepted = [r for r in rows if score(r) >= cutoff]
        if all(r['hits']['connected_region'] for r in accepted):
            permitted.append((len(accepted), cutoff))
    coverage, cutoff = max(permitted, key=lambda x: (x[0], -x[1]))
    return {'cutoff': cutoff, 'development_accepted': coverage, 'development_wrong': 0}


def prepare():
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Do not overwrite a sealed experiment'
    for file in ['benchmark/OSWorld-G_refined.json', 'LICENSE']:
        download(BASE + file, CACHE / Path(file).name)
    rows = read(CACHE / 'OSWorld-G_refined.json')
    assert len(rows) == 564
    selected = []
    used_images = set()
    # Unique screenshots across both categories; no outcome or image inspection.
    for absent in [True, False]:
        candidates = sorted([r for r in rows if (r['box_type'] == 'refusal') == absent], key=lambda r: r['id'])
        random.Random(20260920 + int(absent)).shuffle(candidates)
        chosen = []
        for row in candidates:
            if row['image_path'] not in used_images:
                chosen.append(row)
                used_images.add(row['image_path'])
            if len(chosen) == 25:
                break
        assert len(chosen) == 25
        selected.extend(chosen)
    random.Random(20260922).shuffle(selected)
    crop_ids = [r['id'] for r in selected if r['box_type'] != 'refusal'][:10]
    for i, row in enumerate(selected):
        path = CACHE / row['image_path']
        download(BASE + 'benchmark/images/' + row['image_path'], path)
        with Image.open(path) as im:
            assert list(im.size) == row['image_size']
        row['image_sha256'] = sha(path)
        print('Prepared', i + 1, '/ 50', flush=True)
    old = read(ROOT / 'results/screenspot/predictions.json')
    protocol = {
        'dataset': 'OSWorld-G refined instructions', 'revision': REV,
        'dataset_url': 'https://github.com/xlang-ai/OSWorld-G/tree/' + REV + '/benchmark',
        'annotations_sha256': sha(CACHE / 'OSWorld-G_refined.json'),
        'license': 'Apache-2.0 repository; screenshots depict third-party applications',
        'model': MODEL, 'model_revision': MODEL_REV,
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [Path(__file__), ROOT / 'experiments/screenspot.py', ROOT / 'experiments/coordinates.py', ROOT / 'results/screenspot/predictions.json']},
        'selection': '25 author-labelled refusal and25 present targets; seeded shuffle; unique image across categories; then independently shuffled. All50 are new to this project. This is a balanced stress test, not natural prevalence.',
        'samples': selected, 'crop_ids': crop_ids,
        'methods': ['max_patch', 'connected_region', 'peak_gate', 'relative_peak_gate'],
        'gates': {'peak_gate': gate_cutoff(old, False), 'relative_peak_gate': gate_cutoff(old, True)},
        'gate_training': 'All30 old ScreenSpot development outputs; greatest coverage with zero accepted wrong connected-region points. No absent targets in development. No OSWorld-G target enters the decision rule.',
        'domain_holdout': 'OSWorld-G desktop benchmark is outside ScreenSpot gate development. No claim that its applications or benchmark were unseen in pretrained model training.',
        'runtime': {'threads': 2, 'dtype': 'float32', 'max_visual_tokens': 576, 'min_visual_tokens': 256, 'language_layers': 28, 'generated_text_tokens': 0},
        'crop': 'Preselected10 present requests only: run a second pass on half-width/half-height crop centered on predicted full-screen connected-region point, clamped to image. Map result back. Gold coordinates never choose crop. Spatial instructions retained verbatim; lost context may hurt. Always pay both passes.',
        'output': 'CLICK(x,y) or UNCERTAIN. UNCERTAIN is not a proven semantic NOT_FOUND classification. Count refused absent requests separately from withheld present requests.',
        'timing': 'Per-request diagnostic CPU wall time, image processing through pointer output; model loading excluded. Other research jobs may run. Not a speed comparison or isolated hardware benchmark.'
    }
    write(OUT / 'protocol.json', protocol)
    write(OUT / 'data-manifest.json', {'dataset': protocol['dataset'], 'revision': REV, 'samples': selected})
    (OUT / 'LICENSE.OSWorld-G').write_bytes((CACHE / 'LICENSE').read_bytes())


def hit(point, row):
    if row['box_type'] == 'refusal':
        return False
    x, y = point[0] * row['image_size'][0], point[1] * row['image_size'][1]
    coords = row['box_coordinates']
    if row['box_type'] == 'bbox':
        bx, by, width, height = coords
        return bx <= x <= bx + width and by <= y <= by + height
    assert row['box_type'] == 'polygon'
    pairs = list(zip(coords[::2], coords[1::2]))
    inside = False
    for (ax, ay), (bx, by) in zip(pairs, pairs[1:] + pairs[:1]):
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-8 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def run():
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from coordinates import Pointer
    from screenspot import region_point
    protocol = read(OUT / 'protocol.json')
    for source, checksum in protocol['source_sha256'].items():
        assert sha(ROOT / source) == checksum, 'Sealed source changed: ' + source
    assert not (OUT / 'result.json').exists(), 'Preserve completed experiment'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoint = Path(snapshot_download(MODEL, revision=MODEL_REV, local_files_only=True))
    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.image_processor.size = {'shortest_edge': 256 * 28 * 28, 'longest_edge': 576 * 28 * 28}
    model = Qwen2VLForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32, attn_implementation='eager').eval()
    pointer = Pointer(model.config.hidden_size).eval()
    state = {}
    for shard in checkpoint.glob('*.safetensors'):
        with safe_open(shard, framework='pt') as reader:
            for key in reader.keys():
                if key.startswith('multi_patch_pointer_head.'):
                    state[key.removeprefix('multi_patch_pointer_head.')] = reader.get_tensor(key).float()
    pointer.load_state_dict(state, strict=True)
    layers = []
    hooks = [layer.register_forward_hook(lambda module, args, output, i=i: layers.append(i + 1)) for i, layer in enumerate(model.model.layers)]

    def forward(image, instruction):
        started = time.perf_counter()
        prompt = '<|im_start|>system\nYou are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>' + instruction + '<|im_end|>\n<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)'
        batch = processor(text=[prompt], images=[image], return_tensors='pt')
        assert int(batch['image_grid_thw'].prod()) // 4 <= 576
        visual = model.visual(batch['pixel_values'].to(model.visual.dtype), grid_thw=batch['image_grid_thw'])
        ids = batch['input_ids']
        embedded = model.model.embed_tokens(ids)
        embedded = embedded.masked_scatter((ids == model.config.image_token_id).unsqueeze(-1).expand_as(embedded), visual)
        positions, _ = model.get_rope_index(ids, batch['image_grid_thw'], None, batch['attention_mask'])
        layers.clear()
        hidden = model.model(inputs_embeds=embedded, attention_mask=batch['attention_mask'], position_ids=positions, use_cache=False).last_hidden_state
        assert layers == list(range(1, 29))
        prob = pointer(visual, hidden[ids == model.config.pointer_pad_token_id]).squeeze(0).numpy()
        assert np.isfinite(prob).all() and abs(float(prob.sum()) - 1) < 1e-5
        _, height, width = batch['image_grid_thw'][0].tolist()
        height //= processor.image_processor.merge_size
        width //= processor.image_processor.merge_size
        argmax = int(prob.argmax())
        return {'points': {'max_patch': [(argmax % width + .5) / width, (argmax // width + .5) / height], 'connected_region': region_point(prob, width, height)}, 'peak_probability': float(prob.max()), 'patch_grid': [width, height], 'probabilities': prob.tolist(), 'layers_executed': list(layers), 'elapsed_ms': (time.perf_counter() - started) * 1000}

    traces = read(OUT / 'predictions.json') if (OUT / 'predictions.json').exists() else []
    with torch.inference_mode():
        for index, row in enumerate(protocol['samples']):
            if index < len(traces):
                assert traces[index]['id'] == row['id']
                continue
            path = CACHE / row['image_path']
            assert sha(path) == row['image_sha256']
            with Image.open(path) as im:
                image = im.convert('RGB')
            result = forward(image, row['instruction'])
            result.update({'id': row['id'], 'target_present': row['box_type'] != 'refusal', 'hits': {method: hit(point, row) for method, point in result['points'].items()}})
            if row['id'] in protocol['crop_ids']:
                px, py = result['points']['connected_region']
                w, h = image.size
                cw, ch = w // 2, h // 2
                left = max(0, min(w - cw, round(px * w - cw / 2)))
                top = max(0, min(h - ch, round(py * h - ch / 2)))
                crop = forward(image.crop((left, top, left + cw, top + ch)), row['instruction'])
                crop['crop_xyxy'] = [left, top, left + cw, top + ch]
                crop['global_point'] = [(left + crop['points']['connected_region'][0] * cw) / w, (top + crop['points']['connected_region'][1] * ch) / h]
                crop['hit'] = hit(crop['global_point'], row)
                result['crop'] = crop
            traces.append(result)
            write(OUT / 'predictions.json', traces)
            print(index + 1, '/ 50', row['id'], 'present=', result['target_present'], result['hits'], flush=True)
    for hook in hooks:
        hook.remove()
    summarize(protocol, traces)


def summarize(protocol, traces):
    assert len(traces) == 50
    methods = {}
    decisions = []
    for method in protocol['methods']:
        counts = dict(correct_clicks=0, wrong_present_clicks=0, absent_clicks=0, withheld_present=0, withheld_correct_present=0, refused_absent=0)
        for row in traces:
            point_method = method if method in ['max_patch', 'connected_region'] else 'connected_region'
            accepted = True
            if method in protocol['gates']:
                score = row['peak_probability'] * (row['patch_grid'][0] * row['patch_grid'][1] if method == 'relative_peak_gate' else 1)
                accepted = score >= protocol['gates'][method]['cutoff']
            correct = row['hits'][point_method]
            if accepted:
                counts['correct_clicks' if correct else 'wrong_present_clicks' if row['target_present'] else 'absent_clicks'] += 1
            else:
                counts['withheld_present' if row['target_present'] else 'refused_absent'] += 1
                counts['withheld_correct_present'] += int(row['target_present'] and correct)
            decisions.append({'id': row['id'], 'method': method, 'output': 'CLICK' if accepted else 'UNCERTAIN', 'point': row['points'][point_method] if accepted else None, 'target_present': row['target_present'], 'point_correct': correct})
        counts['accepted'] = counts['correct_clicks'] + counts['wrong_present_clicks'] + counts['absent_clicks']
        methods[method] = counts
    crop = [r for r in traces if 'crop' in r]
    summary = {'protocol_sha256': sha(OUT / 'protocol.json'), 'samples': 50, 'present': 25, 'absent': 25, 'methods': methods,
        'crop_study': {'n': len(crop), 'before_correct': sum(r['hits']['connected_region'] for r in crop), 'after_correct': sum(r['crop']['hit'] for r in crop), 'corrected': sum(not r['hits']['connected_region'] and r['crop']['hit'] for r in crop), 'lost': sum(r['hits']['connected_region'] and not r['crop']['hit'] for r in crop), 'full_screen_total_ms': sum(r['elapsed_ms'] for r in crop), 'additional_crop_ms': sum(r['crop']['elapsed_ms'] for r in crop)},
        'runtime': protocol['runtime'], 'total_full_screen_ms': sum(r['elapsed_ms'] for r in traces),
        'limitations': ['Balanced50-case public benchmark stress test, not production prevalence or full benchmark.', 'Gate fitted on30 old present-only ScreenSpot requests; this tests domain transfer, not trained absence detection.', 'UNCERTAIN is an abstention, not a semantic NOT_FOUND claim; review is not a completed action.', 'All28 language layers and the vision encoder run; zero generated tokens, zero early-exit savings.', 'Model pretraining overlap with public benchmarks cannot be excluded.', 'Refined instructions and576-patch budget; not an official leaderboard replication.', 'Crop costs a second complete model pass on10 prespecified present cases and may discard instruction context.']}
    write(OUT / 'decisions.json', decisions)
    write(OUT / 'result.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    prepare() if args.prepare else run()
