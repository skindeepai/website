"""One saved request rerun with the obtainable Transformers4.50.3 release."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import hashlib
import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/visual-next'
sys.path.insert(0, str(ROOT / 'experiments/.cache/replay-runtime'))
import numpy as np
import torch
import transformers
from PIL import Image
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from coordinates import Pointer
from screenspot import region_point


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output_path):
    assert transformers.__version__ == '4.50.3', 'Use the cached, published release'
    assert not output_path.exists(), 'Preserve completed replay; choose a new --output path'
    protocol = json.loads((OUT / 'protocol.json').read_text())
    replay_protocol = json.loads((OUT / 'published-runtime-protocol.json').read_text())
    assert replay_protocol['primary_protocol_sha256'] == sha(OUT / 'protocol.json')
    for source, checksum in replay_protocol['source_sha256'].items():
        assert sha(ROOT / source) == checksum
    predictions = json.loads((OUT / 'predictions.json').read_text())
    assert len(predictions) == 50, 'Wait until the primary experiment finishes'
    sample, reference = protocol['samples'][0], predictions[0]
    assert sample['id'] == reference['id']
    assert sample['id'] == replay_protocol['sample_id']
    assert hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest() == replay_protocol['reference_sha256']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoint = Path(snapshot_download(protocol['model'], revision=protocol['model_revision'], local_files_only=True))
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
    path = ROOT / 'experiments/.cache/visual-next' / sample['image_path']
    assert sha(path) == sample['image_sha256']
    prompt = '<|im_start|>system\nYou are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>' + sample['instruction'] + '<|im_end|>\n<|im_start|>assistant<|recipient|>os\npyautogui.click(<|pointer_start|><|pointer_pad|><|pointer_end|>)'
    layers = []
    hooks = [layer.register_forward_hook(lambda module, args, output, i=i: layers.append(i + 1)) for i, layer in enumerate(model.model.layers)]
    with Image.open(path) as im:
        image = im.convert('RGB')
    with torch.inference_mode():
        batch = processor(text=[prompt], images=[image], return_tensors='pt')
        assert int(batch['image_grid_thw'].prod()) // 4 <= 576
        visual = model.visual(batch['pixel_values'].to(model.visual.dtype), grid_thw=batch['image_grid_thw'])
        ids = batch['input_ids']
        embedded = model.model.embed_tokens(ids)
        embedded = embedded.masked_scatter((ids == model.config.image_token_id).unsqueeze(-1).expand_as(embedded), visual)
        positions, _ = model.get_rope_index(ids, batch['image_grid_thw'], None, batch['attention_mask'])
        hidden = model.model(inputs_embeds=embedded, attention_mask=batch['attention_mask'], position_ids=positions, use_cache=False).last_hidden_state
        probabilities = pointer(visual, hidden[ids == model.config.pointer_pad_token_id]).squeeze(0).numpy()
    for hook in hooks:
        hook.remove()
    assert layers == list(range(1, 29))
    _, height, width = batch['image_grid_thw'][0].tolist()
    height //= processor.image_processor.merge_size
    width //= processor.image_processor.merge_size
    assert [width, height] == reference['patch_grid']
    index = int(probabilities.argmax())
    points = {'max_patch': [(index % width + .5) / width, (index // width + .5) / height], 'connected_region': region_point(probabilities, width, height)}
    probability_delta = float(np.max(np.abs(probabilities - np.asarray(reference['probabilities']))))
    coordinate_delta = max(abs(a - b) for key in points for a, b in zip(points[key], reference['points'][key]))
    passed = probability_delta < 1e-4 and coordinate_delta < 1e-5
    package_root = Path(transformers.__file__).parent
    artifact = {'status': 'passed' if passed else 'failed', 'sample_id': sample['id'], 'model': protocol['model'], 'revision': protocol['model_revision'], 'transformers': transformers.__version__, 'torch': torch.__version__, 'threads': 2, 'layers_executed': layers, 'max_probability_delta': probability_delta, 'max_coordinate_delta': coordinate_delta, 'tolerances': {'probabilities': 1e-4, 'coordinates': 1e-5}, 'points': points, 'probabilities': probabilities.tolist(), 'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in [Path(__file__), ROOT / 'experiments/coordinates.py', ROOT / 'experiments/screenspot.py']}, 'runtime_sources': {str(p.relative_to(package_root)).replace('\\', '/'): sha(p) for p in [package_root / 'models/qwen2_vl/modeling_qwen2_vl.py', package_root / 'models/qwen2_vl/image_processing_qwen2_vl.py']}, 'limits': 'One fixed request only. Checks compatibility with an obtainable runtime, not all50 requests or an independent implementation of the pointer architecture.'}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in artifact.items() if k != 'probabilities'}, indent=2))
    assert passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT / 'published-runtime-replay.json')
    main(parser.parse_args().output)
