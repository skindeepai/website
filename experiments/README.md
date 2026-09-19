# Reproduce the lab work

These are bounded local pilots, not full executions of 29 protocols. Public JSON contains configurations, predictions and limitations. No live GUI clicks or external data submission occur.

## Preference checks

Python 3.13.2, NumPy 2.2.4 and SciPy 1.15.2 were used. Node tests the actual browser math against shared fixtures.

```sh
python -m pip install -r experiments/requirements.txt
python experiments/run_synthetic.py
python experiments/preference.py
python experiments/check_reference.py
```

The solver is compared with independent SciPy SLSQP. Five seeds cover linear generalization, mixed sampling, disconnected preferences, misspecified optimization, drift and a known analytic constraint. Protocols are only partially covered. The synthetic sampler is 2/2/2; the browser uses 5/4/3.

`check_reference.py` requires Torch and checks the older neural example's small batches, single-item remainder and finite termination. It does not validate real generators.

## Model environment

Measured installation: Torch **2.6.0+cpu**, Transformers **4.50.0.dev0**, Pillow **12.3.0**, huggingface-hub **0.29.3**, safetensors **0.5.3**, Windows, AMD CPU, 32 logical processors. Eight Torch threads and one inter-op thread per model run. The installed Torch build has no CUDA/NPU backend.

Use an isolated environment; do not replace an application's dependencies. The exact development Transformers build may not be available from PyPI. A compatible release is a new run: report API/numerical changes rather than equating it to these results. Runtime versions and model revisions are recorded in JSON.

Download the pinned checkpoints without executing remote model code:

```python
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',
    revision='7ae557604adf67be50417f59c2c2f167def9a775',
    allow_patterns=['*.json','*.safetensors','*.txt'], max_workers=2)
snapshot_download('microsoft/GUI-Actor-2B-Qwen2-VL',
    revision='8f87b366d004425a9823502553e2097c71116ece',
    allow_patterns=['*.json','*.safetensors','*.txt'], max_workers=2)
```

Check model cards/licenses before reuse. Weights stay in the local Hugging Face cache; trained `.pt` heads are excluded from git. Experiment scripts require local checkpoints and do not fetch new revisions.

```sh
python experiments/qwen_decisions.py --threads 8
python experiments/coordinates.py --threads 8
```

Run model pilots sequentially. Cap aggregate work at half the machine's logical processors, counting all builds/pools. The coordinate pilot can take many minutes on CPU and needs memory for a 2B float32 model. It caps visual tokens at 576, unlike higher-resolution upstream evaluations.

### Decision pilot

Programmatic topic-rule fixtures: 80 train, 32 calibration, 48 test. Lexical groups are separated; grammar is shared. Frozen transformer states feed three-way heads at layers 6/12/18/24. Temperature/threshold selection reuses one small calibration set: weaker than separate gate-training/calibration partitions. Test labels do not select thresholds.

Timed adaptive execution counts layers and stops at an accepted head. Full-depth probing alone is not compute savings. The minimal token baseline is zero-shot while heads receive labels, so this is not a fair trained-head-versus-trained-token accuracy comparison. Timings exclude tokenization/loading. No production risk gate passed.

### Coordinate pilot

The adapter reproduces [GUI-Actor](https://github.com/microsoft/GUI-Actor)'s published pointer architecture with pretrained weights. It runs vision and all 28 transformer layers, reads a pointer query and chooses a patch. It never calls autoregressive generation. A paired pass adds one vocabulary projection; it is not a coordinate-text or JSON baseline.

`fixtures/interface.html` is our original inert GUI. `capture_fixtures.cjs` captures two viewports and element boxes using Playwright. Eight prompts include six visible targets and two absent targets. Missing targets still get a patch: this checkpoint has no abstention class. No clicks execute and no account content appears.

```sh
node experiments/capture_fixtures.cjs
```

## Refresh the local record

```sh
python scripts/refresh_research.py
python scripts/build_site.py
python scripts/check_site.py
node scripts/test_browser.cjs
python scripts/record_provenance.py
```

Browser tooling can use an existing module through `PLAYWRIGHT_MODULE`. Provenance records finalized source/artifact hashes, git base and dirty-diff identity; it is not immutable pre-registration. Chromium viewport checks are not real-device validation. None of these commands deploys.
