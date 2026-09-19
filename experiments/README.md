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

Run model pilots sequentially. Cap aggregate work at half the machine's logical processors, counting all builds/pools. The coordinate pilot can take many minutes on CPU and needs memory for a 2B float32 model. The original processor cap was not applied as intended: actual grids were 936 desktop / 392 mobile patches. See [the correction](../docs/coordinate-correction.md). The new ScreenSpot harness explicitly caps and checks 576 patches.

### Decision pilot

Programmatic topic-rule fixtures: 80 train, 32 calibration, 48 test. Lexical groups are separated; grammar is shared. Frozen transformer states feed three-way heads at layers 6/12/18/24. Temperature/threshold selection reuses one small calibration set: weaker than separate gate-training/calibration partitions. Test labels do not select thresholds.

Timed adaptive execution counts layers and stops at an accepted head. Full-depth probing alone is not compute savings. The minimal token baseline is zero-shot while heads receive labels, so this is not a fair trained-head-versus-trained-token accuracy comparison. Timings exclude tokenization/loading. No production risk gate passed.

### Coordinate pilot

The adapter reproduces [GUI-Actor](https://github.com/microsoft/GUI-Actor)'s published pointer architecture with pretrained weights. It runs vision and all 28 transformer layers, reads a pointer query and chooses a patch. It never calls autoregressive generation. A paired pass adds one vocabulary projection; it is not a coordinate-text or JSON baseline.

`fixtures/interface.html` is our original inert GUI. `capture_fixtures.cjs` captures two viewports and element boxes using Playwright. Eight prompts include six visible targets and two absent targets. Missing targets still get a patch: this checkpoint has no abstention class. No clicks execute and no account content appears.

```sh
node experiments/capture_fixtures.cjs
```

## Public-data follow-up and browser check

The approved site styling stays unchanged; new evidence is linked from the existing technical sections.

```sh
python experiments/banking77.py --threads 8 --batch-size 8 --timing-samples 96
python experiments/screenspot.py --threads 4
python experiments/test_research.py
python scripts/refresh_evidence.py
```

BANKING77 downloads three pinned PolyAI files, fits four frozen-Qwen readouts with three initialization seeds, and evaluates all 3,080 official test queries. The manifest records exact split IDs and overlaps. Feature extraction can run for many minutes on CPU; it is cached in `experiments/.cache/`. The actual runtime check still executes Qwen and counts blocks. See [the protocol](../docs/banking77-protocol.md).

ScreenSpot downloads a predetermined public sample and enforces its image-patch cap in the processor field actually used by this runtime. Original source screenshots remain local. See [the protocol](../docs/screenspot-protocol.md) and [the earlier cap correction](../docs/coordinate-correction.md). Its durations are diagnostic; do not compare concurrent runs as isolated speed benchmarks. For the final BANKING77 timing record, stop other launched model benchmarks first and rerun from cached features.

The lexical control uses scikit-learn 1.8.0, joblib 1.5.3 and threadpoolctl 3.6.0. Install these into an isolated environment, or into the ignored `experiments/.cache/tooling` directory with `pip --target`; `banking77_lexical.py` supports that location. It uses one thread and exactly the same training IDs.

`browser-benchmark.html` downloads a pinned ONNX Qwen model only when started. It runs through Transformers.js 3.8.1 in a single-threaded WebAssembly worker. It compares ordinary one-letter and JSON generation; it does **not** run the trained heads or skip layers. The first real-browser output record is `results/ui/browser-qwen.json`, including failures. Stop terminates the worker. No result is uploaded.

## Adversarial-review follow-ups

The independent implementation in `reproduce_early_exit.py` loads portable numeric heads and executes Qwen again; it does not import the original runner or its feature cache. It passed with the original runtime and with an isolated published Transformers 4.50.3 release. See [the replay record](../docs/reproduction.md). `requirements-replay.txt` pins the core obtainable packages; it is not a full transitive lockfile or a claim that a clean environment was tested.

Additional prospective experiments preserve failures and keep their protocols separate:

```sh
python experiments/clinc_validation.py --prepare
python experiments/clinc_unknown.py --prepare
python experiments/clinc_validation.py
python experiments/clinc_unknown.py
python experiments/changing_rules.py --prepare
python experiments/changing_rules.py
python experiments/matched_output.py
python scripts/refresh_validation.py
```

The recorded UNKNOWN variant's `--prepare` was run before the first CLINC result existed. Its existing protocol is checked on replay. For new experiments, preserve that sequence: prepare both CLINC protocols before running the baseline. Do not replace a failed protocol or result to make the record appear stronger.

Download `data/data_full.json` and `data/domains.json` from `https://raw.githubusercontent.com/clinc/oos-eval/828f8093932c8fe6ca7936c3d2e52903b1c523de/` into `experiments/.cache/clinc/` before these commands. `clinc_validation.py --prepare` verifies their SHA-256 values against constants before using them. Keep the upstream CC BY 3.0 attribution; generated changing-rule prompts are an authored adaptation. Portable CLINC NPZ arrays use `{depth}_weight`, `{depth}_bias`, `{depth}_mean`, `{depth}_std` and, where applicable, `{depth}_temperature`. Load with `allow_pickle=False`.

Run the matched-output timing **after other model work finishes**. It controls only output format using identical trained readout rows; equivalent predictions follow by construction. It is not a comparison to an independently trained conversational model or ordinary unrestricted generation.

## Refresh the local record

```sh
python scripts/refresh_research.py
python scripts/build_site.py
python scripts/check_site.py
node scripts/test_browser.cjs
python scripts/record_provenance.py
```

Browser tooling can use an existing module through `PLAYWRIGHT_MODULE`. Provenance records finalized source/artifact hashes, git base and dirty-diff identity; it is not immutable pre-registration. Chromium viewport checks are not real-device validation. None of these commands deploys.

## Practical decision workloads

See [the 600-message protocol](../docs/chat600-protocol.md) and [results](../docs/chat600-results.md) for pinned ToxicChat source files, partition IDs, trained heads, failed quality gates and full-workload timings. `chat600.py --prepare`, `--fit` and `--benchmark` are separate stages; the sealed protocol rejects unrecorded source changes. Use the documented published Transformers replay environment. The benchmark uses eight CPU threads and should run after other model work has ended. Do not treat cached hidden-state extraction as early-exit timing.

[Exact reproduction commands](../docs/chat600-reproduce.md) cover the dataset downloads, isolated dependencies and all stages. Run them in a separate copy: fitting and timing overwrite their corresponding result files.

`maze_actions.py --prepare` followed by `maze_actions.py` runs the [maze experiment](../docs/maze-actions.md), capped at four CPU threads. It records failed moves, illegal actions, reached goals and executed blocks. The [browser replay](../maze-benchmark.html) reads saved artifacts and does no inference.

After model timing finishes, `node scripts/test_maze.cjs` checks every recorded replay step and its controls against the local preview. It uses the same separate Playwright tooling as the other browser checks.

Both studies retain negative results. New training or gate selection needs a new output directory, protocol and untouched evaluation set; do not overwrite these runs to improve their reported scores.

## Bounded exploratory comparisons

The [small comparison](../docs/chat-smoke-results.md) uses `chat_smoke_common.py` to select already inspected messages, explicitly as development exploration. This exception does not turn reused data into a fresh validation set. Scripts and result directories are separate from the earlier runs. Every launched model process is capped at four CPU threads; timing runs are isolated from training.

- `chat_smoke_heads.py`: frozen linear/neural readouts, asymmetric thresholds, agreement, learned error/benefit gates and REVIEW. `chat_smoke_heads_runtime.py` executes the selected gate with real layer stopping.
- `chat_smoke_specialist.py`: tiny BERT and escalation to Qwen; `--benchmark` times all executed fallback work.
- `chat_smoke_adaptation.py --variant full|joint|distill|fixed12`: separate 32-step adapter recipes. Run full first because distillation uses its teacher. `chat_smoke_adaptation_runtime.py --variant ...` checks portable weights with actual 50-message execution.
- `chat_smoke_adaptation_reset.py --variant joint`: separately recorded five-step head-warm-up repair, preserving the original failures.
- `maze_smoke.py` and `coordinate_abstention_smoke.py`: structured-action training and a retrospective coordinate-confidence diagnostic.

The adapters use the isolated published replay dependencies under `experiments/.cache/replay-runtime`, as recorded in source. Read each protocol before reproducing in a separate copy; commands can overwrite corresponding outputs. The `full|joint|distill|fixed12` notation lists alternatives, not a literal shell argument. `python scripts/refresh_smoke.py` rebuilds the reports from completed artifacts. No raw chat text is published.
