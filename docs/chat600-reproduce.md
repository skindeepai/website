# Reproduce the 600-message workload

Use a separate copy of the repository to preserve the recorded results. The runner verifies its source and data hashes, but `--fit` and `--benchmark` write their outputs to `results/chat600`; those result files are not immutable. The checks establish reproducible inputs, not independent attestation or external preregistration.

The recorded run used Windows, Python 3.13.2, Torch 2.6.0+cpu, published Transformers 4.50.3 and eight Torch CPU threads. Other platforms may produce different timings or numerical results. The dependency file pins direct dependencies and is not a complete transitive lockfile.

## Prepare an isolated environment

From that repository copy in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r experiments/requirements-chat600.txt
```

This installs into the copy's virtual environment. The original measured run instead used an isolated Transformers directory with the existing recorded dependencies, as described in the [published-runtime replay](reproduction.md#replay-with-a-published-dependency-release). A new environment installation is a reproduction attempt, not a claim that this exact installation procedure was separately executed here.

## Fetch the pinned research inputs

Run this Python code using the virtual environment, from the repository root. Downloads require network access and disk space. Source messages are CC BY-NC 4.0; model weights have their own upstream license. Keep source text in the ignored cache.

```python
from pathlib import Path
from shutil import copyfile
from huggingface_hub import hf_hub_download, snapshot_download

cache = Path('experiments/.cache/toxicchat')
cache.mkdir(parents=True, exist_ok=True)
revision = '29df8e4dba60e1f4af4b4075c0705c5b313548a8'
for split in ['train', 'test']:
    name = f'toxic-chat_annotation_{split}.csv'
    source = hf_hub_download(
        repo_id='lmsys/toxic-chat', repo_type='dataset',
        revision=revision, filename='data/0124/' + name)
    copyfile(source, cache / name)
snapshot_download(
    'Qwen/Qwen2.5-0.5B-Instruct',
    revision='7ae557604adf67be50417f59c2c2f167def9a775')
```

The runner checks both CSV hashes and uses local-only model loading. It excludes non-human-annotated rows, reproduces the recorded split IDs, and requires conversation separation. The additional clipped-input overlap check is recorded separately; the original protocol remains unchanged.

## Run the stages

```powershell
.\.venv\Scripts\python.exe experiments/chat600.py --prepare
.\.venv\Scripts\python.exe experiments/chat600.py --fit
.\.venv\Scripts\python.exe experiments/audit_chat600.py
```

Stop other model jobs before timing. The benchmark performs all 600 decisions for each of four paths, three times, rather than extrapolating a small subset. Keep the process running until `benchmark.json` is complete.

```powershell
.\.venv\Scripts\python.exe experiments/chat600.py --benchmark
.\.venv\Scripts\python.exe scripts/refresh_chat600.py
```

Use no more than half the machine's logical CPU threads across concurrent work. The runner caps its Torch pool at eight or half the available threads, whichever is smaller; tokenizer parallelism is disabled. Small machines should also lower BLAS/OpenMP limits in a separately registered variant before using other numerical workloads.

`predictions.json` contains labels, confidence scores and selected depths. `timings.json` contains each actual prediction, elapsed time and executed layer trace. `passes.json` records complete-workload durations. `benchmark.json` summarizes all 7,200 timed calls; four warm-up calls are excluded. Keep the failed calibration result alongside all timing claims.

The public `heads.npz` contains only the trained readouts and normalization parameters. It is not the Qwen checkpoint. Feature caches are local `.pt` files; do not load caches from untrusted sources. A repeat from scratch can omit those caches and recompute features from the pinned model.

The separate [review](chat600-review.md) describes checks actually performed. These instructions do not claim cross-machine replication, full-context moderation quality or successful adaptive readiness detection.
