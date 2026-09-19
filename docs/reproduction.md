# Reviewer replay of the early-exit mechanism

A separate reviewer agent wrote a second implementation of the existing BANKING77 stopping rule. It imports none of the original experiment code, reads no saved hidden-feature cache, and runs Qwen again in a fresh process with at most four CPU threads.

This is a **local replay, not an external replication or a fresh-data validation**. The reviewer shares the machine, model weights, source data, and already trained classifiers. Matching earlier predictions does not make a failed reliability guard pass.

The recorded replay passed all 48 forward passes: full and early predictions matched on all 24 queries. Eight candidate requests exited at layer 12; sixteen continued through layer 24. The predetermined sample contained no layer-18 exit, so this replay does not independently confirm termination at that checkpoint. Five groups of saved numerical summaries also matched their per-query evidence.

## What the check does

The script selects 24 official test query IDs by sorting a fixed hash of each ID. Selection does not consult labels, confidence scores, errors or exit depths. It writes the selected IDs and policy before running inference, then executes both full depth and the early candidate for every query.

Hooks record each actual block index. The separate control path raises its own stop signal when the candidate qualifies and verifies that precisely the contiguous sequence of expected blocks ran. It compares the resulting label and exit depth with the saved seed-17 predictions. Original full-dataset summaries, source-file hashes, labels, and timing calculations also receive lightweight recomputation checks.

Elapsed durations are diagnostic only: this replay does not repeat the controlled timing design and may overlap other work. It provides no new speedup estimate.

## Portable classifier weights

The [NPZ array file](../results/banking77/seed17-heads.npz) preserves the trained seed-17 classifier weights, biases, means, standard deviations and temperatures. The [schema](../results/banking77/seed17-heads.schema.json) records shapes, numerical types, label ordering and checksums. Load it with `numpy.load(path, allow_pickle=False)`; no Python object deserialization is needed.

The export checks every stored array for exact equality after loading. For each of the four classifiers it also checks bitwise-equal logits on eight predetermined random vectors. Those are serialization checks, not accuracy tests. The trusted local `.pt` file is only needed to regenerate the export.

## Evidence and reproduction

- [Replay result and limits](../results/reproduction/result.json).
- [Selected IDs and policy](../results/reproduction/protocol.json).
- [Every prediction, checkpoint score and executed block](../results/reproduction/predictions.json).
- [Independent arithmetic checks](../results/reproduction/artifact-checks.json).
- [Environment](../results/reproduction/environment.json) and [installed Transformers source hashes](../results/reproduction/transformers-source-hashes.json).
- [Current separate implementation](../experiments/reproduce_early_exit.py) and [exact source used for the first replay](../results/reproduction/implementation.py).

With the original dataset files and pinned Qwen checkpoint locally available as described in [the experiment setup](../experiments/README.md), run:

```sh
python experiments/reproduce_early_exit.py
```

Only the maintainer regenerating the portable weights from the trusted local original file needs `--export-heads`. Replay loads the NPZ file directly. The dataset-file hashes and category-manifest hash must match before inference proceeds.

The environment record hashes the installed Transformers Python files, including the Qwen implementation. This identifies the code actually replayed; hashes alone do not provide an installable copy. A development-version string without a recoverable source commit remains a reproducibility limitation.

The protocol and outputs are local files that reruns overwrite, not immutable external preregistration or signed independent execution attestation. This check increases confidence in the implementation and arithmetic. It does not test unfamiliar inputs, changing instructions, alternate datasets, different hardware, or production reliability.

## Replay with a published dependency release

A second fresh process used the published **Transformers 4.50.3** wheel in an isolated local directory. It reused the recorded installed dependencies and made no global package changes. Both outputs matched for the same 24 predetermined queries. A separate, explicitly targeted example (`test:1727`) also matched and stopped at layer 18. This additional example was selected because the previous record already showed a layer-18 exit; it verifies that execution path, not its prevalence or generalization.

All **50 forward passes** matched their reference labels and exit depths. The original random sample still contains eight layer-12 exits and sixteen full-depth completions. Layer 18 is reported separately.

This demonstrates compatibility on this small replay with an obtainable published dependency. It does not recover the exact historical development build, repeat head training, or independently reproduce the full accuracy or latency benchmark.

- [Published-release replay result](../results/reproduction-release/result.json), [every executed path](../results/reproduction-release/predictions.json), and [prospective selection record](../results/reproduction-release/protocol.json).
- [Wheel SHA-256, actual import location and installed dependency versions](../results/reproduction-release/environment.json).
- [Exact replay source](../results/reproduction-release/implementation.py) and [installed release source hashes](../results/reproduction-release/transformers-source-hashes.json).

To repeat this isolated dependency check from the repository root, keeping the original environment's other recorded dependencies:

```sh
python -m pip download --no-deps --dest experiments/.cache/replay-wheels transformers==4.50.3
python -m pip install --no-deps --target experiments/.cache/replay-runtime experiments/.cache/replay-wheels/transformers-4.50.3-py3-none-any.whl
```

Then run this Python launcher. The output argument preserves the first replay record:

```python
import runpy, sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / 'experiments/.cache/replay-runtime'))
sys.argv = ['reproduce_early_exit.py', '--output', 'results/reproduction-release',
            '--target-layer18', '--runtime-wheel',
            str(root / 'experiments/.cache/replay-wheels/transformers-4.50.3-py3-none-any.whl')]
runpy.run_path(str(root / 'experiments/reproduce_early_exit.py'), run_name='__main__')
```
