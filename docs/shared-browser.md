# Shared two/four-layer model in the browser

This demo runs actual ONNX Runtime Web inference, with one WASM thread. It exports the existing jointly trained Google BERT four-layer specialist (11,105,796 parameters), not Qwen. The same trained weights are used by both paths.

- Prefix: token embeddings, layers 1 and 2, masked mean pooling, and a 256-to-2 classifier.
- Stop: BLOCK probability <=0.1 returns SAFE; >=0.8 returns BLOCK.
- Continue: the prefix's hidden-state tensor and attention mask go directly to a second graph containing layers 3 and 4, pooling, and the final classifier (BLOCK threshold0.2).
- Full control: an unsplit graph executes embeddings and all four layers, with the same final classifier. It does not pay for exporting the intermediate state or running the layer2 classifier.

The adaptive path never repeats embeddings or earlier blocks. No vocabulary projection or generated text is present. The prefix is 38,141,179 bytes; suffix 6,351,702 bytes; full control 44,506,427 bytes. Trying a typed message loads only prefix+suffix (~44.5 MB), plus tokenizers and runtime. Starting a benchmark additionally loads the full control, for ~89 MB total. Empty or invalid requests are rejected before loading. The control is created and warmed before timed calls; it is reused for later benchmarks in the same worker.

Setup accounting records adaptive setup (including its warmup), additional control setup (including its warmup), cumulative setup, and the increment for the current action. A benchmark after a typed message therefore reports only control initialization as its new setup work while retaining the cumulative total. Dataset download is separate. Stopping the worker discards its runtime and starts a new setup on the next action.

## Checks and scope

The Python/ONNX export check preserved all 100 saved adaptive decisions and exit layers: 83 correct, 10 toxic misses, 7 false blocks, 72 early exits. All final-depth decisions also match. Largest PyTorch/ONNX logit difference: 0.0000025928. These 100 balanced ToxicChat examples were already consumed in earlier experiments. This is export parity, not a fresh accuracy or general reliability result.

The browser runner first checks all 100 adaptive decisions, exit layers and exact input token IDs against Python, then executes three counterbalanced paired passes on the first 50 messages through both adaptive and unsplit-full paths. Raw text bounding with the pinned Qwen tokenizer, BERT tokenization, graph execution, prefix/suffix boundary handling, classifier/gate, and tensor cleanup are timed. Imports, model loading, warmup and dataset download are excluded. Only the tokenizer comes from Qwen; no Qwen model executes.

Every browser call records the invoked graphs. Prefix-only means layers 1/2; prefix+suffix means all four; the full control calls one graph. Browser times depend on device/runtime and are not interchangeable with earlier Python timings. The first 100-message pass is for parity; headline comparison uses the same first 50 and three passes.

Actual Chromium 151/WASM passed all 100 adaptive checks and all 300 paired calls, including token IDs and graph traces. Maximum probability difference from Python was 0.0000003312. The concurrent functional run measured a mean 1.416 seconds full versus 0.873 seconds adaptive per 50 messages; other research jobs were running, so these timings are diagnostic. [Browser records](../results/shared-browser/browser-functional.json) retain every request and all three passes. The separate UI check exercised a typed message, benchmark completion, all 100 result rows and seven populated viewport widths (320 through 1440 pixels).

Those initial browser records used eager loading of all three graphs. Exact original worker/UI/test scripts are archived under `results/shared-browser/original-browser-source/`, matching the original protocol's hashes. The current worker defers the full control until a benchmark, without changing model weights, graph computation, stop rules or measured inference boundaries. The first lazy-loading test caught a local variable shadowing the setup function before any model download; its source and failure are archived, and the accounting variable was renamed.

The corrected worker passed `scripts/shared-decision-ui-test.cjs`: empty input downloaded no model resources, typed input fetched prefix/suffix only, and the subsequent benchmark fetched the full control exactly once. All 400 parity checks and seven populated viewport widths passed, with no page errors. Incremental control setup was 338.5 ms, separately reported from 4,816.7 ms adaptive setup. These setup values and the saved UI inference times came from a concurrent functional run, not isolated performance measurement. [Lazy-loading checks](../results/shared-browser/ui-lazy.json) and the [full browser report](../results/shared-browser/ui-lazy-report.json) preserve the evidence; the old UI results remain unchanged. A separate isolated run must be used for performance claims.

## Isolated browser timing

After the other launched model jobs stopped, the current lazy-loading worker completed a separate three-pass run. The full graph took **943.4 ms** per 50 messages; the adaptive path took **600.4 ms**, or **36.4% less time**. All 100 reference checks and 300 paired timing calls preserved predictions and execution traces. This uses the consumed reference sample, not the fresh 500-message quality test, and excludes setup and downloads.

[Isolated protocol](../results/shared-browser/browser-isolated-protocol.json) and [all browser measurements](../results/shared-browser/browser-isolated.json) retain the source hashes, individual calls and three passes. These are measurements on this CPU/browser; users can run the same comparison on their own devices.

## Reproduction and provenance

`python experiments/export_shared_model.py` exports and checks the frozen checkpoint. It preserves existing completed outputs. The original run completed numerical artifacts, then failed copying the Apache license because its source filename omitted `.txt`. `results/shared-browser/export-attempt.py` preserves the exact source matching the sealed protocol hash; the reusable source now corrects that packaging path, and LICENSE.txt was copied without rerunning numerical evaluation.

`node scripts/shared-decision-test.cjs browser-functional` runs Chromium's actual WASM engine. An isolated later run can use a new result name and `--isolated` after other model jobs stop. The flag records the operator's isolation assertion, not an automatic process audit. Results and script/model hashes are retained under `results/shared-browser/`.

The model and dataset are checksum-checked in the browser. The manifest includes reversible input token IDs from public evaluation data, not anonymized records. Typed messages run locally and do not enter downloaded benchmark reports. The benchmark downloads the pinned [ToxicChat0124 dataset](https://huggingface.co/datasets/lmsys/toxic-chat) (CC BY-NC 4.0); the base BERT is Apache 2.0. These trained research artifacts are intended for noncommercial research.

Training, split selection and confidence-rule limitations: [compact specialist](compact-specialist.md). Export protocol and numerical records: [result](../results/shared-browser/result.json), [protocol](../results/shared-browser/protocol.json), [records](../results/shared-browser/records.json).
