# Real screenshots: locating a target and withholding a click

This experiment runs new image-and-instruction inference on 50 real desktop screenshots. It tests whether our existing pointer confidence rule transfers to a harder benchmark that includes infeasible requests. A separate ten-request comparison asks whether a second, closer look improves localization. It does not train a new vision model or establish a reliable NOT_FOUND classifier.

## Results

Combining nearby patches localized **8/25 present targets**, compared with 3/25 for the highest-scoring patch. This harder desktop sample does not reproduce the earlier 25/30 ScreenSpot result. Different data and small controls matter; it is not evidence that the underlying model generally attains the earlier accuracy.

| Readout | Correct clicks on present targets | Wrong clicks on present targets | Clicked on infeasible requests | Present requests withheld | Infeasible requests withheld |
|---|---:|---:|---:|---:|---:|
| Highest-scoring patch | 3 | 22 | 25 | 0 | 0 |
| Connected region | 8 | 17 | 25 | 0 | 0 |
| Connected region + peak cutoff | 2 | 2 | 0 | 21 | 25 |
| Connected region + grid-adjusted cutoff | 2 | 2 | 0 | 21 | 25 |

Each present/infeasible group contains 25 requests. The cutoffs withheld every infeasible request on this sample, but also 21/25 present requests. Of their four accepted clicks, two were wrong. Six withheld present requests already had correct proposed points. This is very low coverage with remaining wrong clicks, not a validated safe-action policy or a reliable semantic NOT_FOUND classifier. Refusing everything would also avoid all infeasible clicks and would complete no requests.

The ten prespecified crop cases changed from **2/10 correct to 3/10**: one correction, no lost correct location. Every case paid for a second full model pass. The small improvement does not establish a useful general accuracy/cost tradeoff.

The [artifact audit](../results/visual-next/audit.json) independently recomputed the region readouts, reference-region hits, crop geometry, gate selection and all 200 method decisions from retained numeric outputs. It also checked every image/source hash and all 60 primary/crop layer traces. This is internal verification, not external replication.

## Data fixed before inference

The source is the authors' [OSWorld-G benchmark](https://github.com/xlang-ai/OSWorld-G/tree/daa6bd8e0e629f0917ad2984df930bf0bd967540/benchmark), specifically `OSWorld-G_refined.json`, pinned to Git revision `daa6bd8e0e629f0917ad2984df930bf0bd967540`. The 564 requests include 470 rectangular targets, 40 polygonal targets, and 54 author-labelled refusal requests. The [paper's refusal examples](https://arxiv.org/html/2505.13227v1#A1.SS3) explain that the requested action is infeasible on the supplied screen. These are the benchmark authors' labels, not negative examples invented by this project. Refusal includes more than literal absence: a control may exist but the requested operation can be impossible (for example, moving a horizontal slider upward). Internal artifact fields named `absent` or `target_present=false` mean author-labelled refusal; they do not establish that every depicted control is missing. The site therefore calls these infeasible requests. Real screenshots and a public benchmark do not imply live production traffic.

Before inference, the runner selected 25 refusal and 25 present-target requests using seeded shuffles. All 50 use different screenshot files; no screenshot appears in both groups. Refusal selection runs first, then present selection excludes already selected screenshots. Another fixed shuffle determines evaluation order. This balanced sample deliberately overrepresents refusal relative to the benchmark and says nothing about production prevalence. Every request is new to this project's experiments, but unknown overlap with the pretrained model's training data remains possible.

The [sealed protocol](../results/visual-next/protocol.json) records sample IDs, instructions, author coordinates, source and image hashes, the selected cutoffs, runtime settings and the ten crop-study IDs. The [manifest](../results/visual-next/data-manifest.json) retains each image's hash and original filename. The results page shows the first sealed request in an optional example panel, with its original unmodified screenshot and a separate point overlay; it is not selected by outcome. [Image attribution](../results/visual-next/NOTICE.txt). Original images are downloaded without modification from `benchmark/images/` at the pinned revision. The repository's [Apache-2.0 license](../results/visual-next/LICENSE.OSWorld-G) is retained; screenshots depict third-party applications and content.

## What the model computes

The model is [Microsoft GUI-Actor-2B-Qwen2-VL](https://huggingface.co/microsoft/GUI-Actor-2B-Qwen2-VL), revision `8f87b366d004425a9823502553e2097c71116ece`. The experiment uses the same frozen pretrained weights, pointer architecture, prompt and image-processing budget as the earlier ScreenSpot experiment. The processor explicitly applies a minimum 256 and maximum 576 visual-token pixel budget; the actual grid is checked and recorded for each inference.

The vision encoder transforms the screenshot into visual features. All **28 language-model transformer layers** then process those features and the instruction. A trained pointer head compares a special pointer-query representation with image-patch representations and normalizes its scores over the patches. The adapter reads these numerical scores directly, bypassing the vocabulary projection and autoregressive text generation. Every call records an exact contiguous layer trace from 1 through 28. This is direct numerical output, **not early exit**.

Two location readouts share the same pass: the highest-scoring patch center and the upstream connected-region rule, which combines nearby patches above 0.3 times the peak score. Both return normalized `(x, y)` coordinates. Scoring uses the dataset's actual rectangle or polygon; the crop and model never receive the reference region as an input. This test measures a point, not task completion, and never sends a click to an application.

## The abstention rules

Two cutoffs are fixed using all 30 old ScreenSpot development results. One uses peak patch probability; the other multiplies the peak by the number of patches. Each chooses the largest development coverage with zero accepted localization errors, allowing a reject-all cutoff. Both accept 5/30 development requests, with cutoffs approximately 0.290121 and 165.875919 respectively. These are empirical scores, not calibrated correctness probabilities or guaranteed error bounds.

OSWorld-G results never select or change a cutoff. This tests transfer from the old mobile/web/Windows/macOS ScreenSpot development examples to a separate Ubuntu-desktop benchmark. It is not a claim that application families or platforms were withheld from the pretrained model's original training. No absent target was available in the 30 development requests.

A score below the cutoff returns **UNCERTAIN**, not a proved semantic NOT_FOUND. The pointer head has no dedicated missing-target output and still distributes probability over screen patches even when the request is infeasible. A refusal can therefore have a large peak. We separately count correct clicks, wrong clicks on present targets, clicks on infeasible requests, withheld present targets, and correctly located present targets unnecessarily withheld. A present request sent to review is unfinished; no successful human or model fallback is assumed. We avoid a single balanced-set accuracy number that would reward withholding everything.

## A second look around the predicted point

Ten present requests are fixed before inference. After the full-screen connected-region prediction, the runner crops a rectangle of half the original width and half the height, centered on that prediction and clamped to the image. It runs the same model on the crop with the original instruction, maps the new point into full-screen coordinates, and scores it against the reference. The target annotation never determines the crop.

This allocates more visual detail around a predicted location but can discard required context or make spatial language misleading. It always costs a second full model pass; it is not an adaptive speed optimization. The results retain both corrected and newly broken cases, and every crop rectangle, probability distribution, layer trace and diagnostic elapsed time.

## Execution and reproducibility

The runner uses CPU float32. After the first saved example, a runtime-only amendment increased PyTorch from two to four threads. After 13 saved examples, a second amendment raised it to twelve when other training jobs released the shared resource budget. The total launched model-work cap remained 16 threads on a 32-logical-processor machine. One inter-op thread and two-thread environment caps remain. The original runner, four-thread runner, original protocol and both amendments are preserved; subsequent inferences record their actual configured thread count. Each change repeats the first saved example and requires patch-score differences below 0.0001 and normalized-coordinate differences below 0.00001 before proceeding. The original two-thread outcome remains the evaluated result. See the [four-thread amendment](../results/visual-next/runtime-amendment.json), [twelve-thread amendment](../results/visual-next/runtime-amendment-12threads.json), [first parity check](../results/visual-next/thread-parity.json), and [second parity check](../results/visual-next/thread-parity-12threads.json). Timing starts before input processing and ends after the pointer result. Model loading is excluded. Other experiments may run concurrently, so times are diagnostics only, not isolated throughput or speedup evidence. Results checkpoint after each example and can resume without changing the sealed sample or rules.

Run `python experiments/visual_next.py --prepare` once before `python experiments/visual_next.py`. This recorded run resumes with `python experiments/visual_next.py --threads 12` under its explicit runtime amendments. Preparing refuses to overwrite a sealed protocol; executing refuses to overwrite completed results and verifies the sealed source hashes and image hashes. Changing the method requires a new experiment rather than silently refreshing this protocol.

- [Runner](../experiments/visual_next.py)
- [Recorded runtime and package versions](../results/visual-next/environment.json)
- [Every inference, numerical patch score, point and crop](../results/visual-next/predictions.json)
- [Every click or abstention decision](../results/visual-next/decisions.json)
- [Summary](../results/visual-next/result.json)

The primary run uses the historical Transformers 4.50.0.dev0 installation; its exact source revision is unavailable, as already disclosed for the earlier vision experiment. This limits exact environment reconstruction. A separately sealed one-request replay under the cached published **4.50.3** release passed with exactly identical patch probabilities and coordinates (both maximum differences were zero), including the full 28-layer trace. The artifact audit also matched its model and image-processor source hashes to the official wheel. This checks compatibility on one fixed request, not all 50 examples or a clean environment installation, and does not replace the primary outcomes. See the [replay protocol](../results/visual-next/published-runtime-protocol.json) and [complete replay](../results/visual-next/published-runtime-replay.json). To repeat that bounded check without overwriting evidence, run `python experiments/visual_next_replay.py --output experiments/.cache/visual-next/my-runtime-replay.json` after obtaining the pinned model and the cached published runtime described in [the earlier reproduction guide](reproduction.md).

These are internal exploratory measurements, not external replication or a full OSWorld-G leaderboard submission. The obvious next training target is an explicit no-action output, with labelled infeasible requests in development, followed by a separate untouched evaluation. The present study intentionally measures how far the existing location-only model and cutoff get before making that change.
