# Image to action: two different demonstrations

The [real-screen replay](../screenshot-demo.html) displays actual saved GUI-Actor outputs on 30 public screenshots. The [live arrow reader](../image-action-demo.html) runs a newly trained small pixel classifier in the browser. The replay does not perform live vision inference; the live model does not understand general screenshots.

## Real screenshots and recorded clicks

The replay uses every one of the 30 previously selected ScreenSpot cases, including misses. Each original PNG is copied without modification and checked against its saved SHA-256. Only the currently selected screenshot loads. The point and target rectangle are separate display overlays; neither changes the underlying image.

The model is [Microsoft GUI-Actor-2B-Qwen2-VL](https://huggingface.co/microsoft/GUI-Actor-2B-Qwen2-VL), revision `8f87b366d004425a9823502553e2097c71116ece`. Its pretrained pointer head scores visual patches. The two readouts use the same saved model pass: the highest-scoring patch hit 16/30 targets; the published connected-region readout hit 25/30. All 28 transformer layers ran. No coordinate text was generated.

These are the existing measurements, not new model inference or a live screenshot service. No click is sent to another application. The public dataset card describes [ScreenSpot](https://huggingface.co/datasets/bevaya/ScreenSpot) and identifies its Apache-2.0 license. The demo preserves [attribution and license information](../results/action-demo/screenshots/NOTICE.txt); depicted third-party content retains its own rights. The dataset revision is `0be08781e2e188582f6131625ae1598d443b4d5d`.

## New check: can a confidence cutoff avoid bad clicks?

We tested two predefined review rules using saved outputs. Each fold holds out one entire platform family: Android, iOS, macOS, Windows or web. A cutoff is selected using the other 24 examples, maximizing allowed clicks with zero wrong clicks in that training fold. The rule is then evaluated on the six held-out examples. Scores use either the peak patch probability or that peak multiplied by the visual-patch count. Neither score is a calibrated probability of correctness.

| Rule | Correct automatic clicks | Wrong automatic clicks | Sent to review | Already-correct points withheld |
|---|---:|---:|---:|---:|
| Always use connected-region point | 25 | 5 | 0 | 0 |
| Peak-score cutoff | 7 | 1 | 22 | 18 |
| Grid-relative peak cutoff | 7 | 1 | 22 | 18 |

The cutoff avoided four wrong automatic clicks by referring most requests to review. It still allowed one wrong click and withheld 18 already-correct points. This is an expensive coverage tradeoff, not a solved reliability problem. Review remains unresolved; we did not test a human or second-model fallback. Every target in this sample exists, so the experiment cannot establish missing-target detection.

The same 30 screenshots had already been inspected. Holding out platforms for cutoff fitting prevents direct held-out-label selection within this calculation, but does not turn this retrospective study into fresh validation. Both predefined rules and every fold are retained.

- [Sealed review protocol](../results/action-demo/screenstudy/protocol.json)
- [Every training cutoff and held-out fold](../results/action-demo/screenstudy/folds.json)
- [Every review decision](../results/action-demo/screenstudy/decisions.json)
- [Summary](../results/action-demo/screenstudy/result.json)
- [Original screenshot/model study](screenspot-results.md)
- [Replay data with original image hashes](../results/action-demo/gallery.json)

## Live model: pixels to one of four actions

The browser model takes only 1,024 grayscale pixel intensities from a 32×32 canvas. It applies a trained 1,024-to-32 linear layer, ReLU, and a 32-to-4 output layer: 32,932 parameters total. Argmax directly returns UP, RIGHT, DOWN or LEFT. The renderer seed, direction label and geometry are not inputs to the predictor. Uploads are resized locally to this input shape; no uploaded image leaves the browser.

This is a synthetic visual-instruction task. It does not use Qwen, a language model, a screenshot detector or a textual decoder. Every input receives one of four actions; there is no unknown-image or absent-sign class. A photo or unrelated screenshot can therefore receive a meaningless direction. The supported demonstration is a dark arrow on a light background.

Training uses 2,400 filled-arrow cards, with random position, size, shaft width, small angular variation, brightness and pixel noise. A separate 400-card tuning split selects the training epoch by accuracy, breaking ties with cross-entropy. Both filled-card test and outline-style stress sets contain 400 independently seeded renders. Every split has distinct seeds; all filled sets share the same procedural renderer. The unseen outline style never participates in training or epoch selection.

Twenty fixed Adam epochs ran on two CPU threads. Epoch 19 was selected using the tuning split. All weights are exported as JSON and executed by plain browser JavaScript. The reference control compares the same pixels with four untrained, centered, canonical arrow templates using squared pixel distance. It does not align or resize those templates and is a deliberately simple baseline.

| Image set | Trained model | Fixed templates |
|---|---:|---:|
| New filled-arrow seeds | 393/400 (98.25%) | 130/400 (32.50%) |
| Unseen outline style | 349/400 (87.25%) | 126/400 (31.50%) |

The live demo's 800-card check recreates these exact saved test definitions and runs both methods again in the user's browser. It reports results, not a prerecorded animation. All 800 exported JavaScript decisions matched the Python/Torch model; the largest raw-logit difference was 0.00001121. These counts support the bounded pixel-classifier demonstration, not a real-world navigation or screenshot claim.

The saved Node timing experiment retained three passes per method over 400 already rendered test images. Neural prediction took 53.81, 50.22 and 50.12 ms; templates took 42.02, 41.33 and 41.18 ms. The learned model was more accurate but slower than this cheap control. These are local numerical-readout diagnostics with rendering and loading excluded, not an LLM speed comparison or guaranteed browser latency.

- [Sealed arrow protocol](../results/action-demo/arrows/protocol.json)
- [Training history and selected epoch](../results/action-demo/arrows/training.json)
- [Trained weights](../results/action-demo/arrows/model.json)
- [Every test and stress prediction](../results/action-demo/arrows/predictions.json)
- [Summary and deployment parity](../results/action-demo/arrows/result.json)
- [Every timing pass](../results/action-demo/arrows/timings.json)
- [Shared renderer and real pixel inference](../scripts/action-demo-core.js)

Run each experiment's `--prepare` stage before its execution stage, after inspecting the implementation and in a new study copy if changing anything. The runners refuse to overwrite completed results. The browser demos use their saved artifacts; no training is silently performed in the page.

## Internal verification

A separate review agent independently recomputed all ten screenshot folds and cutoff choices, checked target-box hits and image hashes, and regenerated all 800 arrow image hashes, template outputs and JavaScript model logits. It verified the tuning-only epoch choice and the Torch/JavaScript decision agreement. This is an internal artifact audit, not external replication.

Actual Chromium checks reproduced all 800 arrow decisions, classified a local uploaded arrow PNG, and exercised every recorded screenshot and its point. Both pages passed overflow and automated WCAG checks at 320, 390 and 1,440 pixels; keyboard controls were exercised. [Browser check record](../results/action-demo/browser-check.json).
