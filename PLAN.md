# SkinDeep documentation, demonstrations, and validation plan

Updated 2026-09-19. Status: approved site checkpoint pushed as 3881aff; public-data follow-ups added locally. Preserve the approved UI, styling and navigation. New research belongs in linked reference notes, with concise measured updates on existing pages.

The follow-up covers BANKING77 (3,080 official test queries, three head seeds), a stricter gate on 742 fresh calibration / 741 fresh test queries, 30 public ScreenSpot examples, and an actual optional browser Qwen output-format benchmark. Adversarial-review work adds separate implementation replays, a published dependency release, a 30-intent CLINC subset with out-of-scope evaluation, a 31-output UNKNOWN variant, changing-rule pairs, and a matched one-token control. All tested early-exit calibration gates still fail. The changing-rule test also fails its preset criterion. See the measured reports in README.md and docs/evidence-acceptance.md.

Implemented: research-lab sidebar and 29 current pages; preserved 21 historical pages; corrected bounded math and small-batch example; seeded browser sessions, blind evaluation and annotation workbench; 29-entry research register; measured synthetic, Qwen decision and pretrained GUI-Actor coordinate pilots. Results and limitations are in `results.html` and `results/`.

The Qwen exit gate missed the proposed quality tolerance. Coordinate reproduction has no absent-target rejection. Human studies, real generators, model transfer, accelerator/cascade comparisons and domain applications require further work and are not marked complete.

The immediate goal is to make the existing work accurate and reproducible. The next goal is to test whether small task heads attached to pretrained representations can deliver useful preferences, decisions, and coordinates with less computation.

This plan covers the current static HTML/CSS/JavaScript site, its Python examples, and separate model experiments. It preserves the historical record and existing URLs. The responsive-layout changes already in the working tree are separate completed work; they do not establish the scientific claims below. The subsequent implementation trained small Qwen readout heads locally and ran a pretrained coordinate model. The approved site checkpoint was committed and pushed at Steve's request. These follow-up experiments remain local; no hosting or account changes were performed.

## 1. Product and research framing

Use **small task heads over shared model representations** as the proposed research theme. Keep three clearly distinguished demonstrations:

| Track | Question | Output |
| --- | --- | --- |
| Preference learning | Can a small personal model learn useful preferences and guide generation? | Predicted preference, ranked candidates, or a proposed latent edit |
| Direct decisions | Can a pretrained model return a useful categorical answer without generating text? | Task enum and calibrated confidence |
| Direct actions | Can a vision-language model locate a target without spelling out coordinates? | Target region/coordinates, action type, and abstention |

Early exit is an optimization to validate for direct decisions and actions. Full-depth task heads must work first. Representations from different models, layers, or generators are not presumed interchangeable.

Preserve dynamic instructions and full relevant context in the decision experiments. A specialist that works only under fixed training-time rules does not satisfy Steve's proposed use case.

## 2. Priorities and sequence

Effort bands describe implementation scope, not elapsed-time promises: small = local documentation/math work; medium = harness or head training; large = generator integration, human study, or hardware investigation. Actual compute requirements depend on the selected models and available equipment.

| Phase | Deliverable | Dependencies | Exit criterion | Effort |
| --- | --- | --- | --- | --- |
| 0 — Evidence inventory | Claim ledger, implementation map, and experiment protocol | None | Every quantitative/current-status claim has a source, measurement, or explicit hypothesis label | Small |
| 1 — Correct the current site | Consistent terminology, repaired mathematical explanations, page-specific edits | Phase 0 | Documentation describes what the current implementation actually does | Medium |
| 2 — Reproducible preference demo | Correct bounded optimization, repeatable sessions, honest evaluation, runnable reference | Phase 1 and P01 | P01 passes; P02/P03 produce reproducible results including failures | Medium |
| 3 — Direct-output prototypes | Full-depth enum head and coordinate head in an isolated Python experiment environment | Protocol plus D01/C01 | Direct outputs meet task-quality gates; complete latency breakdown recorded | Medium |
| 4 — Adaptive depth | Intermediate probes, trained exit heads, actual runtime stopping | D01/D02 and C01 | Lower measured cost at the registered error limits; verify unexecuted layers | Large |
| 5 — Handoffs and hardware | Separate-model cascade, optional trained activation adapter, CPU/GPU/NPU comparison | Reliable full-depth and early-exit baselines | End-to-end benefit survives transfer, fallback, and scheduling costs | Large |
| 6 — Publish measured findings | Benchmarks page, reproducible configurations, limitations, selected new demos | Completed experiments | Every published result can be traced to an artifact and scoped test | Medium |

Preference work and full-depth direct-output prototypes are independent after the common evaluation protocol exists. Combining all three ideas is conditional on their individual results.

First implementation batch: claim ledger; P01 math/edge-case checks; demo wording and failure states; one consistent reference implementation. First model batch: D01 full-depth enums and C01 full-depth coordinates. Do not start with a multi-hardware chimera.

## 3. Confirmed findings that drive the work

These findings come from the local source, not model benchmarks:

- `scripts/demo.js` trains logistic regression directly on 12- or 16-dimensional renderer parameters. Its reported fit is calculated on the training ratings; its timing covers the local training function, not a real image generator.
- `idealZ()` uses signs of weights but sets weights below 3% of the largest magnitude to zero. With nonzero ignored weights, that is not the exact maximum over the latent box. Its truncation factor further changes the objective value.
- `transform()` takes an unconstrained linear-model step and then clips coordinates. Clipping can prevent it reaching the requested score; it does not generally solve the bounded minimum-distance problem. A minimum-change-to-at-least-target operation should also leave an already-satisfactory input unchanged.
- `nextBatch()` mixes five predicted likes, four uncertain samples, and three exploratory samples. Documentation presenting a universal 70/30 rule does not describe this implementation. The random-comparator sort should become a seeded shuffle for reliable comparisons.
- `examples/core_implementation.py` trains a multilayer model on encoded generated content, whereas the browser demo trains on the original latent vectors. These are different pipelines with different inference/training costs.
- The Python example divides training loss by `len(X_train) // batch_size`, which can be zero for supported small datasets. Single-item batches also need checking against its BatchNorm and squeeze usage. Threshold-based generation has no attempt limit.
- The multimodal-preferences page attributes multiple preference modes to randomized search through a single linear-plus-sigmoid classifier. Over a convex box, that classifier cannot represent disconnected high-preference regions; variation along an acceptable-score set is a different property.
- Some pages contain numerical performance/efficiency claims without a reproducible experiment attached. Historical filing language and current legal-status language are inconsistent; reconcile them with the records instead of inferring status.

Additional suspected example-code issues should be reproduced before being described as confirmed runtime failures. That was the initial inventory; the current implementations and their measured limits are linked above.

## 4. Documentation rules

Create a claim ledger with: claim ID, wording, page/section, status, source or experiment ID, model/data/hardware scope, measurement date, limitation, and next action. Statuses: historical evidence, measured here, reported by another source, hypothesis, illustration, or unsupported.

Apply these rules:

1. Keep original artifacts unchanged. Add corrections and context outside historical documents; distinguish a corrected current technical note from the archived 2025 whitepaper.
2. Use one glossary: latent code, feature representation, classifier/logistic regression/SVM, task head, preference score, optimization, calibration, early exit, and escalation.
3. Write reverse classification as constrained optimization. A score generally has many corresponding latent inputs; it is not a unique inverse and is not a guarantee of user satisfaction.
4. Separate user preference, content validity, and required policy constraints. Negative examples help learn a boundary but do not guarantee compliance.
5. Scope speed claims to the measured computation. Distinguish training, input encoding, head evaluation, generation, transfers, and full interaction latency.
6. Replace absolute market/novelty claims with dated, attributable comparisons. Related work can share a mechanism without proving direct influence or priority.
7. Label incomplete integrations as illustrative code. A working parametric renderer is a useful demo, but is not evidence that every generative model supports the same integration.
8. Explain that storing embeddings or adding unspecified noise does not itself establish privacy. Define and test the relevant exposure before making a claim.

## 5. Page-by-page backlog

P0 = credibility/correctness; P1 = reproducibility/usability; P2 = expansion after evidence.

| Page/file | Priority | Planned changes | Evidence/dependency |
| --- | --- | --- | --- |
| `index.html` | P0 | Keep rate/learn/generate introduction; qualify predicted ideals and timing; use precise historical filing language. Add research-track links only when the destinations contain substantive work. | Claim ledger; P02/P08 |
| `demo.html` | P0/P1 | Put the renderer disclosure near the controls; distinguish training fit, unseen-rating performance, and model confidence; report attainable targets and failures. | P01–P03/P06 |
| `how-it-works.html` | P0 | Explain the learned approximation, bounded search, sample uncertainty, and failure cases. Explicitly distinguish its 2D teaching widget from the classifier demo. | P01/P02 |
| `history.html` | P0 | Keep dated artifacts; distinguish shipped, demonstrated, described, and later proposed features; verify evidence links and filing terminology. | Claim ledger |
| `landscape.html` | P0 | Verify each comparison against primary sources; separate observable features from inferred implementation; replace universal absence claims. Include related work for new tracks. | Source review; D/C tracks |
| `about.html` | P0 | Reconcile dates and filing/status wording; reduce repeated positioning; attach evidence to material scale/biographical claims. | Claim ledger |
| `getting-started.html` | P1 | Give one tested path from setup to ratings to prediction to bounded optimization; distinguish historical requirements from maintained code. | Reference implementation; P01/P02 |
| `whitepaper.html` | P0 | Clearly label the original 2025 material and provide an errata/current technical note; correct inversion, calibration, mode, speed, and constraint claims in current guidance. | P01–P11 |
| `active-learning-strategies.html` | P0 | Remove or label unsupported percentages; define selection quality and sample-efficiency metrics; use actual demo mixture as one baseline. | P03 |
| `multi-modal-preferences-deep-dive.html` | P0 | Correct linear-model geometry; separate alternate acceptable samples from distinct modes; compare nonlinear heads/mixtures and context-conditioned preference. | P04/P07 |
| `future-explorations.html` | P1 | Keep archive status; add links to accepted experiments and mark illustrative snippets. Preserve historical predictions as such. | Experiment registry |
| `roadmap.html` | P0/P1 | Preserve the dated forecast; link to a current milestone roadmap driven by measured outcomes. Do not silently rewrite predictions as achievements. | Phase gates |
| `vc-analysis.html` | P0 | Make hypothetical provenance prominent and retain in archive; replace current-facing claims with measured deployment economics if available. | D07/X02 |
| `investment-response.html` | P0 | Keep as archived correspondence/positioning; move verified evidence to About/History; correct misleading current-status statements with annotations. | Claim ledger |
| `sitemap.html` | P1 | Separate current demos, maintained documentation, experiments/results, and archives. Preserve working old URLs. | Content updates |
| `examples/index.html` | P1 | Mark each entry runnable, illustrative, or proposed; list required model/data/runtime and link to its validation record. | Example smoke checks |
| `examples/art.html` | P1 | First real-generator integration candidate; explain representation, constraints, baseline, and expected result. | P08 |
| `examples/music.html` | P2 | Select a real controllable generator; test blind preference and continuity rather than promising universal musical personalization. | P12 |
| `examples/social-media.html` | P2 | Distinguish preference, engagement, and satisfaction; compare ranking existing candidates with personalized generation under the same budget. | P13 |
| `examples/dating.html` | P0/P2 | Remove absolute privacy/screenshot claims from current guidance; label prototype status; specify matching and privacy evaluation separately. | P11/P14 |
| `examples/science.html` | P0/P2 | Label as a proposal; require validity/property constraints and independent objective evaluation; avoid treating preference score as scientific discovery. | P15 |
| `README.md` | P0/P1 | Align summary and claims with the site; add exact local setup, demo/reference distinction, and links to plan/results. | Phases 1/2 |
| `PLGL_Whitepaper.md` | P0 | Preserve historical body if designated archival; add clear provenance/errata links and keep its status consistent with HTML. | Historical inventory |
| `examples/*.py` | P1 | Reproduce correctness issues; make one canonical minimal latent-classifier path. Retain feature-based alternatives under explicit names and assumptions. | P01/P02/P08 |

Keep the current static-site architecture. Model training belongs in an isolated experiment environment, not in a new dependency chain for serving HTML.

## 6. Demo and reference implementation changes

### Existing browser demo

- Make exact bounded maximization and aesthetic/truncation heuristics distinct controls/results. Do not call a modified heuristic an exact optimum.
- Implement and test minimum-distance-to-target within bounds; report achieved score and unreachable requests. Handle zero weights, already-met targets, extreme probabilities, one-class ratings, undo-to-zero, and reset explicitly.
- Add reproducible seeds and local session export/import, including ratings, model version, sampling policy, and renderer version. Keep data local unless the user explicitly exports it.
- Add a blind evaluation mode: record predictions before seeing a new rating; keep a separate uniform/held-out evaluation stream so active sampling does not bias the reported generalization score.
- Compare random, uncertainty, diversity, and mixed decks with equal label and candidate-generation budgets. Keep “fun to rate” and “efficient to learn” as distinct outcomes.
- Show real timing distributions and sample counts. Include rating, retraining, selection, and rendering costs; measure different history lengths and devices.
- Add an educational two-mode synthetic example to show where linear preferences fail, then compare a small nonlinear alternative.
- Preserve keyboard, touch, menu, mobile-width, and landscape-modal behavior. Check 320/375/390/768/900/1024/1440px, long text, zoom, reduced motion, and physical mobile browsers when available.

### Canonical reference

Start with one CPU-capable implementation matching the browser's latent-logistic model and bounded objective. Give it deterministic fixtures shared with the browser, a minimal dependency specification, and one runnable command. Keep optional real generators and feature-space classifiers as separate examples. Add tests for mathematical invariants, small batches, bounded termination, and device placement; avoid tests that merely restate the implementation.

### Proposed research demonstrations

| Demo | What the visitor can inspect | Ship condition |
| --- | --- | --- |
| Preference lab | Training/evaluation curves, sampling policies, multiple preference modes, feasible edits | P01–P07 |
| Decision lab | Input plus editable rules, per-layer predictions, exit location, result/abstention, measured latency | D01–D04 |
| Coordinate lab | Screenshot plus instruction, candidate regions, click point/no-target, ground truth, latency breakdown | C01/C02 |
| Adaptive action lab | Same grounding task with fixed-depth versus early-exit execution and fallback | C03/X01 |

Prefer recorded benchmark traces with explicit provenance for the public static site. A recorded trace must be labeled as a replay, not live model inference. An optional local endpoint can power live experiments later; hosting/runtime choice is outside this plan.

## 7. Architecture contract for new experiments

Separate the task result from control flow:

- Task result: moderation class, relevance, preference score, action class, region/coordinate, or no-target.
- Control state: `RESOLVED`, `IGNORE`, `CONTINUE`, or `ESCALATE`.
- Diagnostics: calibrated confidence where justified, exit depth, model/head versions, and timing breakdown.

`IGNORE` requires confidence that no action is needed. `CONTINUE` means this layer has insufficient evidence. `ESCALATE` is a tested routing decision, not a claim that the problem is intrinsically impossible. Store a reason code for unsupported inputs or runtime failures separately from semantic uncertainty.

The internal return type can be a tuple/struct/tensor. A surrounding API may serialize it as JSON without making the model generate JSON tokens. Use a query/readout position after the full input for causal text models; for grounding, compare query-to-visual-patch heads with coordinate regression.

Intermediate-state collection during full inference is useful for training/probing, but does not demonstrate early-exit savings. The adaptive runtime must stop subsequent blocks and account for any missing cache/state if it later resumes or processes a continuing session.

## 8. Deliverables and completion checks

Proposed additions during implementation:

- `docs/claims.md`: evidence ledger and corrections.
- `docs/method.md`: maintained technical description, distinct from the historical whitepaper.
- `experiments/`: isolated, pinned training/evaluation configurations and scripts.
- `results/<experiment>/<run>/`: configuration, environment, aggregate metrics, predictions permitted by data terms, and report. Large/private datasets and weights stay outside the public repo.
- A benchmark index linking public claims to exact results and limitations.

A documentation phase is complete when terminology, provenance, links, examples, and claims agree across pages. A research phase is complete when the registered evaluation has a reproducible result, including a negative result. Publication depends on the applicable experiment gate, not an attractive screenshot.

Work directly on the existing branch unless Steve requests another. Preserve unrelated edits. Explicitly limit launched CPU workers to at most half the available logical processors across concurrent work. Do not access Cloudflare, deploy, purchase compute, or send/recruit participants as part of this plan. Those actions are not needed for the initial local validation.

The detailed experiments, comparison methods, and decision gates are in [EXPERIMENTS.md](EXPERIMENTS.md).

## Practical workload findings and the next iteration

The requested workload tests are implemented: [600 real archived chat messages](docs/chat600-results.md) and [closed-loop maze actions](docs/maze-actions.md). The separate [chat review](docs/chat600-review.md) checks the evidence. Shared styling remains unchanged; the maze is a clearly labeled recorded replay.

The current chat candidate always exits at layer 12. It gets 519/600 labels right versus 522/600 at full depth, while missed toxic messages rise from 19 to 26. Both calibration variants reject it, and the full-depth model misses its own quality minimums. The maze policies both reach 0/10 goals. These are completed negative results, not a passing validation phase.

The next study should address those specific failures in this order:

1. **Improve the task model first.** On development data, compare the current frozen linear readout with a small nonlinear readout and task adaptation of the representation. Retain the cheap lexical baseline and an equally supervised one-token output. Select using toxic recall and false blocks together; overall accuracy alone can reward always returning SAFE.
2. **Train an error predictor, separately from the answer classifier.** Use out-of-fold development predictions to label when an intermediate answer is wrong, when it adds a toxic miss, and when more layers correct it. Give the gate intermediate representations and checkpoint changes available at that depth. Evaluate fixed layers 6/12/18 alongside it; a constant-depth policy must not be described as readiness detection.
3. **Permit a useful third result.** Add REVIEW as a control decision, not a new source toxicity label. Measure error among automatic decisions, review load, toxic recall and processing time including fallback. Review cannot silently count as a correct SAFE/BLOCK answer.
4. **Register fresh evaluation before selection.** Existing 600-message outcomes are now development knowledge. Keep them as a named regression set, then seal new disjoint calibration and test IDs from unused source rows, including effective-input deduplication after clipping. Fix quality tolerances and gate-selection rules before inference. Retain full-message and clipped-input results separately.
5. **Measure the accepted policy.** Only after quality passes, repeat isolated full-workload timing, balance every path across order positions, and report every complete pass. Include the rejected-candidate result too. A second runtime or machine is a separate replication, not something a query bootstrap substitutes for.

For mazes, first establish successful full-depth goal completion on new layouts. A legal-action mask may be a separately named system variant, but cannot be silently applied to the recorded model-only failures. Continue reporting collisions, loops, reached goals and a shortest-path reference. Larger games or visual observations should wait until the simple action task works.

These follow-ups are a plan, not experiments claimed to have run. No target score of 10 overrides a failed criterion, and no evaluated dataset is recycled as an untouched test.

## Exploration completed on smaller samples

At Steve's request, the next ideas were tried with 100 previously inspected balanced messages and 50-message adapter timing samples. This is a deliberately smaller exploration phase before fresh confirmation. [All outcomes](docs/chat-smoke-results.md) and the [adversarial review](docs/chat-smoke-review.md) include negative results, matched initialization controls and the separately recorded shorter-warm-up repair. Earlier untouched-data requirements still apply to validation; these reused samples cannot satisfy them.

The learned gate is the clearest follow-up candidate: different messages stop at blocks 6, 12, 18 or 24, with 33.8% measured time saving in one paired run. Distillation improved the final-layer score but did not improve early-layer accuracy. Next freeze a small candidate set, then evaluate unused messages, new toxic misses, false blocks and repeated isolated runtime. Do not continue selecting on the same 100 examples.
