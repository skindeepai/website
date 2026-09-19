# Validation experiments

Updated 2026-09-19. The protocols below define the broader research agenda. Partial synthetic and model pilots now exist; see [results.html](results.html) and [research.html](research.html) for measured scope and next steps. No partial pilot completes its full protocol. Implementation sequence and page-specific changes are in [PLAN.md](PLAN.md).

## 1. Common protocol

Before each experiment, record the hypothesis, primary metric, baseline, splits, quality tolerance, candidate configurations, resource budget, and stopping rule. A failure to demonstrate improvement is a useful result and must remain visible.

### Fair comparisons

- Use the same input, instructions, relevant context, precision, image resolution, candidate budget, and comparable model backbone where the comparison allows it. Record intentional differences.
- Compare direct outputs against minimal constrained text output as well as JSON. Do not manufacture a speed advantage by requiring a verbose explanation from only the baseline.
- Keep data for fitting heads, choosing architectures/exit thresholds, calibrating confidence, and final testing separate. Group splits by user, conversation, app/site/template, and underlying screenshot as applicable. Split before augmenting or generating related examples.
- Begin with small development fixtures. For exploratory stochastic runs use at least three seeds; increase repetitions and independent test groups for confirmatory claims. Seeds are not independent human participants.
- Use real task labels, independently checked where ambiguous. Teacher predictions can supervise distillation, but agreement with the teacher is not ground-truth correctness.
- Report confidence intervals for primary differences; bootstrap at the independent unit (user/conversation/app), not at correlated individual tokens or screenshots. Determine final sample size from the target margin and observed variability. Small pilots do not establish rare-error guarantees.
- Measure natural task prevalence and also report difficult slices separately: unclear/missing targets, negatives, novel instructions, rare rules, long context, policy changes, and domain shift. Do not report only a balanced aggregate that hides the deployment error rate.

### Latency and resource accounting

Measure full input-to-result wall time plus preprocessing, tokenization, image/audio encoding, input transformer computation, head evaluation, exit checks, generated-token steps, transfers, fallback, and postprocessing. Report cold and warm runs, p50/p95/p99 where sample counts support them, peak memory, and energy only with an identified measurement method. Synchronize accelerator measurements appropriately. Report batch-one latency separately from throughput under batching/concurrency.

No parameter-count arithmetic is a substitute for latency, energy, or cost. Count both stages for escalated requests. If an explanation is required, count the model work that produces it. A speedup at constant quality is the claim to test.

Hardware work begins with an inventory of the available machine. Do not assume a supported NPU, available VRAM, or a particular acceleration backend. Cap CPU worker pools and simultaneous workloads to at most half the logical processors, and lower the limit if needed. Pin model revisions, runtime versions, thread counts, quantization, seeds, and power settings.

### Initial decision gates

These are suggested engineering targets to register before a run, not universal requirements or achieved results:

- For a low-impact prototype, compare quality using a predeclared non-inferiority margin, initially at most one percentage point absolute loss on the primary metric. Require uncertainty estimates consistent with that margin; insufficient data means inconclusive.
- Separately set maximum harmful-action and missed-required-action rates. Do not average these away. A low-impact margin does not authorize deployment for moderation or consequential actions.
- Seek at least a 20% median end-to-end latency reduction without a material p95 regression for extra early-exit complexity to be worthwhile. A direct-output head may still be useful for format reliability even if it misses the speed target; report that narrower finding.
- Choose confidence/exit thresholds only on calibration data. Report error versus accepted-request coverage, escalation fraction, and quality on accepted and escalated cases. “Abstain on everything” is not success.
- Count unparseable outputs, invalid coordinates, timeouts, unsupported cases, and unreachable preference targets. Do not silently drop failures.

Every result should contain the run ID, code commit plus dirty-diff identity if applicable, model/head revisions, input/data manifest and usage terms, split hashes, seeds, hardware/runtime settings, training budget, metrics/intervals, and failure examples. Keep sensitive user inputs and restricted datasets out of public artifacts.

## 2. Preference learning and the existing demo

### P01 — Mathematical correctness and edge cases [first]

**Hypothesis:** The documented bounded optimization can be implemented correctly and distinguished from aesthetic heuristics.

**Test:** Compare current and corrected `idealZ()` and `transform()` against analytic low-dimensional cases and a trusted bounded optimizer. Use zero/tiny/mixed-sign weights, boundary inputs, already-met targets, and unattainable target probabilities. Reproduce small-batch reference-code failures, undo-to-zero behavior, and unbounded sampling loops.

**Measure:** Objective gap, target feasibility/residual, distance from the reference solution, bounds violations, stale model state, and bounded termination. For minimum change use the stated norm and an at-least-target constraint; unconstrained closed form is only a baseline.

**Gate/artifact:** Deterministic cases pass documented numerical tolerances; infeasibility is explicit; heuristic outputs are labeled. Publish shared fixtures and a short derivation. No preference or performance conclusion follows from this test alone.

### P02 — Does the classifier learn preferences on unseen samples? [first]

**Hypothesis:** A small model learns a useful ranking from a limited number of ratings.

**Test:** Start with synthetic users whose true utility is known: linear, nonlinear, multiple preferred regions, noisy, and inconsistent. Compare latent logistic regression with constant/majority, random, nearest-neighbor, and a small nonlinear model. Follow with consenting human pilots on held-out parametric outputs.

**Measure:** Pairwise ranking accuracy, balanced classification metrics where relevant, Brier/log loss, calibration, utility/regret against the known synthetic oracle, and learning curves after 5/10/20/50/100 ratings. Report synthetic and human results separately.

**Gate/artifact:** Improvement over simple baselines on unseen samples with intervals; show where the linear model fails. Public demo displays training fit and unseen/prequential evaluation separately. Human preferences define the human result; the model's own score does not.

### P03 — Active sampling, labeling efficiency, and user experience

**Hypothesis:** Mixed sampling learns useful preferences with fewer ratings without making the rating experience worse.

**Test:** Compare uniform random, uncertainty, diversity, greedy preference, and the current 5/4/3 mixture at equal label and candidate budgets. Use reproducible shuffles and a common independent evaluation distribution. In a human pilot counterbalance order and use separate sessions/models.

**Measure:** Ratings needed to reach a specified held-out quality; selection/retraining time; oracle regret; duplicate coverage; human completion/dropout, rating time, and enjoyment separately.

**Gate/artifact:** State exactly which policy wins for which utility/user regime. Do not infer satisfaction from classifier confidence. Replace unsupported “efficiency” percentages with these defined measurements. Depends on P01/P02.

### P04 — Multiple preference modes and diverse outputs

**Hypothesis:** A nonlinear or mixture model captures disconnected preferences that a single linear classifier cannot.

**Test:** Known two-/three-mode utilities with deliberately disliked regions between them. Compare linear logistic, a small MLP, kernel-based preference learning, and mixture/context-conditioned heads. Compare random restarts and randomized coordinate search without treating them as a different model class.

**Measure:** Modes recovered, false preferences between modes, held-out quality, diverse high-utility samples, labels/compute required, and stability across seeds. Separate latent distance from perceptual diversity.

**Gate/artifact:** Synthetic geometry and empirical results agree; publish a demo of the failure as well as the remedy. No claim that random search creates additional modes in a linear model. Depends on P02.

### P05 — Does maximizing predicted preference improve the output?

**Hypothesis:** Optimizing the preference model produces outputs people prefer, within an acceptable generator region.

**Test:** Compare unconstrained/bounded maximization, prior-regularized optimization, truncation, best-of-N reranking, and random generation at matched total generation budgets. Use frozen learned models and blinded new ratings. For synthetic users keep the oracle separate from the learned score.

**Measure:** Human/oracle utility, predicted-versus-observed gap at extrema, realism/validity, diversity, and end-to-end cost. Test deliberate exploitation of model blind spots.

**Gate/artifact:** Demonstrated gain survives independent evaluation; report when a lower predicted score produces better perceived output. Do not call a score of 1 a guaranteed perfect preference. Depends on P02; extend with P08.

### P06 — Minimal-change transformations

**Hypothesis:** A constrained edit can improve preference while preserving the aspects of an input that should remain unchanged.

**Test:** Correct bounded L2 solver versus clipping the unconstrained step, gradient search, and random matched-size edits. Include protected dimensions, impossible targets, and already-met targets. On real outputs compare latent distance with perceptual/task-specific distance.

**Measure:** Feasibility, target attainment, mathematical distance, protected-feature preservation, perceived change, and blinded preference gain.

**Gate/artifact:** P01 establishes numerical correctness; independent judgments establish whether the edit is meaningful. Do not equate minimum latent distance with minimum visual change. Publish paired examples and failure states. Depends on P01/P05.

### P07 — Changing preferences and context

**Hypothesis:** Context-aware or time-weighted heads adapt without destroying useful earlier preferences.

**Test:** Synthetic preference switches and recurring contexts, then separate human sessions. Compare static fit, sliding windows, decay, explicit context-conditioned heads, and separate profiles. Separate context supplied by the user from context inferred by the model.

**Measure:** Adaptation delay, old-context retention, label cost, prequential quality, and whether temporary fatigue or ambiguous skips produce inappropriate permanent updates.

**Gate/artifact:** Report the stability/adaptation trade-off and recovery after a switch. Depends on P02/P04.

### P08 — A real generator and honest scaling

**Hypothesis:** The preference loop remains useful outside the hand-built renderer at realistic dimensions and costs.

**Test:** Start with one accessible, appropriately licensed generator with a defined controllable representation; the historical StyleGAN route is a candidate, not a presumed maintained dependency. Compare latent-only fitting with feature-based fitting and reranking. Measure 16/128/512 dimensions where meaningful, several history sizes, and cold/warm startup.

**Measure:** Preference quality, prior validity, encoder/generator cost, per-rating training/selection/rendering distributions, memory, and CPU/GPU/device differences. A synthetic 512-D timing test is not a real 512-D generator integration.

**Gate/artifact:** Reproducible setup plus independently evaluated outputs and a complete cost breakdown. Scope any millisecond statement to the operation and hardware measured. Depends on P01/P02/P05.

### P09 — Portable preferences and model replacement

**Hypothesis:** A shared representation or learned adapter transfers useful preferences between generators without relearning everything.

**Test:** Compare zero-shot head reuse, a learned alignment adapter, shared perceptual embeddings, and retraining from scratch. Separate same-domain generator replacement from cross-domain transfer such as art to music. Hold out users and generator/model families where feasible.

**Measure:** Quality retained, additional labels, adapter cost, calibration drift, and negative transfer.

**Gate/artifact:** Claim portability only for tested pairs and tasks. Failure of raw weight transfer is an expected possible result, not evidence that all personalized transfer is impossible. Depends on P08; broader domains depend on their pilots.

### P10 — Preference versus independent constraints

**Hypothesis:** Explicit validity/policy constraints outperform relying solely on negative preference labels.

**Test:** In a benign synthetic domain, make a region invalid regardless of user taste. Compare negative labels alone, a separate constraint head/filter, and constrained search. Include optimized boundary cases and shifted inputs.

**Measure:** Constraint violations, valid-output utility, false rejections, and coverage. Use held-out constraints and an independent validator where available.

**Gate/artifact:** Document a measured violation rate and residual failure cases. Do not extrapolate a toy constraint test into a general safety guarantee. Depends on P05.

### P11 — What privacy does encoding actually provide?

**Hypothesis:** A specific storage/matching design can reduce a specified exposure while retaining useful matching quality.

**Test:** Use synthetic or explicitly permitted inputs. Define separately an observer of stored embeddings, a matching-service operator, and another user receiving scores. Test reconstruction/identification and score-query leakage; compare local-only scoring, ordinary embeddings, and specified noise/protection mechanisms. Audit whether source buffers/files remain after the claimed deletion step.

**Measure:** Attack success under the stated access/budget, matching degradation, metadata exposure, and operational costs. If formal differential privacy is proposed, specify sensitivity, clipping, mechanism, privacy accounting, and composition; arbitrary noise is not such a guarantee.

**Gate/artifact:** Publish the threat model and measured limitations. Replace absolute privacy claims before this experiment, not after. Depends on an explicit data flow; real-person data collection is a separate activity.

### P12 — Music pilot

**Hypothesis:** Rating short generated examples improves musical preferences beyond selecting the best of the same candidate pool.

**Test:** One actual controllable music generator; random, reranked, and preference-guided candidates with equal duration and generation budget. Blind evaluation across held-out listeners; add separate continuity tests for playlists.

**Measure/gate:** Pairwise preference, diversity, continuity, artifact rate, labels, and generation latency. Advance only if an independent preference gain survives the matched-budget baseline. Depends on P08, with a music-specific representation.

### P13 — Generative feed pilot

**Hypothesis:** Personalized generation adds value beyond ranking existing generated material.

**Test:** Synthetic/offline simulation first; later an explicitly consented pilot. Compare random, reranked pool, and personalized generation at the same production cost. Separate explicit satisfaction from watch time, skips, novelty, and repeat behavior.

**Measure/gate:** Satisfaction, diversity, wasted generations, cost per positively rated item, adaptation, and fatigue. Simulation cannot establish real retention or business impact. Depends on P03/P07/P08.

### P14 — Mutual matching pilot

**Hypothesis:** Two-sided preference scoring predicts mutual visual preferences better than a generic similarity/ranking baseline.

**Test:** Synthetic profiles first; any later consenting-user study uses separate training and evaluation images. Compare one-sided, mutual, and generic ranking; evaluate exposure separately through P11.

**Measure/gate:** Mutual preference precision/recall, coverage across users, labels needed, and calibration. Describe the result as visual preference, not relationship compatibility. Depends on P02/P11.

### P15 — Constrained scientific-design pilot

**Hypothesis:** Preference feedback helps optimize trade-offs while independent hard constraints remain satisfied.

**Test:** Begin with a cheap synthetic engineering/design task with known objectives. Compare random search, scalarized/multi-objective optimization, and preference-guided search under equal evaluation budgets. A molecular extension requires an actual generator and independently validated property checks.

**Measure/gate:** Feasible solution rate, Pareto regret, expert labeling effort, and evaluation cost. A favorable toy result supports the method on that task, not a drug-discovery outcome. Depends on P06/P10.

## 3. Direct decisions and adaptive computation

### D01 — Full-depth enums without generated text [first model experiment]

**Hypothesis:** A small task head can preserve instruction-conditioned decision quality and eliminate unnecessary autoregressive output work.

**Test:** Select one small Qwen-class text model after hardware inventory. Feed rules plus relevant conversation and a final readout position. Compare minimal constrained label tokens, short JSON, frozen-backbone linear/MLP heads, and limited backbone adaptation if needed. Include a simple lexical/classifier baseline for the easy cases.

**Measure:** Human-labeled task quality, per-class errors, invalid-format rate, calibration, latency breakdown, and training budget. Include cases where the same final message has different labels under different context or rules.

**Gate/artifact:** Meet the registered quality margin and report actual timing. Returning a tensor/enum is a format result; any speed benefit must be measured independently. Maintain the original generative path for controlled comparisons.

### D02 — Is the answer accessible at earlier layers?

**Hypothesis:** Some tasks have a useful readout before the final layer.

**Test:** Cache states offline and train independent probes at a few depth fractions, initially 25/50/75/100%. Compare linear and small nonlinear probes, label supervision and teacher distillation, frozen representations and auxiliary early-exit training. Use only development data to select checkpoints.

**Measure:** Quality by depth and task slice, disagreement with final predictions, calibrated correctness, representation-extraction/training cost, and loss of full-depth performance after adaptation.

**Gate/artifact:** Publish depth-quality curves. Stop the early-exit track if no earlier head is useful. Full-state collection is a probe experiment, not an inference speedup. Depends on D01.

### D03 — Learn when to exit

**Hypothesis:** A trained gate improves the quality/cost trade-off over fixed depth and simple confidence thresholds.

**Test:** Compare fixed-depth outputs, probability/entropy thresholds, adjacent-head agreement, and a learned correctness/utility gate. Keep task labels separate from `IGNORE/RESOLVED/CONTINUE/ESCALATE`. Include correlated confidently wrong heads and confidently wrong final outputs.

**Measure:** Risk versus coverage, exit-depth distribution, false early acceptance, incorrect ignores, unnecessary continuation/escalation, and final task quality. Use an oracle gate only as an explicitly labeled upper bound.

**Gate/artifact:** Thresholds selected on calibration data satisfy registered test limits at useful coverage. A high softmax value or two agreeing heads is not proof of correctness. Depends on D02.

### D04 — Changing rules, full context, and unfamiliar inputs

**Hypothesis:** Adaptive decisions remain useful when the task changes through the prompt rather than retraining.

**Test:** Hold out rule families, paraphrases, topics, conversation lengths, languages where supported, and context-dependent replies such as “same here.” Construct matched inputs whose required decision changes under the preceding context/policy. Include missing information, conflicting instructions in quoted material, and unsupported tasks.

**Measure:** Rule-conditioned correctness, paired decision-flip accuracy, critical missed actions, calibration under shift, escalation recall/coverage, and long-context latency.

**Gate/artifact:** The system follows unseen supplied rules within its measured task envelope, or reliably exposes its limitation through abstention. If it only learns a fixed taxonomy, describe it as that narrower specialist. Depends on D01/D03.

### D05 — Do early exits actually save wall time?

**Hypothesis:** Skipping later blocks yields practical savings after gate overhead and scheduling costs.

**Test:** Instrument a runtime that can stop at checkpoints; compare full depth, fixed truncation, and adaptive exits. Count executed layers/kernels. Test batch-one and heterogeneous batches; include long prompts, growing conversations, cache reuse, cancellation, and escalated requests.

**Measure:** Entire-request latency distributions, throughput, peak memory, transfer/synchronization cost, exit fraction, and correctness after cache/resume paths. Later processing must not silently assume missing deeper-layer cache states exist.

**Gate/artifact:** Meet registered quality and latency gates with actual skipped work. Fewer layers with worse wall time is a negative performance result. Depends on D03/D04.

### D06 — Separate-model cascade versus shared-model continuation

**Hypothesis:** A cheap first stage plus fallback can beat running the expensive model for every input.

**Test:** Compare always-small, always-large, small-to-large text/input cascade, and internal early-exit continuation. Preserve full instructions/context in fallback. Separately test any learned activation adapter against reprocessing the original input; never assume unrelated hidden states or caches are compatible.

**Measure:** Combined quality, routing errors, escalation prevalence, full cost of both stages, information lost through compression/adaptation, training overhead, and tail latency. Report explanation-free decisions and required explanations separately.

**Gate/artifact:** End-to-end advantage at matched quality; an adapter is retained only if it beats the ordinary cascade including its training and transfer costs. No “95% savings” claim from parameter counts. Depends on D01/D04/D05.

### D07 — CPU/GPU/NPU placement and quantization

**Hypothesis:** Heterogeneous placement helps the actual workload more than keeping the chosen model on one processor.

**Test:** Inventory supported runtimes first. Compare all-CPU, all-GPU, and available split/NPU configurations at matched outputs and quantization where possible. Test a small standalone first stage as well as partitioning one model. Include cold start, residency, concurrency, and battery/power conditions if measurable.

**Measure:** Latency/throughput, activation transfers, model loading, memory, energy with instrumented methodology, accuracy/calibration changes, and deployment complexity.

**Gate/artifact:** Hardware-specific reproducible gains; recalibrate gates after quantization/runtime changes. Unsupported hardware stays untested. Depends on D05/D06; no hardware purchase is assumed.

### D08 — Multimodal context and reviewer explanations

**Hypothesis:** Audio/visual context resolves errors that a text-only shortcut misses, and the fallback can supply a useful explanation when required.

**Test:** Start with paired prerecorded, permitted fixtures: transcript alone, transcript plus conversation, and synchronized audio/visual context where the model supports it. Compare shortcut routing with always-multimodal review. Require a reviewer to judge correctness rather than simply justify the first-stage result; blind it to that result in an ablation.

**Measure/gate:** Context-dependent accuracy, missed escalation, synchronization/encoding cost, explanation correctness against known facts, and total latency. Report always-explain and explain-on-escalation as different products. Advancement requires a measured benefit over text-only and honest end-to-end cost. Depends on D04/D06.

## 4. Direct coordinates and GUI actions

### C01 — Full-depth coordinate/action heads [first model experiment]

**Hypothesis:** Direct numerical or spatial outputs can replace coordinate text generation at comparable grounding quality.

**Test:** Begin with a released small GUI-Actor/Qwen-VL setup as a reproducible reference, then compare on the same selected backbone: minimal coordinate text, short JSON, two-number regression, region/patch selection, and region plus local offset. Include a no-target outcome. Train frozen heads first, then measure the benefit/cost of limited adaptation. Distinguish a one-token generation wrapper from a direct forward call that skips vocabulary projection.

**Data:** Licensed ScreenSpot-family evaluation where applicable plus a fresh held-out set of own/allowed interfaces. Split by app/template and underlying screenshot. Supply a target instruction unless the task is deliberately fixed.

**Measure:** Click inside valid target region, target/no-target precision/recall, wrong-action rate, ambiguity handling, coordinate validity, image-encoding/input/head/output/postprocessing time, and memory.

**Gate/artifact:** Meet registered quality limits; report exact operation removed and actual latency reduction. A single forward pass still includes vision and transformer computation. No claim of zero inference cost. Depends on the common protocol, independently of D01.

### C02 — Grounding robustness and coordinate mapping

**Hypothesis:** Useful grounding survives layout/resolution changes and abstains when no valid click exists.

**Test:** Held-out websites/apps, small targets, repeated icons, multiple acceptable targets, overlays, disabled controls, absent targets, zoom, viewport changes, crops, padding/letterboxing, and device pixel ratios. Document screenshot-pixel versus CSS-pixel coordinates and every resize transform. Test matched screenshots with different requested targets.

**Measure/gate:** Target hit rate and wrong-click/abstention rates by slice, position-mapping error, and calibration. An average coordinate between two valid targets counts as a miss if it is not itself valid. Choose a valid point inside a selected region. Depends on C01.

### C03 — Early exits for coordinates

**Hypothesis:** Easy grounding tasks can exit at earlier transformer checkpoints with useful savings.

**Test:** Train coordinate/region heads at selected depths. Compare full depth, fixed depth, and gates based on region confidence, spatial agreement, and learned success prediction. Test partial versus complete visual feature injection for the specific backbone instead of assuming every layer has equivalent image information.

**Measure/gate:** Risk/coverage, localization error, missed small targets, actual blocks skipped, complete image-to-click latency, and per-depth quality. Include image-encoder cost; stop if it dominates so strongly that adaptive depth adds little value. Depends on C01/C02 and D03/D05 methods.

### C04 — Does it help complete tasks?

**Hypothesis:** Better/faster grounding improves usable action sequences, not just isolated coordinate scores.

**Test:** A local sandbox of benign click tasks with deterministic reset and instrumented target regions. Hold planner/instructions constant while comparing grounding methods; test action-head-only and separate-planner modes. Use drag/scroll/type only as separately labeled later tasks. Include stale screenshots and UI changes between perception and action.

**Measure/gate:** Task completion, wrong actions, retries, fallback, and total task time at the same success level. A click-coordinate benchmark alone cannot establish general agent competence. No live consequential account actions. Depends on C02; compare C03 when available.

## 5. Combined and maintenance experiments

### X01 — Personalized decisions/actions without overriding task correctness

**Hypothesis:** A small personal head can choose among multiple valid outcomes while the shared model handles semantics and the exit head controls effort.

**Test:** A local design/selection task with several valid targets and known personal preferences. Compare global, separately personalized, and combined heads; ablate the preference, direct-action, and exit components independently. Include changed instructions that override past preference and users with multiple preference modes.

**Measure/gate:** Task validity, personal utility, labels needed, calibration, and full latency. Personal preference must not convert an invalid action into an accepted one. Claim benefit only beyond the best individual baseline. Depends on P04/P07, D04, and C04.

### X02 — Reproducibility, model upgrades, and deployment economics

**Hypothesis:** A useful method survives reasonable model/runtime updates at an acceptable maintenance cost.

**Test:** Repeat selected positive results on a second model size/family or newer pinned revision and, if available, a second machine. Compare reusing, recalibrating, retraining, and adapting heads. Include head-training time, labels, explanation/fallback rates, cold starts, and realistic request volume in deployment cost accounting.

**Measure/gate:** Reproduced quality/cost curves, head portability, regression rate, adaptation expense, and documented failure boundaries. Publish hardware/model-specific results instead of a blanket claim. Depends on whichever tracks produce promising results.

## 6. Execution order and stop conditions

1. **Local correctness:** P01, claim ledger, documentation corrections, and reference setup. No model downloads are needed.
2. **Cheap learning evidence:** P02–P04 plus P06 synthetic cases. Establish whether the existing preference story holds and where it fails.
3. **Full-depth direct outputs:** D01 and C01 independently. Use available hardware and small development sets; establish trustworthy baselines before optimizing.
4. **Adaptive depth:** D02–D05 and C02/C03. Stop adding exits if useful early representations, calibration, or wall-time gains fail to appear.
5. **Real-world scope:** P05/P07/P08, C04, and D06. Keep generator and human-validation costs visible.
6. **Conditional extensions:** P09–P15, D07/D08, X01/X02 based on evidence and user priorities. Hardware integration and cross-domain portability are not prerequisites for a useful first result.

Before a large training run, use the pilot to estimate memory, elapsed time, and cost and set an explicit budget. If only frozen-head training is needed, do not default to full fine-tuning. If a baseline already meets the use case cheaply, report that rather than manufacturing a need for added architecture.

Further ideas from the archived future-explorations page can be mapped into these experiments: Bayesian/evolutionary search into P05/P06; caching/batching into P08/D05/D07; collaborative preference initialization into P09/P11; explanation of preference edits into P06; learned inverse proposals into P05. Each requires a matched-budget baseline and a separate ablation before receiving its own public claim.

## 7. Primary references and provenance

The shared conversation motivates dynamic-context handling, direct outputs, and hardware-aware cascades. Its performance/market assertions are hypotheses to verify, not benchmark evidence: [Steve's shared discussion](https://chatgpt.com/share/6aa02459-99bc-83e9-bc85-a8c2874a0558).

FastBERT studies intermediate classifiers and adaptive inference; use it as prior art for D02/D03, not evidence of a particular Qwen speedup. [FastBERT paper](https://arxiv.org/abs/2004.02178).

CALM studies confidence-based adaptive depth in generation, including challenges from per-token exits and missing representations. Its generation setting differs from a terminal enum/coordinate result. [CALM paper](https://arxiv.org/abs/2207.07061).

LayerSkip studies training for early exits and self-speculative decoding. It is a comparison point for auxiliary training and runtime design; adding arbitrary probes to an unchanged model does not establish the same result. [LayerSkip paper](https://arxiv.org/abs/2404.16710).

GUI-Actor supplies an attention-based action-head reference over Qwen vision-language backbones, including small checkpoints. Its released placeholder inference still invokes a one-token generation call before reading action-head outputs, so C01 explicitly distinguishes that from a direct-forward implementation. [GUI-Actor repository](https://github.com/microsoft/GUI-Actor), [paper](https://arxiv.org/abs/2506.03143), [inference code](https://github.com/microsoft/GUI-Actor/blob/main/src/gui_actor/inference.py).

Recheck dataset terms, checkpoints, evaluation implementations, and runtime compatibility at experiment setup. Pin the versions actually tested in each run manifest.
