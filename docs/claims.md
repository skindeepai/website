# Claim and evidence ledger

Updated 2026-09-20. Current pages describe this checkout. Archived pages and source documents retain historical assertions and are not current validation.

| Statement | Evidence and status | Limit |
| --- | --- | --- |
| Browser fits preferences locally | Browser source and regression checks | Procedural renderer, no pretrained generator |
| Box optimum and bounded L2 edit work on tested cases | `results/synthetic/math-fixtures.json`, independent SciPy comparison | Numerical geometry, not perceived quality |
| Linear head learns linear synthetic utility | Five seeded held-out pilots | No human study |
| Linear head misses disconnected preferences | P04 linear versus quadratic-feature pilot | One chosen synthetic utility |
| Maximizing predicted preference can harm true utility | P05 independent synthetic oracle | Unequal candidate budgets; failure illustration, not efficiency comparison |
| Active sampling reduces required ratings | [Exact browser 5/4/3 sampler tested](preference-followups.md); it was not consistently better and lost to random sampling with noisy ratings | Synthetic rules and five seeds; human rating savings remain unestablished |
| Qwen can return a direct enum | Frozen 0.5B model, trained heads, pinned fixtures | Small rule grammar; not arbitrary instructions |
| Runtime can skip later layers | [Qwen checkpoint traces](chat-next-methods.md) and [shared BERT execution](shared-browser.md) verify stopping and continuation without repeating earlier blocks | Measured implementations and samples; not general answer-readiness detection |
| Original synthetic early exit preserves quality | **Not supported**: 91.7% versus 97.9% full-depth accuracy | Exceeds proposed one-point tolerance |
| Pointer head returns coordinates without decoding text | GUI-Actor pretrained reproduction | Existing architecture/weights; small local fixtures |
| Missing GUI targets can be rejected | [A separate cutoff withheld all 25 infeasible requests](visual-next.md), but also withheld 21 valid ones and accepted two wrong clicks | The pointer head itself always chooses a patch; the cutoff returns uncertainty, not a learned missing-target verdict |
| Direct coordinates beat JSON at equal quality | Open hypothesis | Vocabulary-projection ablation is not JSON comparison |
| Transfer, music, feeds, matching or scientific value | Registered P04–P15 studies | Need independent data and labels |
| Encodings make private data anonymous | **Not established**; P11 pending | Reconstruction/linkage/membership require evaluation |
| Pages fit mobile and desktop widths | `results/ui/result.json` after checks | Chromium viewports; not physical Safari/iOS/Android verification |

## Provenance and prior work

- `history.html` links original app/server repositories, videos and dated filing material. A filing is not a current patent-status determination; no new legal conclusion is made here.
- Adaptive depth builds on [FastBERT](https://arxiv.org/abs/2004.02178), [CALM](https://arxiv.org/abs/2207.07061), and [LayerSkip](https://arxiv.org/abs/2404.16710).
- Pointer heads are established prior art. This checkout reproduces [Microsoft GUI-Actor](https://github.com/microsoft/GUI-Actor) using its released Qwen2-VL 2B checkpoint. The lab's question is how these mechanisms combine with reliable abstention, dynamic instructions and measured savings.
- Upstream licenses govern downloaded weights separately. Original downloaded upstream checkpoints stay outside the repo; trained experiment weights and readouts are preserved where linked.

No consented human study, real-generator validation, cross-model transfer, production moderation test, end-to-end GUI task benchmark, accelerator comparison, energy measurement or privacy attack study has been run here. `research.html` assigns each an explicit next step. Partial pilots do not complete full protocols.

## Public-data follow-up

| Finding | Evidence | Limit |
| --- | --- | --- |
| BANKING77 adaptive candidate retains similar average accuracy while skipping about 24% of blocks | [Three-seed report](banking77-results.md), full official test split | Independent harmful-exit guard rejects all three candidates |
| Conservative gate reaches 615/741 versus 613/741 at full depth and skips 7.7% of blocks | [Fresh-reserve report](conservative-exits.md) | Calibration upper bound 1.007% exceeds the fixed 1% limit; no passing gate claim |
| Connected-region points improve 16/30 to 25/30 on public screenshots | [ScreenSpot report](screenspot-results.md) | Small predetermined sample; uses published readout, no absent-target check |
| First coordinate run had a 576-patch cap | **Incorrect**; [correction](coordinate-correction.md) | Saved grids were 936 desktop / 392 mobile. New harness asserts the cap |
| Browser can run Qwen locally | Actual saved browser run and benchmark code | Full-depth output-format comparison; not an early-exit implementation |

## Adversarial follow-ups

| Finding | Evidence | Limit |
| --- | --- | --- |
| A separate implementation executes the same real layer exits | [Fresh-process replay](reproduction.md), including a published Transformers release | Same machine and saved heads; 24 predetermined queries plus a targeted layer-18 case |
| Classification shortcuts also appear on a second dataset | [CLINC subset](clinc-results.md): 854/900 versus 856/900; projected 32.7% blocks skipped | 30 selected intents; calibration fails; 66/1,000 OOS inputs accepted early; no new runtime savings measurement |
| Adding UNKNOWN solves unfamiliar-input handling | **Not supported**: [31-output variant](clinc-unknown.md) still fails calibration | Changes supervision and seed too; 299/1,000 OOS inputs prematurely routed to known categories |
| The trained readout follows changing rules | **Not supported**: [paired stress test](changing-rules.md), full-depth only 14/100 pairs both correct | Authored rule templates on public utterances; not arbitrary instructions |
| Numeric enum output has a reliable advantage over one trained token | **Not established**: [matched output control](matched-output.md) | Shared learned rows make outputs equivalent by construction; timing interval includes zero |
| Better evidence justifies a production-ready claim | **Not supported** | [Acceptance requirements](evidence-acceptance.md); independent traffic/hardware validation and passing reliability gates remain missing |

## Practical workload follow-ups

| Finding | Evidence | Limit |
| --- | --- | --- |
| A classifier can stop at layer 12 on real archived chat | [600-message study](chat600-results.md), counted actual execution | Selected threshold is unconditional at layer 12; no input-dependent readiness is demonstrated |
| Similar aggregate chat accuracy preserves moderation quality | **Not supported**: 522/600 full-depth versus 519/600 early, but toxic misses rise from 19 to 26 | Both original and clean calibration reject the candidate; the full-depth classifier also fails its quality floor |
| Numeric output alone avoids substantial processing | Compare equally supervised enum and one-token paths in the [same workload](chat600-results.md) | Both use the same learned rows; a token label need not require a conversation or multiple decoding steps |
| Early heads navigate unseen mazes successfully | **Not supported**: [maze study](maze-actions.md), both learned policies reach 0/10 goals versus BFS 10/10 | Small synthetic fully observed grids; state accuracy does not predict successful closed-loop behavior |
| Recorded maze decisions can be inspected in the browser | [Replay](../maze-benchmark.html), all 459 recorded actions checked | Browser replays saved decisions; it does not run the model |

The [separate chat review](chat600-review.md) verifies the calculations while rejecting an equal-quality acceleration claim. A measured speed improvement cannot override failed quality checks.
## Smaller exploratory comparisons

| Finding | Evidence | Limit |
| --- | --- | --- |
| A learned gate chooses different stopping depths | [100-message comparison](chat-smoke-results.md): exits at blocks 6/12/18/24; 79 correct versus 77 at full depth | Reused balanced examples; one new false block and three corrected toxic misses; no fresh quality acceptance |
| That gate reduces executed request time | [Actual execution](chat-smoke-heads.md): 42.06 versus 63.56 summed seconds | One warm paired CPU pass, 33.84% reduction; loading excluded, text preparation included |
| More backbone training reliably improves early decisions | **Not supported** by [these short adapter recipes](chat-smoke-adaptation.md) | Distillation improves final accuracy but weakens early layers; shorter-warm-up repair worsens toxic recall |
| A small specialist can avoid some Qwen calls | [Cascade](chat-smoke-specialist.md): 51/100 messages handled without Qwen | Adds a toxic miss; different supervision budgets; no validated quality guarantee |
| More action supervision and explicit legal moves help the maze policy | [Structured-policy follow-up](maze-smoke.md): 7/10 goals | Small reused synthetic episodes; three legal loops; not Qwen or evidence of early-exit improvement |
| Confidence identifies absent click targets generally | **Not established** by [two-screenshot diagnostic](coordinate-abstention-smoke.md) | Rejects two absent targets but accepts one wrong point; repeated requests and interface |

The [adversarial audit](chat-smoke-review.md) checks artifacts and code within this session; it is not external replication.
