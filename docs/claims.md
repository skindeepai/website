# Claim and evidence ledger

Updated 2026-09-19. Current pages describe this checkout. Archived pages and source documents retain historical assertions and are not current validation.

| Statement | Evidence and status | Limit |
| --- | --- | --- |
| Browser fits preferences locally | Browser source and regression checks | Procedural renderer, no pretrained generator |
| Box optimum and bounded L2 edit work on tested cases | `results/synthetic/math-fixtures.json`, independent SciPy comparison | Numerical geometry, not perceived quality |
| Linear head learns linear synthetic utility | Five seeded held-out pilots | No human study |
| Linear head misses disconnected preferences | P04 linear versus quadratic-feature pilot | One chosen synthetic utility |
| Maximizing predicted preference can harm true utility | P05 independent synthetic oracle | Unequal candidate budgets; failure illustration, not efficiency comparison |
| Active sampling reduces required ratings | Open hypothesis; fixed-budget pilot only | Browser 5/4/3 policy and human experience not established |
| Qwen can return a direct enum | Frozen 0.5B model, trained heads, pinned fixtures | Small rule grammar; not arbitrary instructions |
| Runtime can skip later layers | Hooks count six executed layers | CPU batch one, no persistent-cache continuation |
| Early exit preserves quality | **Not supported**: 91.7% versus 97.9% full-depth accuracy | Exceeds proposed one-point tolerance |
| Pointer head returns coordinates without decoding text | GUI-Actor pretrained reproduction | Existing architecture/weights; small local fixtures |
| Missing GUI targets can be rejected | **Not implemented** in reproduced head | Always chooses a patch; needs abstention |
| Direct coordinates beat JSON at equal quality | Open hypothesis | Vocabulary-projection ablation is not JSON comparison |
| Transfer, music, feeds, matching or scientific value | Registered P04–P15 studies | Need independent data and labels |
| Encodings make private data anonymous | **Not established**; P11 pending | Reconstruction/linkage/membership require evaluation |
| Pages fit mobile and desktop widths | `results/ui/result.json` after checks | Chromium viewports; not physical Safari/iOS/Android verification |

## Provenance and prior work

- `history.html` links original app/server repositories, videos and dated filing material. A filing is not a current patent-status determination; no new legal conclusion is made here.
- Adaptive depth builds on [FastBERT](https://arxiv.org/abs/2004.02178), [CALM](https://arxiv.org/abs/2207.07061), and [LayerSkip](https://arxiv.org/abs/2404.16710).
- Pointer heads are established prior art. This checkout reproduces [Microsoft GUI-Actor](https://github.com/microsoft/GUI-Actor) using its released Qwen2-VL 2B checkpoint. The lab's question is how these mechanisms combine with reliable abstention, dynamic instructions and measured savings.
- Upstream licenses govern downloaded weights separately. No model weights are committed.

No consented human study, real-generator validation, cross-model transfer, production moderation test, end-to-end GUI task benchmark, accelerator comparison, energy measurement or privacy attack study has been run here. `research.html` assigns each an explicit next step. Partial pilots do not complete full protocols.
