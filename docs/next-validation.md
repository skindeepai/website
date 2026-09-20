# Validation completed and remaining limits

The follow-up now includes fresh moderation and search evaluation, a real shared-computation browser demo, and a selective quantization diagnostic. Complete protocols, negative results and audits remain linked from the focused pages. Fresh test outcomes are not reused to select replacements.

## 1. Shared computation

Completed: [500 unused messages](compact-next.md), six training candidates across three matched seeds, a learned risk gate, a confidence control and the frozen Qwen reference. The original shared BERT rule preserves all full-depth decisions while proposing 400 layer-2 exits. The selected teacher-trained model reduces false blocks but increases toxic misses. The learned gate stops fewer cases than confidence alone.

The sample contains only 18 toxic messages and represents the eligible annotated remainder, not natural deployment prevalence. Next quality validation needs enough independently sourced toxic cases to constrain rare errors, plus a separately reported realistic benign/toxic mix. Do not promote the best-looking test seed without fresh confirmation.

An independently pretrained BERT cannot hand its hidden states directly to Qwen: tokenization, dimensions, attention conventions and representations differ. A learned connector would require its own training and evaluation and may erase the savings. First explore a longer model initialized with the same trained prefix, with later layers trained to continue from it. Do not advertise unrelated-model fusion before actual handoff tests work.

## 2. Browser deployment

Completed: [shared browser execution](shared-browser.md) passes hidden states and masks from a prefix graph to an optional suffix. All 100 reference decisions and exit depths match; the full-depth timing control is one unsplit graph. Typed messages download 44.5 MB; the benchmark additionally loads its 44.5 MB control. Invalid input downloads no model. Network, parity and mobile/desktop checks passed.

A separate Qwen head-only export should compare the trained task head with the same backbone and inputs using a vocabulary output. The present Qwen browser demo reads existing vocabulary scores; the BERT demo uses trained task outputs. Their different datasets and training prevent attributing their entire latency difference to removal of decoding.

## 3. Search

Completed: [the remaining 200 SciFact test IDs](search-next.md), including a two-layer cross-encoder over the keyword top 20 and a development-selected optional reranking rule. Frozen fusion returns a relevant first result on 119 queries versus 105 for keywords. One query duplicates a training claim; excluding it preserves the totals out of 199. The small second model does not outperform fusion here.

All paths include query preparation and required encoding/ranking; index construction is separate. A new search domain, document-family separation and task-specific hard-negative training remain distinct follow-ups. A scientific evidence retrieval benchmark is not a test of whether a paper's claims are true.

## 4. Images and actions

The [new visual study](visual-next.md) uses fresh OSWorld-G screenshots and author-labelled infeasible requests, plus a prespecified second crop around the predicted location. Infeasible does not always mean a control is absent. This is a benchmark/domain holdout; strict application-family disjointness is not established. Coordinate early exit and a live browser screenshot model remain separate work. The live arrow demo is still synthetic recognition; the real-screen gallery remains a labelled replay.

Completed outcome: connected-region selection locates 8/25 present targets. The cutoff withholds all 25 infeasible requests, but permits only four clicks on present requests, two of them wrong. Cropping improves 2/10 to 3/10 at the cost of a second full pass. Better feasibility training and useful accepted-action coverage are still needed; abstaining on almost everything is not solved GUI control.

## 5. Quantization

Completed: [six selective INT8 diagnostics](quant-localize.md). Attention-only conversion is less damaging than converting feed-forward projections, but it changes four answers on 32 development messages. Per-channel conversion did not rescue the full conversion. No quantized model is promoted without fresh error validation and measured runtime.

## 6. Practical task baselines

- **Names/redaction:** separate gold annotation work from the inference timer; preserve full documents and evaluate complete entity misses. Move beyond PERSON spans before describing the result as anonymization.
- **Receipts:** test authentic OCR output, not only human transcription. Keep OCR and extraction costs separate and hold out merchant/template families. Verify source annotations and score missing references explicitly.
- **Request routing:** expand UNKNOWN examples and test new unsupported categories. Choose a handoff policy using the cost of mistaken action and unnecessary rejection; headline overall accuracy hides that tradeoff.

The practical extensions above remain subsequent experiments, not completed features or promises of lossless acceleration. Each measured method has a focused result page, with full sources and failures linked as references.
