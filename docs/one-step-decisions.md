# A label in one step

Part of [Return a decision](../decisions.html). Results: [one-step decisions](../one-step-results.html). Found during the [longer-context work](../context.html).

**Question.** A model that thinks before it answers spends most of a short decision on thinking. For a yes/no, a choice or a label, can it answer in one step with the same result?

**Test.** 80 easy items: yes/no, A-D choices, sentiment and routing, each with labels whose first tokens differ. Qwen3.8-27B (FP8 weights, 16-bit cache) on two Intel Arc Pro B70 cards. Three paths per item: one step with thinking off and the first token restricted to the labels; normal decoding with thinking off; and, on 24 items, decoding with thinking on. One pass.

**Why it is exact.** When the labels start with different tokens, restricting the first token and taking the most likely one gives exactly what restricted greedy decoding would give. In this test the server is asked for one token, with sampling limited to the labels' first tokens, so the reply is always a label. Reading an answer from the model's internal numbers with a trained probe would not be equivalent; this does not do that.

**Limits.** Easy items only. Not measured: hard decisions where thinking changes the answer. One machine, one pass, single-user timings. With drafting, a short label already takes one step, so the saving is the thinking, not the decoding.

## Raw data and lab notes

- [Summary record](../results/long-context/one-step-choice-summary.json)
- [Plan and result (section "One-step answers")](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-context-window-prereg.md) · [Per-item data](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/data/2026-10-05-context/edge/choice.json)
