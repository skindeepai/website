# Managing the context on a long task

Part of [Working past the context limit](../context.html). Results: [comparison table](../context-results.html#comparison). Protocols: [L03](../EXPERIMENTS.md#l03--let-the-model-edit-its-own-context) and [L04](../EXPERIMENTS.md#l04--keep-the-working-state-outside-the-context).

**Task.** A running ledger: 20 batches of counter updates, 121,000 tokens in all, with overwrites, deletes and distracting memo lines. The model fetches each batch with a `next` command and cannot fetch it again. At the end it reports the current value of 24 counters. Graded by exact value. The model has a shell. "Files not allowed" means a grader checks that no stream data was written to disk; piping through a script is allowed.

**Strategies compared.** Keep everything in the full window with the exact cache on; the paper's self-editing agent ("Context Language Models", arXiv 2609.37725) at a 32K budget, where the model rewrites a file that mirrors its own transcript; summarise when 75% of a 32K budget is used; the same with files allowed; and keep everything with the model's earlier thinking dropped from each call.

**How each value was lost.** Every transcript was read call by call. The paper's agent lost five values; its answers equal the true ledger with batches 2, 3, 4, 5 and 7 left out exactly. Those batches were delivered and then removed by the harness's rollback-and-retry when the context ran over. The overflow came from thinking: the first call wrote 14K tokens of planning, and the chat template re-sends earlier thinking on every call. Once settled, the model kept a state table between its own markers, folded each batch in by a short script and deleted the batch, about 650 tokens per batch, losing nothing.

**Improved agent (running).** Keeps the paper's idea and changes the harness: a delivered item is never rolled back, a room check runs before fetching, earlier thinking is dropped, and a small state block that the harness checks is shown last so the transcript stays append-only and the exact cache keeps working. Pass rule: two seeds within one value of keeping everything, with no delivered item lost.

## Files

With files allowed, the model wrote each batch's effect to disk and kept its context under 9K, with or without self-editing. This is the plan-file pattern: state outside the context, rewritten as work proceeds, raw data in other files. An earlier run on a 140K variant showed the one risk: a shell habit (`tee` piped into `head`) cut files short and lost values while saving, not in the context.

**Limits.** One task family, one seed per strategy, one machine. The streams larger than the window are running.

## Raw data and lab notes

- [The comparison rows, including the running ones](../results/long-context/result.json)
- [Call-by-call reading of every run, and the improved agent](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-self-editing-first-comparison.md)
- [Paper review: mechanism, baselines and what applies here](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-context-research-review.md)
- [Task generators, harness and runners](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/scripts/context/)
