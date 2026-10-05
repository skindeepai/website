# Managing a long task's working context

Start with [files and retrieval](../context-files.html) or [CLM self-editing](../context-clm.html). [Full task comparison](../context-results.html#comparison).

**Task:** 20 batches, approximately 121K input tokens, about 160 counters with SET/ADD/DEL operations and irrelevant memos. Each batch can be fetched only once. The final query requests 24 current values. Seed 0; exact-value grading. The model has a shell, and code performs bookkeeping.

With external data files forbidden, the large-window baseline returned 24/24 in about 26 minutes, periodic summaries 24/24 in 41 minutes, and the original CLM agent 19/24 in 64 minutes. The large-window model filtered batches through code rather than retaining all raw memos.

The five CLM errors match a ledger missing delivered batches 2, 3, 4, 5 and 7. Harness rollback removed them after overflow. The transcript analysis did not attribute these misses to the model's state edits.

## Files

The file-using CLM and plain agents both returned 24/24 in about 1.9 minutes, with peak active contexts of 8.2K and 8.9K. They generated about 7K and 8K tokens, versus 89K in the large-window baseline and 211K with summaries. These totals do not isolate file I/O cost or per-token acceleration.

Saving original records intact can preserve their bytes. A compact current-state file instead preserves what this ledger needs, not necessarily the history behind it. Later arbitrary-detail questions require the original records or another adequate representation.

The earlier key-value trials all offloaded into files. Their losses came from truncated shell-pipeline writes, so their score differences do not rank context policies.

## Follow-up status

The revised agent protects delivered batches, checks room before reading and keeps explicit state. The cited snapshot includes code and stub checks, but no completed score for it or the proposed 480K stream. The completed 121K task exceeds the small 32K budget, not the model's full 262K limit.

## Sources

- [Reported results](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/notes/2026-10-05-context-research-results.md) / [Call-by-call analysis and revised design](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/notes/2026-10-05-self-editing-first-comparison.md)
- [Task rules and runners](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/scripts/context/README.md)
