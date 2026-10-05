# Context on disk: text, snapshots and active offload

For the explanation, start with [Can disk replace GPU memory?](../context-memory.html).

**Text files:** the file-using ledger agents each got 24/24 in about 1.9 minutes, with active context below 9K tokens. This is total task time, not a measurement of file-write overhead or disk-streamed attention.

**Numerical cache snapshots:** saving and restoring the complete compatible state could avoid recomputing a prompt. The active state must still fit the execution arrangement after restoration. Byte-preserving storage is not by itself a test of correct model continuation. No completed snapshot benchmark is in this research snapshot.

**Active offload:** processing cache blocks from RAM or disk during generation trades data movement or CPU work for capacity. It requires explicit engine support and working buffers. No completed active-offload benchmark is in this snapshot.

## Payload calculation

For the target model: 16 full-attention layers x 2 arrays x 4 KV heads x 256 values x 2 bytes = 65,536 bytes per token across both GPUs. At 200,000 tokens: 13.1072 GB, or 12.207 GiB, of attention KV payload. Weights, recurrent state, drafting, checkpoints, padding and temporary workspace are additional.

At a hypothetical sustained 10 GB/s, reading that entire payload once takes at least 1.31 seconds; at 5 GB/s, 2.62 seconds. These are transfer lower bounds, not measured output rates. Resident cache portions, overlapped work and multiple accepted draft tokens per pass change the calculation.

A few free kilobytes cannot hold this active state. Disk also does not enlarge the model's 262,144-position limit. KB of text and thousands of model tokens are different units.

## Sources

- [Model shapes and cache accounting](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/notes/2026-10-05-prefix-cache-reuse-rules.md)
- [Original research discussion](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/notes/2026-10-05-context-research-review.md)
- [Website calculations and assumptions](../results/long-context/result.json)
