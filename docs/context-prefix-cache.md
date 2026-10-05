# Exact prefix reuse

Start with [the cache explanation](../context-cache.html). [Measured tables](../context-results.html#cache) separate first-token waiting from output rate.

The add-on retains prompt-reading states and uses fixed 832-token reading chunks. It excludes states produced during generation and retains periodic recurrent-state checkpoints, allowing an edit to resume before the changed region. Later text must still be recomputed.

**Matched comparison:** two fresh servers, both using 832-token chunks, with drafting on. Eleven prompts, nine variants each, including second and third turns; up to 30K input and 48 output tokens. All 99 output token sequences matched. Separate cold checks at 120K matched too. This was not a scores-level comparison with drafting.

**Decode check:** the standard 12-prompt gate returned 12/12 reference answers on both passes of each server. Cache off: 88.5 and 88.7 tokens/s. Exact cache on: 89.6 and 89.7. One server per configuration; the approximately 1% difference is not a reliable speedup claim.

**Costs:** cold latency increased from about 9.8 to 11.4 seconds at 30K, and about 55 to 69.6 seconds at 120K. A saved recurrent checkpoint costs roughly three 832-token attention blocks' worth of memory; drafting and allocator details affect physical allocation. Cached 200K questions started in about 2 seconds, versus 114 seconds in the separate initial cold probe.

The multi-user, one-card and disk-restore cases need separate validation. Approximate reuse of old suffix states after an edit is a different method and was not part of this cache result.

## Sources

- [Experiments and standard decode gate](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/notes/2026-10-05-prefix-cache-exactness-prereg.md) / [Detailed reuse rules](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/notes/2026-10-05-prefix-cache-reuse-rules.md)
- [99-case data](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/data/2026-10-05-context/prefixcache-exact-mtp/cache.json) / [Standard gate data](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/data/2026-10-05-context/pcgate/results.json)
- [Add-on source](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/overlays/b70-prefix-cache-exact/b70_prefix_cache_exact.py)
