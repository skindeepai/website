# Reusing a context exactly: the prefix cache

Part of [Working past the context limit](../context.html). Results: [cache table](../context-results.html#cache). Protocol: [L02](../EXPERIMENTS.md#l02--reuse-an-unchanged-context-exactly).

**The problem.** Without reuse, every turn of a long conversation re-reads the whole prompt: about two minutes at 200K. The serving engine's own cache can skip that, but here it was not guaranteed to give the same answer as a cold read. It also stores pieces made while the model was writing, and with drafting it keeps the model's running-summary layers at every block, four times the memory per token.

**What was built.** A small add-on stores only what was made while reading a prompt, in fixed 832-token pieces, so a cached read runs the same arithmetic as a cold read. It can also keep the running state every 6,656 or 13,312 tokens (a memory-for-reuse dial), so an edit in the middle of a long context resumes from the last kept state before the edit instead of from the start.

**Test.** Two fresh servers with drafting on: a reference without the cache, then the cache server. Eleven prompts, nine cases each: repeat, append, edit in the middle, repeated edit, and the second and third turn of a conversation, where the shared text includes answers the server wrote. Token ids compared, up to 48 tokens per case. Result: 99 of 99 identical; a cold repeat at 120K matched the cached answer.

**Costs.** A cold read is 14-27% slower because of the fixed piece size. Each kept state costs the memory of about 2,500 tokens of context.

**Not tested yet.** Several users at once, the one-card server, a comparison at the level of scores, prompts above 30K in the edit cases.

**Rejected alternative.** The paper's "suffix cache reuse" keeps cache computed for the text before an edit; the authors describe it as an approximation of re-reading. Not exact, so not used.

## Raw data and lab notes

- [All 99 comparisons and the summary line](../results/long-context/cache-probe.log) (cached_tokens = tokens served from the cache; ttft = seconds to the first word)
- [Plan and results](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-prefix-cache-exactness-prereg.md) · [How the engine decides what it can reuse](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-prefix-cache-reuse-rules.md)
- [Add-on code](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/overlays/b70-prefix-cache-exact/) · [Run data](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/data/2026-10-05-context/prefixcache-exact-mtp/)
