# Long-context speed and recall

Start with [the overview](../context.html), then [the speed and recall tables](../context-results.html#window).

Qwen3.8-27B used FP8 weights and a 16-bit attention cache on two Arc Pro B70 GPUs. The server accepted a 262,144-position limit, with chat probes supplying up to about 250K input tokens and leaving room for output.

**Generation speed:** the 8K and 30K points each use one uncached chat recall reply. The 60K–250K points use the median of 20 six-code replies per length, split between ordinary-word filler and a wall of codes, with exact prefix caching enabled. These are observed prompt tests, not a controlled comparison of two optimizations. They range from approximately 127 tokens/s at 8K to 27 at 250K.

**Initial probe:** the original bare-text trial supplies the cold reading speeds. Its 250K response ended immediately with drafting both on and off. Later chat-format tests answered at 213.5K and 250K. Matching the original empty outputs does not prove successful long-context recall.

**Recall:** two styles × six lengths × ten questions × six codes = 720 lookups; 708 correct. All 120 lookups at 60K were correct. Every observed miss was another record's actual code. The preset continuous-range rule required at least 59/60 in both styles at each tested length and all shorter ones; only 60K met it. This is not a universal model limit.

## Original measurements

- [Protocol, corrections and results](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/notes/2026-10-05-context-window-prereg.md)
- [Initial cold probe](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/data/2026-10-05-context/longctx/probe-mtp5.json) / [Chat-format short points](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/data/2026-10-05-context/edge/chat-recall.json)
- [Recall: ordinary words](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/data/2026-10-05-context/recall/recall-prose.json) / [Recall: wall of codes](https://github.com/steveseguin/b70-optimization-lab/blob/83a71180e3abf4eec7f57181fc1433cedc3cdd26/experiments/qwen38-27b-b70/data/2026-10-05-context/recall/recall-ledger.json)
- [Aggregation rules and values](../results/long-context/result.json)
