# CPU text cleanup and earlier reasoning

Start with [the explanation](../context-clm.html#cleaning) or [the census table](../context-results.html#cleaning).

Three existing coding-agent transcripts contained approximately 4.99M tokens. Conservative combined rules for duplication and formatting reduced this by 1.83%. The sliding-window analysis put a 32,768-token window at roughly 33,070 raw tokens, far below 100K.

A separate rule moved the middle of outputs longer than 2,000 tokens out of view, retaining a head, tail and file pointer: 7.50% fewer visible tokens. This percentage is not additive with the combined cleanup result. In 54 of 152 large outputs, a later command quoted a distinctive hidden line. That is a proxy for possible retrieval work, not a measured count of read-back calls.

Earlier reasoning was measured separately in 277 calls from the first key-value comparison: a median 9.9% of input tokens, maximum 55.8%. A later large-window trial that removed earlier reasoning got stuck without a final answer after 2.6 hours. Preserve explicit task state before attempting this optimization.

Cleanup changes the text presented to the model even when information remains recoverable. This census does not establish identical outputs, a trained CPU classifier's quality, or a model-speed improvement.

## Sources

- [Census and method](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/notes/2026-10-05-context-hygiene-census.md) / [Reasoning-removal outcome](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/notes/2026-10-05-context-research-results.md)
- [Census implementation](https://github.com/steveseguin/b70-optimization-lab/blob/9b25b1b19d319c019344dae5884b3abf79827313/experiments/qwen38-27b-b70/scripts/context/hygiene_census.py)
