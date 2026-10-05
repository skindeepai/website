# Keeping the model's cache on disk

Part of [Working past the context limit](../context.html). Results: [estimate table](../context-results.html#disk). Protocol: [L06](../EXPERIMENTS.md#l06--keep-the-models-cache-on-disk).

**Status: estimated from measurements, not built.**

There are three different uses of disk:

1. **Files the model reads and writes (measured).** The model keeps its data in files and searches them. Lossless and fast on the ledger task; see [Managing the context](context-management.md#files).
2. **Streaming the cache from disk during writing (estimated).** The model's cache is exactly 64 KiB per token for this model at 16 bits, so 200K tokens take about 13 GB (12 GiB). Every written token reads the whole cache, so a card that holds only 16K would have to pass the rest through on every step. At 12.7 GB/s from this machine's drive (measured) and roughly 9-16 GB/s uploads to the card, one step takes a little over a second: roughly 3-4 tokens a second with drafting, against about 35 with the cache in video memory, and several times slower on an ordinary drive. Reading a 200K prompt would stream about 1.5 TB in total, roughly doubling the two-minute read. The arithmetic would run in pieces, so results would be repeatable but not bit-identical to the in-memory path.
3. **Parking a whole conversation (designed, not built).** Save a finished conversation's cache, about 13 GB for 200K, and restore it in seconds instead of re-reading for two minutes. Byte-for-byte, so lossless.

Disk does not make the model's own window (262,144 tokens) any bigger. It only changes where the cache for that window is kept, and what that costs in speed.

## Sources

- [Cache size and the disk discussion](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-context-research-results.md) · [Paper review, section on disk](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-context-research-review.md)
- [Numbers used above](../results/long-context/result.json)
