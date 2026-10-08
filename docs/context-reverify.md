# The two quoted-events runs, checked again

Start with [the reading results](../context-results.html#reading) for what the quoted-events agent does.

**What happened.** In the quoted-events method the model writes down each change it reads, with the exact sentence that proves it, and a small program keeps the totals. On 2026-10-06 a review found three holes in that program's checker: a made-up change could slip past the quote check, a batch of mixed changes could be applied in the wrong order, and one script let the shell expand text it should have left alone. The fix was reviewed and applied (9 of 9 unit tests and 22 of 22 stub checks pass).

**What we re-ran.** The two headline runs were repeated on 2026-10-07 with the fixed checker, on the same task files, with Qwen3.8-27B (FP8) and the same 32K working budget.

| Run | Before the fix (2026-10-06) | After the fix (2026-10-07) |
| --- | --- | --- |
| 480K-token story stream | 10/10, 1,173 s, 578 calls, 36,702 tokens written, peak context 22,430 | **10/10**, 1,165 s, 577 calls, 36,600 tokens written, peak context 19,935 |
| Million-token story stream | 24/24, 2,874 s, 1,236 calls, 111,709 tokens written, peak context 22,815 | **24/24**, 2,867 s, 1,236 calls, 111,709 tokens written, peak context 22,815 |

The scores did not change. The million-token run made exactly the same calls and wrote exactly the same number of tokens; the 480K run differed by one call. So the checker holes did not affect the published numbers.

**Limits.**

- The repeat used a 65K server window instead of 262K, to leave the computer more free memory. The agent never holds more than about 23K tokens plus 4K of output, so it never came close to either limit. A first attempt at 262K was stopped early by a memory safety guard; it was not a graphics-card fault.
- The checker still does not prove that each change was understood correctly or that none was missed. Only the final answers are scored.
- One repeat of each run; no new speed claim. The 120K and retention runs were not repeated.

## Sources

- [Lab note: the repeat on the fixed checker](https://github.com/steveseguin/b70-optimization-lab/blob/8a46c0cae87be86002b4fb369525040c505af0b7/experiments/qwen38-27b-b70/notes/2026-10-07-context-reverify-fixed-checker.md)
- [Validated export (manifest, results, site projection)](https://github.com/steveseguin/b70-optimization-lab/tree/8a46c0cae87be86002b4fb369525040c505af0b7/experiments/qwen38-27b-b70/data/2026-10-07-context-reverify)
