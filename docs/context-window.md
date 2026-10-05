# Opening the whole context window

Part of [Working past the context limit](../context.html). Results: [speed](../context-results.html#window) and [recall](../context-results.html#recall). Protocol: [L01](../EXPERIMENTS.md#l01--open-the-whole-context-window-exactly).

**Setup.** Qwen3.8-27B (FP8 weights, 16-bit cache) on two Intel Arc Pro B70 cards, the two cards splitting the model. The server is started with the model's full 262,144-token window; nothing is quantized beyond the published FP8 weights. Drafting (the model's own built-in draft head, every drafted token checked by the model) is on unless stated.

**Speed test.** One prompt per length from 8K to 250K tokens, sent in chat form. Recorded: tokens read per second, the wait before the first word (cold, then again with the exact prefix cache), and tokens written per second. Each length was also answered with drafting off; the answers were identical at every length.

**Correction kept on record.** A first run reported empty answers above 212K. That came from sending the prompt as bare text instead of chat form; in chat form the model answers at 213K and 250K.

**Recall test.** Each prompt holds thousands of records that differ only in their number (up to 11,879 at 250K). Ten questions per prompt ask for six codes each, spread across the start, middle and end. Two kinds of filler: ordinary words around the records, and a wall of look-alike records. 720 codes in all. A miss is scored as wrong whatever it returned; every miss turned out to be another record's real code.

**Limits.** One server, one prompt per length, single-user timings. These are first looks, not published package numbers.

## Raw data and lab notes

- [Every number on the results page](../results/long-context/result.json)
- [Recall summaries, ordinary words](../results/long-context/recall-ordinary-words.log) · [wall of codes](../results/long-context/recall-wall-of-codes.log) (one line per length; codes right by third of the context)
- [Plan, written before each run, and results](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-context-window-prereg.md)
- [Full run data: speed probe, recall and edge tests](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/data/2026-10-05-context/)
- [Plain-words summary of the night's work](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-context-research-results.md)
