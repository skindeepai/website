# A matched one-token output control

The numeric enum and constrained text paths share the same frozen Qwen representations and the same supervised output rows. Each of the 77 labels maps to a single Qwen token. The text path emits that token and converts it to text. Neither path stops early.

All **3080** saved examples produced identical decisions, with **2551** correct. This equality follows from the shared classifier and bijective token mapping; it is an invariant check, not new evidence of equal independently trained model quality.

The constrained readout computes only its 77 supported token scores. It does not run an untouched full-vocabulary language decoder or the ordinary generation API. This is an efficient custom one-token baseline, not an independently fine-tuned conversational model.

## Actual warm CPU timing

| Output | Mean input-to-result | Median |
| --- | --- | --- |
| enum | 286.94 ms | 281.33 ms |
| single_token | 288.54 ms | 284.09 ms |

Timing uses 32 predetermined queries, two alternating repeats per path, eight CPU threads and an otherwise idle launched-model workload. Includes tokenization, all 24 blocks, trained readout and token-to-text conversion where applicable. Loading is excluded.
Text minus enum mean time: **1.60 ms**; paired-query bootstrap 95% interval **-2.94 to 6.13 ms**. This describes one run, not fresh sessions or different hardware.

A single trained output token can implement the same decision. Avoiding text alone does not establish a useful speedup. Skipping transformer blocks is a separate optimization.

- [Implementation](../experiments/matched_output.py), [timing protocol](../results/matched-output/protocol.json), [results](../results/matched-output/result.json).
- [Every equivalent output](../results/matched-output/predictions.json), [all actual timings](../results/matched-output/timings.json).
