# Direct points on public interface screenshots

Tested 30 predetermined ScreenSpot examples across Windows, macOS, iOS, Android and web interfaces. Three examples per platform/type stratum. This is a small stratified sample, not the full 1,272-example benchmark.

- Highest-probability patch center: **16/30 hits**.
- Connected-region weighted center: **25/30 hits**.

The two readouts use exactly the same probability map from one model pass. The region method follows the published GUI-Actor rule; it is not a newly trained model. No text tokens or JSON are generated. All 28 language-model blocks and the vision encoder run.

## Every result

| Source row | Platform | Type | Single patch hit | Region hit |
| --- | --- | --- | --- | --- |
| 695 | android | icon | True | True |
| 524 | android | icon | False | True |
| 716 | android | icon | True | True |
| 784 | android | text | False | False |
| 792 | android | text | True | True |
| 566 | android | text | False | True |
| 340 | ios | icon | False | True |
| 611 | ios | icon | True | True |
| 818 | ios | icon | False | True |
| 827 | ios | text | True | True |
| 356 | ios | text | True | True |
| 343 | ios | text | True | True |
| 300 | macos | icon | False | False |
| 322 | macos | icon | False | True |
| 250 | macos | icon | True | True |
| 298 | macos | text | True | True |
| 304 | macos | text | True | True |
| 308 | macos | text | False | False |
| 987 | shop | icon | True | True |
| 1254 | shop | icon | True | True |
| 1110 | tool | icon | False | False |
| 969 | shop | text | True | True |
| 1158 | tool | text | False | True |
| 1240 | shop | text | False | True |
| 1 | windows | icon | False | False |
| 0 | windows | icon | False | True |
| 20 | windows | icon | True | True |
| 102 | windows | text | True | True |
| 132 | windows | text | True | True |
| 85 | windows | text | False | True |

This run enforces at most 576 visual patches, checked after preprocessing. Smaller elements may disappear at that resolution. It does not test missing targets or prove a speed advantage over JSON. Recorded elapsed times are diagnostic because other CPU work overlapped. Unknown training overlap with this public benchmark remains possible.

## Evidence

- [Protocol](screenspot-protocol.md), [implementation](../experiments/screenspot.py), [resolution-cap correction](coordinate-correction.md).
- [Pinned sample IDs, boxes and image hashes](../results/screenspot/data-manifest.json), [all points and probabilities](../results/screenspot/predictions.json), [summary](../results/screenspot/result.json).
- [ScreenSpot dataset](https://huggingface.co/datasets/bevaya/ScreenSpot), [SeeClick authors](https://github.com/njucckevin/SeeClick), [Microsoft GUI-Actor](https://github.com/microsoft/GUI-Actor).
- Reproduce with `python experiments/screenspot.py --threads 4`. Public screenshots download to the ignored local cache; no live clicks are executed.
