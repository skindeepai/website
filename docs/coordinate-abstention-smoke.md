# Can low pointer confidence mean “target absent”?

On eight saved local examples, a simple confidence threshold rejects **both absent targets**, keeps all six present targets, and still accepts **one wrong coordinate**. This is a tiny retrospective diagnostic, not a trained visual presence classifier or a public absent-target benchmark.

| Held-out screenshot | Threshold learned on | Threshold | Absent rejected | Present wrongly rejected | Accepted wrong points |
| --- | --- | ---: | ---: | ---: | ---: |
| Desktop | Mobile's four examples | 0.03509 | 1/1 | 0/3 | 1 |
| Mobile | Desktop's four examples | 0.03280 | 1/1 | 0/3 | 0 |

Each fold chooses a threshold from the other screenshot's confidence values, maximizing balanced present/absent classification accuracy. No held-out labels select that fold's threshold. The two screenshots show **the same authored interface and repeat the same instructions**, including the same missing “Delete account” target. Their visual sizes differ, but this is not an independent task split. All eight outputs had already been inspected before this reanalysis.

The threshold reads the existing GUI-Actor pointer's peak patch probability. It does not change or retrain the model. That number is a probability over image patches, **not a calibrated probability that the target exists or the click is correct**. Patch counts and image processing can change its scale. Here the absent requests happen to have lower peaks; the erroneous desktop click still has a sufficiently high peak to be accepted.

Applying either fixed threshold to the existing 30 ScreenSpot examples rejects none: 25 connected-region points remain correct and five remain wrong. All 30 requested targets exist. This only checks rejection cost on those positive examples; it supplies **no evidence about absent-target detection** on public screenshots.

The useful next experiment is a presence head trained on varied screens with labeled present and genuinely absent instructions, separating screenshots and applications across training and evaluation. Measure absent false clicks, wrongly rejected present targets, and correct accepted coordinates separately. This diagnostic gives a confidence-threshold baseline to beat.

- [Threshold-selection protocol](../results/coordinate-abstention-smoke/protocol.json)
- [All eight held-out decisions](../results/coordinate-abstention-smoke/predictions.json)
- [Fold results and present-only ScreenSpot diagnostic](../results/coordinate-abstention-smoke/result.json)
- [Reanalysis code](../experiments/coordinate_abstention_smoke.py)
- [Original local pointer results](../results/coordinates/result.json)

No model inference, training, new screenshots or latency measurements were performed. To reproduce into a fresh output directory, run a copy of `experiments/coordinate_abstention_smoke.py` with `OUT` changed; completed diagnostic outputs are protected from overwriting.
