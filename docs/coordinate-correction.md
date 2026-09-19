# Correction to the first coordinate pilot

During the 2026-09-19 public-data follow-up, source inspection and an actual processor check found that the installed Transformers development build uses `image_processor.size` for resizing. Passing `max_pixels` to `AutoProcessor.from_pretrained` changed the legacy attribute without changing the loaded `size` mapping.

The original notes therefore overstated what the code enforced. The **saved predictions** show 36 × 26 = **936 visual patches on desktop**, and 14 × 28 = **392 on mobile**, not a universal cap of 576. Those recorded predictions and timings remain the evidence for the original 5/6 result. No outcomes are being retroactively changed.

The new ScreenSpot harness sets the actual size mapping explicitly and asserts that the produced patch count is at most 576. Its first attempted public-image run was stopped before it produced any predictions after this issue was found. The corrected run is the reported run.

The original pointer adapter's mathematical architecture agrees with the upstream implementation: visual-feature self-attention and residual normalization; separate two-layer GELU projections of image features and the task-query representation; scaled dot products and a softmax over patches. It uses visual embeddings from the vision encoder and the final transformer's pointer-query state. It is not a direct x/y linear regression.

Another difference was deliberate but underspecified: the first pilot used only the highest-scoring patch center, while upstream supports a weighted center over connected activated patches. The public-data pilot compares those two readouts on identical probabilities. It also uses the upstream system instruction, so comparisons with the earlier homemade screenshots are not controlled before/after comparisons.

Sources: [original per-example evidence](../results/coordinates/predictions.json), [original adapter](../experiments/coordinates.py), [corrected public-data harness](../experiments/screenspot.py), [Microsoft's pointer implementation](https://github.com/microsoft/GUI-Actor/blob/d98d1bbd01862f9112114b83b032f492c365a173/src/gui_actor/modeling.py), [region readout](https://github.com/microsoft/GUI-Actor/blob/d98d1bbd01862f9112114b83b032f492c365a173/src/gui_actor/inference.py).
