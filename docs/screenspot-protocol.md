# Public coordinate pilot

Recorded before inference, 2026-09-19. Use ScreenSpot (Cheng et al., SeeClick, 2024), the Apache-2.0-labeled dataset mirror `bevaya/ScreenSpot` at revision `0be08781e2e188582f6131625ae1598d443b4d5d`. Microsoft links its former name, `rootsautomation/ScreenSpot`, in GUI-Actor's example.

Select three examples per platform family (Windows/macOS/iOS/Android/web) and element type (text/icon), using deterministic seeded shuffles before running the model. Retain source row IDs, target boxes, screenshot hashes, and all failures. This is a small stratified sample, not a full benchmark score. Images remain in the ignored local cache.

Reproduce the published 2B GUI-Actor pointer, pinned to the original local pilot's weights. Verify its architecture against Microsoft's source. The original pilot used the maximum-probability patch center. Compare it with the upstream connected-region rule: patches above 0.3 times peak probability, four-neighbour components, highest mean-score region, probability-weighted center. Both points come from exactly the same forward pass, before reading target-box correctness.

Use the upstream system instruction and fixed pointer-placeholder prefix. Keep the local CPU cap of 576 visual tokens. Count a hit only when the normalized point lies inside the supplied `[x1,y1,x2,y2]` box. Run no live clicks. This does not test absent targets, coordinate early exit, or action completion. Record CPU duration for diagnostics only; claim no comparative speedup from this run.

Sources: [ScreenSpot / SeeClick](https://github.com/njucckevin/SeeClick), [dataset card](https://huggingface.co/datasets/bevaya/ScreenSpot), [GUI-Actor](https://github.com/microsoft/GUI-Actor).
