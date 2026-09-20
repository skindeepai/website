# One shared model, with an answer after two or four layers

We trained Google BERT small (four layers, 256 hidden dimensions; 11,105,796 parameters including both heads) in two ways: a final-layer classification loss alone, or an equal average of losses at layers two and four. All encoder parameters are trained. This gives the early classifier a chance to influence the shared representation, rather than reading a frozen large model.

## Quality on the consumed 100-message sample

| Model and route | Correct / 100 | Toxic missed / 50 | Benign blocked / 50 |
|---|---:|---:|---:|
| Previously trained tiny two-layer BERT | 82 | 13 | 5 |
| Four-layer BERT, final-layer loss only | 80 | 11 | 9 |
| Jointly trained two/four-layer BERT, answer at four | 83 | 10 | 7 |
| Same jointly trained model, always answer at two | 82 | 9 | 9 |
| Same jointly trained model, calibrated stop/continue | 83 | 10 | 7 |
| Previous full Qwen reference | 86 | 9 | 5 |

The adaptive route stopped at layer two on **72 messages**, and continued to layer four on 28. It preserved all 100 predictions of its own full four-layer reference, with zero added or corrected errors. Mean depth was **2.56 layers**, or **36% fewer transformer-block executions** than always running four. This is not 36% less memory, model parameters or wall time.

The result fits a single model with a small early exit. It does **not** fuse two independently pretrained models or establish that a two-layer BERT representation can be passed into Qwen. The lower layers are shared by construction. Later computation starts from the same hidden state and does not encode the input again.

## Training and selection

Both variants start from the same pinned Google checkpoint and seed. Both use the deduplicated 1,398 historical training messages from the previous refinement study, the same input boundary, inverse-frequency class weighting, four epochs, AdamW learning rate 0.0001 and batch size 16. Development accuracy uses the separate 500 tuning examples. The final-layer tuning balanced accuracy selects epoch and decision threshold, breaking ties by fewer toxic misses then more correct predictions. Both variants selected epoch two. The selected joint-model thresholds are 0.2 at both depths; the final-only model uses 0.4 at depth four.

After fixing the joint model, 796 development messages select an asymmetric gate with zero added errors relative to that model's full-depth decisions. Stop at layer two if its BLOCK score is at most 0.1 (SAFE) or at least 0.8 (BLOCK); otherwise continue. These thresholds are fitted heuristics, not reliability guarantees. Scores between 0.2 and 0.8 may predict BLOCK but still require full depth.

The final-only variant's layer-two head was not trained by its loss. Its recorded layer-two output is a diagnostic and is not a meaningful trained baseline. The comparison between final-only and joint full-depth models changes the training objective; a one-seed, three-message improvement does not prove a general benefit.

## Actual execution

Quality evaluation collected both depths and then applied the frozen gate offline. Its depth counts alone are policy estimates. The separate [continuation runner](../experiments/compact_continuation.py) then implemented actual stopping through a layer-two hook, checked every executed layer, and verified labels against the quality record.

| Executed route | Mean time / 50 messages | Correct / 50 | Stop after two / 50 |
|---|---:|---:|---:|
| Full four layers | 0.491 s | 41 | 0 |
| Always stop after two | 0.280 s | 39 | 50 |
| Adaptive stop/continue | 0.344 s | 41 | 35 |

The adaptive route used **30.1% less measured time** than its paired full-depth reference. All 450 calls across three rotating passes matched their expected predictions and exact layer traces. On these 50 timed inputs, the adaptive route skipped 35% of block executions; the 36% figure above applies to the full 100-message quality sample.

Loading and warm-up are excluded from request timing; Qwen-tokenizer bounding, BERT tokenization, pooling, classifier/gate checks and hook overhead are included. Other launched model jobs were paused during this run. The runner explicitly sets PyTorch to two threads. A separate agent replayed the exact imports and confirmed PyTorch/OpenMP use two threads with one interop thread. Importing the training helper leaves environment variables and NumPy/SciPy OpenBLAS pools at four; those BLAS pools are not used by the timed execution function. Thus the model-thread count is two, not a blanket two-thread setting for every imported library.

The same agent independently recomputed all 450 labels, executed-layer sequences and per-pass timing sums, plus all 100 quality decisions. This internal artifact audit passed; it is not external replication on another machine.

[Timing protocol](../results/compact-specialist/runtime-protocol.json) · [Three-pass totals](../results/compact-specialist/runtime.json) · [Every call and executed layer](../results/compact-specialist/runtime-records.json).

## What remains unproven

- These 100 ToxicChat messages were fresh for the previous refinement study but have now been consumed by exploration. This is a paired diagnostic, not independent validation.
- Identical predictions here do not guarantee identical decisions on new messages, other devices or shifts in message length or language.
- The compact adaptive model is less accurate than the existing Qwen reference on this sample. The shared-model optimization preserves its own lower-quality baseline.
- The larger compact model was not automatically better: final-only training scored 80/100, below the existing tiny model's 82/100.
- No actual service traffic, rare-error acceptance test or energy measurement was performed.

## Reproduce and inspect

[Training protocol](../results/compact-specialist/protocol.json) · [Every epoch, gate and result](../results/compact-specialist/result.json) · [Every prediction](../results/compact-specialist/predictions.json) · [Training source](../experiments/compact_specialist.py).

The base checkpoint is [google/bert_uncased_L-4_H-256_A-4](https://huggingface.co/google/bert_uncased_L-4_H-256_A-4), revision `387825ce42dbb39b87911cdf8e383ee3b25184f8`. Source files and selected weight hashes are recorded. Selected weights are kept in the local ignored experiment cache; this study is reproduced by training, rather than downloading a newly published larger model.

Run `python experiments/compact_specialist.py` in a fresh results directory, then `python experiments/compact_continuation.py` with other model jobs stopped. Completed results are protected from overwrite. Four CPU threads are used for training, two for the continuation timing. [ToxicChat provenance and split exclusions](chat-refinement.md) apply unchanged.
