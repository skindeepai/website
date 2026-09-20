# A real two-output model in the browser

The trained 4.4-million-parameter BERT specialist now runs locally in the browser as a 17.5 MB ONNX model. Its output has exactly two numbers. It has no vocabulary projection, text decoder or generated reply tokens. This is a different model from the pretrained Qwen output-format benchmark, not a compressed export of Qwen.

The model retains both transformer layers and the trained masked-mean-pooling 128-to-2 classifier. The frozen threshold is 0.4 for BLOCK. Training and threshold selection are unchanged from the [refinement study](chat-refinement.md): 1,398 training messages, separate development data, and a previously unused 100-message evaluation that is now consumed research data.

## Export and browser checks

- Python ONNX Runtime preserved all 100 decisions. Maximum absolute logit difference from PyTorch: 0.00000334.
- Actual Chromium/WASM execution processed all 100 real messages and preserved every decision. The complete BERT token sequences also matched the Python reference.
- Accuracy remained **82/100**: 13 toxic messages missed and 5 benign messages incorrectly blocked.
- The first functional browser pass took about **0.54 seconds summed across requests**. This is a single-run observation during other research work, not an isolated latency benchmark or a paired speedup against Qwen.

Every request timer includes Qwen-tokenizer input bounding, BERT tokenization, model execution, output reading and tensor cleanup. The Qwen tokenizer preserves the earlier study's first-256-Qwen-token input boundary; no Qwen model runs. Loading, dataset download and a warm-up are outside request timing. The report separately includes whole-loop time and setup time. ONNX Runtime Web 1.23.2 uses one WASM thread. Transformers.js 3.8.1 supplies the tokenizers.

The page also accepts a typed message. It sends no message text to a server; models, runtime libraries and public benchmark data are downloaded. Saved benchmark reports omit message text. The reproducibility manifest includes public benchmark token IDs to check exact preprocessing parity; those IDs can reconstruct the bounded public dataset text and should not be treated as anonymized data.

## Limits

This is not a production moderation service. The balanced 50/50 sample has been inspected before and cannot establish live-traffic quality. Timing and accuracy cannot be compared directly with the separate Qwen browser page: it uses different evaluation IDs, model weights and training. A proper head-versus-vocabulary comparison within the same Qwen model remains a separate experiment.

The export uses a float32 model. Quantization, a larger specialist and a trained early exit are distinct experiments. A smaller download does not by itself establish lower peak memory or energy use.

## Evidence

- [Try the live model](../tiny-decision-demo.html).
- [Export protocol](../results/decision-export/protocol.json), [parity summary](../results/decision-export/result.json), [per-message export checks](../results/decision-export/records.json).
- [Actual 100-message browser report](../results/decision-export/browser.json).
- [Exporter](../experiments/export_decision_model.py), [browser worker](../scripts/tiny-decision-worker.js), [model manifest](../models/moderation-tiny/manifest.json).
- Base model: [Google BERT tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2), Apache-2.0. Training dataset: [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat), CC BY-NC 4.0; this trained research artifact is provided for noncommercial research.

The exporter seals its source, source weights and input protocol before checking runtime parity. Original completed artifacts are preserved. ONNX 1.19.1 and ONNX Runtime 1.23.2 were installed in a local ignored experiment dependency directory; no project-wide runtime was changed.

For a clean environment, install [the direct dependency set](../experiments/requirements-decision-export.txt). Existing pinned model/data caches and the refinement checkpoint are required. The exporter refuses to overwrite completed evaluation artifacts.
