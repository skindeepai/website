"""Render the measured output-path comparison as a standalone research figure."""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/chat-output-steps'
result = json.loads((OUT / 'result.json').read_text())
rows = result['methods']
labels = [
    'Trained classifier · 24 layers',
    'Two vocabulary scores · no text',
    'Forced SAFE/BLOCK · one token',
    'First token · SAFE',
    'Longer reply · SAFE',
    'First token · OK',
    'Longer reply · OK',
    'First token · THIS IS SAFE',
    'Longer reply · THIS IS SAFE',
]
assert len(rows) == len(labels)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'axes.spines.top':False, 'axes.spines.right':False})
fig, (speed, quality, misses) = plt.subplots(1, 3, figsize=(16, 8), gridspec_kw={'width_ratios':[1.3, 1, .8]}, sharey=True)
fig.subplots_adjust(left=.27, right=.98, top=.80, bottom=.20, wspace=.36)
colors = ['#2854da'] + ['#637486']*8
positions = list(range(len(rows)))
speed.barh(positions, [r['mean_ms'] for r in rows], color=colors, height=.60)
quality.barh(positions, [r['correct'] for r in rows], color=colors, height=.60)
misses.barh(positions, [r['missed_toxic'] for r in rows], color=colors, height=.60)
speed.set_yticks(positions, labels)
speed.invert_yaxis()
speed.set_xlim(0, max(r['mean_ms'] for r in rows)*1.22)
quality.set_xlim(0, 56)
quality.set_xticks([0, 10, 20, 30, 40, 50])
misses.set_xlim(0, 25)
misses.set_xticks([0, 5, 10, 15, 20, 25])
speed.set_xlabel('Milliseconds per message · lower is faster')
quality.set_xlabel('Correct / 50 · higher is better')
misses.set_xlabel('Missed / 25 · lower is better')
speed.set_title('Measured runtime', loc='left', fontsize=14, fontweight='bold')
quality.set_title('Accuracy', loc='left', fontsize=14, fontweight='bold')
misses.set_title('Toxic messages missed', loc='left', fontsize=14, fontweight='bold')
for i, row in enumerate(rows):
    speed.text(row['mean_ms']+10, i, f"{row['mean_ms']:.0f}", va='center', fontsize=11)
    quality.text(row['correct']+.8, i, f"{row['correct']}/50", va='center', fontsize=11)
    misses.text(row['missed_toxic']+.5, i, str(row['missed_toxic']), va='center', fontsize=11)
for axis in [speed, quality, misses]:
    axis.set_axisbelow(True)
    axis.grid(axis='x', color='#e2e7ed')
    axis.tick_params(axis='y', length=0)
    axis.spines['left'].set_visible(False)
fig.text(.035, .945, 'Stop after one token, or skip text generation?', fontsize=23, fontweight='bold')
fig.text(.035, .90, 'Qwen2.5-0.5B · same 50 ToxicChat messages · 25 toxic / 25 benign · two warm CPU timing passes', fontsize=12)
fig.legend(handles=[Patch(color=colors[0],label='Trained on 384 task examples'), Patch(color=colors[1],label='Original language-model readout; no task-head training')],
    loc='lower left', bbox_to_anchor=(.025,.09), frameon=False, ncol=2)
fig.text(.035, .075, 'First-token rows use the first letter; longer replies allow up to 8 tokens and use the existing tolerant parser. Unparsed output blocks.', fontsize=10)
fig.text(.035, .045, 'Exploratory, reused sample. All rows here run all 24 layers. Training differs: accuracy gains cannot be attributed to output format alone.', fontsize=10)
fig.savefig(OUT / 'comparison.png', dpi=150, facecolor='white')
fig.savefig(OUT / 'comparison.svg', facecolor='white')
svg = OUT / 'comparison.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8', newline='\n')
print('Saved comparison.png and comparison.svg.')

lines = [
    '# One output token versus a trained classifier',
    '',
    '[Website comparison](../output-results.html) · [All decision approaches](../decision-results.html)',
    '',
    'This compares actual execution on the same 50 previously inspected [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) messages, with 25 toxic and 25 benign examples. All output-path rows use the same Qwen2.5-0.5B weights and run all 24 transformer blocks. The trained classifier has additional supervision on 384 separate examples; the language-model rows do not. This is an exploratory comparison, not a production moderation validation.',
    '',
    '![Measured runtime and accuracy](../results/chat-output-steps/comparison.png)',
    '',
    '| Method | Correct / 50 | Toxic missed / 25 | Benign blocked / 25 | Unparsed / 50 | Mean ms/message | Mean output tokens |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for label, row in zip(labels, rows):
    lines.append(f"| {label} | {row['correct']} | {row['missed_toxic']} | {row['false_block']} | {row['unparsed']} | {row['mean_ms']:.1f} | {row['mean_output_tokens']:.2f} |")
lines += [
    '',
    '## What the output labels mean',
    '',
    '- **Trained classifier:** reads the final 896-number hidden state with a trained linear 896-to-2 head. No vocabulary projection, output tokens or text parsing.',
    '- **Two vocabulary scores:** uses only the original SAFE and BLOCK output-weight rows. No training, full-vocabulary projection or text generation. This is different from our browser direct-score implementation, which computes the full vocabulary projection.',
    '- **Forced one token:** normal generation API with output restricted to SAFE or BLOCK. It computes the vocabulary projection, chooses one token and stops. Its decisions matched the two-row numerical path on every example and repeat.',
    '- **First token:** unrestricted greedy generation stops after one token. The parser looks only at its first non-whitespace character: S, O or T means allow under that particular prompt; B means block. Unrecognized output blocks. This deliberately tests the proposed shortcut; it is not a reliable general natural-language parser.',
    '- **Longer reply:** unrestricted greedy generation, natural EOS stopping, capped at eight tokens. Uses the existing enhanced parser, with the same blocking fallback. Exact and supplied-parser results are also retained.',
    '',
    'SAFE, OK and BLOCK each occupy one token in this tokenizer. A model emits tokens rather than individual letters, so the first letter does not arrive earlier than the rest of that token. The first token comes from the input pass; subsequent output tokens require additional cached transformer passes. A classifier avoids the language vocabulary readout, but still needs its chosen input-processing layers.',
    '',
    'THIS IS SAFE is a longer output phrase. Stopping after its first token avoids completing the phrase only when that first token identifies the intended label. An unexpected continuation such as THIS IS NOT SAFE would invalidate the shortcut. The prompt wording and input lengths differ between label variants, so identical output-token counts do not guarantee identical latency or accuracy.',
    '',
    '## Did waiting change decisions?',
    '',
    '| Allow wording | Changed actions after longer reply | First token right, longer wrong | First token wrong, longer right |',
    '|---|---:|---:|---:|',
]
for pair in result['paired_first_vs_longer']:
    lines.append(f"| {pair['allow']} | {pair['changed_actions']} | {pair['first_correct_longer_wrong']} | {pair['first_wrong_longer_correct']} |")
lines += [
    '',
    'Every one-token run matched the actual first token of the corresponding longer run. Changed actions therefore come from later wording and parser behavior, not from different first-token model predictions. Raw generated outputs and parser outcomes are retained in the records and summary. Ambiguous prefixes must not be silently counted as valid full-word answers.',
    '',
    'A concrete failure occurred on `test:961`: the first token was `Story`, which the S-prefix rule interpreted as SAFE despite the toxic label. The longer reply continued as unrelated story text; the enhanced parser rejected it and used the blocking fallback. `Sketch` and `The` also triggered allow initials on benign examples. These are guesses from coincidental initials, not valid classification responses.',
    '',
    'The first-token SAFE path kept the same total accuracy but introduced one toxic miss and corrected one false block relative to the longer parsed reply. The OK path produced identical actions but blocked 24 of the 25 benign messages. The classifier had higher total accuracy while missing more toxic messages than constrained vocabulary scoring. None of these aggregate scores establishes an adequate moderation policy.',
    '',
    '## Timing and limits',
    '',
    'A [matched follow-up](chat-readout-control.md) isolates full-vocabulary versus two-row scoring with the same cache setting: 6.4% less total time, identical decisions.',
    '',
    'Two rotating/reversed passes, 50 messages, batch size one, four CPU compute threads and one interop thread, float32, eager attention. Timing includes input tokenization/truncation, prompt construction, actual forward/generation work, classifier or vocabulary readout, and token-to-text conversion. Model loading, warm-up, file writes and offline output parsing are excluded. The earlier parser study measured negligible parser overhead; no parser-only speed gain is claimed here.',
    '',
    'All 900 timed calls passed layer-trace checks; both timing repeats produced identical outputs. Timing repeats do not create 100 independent quality examples. This is one warmed CPU session, not GPU/NPU, browser or CtrlVox performance. The trained classifier has different task supervision, so its accuracy cannot be attributed solely to returning numbers.',
    '',
    '## Skipping the last layers',
    '',
    'A separate paired three-pass run on these same 50 messages tested trained linear heads at blocks 22, 23 and 24. Its full-depth timing is a separate reference; use within-run differences rather than assuming timings from separate phases are interchangeable.',
    '',
    '| Stop after block | Correct / 50 | Blocks skipped | Mean ms/message | New toxic misses vs full depth |',
    '|---|---:|---:|---:|---:|',
]
late = json.loads((ROOT / 'results/chat-late-exit/result.json').read_text())
for depth in ['22', '23', '24']:
    row = late['methods'][depth]
    lines.append(f"| {depth} | {row['correct']} | {row['blocks_skipped']}/24 ({100*row['fraction_blocks_skipped']:.1f}%) | {row['mean_ms_per_message']:.1f} | {row['additional_missed_toxic']} |")
lines += [
    '',
    'Layer 22 improved total accuracy by one but introduced two toxic-message misses full depth avoided. Layer 23 preserved total accuracy but introduced one toxic miss. Neither establishes a quality-preserving early exit. [Complete late-layer notes](chat-late-exit.md).',
    '',
    '## Reproduction and evidence',
    '',
    '- [Pinned protocol, prompts, sample IDs and source hashes](../results/chat-output-steps/protocol.json)',
    '- [Raw timings, output tokens and executed layers](../results/chat-output-steps/records.json)',
    '- [Results, parser variants and audit checks](../results/chat-output-steps/result.json)',
    '- [Inference runner](../experiments/chat_output_steps.py), [analysis and checks](../experiments/analyze_chat_output_steps.cjs), [figure/report builder](../experiments/plot_chat_output_steps.py)',
    '- [Earlier matched enum-versus-single-token control](matched-output.md)',
    '- [Classifier methods already used in the site](chat-late-exit.md)',
    '',
    'The inference runner protects existing output directories from overwrite. A rerun requires a new output directory and retention of the old protocol/source. Analysis: `node experiments/analyze_chat_output_steps.cjs`; figure/report: `python experiments/plot_chat_output_steps.py`.',
]
(ROOT / 'docs/chat-output-steps.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
