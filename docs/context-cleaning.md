# Cleaning the context on the CPU

Part of [Working past the context limit](../context.html). Results: [cleaning table](../context-results.html#cleaning). Protocol: [L05](../EXPERIMENTS.md#l05--clean-the-context-on-the-cpu-without-loss).

**Question.** Can a small program or classifier running on the CPU tidy an agent's context so a 32K window holds about 100K, with nothing lost?

**Data.** 5.0M tokens of real agent sessions (three long coding-agent transcripts), counted with the Qwen tokenizer, plus 277 calls from the 27B's own runs on the ledger task.

**Rules.** Strictly no loss: replace exact duplicate outputs and superseded file reads with a pointer, drop repeated lines, whitespace and terminal noise. Not deleted but moved: large tool outputs (over 2,000 tokens) go to disk, keeping their first and last 400 tokens in view. Lossy: drop the model's earlier thinking.

**Why the gain is small.** Most of an agent's context is new material: its own commands (48%) and fresh tool output (43%). There is little exact repetition to remove. About one moved output in three was quoted later, so moving to disk costs read-backs.

**Earlier thinking.** The model's chat template re-sends all earlier thinking on every call: about 10% of a typical call, up to 56%. Dropping it blindly made one long run re-derive its state on every call and never act. It is safe only when the working state is kept elsewhere, as in the improved agent.

## Raw data and lab notes

- [Census: tables per rule, per session and window multipliers](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/notes/2026-10-05-context-hygiene-census.md)
- [Census script](https://github.com/steveseguin/b70-optimization-lab/blob/main/experiments/qwen38-27b-b70/scripts/context/hygiene_census.py)
