# Local research record

- `synthetic/`: five-seed preference pilots and 83 shared numerical fixtures.
- `reference/`: CPU mock-generator regressions for the older neural example.
- `qwen-decisions/`: frozen Qwen 0.5B readout/exit pilot, fixtures and predictions. Early exit missed the proposed quality tolerance.
- `coordinates/`: pretrained GUI-Actor 2B pointer reproduction, eight prompts from two original interface screenshots. Five of six visible targets hit; two absent targets still received points. No live clicks.
- `ui/`: Chromium viewport/functional results and reviewed screenshots. Not physical-device certification.
- `provenance.json`: finalized local source/artifact hashes, git base and tracked-diff identity. Untracked research files are identified by their individual hashes.

The artifact hashes describe the final record, not immutable pre-registration. Model timings are warm CPU execution measurements, excluding loading/preprocessing; they are not service-level latency. Static-site browser checks overlapped portions of the coordinate pilot, so its timings were not collected on an idle machine. It does not demonstrate a significant speed advantage from removing one vocabulary projection.

Exact broader protocols and unmet requirements remain in `EXPERIMENTS.md` and the generated `research.html`. No human ratings, private account screenshots or downloaded model weights are included.
