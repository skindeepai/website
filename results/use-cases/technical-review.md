# Adversarial technical review of use cases

Reviewed 20 September 2026 by an independent review agent. This was a content and source review, not an experiment or clinical validation.

## Scope

Reviewed `content/use-cases.json`, `scripts/refresh_use_cases.py`, `docs/use-case-history.md`, the original `caadd36` homepage and future-exploration material, and relevant existing demo/evidence descriptions. Checked the regenerated preference page after revisions. All 14 named application cards from the original homepage have current counterparts. Historical repository snapshots are not proof of deployment on a particular date.

## Substantive revisions verified

- **Original preference mechanism:** art, music, social-media and beauty now explain selecting or adjusting compatible generator inputs before rendering. The preference topic distinguishes that route from ranking existing items. Neither route assumes universal generator compatibility.
- **Useful medical comparisons:** ECG labels, ECG quality, sleep staging, MRI segmentation and MRI reconstruction now require conventional task-specific baselines and measured processing costs. Structured output alone is not presented as a new medical capability.
- **Cancelled-image evaluation:** image-safety now calls for completing matched generations offline with the same prompts and seeds to inspect what rejected candidates would have become. Its checks include false cancellations, missed unsafe outputs and checking costs.
- **Consistent health scope:** patient-preferences now carries the medical-proposal flag, alongside its existing limits on clinical suitability and treatment decisions.

## Remaining limits

The domain applications remain proposals unless explicitly tied to a narrow prototype. No medical model, faster general-purpose chat model or intermediate-image safety system was validated in this work. Confidence is not a correctness guarantee. Separate-model routing and early exits within one model are distinct mechanisms. Related drawing, OCR, BERT and recorded-Qwen demos do not implement every proposed application; their qualifications should remain visible.

The catalog appropriately distinguishes model processing time from ECG observation time or MRI acquisition time. It also separates personal preference, physical feasibility and clinical suitability. No new quantitative performance claims were found in the reviewed use-case copy. Visual layout and navigation were assigned to a separate reviewer.

## Primary sources checked

The linked descriptions support the proposed dataset starting points; they do not establish SkinDeep performance:

- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/): multi-label ECG recordings and patient-separated recommended folds.
- [PhysioNet 2011 ECG quality challenge](https://physionet.org/content/challenge-2011/1.0.0/): recording quality, distinct from disease classification.
- [BraTS 2020 tasks](https://www.med.upenn.edu/cbica/brats2020/tasks.html): multimodal brain-tumor MRI segmentation.
- [NYU fastMRI](https://fastmri.med.nyu.edu/): MRI reconstruction research data and access terms.
- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/): sleep recordings and stage annotations.
