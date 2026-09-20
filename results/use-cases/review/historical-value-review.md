# Historical value and claims review

Reviewed 20 September 2026 by an independent adversarial review agent. This is a source/content review, not a new model experiment or product validation.

## Historical sources

- `caadd36:index.html` (26 June 2025): rating-first explanation, application cards and old marketing claims. Lines 133, 385 and 405 respectively illustrate the accessible explanation, unsupported speed claim and overly broad privacy claim.
- `29faac5:index.html` (5 August 2026): ratings rather than descriptions (244-254), a small personal classifier (262), group preferences (279), active sampling (215), and the original working app record.
- `ProvisionalPatent.txt`: group preferences (94-96), targeted edits (108-112), choosing useful rating examples (152), learning from existing playlists (183), and private member matching (187-190).
- `docs/use-case-history.md` correctly distinguishes saved repository versions from independently verified live deployment dates.

## Valuable ideas to retain

1. **Show preferences through examples.** The new preference introduction explains the user benefit without promising that every taste can be learned accurately.
2. **Update the small personal scorer.** The how-it-works addition accurately describes the current drawing demo: training changes the preference model, not the renderer. `scripts/demo.js:530` and `scripts/preference-core.js:9` support this mechanism.
3. **Start from existing likes.** The music page now restores the original playlist idea as a future proposal. There is no validated audio generator or listener study here.
4. **Find shared choices.** The architecture page restores household preferences as a proposal and checks each person's satisfaction, rather than accepting a favorable average alone.
5. **Edit an existing choice.** Existing makeup/design examples preserve this useful original distinction. A small latent edit is not automatically a small visible or perceptual edit.
6. **Choose the next example to ask about.** The existing approach page provides an appropriate home. Do not market a general reduction in required ratings: `docs/preference-followups.md` reports mixed synthetic sampling results.

These ideas fit existing topic, approach and use-case pages. They do not require another marketing page or a new global navigation item. Historical screenshots and source links remain most useful in History.

## Dating proposal: final content review

The reviewed `content/use-cases.json` entry appropriately distinguishes:

- Clearly labelled generated faces used as rating examples from real member identities.
- Photos supplied privately by consenting adults from a scraped public-photo collection.
- A two-way predicted preference from either person's consent to reveal a profile.
- Possible synthetic-to-real transfer from demonstrated matching performance.
- Keeping a catalog out of public view from preventing copying of a revealed photo.
- Preference matching from identity verification, catfishing prevention and relationship compatibility.

No blocking substantive claims issue found in this revision. It is explicitly a proposal, not an implemented private matching service. The word "privately" describes the intended product workflow; it must not be expanded into a security guarantee without an implementation and security review.

## Metadata and social card

The reviewed title/description wording presents research, experiments, demos and measured results without claiming superiority. The new social-card SVG lists the research topics without quantitative or privacy claims.

One concrete defect was reported to the parent agent: all five new `meta_title` strings contained literal ASCII ` ? ` separators, also present in generated HTML titles. Replace with an intentional separator and rebuild before publishing; this finding is an encoding/content issue, not a claims issue.

One preservation suggestion was reported: the music scenario replacement removed the prior explicit game-soundtrack and meditation-playlist examples. Retain those applications where concise, since the user asked to carry forward the original examples.

## Claims deliberately not restored

Do not bring back universal "1000x faster" or millisecond-training promises, "100% ideal" outputs, privacy guarantees from latent vectors, "works with any generator," or claims that nobody else has implemented the overall approach. A historical provisional filing documents an idea; it does not establish measured performance, an issued patent, universal novelty or clinical validity.

No source content was edited by this reviewer. Only this review record was added.

## Resolution check

Independently rechecked the current files after the parent agent's fixes. All five `meta_title` values now use the intentional ASCII ` | ` separator, and each generated HTML title exactly matches its source value. The music scenario again names game soundtracks and meditation playlists, both in `content/use-cases.json` and the generated `examples/music.html`. Both reported findings are resolved; no open findings remain from this review.
