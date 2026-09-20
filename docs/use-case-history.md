# Use-case catalog: historical coverage

The catalog restores the original preference-learning applications and adds separate proposals for direct decisions, spatial outputs and adaptive computation. Historical ideas are not treated as completed products or measured results.

## Versions reviewed

Requested reference dates, relative to 20 September 2026: about one week earlier (13 September 2026) and about nine months earlier (20 December 2025).

| Reference | Latest saved repository version before that date | What was reviewed |
|---|---|---|
| One week earlier | `29faac5128f83a05ce21b70af5b71ec1d766ca97`, 5 August 2026 | Homepage, example index and domain pages; the generate / match-and-score / transform explanation |
| About nine months earlier | `caadd362247dcaf821ab20e8b4d9fe91d21a9766`, 26 June 2025 | Homepage application cards, examples, whitepaper and future explorations |

These are repository snapshots as of the requested dates, **not independently verified live deployment dates**. No intervening commits were found before those dates. The Internet Archive availability endpoint returned HTTP 429 during the review, so it did not supply a dated production capture. Search-index copies also exposed older content, but their crawl dates were not treated as exact deployment evidence.

- [August homepage source](https://github.com/skindeepai/website/blob/29faac5128f83a05ce21b70af5b71ec1d766ca97/index.html)
- [June homepage source](https://github.com/skindeepai/website/blob/caadd362247dcaf821ab20e8b4d9fe91d21a9766/index.html)
- [June future explorations](https://github.com/skindeepai/website/blob/caadd362247dcaf821ab20e8b4d9fe91d21a9766/future-explorations.html)
- [Preserved pre-lab site](../archive/2026-09/index.html)
- [Original 2019 filing](../ProvisionalPatent.txt)
- [Earlier Markdown whitepaper](../PLGL_Whitepaper.md)

## Original homepage applications

Every named application card now has its own current page. The original generic “Your Application” card maps to the local implementation guide rather than a duplicate use case.

| Original application | Current dedicated page |
|---|---|
| Music Generation | [Music and playlists](../examples/music.html) |
| Art & Design | [Art and visual design](../examples/art.html) |
| Drug Discovery | [Molecule and drug research](../examples/molecules.html) |
| Architecture | [Architecture and interiors](../examples/architecture.html) |
| Story Generation | [Stories and interactive fiction](../examples/stories.html) |
| Fashion Design | [Fashion and outfit choices](../examples/fashion.html) |
| Material Science | [Material research](../examples/materials.html) |
| Game Design | [Game levels and difficulty](../examples/games.html) |
| Private Dating | [Dating through mutual preferences](../examples/dating.html) |
| Zero-Prompt Social Media | [A feed without repeated prompting](../examples/social-media.html) |
| News & Content Curation | [News and content curation](../examples/news.html) |
| Beauty & Makeup | [Makeup and appearance previews](../examples/beauty.html) |
| Automotive Design | [Automotive design](../examples/automotive.html) |
| DNA & Genetics | [Genetics research](../examples/genetics.html) |

The related historical material also supplies recipes, 3D objects, protein research, video, voice, education, shopping, interface design, healthcare preferences and engineering constraints. Each is retained as a separate, bounded proposal. The catalog does not repeat unsupported historical claims of universal compatibility, guaranteed privacy, perfect preferences or specific performance gains.

## Historical value restored in the current wording

The preference introduction now explains learning through examples when taste is hard to describe. How it works clarifies that the drawing renderer stays fixed while the small personal scorer changes. Music retains the proposal to start from existing playlists; architecture adds shared household choices, evaluated for each person. Dating now explains synthetic-face ratings, private scoring of consenting members, and profile reveal only by mutual agreement. Its intended privacy benefits are separated from unproven security and identity-verification claims.

The [adversarial claims review](../results/use-cases/review/historical-value-review.md) records the useful ideas and rejected marketing claims. These additions fit existing pages and links; no new navigation category was needed.

The current favicon is a byte-for-byte copy of `705fa7e:favicon.ico` (25 September 2019), served as `favicon.png` because the original file contains PNG data. The three-stripe pixel artwork is unchanged. Current pages use a new, code-rendered research sharing card, replacing the old card's unsupported ideal-result and universal timing language. Archived pages retain their historical metadata and artwork.

## How the topics connect

Preference learning remains the basis of the original cases: ratings train a personal scorer, then a system ranks candidates, proposes new ones or makes a constrained edit. A domain encoder/generator and a valid evaluation are still required. The drawing demo illustrates that loop; it does not implement every domain.

Stopping early has four leading application pages: model routing, faster LLM results, cancelling unsafe image generation, and cancelling generation that misses learned preferences. The image proposals distinguish intermediate features from partial previews, and denoising steps from transformer layers. They are not represented as completed experiments.

A case that spans topics has **one canonical page**. For example, chat moderation and ECG classification appear under decisions and adaptive computation, but their data and validation requirements are distinct. The case page links back to every relevant topic. Topic teasers, the use-case index and the sitemap all come from [one catalog](../content/use-cases.json).

## Medical sources and scope

These sources identify possible research tasks and datasets; they do not validate a SkinDeep medical model:

- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/): annotated ECG recordings and recommended splits.
- [PhysioNet ECG quality challenge](https://physionet.org/content/challenge-2011/1.0.0/): recording quality, distinct from disease classification.
- [BraTS MRI tasks](https://www.med.upenn.edu/cbica/brats2020/tasks.html): brain-tumor segmentation and uncertainty evaluation, not every MRI use case.
- [NYU fastMRI](https://fastmri.med.nyu.edu/): MRI reconstruction data and project terms.
- [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/): sleep recordings and annotations.

Medical pages are marked as proposals without a clinical model or patient-use demo. Their checks distinguish participant/patient separation, missed findings, false positives, uncertainty and actual processing time. Shorter model execution is not a claim of shorter ECG observation or MRI acquisition.
