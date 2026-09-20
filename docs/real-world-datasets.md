# Five real datasets for the next experiments

These are proposed benchmarks, not new performance results. Sources checked on 19 September 2026. Each contains collected human messages, URLs, or physical measurements rather than generated examples. A real dataset still does not establish production performance: collection conditions, old data, label ambiguity, and leakage all matter.

Start with SMS filtering and complaint routing because they fit the current text-classifier pipeline. Emotion tagging adds multiple simultaneous labels. Phishing and activity recognition test whether the same decision-first approach helps outside conversational language. The application ideas and experiment designs below are our proposals.

| Dataset | Practical decision | First comparison | Important quality measure |
| --- | --- | --- | --- |
| SMS Spam Collection | Show a message or put it in spam | Sparse text classifier, tiny encoder, full Qwen, cascade | Legitimate messages incorrectly hidden |
| CFPB complaints | Route a written complaint to the right product team | Text classifier, tiny encoder, full Qwen, cascade | Correct routing and missed rare categories |
| GoEmotions | Suggest appropriate reaction tags for a comment | Multilabel tiny encoder versus deeper encoder | Per-label precision/recall and exact label-set agreement |
| PhiUSIIL | Warn about a suspicious link | URL-only classifier versus richer feature classifier | Phishing missed and legitimate links incorrectly flagged |
| Smartphone activity recognition | Identify walking, sitting, or other activity on a device | Linear/MLP baseline, small temporal network, early exits | Per-activity accuracy on unseen people |

## 1. SMS Spam Collection: stop unwanted messages

**Data and access.** Roughly 5,600 English messages labeled `ham` or `spam`, assembled from received-message collections and reported spam. The underlying sources include volunteer-contributed Singapore messages and a UK spam-reporting forum. UCI lists CC BY 4.0 and provides a small download without an account. There is no official train/calibration/test split. [UCI source, collection details, license, and download](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

**Smoke test.** Freeze duplicate and near-duplicate message groups before a stratified 60/20/20 split. Train on 512 messages, calibrate on a separate 200, and initially time 100 held-out messages. Compare word/character TF-IDF plus logistic regression, the existing two-layer BERT, full Qwen, and a confidence cascade. Keep natural class prevalence in the main evaluation; report minority-class recall separately.

**What could go wrong.** A missed personal message can matter more than an extra spam notification, so select thresholds using a false-positive budget. Repeated spam templates must not appear on both sides of the split. This is an older, geographically limited corpus; it cannot establish resistance to current scams or multilingual performance. A learned confidence score cannot guarantee that a message is safe.

## 2. CFPB Consumer Complaint Database: route support requests

**Data and access.** These are submitted consumer complaints, not invented intent examples. Use only published, nonempty complaint narratives as input and the consumer-selected `Product` as the routing label. CFPB provides filtered downloads and an open API, and explicitly permits use, analysis, and building on published data. The database changes over time and has no benchmark split. [Official database and reuse statement](https://www.consumerfinance.gov/data-research/consumer-complaints/), [API](https://cfpb.github.io/api/ccdb/api.html), [field definitions](https://cfpb.github.io/api/ccdb/fields.html), [publication policy](https://www.consumerfinance.gov/complaint/data-use/).

**Smoke test.** Export a bounded 2024 snapshot with narratives, pin its hash, and use January–June for training, July–September for calibration, and October–December for testing. Start with 1,000/300/100 examples. Define the product vocabulary from training data; unseen categories must remain visible in evaluation. Remove duplicate narrative groups across periods. Give models only narrative text, never product, issue, sub-product, or company response fields.

**What could go wrong.** Incorrect routing delays help. Product labels reflect submitter choices, not independently adjudicated truth; ambiguous cases may need human review. Report macro-F1, per-product recall, truncation rate, and routing agreement with the full model. Long narratives make tokenization and input processing important costs. The published sample is not representative of all customers, and evolving form categories require an explicit mapping rather than silent label merging.

## 3. GoEmotions: suggest reaction tags

**Data and access.** Human annotators labeled collected Reddit comments with 27 emotions plus neutral. A comment can have several labels. The agreement-filtered release supplies 43,410 training, 5,426 validation, and 5,427 test examples. Google provides the data directly; its dataset card lists Apache 2.0. [Official collection and splits](https://github.com/google-research/google-research/blob/master/goemotions/README.md), [dataset files and license metadata](https://huggingface.co/datasets/google-research-datasets/go_emotions).

**Smoke test.** Preserve the supplied split, initially train on 2,000 examples, calibrate on 500 validation examples, and time 100 test examples. Use a 28-output sigmoid head, not a single mutually exclusive enum. Compare shallow and full encoder heads, then a cascade. A skipped deeper pass is justified only when the whole predicted label set meets the chosen acceptance rule. Report macro/micro-F1, label-set agreement, and coverage.

**What could go wrong.** This measures agreement with subjective annotations of text, not access to a person's internal emotions. Sarcasm and missing conversation context are difficult. Incorrect emoji suggestions are a lower-stakes target than automated support escalation. Reddit-derived performance does not establish customer-support accuracy. Keep every annotation of the same comment together; audit duplicate texts and thread overlap. A 100-example smoke test cannot evaluate every rare emotion reliably.

## 4. PhiUSIIL: flag suspicious URLs

**Data and access.** UCI provides 235,795 examples: 134,850 legitimate and 100,945 phishing URLs. The release includes URL and webpage-derived features and labels `1 = legitimate`, `0 = phishing`. It is an approximately 15 MB compressed public download under CC BY 4.0. UCI does not provide an official held-out benchmark partition. [Official data, feature descriptions, labels, and license](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset).

**Smoke test.** Use URL text alone first: a character classifier and small encoder are fair baselines. Split by registrable domain, not individual URL, before choosing 2,000 training, 500 calibration, and 100 test rows. Compare against a richer classifier in a separate experiment; use only features obtainable at decision time. Fit corpus-derived statistics on training data only. Do not visit dataset URLs: the archived text suffices for the first test.

**What could go wrong.** Missing a phishing link can expose a user; false alarms block legitimate work. Report both errors, not just total accuracy. Published engineered similarity/probability features need a leakage audit before use. Precomputed webpage features omit their acquisition cost, so their inference time cannot be called end-to-end link checking. Domain holdout is stronger than row shuffling but does not establish performance on future attacks. This benchmark may favor an ordinary URL classifier over any language-model cascade; that is a useful result.

## 5. Smartphone Human Activity Recognition: classify motion locally

**Data and access.** Thirty volunteers performed six activities while wearing a waist-mounted phone; sensor recordings were manually labeled using video. The release has 10,299 windows with 561 derived features and windowed inertial signals. The official split separates training and test participants, using 70% and 30% of volunteers. UCI offers the approximately 58 MB download under CC BY 4.0. [Official protocol, labels, license, and download](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

**Smoke test.** Preserve participant holdout and reserve complete training participants for calibration. Start with 1,000 training windows and 100 held-out windows. First compare a linear classifier and small MLP on the provided features. Separately compare full and early-exit temporal networks on the same inertial input. Measure classifier time and preprocessing separately; do not turn the sensor values into a long prompt merely to create a slow baseline.

**What could go wrong.** Mistakes produce incorrect activity logs; these labels do not validate falls or medical decisions. Overlapping windows must remain with their participant. The recordings are real but collected under instructed conditions with one phone placement, not unrestricted daily use. Accuracy across different devices and pocket placements remains untested. Skipping network layers does not shorten the physical observation window or prove battery savings; those need separate measurements.

## Acceptance rules shared by all five

- Save download URL, retrieval date, content hash, label mapping, row IDs, duplicate policy, and split manifest before training. Preserve licenses and attribution; avoid reproducing personal message text in public reports when IDs suffice.
- Use training data to fit models and calibration data to select exit thresholds. Freeze the selected policy before inspecting test labels. Repeated exploration consumes a holdout; reserve a fresh final test.
- Treat 50–100 examples as a smoke test. Keep full-test evaluation for the final candidate, including rare classes and shifted inputs. Report paired errors and uncertainty, not a claim of universal quality preservation.
- Report both accuracy against dataset labels and agreement with the deeper baseline. Identical accuracy can hide different wrong answers. Define an acceptable quality-loss margin before testing; zero observed loss is not proof of zero future loss.
- Time the exact model and policy that produced the accuracy numbers. Include tokenization, feature extraction where available, shallow passes, routing, fallback work, and synchronization. Use warmup and repeated runs on fixed hardware with explicit thread caps.
- For an internal exit, record actual layer counts executed and skipped for every example. For a separate tiny-model cascade, record each model's work separately; a fallback still executes the larger model in full.
- Include an ordinary small classifier baseline. If it solves the task adequately, do not make a cascade look useful by comparing only against an unnecessarily expensive generative answer.
