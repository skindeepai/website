# Site presentation review — 20 September 2026

The historical-use-case follow-up restores the original favicon, clarifies selected preference-learning benefits, expands the dating proposal, and updates current-page search/social metadata. It does not change benchmark results or model behavior.

## Search and sharing

- All current pages have matching canonical and Open Graph URLs, page-specific titles/descriptions, a site name, and large-image Twitter/Open Graph cards with image dimensions and alternative text.
- The homepage and four topic pages have more descriptive search metadata without lengthening their visible introductions. The homepage includes factual WebSite/creator structured data.
- `images/skindeep-research-card.svg` is the editable source for the new 1200×630 PNG. Rebuild with `scripts/build_share_card.cjs`; it uses the original favicon artwork and local typography.
- `scripts/check_seo.py` verifies all 111 pages, the sitemap, original favicon bytes and actual PNG dimensions. Historical archives and crawler access policies remain unchanged.
- Search engines and sharing services choose when to refresh cached previews. Local metadata validation does not establish indexing or ranking improvements.

## Mobile and navigation

The standard browser suite passed 777 page/viewport checks: every current page at 320, 375, 390, 768, 900, 1024 and 1440 pixels. It also checks mobile menus and the preference/coordinate interaction flows.

An independent agent inspected all 111 current pages at 320 and 390 pixels with details collapsed and expanded (444 states), plus 21 archive pages (84 states). Current pages have no detected document overflow, clipped text or controls. Four demo fields now use 16px mobile text to address a potential focus-zoom issue. The archived demo retains a small reset button; original archive presentation is preserved.

These are headless Chromium checks, not physical Android/iOS, Safari or assistive-technology verification. Automated inspection does not establish complete accessibility conformance.

Use-case links still follow topic → canonical application page → clearly labelled related demo/evidence, with return links. No extra global navigation items were added. Audit records and representative screenshots are in `results/use-cases/review/`; the complete link map is `docs/navigation-map.html`.
