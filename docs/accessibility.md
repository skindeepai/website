# Current-site accessibility review

Reviewed 2026-09-19. Covers all 30 current pages; the historical archive retains its original presentation.

## Changes

- Early stopping: filled and dashed empty blocks with visible counts. The current 600-message diagram shows twelve layers used and twelve skipped. The distinction survives forced colors and does not depend on color alone.
- Darker secondary text, larger captions and status labels, clearer control borders, and visible keyboard focus on light pages and the dark sidebar.
- Demo: removed faded locked content from keyboard navigation; readable instructions and disabled labels; wrapping controls and session imports at narrow widths with enlarged text.
- Demo dialog: background becomes inert, focus returns on dismissal, Enter activates the selected button, and opening the breakdown respects reduced motion. Rating shortcuts only operate within the rating component.
- Visible signed weight values accompany the demo's colored bars. Rating progress is announced as a short status.
- Research questions are grouped under four topic headings. Filtering hides empty groups. Tables have named keyboard-scrollable regions.
- Coordinate markers have contrasting outlines so they remain visible over screenshots.

## Verification

Chromium 151.0.7922.34 and axe-core 4.13.0, run sequentially with two renderer processes per browser:

- 210 page/viewport checks across 320, 375, 390, 768, 900, 1024 and 1440 CSS pixels; no document overflow.
- 184 accessibility/layout checks: initial and expanded pages at 320 and 1440 pixels, enlarged text with increased spacing, trained demo, suggestion dialog, and open mobile navigation. No detected violations in the selected WCAG A/AA rules and no document overflow.
- Functional checks for menu dismissal, scoped shortcuts, locked controls, dialog focus and button activation, research filters, training, undo, import/export, blind evaluation, and coordinate annotation.
- Additional checks for skip-link focus, deep links opening technical sections, empty search results, and forced-color distinction in the early-stop graphic.
- Visual review of representative mobile/desktop renders, alongside content and shared-layout review across all current pages.

The first pass found demo text contrast as low as 1.90:1 and a session import overflow with enlarged text. Current secondary text is 7.28:1 against the page; used blocks are 6.80:1 and skipped outlines 5.47:1 against white. Measurements and automated findings are in [accessibility.json](../results/ui/accessibility.json); functional results are in [result.json](../results/ui/result.json).

Axe leaves decorative glyphs and overlays for manual contrast review. The report records their palette ratios and conservative composited backgrounds separately. These checks do not establish full WCAG conformance. Screen readers, physical iOS/Android devices, Safari, and actual browser zoom have not been tested. Enlarged-text checks increase the root font size and text spacing; they are not a browser-zoom simulation.

Reference criteria: [WCAG 2.2](https://www.w3.org/TR/WCAG22/) and [reflow guidance](https://www.w3.org/WAI/WCAG22/Understanding/reflow.html).

## Reproduce

Serve the site locally as described in the README. Provide Playwright/Chromium and axe-core in a separate tooling environment; no dependency is loaded into the public site.

```powershell
$env:PLAYWRIGHT_MODULE = 'C:/path/to/node_modules/playwright-core'
$env:AXE_CORE_PATH = 'C:/path/to/node_modules/axe-core/axe.min.js'
$env:AUDIT_ORIGIN = 'http://127.0.0.1:8768'
node scripts/audit_accessibility.cjs
node scripts/test_browser.cjs
python scripts/check_site.py
```

The research follow-up also exercised an actual Qwen download/inference run in the browser, plus cancellation, lazy loading and network-failure recovery. The browser model uses full depth; this does not validate browser early exits.

The recorded maze replay adds 30 checked recordings and 459 verified move displays, including blocked moves, executed-layer labels, play/pause and network failure. Its selectors were constrained to their container after the enlarged-text check found an overflow at 320 pixels. The final 184-check accessibility run passes. See [maze replay checks](../results/ui/maze-replay.json); reproduce with `node scripts/test_maze.cjs`.

## Focused results pages

The reorganized site has 41 current pages. Its browser regression run passed 287 page/viewport combinations across seven widths. The full automated accessibility run passed 250 checks, including expanded details and enlarged text. Final targeted checks cover the subsequent maze illustration, training error counts and table layout; see [organized-page checks](../results/ui/organized-pages.json).

The demo keeps unready suggestion panels out of the focus order and shows them when its existing rating requirement is met. Layer diagrams use filled versus outlined blocks and explicit numerical labels; the maze diagram includes player/goal/wall letters and an accessible description. These are Chromium and axe checks, not physical-device or screen-reader certification.
