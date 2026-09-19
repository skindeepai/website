// Check the recorded replay against every saved episode; no model is loaded.
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const origin = process.env.AUDIT_ORIGIN || 'http://127.0.0.1:8768';
const episodes = JSON.parse(fs.readFileSync(path.join(root, 'results/maze-actions/episodes.json')));

(async () => {
    const browser = await chromium.launch({headless: true, args: ['--renderer-process-limit=2']});
    try {
        const page = await browser.newPage({reducedMotion: 'reduce'});
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.route('**/*', route => route.request().url().startsWith(origin) ? route.continue() : route.abort());
        await page.goto(origin + '/maze-benchmark.html');
        await page.waitForFunction(() => !document.getElementById('maze-choice').disabled);
        let moves = 0;
        for (const record of episodes) {
            await page.locator('#maze-choice').selectOption(record.maze);
            await page.locator('#maze-method').selectOption(record.path);
            assert(await page.locator('#maze-back').isDisabled());
            assert.equal(await page.locator('#maze-status').textContent(), `Move 0 of ${record.steps.length}. Choose Next move to inspect a decision.`);
            for (const [index, step] of record.steps.entries()) {
                await page.locator('#maze-next').click();
                const status = await page.locator('#maze-status').textContent();
                assert(status.startsWith(`Move ${index + 1} of ${record.steps.length}. ${['UP', 'RIGHT', 'DOWN', 'LEFT'][step.prediction]}.`));
                assert(status.includes(step.legal ? 'Legal move.' : 'Blocked move; the agent stayed in place.'));
                assert(status.includes(step.depth ? `Used ${step.depth} of 24 layers.` : 'Shortest-path reference.'));
                const label = await page.locator('#maze-grid').getAttribute('aria-label');
                assert(label.includes(`Agent at row ${Math.floor(step.next_position / 4) + 1}, column ${step.next_position % 4 + 1}.`));
                moves++;
            }
            assert(await page.locator('#maze-next').isDisabled());
            assert((await page.locator('#maze-outcome').textContent()).includes(record.goal_reached ? 'goal reached' : 'goal not reached'));
        }
        await page.locator('#maze-back').click();
        assert(!(await page.locator('#maze-next').isDisabled()));
        await page.locator('#maze-method').selectOption('candidate');
        await page.locator('#maze-play').click();
        await page.waitForFunction(() => !document.getElementById('maze-status').textContent.startsWith('Move 0 '));
        await page.locator('#maze-play').click();
        const paused = await page.locator('#maze-status').textContent();
        await page.waitForTimeout(800);
        assert.equal(await page.locator('#maze-status').textContent(), paused);
        const widths = [320, 390, 768, 1440];
        for (const width of widths) {
            await page.setViewportSize({width, height: 900});
            await page.goto(origin + '/maze-benchmark.html');
            await page.waitForFunction(() => !document.getElementById('maze-choice').disabled);
            assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)));
            if (width === 390 || width === 1440) {
                await page.evaluate(() => { document.activeElement.blur(); window.scrollTo(0, 0); });
                await page.screenshot({path: path.join(root, `results/ui/maze-${width}.png`), fullPage: true});
            }
        }
        await page.route('**/results/maze-actions/episodes.json', route => route.abort());
        await page.reload();
        await page.waitForFunction(() => document.getElementById('maze-status').textContent.includes('could not be loaded'));
        assert(await page.locator('#maze-play').isDisabled());
        assert.deepEqual(errors, []);
        const report = {status: 'passed', recordings: episodes.length, checkedMoves: moves, widths, browser: await browser.version(), checks: ['all recorded moves, depths and agent positions', 'blocked moves stay visible', 'episode outcomes', 'previous/next limits', 'play and pause', 'mobile and desktop width', 'network failure'], limitations: ['Saved-record replay only; this does not test model inference.', 'Headless Chromium, not physical mobile devices or assistive technology.']};
        fs.writeFileSync(path.join(root, 'results/ui/maze-replay.json'), JSON.stringify(report, null, 2) + '\n');
        console.log(JSON.stringify(report));
    } finally { await browser.close(); }
})().catch(error => {console.error(error); process.exitCode = 1;});
