// Render the source SVG at the exact social preview dimensions; no network.
const fs = require('fs');
const path = require('path');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
    const root = path.resolve(__dirname, '..');
    const browser = await chromium.launch({headless:true, args:['--renderer-process-limit=2']});
    try {
        const page = await browser.newPage({viewport:{width:1200,height:630},deviceScaleFactor:1});
        await page.setContent('<style>html,body{margin:0;width:1200px;height:630px}svg{display:block}</style>' + fs.readFileSync(path.join(root,'images/skindeep-research-card.svg'),'utf8'));
        await page.screenshot({path:path.join(root,'images/skindeep-research-card.png')});
    } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
