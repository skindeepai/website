// Local-only functional/responsive regression checks. No account or remote service used.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs');const path=require('path');const http=require('http');const assert=require('assert/strict');
const root=path.resolve(__dirname,'..');const pages=Object.keys(JSON.parse(fs.readFileSync(path.join(root,'content/pages.json'))));
const server=http.createServer((req,res)=>{
    const file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));
    if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);return res.end();}
    res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.png':'image/png','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');fs.createReadStream(file).pipe(res);
});
(async()=>{
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));const origin='http://127.0.0.1:'+server.address().port;
    const browser=await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
    try {
        const page=await browser.newPage({reducedMotion:'reduce'});const errors=[];const failures=[];
        page.on('pageerror',e=>errors.push(e.message));page.on('response',r=>{if(r.status()>=400)failures.push(r.url()+': '+r.status());});
        await page.route('**/*',route=>route.request().url().startsWith(origin)||route.request().url().startsWith('blob:') ? route.continue():route.abort());
        let views=0;
        for(const width of [320,375,390,768,900,1024,1440]){
            await page.setViewportSize({width,height:900});
            for(const name of pages){
                await page.goto(origin+'/'+name);views++;
                const overflow=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth}));
                assert(overflow.scroll<=width+1,`${name} at ${width}: document width ${overflow.scroll}`);
                assert.equal(await page.locator('main h1').count(),1);
                if(width<=800){
                    await page.locator('.lab-menu').click();assert.equal(await page.locator('.lab-menu').getAttribute('aria-expanded'),'true');
                    await page.keyboard.press('Escape');assert.equal(await page.locator('.lab-menu').getAttribute('aria-expanded'),'false');
                }
            }
        }
        await page.goto(origin+'/research.html');assert.equal(await page.locator('.experiment:visible').count(),29);
        await page.locator('#research-filter').selectOption('coordinates');assert.equal(await page.locator('.experiment:visible').count(),4);
        await page.locator('#research-search').fill('early exits');assert.equal(await page.locator('.experiment:visible').count(),1);
        await page.goto(origin+'/demo.html');const initial=await page.locator('#card-art').innerHTML();
        await page.locator('#generate > summary').click();
        assert(!(await page.locator('#realism').isVisible()),'Locked controls must not be reachable');
        await page.locator('#generate > summary').click();
        await page.locator('.lab-header .wordmark').focus();await page.keyboard.press('s');
        assert.equal(await page.locator('#card-art').innerHTML(),initial,'Rating shortcuts must stay inside the rating component');
        await page.locator('#card').focus();await page.keyboard.press('ArrowRight');
        assert.equal(await page.locator('#stat-n').textContent(),'1');await page.keyboard.press('u');
        assert.equal(await page.locator('#stat-n').textContent(),'0');
        await page.locator('#like-btn').click();await page.locator('#undo-btn').click();assert.equal(await page.locator('#stat-n').textContent(),'0');
        assert(await page.locator('#undo-btn').isDisabled());await page.locator('#reset-btn').click();assert.equal(await page.locator('#card-art').innerHTML(),initial);
        for(let i=0;i<12;i++)await page.locator(i%2?'#like-btn':'#pass-btn').click();
        assert.equal(await page.locator('#stat-n').textContent(),'12');assert((await page.locator('#transform-status').textContent()).length>10);
        await page.locator('#transform > summary').click();
        for(const width of [320,390,768,1024,1440]){
            await page.setViewportSize({width,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Trained demo overflow at '+width);
            const sizes=await page.locator('.tf-card').evaluateAll(cards=>cards.map(card=>card.getBoundingClientRect().width));assert(Math.abs(sizes[0]-sizes[1])<2,'Unequal transform cards at '+width);
        }
        await page.locator('.session-settings > summary').click();
        await page.locator('#evaluation-toggle').check();for(let i=0;i<4;i++)await page.locator('#like-btn').click();
        assert.equal(await page.locator('#stat-n').textContent(),'12');assert.match(await page.locator('#evaluation-summary').textContent(),/4 ratings/);
        assert(!(await page.locator('#card-pred').isVisible()));
        const download=page.waitForEvent('download');await page.locator('#export-session').click();const saved=await download;const payload=JSON.parse(fs.readFileSync(await saved.path(),'utf8'));
        assert.equal(payload.ratings.length,12);assert.equal(payload.evaluation.length,4);
        await page.locator('#reset-btn').click();await page.locator('#import-session').setInputFiles({name:'session.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(payload))});
        await page.waitForFunction(()=>document.querySelector('#stat-n').textContent==='12');assert.match(await page.locator('#session-status').textContent(),/imported/);
        await page.locator('#import-session').setInputFiles({name:'bad.json',mimeType:'application/json',buffer:Buffer.from('{"version":1,"domain":"__proto__"}')});
        await page.waitForFunction(()=>document.querySelector('#session-status').textContent.includes('Could not import'));assert.equal(await page.locator('#stat-n').textContent(),'12');
        await page.locator('#evaluation-toggle').uncheck();await page.locator('#like-btn').click();assert.match(await page.locator('#evaluation-summary').textContent(),/No blind/);
        for(let i=0;i<7;i++)await page.locator(i%2?'#like-btn':'#pass-btn').click();assert(await page.locator('#reveal-overlay').isVisible());
        await page.keyboard.press('Tab');assert.equal(await page.evaluate(()=>document.activeElement.id),'reveal-more');await page.keyboard.press('Escape');
        assert.equal(await page.evaluate(()=>document.querySelector('.lab-header').inert),false);
        await page.locator('#undo-btn').click();await page.locator('#like-btn').click();
        assert.equal(await page.evaluate(()=>document.querySelector('.lab-header').inert),true);
        await page.keyboard.press('Tab');await page.keyboard.press('Enter');
        assert(!(await page.locator('#reveal-overlay').isVisible()));
        assert.equal(await page.evaluate(()=>document.activeElement===document.querySelector('#generate > summary')),true);
        assert.equal(await page.evaluate(()=>document.querySelector('.lab-header').inert),false);
        await page.locator('[data-domain="art"]').click();assert.equal(await page.locator('#stat-n').textContent(),'0');
        await page.goto(origin+'/coordinate-lab.html');await page.locator('.manual-coordinate > summary').click();await page.locator('#coordinate-image').scrollIntoViewIfNeeded();const box=await page.locator('#coordinate-image').boundingBox();await page.mouse.click(box.x+box.width/2,box.y+box.height/2);
        const values=(await page.locator('#coordinate-output').textContent()).match(/Normalized: \(([\d.]+), ([\d.]+)\)/);assert(values&&Math.abs(Number(values[1])-.5)<.002&&Math.abs(Number(values[2])-.5)<.002);
        await page.locator('#coordinate-stage').focus();await page.keyboard.press('ArrowRight');const shifted=(await page.locator('#coordinate-output').textContent()).match(/Normalized: \(([\d.]+)/);assert(Math.abs(Number(shifted[1])-.51)<.002);
        await page.locator('#coordinate-clear').click();assert(await page.locator('#coordinate-export').isDisabled());
        await page.locator('#recorded-fixture').selectOption('3');assert.match(await page.locator('#recorded-output').textContent(),/No target exists/);assert(!(await page.locator('#recorded-box').isVisible()));
        const out=path.join(root,'results/ui');fs.mkdirSync(out,{recursive:true});
        for(const [name,width] of [['desktop',1440],['mobile',390]]){await page.setViewportSize({width,height:1000});await page.goto(origin+'/index.html');await page.screenshot({path:path.join(out,name+'.png'),fullPage:true});}
        assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
        const result={status:'passed',pages:pages.length,widths:[320,375,390,768,900,1024,1440],pageViewportChecks:views,browser:await browser.version(),checks:['document width','mobile navigation and Escape','research filtering','training and undo','seed reset','blind evaluation isolation','session export/import and rejection','reveal keyboard loop','domain isolation','normalized coordinate annotation and keyboard'],limitations:['Headless Chromium, CSS viewports; not physical iOS/Android or Safari verification.','No assistive-technology or real-device performance audit.']};
        fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
    }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
