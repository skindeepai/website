// Validate mode-specific demo/results journeys without loading or running models.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'C:/Users/steve/Code/LUT-maker/node_modules/playwright-core');
const fs=require('fs'),path=require('path'),http=require('http'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),pages=JSON.parse(fs.readFileSync(path.join(root,'content/pages.json'),'utf8'));
const server=http.createServer((req,res)=>{const f=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));if(!f.startsWith(root+path.sep)||!fs.existsSync(f)||!fs.statSync(f).isFile()){res.writeHead(404);return res.end();}res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml'})[path.extname(f)]||'application/octet-stream');fs.createReadStream(f).pipe(res);});
(async()=>{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));const origin='http://127.0.0.1:'+server.address().port,browser=await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
  try{
    const page=await browser.newPage(),errors=[],downloads=[];
    page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/\.onnx(?:\?|$)/.test(r.url()))downloads.push(r.url());});
    await page.route('**/*',r=>r.request().url().startsWith(origin)?r.continue():r.abort());
    let modes=0;
    for(const [name,p] of Object.entries(pages)){
      const j=p.journey;if(!j)continue;
      for(const item of [j,...Object.values(j.modes||{})])for(const key of ['topic','approach','results','evidence'])if(item[key]){
        const u=new URL(item[key].href,origin+'/');assert(fs.existsSync(path.join(root,decodeURIComponent(u.pathname))),name+' missing '+key);
      }
      if(!j.modes)continue;
      for(const [mode,item] of Object.entries(j.modes)){
        const url=origin+'/'+name+'?'+j.parameter+'='+mode;
        await page.goto(url);assert.equal(await page.locator('#'+j.selector).inputValue(),mode);
        assert.equal(await page.locator('[data-journey="results"]').getAttribute('href'),item.results.href);
        assert.equal(await page.locator('[data-journey="approach"]').getAttribute('href'),item.approach.href);
        await page.locator('[data-journey="results"]').click();assert(page.url().endsWith('/'+item.results.href));
        if(item.results.href.includes('#'))assert(await page.locator('#'+item.results.href.split('#')[1]).isVisible());
        if(name!=='practical-demo.html'){
          const matching=page.locator('main a[href="'+name+'?'+j.parameter+'='+mode+'"]');assert(await matching.count()>0);await matching.first().click();assert.equal(await page.locator('#'+j.selector).inputValue(),mode);
        }else{
          await page.locator('#live-demo-link a').click();assert.equal(await page.locator('#'+j.selector).inputValue(),mode);
        }
        modes++;
      }
      // Changing a selector must update both the shareable URL and related evidence.
      for(const [mode,item] of Object.entries(j.modes)){
        await page.locator('#'+j.selector).selectOption(mode);
        assert.equal(new URL(page.url()).searchParams.get(j.parameter),mode);
        assert.equal(await page.locator('[data-journey="results"]').getAttribute('href'),item.results.href);
      }
      await page.reload();const selected=await page.locator('#'+j.selector).inputValue();assert.equal(new URL(page.url()).searchParams.get(j.parameter),selected);
    }
    await page.goto(origin+'/sitemap.html');const targets=await page.locator('main a').evaluateAll(a=>a.map(n=>n.getAttribute('href')));
    for(const name of Object.keys(pages))if(name!=='sitemap.html')assert(targets.includes(name),'Sitemap missing '+name);
    for(const [source,target] of [['preferences.html','active-learning-strategies.html'],['coordinate-results.html','screenshot-demo.html'],['coordinate-results.html','visual-refusal-results.html'],['maze-live.html','maze-results.html'],['demo-directory.html','tiny-decision-demo.html'],['results.html','search-next-results.html']]){
      await page.goto(origin+'/'+source);assert(await page.locator('main a[href="'+target+'"]').count()>0,source+' missing '+target);
    }
    const dir=path.join(root,'results/navigation-workflow');fs.mkdirSync(dir,{recursive:true});let views=0;
    for(const width of [320,390,768,1440]){
      await page.setViewportSize({width,height:1000});
      for(const name of ['method-demo.html?method=cascade','execution-demo.html?method=batch','practical-demo.html?task=privacy','browser-method-results.html#adaptive','browser-execution-results.html#combined','demo-directory.html','sitemap.html']){
        await page.goto(origin+'/'+name);assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),name+' overflow '+width);views++;
        if(width===390&&name.startsWith('method-demo'))await page.screenshot({path:path.join(dir,'method-mobile.png'),fullPage:true});
      }
    }
    assert.deepEqual(errors,[]);assert.deepEqual(downloads,[]);
    const result={status:'passed',modeRoundTrips:modes,viewportChecks:views,pagesInSitemap:Object.keys(pages).length-1,modelDownloads:downloads.length,checks:['deep links select correct mode','matching result destinations','results return to the same demo mode','mode changes update links and URL','refresh retains selection','all journey destinations exist','sitemap complete','specific missing connections repaired'],browser:await browser.version()};
    fs.writeFileSync(path.join(dir,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
