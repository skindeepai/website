// Verify the audit against actual DOM structure, then exercise its offline explorer.
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'C:/Users/steve/Code/LUT-maker/node_modules/playwright-core');
const fs=require('fs'),path=require('path'),http=require('http'),assert=require('assert/strict'),crypto=require('crypto');
const root=path.resolve(__dirname,'..'),data=JSON.parse(fs.readFileSync(path.join(root,'docs/navigation-map.json'),'utf8'));
const server=http.createServer((req,res)=>{
  const file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));
  if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);return res.end();}
  res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');
  fs.createReadStream(file).pipe(res);
});
(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const origin='http://127.0.0.1:'+server.address().port,browser=await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
  const output=path.join(root,'results/navigation-audit');fs.mkdirSync(output,{recursive:true});
  try{
    const sourceContext=await browser.newContext({javaScriptEnabled:false});
    await sourceContext.route('**/*',r=>r.request().url().startsWith(origin)?r.continue():r.abort());
    const source=await sourceContext.newPage();let anchors=0;
    for(const [name,item] of Object.entries(data.pages)){
      assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,name))).digest('hex'),item.sha256,name+' changed since map generation');
      await source.goto(origin+'/'+name);
      const actual=await source.locator('a[href]').evaluateAll(nodes=>nodes.map(n=>({
        href:n.getAttribute('href'),zone:n.closest('aside')?'sidebar':n.closest('footer')?'footer':n.closest('main')?'content':n.closest('header')?'header':'utility',
        collapsed:!!n.closest('details:not([open])'),hidden:!!n.closest('[hidden]')
      })));
      assert.deepEqual(actual,item.links.map(({href,zone,collapsed,hidden})=>({href,zone,collapsed,hidden})),name+' link extraction');anchors+=actual.length;
    }
    await sourceContext.close();
    const context=await browser.newContext(),page=await context.newPage(),errors=[];
    await context.route('**/*',r=>r.request().url().startsWith(origin)?r.continue():r.abort());
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(origin+'/docs/navigation-map.html');
    assert.equal(await page.locator('#page-select option').count(),data.summary.pages);
    assert.equal(await page.locator('#selected-path').textContent(),'index.html');
    assert.equal(await page.locator('#outgoing button').count(),7);
    await page.locator('#outgoing button').filter({hasText:'adaptive.html'}).click();
    await page.waitForFunction(()=>document.getElementById('selected-path').textContent==='adaptive.html');
    await page.goBack();await page.waitForFunction(()=>document.getElementById('selected-path').textContent==='index.html');
    await page.locator('#page-select').selectOption('method-demo.html');
    await page.waitForFunction(()=>document.getElementById('selected-path').textContent==='method-demo.html');
    assert.equal(await page.locator('#outgoing button').count(),data.pages['method-demo.html'].outgoing.length);
    assert.equal(await page.locator('#incoming button').count(),data.pages['method-demo.html'].incoming.length);
    assert.equal(await page.locator('#source-page').getAttribute('href'),'../method-demo.html');
    await page.locator('#global-links').check();assert((await page.locator('#outgoing button').count())>1);
    await page.locator('#global-links').uncheck();
    await page.locator('#page-select').selectOption('preferences.html');
    await page.waitForFunction(()=>document.getElementById('selected-path').textContent==='preferences.html');
    assert.equal(await page.locator('#outgoing button').filter({hasText:'preference-results.html'}).count(),1);
    await page.locator('#closed-links').uncheck();assert.equal(await page.locator('#outgoing button').filter({hasText:'history.html'}).count(),0);
    await page.locator('#closed-links').check();
    await page.locator('#page-select').selectOption('active-learning-strategies.html');
    await page.waitForFunction(()=>document.getElementById('selected-path').textContent==='active-learning-strategies.html');
    assert.match(await page.locator('#route').textContent(),/SkinDeep/);
    let views=0;
    for(const width of [320,390,768,1440]){
      await page.setViewportSize({width,height:1000});
      for(const name of ['index.html','method-demo.html','sitemap.html','research.html']){
        await page.locator('#page-select').selectOption(name);await page.waitForFunction(n=>document.getElementById('selected-path').textContent===n,name);
        assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),name+' overflow at '+width);views++;
      }
    }
    await page.locator('#page-select').selectOption('method-demo.html');await page.waitForFunction(()=>document.getElementById('selected-path').textContent==='method-demo.html');
    for(const width of [390,1440]){await page.setViewportSize({width,height:1000});await page.screenshot({path:path.join(output,'map-'+width+'.png'),fullPage:true});}
    assert.deepEqual(errors,[]);
    const result={status:'passed',pagesChecked:Object.keys(data.pages).length,anchorsChecked:anchors,mapViewports:views,
      checks:['all static hrefs, zones and disclosure states agree with browser DOM','all page source hashes match','homepage and incoming/outgoing connections','page selection and browser Back','global-navigation and disclosure filters','active learning reachable without the sitemap','local source link','responsive explorer'],
      browser:await browser.version(),mapSha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'docs/navigation-map.html'))).digest('hex'),
      limitations:['Static navigation; not analytics or a user study.','Chromium viewport checks, not physical devices.']};
    fs.writeFileSync(path.join(output,'result.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result,null,2));
  }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
