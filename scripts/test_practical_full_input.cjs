'use strict';
const fs=require('fs'),path=require('path'),http=require('http'),assert=require('assert/strict'),crypto=require('crypto');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'C:/Users/steve/Code/LUT-maker/node_modules/playwright-core');
const P=require('./practical-demo-core.js'),root=path.resolve(__dirname,'..');
const model=JSON.parse(fs.readFileSync(path.join(root,'models/practical/privacy.json'),'utf8'));
const text='word '.repeat(1250)+'😀 Dr. Alice Morgan spoke with Mr. David Clark.';
const prefix=P.privacy(text,model),full=P.privacy(text,model,{fullDocument:true});
assert.equal(prefix.tokens.length,1200);assert.equal(prefix.truncated,true);assert.equal(full.truncated,false);
assert(full.tokens.length>1250);assert(full.tokens.some(t=>t.marked&&t.start>6250));
for(const token of full.tokens)assert.equal(text.slice(token.start,token.end),token.text);
assert.deepEqual(full.tokens.slice(0,1199),prefix.tokens.slice(0,1199));
for(const input of ['', 'Dr. Alice Morgan.', '😀 Alice Kelly met ² ١ ²².', ' ']){
 assert.deepEqual(P.privacy(input,model,{fullDocument:true}),P.privacy(input,model));
}
const server=http.createServer((req,res)=>{
 const file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));
 if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);return res.end();}
 res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');
 fs.createReadStream(file).pipe(res);
});
(async()=>{
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const origin='http://127.0.0.1:'+server.address().port,browser=await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
 try{
  const page=await browser.newPage(),errors=[],external=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await page.route('**/*',route=>{if(route.request().url().startsWith(origin))return route.continue();external.push(route.request().url());return route.abort();});
  await page.goto(origin+'/practical-demo.html?task=privacy');
  assert.equal(await page.locator('#practical-task').inputValue(),'privacy');
  await page.locator('#practical-input').fill(' ');await page.locator('#practical-run').click();
  assert.match(await page.locator('#practical-status').innerText(),/Enter/);
  assert(!requests.some(u=>u.includes('models/practical/')));
  await page.locator('#practical-input').fill(text);await page.locator('#practical-run').click();
  await page.waitForFunction(()=>document.querySelector('#practical-status').textContent.startsWith('Finished'));
  assert.equal(await page.locator('#practical-output p').first().textContent(),text);
  assert.deepEqual(await page.locator('#practical-output mark').allTextContents(),full.tokens.filter(t=>t.marked).map(t=>t.text));
  assert.match((await page.locator('#practical-output').innerText()).replace(/,/g,''),new RegExp(full.tokens.length+' tokens examined'));
  assert(!(await page.locator('#practical-output').innerText()).includes('Only the first'));
  assert.equal(await page.locator('[data-journey="evidence"]').getAttribute('href'),'docs/privacy-full-document.md');
  let views=0;
  const widths=[320,375,390,768,900,1024,1440];
  for(const width of widths){await page.setViewportSize({width,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));views++;}
  // Exact current limit: rendering a long token must wrap without widening mobile layout.
  await page.locator('#practical-input').fill('a'.repeat(30000));await page.locator('#practical-run').click();
  await page.waitForFunction(()=>document.querySelector('#practical-status').textContent.startsWith('Finished'));
  await page.setViewportSize({width:320,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.evaluate(()=>{document.querySelector('#practical-input').value='a'.repeat(30001);});
  await page.locator('#practical-run').click();assert.match(await page.locator('#practical-status').innerText(),/Enter 1 to 30,000/);
  // Shared demo still executes the untouched routing and receipt paths.
  for(const task of ['routing','receipts']){
   await page.locator('#practical-task').selectOption(task);await page.locator('#practical-run').click();
   await page.waitForFunction(()=>document.querySelector('#practical-status').textContent.startsWith('Finished'));
   assert((await page.locator('#practical-output').innerText()).length>0);
   assert.equal(await page.locator('[data-journey="evidence"]').getAttribute('href'),'docs/practical-baselines.md');
  }
  await page.goto(origin+'/redaction-results.html');
  assert.deepEqual(await page.locator('tbody tr').allTextContents(),['First 1,200 tokens290 / 445339','Entire input319 / 445378']);
  await page.locator('details').first().evaluate(e=>e.open=true);
  for(const width of widths){await page.setViewportSize({width,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));views++;}
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  const hashes=Object.fromEntries(['scripts/practical-demo-core.js','scripts/practical-demo.js','scripts/test_practical_full_input.cjs','models/practical/privacy.json'].map(name=>[name,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,name))).digest('hex')]));
  const report={status:'passed',browser:await browser.version(),populatedViewportChecks:views,checkedLongInputTokens:full.tokens.length,
   checks:['Original prefix control retained','Short and Unicode input parity','Full-input tail classified and correctly highlighted','Empty input does not download models','30000-character limit retained','Maximum-length token wraps on mobile','Receipt and routing continue to work','Privacy evidence links to matching new study','Published full-input counts match evidence','No page errors or external requests'],hashes};
  fs.writeFileSync(path.join(root,'results/privacy-full-document/browser.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1;});
