'use strict';
const fs=require('fs'),path=require('path'),http=require('http'),assert=require('assert/strict');
const {chromium}=require('C:/Users/steve/Code/LUT-maker/node_modules/playwright-core');
const root=path.resolve(__dirname,'..'),out=path.join(root,'results/moderation-transfer');
const server=http.createServer((req,res)=>{
 const file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));
 if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.statusCode=404;return res.end();}
 res.setHeader('Content-Type',({'.html':'text/html; charset=utf-8','.js':'text/javascript','.css':'text/css','.json':'application/json','.wasm':'application/wasm','.svg':'image/svg+xml'})[path.extname(file)]||'application/octet-stream');
 fs.createReadStream(file).pipe(res);
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const browser=await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
 try{
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:'+server.address().port+'/moderation-benchmark.html');
  const report=process.env.SAVED?JSON.parse(fs.readFileSync(path.join(out,'browser-qwen25.json'),'utf8')):await page.evaluate(()=>new Promise((resolve,reject)=>{
   const worker=new Worker('scripts/moderation-benchmark-worker.js',{type:'module'});
   const timer=setTimeout(()=>{worker.terminate();reject(Error('30-minute timeout'));},1800000);
   worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};
   worker.onmessage=({data})=>{if(data.type==='complete'||data.type==='error'){clearTimeout(timer);worker.terminate();data.type==='error'?reject(Error(data.message)):resolve(data.result);}};
   worker.postMessage({type:'run',count:10});
  }));
  fs.writeFileSync(path.join(out,'browser-qwen25.json'),JSON.stringify(report,null,2));
  assert.equal(report.rows.length,40);
  const cached=report.rows.filter(r=>r.format==='cached');assert.equal(cached.length,10);
  assert(cached[0].cacheBuildMs>0);assert(cached.slice(1).every(r=>r.cacheBuildMs===0));
  assert(cached.every(r=>r.processedInputTokens<r.inputTokens));
  // Feed the real results through the production UI without downloading the model again.
  await page.evaluate(report=>{
   window.Worker=class {postMessage(){setTimeout(()=>{
    const {rows,summary,complete,...metadata}=report;
    this.onmessage({data:{type:'metadata',metadata}});
    report.rows.forEach((row,index)=>this.onmessage({data:{type:'row',row,completed:index+1,total:report.rows.length}}));
    this.onmessage({data:{type:'complete',result:report}});
   },0);}terminate(){}};
  },report);
  await page.selectOption('#moderation-count','10');await page.click('#moderation-run');
  await page.waitForFunction(()=>document.querySelector('#moderation-status').textContent.startsWith('Finished'));
  assert.equal(await page.locator('#moderation-summary tr').count(),4);
  assert.equal(await page.locator('#moderation-rows tr').count(),40);
  assert.equal(await page.locator('#moderation-progress').getAttribute('value'),'40');
  assert.equal(await page.locator('#moderation-summary tr').last().locator('th').innerText(),'Cached instructions');
  const widths=[320,375,390,768,1440];
  for(const width of widths){await page.setViewportSize({width,height:900});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));if(width===390||width===1440)await page.screenshot({path:path.join(out,'browser-'+width+'.png'),fullPage:true});}
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser-ui.json'),JSON.stringify({widths,errors,status:await page.locator('#moderation-status').innerText()},null,2));
  console.log(JSON.stringify(report.summary));
 }finally{await browser.close();server.close();}
})().catch(e=>{console.error(e);fs.writeFileSync(path.join(out,'browser-error.json'),JSON.stringify({error:e.stack},null,2));process.exitCode=1;server.close();});
