// Independent mobile review. No model downloads, external requests, or source changes.
const {chromium}=require('C:/Users/steve/Code/LUT-maker/node_modules/playwright-core');
const fs=require('fs'),path=require('path'),http=require('http');
const root=path.resolve(__dirname,'../../..');
const archiveMode=process.argv.includes('--archive');
const walk=d=>fs.readdirSync(d,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(path.join(d,e.name)):e.name.endsWith('.html')?[path.relative(root,path.join(d,e.name)).split(path.sep).join('/')]:[]);
const pages=archiveMode?walk(path.join(root,'archive')):Object.keys(JSON.parse(fs.readFileSync(path.join(root,'content/pages.json'),'utf8')));
const server=http.createServer((req,res)=>{
  const file=path.resolve(root,'.'+decodeURIComponent(req.url.split('?')[0]));
  if(!file.startsWith(root+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);return res.end();}
  res.setHeader('Content-Type',({'.html':'text/html;charset=utf-8','.js':'text/javascript','.css':'text/css','.svg':'image/svg+xml','.png':'image/png'})[path.extname(file)]||'application/octet-stream');
  fs.createReadStream(file).pipe(res);
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 const origin='http://127.0.0.1:'+server.address().port;
 const browser=await chromium.launch({headless:true,args:['--renderer-process-limit=2']});
 const rows=[],errors=[];
 try {
  const page=await browser.newPage({reducedMotion:'reduce'});
  page.on('pageerror',e=>errors.push({url:page.url().replace(origin,''),message:e.message}));
  await page.route('**/*',r=>r.request().url().startsWith(origin)||r.request().url().startsWith('blob:')?r.continue():r.abort());
  for(const width of [320,390]){
   await page.setViewportSize({width,height:850});
   for(const name of pages){
    await page.goto(origin+'/'+name);
    if(name==='maze-live.html')await page.waitForFunction(()=>document.querySelector('#maze-live-layout').options.length>0);
    for(const expanded of [false,true]){
     if(expanded)await page.locator('details').evaluateAll(xs=>xs.forEach(x=>{if(!x.hidden)x.open=true;}));
     rows.push(await page.evaluate(({name,width,expanded})=>{
      const visible=e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden';
      const label=e=>(e.id?'#'+e.id:e.tagName.toLowerCase()+(e.className&&typeof e.className==='string'?'.'+e.className.trim().split(/\s+/).join('.'):''));
      const text=e=>(e.innerText||e.getAttribute('aria-label')||'').trim().slice(0,90);
      const content=document.querySelector('main')||document.body;
      const all=[...content.querySelectorAll('*')].filter(visible);
      const smallText=all.filter(e=>['P','LI','SUMMARY','LABEL','TH','TD'].includes(e.tagName)&&parseFloat(getComputedStyle(e).fontSize)<12).map(e=>({element:label(e),text:text(e),size:getComputedStyle(e).fontSize}));
      const clippedText=all.filter(e=>e.childElementCount===0&&text(e)&&['hidden','clip'].includes(getComputedStyle(e).overflowX)&&e.scrollWidth>e.clientWidth+3).map(e=>({element:label(e),text:text(e),width:e.clientWidth,scroll:e.scrollWidth}));
      const controls=all.filter(e=>['BUTTON','INPUT','SELECT','TEXTAREA','SUMMARY'].includes(e.tagName));
      const clippedControls=controls.filter(e=>{const r=e.getBoundingClientRect();return r.width>0&&(r.left<-1||r.right>innerWidth+1)}).map(e=>({element:label(e),text:text(e)}));
      const tinyTargets=controls.filter(e=>{const r=e.getBoundingClientRect();return !['checkbox','radio','range'].includes(e.type)&&!e.disabled&&r.height>0&&(r.height<24||r.width<24)}).map(e=>({element:label(e),text:text(e),height:e.getBoundingClientRect().height}));
      const smallInputs=controls.filter(e=>['INPUT','SELECT','TEXTAREA'].includes(e.tagName)&&!['checkbox','radio','range','hidden'].includes(e.type)&&parseFloat(getComputedStyle(e).fontSize)<16).map(e=>({element:label(e),size:getComputedStyle(e).fontSize}));
      const tables=[...content.querySelectorAll('table')].filter(visible).map(t=>{let p=t.parentElement;while(p&&p!==document.body&&!['auto','scroll'].includes(getComputedStyle(p).overflowX))p=p.parentElement;return{width:t.getBoundingClientRect().width,scrollContainer:p&&p!==document.body?label(p):null,minimumCell:[...t.querySelectorAll('td,th')].reduce((m,e)=>Math.min(m,e.getBoundingClientRect().width),Infinity)}});
      return{name,width,expanded,documentWidth:document.documentElement.scrollWidth,smallText,clippedText,clippedControls,tinyTargets,smallInputs,tables};
     },{name,width,expanded}));
    }
   }
  }
  for(const name of archiveMode?['archive/2026-09/index.html','archive/2026-09/examples/dating.html']:['preferences.html','how-it-works.html','examples/dating.html','examples/ecg.html','examples/image-safety.html','decision-results.html']){
   await page.setViewportSize({width:390,height:850});await page.goto(origin+'/'+name);
   if(archiveMode){for(const section of await page.locator('.section').all()){await section.scrollIntoViewIfNeeded();await page.waitForTimeout(650);}await page.evaluate(()=>scrollTo(0,0));await page.waitForTimeout(650);}
   await page.screenshot({path:path.join(__dirname,name.replaceAll('/','-').replace('.html','-390.png')),fullPage:true});
  }
  const result={pages:pages.length,viewStates:rows.length,widths:[320,390],errors,rows,limits:'Headless Chromium CSS viewports; expanded details do not unlock gated demos or exercise downloaded models. Small inputs are potential iOS focus-zoom issues, not confirmed Safari failures.'};
  fs.writeFileSync(path.join(__dirname,archiveMode?'archive-mobile-audit.json':'mobile-audit.json'),JSON.stringify(result,null,2)+'\n');
  const counts={};for(const key of ['smallText','clippedText','clippedControls','tinyTargets','smallInputs'])counts[key]=rows.filter(r=>r[key].length).length;
  console.log(JSON.stringify({pages:pages.length,viewStates:rows.length,errors,overflow:rows.filter(r=>r.documentWidth>r.width+1),counts},null,2));
 } finally {await browser.close();server.close();}
})().catch(e=>{console.error(e);server.close();process.exitCode=1});
