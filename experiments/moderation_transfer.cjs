'use strict';
// Actual SSApp runtime, isolated profile, localhost-only network and pinned public model.
const fs=require('fs'),path=require('path'),os=require('os'),http=require('http');
const {pathToFileURL}=require('url');
const ROOT=path.resolve(__dirname,'..'),SSN='C:/Users/steve/Code/social_stream',APP='C:/Users/steve/Code/ssapp';
const {_electron}=require(path.join(SSN,'node_modules/playwright'));
const OUT=path.join(ROOT,'results/moderation-transfer');
const network=[];
const types={'.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json','.html':'text/html'};
const modes=(process.env.MODES||'baseline,original,original-one,original-direct,short,short-one,short-direct,short-cache').split(',');
const tasks=JSON.parse(fs.readFileSync(path.join(OUT,process.env.TASKS||'tasks.json'))).slice(0,Number(process.env.COUNT||2));
const source=fs.readFileSync(process.env.PRODUCTION?path.join(SSN,'local-browser-model-worker.js'):path.join(OUT,'original/local-browser-model-worker.js'),'utf8');
const marker="self.addEventListener('message'";
const worker=source.slice(0,source.lastIndexOf(marker))+fs.readFileSync(path.join(__dirname,'moderation_transfer_worker.js'),'utf8');
const server=http.createServer((req,res)=>{
 res.setHeader('Access-Control-Allow-Origin','*');
 network.push(req.url);const u=decodeURIComponent(req.url.split('?')[0]);
 if(u==='/lab.js'){res.setHeader('Content-Type','text/javascript');return res.end(worker);}
 if(u==='/lab.html'){res.setHeader('Content-Type','text/html');return res.end('<title>Local moderation experiment</title>');}
 const base=u.startsWith('/models/qwen3.5-0.8b-onnx-opt/')?path.join(ROOT,'experiments/.cache/qwen35-opt'):SSN;
 const rel=u.startsWith('/models/qwen3.5-0.8b-onnx-opt/')?u.slice('/models/qwen3.5-0.8b-onnx-opt/'.length):u.slice(1);
 const file=path.resolve(base,rel);
 if(!file.startsWith(path.resolve(base)+path.sep)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.statusCode=404;return res.end();}
 res.setHeader('Content-Type',types[path.extname(file)]||'application/octet-stream');
 res.setHeader('Content-Length',fs.statSync(file).size);fs.createReadStream(file).pipe(res);
});
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r));const host='http://127.0.0.1:'+server.address().port;
 const profile=fs.mkdtempSync(path.join(os.tmpdir(),'ssapp-moderation-lab-'));
 fs.writeFileSync(path.join(profile,'savedSync.json'),JSON.stringify({streamID:'local_moderation_lab',password:'false',state:false,settings:{},wsServer:false}));
 const wrapper=path.join(profile,'bootstrap.cjs');
 fs.writeFileSync(wrapper,`const {app}=require('electron');app.setAppPath(${JSON.stringify(APP)});app.on('session-created',s=>s.webRequest.onBeforeRequest({urls:['http://*/*','https://*/*','ws://*/*','wss://*/*']},(d,cb)=>cb({cancel:new URL(d.url).hostname!=='127.0.0.1'})));require(${JSON.stringify(path.join(APP,'bootstrap.js'))});`);
 let app;
 const records=[];
 fs.writeFileSync(path.join(OUT,(process.env.RECORDS||'records.json').replace('.json','-worker.js')),worker);
 try{
  app=await _electron.launch({executablePath:require(path.join(APP,'node_modules/electron')),cwd:APP,args:[wrapper,'--running-from-source','--multiinstance','--filesource',pathToFileURL(SSN+path.sep).href],env:{...process.env,SSAPP_USER_DATA_DIR:profile}});
  const main=await app.firstWindow();main.on('console',m=>{if(m.type()==='error')console.log('APP',m.text().slice(0,800));});main.on('requestfailed',r=>console.log('FAILED',r.url(),r.failure()?.errorText));await main.waitForFunction(()=>typeof document.getElementById('frame2')?.contentWindow?.callLLMAPI==='function',null,{timeout:60000});
  const pagePromise=app.waitForEvent('window');
  await app.evaluate(({BrowserWindow},url)=>new BrowserWindow({show:false,webPreferences:{backgroundThrottling:false}}).loadURL(url),host+'/lab.html');
  const page=await pagePromise;await page.waitForLoadState('domcontentloaded');
  page.on('console',m=>{if(m.type()==='error')console.log('BROWSER',m.text().slice(0,350));});
  await page.evaluate(()=>{window.worker=new Worker('/lab.js',{type:'module'});window.serial=0;window.pending=new Map();worker.onmessage=({data:m})=>{if(m.lab){const p=pending.get(m.id);if(p){pending.delete(m.id);m.error?p.reject(Error(m.error)):p.resolve(m.result);}}};worker.onerror=e=>{for(const p of pending.values())p.reject(Error(e.message));pending.clear();};window.ask=data=>new Promise((resolve,reject)=>{const id=++serial;pending.set(id,{resolve,reject});worker.postMessage({...data,lab:true,id});});});
  const init=await page.evaluate(host=>ask({op:'init',host}),host);console.log('INIT',JSON.stringify(init));
  fs.writeFileSync(path.join(OUT,(process.env.RECORDS||'records.json').replace('.json','-runtime.json')),JSON.stringify({init,date:new Date().toISOString(),electron:await app.evaluate(()=>process.versions),gpu:await app.evaluate(async({app})=>app.getGPUInfo('basic')),cpu:os.cpus()[0].model,modes,scope:'Exploratory previously inspected real ToxicChat examples. Single-message inputs; no deployment quality claim.'},null,2));
  const warmups=[];for(const mode of modes){const r=await page.evaluate(({mode})=>ask({mode,text:'Thank you for your help!'}),{mode});warmups.push({mode,...r});console.log('WARM',mode,JSON.stringify(r));}fs.writeFileSync(path.join(OUT,(process.env.RECORDS||'records.json').replace('.json','-warmups.json')),JSON.stringify(warmups,null,2));
  for(let i=0;i<tasks.length;i++){
   const order=i%2?modes.slice().reverse():modes.slice();
   for(const mode of order){const task=tasks[i],r=await page.evaluate(({mode,task})=>ask({mode,text:task.text,history:task.history}),{mode,task});const record={id:task.id,expected:task.expected,mode,...r,correct:r.label===task.expected};records.push(record);fs.writeFileSync(path.join(OUT,process.env.RECORDS||'records.json'),JSON.stringify(records,null,2));console.log(i+1,mode,r.label,record.correct,Math.round(r.ms)+'ms');}
  }
  if(process.env.PRODUCTION){
   const contextChecks=[];
   for(const task of [
    {text:'yourself',history:'1. User: go\n2. User: kill'},
    {text:'K',history:'1. A: F\n2. B: U\n3. C: C'},
    {text:'yourself',history:'1. User: believe in'},
    {text:'Thank you for explaining that.',history:'1. User: Can you help me?'}
   ]){contextChecks.push({...task,baseline:await page.evaluate(task=>ask({mode:'baseline',...task}),task),result:await page.evaluate(task=>ask({mode:'production',...task}),task)});}
   const clean=await page.evaluate(()=>ask({mode:'production',text:'I enjoyed the music today.'}));
   await page.evaluate(()=>ask({op:'poison'}));
   const poisoned=await page.evaluate(()=>ask({mode:'production',text:'I enjoyed the music today.'}));
   fs.writeFileSync(path.join(OUT,'production-context.json'),JSON.stringify({contextChecks,clean,poisoned},null,2));
   if(clean.label!==poisoned.label)throw Error('Co-host memory leaked into moderation');
   await page.evaluate(()=>worker.terminate());
   const bg=main.frames().find(f=>f.url().includes('background.html'));
   const integration=await bg.evaluate(async host=>{
    const options={settings:{aiProvider:{optionsetting:'localqwen'},localgemmahost:{textsetting:host+'/models/'}},localBrowserStateless:true,localBrowserModeration:true};
    const prompt=buildCensorPrompt({chatname:'User'},'I enjoyed the music today.',[],[],'localqwen');
    const start=performance.now();
    const reply=await callLLMAPI(prompt,null,null,null,null,null,options);
    const result={reply,decision:parseBinaryCensorDecision(reply),ms:performance.now()-start};
    await disposeLocalBrowserLLMClient();return result;
   },host);
   fs.writeFileSync(path.join(OUT,'production-integration.json'),JSON.stringify(integration,null,2));
   if(integration.reply!=='OK')throw Error('Actual ai.js -> client -> worker integration failed');
   console.log('INTEGRATION',JSON.stringify(integration));
  }
 }finally{fs.writeFileSync(path.join(OUT,(process.env.RECORDS||'records.json').replace('.json','-network.json')),JSON.stringify(network,null,2));if(app)await app.close();server.close();}
})().catch(e=>{console.error(e);fs.writeFileSync(path.join(OUT,(process.env.RECORDS||'records.json').replace('.json','-error.json')),JSON.stringify({error:e.stack,date:new Date().toISOString()},null,2));process.exitCode=1;server.close();});
