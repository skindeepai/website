'use strict';
// Reuse the verified runtime; this worker replaces only its message handler.
importScripts('shared-decision-worker.js');
let labBusy=false,labManifest=null,labTiny=null;
const labModes={fixed:['full','fixed2'],adaptive:['full','adaptive'],cascade:['full','cascade'],training:['full','trained'],batch:['full','batch4'],quantization:['full','int8']};
async function labPrepare(mode){
 const r=await setup();await ensureControl(r);
 if(!labManifest){const f=await fetch('../models/method-lab/manifest.json');if(!f.ok)throw Error('Method manifest unavailable.');labManifest=await f.json();}
 if(mode==='cascade'&&!labTiny){const f=await fetch('../models/moderation-tiny/manifest.json');if(!f.ok)throw Error('Tiny manifest download failed.');const m=await f.json();const b=await fetch('../models/moderation-tiny/model.onnx');if(!b.ok)throw Error('Tiny model download failed.');const bytes=await b.arrayBuffer();if(await digest(bytes)!==m.model_sha256)throw Error('Tiny model checksum mismatch.');labTiny={manifest:m,session:await r.ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'})};}
 const name=mode==='training'?'trained':mode==='quantization'?'int8':null;
 if(name&&!r.sessions[name]){const f=await fetch('../models/method-lab/'+name+'.onnx');if(!f.ok)throw Error('Model download failed.');const b=await f.arrayBuffer();if(await digest(b)!==labManifest.models[name].sha256)throw Error('Model checksum mismatch.');r.sessions[name]=await r.ort.InferenceSession.create(b,{executionProviders:['wasm'],graphOptimizationLevel:'all'});}
 return r;
}
async function labGraph(texts,r,session,threshold,layers,graph){
 const bounded=texts.map(text=>{const ids=r.qt.encode(text,{add_special_tokens:false});return ids.length>256?r.qt.decode(ids.slice(0,256),{skip_special_tokens:false}):text;});
 const input=r.bt(bounded,{padding:true,truncation:true,max_length:512}),feeds={};
 for(const n of ['input_ids','attention_mask','token_type_ids'])feeds[n]=new r.ort.Tensor('int64',BigInt64Array.from(input[n].data),input[n].dims);
 let output;try{output=await session.run(feeds);const data=(output.logits4||output.logits2||output.logits).data;if(data.length!==texts.length*2||Array.from(data,Number).some(v=>!Number.isFinite(v)))throw Error('Invalid classifier scores.');return texts.map((_,i)=>{const p=1/(1+Math.exp(Number(data[i*2])-Number(data[i*2+1])));return {prediction:Number(p>=threshold),probability:p,blocks:layers,calls:[graph]};});}finally{if(output)for(const t of Object.values(output))t.dispose();for(const t of Object.values(feeds))t.dispose();}
}
async function labOne(text,r,path){
 if(path==='full'||path==='adaptive'){const v=await classify(text,r,path);return {prediction:v.prediction,probability:v.probability,blocks:v.depth,calls:v.calls};}
 if(path==='fixed2')return (await labGraph([text],r,r.sessions.prefix,labManifest.fixed2_threshold,2,'prefix'))[0];
 if(path==='cascade'){
  const v=(await labGraph([text],r,labTiny.session,labTiny.manifest.threshold,2,'independent-tiny'))[0];
  if(v.probability<=labManifest.cascade.low||v.probability>=labManifest.cascade.high)return v;
  const fallback=await labOne(text,r,'full');return {...fallback,blocks:6,calls:[...v.calls,...fallback.calls]};
 }
 return (await labGraph([text],r,r.sessions[path],labManifest.models[path].threshold,4,path))[0];
}
async function labPath(tasks,r,path){
 const start=performance.now(),records=[];let graphCalls=0;
 if(path==='batch4'){for(let i=0;i<tasks.length;i+=4){const group=tasks.slice(i,i+4);const values=await labGraph(group.map(x=>x.text),r,r.sessions.full,r.manifest.threshold,4,'full-batch');for(let j=0;j<group.length;j++)records.push({id:group[j].id,expected:group[j].expected,batchId:i/4,...values[j]});graphCalls++;}}
 else for(const task of tasks)records.push({id:task.id,expected:task.expected,...await labOne(task.text,r,path)});
 if(path!=='batch4')graphCalls=records.reduce((n,r)=>n+r.calls.length,0);return {path,ms:performance.now()-start,records,graphCalls};
}
onmessage=async({data:m})=>{
 if(labBusy)return;labBusy=true;
 try{
  if(!m||!labModes[m.mode]||!['classify','benchmark'].includes(m.type))throw Error('Choose a supported comparison.');
  if(m.type==='classify'&&(typeof m.text!=='string'||!m.text.trim()||m.text.length>10000))throw Error('Enter 1 to 10,000 characters.');
  if(m.type==='benchmark'&&(!Number.isInteger(m.count)||m.count<1||m.count>100))throw Error('Choose 1 to 100 messages.');
  const begin=performance.now();const r=await labPrepare(m.mode);const setupMs=performance.now()-begin;let tasks;
  if(m.type==='classify')tasks=[{id:'your-message',expected:null,text:m.text}];
  else{send('status',{text:'Checking the real ToxicChat dataset...'});const f=await fetch(r.manifest.dataset.url);if(!f.ok)throw Error('Dataset download failed.');const b=await f.arrayBuffer();if(await digest(b)!==r.manifest.dataset.sha256)throw Error('Dataset checksum mismatch.');const rows=ModerationBenchmark.parseCSV(new TextDecoder().decode(b));tasks=r.manifest.evaluation.slice(0,m.count).map(e=>{const row=rows[Number(e.id.split(':')[1])];if(!row||row.human_annotation!=='True'||Number(row.toxicity)!==e.label)throw Error('Dataset labels mismatch.');return {id:e.id,expected:e.label,text:row.user_input};});}
  const paths=labModes[m.mode];for(const path of paths)await labPath([{id:'warmup',text:'Thank you for your help.',expected:null}],r,path);
  const runs=[];const repeats=m.type==='benchmark'?2:1;
  for(let repeat=0;repeat<repeats;repeat++)for(const path of repeat%2?[...paths].reverse():paths){send('status',{text:'Running '+path+' on '+tasks.length+' message(s), pass '+(repeat+1)+'...'});runs.push({repeat,...await labPath(tasks,r,path)});}
  for(const path of paths){const matches=runs.filter(x=>x.path===path);if(matches.length>1&&JSON.stringify(matches[0].records.map(x=>[x.id,x.prediction,x.blocks,x.calls]))!==JSON.stringify(matches[1].records.map(x=>[x.id,x.prediction,x.blocks,x.calls])))throw Error('Repeated runs changed decisions or routes. No aggregate published.');}
  const reference=runs.find(x=>x.path==='full').records,summary=paths.map(path=>{const own=runs.filter(x=>x.path===path),rows=own[0].records;return {path,n:rows.length,ms:own.reduce((s,x)=>s+x.ms,0)/own.length,correct:m.type==='benchmark'?rows.filter(x=>x.prediction===x.expected).length:null,missedToxic:m.type==='benchmark'?rows.filter(x=>x.expected===1&&x.prediction===0).length:null,falseBlocks:m.type==='benchmark'?rows.filter(x=>x.expected===0&&x.prediction===1).length:null,changed:rows.filter((x,i)=>x.prediction!==reference[i].prediction).length,addedErrors:m.type==='benchmark'?rows.filter((x,i)=>x.prediction!==x.expected&&reference[i].prediction===x.expected).length:null,correctedErrors:m.type==='benchmark'?rows.filter((x,i)=>x.prediction===x.expected&&reference[i].prediction!==x.expected).length:null,meanBlocks:rows.reduce((s,x)=>s+x.blocks,0)/rows.length,prediction:rows.length===1?rows[0].prediction:null};});
  send('complete',{report:{mode:m.mode,kind:m.type,model:'Task-trained BERT, four layers / 11.1M parameters; cascade adds separate two-layer / 4.37M BERT',threads:1,setupMs,passes:repeats,summary,runs,manifest:r.manifest.models,extraModels:labManifest.models,repeatDecisionAndRouteParity:true,tinyModel:labTiny?labTiny.manifest:null,thresholds:{full:r.manifest.threshold,fixed2:labManifest.fixed2_threshold,adaptive:r.manifest.gate,cascade:labManifest.cascade},dataset:m.type==='benchmark'?r.manifest.dataset:null,limitations:['Consumed real-message sample, not a fresh accuracy test.','Timing includes tokenization, inference, checks and output reading. Setup/download/warmup excluded. Two reverse-order passes; device-specific.','Cascade is a new BERT-to-BERT illustration; fallback repeats input encoding. 2+4 blocks are not equivalent FLOPs because widths differ.','Training mode runs existing trained weights, not browser training. Quantization mode converts BERT, not the Qwen experiment.','Only adaptive/fixed modes skip layers inside the four-layer model.'],userAgent:navigator.userAgent}});
 }catch(e){send('error',{text:e.message||String(e)});}finally{labBusy=false;}
};
