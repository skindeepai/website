'use strict';
importScripts('moderation-benchmark-core.js');
let runtime=null;
const send=(type,data)=>postMessage({type,...data});
async function digest(bytes){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');}
function probability(t){const a=Array.from(t.data,Number);if(a.length!==2||a.some(x=>!Number.isFinite(x)))throw Error('Invalid classifier scores.');return 1/(1+Math.exp(a[0]-a[1]));}
async function setup(){
 if(runtime)return runtime;
 const start=performance.now();send('status',{text:'Loading the shared model (about 44.5 MB), tokenizers and runtime...'});
 const base=new URL('../models/shared-moderation/',location.href);const response=await fetch(new URL('manifest.json',base));if(!response.ok)throw Error('Manifest could not be loaded.');const manifest=await response.json();
 const [{AutoTokenizer,env},ort]=await Promise.all([import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js'),import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.wasm.min.mjs')]);
 env.allowLocalModels=true;env.localModelPath=new URL('../',location.href).href;ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
 const bt=await AutoTokenizer.from_pretrained('models/shared-moderation',{local_files_only:true});const qt=await AutoTokenizer.from_pretrained(manifest.qwen_tokenizer.model,{revision:manifest.qwen_tokenizer.revision});const sessions={};
 const r={manifest,bt,qt,ort,sessions,base,adaptiveSetupMs:0,controlSetupMs:0,setupMs:0};
 for(const name of ['prefix','suffix'])await loadGraph(r,name);
 await classify('Hello, can you help me?',r,'adaptive',true);
 r.adaptiveSetupMs=performance.now()-start;r.setupMs=r.adaptiveSetupMs;runtime=r;return r;
}
async function loadGraph(r,name){
 const response=await fetch(new URL(name+'.onnx',r.base));if(!response.ok)throw Error(name+' download failed.');const bytes=await response.arrayBuffer();if(await digest(bytes)!==r.manifest.models[name].sha256)throw Error(name+' checksum mismatch.');r.sessions[name]=await r.ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
}
async function ensureControl(r){
 if(r.sessions.full)return;
 const start=performance.now();send('status',{text:'Loading the additional 44.5 MB full-depth control for the benchmark...'});
 await loadGraph(r,'full');await classify('Hello, can you help me?',r,'full');
 r.controlSetupMs=performance.now()-start;r.setupMs=r.adaptiveSetupMs+r.controlSetupMs;
}
async function classify(text,r,path,forceContinue=false){
 const start=performance.now();const q=r.qt.encode(text,{add_special_tokens:false});const bounded=q.length>256?r.qt.decode(q.slice(0,256),{skip_special_tokens:false}):text;const input=r.bt(bounded,{truncation:true,max_length:512});const feeds={};
 for(const name of ['input_ids','attention_mask','token_type_ids']){const t=input[name];if(!t)throw Error('Missing input '+name);feeds[name]=new r.ort.Tensor('int64',BigInt64Array.from(t.data),t.dims);}
 const ids=Array.from(input.input_ids.data,Number);let p,p2=null,depth=4;const owned=[];const calls=[];
 try{
  if(path==='full'){calls.push('full');const output=await r.sessions.full.run(feeds);owned.push(...Object.values(output));p=probability(output.logits4);}
  else{calls.push('prefix');const prefix=await r.sessions.prefix.run(feeds);owned.push(...Object.values(prefix));p2=probability(prefix.logits2);
   if(!forceContinue&&(p2<=r.manifest.gate.low||p2>=r.manifest.gate.high)){p=p2;depth=2;}
   else{calls.push('suffix');const output=await r.sessions.suffix.run({hidden_states:prefix.hidden_states,attention_mask:feeds.attention_mask});owned.push(...Object.values(output));p=probability(output.logits4);}
  }
 }finally{for(const t of owned)t.dispose();for(const t of Object.values(feeds))t.dispose();}
 const prediction=depth===2?Number(p2>=r.manifest.gate.high):Number(p>=r.manifest.threshold);
 return {prediction,label:prediction?'BLOCK':'SAFE',probability:p,layer2Probability:p2,depth,calls,ms:performance.now()-start,truncated:q.length>256,inputIds:ids};
}
onmessage=async({data:message})=>{try{
 if(!message||!['classify','benchmark'].includes(message.type))throw Error('Unknown operation.');
 if(message.type==='classify'&&(typeof message.text!=='string'||!message.text.trim()||message.text.length>10000))throw Error('Enter between 1 and 10,000 characters.');
 const previousSetupMs=runtime?runtime.setupMs:0;const r=await setup();
 if(message.type==='benchmark')await ensureControl(r);
 const setupInfo={adaptiveMs:r.adaptiveSetupMs,controlMs:r.controlSetupMs,cumulativeMs:r.setupMs,incrementalMs:r.setupMs-previousSetupMs};send('setup',setupInfo);
 if(message.type==='classify'){const result=await classify(message.text,r,'adaptive');delete result.inputIds;send('classified',{result});return;}
 send('status',{text:'Downloading and checking the public dataset...'});const response=await fetch(r.manifest.dataset.url);if(!response.ok)throw Error('Dataset download failed.');const bytes=await response.arrayBuffer();if(await digest(bytes)!==r.manifest.dataset.sha256)throw Error('Dataset checksum mismatch.');const data=ModerationBenchmark.parseCSV(new TextDecoder().decode(bytes));
 const rowFor=e=>{const row=data[Number(e.id.split(':')[1])];if(!row||row.human_annotation!=='True'||Number(row.toxicity)!==e.label)throw Error('Dataset labels differ from protocol.');return row;};
 const records=[],timing=[];
 for(const e of r.manifest.evaluation){const result=await classify(rowFor(e).user_input,r,'adaptive');const tokenizerMatch=JSON.stringify(result.inputIds)===JSON.stringify(e.input_ids);delete result.inputIds;records.push({id:e.id,expected:e.label,...result,tokenizerMatch,matchesPython:result.prediction===e.prediction&&result.depth===(e.early?2:4),probabilityDelta:Math.abs(result.probability-(e.early?e.layer2_probability:e.layer4_probability))});send('progress',{completed:records.length,total:400,phase:'Checking 100 messages'});}
 for(let repeat=0;repeat<3;repeat++)for(let i=0;i<50;i++){const e=r.manifest.evaluation[i];const paths=(i+repeat)%2?['adaptive','full']:['full','adaptive'];for(const path of paths){const result=await classify(rowFor(e).user_input,r,path);const tokenizerMatch=JSON.stringify(result.inputIds)===JSON.stringify(e.input_ids);delete result.inputIds;timing.push({repeat,path,id:e.id,expected:e.label,...result,tokenizerMatch,matchesPython:result.prediction===(path==='full'?e.full_prediction:e.prediction)&&result.depth===(path==='full'||!e.early?4:2)});send('progress',{completed:100+timing.length,total:400,phase:'Timing paired runs'});}}
 const totals=path=>[0,1,2].map(n=>timing.filter(x=>x.path===path&&x.repeat===n).reduce((s,x)=>s+x.ms,0));const full=totals('full'),adaptive=totals('adaptive');const mean=a=>a.reduce((s,x)=>s+x,0)/a.length;
 const report={kind:'Actual browser shared 2/4-layer BERT, consumed100 parity and paired first50 timing',model:r.manifest.model,models:r.manifest.models,gate:r.manifest.gate,threshold:r.manifest.threshold,generatedTokens:0,setupMs:r.setupMs,setup:setupInfo,correct:records.filter(x=>x.prediction===x.expected).length,missedToxic:records.filter(x=>x.expected===1&&x.prediction===0).length,falseBlocks:records.filter(x=>x.expected===0&&x.prediction===1).length,earlyCount:records.filter(x=>x.depth===2).length,blocksSkippedFraction:records.reduce((s,x)=>s+4-x.depth,0)/400,allTokenizersMatch:[...records,...timing].every(x=>x.tokenizerMatch),allPredictionsAndExitsMatch:[...records,...timing].every(x=>x.matchesPython),maxProbabilityDelta:Math.max(...records.map(x=>x.probabilityDelta)),timing:{n:50,passes:3,fullMs:full,adaptiveMs:adaptive,fullMeanMs:mean(full),adaptiveMeanMs:mean(adaptive),lessTimeFraction:1-mean(adaptive)/mean(full)},records,timingRecords:timing,userAgent:navigator.userAgent,limitations:['Consumed balanced sample; not fresh quality validation.','Paired request timings include tokenization, graph calls, boundary transfer, classifier and cleanup; setup and dataset download excluded.','One WASM thread, same trained weights; full control uses one unsplit graph.','Browser timing is device-specific; 100 quality messages differ in count from 50 timed messages.']};send('complete',{report});
 }catch(error){send('error',{text:error.message||String(error)});}};
