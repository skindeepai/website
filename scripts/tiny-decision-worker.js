'use strict';
importScripts('moderation-benchmark-core.js');
let runtime=null;
const send=(type,data)=>postMessage({type,...data});
async function digest(bytes){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');}
async function setup(){
 if(runtime)return runtime;
 const start=performance.now();send('status',{text:'Loading the 17.5 MB classifier, tokenizers and browser runtime…'});
 const base=new URL('../models/moderation-tiny/',location.href);
 const response=await fetch(new URL('manifest.json',base));if(!response.ok)throw Error('Model manifest could not be loaded.');
 const manifest=await response.json();
 const [{AutoTokenizer,env},ort]=await Promise.all([
  import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js'),
  import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.wasm.min.mjs')]);
 env.allowLocalModels=true;env.localModelPath=new URL('../',location.href).href;
 ort.env.wasm.numThreads=1;ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
 const [bt,qt,modelResponse]=await Promise.all([
  AutoTokenizer.from_pretrained('models/moderation-tiny',{local_files_only:true}),
  AutoTokenizer.from_pretrained(manifest.qwen_tokenizer.model,{revision:manifest.qwen_tokenizer.revision}),
  fetch(new URL('model.onnx',base))]);
 if(!modelResponse.ok)throw Error('Classifier download failed.');
 const bytes=await modelResponse.arrayBuffer();if(await digest(bytes)!==manifest.model_sha256)throw Error('Classifier checksum mismatch.');
 const session=await ort.InferenceSession.create(bytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
 runtime={manifest,bt,qt,ort,session,setupMs:performance.now()-start};
 await classify('Hello, can you help me?',runtime);
 return runtime;
}
async function classify(text,r){
 const start=performance.now();const q=r.qt.encode(text,{add_special_tokens:false});
 const bounded=q.length>256?r.qt.decode(q.slice(0,256),{skip_special_tokens:false}):text;
 const input=r.bt(bounded,{truncation:true,max_length:512});const feeds={};
 for(const name of r.session.inputNames){const t=input[name];if(!t)throw Error('Missing tokenizer input: '+name);feeds[name]=new r.ort.Tensor('int64',BigInt64Array.from(t.data),t.dims);}
 const ids=Array.from(input.input_ids.data,Number);let logits;
 try{const output=await r.session.run(feeds);logits=Array.from(output.logits.data,Number);for(const tensor of Object.values(output))tensor.dispose();}
 finally{for(const tensor of Object.values(feeds))tensor.dispose();}
 if(logits.length!==2||logits.some(x=>!Number.isFinite(x)))throw Error('Invalid classifier scores.');
 const probability=1/(1+Math.exp(logits[0]-logits[1]));
 return {label:probability>=r.manifest.threshold?'BLOCK':'SAFE',probability,ms:performance.now()-start,truncated:q.length>256,inputIds:ids};
}
onmessage=async event=>{
 try{
  const message=event.data;const r=await setup();send('setup',{ms:r.setupMs});
  if(message.type==='classify'){
   if(typeof message.text!=='string'||!message.text.trim()||message.text.length>10000)throw Error('Enter between 1 and 10,000 characters.');
   const result=await classify(message.text,r);delete result.inputIds;send('classified',{result});return;
  }
  if(message.type!=='benchmark')throw Error('Unknown operation.');
  send('status',{text:'Downloading the pinned public dataset…'});
  const response=await fetch(r.manifest.dataset.url);if(!response.ok)throw Error('Dataset download failed.');
  const bytes=await response.arrayBuffer();if(await digest(bytes)!==r.manifest.dataset.sha256)throw Error('Dataset checksum mismatch.');
  const data=ModerationBenchmark.parseCSV(new TextDecoder().decode(bytes));const records=[];const start=performance.now();
  for(const expected of r.manifest.evaluation){
   const row=data[Number(expected.id.split(':')[1])];if(!row||row.human_annotation!=='True'||Number(row.toxicity)!==expected.label)throw Error('Dataset labels differ from the pinned protocol.');
   const result=await classify(row.user_input,r);
   const tokenizerMatch=JSON.stringify(result.inputIds)===JSON.stringify(expected.input_ids);delete result.inputIds;
   records.push({id:expected.id,expected:expected.label?'BLOCK':'SAFE',...result,tokenizerMatch,
    matchesPython:result.label===(expected.prediction?'BLOCK':'SAFE'),probabilityDelta:Math.abs(result.probability-expected.block_probability)});
   send('progress',{completed:records.length,total:100});
  }
  const report={kind:'Actual browser execution, previously inspected balanced100 ToxicChat messages',model:r.manifest.model,modelSha256:r.manifest.model_sha256,
   layers:2,generatedTokens:0,threshold:r.manifest.threshold,setupMs:r.setupMs,wallMs:performance.now()-start,
   measuredMs:records.reduce((s,x)=>s+x.ms,0),correct:records.filter(x=>x.label===x.expected).length,
   missedToxic:records.filter(x=>x.expected==='BLOCK'&&x.label==='SAFE').length,falseBlocks:records.filter(x=>x.expected==='SAFE'&&x.label==='BLOCK').length,
   allTokenizersMatch:records.every(x=>x.tokenizerMatch),allPredictionsMatch:records.every(x=>x.matchesPython),maxProbabilityDelta:Math.max(...records.map(x=>x.probabilityDelta)),
   records,limitations:['Consumed balanced100, not new accuracy validation.','No paired Qwen timing; different model, training and browser runtime.','One browser pass; loading and dataset fetch excluded from measured request times.'],userAgent:navigator.userAgent};
  send('complete',{report});
 }catch(error){send('error',{text:error.message||String(error)});}
};
