import './moderation-benchmark-core.js';
const B=self.ModerationBenchmark;
const send=(type,data)=>postMessage({type,...data});
function disposeTree(value,seen=new Set()){
 if(!value||typeof value!=='object'||seen.has(value))return;
 seen.add(value);
 if(typeof value.dispose==='function'){value.dispose();return;}
 if(ArrayBuffer.isView(value)||value instanceof ArrayBuffer)return;
 for(const child of Object.values(value))disposeTree(child,seen);
}
let running=false;
self.onmessage=async({data})=>{
 if(data.type!=='run'||running)return;
 running=true;let model;
 try{
  if(!self.crypto?.subtle||typeof WebAssembly==='undefined')throw new Error('This benchmark needs WebAssembly and a secure page (HTTPS or localhost) with SHA-256 support.');
  const count=Number(data.count);
  if(!Number.isInteger(count)||count<1||count>100)throw new Error('Invalid message count.');
  const started=performance.now();
  send('status',{message:'Fetching the pinned ToxicChat test file and checking its SHA-256.'});
  const response=await fetch(B.DATA_URL);
  if(!response.ok)throw new Error('Dataset download failed: HTTP '+response.status);
  const bytes=await response.arrayBuffer();
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==B.DATA_SHA256)throw new Error('Dataset checksum differs. No test was run.');
  const tasks=B.selectRows(B.parseCSV(new TextDecoder().decode(bytes)),count);
  const datasetMs=performance.now()-started;
  send('status',{message:'Loading Qwen 0.5B. The first download is about 800 MB; inference uses one CPU thread.'});
  const loadStarted=performance.now();
  const {AutoTokenizer,AutoModelForCausalLM,LogitsProcessorList,env}=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js');
  env.allowLocalModels=false;env.backends.onnx.wasm.numThreads=1;
  const tokenizer=await AutoTokenizer.from_pretrained(B.MODEL,{revision:B.REVISION});
  let downloadPercent=-1;
  model=await AutoModelForCausalLM.from_pretrained(B.MODEL,{revision:B.REVISION,dtype:'q4',device:'wasm',progress_callback:p=>{
   if(p.status==='progress'&&p.file?.endsWith('.onnx')&&Math.round(p.progress)!==downloadPercent){downloadPercent=Math.round(p.progress);send('status',{message:'Downloading Qwen: '+downloadPercent+'%. The model may be cached for another run.'});}
  }});
  if(model.config.num_hidden_layers!==24)throw new Error('Unexpected number of model layers.');
  const encodedLabels=['SAFE','BLOCK'].map(label=>tokenizer.encode(label,{add_special_tokens:false}));
  if(encodedLabels.some(ids=>ids.length!==1))throw new Error('SAFE and BLOCK must each be one token for this comparison.');
  const tokenIds=encodedLabels.map(ids=>Number(ids[0]));
  const labelOnly=new LogitsProcessorList();
  labelOnly.push((_ids,logits)=>{
   const values=logits.data,vocab=logits.dims.at(-1);
   for(let offset=0;offset<values.length;offset+=vocab){
    const safe=values[offset+tokenIds[0]],block=values[offset+tokenIds[1]];
    values.fill(-Infinity,offset,offset+vocab);
    values[offset+tokenIds[0]]=safe;values[offset+tokenIds[1]]=block;
   }
   return logits;
  });
  const loadMs=performance.now()-loadStarted;
  const metadata={model:B.MODEL,modelRevision:B.REVISION,runtime:'Transformers.js 3.8.1',device:'wasm',dtype:'q4',threads:1,
   transformerLayers:24,layersSkipped:0,dataset:'ToxicChat0124 human-annotated subset',datasetRevision:B.DATA_REVISION,datasetSHA256:hash,
   datasetLicense:'CC-BY-NC-4.0',requested:count,ids:tasks.map(task=>task.id),datasetMs,loadMs,tokenIds,
   date:new Date().toISOString(),task:B.TASK,maxMessageTokens:256,jsonMaxNewTokens:16,
   limitations:['Previously inspected exploratory subset; not fresh or deployment-representative validation.',
    'Direct path reads two final vocabulary logits. It still computes the full vocabulary projection and all 24 layers; no trained task head or early exit.',
    'JSON is greedy, capped at 16 new tokens, and must parse as exactly one label field. Invalid outputs count as incorrect.',
    'Direct and one-token control share the exact prompt. JSON changes only the output instruction.',
    'One timed pass per format/message, rotating and reversing order. Download, loading and warm-up excluded from per-format totals.',
    'Timings include message tokenization/truncation, prompt preparation, inference and output parsing. Browser and hardware affect timings.',
    'Equal accuracy, if observed, does not prove identical errors or guarantee future quality.']};
  send('metadata',{metadata});
  async function run(task,format){
   const before=performance.now();
   const messageIds=tokenizer.encode(task.text,{add_special_tokens:false});
   const truncated=messageIds.length>256;
   const message=truncated?tokenizer.decode(messageIds.slice(0,256),{skip_special_tokens:false}):task.text;
   const instruction=format!=='json'?' Reply with exactly SAFE or BLOCK.':' Reply with only JSON containing exactly one label field: {"label":"SAFE"} or {"label":"BLOCK"}.';
   const inputs=tokenizer.apply_chat_template([{role:'system',content:B.TASK+instruction},{role:'user',content:message}],{tokenize:true,add_generation_prompt:true,return_dict:true});
   const inputTokens=inputs.input_ids.dims.at(-1),prepared=performance.now();
   let output,label,scores=null,outputTokens=0,modelOutput;
   try{
    if(format==='direct'){
     modelOutput=await model(inputs);
     ({label,scores}=B.chooseLogits(modelOutput.logits.data,modelOutput.logits.dims,tokenIds));
     output=label;
    }else{
     modelOutput=await model.generate({...inputs,do_sample:false,repetition_penalty:1,max_new_tokens:format==='token'?1:16,...(format==='token'?{logits_processor:labelOnly}:{})});
     const newIds=Array.from(modelOutput.data).slice(inputTokens);
     outputTokens=newIds.length;output=tokenizer.decode(newIds,{skip_special_tokens:true}).trim();label=format==='token'?(['SAFE','BLOCK'].includes(output)?output:null):B.parseReply(output);
    }
    const finished=performance.now();
    return {id:task.id,format,expected:task.expected,label,output,scores,correct:label===task.expected,
     inputTokens,originalMessageTokens:messageIds.length,truncated,outputTokens,
     preparationMs:prepared-before,inferenceAndReadoutMs:finished-prepared,endToEndMs:finished-before,layers:24};
   }finally{disposeTree(modelOutput);disposeTree(inputs);}
  }
  send('status',{message:'Warming all three paths on one separate example. Warm-up is reported separately.'});
  const warmStarted=performance.now();
  for(const format of ['direct','token','json'])await run({id:'warmup',text:'Thank you for your help.',expected:'SAFE'},format);
  metadata.warmupMs=performance.now()-warmStarted;send('metadata',{metadata});
  const rows=[],benchmarkStarted=performance.now();
  for(let i=0;i<tasks.length;i++){
   const orders=[['direct','token','json'],['json','token','direct'],['token','json','direct'],['direct','json','token'],['json','direct','token'],['token','direct','json']];
   for(const format of orders[i%orders.length]){
    send('status',{message:'Message '+(i+1)+' of '+tasks.length+': '+({direct:'reading the decision',token:'generating one label token',json:'writing the JSON reply'}[format])+'.'});
    const row=await run(tasks[i],format);rows.push(row);send('row',{row,completed:rows.length,total:tasks.length*3});
   }
  }
  metadata.benchmarkWallMs=performance.now()-benchmarkStarted;
  send('complete',{result:{...metadata,complete:true,rows,summary:B.summarize(rows)}});
 }catch(error){send('error',{message:error.message||String(error)});}
 finally{if(model)await model.dispose();running=false;}
};
