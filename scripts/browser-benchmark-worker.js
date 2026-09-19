// Optional local model run. Nothing loads until the user presses Run.
const MODEL='onnx-community/Qwen2.5-0.5B-Instruct';
const REVISION='cc5cc01a65cc3ff17bdb73a7de33d879f62599b0';
const TASKS=[
 {text:'My new card still has not arrived.',label:'A'},
 {text:'Someone stole my bank card. I need to freeze it.',label:'B'},
 {text:'The cash machine gave me less money than I requested.',label:'C'}
];
const send=(type,data)=>postMessage({type,...data});
onmessage=async()=>{
 try {
  const {AutoTokenizer,AutoModelForCausalLM,env}=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js');
  env.allowLocalModels=false;
  env.backends.onnx.wasm.numThreads=1;
  const started=performance.now();
  const tokenizer=await AutoTokenizer.from_pretrained(MODEL,{revision:REVISION});
  const model=await AutoModelForCausalLM.from_pretrained(MODEL,{revision:REVISION,dtype:'q4',device:'wasm',progress_callback:p=>{
   if(p.status==='progress'&&p.file?.endsWith('.onnx'))send('status',{message:'Downloading model: '+Math.round(p.progress)+'% (about 800 MB).'});
   else if(p.status==='done')send('status',{message:'Preparing the model. The first run can take a while.'});
  }});
  const loadMs=performance.now()-started;
  async function run(task,format){
   const start=performance.now();
   const prompt='Classify this banking request. A = card delivery. B = stolen card. C = incorrect cash withdrawal.\nRequest: '+task.text+'\n'+(format==='code'?'Reply with exactly one letter: A, B, or C.':'Reply with only JSON in this format: {"label":"A"}. Choose A, B, or C.');
   const input=tokenizer.apply_chat_template([{role:'user',content:prompt}],{tokenize:true,add_generation_prompt:true,return_dict:true});
   const prepared=performance.now();
   const generated=await model.generate({...input,do_sample:false,max_new_tokens:format==='code'?1:16});
   const finished=performance.now();
   const newIds=Array.from(generated.data).slice(input.input_ids.dims.at(-1));
   const output=tokenizer.decode(newIds,{skip_special_tokens:true}).trim();
   let label=null;
   if(format==='code'&&/^[ABC]$/.test(output))label=output;
   if(format==='json'){try{const value=JSON.parse(output);if(/^[ABC]$/.test(value.label))label=value.label;}catch{}}
   const result={format,text:task.text,expected:task.label,output,label,valid:label!==null,correct:label===task.label,
    inputTokens:input.input_ids.dims.at(-1),outputTokens:newIds.length,tokenizationMs:prepared-start,inferenceMs:finished-prepared,endToEndMs:finished-start};
   generated.dispose?.();
   for(const value of Object.values(input))value.dispose?.();
   return result;
  }
  send('status',{message:'Warming both output formats. Download and warm-up are excluded from the comparison.'});
  await run(TASKS[0],'code');await run(TASKS[0],'json');
  const rows=[];
  for(let repeat=0;repeat<2;repeat++)for(let i=0;i<TASKS.length;i++){
   for(const format of ((i+repeat)%2?['json','code']:['code','json'])){
    send('status',{message:'Running comparison '+(rows.length+1)+' of 12…'});
    const row={repeat,...await run(TASKS[i],format)};rows.push(row);send('row',{row});
   }
  }
  await model.dispose();
  send('complete',{result:{model:MODEL,revision:REVISION,library:'Transformers.js 3.8.1',device:'wasm',dtype:'q4',threads:1,loadMs,
   date:new Date().toISOString(),rows,limitations:['Both paths run all 24 layers; this is not early exit or the trained internal classifier.',
   'Three authored examples, two repeats: a local timing demonstration, not task-quality validation.',
   'One-code output is unconstrained; invalid outputs and failed JSON are retained.',
   'Prompts differ by output-format instruction. Timing includes tokenization and generation but excludes loading and warm-up.',
   'Browser/device performance varies. No server inference or result upload.']}});
 }catch(error){send('error',{message:error.message||String(error)});}
};
