'use strict';
(()=>{
 const B=window.ModerationBenchmark;
 const run=document.getElementById('moderation-run'),stop=document.getElementById('moderation-stop'),save=document.getElementById('moderation-save');
 const count=document.getElementById('moderation-count'),status=document.getElementById('moderation-status'),summary=document.getElementById('moderation-summary');
 const progress=document.getElementById('moderation-progress'),body=document.getElementById('moderation-rows'),setup=document.getElementById('moderation-setup');
 const comparison=document.getElementById('moderation-comparison');
 let worker=null,result=null;
 function finish(){if(worker)worker.terminate();worker=null;run.disabled=false;stop.disabled=true;count.disabled=false;}
 function seconds(ms){return(ms/1000).toFixed(2)+' s';}
 function update(){
  summary.replaceChildren();
  comparison.textContent='';
  if(!result)return;
  result.summary=B.summarize(result.rows);
  for(const row of result.summary){
   const tr=document.createElement('tr');
   const recall=row.toxic?(100*row.toxicCaught/row.toxic).toFixed(1)+'% ('+row.toxicCaught+'/'+row.toxic+')':'No toxic examples yet';
   for(const value of [{direct:'Direct decision',token:'One-token control',json:'Written JSON'}[row.format],row.correct+' / '+row.count,seconds(row.totalMs),recall,row.safeBlocked,row.invalid]){
    const cell=document.createElement(tr.childElementCount?'td':'th');if(!tr.childElementCount)cell.scope='row';cell.textContent=String(value);tr.append(cell);
   }summary.append(tr);
  }
  save.disabled=!result.rows.length;
  if(result.complete&&result.requested>0&&result.summary.every(row=>row.count===result.requested&&row.totalMs>0)){
   const direct=result.summary[0],token=result.summary[1],json=result.summary[2];
   const speed=reference=>direct.totalMs<=reference.totalMs?(reference.totalMs/direct.totalMs).toFixed(2)+' times as fast as':(direct.totalMs/reference.totalMs).toFixed(2)+' times slower than';
   comparison.textContent='Direct was '+speed(json)+' JSON, with '+direct.correct+'/'+direct.count+' correct versus '+json.correct+'/'+json.count+'. Direct was '+speed(token)+' the one-token control, with '+direct.correct+'/'+direct.count+' correct versus '+token.correct+'/'+token.count+'.';
  }
  const completedIds=new Set(result.rows.map(row=>row.id));
  setup.textContent='Model and runtime loading: '+(result.loadMs===undefined?'pending':seconds(result.loadMs))+'. Dataset download and check: '+(result.datasetMs===undefined?'pending':seconds(result.datasetMs))+'. Warm-up: '+(result.warmupMs===undefined?'pending':seconds(result.warmupMs))+'. '+completedIds.size+' message IDs reached; '+result.rows.filter(row=>row.format==='direct'&&row.truncated).length+' truncated to 256 message tokens.';
 }
 run.addEventListener('click',()=>{
  if(typeof Worker==='undefined'||typeof WebAssembly==='undefined'||!window.crypto?.subtle){status.textContent='This benchmark needs a browser with Web Workers, WebAssembly and SHA-256 on HTTPS or localhost. No download was started.';return;}
  body.replaceChildren();summary.replaceChildren();comparison.textContent='';setup.textContent='';result={complete:false,requested:Number(count.value),rows:[]};save.disabled=true;
  run.disabled=true;stop.disabled=false;count.disabled=true;progress.value=0;progress.max=result.requested*3;
  status.textContent='Starting. Downloads come from Hugging Face and jsDelivr. Messages are processed on this device.';
  try{worker=new Worker('scripts/moderation-benchmark-worker.js',{type:'module'});}catch(error){status.textContent=error.message;finish();return;}
  worker.onerror=event=>{result.error=event.message;status.textContent='Run failed: '+event.message+'. Any completed rows can still be saved.';finish();update();};
  worker.onmessage=({data})=>{
   if(data.type==='status')status.textContent=data.message;
   if(data.type==='metadata'){Object.assign(result,data.metadata);update();}
   if(data.type==='error'){result.error=data.message;status.textContent='Run failed: '+data.message+'. Any completed rows can still be saved.';finish();update();}
   if(data.type==='row'){
    result.rows.push(data.row);progress.value=data.completed;
    const row=data.row,tr=document.createElement('tr');
    for(const value of [row.id,{direct:'Direct',token:'One token',json:'JSON'}[row.format],row.expected,row.output||'(empty)',row.correct?'Correct':row.label?'Wrong':'Invalid',row.outputTokens,seconds(row.endToEndMs)]){
     const td=document.createElement('td');td.textContent=String(value);tr.append(td);
    }body.append(tr);update();
   }
   if(data.type==='complete'){
    result=data.result;update();
    const byId=new Map();for(const row of result.rows){if(!byId.has(row.id))byId.set(row.id,{});byId.get(row.id)[row.format]=row.label;}
    const different=Array.from(byId.values()).filter(pair=>pair.direct!==pair.json).length;
    const controlDifferent=Array.from(byId.values()).filter(pair=>pair.direct!==pair.token).length;
    status.textContent='Finished '+result.requested+' messages in all three formats. Direct and JSON disagreed on '+different+' messages; direct and one-token control on '+controlDifferent+'. Totals exclude loading and warm-up.';finish();
   }
  };
  worker.postMessage({type:'run',count:result.requested});
 });
 stop.addEventListener('click',()=>{if(result){result.complete=false;result.stopped=true;}finish();update();status.textContent='Stopped. The table and download contain only completed decisions; the paths may have different counts.';});
 save.addEventListener('click',()=>{
  if(!result)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=document.createElement('a');
  a.href=url;a.download='skindeep-browser-moderation'+(result.complete?'':'-partial')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
})();
