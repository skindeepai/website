'use strict';
(()=>{
 const byId=id=>document.getElementById(id);let worker=null,report=null,busy=false;
 function controls(active){busy=active;byId('tiny-run').disabled=active;byId('tiny-classify').disabled=active;byId('tiny-stop').disabled=!active;}
 function start(type){
  if(busy)return;
  if(typeof Worker==='undefined'||typeof WebAssembly==='undefined'||!window.crypto?.subtle){byId('tiny-status').textContent='This demo needs Web Workers, WebAssembly and HTTPS or localhost.';return;}
  if(!worker){
   try{worker=new Worker('scripts/tiny-decision-worker.js');}
   catch(error){byId('tiny-status').textContent='Could not start the model worker: '+error.message;controls(false);return;}
   worker.onmessage=({data})=>{
   if(data.type==='status')byId('tiny-status').textContent=data.text;
   if(data.type==='setup')byId('tiny-setup').textContent='Model and tokenizer setup: '+(data.ms/1000).toFixed(2)+' s (separate from inference).';
   if(data.type==='progress'){byId('tiny-progress').value=data.completed;byId('tiny-status').textContent='Processed '+data.completed+' of 100 messages.';}
   if(data.type==='classified'){byId('tiny-output').textContent=data.result.label+' · '+data.result.ms.toFixed(1)+' ms'+(data.result.truncated?' · input shortened':'');byId('tiny-status').textContent='Finished. This is an experimental prediction.';controls(false);}
   if(data.type==='complete'){report=data.report;byId('tiny-summary').textContent=report.correct+'/100 correct · '+report.missedToxic+' toxic messages missed · '+report.falseBlocks+' benign messages blocked · '+(report.measuredMs/1000).toFixed(2)+' s total processing time.';byId('tiny-save').disabled=false;byId('tiny-status').textContent=report.allTokenizersMatch&&report.allPredictionsMatch?'Finished. All 100 decisions match the saved Python reference.':'Finished, but this runtime differs from the saved reference. See the downloaded report.';controls(false);}
   if(data.type==='error'){byId('tiny-status').textContent='Could not finish: '+data.text;worker.terminate();worker=null;controls(false);}
  };worker.onerror=event=>{byId('tiny-status').textContent='Worker failed: '+event.message;worker.terminate();worker=null;controls(false);};}
  controls(true);byId('tiny-status').textContent='Starting…';worker.postMessage({type,text:byId('tiny-text').value});
 }
 byId('tiny-run').addEventListener('click',()=>{byId('tiny-progress').value=0;start('benchmark');});
 byId('tiny-classify').addEventListener('click',()=>start('classify'));
 byId('tiny-stop').addEventListener('click',()=>{if(worker)worker.terminate();worker=null;controls(false);byId('tiny-status').textContent='Stopped. Any previous complete report remains available.';});
 byId('tiny-save').addEventListener('click',()=>{if(!report)return;const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='tiny-decision-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
})();
