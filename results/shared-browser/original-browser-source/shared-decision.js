'use strict';
(()=>{
 const byId=id=>document.getElementById(id);let worker=null,report=null,busy=false;
 function controls(active){busy=active;byId('shared-run').disabled=active;byId('shared-classify').disabled=active;byId('shared-stop').disabled=!active;}
 function start(type){
  if(busy)return;if(typeof Worker==='undefined'||typeof WebAssembly==='undefined'||!window.crypto?.subtle){byId('shared-status').textContent='This demo needs Web Workers, WebAssembly and HTTPS or localhost.';return;}
  if(!worker){try{worker=new Worker('scripts/shared-decision-worker.js');}catch(e){byId('shared-status').textContent=e.message;return;}
   worker.onmessage=({data})=>{
    if(data.type==='status')byId('shared-status').textContent=data.text;
    if(data.type==='setup')byId('shared-setup').textContent='Setup: '+(data.ms/1000).toFixed(2)+' s, separate from processing.';
    if(data.type==='progress'){byId('shared-progress').value=data.completed;byId('shared-status').textContent=data.phase+': '+data.completed+' / '+data.total+' model runs.';}
    if(data.type==='classified'){byId('shared-output').textContent=data.result.label+' after layer '+data.result.depth+' of 4; '+data.result.ms.toFixed(1)+' ms'+(data.result.truncated?' (input shortened)':'');byId('shared-status').textContent='Finished. Experimental prediction.';controls(false);}
    if(data.type==='complete'){report=data.report;byId('shared-summary').textContent=report.correct+'/100 correct; '+report.missedToxic+' toxic messages missed, '+report.falseBlocks+' benign messages blocked. '+report.earlyCount+'/100 stopped at layer 2; '+(100*report.blocksSkippedFraction).toFixed(0)+'% of blocks skipped.';const t=report.timing;byId('shared-timing').textContent='Same 50 messages, mean of 3 paired passes: full model '+(t.fullMeanMs/1000).toFixed(2)+' s; adaptive '+(t.adaptiveMeanMs/1000).toFixed(2)+' s. '+Math.abs(t.lessTimeFraction*100).toFixed(1)+'% '+(t.lessTimeFraction>=0?'less':'more')+' time.';const list=byId('shared-rows');list.replaceChildren();for(const r of report.records){const tr=document.createElement('tr');for(const value of [r.id,r.label,r.expected?'BLOCK':'SAFE',r.depth+' / 4',r.ms.toFixed(1)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}list.append(tr);}byId('shared-save').disabled=false;byId('shared-status').textContent=report.allTokenizersMatch&&report.allPredictionsAndExitsMatch?'Finished. All decisions, exit layers and tokenized inputs match the saved reference.':'Finished with differences from the reference. See the saved report.';controls(false);}
    if(data.type==='error'){byId('shared-status').textContent='Could not finish: '+data.text;worker.terminate();worker=null;controls(false);}
   };worker.onerror=e=>{byId('shared-status').textContent='Worker failed: '+e.message;worker.terminate();worker=null;controls(false);};
  }controls(true);byId('shared-status').textContent='Starting...';worker.postMessage({type,text:byId('shared-text').value});
 }
 byId('shared-run').addEventListener('click',()=>{byId('shared-progress').value=0;start('benchmark');});byId('shared-classify').addEventListener('click',()=>start('classify'));
 byId('shared-stop').addEventListener('click',()=>{if(worker)worker.terminate();worker=null;controls(false);byId('shared-status').textContent='Stopped. Previous complete results remain available.';});
 byId('shared-save').addEventListener('click',()=>{if(!report)return;const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='shared-decision-results.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
})();
