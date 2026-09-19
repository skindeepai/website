'use strict';
(() => {
 const run=document.getElementById('benchmark-run'),stop=document.getElementById('benchmark-stop'),save=document.getElementById('benchmark-save');
 const status=document.getElementById('benchmark-status'),body=document.getElementById('benchmark-rows'),summary=document.getElementById('benchmark-summary');
 let worker=null,result=null;
 function finish(){worker?.terminate();worker=null;run.disabled=false;stop.disabled=true;}
 run.addEventListener('click',()=>{
  body.replaceChildren();summary.textContent='';result=null;save.disabled=true;run.disabled=true;stop.disabled=false;
  status.textContent='Loading the local model. Download comes from Hugging Face; runtime comes from jsDelivr.';
  worker=new Worker('scripts/browser-benchmark-worker.js',{type:'module'});
  worker.onerror=event=>{status.textContent='The browser could not run the model: '+event.message;finish();};
  worker.onmessage=({data})=>{
   if(data.type==='status')status.textContent=data.message;
   if(data.type==='error'){status.textContent='The browser could not run the model: '+data.message;finish();}
   if(data.type==='row'){
    const r=data.row,tr=document.createElement('tr');
    for(const value of [r.text,r.format==='code'?'One letter':'JSON',r.output||'(empty)',r.expected,r.correct?'Correct':r.valid?'Wrong label':'Invalid output',r.outputTokens,Math.round(r.endToEndMs)+' ms']){
     const td=document.createElement('td');td.textContent=String(value);tr.append(td);
    }
    body.append(tr);
   }
   if(data.type==='complete'){
    result=data.result;save.disabled=false;
    const median=values=>{values.sort((a,b)=>a-b);return(values[2]+values[3])/2;};
    summary.textContent=['code','json'].map(format=>{
     const rows=result.rows.filter(r=>r.format===format);
     return(format==='code'?'One letter':'JSON')+': '+Math.round(median(rows.map(r=>r.endToEndMs)))+' ms median; '+rows.filter(r=>r.correct).length+'/'+rows.length+' correct.';
    }).join(' ');
    status.textContent='Finished. All outputs, including errors, are shown below.';finish();
   }
  };
  worker.postMessage({type:'run'});
 });
 stop.addEventListener('click',()=>{finish();status.textContent='Stopped. No complete benchmark result was produced.';});
 save.addEventListener('click',()=>{
  if(!result)return;
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=url;a.download='skindeep-browser-benchmark.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 });
})();
