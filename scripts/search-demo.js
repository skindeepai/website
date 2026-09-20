'use strict';
(()=>{
 const form=document.getElementById('search-form'),query=document.getElementById('search-query'),run=document.getElementById('search-run'),compare=document.getElementById('search-compare'),stop=document.getElementById('search-stop');
 const status=document.getElementById('search-status'),timing=document.getElementById('search-timing'),keyword=document.getElementById('search-keywords'),meaning=document.getElementById('search-meaning'),combined=document.getElementById('search-combined'),learned=document.getElementById('search-learned');
 let worker;
 function controls(running){run.disabled=running;compare.disabled=running;stop.disabled=!running;}
 function list(target,rows){
  target.replaceChildren();
  for(const row of rows){
   const item=document.createElement('li'),title=document.createElement('h3'),snippet=document.createElement('p'),details=document.createElement('details'),summary=document.createElement('summary'),abstract=document.createElement('p'),source=document.createElement('a');
   title.textContent=row.title;snippet.textContent=row.text.slice(0,220)+(row.text.length>220?'…':'');summary.textContent='Read abstract';abstract.textContent=row.text;
   source.href='https://huggingface.co/datasets/BeIR/scifact';source.textContent='SciFact corpus · document '+row.id;
   details.append(summary,abstract,source);item.append(title,snippet,details);target.append(item);
  }
 }
 function search(withModel){
  if(!query.value.trim()){status.textContent='Enter a search query first.';query.focus();return;}
  if(typeof Worker==='undefined'||!window.crypto?.subtle){status.textContent='This demo needs Web Workers and HTTPS or localhost for dataset checks.';return;}
  if(withModel&&typeof WebAssembly==='undefined'){status.textContent='Meaning search needs WebAssembly. Keyword search is still available.';return;}
  controls(true);timing.textContent='';keyword.replaceChildren();meaning.replaceChildren();combined.replaceChildren();learned.replaceChildren();
  status.textContent=withModel?'Comparing keywords and meaning. The model runs on your device.':'Searching by keywords.';
  if(!worker){
   try{worker=new Worker('scripts/search-demo-worker.js',{type:'module'});}catch(error){status.textContent='Search could not start: '+error.message;controls(false);return;}
   worker.onerror=event=>{status.textContent='Search could not run: '+event.message;worker.terminate();worker=null;controls(false);};
   worker.onmessage=({data})=>{
    if(data.type==='status')status.textContent=data.message;
    if(data.type==='error'){status.textContent='Search could not run: '+data.message;controls(false);}
    if(data.type==='result'){
     const result=data.result;list(keyword,result.bm25.results);
     if(result.minilm){list(meaning,result.minilm.results);list(combined,result.hybrid.results);list(learned,result.learned.results);}
     else{const note=document.createElement('li');note.textContent='Choose Compare meaning to run MiniLM on the same query.';meaning.append(note);}
     const ms=value=>value.toFixed(1)+' ms';
     timing.textContent='Keywords: '+ms(result.bm25.ms)+(result.minilm?'. Meaning: '+ms(result.minilm.ms)+', including query encoding.':'')+' Collection setup: '+(result.setup.corpusMs/1000).toFixed(2)+' s'+(result.setup.modelMs===undefined?'':'; model and vector loading: '+(result.setup.modelMs/1000).toFixed(2)+' s')+'. Setup is excluded from query times.';
     status.textContent='Searched '+result.candidates.toLocaleString()+' abstracts. Results are ranked documents, not a generated answer.'+(result.minilm?.truncated?' The model used the first 256 query tokens.':'');
     controls(false);
    }
   };
  }
  worker.postMessage({type:'search',query:query.value,compare:withModel});
 }
 form.addEventListener('submit',event=>{event.preventDefault();search(false);});compare.addEventListener('click',()=>search(true));
 stop.addEventListener('click',()=>{worker?.terminate();worker=null;controls(false);status.textContent='Stopped. You can start another search.';});
 document.querySelectorAll('[data-search-example]').forEach(button=>button.addEventListener('click',()=>{query.value=button.dataset.searchExample;query.focus();}));
})();
