'use strict';
(async()=>{
 const select=document.getElementById('action-example'),method=document.getElementById('action-readout'),stage=document.getElementById('action-screen-stage'),image=document.getElementById('action-screen-image');
 const dot=document.getElementById('action-screen-point'),box=document.getElementById('action-screen-box'),status=document.getElementById('action-screen-status'),reveal=document.getElementById('action-reveal'),gate=document.getElementById('action-review');
 let rows=[];
 function show(){
  const row=rows[Number(select.value)];if(!row)return;
  const point=row.points[method.value],hit=row.hits[method.value],review=gate.checked&&method.value==='connected_region'&&!row.review_policies.peak_gate.accepted;
  image.src=row.image;image.alt=row.data_source+' screenshot for the instruction: '+row.instruction;
  stage.style.width=Math.min(620,360*row.width/row.height)+'px';
  document.getElementById('action-instruction').textContent=row.instruction;
  dot.style.left=point[0]*100+'%';dot.style.top=point[1]*100+'%';dot.style.opacity=review?'.4':'1';
  box.hidden=!reveal.checked;box.style.left=row.bbox[0]*100+'%';box.style.top=row.bbox[1]*100+'%';box.style.width=(row.bbox[2]-row.bbox[0])*100+'%';box.style.height=(row.bbox[3]-row.bbox[1])*100+'%';
  status.textContent=(review?'The review rule would hold this click. ':'')+(hit?'The saved point lands inside the target.':'The saved point misses the target.')+' No click is sent anywhere.';
  document.getElementById('action-screen-coordinates').textContent='Recorded point: x '+point[0].toFixed(4)+', y '+point[1].toFixed(4)+' (0 to 1).';
  document.getElementById('action-screen-full').href=row.image;
  gate.disabled=method.value!=='connected_region';
  document.getElementById('action-review-detail').textContent='The review cutoff for this platform was fitted using the other 24 screenshots. This is an exploratory replay, not a validated safety gate.';
 }
 try{
  const response=await fetch('results/action-demo/gallery.json');if(!response.ok)throw new Error('Recorded screenshots could not load.');
  const data=await response.json();rows=data.samples;
  for(let i=0;i<rows.length;i++){const option=document.createElement('option');option.value=i;option.textContent=(i+1)+' / '+rows.length+' — '+rows[i].data_source+': '+rows[i].instruction;select.append(option);}
  select.disabled=false;method.disabled=false;show();
  select.addEventListener('change',show);method.addEventListener('change',show);reveal.addEventListener('change',show);gate.addEventListener('change',show);
  document.getElementById('action-next').addEventListener('click',()=>{select.value=(Number(select.value)+1)%rows.length;show();});
 }catch(error){status.textContent=error.message||String(error);}
 image.addEventListener('error',()=>{status.textContent='The screenshot could not load. Try another recorded example.';});
})();
