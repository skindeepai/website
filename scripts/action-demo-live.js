'use strict';
(async()=>{
 const A=window.ActionDemo,canvas=document.getElementById('action-card'),ctx=canvas.getContext('2d',{willReadFrequently:true}),status=document.getElementById('action-live-status');
 const result=document.getElementById('action-live-result'),next=document.getElementById('action-new-card'),upload=document.getElementById('action-upload'),benchmark=document.getElementById('action-test');
 let model,generatedLabel=null,seed=Date.now()>>>0;
 const arrows=['↑','→','↓','←'];
 function pixels(){const rgba=ctx.getImageData(0,0,32,32).data,gray=new Uint8ClampedArray(1024);for(let i=0;i<1024;i++)gray[i]=.2126*rgba[i*4]+.7152*rgba[i*4+1]+.0722*rgba[i*4+2];return gray;}
 function classify(){
  if(!model)return;
  const before=performance.now(),answer=A.predict(pixels(),model),elapsed=performance.now()-before;
  result.textContent=arrows[answer.index]+' '+answer.label;
  status.textContent=generatedLabel===null?'Read from your uploaded image. This model only learned synthetic dark arrows on a light background.':(answer.index===generatedLabel?'Correct for this generated card.':'Wrong for this generated card.')+' The classifier read only the pixels.';
  document.getElementById('action-live-detail').textContent='One model evaluation: '+elapsed.toFixed(2)+' ms on this device, including reading the 32×32 pixels. Output is one of four directions; no text tokens are generated.';
 }
 function newCard(){
  seed=(seed+1)>>>0;generatedLabel=seed%4;const gray=A.renderCard(seed,generatedLabel),data=ctx.createImageData(32,32);
  for(let i=0;i<gray.length;i++){data.data[i*4]=data.data[i*4+1]=data.data[i*4+2]=gray[i];data.data[i*4+3]=255;}
  ctx.putImageData(data,0,0);canvas.setAttribute('aria-label','Generated pixel arrow pointing '+A.LABELS[generatedLabel]);classify();
 }
 next.addEventListener('click',newCard);
 upload.addEventListener('change',()=>{
  const file=upload.files[0];if(!file)return;
  if(!/^image\/(png|jpeg|webp)$/.test(file.type)||file.size>5000000){status.textContent='Choose a PNG, JPEG or WebP arrow image under 5 MB.';return;}
  const url=URL.createObjectURL(file),image=new Image();
  image.onload=()=>{try{const scale=Math.min(32/image.naturalWidth,32/image.naturalHeight),w=image.naturalWidth*scale,h=image.naturalHeight*scale;ctx.fillStyle='#fff';ctx.fillRect(0,0,32,32);ctx.drawImage(image,(32-w)/2,(32-h)/2,w,h);generatedLabel=null;canvas.setAttribute('aria-label','Uploaded image reduced to32 by32 pixels');classify();}finally{URL.revokeObjectURL(url);}};
  image.onerror=()=>{URL.revokeObjectURL(url);status.textContent='This image could not be opened.';};image.src=url;
 });
 benchmark.addEventListener('click',async()=>{
  benchmark.disabled=true;const output=document.getElementById('action-test-output');output.textContent='Loading the fixed test cards…';
  try{
   const response=await fetch('results/action-demo/arrows/evaluation.json');if(!response.ok)throw new Error('Test definitions could not load.');const rows=await response.json();
   const counts={test:{n:0,neural:0,template:0},outline_stress:{n:0,neural:0,template:0}};let parity=0;
   for(let i=0;i<rows.length;i++){
    const row=rows[i],gray=A.renderCard(row.seed,row.label,row.style),neural=A.predict(gray,model),template=A.templatePredict(gray),c=counts[row.split];
    c.n++;c.neural+=neural.index===row.label;c.template+=template.index===row.label;parity+=neural.index===row.torch_prediction;
    if(i%25===0){output.textContent='Checking pixels: '+i+' / '+rows.length;await new Promise(resolve=>setTimeout(resolve,0));}
   }
   output.replaceChildren();
   for(const [name,c] of Object.entries(counts)){const p=document.createElement('p');p.textContent=(name==='test'?'New filled arrows':'Unseen outline style')+': trained model '+c.neural+' / '+c.n+'; simple templates '+c.template+' / '+c.n+'.';output.append(p);}
   const note=document.createElement('p');note.textContent='Your browser matched the saved model decisions on '+parity+' / '+rows.length+' cards. These are synthetic images, not real-screen accuracy.';output.append(note);
  }catch(error){output.textContent=error.message||String(error);}finally{benchmark.disabled=false;}
 });
 try{
  const response=await fetch('results/action-demo/arrows/model.json');if(!response.ok)throw new Error('The small trained model could not load.');model=await response.json();
  if(model.input_pixels!==1024||model.hidden!==32||model.parameter_count!==32932)throw new Error('Unexpected model shape.');
  next.disabled=upload.disabled=benchmark.disabled=false;newCard();
 }catch(error){status.textContent=error.message||String(error);}
})();
