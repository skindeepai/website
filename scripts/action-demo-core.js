'use strict';
(function(root){
 const SIZE=32,LABELS=['UP','RIGHT','DOWN','LEFT'];
 function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
 function inside(x,y,p){let hit=false;for(let i=0,j=p.length-1;i<p.length;j=i++){const a=p[i],b=p[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])hit=!hit;}return hit;}
 function distance(x,y,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)));return Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy);}
 function renderCard(seed,label,style='filled',canonical=false){
  if(!Number.isInteger(label)||label<0||label>3)throw new Error('Invalid arrow label');
  const random=rng(seed),cx=canonical?16:11+10*random(),cy=canonical?16:11+10*random(),scale=canonical?21:15+9*random();
  const angle=label*Math.PI/2+(canonical?0:(random()-.5)*.55),shaft=canonical ? 0.18 : 0.13+0.13*random();
  const p=[[0,-.5],[.48,0],[shaft,0],[shaft,.5],[-shaft,.5],[-shaft,0],[-.48,0]].map(([x,y])=>[cx+scale*(x*Math.cos(angle)-y*Math.sin(angle)),cy+scale*(x*Math.sin(angle)+y*Math.cos(angle))]);
  const bg=canonical?255:220+35*random(),ink=canonical?0:60*random(),pixels=new Uint8ClampedArray(SIZE*SIZE);
  for(let y=0;y<SIZE;y++)for(let x=0;x<SIZE;x++){
   const solid=inside(x+.5,y+.5,p),edge=solid&&p.some((a,i)=>distance(x+.5,y+.5,a,p[(i+1)%p.length])<1.35);
   const marked=style==='outline'?edge:solid;
   pixels[y*SIZE+x]=Math.max(0,Math.min(255,(marked?ink:bg)+(canonical?0:(random()-.5)*22)));
  }
  return pixels;
 }
 function inkPixels(gray){if(gray.length!==SIZE*SIZE)throw new Error('Expected32×32 pixels');return Float32Array.from(gray,value=>1-value/255);}
 let templates;
 function templatePredict(gray){
  if(!templates)templates=LABELS.map((_,i)=>inkPixels(renderCard(0,i,'filled',true)));
  const x=inkPixels(gray),scores=templates.map(t=>{let error=0;for(let i=0;i<x.length;i++)error+=(x[i]-t[i])**2;return-error;});
  const index=scores.indexOf(Math.max(...scores));return{index,label:LABELS[index],scores};
 }
 function predict(gray,model){
  const x=inkPixels(gray),hidden=new Float64Array(model.hidden),scores=new Array(4).fill(0);
  for(let j=0;j<model.hidden;j++){let sum=model.b1[j];for(let i=0;i<x.length;i++)sum+=model.w1[j][i]*x[i];hidden[j]=Math.max(0,sum);}
  for(let k=0;k<4;k++){let sum=model.b2[k];for(let j=0;j<hidden.length;j++)sum+=model.w2[k][j]*hidden[j];scores[k]=sum;}
  const index=scores.indexOf(Math.max(...scores));return{index,label:LABELS[index],scores};
 }
 const api={SIZE,LABELS,renderCard,inkPixels,templatePredict,predict};
 if(typeof module==='object'&&module.exports)module.exports=api;else root.ActionDemo=api;
})(typeof self==='object'?self:globalThis);
