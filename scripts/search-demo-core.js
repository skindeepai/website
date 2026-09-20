'use strict';
(function(root){
 function words(text){return text.toLowerCase().match(/[a-z0-9]+/g)||[];}
 function index(documents){
  const postings=new Map(),lengths=[];
  documents.forEach((document,i)=>{
   const terms=new Map();for(const term of words(document.title+'\n'+document.text))terms.set(term,(terms.get(term)||0)+1);
   lengths.push(Array.from(terms.values()).reduce((a,b)=>a+b,0));
   for(const [term,count] of terms){if(!postings.has(term))postings.set(term,[]);postings.get(term).push([i,count]);}
  });
  const average=lengths.reduce((a,b)=>a+b,0)/documents.length;
  for(const list of postings.values()){
   const idf=Math.log(1+(documents.length-list.length+.5)/(list.length+.5));
   for(const entry of list){const [i,count]=entry;entry[1]=idf*count*2.2/(count+1.2*(.25+.75*lengths[i]/average));}
  }
  return postings;
 }
 function keywordScores(query,postings,count){
  const scores=new Float32Array(count);
  for(const term of new Set(words(query)))for(const [i,value] of postings.get(term)||[])scores[i]+=value;
  return scores;
 }
 function denseScores(query,vectors,count){
  if(query.length!==384||vectors.length!==count*384)throw new Error('Embedding dimensions do not match.');
  const scores=new Float32Array(count);
  for(let i=0;i<count;i++){let score=0;for(let j=0;j<384;j++)score+=query[j]*vectors[i*384+j];scores[i]=score;}
  return scores;
 }
 function rank(scores){return Array.from(scores,(_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b);}
 function fuse(left,right){
  if(left.length!==right.length)throw new Error('Candidate pools differ.');
  const scores=new Float32Array(left.length);
  for(let i=0;i<left.length;i++)scores[left[i]]+=1/(60+i+1);
  for(let i=0;i<right.length;i++)scores[right[i]]+=1/(60+i+1);
  return rank(scores);
 }
 function learnedScores(lexical,dense,left,right,model){
  if(model.coefficients.length!==5||lexical.length!==dense.length||left.length!==lexical.length||right.length!==lexical.length)throw new Error('Learned ranker inputs do not match.');
  const count=lexical.length,lr=new Float64Array(count),dr=new Float64Array(count),scores=new Float64Array(count);
  let maximum=1e-9;for(const score of lexical)maximum=Math.max(maximum,score);
  for(let i=0;i<count;i++){lr[left[i]]=i+1;dr[right[i]]=i+1;}
  const weights=model.coefficients;
  for(let i=0;i<count;i++){
   const normalized=lexical[i]/maximum;
   scores[i]=model.intercept+weights[0]*normalized+weights[1]*dense[i]+weights[2]*60/(60+lr[i])+weights[3]*60/(60+dr[i])+weights[4]*normalized*dense[i];
  }
  return scores;
 }
 function top(order,scores,documents){return order.slice(0,5).map(i=>({id:documents[i].id,title:documents[i].title,text:documents[i].text,score:scores?Number(scores[i]):null}));}
 const api={words,index,keywordScores,denseScores,rank,fuse,learnedScores,top};
 if(typeof module==='object'&&module.exports)module.exports=api;else root.SearchDemo=api;
})(typeof self==='object'?self:globalThis);
