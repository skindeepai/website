'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert/strict'),crypto=require('crypto');
const {enhanced,supplied,exactWithFallback}=require('./censor_decision_parsers.cjs');
const root=path.resolve(__dirname,'..'),out=path.join(root,'results/chat-output-steps');
const read=name=>JSON.parse(fs.readFileSync(path.join(out,name+'.json'),'utf8'));
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const protocol=read('protocol'),records=read('records'),complete=read('complete');
assert.equal(records.length,900);assert.equal(complete.timed_calls,900);
assert.equal(hash(path.join(out,'source.py')),protocol.source_sha256);
assert.equal(hash(path.join(root,'experiments/chat_output_steps.py')),protocol.source_sha256);
for(const [file,digest] of Object.entries(protocol.inputs))assert.equal(hash(path.join(root,file)),digest,file);
function prefix(raw,allow){
 const text=String(raw||'').trimStart().toUpperCase();
 const found=text.startsWith(allow[0])?0:text.startsWith('B')?1:null;
 return {blocked:found===null||found===1,parseOk:found!==null,normalized:text[0]||''};
}
function parse(row,mode){
 if(row.prediction!==null)return{blocked:row.prediction===1,parseOk:true};
 if(mode==='prefix')return prefix(row.raw,row.allow);
 if(mode==='supplied')return supplied(row.raw);
 if(mode==='exact')return exactWithFallback(row.raw,row.allow);
 return enhanced(row.raw);
}
function primary(row){return parse(row,row.method==='greedy1'?'prefix':row.method==='greedy8'?'enhanced':'exact');}
const mean=values=>values.reduce((a,b)=>a+b,0)/values.length;
function metrics(rows,mode){
 const scored=rows.map(row=>({row,p:mode?parse(row,mode):primary(row)}));
 return{correct:scored.filter(x=>Number(x.p.blocked)===x.row.label).length,
  missed_toxic:scored.filter(x=>x.row.label===1&&!x.p.blocked).length,
  false_block:scored.filter(x=>x.row.label===0&&x.p.blocked).length,
  unparsed:scored.filter(x=>!x.p.parseOk).length};
}
let cursor=0;
for(let pass=0;pass<2;pass++)for(let i=0;i<protocol.evaluation.length;i++){
 const shift=(i+pass)%9;let order=protocol.variants.slice(shift).concat(protocol.variants.slice(0,shift));
 if(pass===1)order.reverse();
 for(let position=0;position<order.length;position++){
  const row=records[cursor++],variant=order[position],expected=protocol.evaluation[i];
  assert.deepEqual([row.id,row.label,row.method,row.allow,row.pass,row.position],[expected.id,expected.label,...variant,pass+1,position+1]);
  const calls=row.output_ids.length||1;
  assert.equal(row.forward_passes,calls);
  assert.deepEqual(row.executed_layers,Array.from({length:24*calls},(_,i)=>i%24+1));
  assert.equal(row.vocabulary_calls,['classifier','vocabulary2'].includes(row.method)?0:calls);
  assert(Number.isFinite(row.ms)&&row.ms>0);
 }
}
const summaries=[];
for(const [method,allow]of protocol.variants){
 const all=records.filter(r=>r.method===method&&r.allow===allow),first=all.filter(r=>r.pass===1),second=all.filter(r=>r.pass===2);
 assert.equal(first.length,50);assert.equal(second.length,50);
 for(let i=0;i<50;i++){
  assert.equal(first[i].prediction,second[i].prediction);
  assert.deepEqual(first[i].output_ids,second[i].output_ids);
 }
 summaries.push({method,allow,n:50,...metrics(first),mean_ms:mean(all.map(r=>r.ms)),
  pass_ms:[1,2].map(pass=>mean(all.filter(r=>r.pass===pass).map(r=>r.ms))),
  mean_output_tokens:mean(first.map(r=>r.output_ids.length)),mean_forward_passes:mean(first.map(r=>r.forward_passes)),
  exact:metrics(first,'exact'),prefix:metrics(first,'prefix'),supplied:metrics(first,'supplied'),enhanced:metrics(first,'enhanced')});
}
for(const {id}of protocol.evaluation)for(const pass of [1,2]){
 const get=(method,allow)=>records.find(r=>r.id===id&&r.pass===pass&&r.method===method&&r.allow===allow);
 assert.equal(get('vocabulary2','SAFE').prediction,Number(get('constrained1','SAFE').raw==='BLOCK'));
 for(const allow of ['SAFE','OK','THIS IS SAFE'])assert.equal(get('greedy1',allow).output_ids[0],get('greedy8',allow).output_ids[0]);
}
const paired=[];
for(const allow of ['SAFE','OK','THIS IS SAFE']){
 const one=records.filter(r=>r.pass===1&&r.method==='greedy1'&&r.allow===allow);
 const longer=records.filter(r=>r.pass===1&&r.method==='greedy8'&&r.allow===allow);
 const changes=one.flatMap((row,i)=>{
  const a=primary(row),b=primary(longer[i]);
  return a.blocked===b.blocked?[]:[{id:row.id,label:row.label,first:row.raw,longer:longer[i].raw,
   first_prediction:Number(a.blocked),longer_prediction:Number(b.blocked),
   first_parse_ok:a.parseOk,longer_parse_ok:b.parseOk}];
 });
 paired.push({allow,changed_actions:changes.length,first_correct_longer_wrong:changes.filter(r=>r.first_prediction===r.label).length,
  first_wrong_longer_correct:changes.filter(r=>r.longer_prediction===r.label).length,changes});
}
const result={status:'passed',unique_messages:50,timed_calls:900,methods:summaries,
 paired_first_vs_longer:paired,
 checks:['Sealed source and dependency hashes','Exact predeclared execution order','Every layer/vocabulary trace','Identical decisions across timing repeats','Two vocabulary rows match constrained one-token decisions','First token matches longer generation for every paired case'],
 parser_timing:'Output parsing performed offline for all generated paths; measured runtime includes detokenization. Prior parser studies place this cost far below1ms; no parser speedup is claimed.',
 caveats:['Reused balanced50; no independent validation or production-quality claim.','Trained classifier and untrained vocabulary rows have different task supervision.','First-character scoring is only meaningful under the fixed output contract; unexpected words can have the same initial.','Two timing passes on one warmed CPU, not a statistical hardware study.'],
 analysis_sha256:hash(__filename),parser_sha256:hash(path.join(__dirname,'censor_decision_parsers.cjs')),records_sha256:hash(path.join(out,'records.json'))};
fs.writeFileSync(path.join(out,'result.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({status:result.status,methods:summaries.map(({method,allow,correct,unparsed,mean_ms,mean_output_tokens})=>({method,allow,correct,unparsed,mean_ms,mean_output_tokens}))},null,2));
