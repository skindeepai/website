'use strict';
// Same frozen classifier, same 50 documents; remove only the prefix boundary.
process.env.UV_THREADPOOL_SIZE='1';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const root=path.resolve(__dirname,'..'),out=path.join(root,'results/privacy-full-document');
const read=name=>JSON.parse(fs.readFileSync(path.join(root,name),'utf8'));
const hash=name=>crypto.createHash('sha256').update(fs.readFileSync(path.join(root,name))).digest('hex');
const write=(name,value)=>fs.writeFileSync(path.join(out,name+'.json'),JSON.stringify(value,null,2)+'\n');
const P=require('../scripts/practical-demo-core.js');
assert(!fs.existsSync(out),'Preserve the completed experiment; repeat in a new directory.');
const original=read('results/practical-privacy/protocol.json'),reference=read('results/practical-privacy/records.json');
const spansReference=read('results/privacy-spans/result.json');
const dataPath='experiments/.cache/practical/echr_test.json';
assert.equal(hash(dataPath),original.data_sha256.test);
const modelPath='models/practical/privacy.json',model=read(modelPath);
assert.equal(hash(modelPath),read('models/practical/manifest.json').privacy.sha256);
P.prepare(model);
const documents=new Map(read(dataPath).map(d=>[d.doc_id,d]));
const dependencies=[dataPath,modelPath,'models/practical/manifest.json','results/practical-privacy/protocol.json',
 'results/practical-privacy/records.json','results/privacy-spans/result.json','scripts/practical-demo-core.js',
 'scripts/practical-demo.js','experiments/privacy_full_document.cjs'];
const protocol={scope:'Retrospective same-model coverage fix, reused test documents; no retraining, threshold search or fresh quality acceptance.',
 documents:original.ids.test,threshold:model.threshold,variants:['prefix','full'],passes:2,
 input:'Original complete text. Full uses both immediate neighbors across the old boundary. Prefix keeps original 1200-token behavior.',
 spanDefinition:spansReference.definition,
 timing:'Two alternating-order passes; one JavaScript inference thread. Tokenization, features and scoring included; labels, scoring against annotations, IO, loading and warmup excluded. Diagnostic costs, not an LLM speed comparison.',
 browserScope:'The library evaluates full documents here. The editable browser still accepts at most30000 UTF-16 code units. Documents longer than that are separately counted.',
 hashes:Object.fromEntries(dependencies.map(name=>[name,hash(name)]))};
fs.mkdirSync(out,{recursive:true});write('protocol',protocol);
fs.copyFileSync(__filename,path.join(out,'source.cjs'));
fs.copyFileSync(path.join(root,'scripts/practical-demo-core.js'),path.join(out,'core.cjs'));

function spansFor(doc){
 const offsets=[0];for(const c of doc.text)offsets.push(offsets[offsets.length-1]+c.length);
 const unique=new Map();
 for(const annotation of Object.values(doc.annotations))for(const m of annotation.entity_mentions){
  if(m.entity_type!=='PERSON')continue;
  assert(m.start_offset>=0&&m.end_offset>m.start_offset&&m.end_offset<offsets.length);
  unique.set(m.start_offset+':'+m.end_offset,{start:m.start_offset,end:m.end_offset,jsStart:offsets[m.start_offset],jsEnd:offsets[m.end_offset]});
 }
 return [...unique.values()].sort((a,b)=>a.start-b.start||a.end-b.end);
}
function score(tokens,spans){
 const confusion={tp:0,fn:0,fp:0};
 for(const token of tokens){const label=spans.some(s=>token.start<s.jsEnd&&token.end>s.jsStart);
  if(label)confusion[token.marked?'tp':'fn']++;else if(token.marked)confusion.fp++;
 }
 const counts={fully_marked:0,partly_marked:0,unmarked:0,beyond_prefix:0,crosses_prefix:0,no_token_overlap:0},occurrences=[];
 const boundary=tokens.length?tokens[tokens.length-1].end:0;
 for(const span of spans){const overlap=tokens.filter(t=>t.start<span.jsEnd&&t.end>span.jsStart);
  const marked=overlap.filter(t=>t.marked).length;
  const status=span.jsStart>=boundary?'beyond_prefix':span.jsEnd>boundary?'crosses_prefix':!overlap.length?'no_token_overlap':marked===overlap.length?'fully_marked':marked?'partly_marked':'unmarked';
  counts[status]++;occurrences.push({start:span.start,end:span.end,status,observed_tokens:overlap.length,marked_tokens:marked});
 }
 return {tokens:tokens.length,confusion,counts,occurrences,all_person_occurrences_marked:spans.length>0&&counts.fully_marked===spans.length};
}
P.privacy('Dr. Alice Morgan spoke with Mr. David Clark.',model,{fullDocument:true});
const records=[],timing=[];
for(let pass=0;pass<2;pass++)for(let index=0;index<protocol.documents.length;index++){
 const id=protocol.documents[index],doc=documents.get(id),spans=spansFor(doc),outputs={};
 const order=(index+pass)%2?['full','prefix']:['prefix','full'];
 for(const [position,variant] of order.entries()){
  const start=performance.now(),output=P.privacy(doc.text,model,{fullDocument:variant==='full'}),ms=performance.now()-start;
  timing.push({id,pass:pass+1,position:position+1,variant,ms,tokens:output.tokens.length});outputs[variant]=output;
 }
 const prefix=outputs.prefix.tokens,full=outputs.full.tokens;
 assert.equal(outputs.full.truncated,false);
 for(let i=0;i<prefix.length;i++){
  // Only the old final token can acquire a different next-token feature.
  if(i===1199&&full.length>1200)continue;
  assert.deepEqual(full[i],prefix[i]);
 }
 const values={prefix:score(prefix,spans),full:score(full,spans)};
 const old=reference.find(r=>r.id===id),oldSpans=spansReference.records.find(r=>r.id===id);
 assert.equal(prefix.length,old.tokens);
 for(const k of ['tp','fn','fp'])assert.equal(values.prefix.confusion[k],old.metrics[k],id+' '+k);
 assert.deepEqual(values.prefix.counts,oldSpans.counts);
 assert.deepEqual(values.prefix.occurrences,oldSpans.occurrences);
 assert.equal(values.full.counts.beyond_prefix+values.full.counts.crosses_prefix,0);
 if(pass===0)records.push({id,input_code_units:doc.text.length,within_browser_limit:doc.text.length<=30000,...values});
 else assert.deepEqual(values,{prefix:records[index].prefix,full:records[index].full});
}
write('records',records);write('timing',timing);
const methods={};
for(const variant of ['prefix','full']){
 const sum=key=>records.reduce((s,r)=>s+r[variant].counts[key],0);
 const calls=timing.filter(r=>r.variant===variant);
 methods[variant]={tokens:records.reduce((s,r)=>s+r[variant].tokens,0),
  confusion:Object.fromEntries(['tp','fn','fp'].map(k=>[k,records.reduce((s,r)=>s+r[variant].confusion[k],0)])),
  span_counts:Object.fromEntries(Object.keys(records[0][variant].counts).map(k=>[k,sum(k)])),
  complete_documents:records.filter(r=>r[variant].all_person_occurrences_marked).length,
  mean_ms:calls.reduce((s,r)=>s+r.ms,0)/calls.length};
}
const result={documents:records.length,person_occurrences:spansReference.person_occurrences,methods,
 documents_exceeding_browser_limit:records.filter(r=>!r.within_browser_limit).length,
 new_complete_spans:records.reduce((sum,r)=>sum+r.full.occurrences.filter((s,i)=>s.status==='fully_marked'&&r.prefix.occurrences[i].status!=='fully_marked').length,0),
 lost_complete_spans:records.reduce((sum,r)=>sum+r.full.occurrences.filter((s,i)=>s.status!=='fully_marked'&&r.prefix.occurrences[i].status==='fully_marked').length,0),
 checks:['Model and source data hashes','Original token and span metrics reproduced on every document','All tokens before old boundary retain exact features and scores','Two repeated passes produce identical predictions','No full-document occurrence beyond processed input'],
 record_sha256:hash('results/privacy-full-document/records.json'),timing_sha256:hash('results/privacy-full-document/timing.json')};
write('result',result);console.log(JSON.stringify(result,null,2));
