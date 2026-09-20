'use strict';
// Arithmetic and frozen ranking parity only; no model download or inference.
const assert=require('assert/strict'),fs=require('fs'),path=require('path');
const S=require('./search-demo-core.js'),root=path.resolve(__dirname,'..','results','search-ranking');
const read=name=>JSON.parse(fs.readFileSync(path.join(root,name),'utf8'));
const documents=read('corpus.json').documents,queries=read('queries.json'),predictions=read('predictions.json'),index=S.index(documents);
assert.deepEqual(S.words('IL-6, A1 and C++'),['il','6','a1','and','c']);
assert.deepEqual(S.rank(new Float32Array([1,1,0])),[0,1,2]);
assert.throws(()=>S.denseScores([1],[1],1),/dimensions/);
assert.throws(()=>S.fuse([0],[0,1]),/pools/);
for(let i=0;i<queries.length;i++){
 assert.equal(queries[i].id,predictions[i].query_id);
 assert.deepEqual(S.rank(S.keywordScores(queries[i].text,index,documents.length)).slice(0,10).map(i=>documents[i].id),predictions[i].methods.bm25.top10,'BM25 '+queries[i].id);
}
const bytes=fs.readFileSync(path.join(root,'embeddings.f32')),vectors=new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4),model=read('learned/model.json');
for(const query of read('browser-query-fixtures.json').queries){
 const lexical=S.keywordScores(query.text,index,documents.length),dense=S.denseScores(query.vector,vectors,documents.length),left=S.rank(lexical),right=S.rank(dense);
 const actual={bm25:left,minilm:right,hybrid:S.fuse(left,right),learned:S.rank(S.learnedScores(lexical,dense,left,right,model))};
 for(const key of Object.keys(actual))assert.deepEqual(actual[key].slice(0,5).map(i=>documents[i].id),query.top5[key],key+' '+query.id);
}
const protocol=read('learned/protocol.json'),fit=new Set(protocol.train_ids),dev=new Set(protocol.development_ids);
assert.equal(fit.size,600);assert.equal(dev.size,209);
for(const id of dev)assert(!fit.has(id));
for(const id of protocol.test_ids)assert(!fit.has(id)&&!dev.has(id));
assert.equal(read('learned/predictions.json').reduce((sum,row)=>sum+row['hit@1'],0),read('learned/result.json').metrics.correct_first);
console.log('Passed: 100 BM25 top-ten rankings; two-query, four-method top-five parity; split isolation and learned result total.');
