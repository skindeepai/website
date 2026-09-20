'use strict';
// Run with node scripts/moderation-benchmark-test.js. No network or model download.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const B=require('./moderation-benchmark-core.js');
const root=path.resolve(__dirname,'..');
const protocol=JSON.parse(fs.readFileSync(path.join(root,'results/chat-smoke/protocol.json'),'utf8'));
assert.deepEqual(B.IDS.map(index=>'test:'+index),protocol.splits.evaluation);
assert.equal(new Set(B.IDS).size,100);
assert.deepEqual(B.parseCSV('a,b\r\n"x,y","a""b"\r\n"multi\nline",z\r\n'),[{a:'x,y',b:'a"b'},{a:'multi\nline',b:'z'}]);
assert.deepEqual(B.parseCSV('\ufeffa,b\nx,y'),[{a:'x',b:'y'}]);
for(const text of ['a,b\n"bad,z','a,b\nx,y,z','a,a\nx,y','a,b\n"x"tail,y'])assert.throws(()=>B.parseCSV(text));
assert.equal(B.parseReply('{"label":"BLOCK"}'),'BLOCK');
for(const output of ['SAFE','```json\n{"label":"SAFE"}\n```','{"label":"SAFE","extra":1}','null','[]','{"label":"safe"}','{"label":'])assert.equal(B.parseReply(output),null);
assert.deepEqual(B.chooseLogits([99,9,1,99,2,8],[1,2,3],[1,2]),{label:'BLOCK',scores:[2,8]});
assert.equal(B.chooseLogits([0,5,5],[1,1,3],[2,1]).label,'BLOCK');
assert.equal(B.chooseLogits([0,5,5],[1,1,3],[1,2]).label,'SAFE');
assert.throws(()=>B.chooseLogits([0,NaN,2],[1,1,3],[1,2]));
assert.throws(()=>B.chooseLogits([1,2],[1,2],[0,1]));
const totals=B.summarize([
 {format:'direct',expected:'BLOCK',label:'SAFE',endToEndMs:3,outputTokens:0},
 {format:'json',expected:'BLOCK',label:null,endToEndMs:8,outputTokens:16},
 {format:'json',expected:'SAFE',label:'BLOCK',endToEndMs:7,outputTokens:8},
 {format:'token',expected:'SAFE',label:'SAFE',endToEndMs:4,outputTokens:1,truncated:true}
]);
assert.equal(totals[0].toxicMissed,1);
assert.equal(totals[1].correct,1);assert.equal(totals[1].truncated,1);
assert.equal(totals[2].invalid,1);assert.equal(totals[2].safeBlocked,1);assert.equal(totals[2].totalMs,15);
assert.equal(totals[2].toxicMissed,1);assert.equal(totals[2].toxicCaught,0);
const csv=path.join(root,'experiments/.cache/toxicchat/toxic-chat_annotation_test.csv');
if(fs.existsSync(csv)){
 const bytes=fs.readFileSync(csv);assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),B.DATA_SHA256);
 const rows=B.selectRows(B.parseCSV(bytes.toString('utf8')),100);
 assert.equal(rows.filter(row=>row.expected==='BLOCK').length,50);
 assert.equal(rows.filter(row=>row.expected==='SAFE').length,50);
 console.log('Verified cached official dataset checksum, parser and all 100 selected labels.');
}
console.log('Passed pinned IDs, CSV edge cases, strict outputs, final-position logits, ties and error metrics.');
