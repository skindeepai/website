// Small declared synthetic checks for remaining local preference questions.
process.env.UV_THREADPOOL_SIZE='1';
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const core=require('../scripts/preference-core.js');
const root=path.resolve(__dirname,'..'),out=path.join(root,'results/preference-followups');
const sha=x=>crypto.createHash('sha256').update(x).digest('hex');
function rng(seed){return()=>{seed|=0;seed=(seed+0x6D2B79F5)|0;let t=Math.imul(seed^(seed>>>15),1|seed);t=(t+Math.imul(t^(t>>>7),61|t))^t;return((t^(t>>>14))>>>0)/4294967296;};}
const seeds=[11,23,37,53,71];fs.mkdirSync(out,{recursive:true});
const write=(n,v)=>fs.writeFileSync(path.join(out,n),JSON.stringify(v,null,2)+'\n');
const protocol={scope:'Synthetic P05/P07/P10 follow-ups, not real generators, people, privacy or learned safety.',seeds,dimensions:4,
 P05:'Train64 uniform examples for center utility -||z-[.4,-.2,0,0]||^2, likes ifutility>-1. Generate64 new shared candidates; compare uniformly selected first candidate versus highest preference score. All64 supplied to both; no speed/rendering claim.',
 P07:'Known context IDs A/B/A/B,24 ratings per phase, opposite linear preferences z0-.6z1+.3z2>0 versus<0. Compare one all-history model, phase-reset model, and one retained model per supplied context. Retrain after each rating. Same100 held-out uniform points per seed; measure each current phase plus retainedA at end.',
 P10:'Train64 examples preferring z0+.4z1>0. Independently require z0<=0. Compare unconstrained box optimum with explicit coordinate cap. This deliberately conflict-inducing fixture tests hardknownconstraint, not learned safety.',
 source_sha256:{runner:sha(fs.readFileSync(__filename)),core:sha(fs.readFileSync(path.join(root,'scripts/preference-core.js')))},threads:1,limits:'Five fixed seeds, constructed utilities, no independent human or real-generator validation. Report every outcome.'};
const pp=path.join(out,'protocol.json');if(fs.existsSync(pp))assert.deepEqual(JSON.parse(fs.readFileSync(pp)),protocol);else write('protocol.json',protocol);
if(process.argv.includes('--prepare')){console.log('Recorded local follow-up protocol.');process.exit(0);}
assert(!fs.existsSync(path.join(out,'result.json')),'Preserve completed results.');
const records=[];
function fit(rows){const m=core.makeModel(4);m.data=rows;core.train(m);return m;}
for(const seed of seeds){
 const rand=rng(seed),utility=z=>-Array.from(z).reduce((v,x,i)=>v+(x-[.4,-.2,0,0][i])**2,0);
 const train=Array.from({length:64},()=>{const z=core.randZ(4,rand);return {z,y:Number(utility(z)>-1)};});
 const m=fit(train),candidates=Array.from({length:64},()=>core.randZ(4,rand));
 const best=candidates.reduce((a,z)=>core.predict(m,z)>core.predict(m,a)?z:a,candidates[0]);
 records.push({experiment:'P05',seed,training:train.map(r=>({z:Array.from(r.z),y:r.y})),candidates:candidates.map(z=>({z:Array.from(z),score:core.predict(m,z),utility:utility(z)})),random_utility:utility(candidates[0]),reranked_utility:utility(best)});
 const r=rng(seed+1000),test=Array.from({length:100},()=>core.randZ(4,r)),label=(z,c)=>Number((z[0]-.6*z[1]+.3*z[2])*(c==='A'?1:-1)>0);
 const global=core.makeModel(4),context={A:core.makeModel(4),B:core.makeModel(4)};let recent;
 for(const [phase,c] of ['A','B','A','B'].entries()){
  recent=core.makeModel(4);
  for(let i=0;i<24;i++){const z=core.randZ(4,r),row={z,y:label(z,c)};for(const model of [global,recent,context[c]]){model.data.push(row);core.train(model);}}
  for(const [method,model] of [['all_history',global],['phase_reset',recent],['known_context',context[c]]])records.push({experiment:'P07',seed,phase:phase+1,context:c,method,correct:test.filter(z=>Number(core.predict(model,z)>.5)===label(z,c)).length,n:100});
 }
 for(const [method,model] of [['all_history',global],['phase_reset',recent],['known_context',context.A]])records.push({experiment:'P07_retention',seed,method,context:'A',correct:test.filter(z=>Number(core.predict(model,z)>.5)===label(z,'A')).length,n:100});
 const cr=rng(seed+2000),data=Array.from({length:64},()=>{const z=core.randZ(4,cr);return{z,y:Number(z[0]+.4*z[1]>0)};});
 const cm=fit(data),plain=core.idealZ(cm,1),constrained=Array.from(plain);constrained[0]=Math.min(0,constrained[0]);
 records.push({experiment:'P10',seed,weights:Array.from(cm.w),bias:cm.b,unconstrained:Array.from(plain),constrained,unconstrained_violation:plain[0]>0,constrained_violation:constrained[0]>0,known_constraint:'z0 <= 0'});
}
write('result.json',{protocol_sha256:sha(fs.readFileSync(pp)),node:process.version,records});console.log('Completed matched candidate selection, recurring contexts and constraint-conflict fixtures.');
