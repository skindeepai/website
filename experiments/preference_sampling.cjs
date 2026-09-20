// Single-threaded synthetic check of the actual browser sampler and training core.
process.env.UV_THREADPOOL_SIZE = '1';
const fs = require('fs'), path = require('path'), vm = require('vm'), crypto = require('crypto');
const assert = require('assert/strict');
const root = path.resolve(__dirname, '..');
const out = path.join(root, 'results/preference-sampling');
const core = require('../scripts/preference-core.js');
const source = fs.readFileSync(path.join(root, 'scripts/demo.js'), 'utf8');
function extract(name, end) {
    const start = source.indexOf('    function ' + name + '(');
    assert(start >= 0);
    const finish = source.indexOf('    function ' + end + '(', start);
    assert(finish > start);
    return source.slice(start, finish);
}
const sampler = extract('nextBatch', 'nearIdeals');
const rngSource = extract('mulberry32', 'lerp');
const context = vm.createContext({predict: core.predict, randZ: core.randZ});
vm.runInContext(rngSource + sampler, context);
const seeds = [11,23,37,53,71], checkpoints = [12,24,48,72,96];
const policies = ['random', 'uncertainty', 'browser_mixed'];
const scenarios = ['linear', 'noisy_linear', 'two_modes'];
const hash = x => crypto.createHash('sha256').update(x).digest('hex');
const write = (name, value) => fs.writeFileSync(path.join(out,name),JSON.stringify(value,null,2)+'\n');
fs.mkdirSync(out,{recursive:true});
const protocol = {
    scope:'P03 exploratory synthetic comparison; no people or image-generator inference.',
    seeds,checkpoints,policies,scenarios,dimensions:16,evaluation_per_seed:100,
    training:'Exact browser preference-core.train after every rating; 260 epochs, reset weights, lr0.5, L2 .02.',
    sampling:'Exact extracted browser 5/4/3 nextBatch function, 380 candidates, 12 shuffled selections. Queue refreshes after every six ratings as in the UI, so only six of each batch are rated.',
    controls:'Random and uncertainty use identical 380-candidate pools per refresh and consume identical RNG counts. All policies receive the same first six examples. All score every candidate; no timing/satisfaction claim.',
    labels:'Linear: z0-.6*z1+.3*z2>0. Noisy: same utility with independent 15% training-label flips, evaluation uses clean utility. Two modes: abs(z0)>.55 and abs(z1)<.65.',
    selection:'All three methods, scenarios, seeds and five checkpoints reported. No outcome selects configurations.',
    metric:'Balanced accuracy on an independent uniform 100-example pool per seed, plus ordinary accuracy and clean oracle utility among rated examples. First sampled checkpoint >=90% balanced accuracy, null if none.',
    source_sha256:{demo:hash(source),core:hash(fs.readFileSync(path.join(root,'scripts/preference-core.js'))),runner:hash(fs.readFileSync(__filename)),sampler:hash(sampler)},
    limits:'Five seeds and authored preferences; seed ranges are not confidence intervals. Does not finish human UX/label-efficiency validation.'
};
const pp=path.join(out,'protocol.json');
if(fs.existsSync(pp)) assert.deepEqual(JSON.parse(fs.readFileSync(pp)),protocol);
else write('protocol.json',protocol);
if(process.argv.includes('--prepare')) {console.log('Recorded prospective preference sampling protocol.');process.exit(0);}
assert(!fs.existsSync(path.join(out,'result.json')),'Preserve completed run; reproduce in a separate copy.');
function shuffle(a,rand) {for(let i=a.length-1;i>0;i--){const j=Math.floor(rand()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
function batch(m,policy,rand) {
    if(policy==='browser_mixed')return context.nextBatch(m,true,rand);
    if(m.data.length<6)return Array.from({length:12},()=>core.randZ(m.d,rand));
    const pool=Array.from({length:380},()=>{const z=core.randZ(m.d,rand);return {z,p:core.predict(m,z)};});
    const uncertain=pool.slice().sort((a,b)=>Math.abs(a.p-.5)-Math.abs(b.p-.5)).slice(0,12);
    shuffle(pool,rand);
    return shuffle((policy==='random'?pool.slice(0,12):uncertain).map(r=>r.z),rand);
}
function label(z,scenario){return Number(scenario==='two_modes'?Math.abs(z[0])>.55&&Math.abs(z[1])<.65:z[0]-.6*z[1]+.3*z[2]>0);}
const records=[],runs=[],fixtures=[];
for(const seed of seeds){
    const testRand=context.mulberry32(seed+900001);
    const test=Array.from({length:100},()=>Array.from(core.randZ(16,testRand)));
    fixtures.push({seed,points:test});
    for(const scenario of scenarios){
        const finalRng=[];
        for(const policy of policies){
            const rand=context.mulberry32(seed),noise=context.mulberry32(seed+1800001),m=core.makeModel(16);
            let queue=batch(m,policy,rand),first90=null;const history=[];
            for(let n=1;n<=96;n++){
                const z=queue.shift(),clean=label(z,scenario),flip=noise()<.15;
                const y=scenario==='noisy_linear'&&flip?1-clean:clean;
                m.data.push({z,y});history.push({z:Array.from(z),y,clean});core.train(m);
                if(queue.length===0||n%6===0)queue=batch(m,policy,rand);
                if(!checkpoints.includes(n))continue;
                const decisions=test.map(z=>({expected:label(z,scenario),prediction:Number(core.predict(m,z)>.5)}));
                const positive=decisions.filter(r=>r.expected),negative=decisions.filter(r=>!r.expected);
                assert(positive.length&&negative.length);
                const recall=positive.filter(r=>r.prediction).length/positive.length;
                const specificity=negative.filter(r=>!r.prediction).length/negative.length;
                const balanced_accuracy=(recall+specificity)/2;
                if(first90===null&&balanced_accuracy>=.9)first90=n;
                records.push({seed,scenario,policy,ratings:n,correct:decisions.filter(r=>r.prediction===r.expected).length,balanced_accuracy,positive:positive.length,recall,specificity,clean_likes_shown:history.reduce((a,r)=>a+r.clean,0),weights:Array.from(m.w),bias:m.b,decisions});
            }
            finalRng.push(rand.state());runs.push({seed,scenario,policy,first_sampled_90:first90,history});
        }
        assert(finalRng.every(r=>r===finalRng[0]),'Candidate RNG budgets differ');
        const starts=runs.slice(-3).map(r=>JSON.stringify(r.history.slice(0,6)));
        assert(starts.every(r=>r===starts[0]),'Warmup ratings differ');
    }
    console.log('Completed seed '+seed);
}
write('fixtures.json',fixtures);write('runs.json',runs);write('result.json',{protocol_sha256:hash(fs.readFileSync(pp)),node:process.version,compute_threads:1,records});
console.log('Completed '+runs.length+' learning curves / '+records.length+' checkpoints.');
