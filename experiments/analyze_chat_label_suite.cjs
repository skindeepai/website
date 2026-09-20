'use strict';
// Score frozen outputs with the pre-existing parsers; never regenerate or fit.
const fs = require('fs'), path = require('path'), crypto = require('crypto'), assert = require('assert/strict');
const root = path.resolve(__dirname, '..');
const dir = path.join(root, 'results/chat-label-suite/replay-100');
const parserPath = path.join(__dirname, 'censor_decision_parsers.cjs');
const parsers = require(parserPath);
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = name => JSON.parse(fs.readFileSync(path.join(dir, name + '.json'), 'utf8'));
const write = (name, value) => fs.writeFileSync(path.join(dir, name + '.json'), JSON.stringify(value, null, 2) + '\n');
assert(!fs.existsSync(path.join(dir, 'parser-result.json')), 'Preserve the completed analysis.');
const protocol = read('protocol'), result = read('result'), records = read('records');
assert.equal(result.protocol_sha256, hash(path.join(dir, 'protocol.json')));
assert.equal(protocol.inputs['experiments/censor_decision_parsers.cjs'], hash(parserPath));
assert.equal(records.length, protocol.planned_calls);
write('parser-protocol', {createdUTC: new Date().toISOString(), analysisSourceSha256: hash(__filename),
    parserSourceSha256: hash(parserPath), sourceRecordsSha256: hash(path.join(dir, 'records.json')),
    sourceResultSha256: hash(path.join(dir, 'result.json')), parsers: Object.keys(parsers),
    policy: 'Unparsed outputs block. Report parse failure separately from correctness.',
    timing: 'Separate repeated-string Node microbenchmark; not added to model timers or claimed as fresh end-to-end latency.',
    limitation: 'One shared set of generated outputs per wording, no parser-specific generation. Reused exploratory data.'});
const parsed = [];
const summary = {};
let numericChecks = 0;
for (const r of records) {
    for (const [name, fn] of Object.entries(parsers)) {
        const decision = fn(r.output, r.allow === 'unchanged' ? 'SAFE' : r.allow);
        if (r.method !== 'generated') {
            assert.equal(decision.parseOk, true);
            assert.equal(Number(decision.blocked), r.prediction);
            numericChecks++;
        } else {
            parsed.push({id: r.id, allow: r.allow, expected: r.expected, output: r.output, parser: name,
                ...decision, prediction: Number(decision.blocked), original_prediction: r.prediction});
        }
    }
}
for (const [name, fn] of Object.entries(parsers)) {
    summary[name] = {};
    for (const allow of ['SAFE', 'OK']) {
        const rows = parsed.filter(r => r.parser === name && r.allow === allow);
        assert.equal(rows.length, 100);
        const baseline = new Map(parsed.filter(r => r.parser === 'exactWithFallback' && r.allow === allow).map(r => [r.id, r]));
        const recovered = rows.filter(r => r.parseOk && r.original_prediction === null);
        const fallback = rows.filter(r => !r.parseOk);
        const data = records.filter(r => r.method === 'generated' && r.allow === allow);
        // Warm and time parser CPU work separately; duplicates do not add accuracy samples.
        let checksum = 0;
        for (let i = 0; i < 100; i++) for (const r of data) checksum += Number(fn(r.output, allow).blocked);
        const times = [];
        for (let pass = 0; pass < 5; pass++) {
            const start = process.hrtime.bigint();
            for (let i = 0; i < 1000; i++) for (const r of data) checksum += Number(fn(r.output, allow).blocked);
            times.push(Number(process.hrtime.bigint() - start) / 1e6 / 100000);
        }
        assert(checksum > 0);
        summary[name][allow] = {n: 100, correct: rows.filter(r => r.prediction === r.expected).length,
            parsed: rows.filter(r => r.parseOk).length,
            parsedCorrect: rows.filter(r => r.parseOk && r.prediction === r.expected).length,
            fallbackBlocked: fallback.length, fallbackCorrect: fallback.filter(r => r.expected === 1).length,
            fallbackFalseBlocks: fallback.filter(r => r.expected === 0).length,
            recoveredInvalid: recovered.length, recoveredCorrect: recovered.filter(r => r.prediction === r.expected).length,
            toxicMissed: rows.filter(r => r.expected === 1 && r.prediction === 0).length,
            safeBlocked: rows.filter(r => r.expected === 0 && r.prediction === 1).length,
            changesFromExactFallback: rows.filter(r => r.prediction !== baseline.get(r.id).prediction).length,
            correctedFromExactFallback: rows.filter(r => r.prediction === r.expected && baseline.get(r.id).prediction !== r.expected).length,
            addedErrorsFromExactFallback: rows.filter(r => r.prediction !== r.expected && baseline.get(r.id).prediction === r.expected).length,
            parserMedianMs: times.sort((a, b) => a - b)[2], modelMeanMs: result.summaries['generated_' + allow].mean_ms};
    }
}
const promptComparisons = {};
for (const method of ['full', 'fixed12', 'learned', 'cascade', 'distill', 'direct', 'generated']) {
    const safe = records.filter(r => r.method === method && r.allow === 'SAFE');
    const ok = new Map(records.filter(r => r.method === method && r.allow === 'OK').map(r => [r.id, r]));
    assert(safe.every(r => r.input_tokens === ok.get(r.id).input_tokens));
    const changed = safe.filter(r => r.prediction !== ok.get(r.id).prediction);
    promptComparisons[method] = {changed: changed.length,
        okCorrected: safe.filter(r => r.prediction !== r.expected && ok.get(r.id).prediction === r.expected).length,
        okAddedErrors: safe.filter(r => r.prediction === r.expected && ok.get(r.id).prediction !== r.expected).length,
        changedDepth: safe.filter(r => r.qwen_depth !== ok.get(r.id).qwen_depth).length,
        pairedInputLengthsEqual: true};
}
write('parser-records', parsed);
write('parser-result', {numericChecks, numericPredictionsChanged: 0, generatedCalls: 200, uniqueMessages: 100,
    summaries: summary, promptComparisons,
    interpretation: 'Numeric predictions are already enums. Prompt-word changes are a separate intervention. Successful parsing and default blocking are reported separately.'});
console.log(JSON.stringify({numericChecks, summary, promptComparisons}, null, 2));
