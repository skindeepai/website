'use strict';
let tokenizer, encoder, vectors, count, recordIds, loading;
async function sha(bytes) {
    return [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
}
async function load(expectedHash) {
    const response = await fetch('../search/manifest.json');
    if (!response.ok) throw new Error('Search manifest unavailable.');
    const manifest = await response.json();
    if (manifest.answers_sha256 !== expectedHash || manifest.dimensions !== 384 || manifest.model !== 'all-MiniLM-L6-v2') throw new Error('Search vectors are being updated.');
    const vectorResponse = await fetch('../search/vectors.f32');
    if (!vectorResponse.ok) throw new Error('Search vectors unavailable.');
    const bytes = await vectorResponse.arrayBuffer();
    if (bytes.byteLength !== manifest.vector_count * 384 * 4 || await sha(bytes) !== manifest.vectors_sha256) throw new Error('Search vectors do not match.');
    recordIds = manifest.record_ids;
    if (!Array.isArray(recordIds) || recordIds.length !== manifest.vector_count || recordIds.some(i => !Number.isInteger(i) || i < 0 || i >= manifest.count)) throw new Error('Search mapping does not match.');
    vectors = new Float32Array(bytes); count = manifest.count;
    const runtime = await import('../thirdparty/transformersjs/transformers.web.min.js');
    const env = runtime.env;
    env.allowRemoteModels = false; env.allowLocalModels = true;
    env.localModelPath = new URL('../models/', self.location.href).href;
    env.useFSCache = false; env.useWasmCache = false; env.useBrowserCache = false;
    // Same-origin fetches use the browser HTTP cache; no CDN or model-host requests.
    env.useCustomCache = true;
    env.customCache = { match: async request => {
        const url = typeof request === 'string' ? request : request.url;
        if (!url.startsWith(env.localModelPath + 'all-MiniLM-L6-v2/')) return undefined;
        const response = await fetch(url);
        return response.ok ? response : undefined;
    }, put: async () => {} };
    const base = new URL('../thirdparty/transformersjs/ort/', self.location.href).href;
    env.backends.onnx.wasm.wasmPaths = { wasm: base + 'ort-wasm-simd-threaded.asyncify.wasm', mjs: base + 'ort-wasm-simd-threaded.asyncify.mjs' };
    env.backends.onnx.wasm.numThreads = 1; env.backends.onnx.wasm.proxy = false;
    tokenizer = await runtime.AutoTokenizer.from_pretrained('all-MiniLM-L6-v2');
    encoder = await runtime.AutoModel.from_pretrained('all-MiniLM-L6-v2', { device: 'wasm', dtype: 'q8' });
}
async function search(query) {
    const inputs = tokenizer(query.slice(0, 400), { truncation: true, max_length: 256, padding: true });
    const result = await encoder(inputs), hidden = result.last_hidden_state;
    const vector = new Float32Array(384);
    let tokens = 0, norm = 0;
    for (let i = 0; i < hidden.dims[1]; i++) {
        if (!Number(inputs.attention_mask.data[i])) continue;
        tokens++;
        for (let j = 0; j < 384; j++) vector[j] += hidden.data[i * 384 + j];
    }
    for (let j = 0; j < 384; j++) { vector[j] /= Math.max(1, tokens); norm += vector[j] * vector[j]; }
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < 384; j++) vector[j] /= norm;
    const scores = new Float32Array(count);
    for (let i = 0; i < recordIds.length; i++) {
        let score = 0;
        for (let j = 0; j < 384; j++) score += vector[j] * vectors[i * 384 + j];
        scores[recordIds[i]] = Math.max(scores[recordIds[i]], score);
    }
    return scores;
}
// Coalesce pending queries; old searches cannot build an unbounded queue.
let pending, running = false;
async function drain() {
    if (running) return;
    running = true;
    while (pending) {
        const request = pending; pending = null;
        try {
            if (!loading) loading = load(request.hash);
            await loading;
            const scores = await search(request.query);
            postMessage({ id: request.id, scores }, [scores.buffer]);
        } catch (error) { postMessage({ id: request.id, error: error.message }); }
    }
    running = false;
}
self.onmessage = event => { pending = event.data; drain(); };
