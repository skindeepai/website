/* Shared browser/Node reference: logistic preference learning on a bounded box. */
(function (root) {
    'use strict';
    const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
    function sigmoid(x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }
    function makeModel(d) { return { d, w: new Float64Array(d), b: 0, data: [], lastMs: 0, acc: 0 }; }
    function logit(m, z) { let s = m.b; for (let i = 0; i < m.d; i++) s += m.w[i] * z[i]; return s; }
    function predict(m, z) { return sigmoid(logit(m, z)); }
    function train(m) {
        const start = performance.now();
        m.w.fill(0); m.b = 0; m.acc = 0;
        const n = m.data.length;
        if (!n) { m.lastMs = performance.now() - start; return; }
        const gradient = new Float64Array(m.d);
        for (let epoch = 0; epoch < 260; epoch++) {
            gradient.fill(0); let bias = 0;
            for (const sample of m.data) {
                const error = predict(m, sample.z) - sample.y;
                for (let i = 0; i < m.d; i++) gradient[i] += error * sample.z[i];
                bias += error;
            }
            for (let i = 0; i < m.d; i++) m.w[i] -= 0.5 * (gradient[i] / n + 0.02 * m.w[i]);
            m.b -= 0.5 * bias / n;
        }
        m.acc = m.data.reduce((n, sample) => n + ((predict(m, sample.z) > 0.5) === (sample.y === 1)), 0) / m.data.length;
        m.lastMs = performance.now() - start;
    }
    function randZ(d, rand) { return Float64Array.from({ length: d }, () => rand() * 2 - 1); }
    function idealZ(m, scale = 1) {
        // Exact maximum on [-scale, scale]^d; zero-weight coordinates stay at zero.
        return Float64Array.from(m.w, w => Math.sign(w) * clamp(scale, 0, 1));
    }
    function transform(m, zRef, targetP) {
        if (!(targetP > 0 && targetP < 1)) throw new RangeError('Target probability must be between zero and one.');
        if (zRef.length !== m.d || Array.from(zRef).some(v => !Number.isFinite(v) || v < -1 || v > 1)) throw new RangeError('Reference must be inside the latent box.');
        const target = Math.log(targetP / (1 - targetP));
        const maximum = m.b + Array.from(m.w).reduce((s, w) => s + Math.abs(w), 0);
        let z = Float64Array.from(zRef), status = 'already-met';
        if (logit(m, z) < target) {
            if (maximum < target) {
                z = Float64Array.from(m.w, (w, i) => w === 0 ? zRef[i] : Math.sign(w));
                status = 'unreachable';
            } else {
                // KKT solution: clamp(zRef + lambda*w). Find its monotone multiplier.
                const scale = Math.max(...Array.from(m.w, Math.abs));
                const at = lambda => Float64Array.from(m.w, (w, i) => clamp(zRef[i] + lambda * (w / scale), -1, 1));
                let lo = 0, hi = 1;
                while (logit(m, at(hi)) < target && hi < Number.MAX_VALUE / 2) hi *= 2;
                for (let i = 0; i < 90; i++) {
                    const mid = (lo + hi) / 2;
                    if (logit(m, at(mid)) >= target) hi = mid; else lo = mid;
                }
                z = at(hi); status = 'reached';
            }
        }
        const moved = Array.from(z, (value, i) => ({ i, dz: value - zRef[i] })).filter(v => Math.abs(v.dz) > 1e-8).sort((a, b) => Math.abs(b.dz) - Math.abs(a.dz));
        return { z, moved, status, achieved: predict(m, z), maximum: sigmoid(maximum), distance: Math.sqrt(Array.from(z).reduce((s, v, i) => s + (v - zRef[i]) ** 2, 0)) };
    }
    const api = { sigmoid, makeModel, logit, predict, train, randZ, idealZ, transform };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.PreferenceCore = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
