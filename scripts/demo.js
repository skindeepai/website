// SkinDeep.ai — in-browser latent preference learning demo.
// Everything runs client-side: a logistic classifier over a toy latent space,
// retrained on every rating, then run in reverse. No uploads, no server.

(function () {
    'use strict';

    // ---------------------------------------------------------------
    // Deterministic PRNG (fixed layout tables so nearby latents render
    // nearby outputs — the decoder must be continuous, like a GAN's).
    // ---------------------------------------------------------------
    function mulberry32(seed) {
        const next = function () {
            seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
            let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        };
        next.state = () => seed >>> 0;
        return next;
    }

    function lerp(a, b, t) { return a + (b - a) * t; }
    function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
    function u(z) { return (z + 1) / 2; } // [-1,1] -> [0,1]

    function lerpHex(h1, h2, t) {
        const p = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
        const a = p(h1), b = p(h2);
        const c = a.map((v, i) => Math.round(lerp(v, b[i], t)));
        return '#' + c.map((v) => v.toString(16).padStart(2, '0')).join('');
    }
    function lerp3(h1, h2, h3, t) {
        return t < 0.5 ? lerpHex(h1, h2, t * 2) : lerpHex(h2, h3, (t - 0.5) * 2);
    }

    // ---------------------------------------------------------------
    // Domain A: parametric cartoon faces (16-D latent)
    // ---------------------------------------------------------------
    const FACE_DIMS = [
        { label: 'Wider face' }, { label: 'Longer face' }, { label: 'Deeper skin tone' },
        { label: 'Longer hair' }, { label: 'Darker hair' }, { label: 'Curlier hair' },
        { label: 'Bigger eyes' }, { label: 'Wider-set eyes' }, { label: 'Darker eyes' },
        { label: 'Stronger brows' }, { label: 'Larger nose' }, { label: 'Wider mouth' },
        { label: 'Bigger smile' }, { label: 'Glasses' }, { label: 'Freckles' }, { label: 'Squarer jaw' }
    ];

    const FRECKLE_POS = (() => {
        const r = mulberry32(77), pts = [];
        for (let i = 0; i < 14; i++) {
            const side = i % 2 === 0 ? -1 : 1;
            pts.push({ x: side * (14 + r() * 20), y: 16 + r() * 14 });
        }
        return pts;
    })();

    function renderFace(z) {
        const cx = 100;
        const rx = 44 + 9 * z[0];
        const ry = 58 + 9 * z[1];
        const cy = 104;
        const skin = lerp3('#F7D9BF', '#C98D5F', '#7C4F30', u(z[2]));
        const skinEdge = lerp3('#E3B994', '#A96F45', '#5E3A22', u(z[2]));
        const hairLen = u(z[3]);                    // 0 buzz .. 1 long
        const hairCol = lerp3('#E7C87E', '#8A5A2B', '#241A12', u(z[4]));
        const curl = Math.max(0, z[5]);
        const eyeR = 6.4 + 2.4 * z[6];
        const eyeDX = 25 + 6 * z[7];
        const iris = lerp3('#7FB6D9', '#7E9455', '#4E3218', u(z[8]));
        const browW = 2.2 + 1.6 * u(z[9]);
        const browTilt = 8 * z[9];
        const noseS = 1 + 0.42 * z[10];
        const mouthW = 17 + 7 * z[11];
        const smile = z[12];
        const glassesOp = clamp((z[13] + 0.15) / 1.15, 0, 1);
        const freckleN = Math.round(Math.max(0, z[14]) * 14);
        const jaw = u(z[15]);                       // 0 pointed .. 1 square

        const eyeY = cy - 4;
        const chinY = cy + ry;
        const cheekW = lerp(0.42, 0.86, jaw);       // how wide the chin stays

        // Head: top = ellipse arc, bottom = two cubics whose side controls
        // widen with the jaw parameter.
        const head =
            `M ${cx - rx} ${cy} A ${rx} ${ry * 0.92} 0 0 1 ${cx + rx} ${cy} ` +
            `C ${cx + rx} ${cy + ry * 0.6} ${cx + rx * cheekW} ${chinY} ${cx} ${chinY} ` +
            `C ${cx - rx * cheekW} ${chinY} ${cx - rx} ${cy + ry * 0.6} ${cx - rx} ${cy} Z`;

        // Back hair: rounded sheet behind the head, bottom edge scalloped by
        // curl. Capped above the card edge so the scallops always resolve.
        let backHair = '';
        if (hairLen > 0.12) {
            const hw = rx + 10;
            const bot = Math.min(cy + ry * 0.25 + hairLen * (ry * 0.75 + 58), 206);
            const amp = 4 + curl * 9, waves = 5;
            let bottomEdge = '';
            for (let i = 0; i <= waves; i++) {
                const x1 = cx + hw - (2 * hw * i) / waves;
                const midX = x1 + hw / waves;
                if (i > 0) bottomEdge += `Q ${midX} ${bot + amp} ${x1} ${bot} `;
            }
            backHair = `<path d="M ${cx - hw} ${cy + 6} A ${hw} ${ry} 0 0 1 ${cx + hw} ${cy + 6} L ${cx + hw} ${bot} ${bottomEdge}Z" fill="${hairCol}"/>`;
        }

        // Neck and shoulders ground the head; long hair falls behind them.
        const shirt = '#6D82A8', shirtEdge = '#5A6E93';
        const neckTop = chinY - 16;
        const neck = `<path d="M ${cx - 9.5} ${neckTop} L ${cx - 9.5} ${chinY + 8} Q ${cx} ${chinY + 13} ${cx + 9.5} ${chinY + 8} L ${cx + 9.5} ${neckTop} Z" fill="${skin}" stroke="${skinEdge}" stroke-width="1"/>`;
        const shoulders = `<path d="M ${cx - 64} 228 L ${cx - 62} 222 Q ${cx - 56} ${chinY + 15} ${cx - 15} ${chinY + 7} L ${cx + 15} ${chinY + 7} Q ${cx + 56} ${chinY + 15} ${cx + 62} 222 L ${cx + 64} 228 Z" fill="${shirt}" stroke="${shirtEdge}" stroke-width="1"/>`;

        // Front hair: cap over the skull with a fringe line; recedes when short.
        const fringeY = cy - ry * lerp(0.72, 0.46, Math.min(1, hairLen * 1.6));
        const capTop = cy - ry * 0.98;
        const wig = 3 + curl * 6;
        const frontHair = hairLen < 0.04 ? '' :
            `<path d="M ${cx - rx - 2} ${cy - 2} A ${rx + 2} ${ry} 0 0 1 ${cx + rx + 2} ${cy - 2} ` +
            `L ${cx + rx - 4} ${fringeY + wig} Q ${cx + rx * 0.5} ${fringeY - wig} ${cx} ${fringeY + wig * 0.6} ` +
            `Q ${cx - rx * 0.5} ${fringeY + wig * 1.6} ${cx - rx + 4} ${fringeY - wig * 0.4} Z" fill="${hairCol}"/>`;
        const buzz = hairLen < 0.04 ?
            `<path d="M ${cx - rx * 0.92} ${cy - ry * 0.35} A ${rx * 0.94} ${ry * 0.9} 0 0 1 ${cx + rx * 0.92} ${cy - ry * 0.35}" fill="none" stroke="${hairCol}" stroke-width="5" opacity="0.55"/>` : '';

        // Eyes / brows / glasses / nose / mouth
        const eye = (sx) => {
            const ex = cx + sx * eyeDX;
            return `<ellipse cx="${ex}" cy="${eyeY}" rx="${eyeR * 1.32}" ry="${eyeR * 1.02}" fill="#fff" stroke="${skinEdge}" stroke-width="0.8"/>` +
                `<circle cx="${ex}" cy="${eyeY}" r="${eyeR * 0.66}" fill="${iris}"/>` +
                `<circle cx="${ex}" cy="${eyeY}" r="${eyeR * 0.3}" fill="#1A1A1A"/>` +
                `<circle cx="${ex - eyeR * 0.22}" cy="${eyeY - eyeR * 0.24}" r="${eyeR * 0.13}" fill="#fff"/>`;
        };
        const brow = (sx) => {
            const ex = cx + sx * eyeDX, by = eyeY - eyeR * 1.1 - 7;
            const x1 = ex - 11, x2 = ex + 11;
            const y1 = by + sx * browTilt * -0.5, y2 = by + sx * browTilt * 0.5;
            return `<path d="M ${x1} ${sx < 0 ? y2 : y1} Q ${ex} ${by - 3} ${x2} ${sx < 0 ? y1 : y2}" stroke="#3B2A1A" stroke-width="${browW}" fill="none" stroke-linecap="round"/>`;
        };
        const glasses = glassesOp < 0.03 ? '' :
            `<g stroke="#2F3542" fill="none" stroke-width="${2 + glassesOp}" opacity="${glassesOp}">` +
            `<circle cx="${cx - eyeDX}" cy="${eyeY}" r="${eyeR * 1.32 + 4.5}"/>` +
            `<circle cx="${cx + eyeDX}" cy="${eyeY}" r="${eyeR * 1.32 + 4.5}"/>` +
            `<path d="M ${cx - eyeDX + eyeR * 1.32 + 4.5} ${eyeY - 2} Q ${cx} ${eyeY - 7} ${cx + eyeDX - eyeR * 1.32 - 4.5} ${eyeY - 2}"/>` +
            `<path d="M ${cx - eyeDX - eyeR * 1.32 - 4.5} ${eyeY} L ${cx - rx} ${eyeY - 3}"/>` +
            `<path d="M ${cx + eyeDX + eyeR * 1.32 + 4.5} ${eyeY} L ${cx + rx} ${eyeY - 3}"/></g>`;
        const noseY = eyeY + 14;
        const nose = `<path d="M ${cx - 1.5 * noseS} ${noseY} Q ${cx + 4.5 * noseS} ${noseY + 8 * noseS} ${cx - 1 * noseS} ${noseY + 13 * noseS}" stroke="${skinEdge}" stroke-width="2.4" fill="none" stroke-linecap="round"/>`;
        const mouthY = cy + ry * 0.58;
        const lipCol = '#B0524F';
        const openSmile = smile > 0.55;
        const mouth = openSmile ?
            `<path d="M ${cx - mouthW} ${mouthY} Q ${cx} ${mouthY + 16 * smile} ${cx + mouthW} ${mouthY} Q ${cx} ${mouthY + 4} ${cx - mouthW} ${mouthY} Z" fill="${lipCol}"/>` +
            `<path d="M ${cx - mouthW * 0.66} ${mouthY + 1.5} Q ${cx} ${mouthY + 5.5} ${cx + mouthW * 0.66} ${mouthY + 1.5} L ${cx + mouthW * 0.6} ${mouthY + 1} Q ${cx} ${mouthY + 4} ${cx - mouthW * 0.6} ${mouthY + 1} Z" fill="#fff"/>` :
            `<path d="M ${cx - mouthW} ${mouthY} Q ${cx} ${mouthY + 13 * smile} ${cx + mouthW} ${mouthY}" stroke="${lipCol}" stroke-width="3.4" fill="none" stroke-linecap="round"/>`;
        const freckles = FRECKLE_POS.slice(0, freckleN).map((p) =>
            `<circle cx="${cx + p.x}" cy="${eyeY + p.y}" r="1.5" fill="rgba(122,74,40,0.45)"/>`).join('');
        const blush = `<ellipse cx="${cx - eyeDX - 4}" cy="${eyeY + 17}" rx="6.5" ry="3.6" fill="rgba(226,120,120,0.18)"/>` +
            `<ellipse cx="${cx + eyeDX + 4}" cy="${eyeY + 17}" rx="6.5" ry="3.6" fill="rgba(226,120,120,0.18)"/>`;
        const ears = `<circle cx="${cx - rx - 2}" cy="${cy + 2}" r="7.5" fill="${skin}" stroke="${skinEdge}" stroke-width="1"/>` +
            `<circle cx="${cx + rx + 2}" cy="${cy + 2}" r="7.5" fill="${skin}" stroke="${skinEdge}" stroke-width="1"/>`;

        return `<svg viewBox="0 0 200 224" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Generated cartoon face">` +
            `<rect width="200" height="224" rx="10" fill="#EEF2F7"/>` +
            backHair + shoulders + neck + ears +
            `<path d="${head}" fill="${skin}" stroke="${skinEdge}" stroke-width="1.4"/>` +
            blush + freckles + eye(-1) + eye(1) + brow(-1) + brow(1) + nose + mouth +
            frontHair + buzz + glasses +
            `</svg>`;
    }

    // ---------------------------------------------------------------
    // Domain B: parametric abstract compositions (12-D latent)
    // ---------------------------------------------------------------
    const ART_DIMS = [
        { label: 'Warmer palette' }, { label: 'More color variety' }, { label: 'More saturated' },
        { label: 'Lighter tones' }, { label: 'More elements' }, { label: 'Varied sizes' },
        { label: 'Angular shapes' }, { label: 'Symmetry' }, { label: 'Darker background' },
        { label: 'Outlined style' }, { label: 'Grid arrangement' }, { label: 'Looser placement' }
    ];

    const ART_TABLE = (() => {
        const r = mulberry32(4242), els = [];
        const grid = [];
        for (let gy = 0; gy < 5; gy++) for (let gx = 0; gx < 5; gx++) grid.push({ x: 34 + gx * 33, y: 34 + gy * 33 });
        for (let i = 0; i < 26; i++) {
            els.push({
                sx: 20 + r() * 160, sy: 20 + r() * 160,
                gx: grid[i % 25].x, gy: grid[i % 25].y,
                rot: r() * 360, size: 9 + r() * 20,
                shapeR: r(), hueR: r(), jx: (r() - 0.5) * 2, jy: (r() - 0.5) * 2, op: 0.75 + r() * 0.25
            });
        }
        return els;
    })();

    function renderArt(z) {
        const hue = 215 - u(z[0]) * 195;             // cool 215 -> warm 20
        const spread = 14 + u(z[1]) * 110;
        const sat = 25 + u(z[2]) * 62;
        const lig = 34 + u(z[3]) * 30;
        const count = Math.round(5 + u(z[4]) * 21);
        const sizeVar = u(z[5]);
        const angular = u(z[6]);
        const sym = u(z[7]);
        const bgL = 94 - u(z[8]) * 80;
        const outlined = u(z[9]);
        const gridT = u(z[10]);
        const jit = u(z[11]) * 16;

        const bg = `hsl(${hue}, ${Math.round(sat * 0.25)}%, ${bgL}%)`;
        let shapes = '';
        for (let i = 0; i < count; i++) {
            const e = ART_TABLE[i];
            const x = lerp(e.sx, e.gx, gridT) + e.jx * jit;
            const y = lerp(e.sy, e.gy, gridT) + e.jy * jit;
            const s = e.size * (1 + sizeVar * (e.hueR - 0.5) * 1.7) * lerp(1.15, 0.85, gridT);
            const h = (hue + (e.hueR - 0.5) * 2 * spread + 360) % 360;
            const col = `hsl(${h}, ${sat}%, ${lig}%)`;
            const paint = outlined > 0.55
                ? `fill="none" stroke="${col}" stroke-width="${2.2 + outlined * 1.6}"`
                : `fill="${col}" stroke="none"`;
            let el;
            if (e.shapeR > angular) {
                el = `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(s * 0.62).toFixed(1)}" ${paint} opacity="${e.op}"/>`;
            } else if (e.shapeR > angular * 0.45) {
                el = `<rect x="${(-s * 0.55).toFixed(1)}" y="${(-s * 0.55).toFixed(1)}" width="${(s * 1.1).toFixed(1)}" height="${(s * 1.1).toFixed(1)}" ${paint} opacity="${e.op}" transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${(e.rot * (0.2 + angular)).toFixed(0)})"/>`;
            } else {
                const p = s * 0.72;
                el = `<path d="M 0 ${-p} L ${p * 0.87} ${p * 0.5} L ${-p * 0.87} ${p * 0.5} Z" ${paint} opacity="${e.op}" transform="translate(${x.toFixed(1)} ${y.toFixed(1)}) rotate(${(e.rot * (0.2 + angular)).toFixed(0)})"/>`;
            }
            shapes += el;
            if (sym > 0.06) {
                shapes += `<g opacity="${(sym * 0.9).toFixed(2)}" transform="translate(200 0) scale(-1 1)">${el}</g>`;
            }
        }
        return `<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Generated abstract composition">` +
            `<rect width="200" height="200" rx="10" fill="${bg}"/>${shapes}</svg>`;
    }

    // ---------------------------------------------------------------
    // The model: logistic regression on the latent vector.
    // Exactly the "single dense layer + sigmoid" from the 2019 patent.
    // ---------------------------------------------------------------
    const { sigmoid, makeModel, logit, predict, train, randZ, idealZ, transform } = window.PreferenceCore;

    // Active learning: sample a pool, keep a deliberate mix of
    // predicted-likes, uncertain cases, and pure exploration.
    function nextBatch(m, active, rand) {
        const B = 12;
        if (!active || m.data.length < 6) {
            const out = [];
            for (let i = 0; i < B; i++) out.push(randZ(m.d, rand));
            return out;
        }
        const pool = [];
        for (let i = 0; i < 380; i++) {
            const z = randZ(m.d, rand);
            pool.push({ z, p: predict(m, z) });
        }
        const byScore = pool.slice().sort((a, b) => b.p - a.p);
        const byUncert = pool.slice().sort((a, b) => Math.abs(a.p - 0.5) - Math.abs(b.p - 0.5));
        const picked = [], used = new Set();
        const take = (arr, k) => {
            for (let i = 0; i < arr.length && k > 0; i++) {
                if (!used.has(arr[i])) { used.add(arr[i]); picked.push(arr[i].z); k--; }
            }
        };
        take(byScore.slice(0, 46), 5);              // likely likes — keeps it fun
        take(byUncert, 4);                          // most informative
        for (let i = pool.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [pool[i], pool[j]] = [pool[j], pool[i]];
        }
        take(pool, 3);     // exploration
        for (let i = picked.length - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [picked[i], picked[j]] = [picked[j], picked[i]];
        }
        return picked;
    }

    function nearIdeals(m, t, rand, count) {
        const zStar = idealZ(m, t);
        const base = logit(m, zStar);
        const out = [];
        let guard = 0;
        while (out.length < count && guard++ < 4000) {
            const z = new Float64Array(m.d);
            for (let i = 0; i < m.d; i++) z[i] = clamp(zStar[i] + (rand() * 2 - 1) * 0.62, -1, 1);
            if (logit(m, z) > base - 2.0) out.push(z);
        }
        return out;
    }


    // ---------------------------------------------------------------
    // UI wiring
    // ---------------------------------------------------------------
    const DOMAINS = {
        faces: { dims: FACE_DIMS, render: renderFace, name: 'faces' },
        art: { dims: ART_DIMS, render: renderArt, name: 'compositions' }
    };

    const el = (id) => document.getElementById(id);
    const POS_COLOR = '#4F46E5', NEG_COLOR = '#a45409';
    const REVEAL_EVERY = 20;           // the 2019 app's loop: an ideal every 20th rating
    const RING_C = 100.53;             // 2 * pi * r16

    const state = {};
    for (const key of Object.keys(DOMAINS)) {
        state[key] = {
            model: makeModel(DOMAINS[key].dims.length),
            seed: 17, rand: mulberry32(17 + (key === 'art' ? 991 : 0)),
            queue: [], current: null, seen: 0, history: [], active: true, msLog: [], prevReveal: null, evaluation: []
        };
    }
    let domainKey = 'faces';

    const S = () => state[domainKey];
    const D = () => DOMAINS[domainKey];

    function ensureCard() {
        const s = S();
        if (!s.current) {
            if (el('evaluation-toggle').checked) s.current = randZ(s.model.d, s.rand);
            else {
                if (s.queue.length === 0) s.queue = nextBatch(s.model, s.active, s.rand);
                s.current = s.queue.shift();
            }
        }
    }

    function likes(s) { return s.model.data.filter((d) => d.y === 1).length; }
    function passes(s) { return s.model.data.filter((d) => d.y === 0).length; }
    function unlocked(s) { return likes(s) >= 3 && passes(s) >= 3 && s.model.data.length >= 10; }

    function fmtMs(ms) { return ms < 1 ? ms.toFixed(2) : ms.toFixed(1); }

    function fmtPct(p, dp) {
        const hi = 100 - Math.pow(10, -(dp + 1)) * 5; // e.g. 99.95 for dp=1
        if (p * 100 >= hi) return (dp > 1 ? '99.99' : '99.9') + '+%';
        return (p * 100).toFixed(dp) + '%';
    }

    function renderCurrent() {
        ensureCard();
        const s = S();
        el('card-art').innerHTML = D().render(s.current);
        el('card-num').textContent = '#' + (s.seen + 1);
        const chip = el('card-pred');
        if (s.model.data.length >= 8 && !el('evaluation-toggle').checked) {
            chip.style.display = 'inline-flex';
            chip.textContent = 'model guess ' + Math.round(predict(s.model, s.current) * 100) + '%';
        } else {
            chip.style.display = 'none';
        }
        el('undo-btn').disabled = s.history.length === 0;

        // Progress ring toward the next reveal, like the 2019 app's.
        const n = s.model.data.length;
        const p = (n % REVEAL_EVERY) / REVEAL_EVERY;
        el('ring-fg').style.strokeDashoffset = (RING_C * (1 - p)).toFixed(1);
        el('ring-label').textContent = Math.round(p * 100) + '%';
        el('rating-progress').textContent = n + ' ratings. ' + (unlocked(s) ? 'Your suggestions are ready below.' : 'Rate a mix of likes and passes to reveal suggestions.');
    }

    function revealOpen() { return el('reveal-overlay').classList.contains('open'); }
    let revealReturnFocus = null;
    let revealBackground = [];

    function showReveal() {
        const s = S();
        const n = s.model.data.length;
        const z = idealZ(s.model, 0.75);
        const cur = D().render(z);
        el('reveal-art').innerHTML = cur;
        el('reveal-title').textContent = 'Rating #' + n + ' — your predicted favorite, generated';
        el('reveal-score').textContent = fmtPct(predict(s.model, z), 2);
        const cmp = el('reveal-compare');
        if (s.prevReveal && s.prevReveal.n !== n) {
            cmp.style.display = 'flex';
            cmp.innerHTML =
                '<figure><div>' + s.prevReveal.svg + '</div><figcaption>at #' + s.prevReveal.n + '</figcaption></figure>' +
                '<span class="rc-arrow">→</span>' +
                '<figure><div>' + cur + '</div><figcaption>now</figcaption></figure>';
        } else {
            cmp.style.display = 'none';
            cmp.innerHTML = '';
        }
        s.prevReveal = { svg: cur, n };
        el('reveal-note').textContent =
            (likes(s) === 0 || passes(s) === 0)
                ? 'You have only rated one way so far — mixing 👍 and 👎 gives a much sharper ideal.'
                : 'Reverse classification, solved from your ' + n + ' ratings — not picked from a pool. Check blind evaluation to see whether it improves.';
        revealReturnFocus = document.activeElement;
        revealBackground = Array.from(document.querySelectorAll('.lab-header, .lab-sidebar, .skip-link, .lab-main > :not(#reveal-overlay)'))
            .map(node => ({ node, inert: node.inert }));
        revealBackground.forEach(({ node }) => { node.inert = true; });
        el('reveal-overlay').classList.add('open');
        el('reveal-keep').focus();
    }

    function closeReveal(goBreakdown) {
        el('reveal-overlay').classList.remove('open');
        revealBackground.forEach(({ node, inert }) => { node.inert = inert; });
        revealBackground = [];
        (revealReturnFocus || el('like-btn')).focus({ preventScroll: true });
        if (goBreakdown) {
            const sec = document.getElementById('generate');
            if (sec) {
                sec.open = true;
                sec.querySelector('summary').focus({ preventScroll: true });
                sec.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
            }
        }
    }

    function renderStats() {
        const s = S();
        const evaluation = s.evaluation;
        el('evaluation-summary').textContent = evaluation.length
            ? 'Blind evaluation: ' + evaluation.length + ' ratings; ' + Math.round(100 * evaluation.filter(row => (row.p > 0.5) === (row.y === 1)).length / evaluation.length) + '% accuracy. Small sessions are exploratory.'
            : 'No blind evaluation ratings for the current model.';
        el('stat-n').textContent = s.model.data.length;
        el('stat-likes').textContent = likes(s) + '👍 ' + passes(s) + '👎';
        if (s.msLog.length) {
            const sorted = s.msLog.slice(-9).sort((a, b) => a - b);
            el('stat-ms').textContent = fmtMs(sorted[Math.floor(sorted.length / 2)]) + ' ms';
        } else {
            el('stat-ms').textContent = '—';
        }
        el('stat-acc').textContent = s.model.data.length >= 4 ? Math.round(s.model.acc * 100) + '%' : '—';
    }

    function renderWeights() {
        const s = S(), dims = D().dims;
        const wmax = Math.max(...Array.from(s.model.w, Math.abs), 0.15);
        let html = '';
        for (let i = 0; i < dims.length; i++) {
            const w = s.model.w[i];
            const frac = Math.abs(w) / wmax;
            const pct = (frac * 50).toFixed(1);
            const side = w >= 0 ? 'pos' : 'neg';
            const col = w >= 0 ? POS_COLOR : NEG_COLOR;
            html +=
                `<div class="wrow" title="weight ${w.toFixed(3)}">` +
                `<span class="wlabel">${dims[i].label}</span>` +
                `<span class="wtrack" aria-hidden="true"><span class="wzero"></span>` +
                `<span class="wbar ${side}" style="width:${pct}%;background:${col};"></span></span>` +
                `<span class="wvalue">${w > 0 ? '+' : ''}${w.toFixed(2)}</span>` +
                `</div>`;
        }
        el('weights').innerHTML = html;
        el('weights-note').textContent =
            'The model stores ' + (dims.length + 1) + ' learned numbers. The bars show which drawing settings affect its predictions.';
    }

    function renderSections() {
        const s = S();
        const ok = unlocked(s);
        document.querySelectorAll('.locked-section').forEach((sec) => {
            sec.hidden = !ok;
            if (!ok) sec.open = false;
            sec.classList.toggle('is-locked', !ok);
            sec.querySelector('.op-body').hidden = !ok;
        });
        if (ok) {
            renderIdeal();
            renderScoring(false);
            renderTransform();
        }
    }

    function renderIdeal() {
        const s = S();
        const t = parseFloat(el('realism').value);
        const zStar = idealZ(s.model, t);
        el('ideal-art').innerHTML = D().render(zStar);
        el('ideal-score').textContent = fmtPct(predict(s.model, zStar), 2);
        const near = nearIdeals(s.model, t, mulberry32(s.seed + s.model.data.length), 6);
        el('near-grid').innerHTML = near.map((z) =>
            `<figure class="mini"><div class="mini-art">${D().render(z)}</div>` +
            `<figcaption>${fmtPct(predict(s.model, z), 1)}</figcaption></figure>`).join('');
    }

    let scoreSamples = null;
    function renderScoring(resample) {
        const s = S();
        if (!scoreSamples || resample || scoreSamples.d !== s.model.d) {
            scoreSamples = { d: s.model.d, zs: [] };
            for (let i = 0; i < 8; i++) scoreSamples.zs.push(randZ(s.model.d, mulberry32(s.seed + i * 31 + (resample ? s.seen : 0))));
        }
        el('score-grid').innerHTML = scoreSamples.zs.map((z) => {
            const p = Math.round(predict(s.model, z) * 100);
            return `<figure class="mini"><div class="mini-art">${D().render(z)}</div>` +
                `<figcaption><span class="scorebar"><span style="width:${p}%"></span></span>${p}%</figcaption></figure>`;
        }).join('');
    }

    function renderTransform() {
        const s = S();
        const target = parseFloat(el('target-score').value);
        el('target-label').textContent = Math.round(target * 100) + '%';
        const zRef = new Float64Array(s.model.d); // the "average" sample
        const before = predict(s.model, zRef);
        const res = transform(s.model, zRef, target);
        el('tf-before').innerHTML = D().render(zRef);
        el('tf-after').innerHTML = D().render(res.z);
        el('tf-before-score').textContent = Math.round(before * 100) + '%';
        el('tf-after-score').textContent = Math.round(predict(s.model, res.z) * 100) + '%';
        el('transform-status').textContent = res.status === 'unreachable' ? 'Target is outside the attainable range. Showing its maximum: ' + fmtPct(res.maximum, 1) + '.' : res.status === 'already-met' ? 'The original already meets the target; no change needed.' : 'Target reached within the latent bounds. L2 distance: ' + res.distance.toFixed(3) + '.';
        el('tf-changes').innerHTML = res.moved.length === 0 ? '<li>No coordinates changed.</li>' :
            res.moved.slice(0, 6).map((mv) =>
                `<li>${mv.dz > 0 ? '<span class="up">▲</span>' : '<span class="down">▼</span>'} ${D().dims[mv.i].label}</li>`).join('');
    }

    function refreshAll() {
        renderCurrent(); renderStats(); renderWeights(); renderSections();
    }

    function rate(y) {
        const s = S();
        ensureCard();
        if (el('evaluation-toggle').checked) {
            const row = { z: Array.from(s.current), y, p: predict(s.model, s.current) };
            s.evaluation.push(row); s.history.push({ ...row, kind: 'evaluation' });
            s.seen++; s.current = null; refreshAll(); return;
        }
        s.model.data.push({ z: s.current, y });
        s.history.push({ z: s.current, y, kind: 'training' });
        s.evaluation = []; s.history = s.history.filter(row => row.kind !== 'evaluation');
        train(s.model);
        s.msLog.push(s.model.lastMs);
        s.seen++;
        s.current = null;
        // Each new batch is drawn against the freshly retrained model.
        if (s.queue.length === 0 || s.model.data.length % 6 === 0) {
            s.queue = nextBatch(s.model, s.active, s.rand);
        }
        flashCard(y);
        refreshAll();
        if (s.model.data.length > 0 && s.model.data.length % REVEAL_EVERY === 0) {
            showReveal();
        }
    }

    function skip() {
        const s = S();
        s.current = null; s.seen++;
        refreshAll();
    }

    function undo() {
        const s = S();
        const last = s.history.pop();
        if (!last) return;
        // Ratings and history are pushed in lockstep, so the last data
        // entry is always the rating being undone.
        if (last.kind === 'evaluation') s.evaluation.pop();
        else { s.model.data.pop(); train(s.model); s.evaluation = []; s.history = s.history.filter(row => row.kind !== 'evaluation'); s.msLog.pop(); }
        s.queue = [];
        s.prevReveal = null;
        s.current = last.z; s.seen = Math.max(0, s.seen - 1);
        refreshAll();
    }

    function flashCard(y) {
        const card = el('card');
        card.classList.remove('flash-like', 'flash-pass');
        void card.offsetWidth;
        card.classList.add(y === 1 ? 'flash-like' : 'flash-pass');
    }

    function setDomain(key) {
        domainKey = key;
        document.querySelectorAll('[data-domain]').forEach((b) =>
            (b.classList.toggle('active', b.dataset.domain === key), b.setAttribute('aria-pressed', String(b.dataset.domain === key))));
        el('active-toggle').checked = S().active;
        el('session-seed').value = S().seed;
        scoreSamples = null;
        refreshAll();
    }

    function resetDomain() {
        const key = domainKey;
        const seed = Number(el('session-seed').value);
        if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295 || el('session-seed').value === '') {
            el('session-status').textContent = 'Choose an integer seed from 0 to 4294967295.'; return;
        }
        state[key] = {
            model: makeModel(DOMAINS[key].dims.length),
            seed, rand: mulberry32(seed + (key === 'art' ? 991 : 0)),
            queue: [], current: null, seen: 0, history: [], active: el('active-toggle').checked, msLog: [], prevReveal: null, evaluation: []
        };
        scoreSamples = null;
        el('session-status').textContent = 'Current domain reset from seed ' + seed + '.';
        refreshAll();
    }

    function exportSession() {
        const s = S();
        const payload = { version: 1, domain: domainKey, seed: s.seed, randomState: s.rand.state(), active: s.active, evaluationMode: el('evaluation-toggle').checked,
            ratings: s.model.data.map(row => ({ z: Array.from(row.z), y: row.y })), evaluation: s.evaluation,
            current: Array.from(s.current), queue: s.queue.map(z => Array.from(z)), seen: s.seen };
        const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
        const link = document.createElement('a'); link.href = url; link.download = 'skindeep-session.json'; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    async function importSession(file) {
        if (!file) return;
        try {
            if (file.size > 2000000) throw new Error('Session is too large.');
            const data = JSON.parse(await file.text());
            if (data.version !== 1 || !Object.prototype.hasOwnProperty.call(DOMAINS, data.domain)) throw new Error('Unsupported session.');
            const d = DOMAINS[data.domain].dims.length;
            const validZ = z => Array.isArray(z) && z.length === d && z.every(v => Number.isFinite(v) && Math.abs(v) <= 1);
            if (!Array.isArray(data.ratings) || data.ratings.length > 5000 || !data.ratings.every(row => validZ(row.z) && (row.y === 0 || row.y === 1)) ||
                !validZ(data.current) || !Array.isArray(data.queue) || data.queue.length > 12 || !data.queue.every(validZ) ||
                !Number.isInteger(data.seed) || data.seed < 0 || data.seed > 4294967295 || !Number.isInteger(data.randomState) || data.randomState < 0 || data.randomState > 4294967295 ||
                !Number.isInteger(data.seen) || data.seen < 0 || typeof data.active !== 'boolean') throw new Error('Invalid session data.');
            const m = makeModel(d); m.data = data.ratings; train(m);
            state[data.domain] = { model: m, seed: data.seed, rand: mulberry32(data.randomState), active: data.active,
                queue: data.queue, current: data.current, seen: data.seen, msLog: [], prevReveal: null, evaluation: [],
                history: data.ratings.map(row => ({ ...row, kind: 'training' })) };
            // Evaluation results are recomputed locally rather than trusted from an imported file.
            if (Array.isArray(data.evaluation) && data.evaluation.length <= 5000 && data.evaluation.every(row => validZ(row.z) && (row.y === 0 || row.y === 1))) {
                state[data.domain].evaluation = data.evaluation.map(row => ({ z: row.z, y: row.y, p: predict(m, row.z) }));
                state[data.domain].history.push(...state[data.domain].evaluation.map(row => ({ ...row, kind: 'evaluation' })));
            }
            el('evaluation-toggle').checked = data.evaluationMode === true;
            setDomain(data.domain); el('session-status').textContent = 'Session imported locally. Model recomputed from ratings.';
        } catch (error) { el('session-status').textContent = 'Could not import: ' + error.message; }
    }

    // ---- events ----
    document.addEventListener('DOMContentLoaded', () => {
        el('export-session').addEventListener('click', exportSession);
        el('import-session').addEventListener('change', event => importSession(event.target.files[0]));
        el('evaluation-toggle').addEventListener('change', () => { S().current = null; S().queue = []; refreshAll(); });
        el('like-btn').addEventListener('click', () => rate(1));
        el('pass-btn').addEventListener('click', () => rate(0));
        el('skip-btn').addEventListener('click', skip);
        el('undo-btn').addEventListener('click', undo);
        el('reset-btn').addEventListener('click', resetDomain);
        el('resample-btn').addEventListener('click', () => renderScoring(true));
        el('realism').addEventListener('input', () => { if (unlocked(S())) renderIdeal(); });
        el('target-score').addEventListener('input', () => { if (unlocked(S())) renderTransform(); });
        el('active-toggle').addEventListener('change', (e) => {
            S().active = e.target.checked;
            S().queue = nextBatch(S().model, S().active, S().rand);
        });
        document.querySelectorAll('[data-domain]').forEach((b) =>
            b.addEventListener('click', () => setDomain(b.dataset.domain)));
        el('reveal-keep').addEventListener('click', () => closeReveal(false));
        el('reveal-more').addEventListener('click', () => closeReveal(true));
        el('reveal-overlay').addEventListener('click', (e) => {
            if (e.target === el('reveal-overlay')) closeReveal(false);
        });
        document.addEventListener('keydown', (e) => {
            if (revealOpen()) {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    (document.activeElement === el('reveal-keep') ? el('reveal-more') : el('reveal-keep')).focus();
                    return;
                }
                if (e.key === 'Escape') {
                    e.preventDefault();
                    closeReveal(false);
                }
                return;
            }
            // Single-key shortcuts only operate while the rating component has focus.
            if (e.ctrlKey || e.altKey || e.metaKey || e.isComposing || e.target.isContentEditable ||
                ['INPUT', 'TEXTAREA', 'SELECT', 'SUMMARY'].includes(e.target.tagName) ||
                !e.target.closest('#card, .rate-row')) return;
            if (e.key === 'ArrowRight') { e.preventDefault(); rate(1); }
            else if (e.key === 'ArrowLeft') { e.preventDefault(); rate(0); }
            else if (e.key.toLowerCase() === 'u') undo();
            else if (e.key.toLowerCase() === 's') skip();
        });

        // Drag-to-swipe on the card, like the app.
        const card = el('card');
        let drag = null;
        card.addEventListener('pointerdown', (e) => {
            if (revealOpen() || !e.isPrimary || e.button !== 0) return;
            drag = { x0: e.clientX, y0: e.clientY, dx: 0 };
            card.setPointerCapture(e.pointerId);
            card.style.transition = 'none';
        });
        card.addEventListener('pointermove', (e) => {
            if (!drag) return;
            drag.dx = e.clientX - drag.x0;
            card.style.transform = 'translateX(' + drag.dx + 'px) rotate(' + drag.dx / 18 + 'deg)';
            card.style.opacity = String(Math.max(0.55, 1 - Math.abs(drag.dx) / 500));
        });
        const endDrag = (e) => {
            if (!drag) return;
            const dx = drag.dx;
            drag = null;
            if (e.type !== 'pointercancel' && Math.abs(dx) > 90) {
                const dir = dx > 0 ? 1 : 0;
                card.style.transition = 'transform 0.18s ease-in, opacity 0.18s';
                card.style.transform = 'translateX(' + (dx > 0 ? 480 : -480) + 'px) rotate(' + dx / 10 + 'deg)';
                card.style.opacity = '0';
                setTimeout(() => {
                    card.style.transition = 'none';
                    card.style.transform = 'none';
                    card.style.opacity = '1';
                    rate(dir);
                }, 160);
            } else {
                card.style.transition = 'transform 0.2s, opacity 0.2s';
                card.style.transform = 'none';
                card.style.opacity = '1';
            }
        };
        card.addEventListener('pointerup', endDrag);
        card.addEventListener('pointercancel', endDrag);

        setDomain(domainKey);
    });
})();
