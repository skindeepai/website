'use strict';
(function (root) {
    const stop = new Set(('a an and are as at be been but by can could did do does for from had has have how i if in into is it its me more my of on or our s should so than that the their them there these they this to use using was we were what when where which who why will with would you your about here work works make get tell know explain research').split(' '));
    function words(text) {
        return (text.toLowerCase().normalize('NFKD').replace(/[\u0300-\u036f]/g, '').match(/[a-z0-9]+/g) || [])
            .filter(word => !stop.has(word)).map(word => word.length > 5 && word.endsWith('s') ? word.slice(0, -1) : word);
    }
    function near(a, b) {
        if (Math.abs(a.length - b.length) > 1) return false;
        let i = 0, j = 0, errors = 0;
        while (i < a.length && j < b.length) {
            if (a[i] === b[j]) { i++; j++; continue; }
            if (++errors > 1) return false;
            if (a.length >= b.length) i++;
            if (a.length <= b.length) j++;
        }
        return errors + (a.length - i) + (b.length - j) <= 1;
    }
    function index(rows) {
        const postings = new Map(), lengths = [];
        rows.forEach((row, i) => {
            const terms = new Map();
            for (const [text, weight] of [[row.answer, 1], [row.title, 4], [row.questions.join(' '), 3], [row.page, .5]]) {
                for (const word of words(text)) terms.set(word, (terms.get(word) || 0) + weight);
            }
            lengths[i] = words(row.answer).length + 30;
            for (const [word, count] of terms) {
                if (!postings.has(word)) postings.set(word, []);
                postings.get(word).push([i, count]);
            }
        });
        const average = lengths.reduce((a, b) => a + b, 0) / rows.length;
        for (const entries of postings.values()) {
            const idf = Math.log(1 + (rows.length - entries.length + .5) / (entries.length + .5));
            for (const entry of entries) entry[1] = idf * entry[1] * 2.2 / (entry[1] + 1.2 * (.25 + .75 * lengths[entry[0]] / average));
        }
        return { rows, postings };
    }
    function lexical(query, catalog) {
        const scores = new Float32Array(catalog.rows.length), hits = new Uint16Array(scores.length);
        const terms = [...new Set(words(query))].slice(0, 40);
        for (const term of terms) {
            const alternatives = catalog.postings.has(term) ? [term] : term.length >= 5 ? [...catalog.postings.keys()].filter(w => near(term, w)).slice(0, 4) : [];
            const matched = new Set();
            for (const word of alternatives) for (const [i, value] of catalog.postings.get(word)) {
                scores[i] += value * (word === term ? 1 : .7);
                matched.add(i);
            }
            for (const i of matched) hits[i]++;
        }
        return { scores, hits, terms: terms.length };
    }
    function rank(query, catalog, dense) {
        const lex = lexical(query, catalog);
        if (!lex.terms) return [];
        const maximum = Math.max(1e-9, ...lex.scores);
        const candidates = catalog.rows.map((row, i) => {
            const coverage = lex.hits[i] / lex.terms;
            const keyword = lex.scores[i] / maximum * coverage;
            const meaning = dense ? dense[i] : 0;
            const eligible = (coverage >= .6 && lex.scores[i] > 0 && (!dense || meaning >= .28)) || (dense && meaning >= .43);
            const score = dense ? .3 * keyword + .7 * Math.max(0, (meaning - .2) / .6) : keyword;
            return { row, score: score + (row.curated ? .025 : 0), eligible };
        }).filter(item => item.eligible).sort((a, b) => b.score - a.score || a.row.id - b.row.id);
        const found = [], urls = new Set(), texts = new Set(), titles = new Set();
        for (const item of candidates) {
            if (item.score < candidates[0].score * .4) continue;
            const excerpt = item.row.answer.slice(0, 180).toLowerCase();
            const title = item.row.page + '|' + item.row.title;
            if (urls.has(item.row.url) || texts.has(excerpt) || titles.has(title)) continue;
            urls.add(item.row.url); texts.add(excerpt); titles.add(title); found.push(item.row);
            if (found.length === 6) break;
        }
        return found;
    }
    const api = { words, index, lexical, rank };
    if (typeof module === 'object' && module.exports) module.exports = api; else root.ResearchSearch = api;
})(typeof self === 'object' ? self : globalThis);
