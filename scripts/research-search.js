'use strict';
(() => {
    const form = document.getElementById('answer-search'), input = document.getElementById('answer-query');
    const results = document.getElementById('answer-results'), status = document.getElementById('answer-status');
    form.hidden = false;
    let catalog, hash, worker, generation = 0, deadline, idle, unavailable = false, ready = false, indexFailed = false;
    function node(tag, text, className) {
        const el = document.createElement(tag); if (text) el.textContent = text; if (className) el.className = className; return el;
    }
    function render(rows, searching = false) {
        const opened = new Set([...results.querySelectorAll('details[open]')].map(n => n.dataset.answer));
        results.replaceChildren();
        if (!rows.length) {
            if (searching) return;
            results.append(node('h2', 'No close answer found'), node('p', 'Try a method or topic such as keeper, summaries, learning preferences, or early stopping.'));
            return;
        }
        results.append(node('h2', 'Closest answers'));
        rows.forEach((row, index) => {
            const item = node('details', '', 'search-answer'); item.dataset.answer = String(row.id);
            item.open = index === 0 || opened.has(String(row.id));
            const title = /^(Measured results|How it works|Evidence so far|What it shows|Tradeoffs)$/.test(row.title) ? row.page + ': ' + row.title.toLowerCase() : row.title;
            item.append(node('summary', title), node('p', row.topic, 'search-topic'));
            for (const paragraph of row.answer.split('\n\n')) item.append(node('p', paragraph));
            const link = node('a', 'Read the source →'); link.href = row.url;
            const source = node('p', '', 'search-source'); source.append(link, document.createTextNode(' · ' + row.page)); item.append(source);
            results.append(item);
        });
    }
    function stopWorker() {
        if (worker) worker.terminate(); worker = null; ready = false;
        clearTimeout(deadline); clearTimeout(idle);
    }
    function fallback(message) {
        stopWorker(); unavailable = true;
        if (catalog && input.value.trim()) render(ResearchSearch.rank(input.value.trim(), catalog));
        if (input.value.trim()) status.textContent = message + ' Keyword results are shown.';
    }
    function semantic(query, id) {
        if (unavailable || !hash) {
            render(ResearchSearch.rank(query, catalog));
            status.textContent = 'Keyword results. Meaning search is unavailable in this browser.'; return;
        }
        try {
            if (!worker) {
                worker = new Worker('scripts/research-search-worker.js');
                const current = worker;
                worker.onmessage = event => {
                    if (worker !== current || event.data.id !== generation) return;
                    clearTimeout(deadline);
                    if (event.data.error) { fallback('Meaning search could not load.'); return; }
                    ready = true;
                    render(ResearchSearch.rank(input.value.trim(), catalog, event.data.scores));
                    status.textContent = 'Matched by words and meaning across ' + catalog.rows.length + ' prepared answers.';
                    clearTimeout(idle); idle = setTimeout(stopWorker, 120000);
                };
                worker.onerror = () => { if (worker === current) fallback('Meaning search could not load.'); };
            }
            status.textContent = ready ? 'Checking meaning… Keyword results are ready.' : 'Keyword results are ready. Loading meaning search on this device (about 50 MB the first time)…';
            clearTimeout(deadline); clearTimeout(idle);
            deadline = setTimeout(() => fallback('Meaning search took too long.'), ready ? 15000 : 120000);
            worker.postMessage({ id, query, hash });
        } catch (_) { fallback('Meaning search is unavailable in this browser.'); }
    }
    function run(save) {
        const query = input.value.trim().slice(0, 400); input.value = query;
        document.querySelector('.search-examples').hidden = Boolean(query);
        generation++; clearTimeout(deadline); clearTimeout(idle);
        if (save) history.replaceState(null, '', query ? '#q=' + encodeURIComponent(query) : location.pathname);
        if (!query) {
            results.replaceChildren(); status.textContent = catalog ? 'Search ' + catalog.rows.length + ' prepared answers.' : 'Loading the answer index…';
            if (worker) idle = setTimeout(stopWorker, 120000);
            return;
        }
        if (!catalog) { status.textContent = indexFailed ? 'The answer index could not load. Reload to retry, or browse the research FAQ below.' : 'Loading the answer index…'; return; }
        render(ResearchSearch.rank(query, catalog), true);
        semantic(query, generation);
    }
    function fromHash() {
        try { input.value = new URLSearchParams(location.hash.slice(1)).get('q') || ''; } catch (_) { input.value = ''; }
        run(false);
    }
    form.addEventListener('submit', event => { event.preventDefault(); run(true); });
    input.addEventListener('input', () => {
        generation++; clearTimeout(deadline); clearTimeout(idle);
        if (worker) idle = setTimeout(stopWorker, 120000);
        if (!input.value) run(true);
        else if (results.children.length) status.textContent = 'Press Search to update the answers.';
    });
    document.querySelectorAll('[data-question]').forEach(button => button.addEventListener('click', () => { input.value = button.dataset.question; run(true); }));
    window.addEventListener('hashchange', fromHash);
    window.addEventListener('pagehide', stopWorker);
    fetch('search/answers.json').then(async response => {
        if (!response.ok) throw new Error('Index unavailable');
        const bytes = await response.arrayBuffer();
        catalog = ResearchSearch.index(JSON.parse(new TextDecoder().decode(bytes)));
        try {
            if (crypto.subtle) hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(n => n.toString(16).padStart(2, '0')).join('');
        } catch (_) { hash = null; }
        if (location.hash.startsWith('#q=')) fromHash(); else run(false);
    }).catch(() => { indexFailed = true; status.textContent = 'The answer index could not load. Reload to retry, or browse the research FAQ below.'; });
})();
