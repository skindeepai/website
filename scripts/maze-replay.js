// Replay recorded decisions. This page does not run a model or use a solver.
(async function () {
    'use strict';
    const status = document.getElementById('maze-status');
    const svg = document.getElementById('maze-grid');
    const mazeSelect = document.getElementById('maze-choice');
    const pathSelect = document.getElementById('maze-method');
    const back = document.getElementById('maze-back');
    const next = document.getElementById('maze-next');
    const play = document.getElementById('maze-play');
    const outcome = document.getElementById('maze-outcome');
    let protocol, recordings, step = 0, timer = null;
    function stop() {
        if (timer !== null) clearInterval(timer);
        timer = null;
        play.textContent = 'Play recording';
    }
    function element(tag, attrs, text) {
        const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
        for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
        if (text !== undefined) node.textContent = text;
        return node;
    }
    function draw() {
        const record = recordings.find(r => r.maze === mazeSelect.value && r.path === pathSelect.value);
        if (!record) throw new Error('Recording missing');
        const maze = protocol.mazes[record.maze];
        const move = step ? record.steps[step - 1] : null;
        const position = move ? move.next_position : record.start;
        svg.replaceChildren();
        for (let cell = 0; cell < 16; cell++) {
            const x = (cell % 4) * 60, y = Math.floor(cell / 4) * 60;
            const wall = maze.walls.includes(cell);
            svg.append(element('rect', {x: x + 1, y: y + 1, width: 58, height: 58, fill: wall ? '#182438' : '#ffffff', stroke: '#596b80'}));
            if (cell === record.goal) svg.append(element('text', {x: x + 30, y: y + 39, 'text-anchor': 'middle', fill: '#17635c', 'font-size': 25, 'font-weight': 700}, 'G'));
            if (cell === position) {
                svg.append(element('circle', {cx: x + 30, cy: y + 30, r: 21, fill: '#315d8e', stroke: '#ffffff', 'stroke-width': 2}));
                svg.append(element('text', {x: x + 30, y: y + 38, 'text-anchor': 'middle', fill: '#ffffff', 'font-size': 22, 'font-weight': 700}, cell === record.goal ? '✓' : 'A'));
            }
        }
        const coordinates = c => `row ${Math.floor(c / 4) + 1}, column ${(c % 4) + 1}`;
        svg.setAttribute('aria-label', `Four by four maze. Agent at ${coordinates(position)}. Goal at ${coordinates(record.goal)}. Walls at ${maze.walls.map(coordinates).join('; ')}.`);
        const detail = move ? ` ${['UP', 'RIGHT', 'DOWN', 'LEFT'][move.prediction]}. ${move.legal ? 'Legal move.' : 'Blocked move; the agent stayed in place.'} ${move.depth ? `Used ${move.depth} of 24 layers.` : 'Shortest-path reference.'}` : ' Choose Next move to inspect a decision.';
        status.textContent = `Move ${step} of ${record.steps.length}.${detail}`;
        outcome.textContent = `Recorded outcome: ${record.goal_reached ? 'goal reached' : 'goal not reached'} in ${record.actions} actions. Shortest possible route: ${record.shortest_path_length} moves.`;
        back.disabled = step === 0;
        next.disabled = step === record.steps.length;
        play.disabled = record.steps.length === 0;
        if (step === record.steps.length) stop();
    }
    try {
        const responses = await Promise.all(['results/maze-actions/protocol.json', 'results/maze-actions/episodes.json', 'results/maze-actions/result.json'].map(p => fetch(p)));
        if (responses.some(r => !r.ok)) throw new Error('Recording unavailable');
        [protocol, recordings] = await Promise.all(responses.slice(0, 2).map(r => r.json()));
        for (const [i, id] of protocol.episode_ids.entries()) {
            const option = document.createElement('option');
            option.value = id; option.textContent = `Maze ${i + 1}`; mazeSelect.append(option);
        }
        mazeSelect.disabled = pathSelect.disabled = false;
        for (const select of [mazeSelect, pathSelect]) select.addEventListener('change', () => { stop(); step = 0; draw(); });
        back.addEventListener('click', () => { stop(); step--; draw(); });
        next.addEventListener('click', () => { stop(); step++; draw(); });
        play.addEventListener('click', () => {
            if (timer !== null) { stop(); return; }
            const record = recordings.find(r => r.maze === mazeSelect.value && r.path === pathSelect.value);
            if (step === record.steps.length) { step = 0; draw(); }
            play.textContent = 'Pause recording';
            timer = setInterval(() => { step++; draw(); }, 700);
        });
        draw();
    } catch (error) {
        stop(); back.disabled = next.disabled = play.disabled = true;
        status.textContent = 'The completed recording could not be loaded. Try reloading this page.';
    }
})();
