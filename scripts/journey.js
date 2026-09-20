// Navigation only: keep the selected demo's approach and evidence together.
(function () {
    'use strict';
    const node = document.getElementById('journey-config');
    if (!node) return;
    const config = JSON.parse(node.textContent);
    const select = document.getElementById(config.selector);
    if (!select) return;
    function update() {
        const selected = config.modes[select.value];
        if (!selected) return;
        for (const key of ['approach', 'results', 'evidence']) {
            const anchor = document.querySelector('[data-journey="' + key + '"]');
            if (anchor && selected[key]) {
                anchor.href = selected[key].href;
                anchor.textContent = selected[key].label;
            }
        }
        const note = document.getElementById('journey-note');
        if (note) note.textContent = selected.note || '';
        const url = new URL(location.href);
        url.searchParams.set(config.parameter, select.value);
        history.replaceState(history.state, '', url);
    }
    select.addEventListener('change', update);
    update();
})();
