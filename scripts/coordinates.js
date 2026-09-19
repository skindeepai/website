'use strict';
(() => {
    const image = document.getElementById('coordinate-image');
    const stage = document.getElementById('coordinate-stage');
    const marker = document.getElementById('coordinate-marker');
    const output = document.getElementById('coordinate-output');
    const instruction = document.getElementById('coordinate-instruction');
    const save = document.getElementById('coordinate-export');
    let point = null, objectURL = null;
    function render() {
        marker.hidden = !point;
        save.disabled = !point;
        if (!point) { output.textContent = 'Choose a point in the image. No model runs in this workbench.'; return; }
        marker.style.left = (point.x * 100) + '%';
        marker.style.top = (point.y * 100) + '%';
        output.textContent = `Normalized: (${point.x.toFixed(4)}, ${point.y.toFixed(4)}) · Image pixels: (${Math.round(point.x * (image.naturalWidth - 1))}, ${Math.round(point.y * (image.naturalHeight - 1))})`;
    }
    stage.addEventListener('click', event => {
        if (!image.complete || !image.naturalWidth) return;
        const box = image.getBoundingClientRect();
        point = {x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)), y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height))}; render();
    });
    stage.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Enter', ' '].includes(event.key)) return;
        event.preventDefault();
        if (!point) point = {x: .5, y: .5};
        const step = event.shiftKey ? .05 : .01;
        if (event.key === 'ArrowLeft') point.x -= step;
        if (event.key === 'ArrowRight') point.x += step;
        if (event.key === 'ArrowUp') point.y -= step;
        if (event.key === 'ArrowDown') point.y += step;
        point.x = Math.max(0, Math.min(1, point.x)); point.y = Math.max(0, Math.min(1, point.y)); render();
    });
    document.getElementById('coordinate-file').addEventListener('change', event => {
        const file = event.target.files[0];
        if (!file) return;
        if (!/^image\/(png|jpeg|webp)$/.test(file.type) || file.size > 15000000) { output.textContent = 'Use a PNG, JPEG or WebP image smaller than 15 MB.'; return; }
        if (objectURL) URL.revokeObjectURL(objectURL);
        objectURL = URL.createObjectURL(file); point = null; render(); image.src = objectURL;
    });
    image.addEventListener('error', () => { point = null; render(); output.textContent = 'This image could not be opened.'; });
    document.getElementById('coordinate-clear').addEventListener('click', () => { point = null; render(); });
    save.addEventListener('click', () => {
        if (!point) return;
        const record = {version: 1, source: 'manual annotation', instruction: instruction.value, image: {width: image.naturalWidth, height: image.naturalHeight}, normalized: point, pixels: {x: Math.round(point.x * (image.naturalWidth - 1)), y: Math.round(point.y * (image.naturalHeight - 1))}};
        const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], {type: 'application/json'}));
        const link = document.createElement('a'); link.href = url; link.download = 'skindeep-coordinate.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    render();
    const recorded = document.getElementById('recorded-fixture');
    if (recorded) {
        fetch('results/coordinates/predictions.json').then(response => {
            if (!response.ok) throw new Error('Recorded result unavailable');
            return response.json();
        }).then(rows => {
            rows.forEach((row, index) => { const option = document.createElement('option'); option.value = index; option.textContent = (row.id.startsWith('mobile') ? 'Phone' : 'Desktop') + ': ' + row.instruction; recorded.append(option); });
            function show() {
                const row = rows[Number(recorded.value)];
                document.querySelector('.recorded-stage').style.width = Math.min(420, 300 * row.width / row.height) + 'px';
                document.getElementById('recorded-image').src = row.image;
                const dot = document.getElementById('recorded-point'); dot.style.left = row.x * 100 + '%'; dot.style.top = row.y * 100 + '%';
                const target = document.getElementById('recorded-box'); target.hidden = !row.box;
                if (row.box) {
                    target.style.left = row.box.x / row.width * 100 + '%'; target.style.top = row.box.y / row.height * 100 + '%';
                    target.style.width = row.box.width / row.width * 100 + '%'; target.style.height = row.box.height / row.height * 100 + '%';
                }
                document.getElementById('recorded-output').textContent = row.box ? (row.hit ? 'The click landed inside the target.' : 'The click missed the target.') : 'No target exists. The model guessed anyway.';
                document.getElementById('recorded-coordinates').textContent = `Relative image position: x = ${row.x.toFixed(4)}, y = ${row.y.toFixed(4)} (0 to 1).`;
                document.getElementById('recorded-full').href = row.image;
            }
            recorded.addEventListener('change', show); show();
        }).catch(() => { document.getElementById('recorded-output').textContent = 'Recorded predictions could not load. Serve this checkout with the local Python server to view them.'; });
    }
})();
