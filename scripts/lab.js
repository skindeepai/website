'use strict';
const labToggle = document.querySelector('.lab-menu');
const labNav = document.getElementById('lab-nav');
function closeLabMenu() {
    labNav.classList.remove('open');
    labToggle.setAttribute('aria-expanded', 'false');
}
labToggle.addEventListener('click', () => {
    const open = labNav.classList.toggle('open');
    labToggle.setAttribute('aria-expanded', String(open));
});
document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && labNav.classList.contains('open')) { closeLabMenu(); labToggle.focus(); }
});
window.addEventListener('resize', () => { if (window.innerWidth > 800) closeLabMenu(); });
document.querySelectorAll('pre').forEach(pre => { pre.tabIndex = 0; });
document.querySelectorAll('.table-scroll').forEach(region => {
    region.tabIndex = 0;
    region.setAttribute('role', 'region');
    if (!region.hasAttribute('aria-label')) {
        const section = region.closest('section');
        const heading = section && section.querySelector('h2, h3');
        region.setAttribute('aria-label', (heading ? heading.textContent + ': ' : '') + 'table; scroll horizontally if needed');
    }
});
const filter = document.getElementById('research-filter');
const search = document.getElementById('research-search');
function filterResearch() {
    let count = 0;
    document.querySelectorAll('.experiment').forEach(item => {
        const matches = (filter.value === 'all' || item.dataset.track === filter.value) && item.textContent.toLowerCase().includes(search.value.toLowerCase());
        item.hidden = !matches;
        if (matches) count++;
    });
    document.getElementById('research-count').textContent = count + ' experiments shown';
    document.querySelectorAll('.research-group').forEach(group => {
        group.hidden = !group.querySelector('.experiment:not([hidden])');
    });
}
if (filter && search) { filter.addEventListener('change', filterResearch); search.addEventListener('input', filterResearch); }
// Deep links still expose their target when technical material is collapsed.
function revealLinkedDetail() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (_) { return; }
    const target = id && document.getElementById(id);
    if (!target) return;
    let parent = target.parentElement;
    while (parent) { if (parent.tagName === 'DETAILS') parent.open = true; parent = parent.parentElement; }
    if (target.tagName === 'DETAILS') target.open = true;
    target.scrollIntoView();
}
window.addEventListener('hashchange', revealLinkedDetail);
revealLinkedDetail();
