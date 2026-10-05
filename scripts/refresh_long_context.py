"""Render the context explainers from templates and the published measurement record."""
import json
import re
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = 'results/long-context/result.json'


def link(href, label):
    return '<a href="' + escape(href, quote=True) + '">' + escape(label) + '</a>'


def table(headers, rows, label, extra=''):
    return ('<div class="table-scroll ctx-table ' + extra + '" tabindex="0" role="region" aria-label="' + escape(label, quote=True) + '"><table><thead><tr>'
            + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(('<th scope="row">' + escape(str(v)) + '</th>') if i == 0 else '<td>' + escape(str(v)) + '</td>' for i, v in enumerate(row)) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def section(pages, name, ident, html):
    body = re.sub(r'<section id="' + ident + r'">.*?</section>', '', pages[name].get('body', ''), flags=re.S)
    pages[name]['body'] = body + '<section id="' + ident + '">' + html + '</section>'


def update(pages):
    d = json.loads((ROOT / DATA).read_text(encoding='utf-8'))
    base = d['lab']
    note = lambda key: base + d['sources'][key]
    raw = lambda path: base + 'data/2026-10-05-context/' + path
    c = d['cache']
    measured = [r for r in d['comparison']['rows'] if r['status'] == 'measured']
    decode = d['window']['decode_rows']
    cache_wait = table(['Prompt or edit', 'First token without reuse', 'First token with reuse'], [
        ['Repeated 30K prompt', c['repeat_30k'][0], c['repeat_30k'][1]],
        ['Middle edit in 30K, first request', c['edit_30k_first'][0], c['edit_30k_first'][1]],
        ['Same edit, repeated', c['edit_30k_first'][0], c['edit_30k_again']],
        ['Question over 200K (separate probes)', c['cached_200k'][0], c['cached_200k'][1]]], 'Wait for the first output token')
    cache_decode = table(['Standard 12-prompt check', 'First-pass output rate', 'Second-pass output rate', 'Reference answers matched'],
                         [[r['configuration'], f'{r["first_tok_s"]:.1f} tokens/s', f'{r["second_tok_s"]:.1f} tokens/s', '12/12 on each pass'] for r in c['decode_gate']],
                         'Decode rate with and without prefix caching')
    memory = d['disk_estimate']
    transfer = table(['Assumed sustained storage read rate', 'Time to move the 200K cache once'],
                     [[f'{r["gb_s"]:g} GB/s', f'At least {r["seconds"]:.2f} seconds'] for r in memory['transfer_examples']], 'Illustrative transfer times; not a model benchmark')
    comparison = table(['Strategy', 'External files', 'Context budget', 'Correct / 24', 'Elapsed time', 'Tokens generated'],
                       [[r['strategy'], r['files'], r['budget'], r['right'], r['time'], r['written']] for r in measured],
                       'One 121K-token ledger, seed 0', 'ctx-comparison')
    short_comparison = table(['Strategy', 'Correct / 24', 'Elapsed time'],
                             [[r['strategy'], r['right'], r['time']] for r in measured if r['id'] in ['keep', 'summary', 'files-edit', 'files-plain']],
                             'Equal-score strategies on the same ledger')
    method_rows = [
        ['Bigger active window', 'More text visible at once, up to the model and engine limits.', 'Generation slows with length; about 127 to ' + str(round(decode[-1]['tokens_per_second'])) + ' tokens/s in separate 8K to 250K probes.', 'Measured here; finite window.'],
        ['Exact prefix cache', 'Reuse previously computed prompt state.', 'Shorter wait to start; roughly unchanged decode rate in the standard check.', 'Measured here; no larger active window.'],
        ['Files and retrieval', 'Archive beyond the window; select what to load.', 'Smaller active context and less generation can help; tools and reads cost time.', '1.9-minute ledger measured; archive limited by storage and retrieval.'],
        ['CLM self-editing', 'Rewrite the conversation into useful current state.', 'Edits and rereading cost time; shorter subsequent context can help.', '19/24 in the original trial; delivery rollback caused the five misses.'],
        ['Periodic summaries', 'Replace history with a shorter account.', 'Summary generation and cache invalidation can outweigh the savings.', '24/24 in 41 minutes here; omitted details may be lost.'],
        ['Park a cache on disk', 'Retain an inactive session for later restoration.', 'Load/save wait; no per-token disk read needed after a full resident restore.', 'Proposed here; does not make the active state smaller.'],
        ['Stream active cache from disk/RAM', 'Trade data movement or CPU work for memory capacity.', 'Repeated transfers can slow generation; no measured decode rate here.', 'Not implemented or benchmarked in these runs.'],
        ['CPU text cleanup', 'Remove repetition and formatting noise.', 'No separate model-speed result; 1.83% fewer tokens in the census.', 'Measured text reduction; not a large window multiplier.']]
    methods = table(['Approach', 'What it buys', 'Effect on speed', 'Evidence and limit'], method_rows, 'Context approaches compared', 'ctx-methods')
    for label, path in [('Bigger active window', 'context-results.html#window'), ('Exact prefix cache', 'context-cache.html'),
                        ('Files and retrieval', 'context-files.html'), ('CLM self-editing', 'context-clm.html'),
                        ('Periodic summaries', 'context-clm.html'), ('Park a cache on disk', 'context-memory.html#parking'),
                        ('Stream active cache from disk/RAM', 'context-memory.html#streaming'), ('CPU text cleanup', 'context-clm.html#cleaning')]:
        methods = methods.replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + path, label) + '</th>')
    chart = '<figure class="ctx-chart"><figcaption>Observed output tokens per second. A shorter active context was faster in these recall probes.</figcaption>'
    for r in decode:
        chart += ('<div class="ctx-bar-row"><span>' + escape(r['context']) + '</span><span class="ctx-bar" aria-hidden="true"><i style="width:'
                  + f'{r["tokens_per_second"] / 130 * 100:.1f}' + '%"></i></span><strong>' + str(round(r['tokens_per_second'])) + ' tok/s</strong></div>')
    chart += '</figure>'
    source_list = '<ul class="ctx-source-list">'
    for key, title, description in [
        ('results', 'Research summary', 'Completed findings and pending work in the October 5 snapshot.'),
        ('window', 'Window, recall and one-step decisions', 'Preregistered probes, corrections and results.'),
        ('cache_test', 'Exact prefix cache tests', '99-case comparison and the separate decode-rate gate.'),
        ('cache_rules', 'How the cache works', 'Detailed engine analysis, state sizes and reuse rules.'),
        ('self_editing', 'Self-editing and ledger comparisons', 'Task rules, failure analysis and revised-agent design.'),
        ('cleaning', 'CPU cleanup census', 'Token accounting and potential retrieval costs.'),
        ('review', 'Paper and related research', 'CLM, alternatives and proposed disk-cache experiments.')]:
        source_list += '<li>' + link(note(key), title + ' (Markdown)') + '<span>' + escape(description) + '</span></li>'
    source_list += '<li>' + link(note('scripts') + 'README.md', 'Run instructions and scripts') + '</li></ul>'
    replacements = {
        'SETUP': escape(d['setup']), 'RESEARCH_TIME': escape(d['snapshot']['time']),
        'DECODE_250': str(round(decode[-1]['tokens_per_second'])),
        'RECALL_CORRECT': str(d['recall']['correct']), 'RECALL_ASKED': str(d['recall']['asked']),
        'COMPARISON_TABLE': comparison, 'SHORT_COMPARISON': short_comparison, 'METHOD_TABLE': methods,
        'DECODE_CHART': chart, 'DECODE_TABLE': table(['Context', 'Output tokens/s', 'Replies measured'],
            [[r['context'], str(round(r['tokens_per_second'])), r['samples']] for r in decode], 'Observed generation speed by context length'),
        'COLD_TABLE': table(d['window']['cold_columns'], d['window']['cold_rows'], 'First uncached prompt-reading probe'),
        'RECALL_TABLE': table(d['recall']['columns'], d['recall']['rows'], 'Exact code lookup by length'),
        'CACHE_WAIT_TABLE': cache_wait, 'CACHE_DECODE_TABLE': cache_decode,
        'MEMORY_TABLE': table(['Active tokens', 'Attention-cache payload across both GPUs'],
            [[f'{n:,}', f'{n * memory["bytes_per_token"] / 2**30:.2f} GiB'] for n in [8192, 32768, 60000, 120000, 200000, 262144]], 'Calculated attention cache payload'),
        'TRANSFER_TABLE': transfer, 'CLEANING_TABLE': table(d['cleaning']['columns'], d['cleaning']['rows'], 'Separate token-reduction and reasoning-history measurements'),
        'SOURCE_LIST': source_list,
        'RAW_RECALL_PROSE': raw('recall/recall-prose.json'), 'RAW_RECALL_LEDGER': raw('recall/recall-ledger.json'),
        'RAW_CACHE': raw('prefixcache-exact-mtp/cache.json'), 'RAW_GATE': raw('pcgate/results.json'),
        'CONTEXT_README': note('scripts') + 'README.md',
    }
    for key in ['results', 'window', 'cache_test', 'cache_rules', 'self_editing', 'cleaning', 'review']:
        replacements['NOTE_' + key.upper()] = note(key)

    registry = [
        ('context.html', 'Working past the context limit', 'What long-context experiments taught us about files, memory and speed.', None),
        ('context-questions.html', 'Longer context, plain answers', 'Common questions about memory, disk, speed and what “unlimited” can mean.', None),
        ('context-methods.html', 'Ways to work with more context', 'Compare what each approach keeps, what it costs and what the experiments showed.', None),
        ('context-files.html', 'Keep records in files, work with what matters', 'Why a 121K-token task finished with less than 9K of active context.', 'context-methods.html'),
        ('context-cache.html', 'Read once, reuse the unchanged start', 'Prefix caching reduces the wait before an answer; long-context generation still has a cost.', 'context-methods.html'),
        ('context-memory.html', 'Can disk replace GPU memory?', 'Text files, saved numerical state and active offloading solve different problems.', 'context-methods.html'),
        ('context-clm.html', 'Let the model edit its own context', 'CLM, summaries and state files: how a long task can use a short working history.', 'context-methods.html'),
        ('context-results.html', 'Longer context: the measured results', 'Task outcomes, speed, recall and the source records behind the explanations.', None),
    ]
    for name, title, description, parent in registry:
        body = (ROOT / 'content' / name).read_text(encoding='utf-8')
        for key, value in replacements.items():
            body = body.replace('{{' + key + '}}', value)
        assert not re.search(r'\{\{[A-Z_0-9]+\}\}', body), name + ': unresolved template value'
        if name == 'context.html':
            body += '<details><summary>Go directly to an approach</summary><ul class="ctx-source-list">' + ''.join(
                '<li id="approach-' + ident + '">' + link('@/' + target, label) + '</li>' for ident, target, label in [
                    ('window', 'context-results.html#window', 'The full window'), ('cache', 'context-cache.html', 'Exact prefix caching'),
                    ('self-editing', 'context-clm.html', 'Self-editing context'), ('files', 'context-files.html', 'Files and retrieval'),
                    ('cleaning', 'context-clm.html#cleaning', 'CPU cleanup'), ('disk', 'context-memory.html', 'Cache on disk'),
                    ('decisions', 'one-step-results.html', 'One-step decisions')]) + '</ul></details>'
        p = dict(title=title, description=description, status='October 5, 2026 research', styles=['results.css', 'context.css'], body=body)
        if name != 'context.html':
            p['journey'] = {'topic': {'href': 'context.html', 'label': 'Longer context overview'}}
            if parent:
                p['journey']['approach'] = {'href': parent, 'label': 'Compare approaches'}
            if name != 'context-results.html':
                p['body'] += '<p class="next-links">' + link('@/context-results.html', 'Detailed results and original research') + '</p>'
        pages[name] = p

    o = d['one_step']
    body = ('<p class="study-sample">80 easy yes/no, A–D, sentiment and routing items. Qwen3.8-27B on two B70 GPUs. One pass; thinking was also tested on a 24-item subset.</p>'
            + table(['Answer path', 'Correct', 'Median time'], [
                ['One token, labels restricted, thinking off', o['right_one_step'], o['median_seconds']['one_step'] + ' seconds'],
                ['Ordinary label decoding, thinking off', o['right_decoding'], o['median_seconds']['decoding'] + ' seconds'],
                ['Reasoning then label, 24-item subset', o['right_thinking'], o['median_seconds']['thinking'] + ' seconds']], 'Easy label decisions')
            + '<p>The one-step and ordinary decoding answers matched on 80/80 items. Both got 78/80 right. On the 24 items with thinking enabled, one-step and thinking answers matched on 24/24; both got 23/24 right.</p>'
            '<p>Median response time was about 0.09 seconds for either non-thinking path, versus 0.59 seconds with thinking. The roughly 6.5× ratio describes these easy-item timing samples; it is not a speedup over ordinary non-thinking decoding.</p>'
            '<p>The allowed labels began with distinct tokens. Restricting the first token ensures a valid label choice; it does not guarantee the choice is correct. All model layers still run. Hard decisions where reasoning changes the answer need a separate accuracy comparison.</p>'
            '<p>' + link('@/context-clm.html#thinking', 'Removing earlier reasoning is a different experiment') + ' · ' + link('@/output-results.html', 'Related: trained label outputs') + '</p>'
            '<details><summary>Original measurements and method</summary><p>' + link(raw('edge/choice.json'), 'Every item and timing') + ' · '
            + link(note('window'), 'Original protocol and result') + ' · ' + link('@/docs/one-step-decisions.md', 'Method note') + '</p></details>')
    pages['one-step-results.html'] = dict(title='A label in one step', description='On easy decisions, skipping reasoning saved time; shortening an already short label added little.', status='Measured pilot', styles=['results.css', 'context.css'], body=body)
    section(pages, 'decisions.html', 'one-step-link', '<h2>Short labels on a larger model</h2><p>One-step labels matched ordinary decoding on 80 easy items. On a 24-item subset, skipping reasoning kept the same answers; median response time was about 0.09 seconds without reasoning versus 0.59 with it.</p><p>' + link('@/one-step-results.html', 'One-step decision results') + '</p>')
    section(pages, 'results.html', 'context', '<h2>Working past the context limit</h2><p>A 121K-token ledger with under 9K active context, faster repeat questions, and the limits of disk-backed memory.</p><p>'
            + link('@/context.html', 'Plain-language overview') + ' · ' + link('@/context-results.html', 'Measured results') + ' · ' + link('@/context-questions.html', 'Common questions') + '</p>')
