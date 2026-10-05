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


def chart_image(name, alt, caption, height):
    return ('<figure class="ctx-plot"><picture><source media="(max-width:600px)" srcset="images/' + name + '-mobile.png">'
            '<img src="@/images/' + name + '.png" width="1548" height="' + str(height) + '" loading="lazy" decoding="async" alt="' + escape(alt, quote=True)
            + '"></picture><figcaption>' + caption + '</figcaption></figure>')


def update(pages):
    d = json.loads((ROOT / DATA).read_text(encoding='utf-8'))
    base = d['lab']
    note = lambda key: base + d['sources'][key]
    raw = lambda path: base + 'data/2026-10-05-context/' + path
    c = d['cache']
    measured = []
    for r in d['comparison']['rows']:
        if r['status'] != 'measured':
            continue
        if 'trials' in r:
            for trial in r['trials']:
                measured.append({**r, **trial, 'id': r['id'] + ('-' + trial['variant'] if 'variant' in trial else '-s' + str(trial['seed'])),
                                 'strategy': trial['strategy'] if 'strategy' in trial else r['strategy'] + ', seed ' + str(trial['seed'])})
        else:
            measured.append(r)
    decode = d['window']['decode_rows']
    overview_labels = {
        'keep-s0': ('Large window, run 1', 'context-large-window.html'),
        'keep-s1': ('Large window, run 2', 'context-large-window.html'),
        'paper': ('Self-editing (CLM)', 'context-self-editing.html'),
        'summary': ('Summaries', 'context-summaries.html'),
        'files-edit': ('Files, editing available', 'context-files-editing.html'),
        'files-plain': ('Files, no context editing', 'context-files-code.html'),
        'improved-121k-s0': ('Revised self-editing, run 1', 'context-revised-editing.html'),
        'improved-121k-s1': ('Revised self-editing, run 2', 'context-revised-editing.html'),
        'no-thinking': ('Remove old thinking', 'context-drop-thinking.html')}
    overview_order = ['keep-s0', 'keep-s1', 'summary', 'paper', 'improved-121k-s0', 'improved-121k-s1', 'files-plain', 'files-edit', 'no-thinking']
    by_id = {r['id']: r for r in measured}
    score = lambda r: 'No answer' if r.get('no_answer') or not r['right'].isdigit() else r['right'] + '/24'
    overview = table(['Approach', 'Correct', 'Time'], [[overview_labels[key][0], score(by_id[key]), by_id[key]['time']]
        for key in overview_order], 'Long-task approaches and results', 'ctx-summary-table')
    for label, path in overview_labels.values():
        overview = overview.replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + path, label) + '</th>')
    other = table(['Approach', 'Result'], [
        ['Prefix caching', 'Repeated 30K prompt: first token in 0.8 s instead of 11.4 s. Standard-check writing speed stayed near 89 tokens/s.'],
        ['Bigger active context', 'Observed writing speed: about 127 tokens/s at 8K, versus ' + str(round(decode[-1]['tokens_per_second'])) + ' at 250K, in separate prompt tests.'],
        ['CPU text cleanup', '1.83% fewer tokens with conservative cleanup.'],
        ['Active cache on disk', 'Not benchmarked. The 200K-token attention cache alone needs about 12.2 GiB.']],
        'Other context approaches and their findings', 'ctx-summary-table ctx-other-table')
    for label, path in [('Prefix caching', 'context-prefix-cache.html'), ('Bigger active context', 'context-large-window.html'),
                        ('CPU text cleanup', 'context-cleanup.html'), ('Active cache on disk', 'context-cache-offload.html')]:
        other = other.replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + path, label) + '</th>')
    cache_wait = table(['Prompt or edit', 'First token without reuse', 'First token with reuse'], [
        ['Repeated 30K prompt', c['repeat_30k'][0], c['repeat_30k'][1]],
        ['Middle edit in 30K, first request', c['edit_30k_first'][0], c['edit_30k_first'][1]],
        ['Same edit, repeated', c['edit_30k_first'][0], c['edit_30k_again']],
        ['Question over 200K (separate probes)', c['cached_200k'][0], c['cached_200k'][1]]], 'Wait for the first output token')
    cache_decode = table(['Server', 'First pass', 'Repeat pass', 'Exact per pass'],
                         [['Cache on' if i else 'Cache off', f'{r["first_tok_s"]:.1f} tokens/s', f'{r["second_tok_s"]:.1f} tokens/s', '12/12'] for i, r in enumerate(c['decode_gate'])],
                         'Decode rate with and without prefix caching')
    memory = d['disk_estimate']
    transfer = table(['Assumed sustained storage read rate', 'Time to move the 200K cache once'],
                     [[f'{r["gb_s"]:g} GB/s', f'At least {r["seconds"]:.2f} seconds'] for r in memory['transfer_examples']], 'Illustrative transfer times; not a model benchmark')
    comparison = table(['Strategy', 'External files', 'Context budget', 'Correct', 'Elapsed time', 'Tokens generated'],
                       [[r['strategy'], r['files'], r['budget'], score(r), r['time'], r['written']] for r in measured if r.get('input_tokens', 121000) == 121000],
                       '121K-token ledger trials, including both large-window and revised-agent seeds', 'ctx-comparison')
    method_rows = [
        ['Bigger active window', 'Holds more text at once. The window opened to 262K positions; longer active context slowed generation.'],
        ['Exact prefix cache', 'Reuses the unchanged start. Repeat questions began sooner; writing stayed near 89 tokens/s in the standard check.'],
        ['Files, no context editing', 'Code reads saved records and keeps the totals. 24/24 in 1.9 minutes on the 121K ledger.'],
        ['Files, editing available', 'Adds permission to rewrite the conversation. 24/24 in 1.9 minutes on the 121K ledger; no edit was needed.'],
        ['CLM self-editing', 'Rewrites selected parts of the working history. Original agent: 19/24 in 64 minutes; its harness lost five delivered batches.'],
        ['Periodic summaries', 'Replaces old history with a short account. It got 24/24 in 41 minutes; making summaries added work.'],
        ['Revised self-editing', 'Protects incoming batches and pins the current state. 24/24 and 21/24 on two 121K runs; 24/24 on a 478K stream.'],
        ['Remove old thinking', 'Drops earlier reasoning from later calls. One ledger run repeatedly rebuilt its state and returned no answer after 2.6 hours.'],
        ['Park a cache on disk', 'Saves an inactive session for later restoration. Save/restore takes time; not benchmarked here.'],
        ['Stream active cache from disk/RAM', 'Trades repeated transfers or CPU work for capacity. Requires engine support; not benchmarked here.'],
        ['CPU text cleanup', 'Removes repetition and formatting noise. Conservative cleanup reduced tokens by 1.83%.']]
    methods = table(['Approach', 'What changes, and the result'], method_rows, 'Context approaches compared', 'ctx-summary-table ctx-other-table')
    for label, path in [('Bigger active window', 'context-large-window.html'), ('Exact prefix cache', 'context-prefix-cache.html'),
                        ('Files, no context editing', 'context-files-code.html'), ('Files, editing available', 'context-files-editing.html'), ('CLM self-editing', 'context-self-editing.html'),
                        ('Periodic summaries', 'context-summaries.html'), ('Revised self-editing', 'context-revised-editing.html'), ('Remove old thinking', 'context-drop-thinking.html'),
                        ('Park a cache on disk', 'context-cache-parking.html'), ('Stream active cache from disk/RAM', 'context-cache-offload.html'), ('CPU text cleanup', 'context-cleanup.html')]:
        methods = methods.replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + path, label) + '</th>')
    chart = chart_image('context-decode-speed', 'Observed output rate: 127 tokens per second at 8K context, 99 at 30K, 70 at 60K, 49 at 120K, 40 at 160K, 35 at 200K, 31 at 230K and 27 at 250K.',
                        'Separate prompt tests: one reply each at 8K and 30K; medians of 20 replies at the longer lengths.', 936)
    source_list = '<ul class="ctx-source-list">'
    for key, title, description in [
        ('results', 'Research summary', 'Completed findings and pending work in the October 5 snapshot.'),
        ('window', 'Window, recall and one-step decisions', 'Preregistered probes, corrections and results.'),
        ('cache_test', 'Exact prefix cache tests', '99-case comparison and the separate decode-rate gate.'),
        ('cache_rules', 'How the cache works', 'Detailed engine analysis, state sizes and reuse rules.'),
        ('self_editing', 'Self-editing and ledger comparisons', 'Task rules, failure analysis and revised-agent design.'),
        ('cleaning', 'CPU cleanup census', 'Token accounting and potential retrieval costs.'),
        ('timing', 'Where the time went', 'Saved-run timing reconstruction and prefix reuse.'),
        ('review', 'Paper and related research', 'CLM, alternatives and proposed disk-cache experiments.')]:
        source_list += '<li>' + link(note(key), title + ' (Markdown)') + '<span>' + escape(description) + '</span></li>'
    source_list += '<li>' + link(note('scripts') + 'README.md', 'Run instructions and scripts') + '</li></ul>'
    replacements = {
        'SETUP': escape(d['setup']), 'RESEARCH_TIME': escape(d['snapshot']['time']),
        'DECODE_250': str(round(decode[-1]['tokens_per_second'])),
        'RECALL_CORRECT': str(d['recall']['correct']), 'RECALL_ASKED': str(d['recall']['asked']),
        'COMPARISON_TABLE': comparison, 'METHOD_TABLE': methods,
        'CLM_RESULT_TABLE': table(['Self-editing agent', 'Correct', 'Time'],
            [[overview_labels[key][0], by_id[key]['right'] + '/24', by_id[key]['time']] for key in ['paper', 'improved-121k-s0', 'improved-121k-s1']],
            'Original and revised self-editing trials', 'ctx-summary-table'),
        'TIMING_TABLE': table(d['timing_analysis']['columns'], d['timing_analysis']['rows'], 'Measured totals and estimated time breakdown', 'ctx-comparison'),
        'OVERVIEW_TABLE': overview, 'OVERVIEW_OTHER_TABLE': other,
        'TASK_CHART': chart_image('context-task-time', 'Large window: 26 minutes, 24 of 24 correct. Summaries: 41 minutes, 24 correct. Original self-editing: 64 minutes, 19 correct. Revised self-editing: 15 minutes and 24 correct on seed 0, 21 minutes and 21 correct on seed 1. Both file-using agents: 1.9 minutes, 24 correct.',
            '121K ledger: seed 0 for each approach, plus seed 1 for revised self-editing. The large-window seed-1 run returned no answer; see the ' + link('@/context-results.html#comparison', 'full comparison') + '.', 1170),
        'RECALL_CHART': chart_image('context-recall', 'Codes correct out of 60 per style. Ordinary words and look-alike codes: 60K, 60 and 60; 120K, 60 and 58; 160K, 59 and 58; 200K, 58 and 57; 230K, 60 and 60; 250K, 60 and 58.',
            'Correct codes out of 60 for each context length and filler style.', 864),
        'CACHE_CHART': chart_image('context-cache-speed', 'Repeated 30K prompt: cache off 11.4 seconds to first token, cache on 0.8 seconds. Separate standard 12-prompt check, second pass: 88.7 output tokens per second with cache off and 89.7 with cache on.',
            'Caching shortened the wait. The roughly 1% writing-speed difference is within normal variation.', 1206),
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
    for key in ['results', 'window', 'cache_test', 'cache_rules', 'self_editing', 'cleaning', 'review', 'timing']:
        replacements['NOTE_' + key.upper()] = note(key)

    def outcomes(items):
        return table(['Trial', 'Correct', 'Time'], [[label, score(by_id[key]), by_id[key]['time']] for key, label in items],
                     'Measured task outcomes', 'ctx-summary-table')

    replacements.update({
        'RESULT_WINDOW': outcomes([('keep-s0', '121K, run 1'), ('keep-s1', '121K, run 2')]),
        'RESULT_SUMMARIES': outcomes([('summary', '121K ledger')]),
        'RESULT_SELF_EDITING': outcomes([('paper', '121K ledger')]),
        'RESULT_REVISED': outcomes([('improved-121k-s0', '121K, run 1'), ('improved-121k-s1', '121K, run 2'), ('improved-480k', '478K stream')]),
        'RESULT_FILES_CODE': outcomes([('files-plain', '121K ledger'), ('files-480k-plain', '478K stream')]),
        'RESULT_FILES_EDITING': outcomes([('files-edit', '121K ledger'), ('files-480k-editing-available', '478K stream')]),
        'RESULT_DROP_THINKING': outcomes([('no-thinking', '121K ledger')]),
        'STREAM_TABLE': outcomes([('improved-480k', 'Revised self-editing'), ('files-480k-plain', 'Files, no context editing'), ('files-480k-editing-available', 'Files, editing available')]),
    })
    for label, path in [('Revised self-editing', 'context-revised-editing.html'), ('Files, no context editing', 'context-files-code.html'), ('Files, editing available', 'context-files-editing.html')]:
        replacements['STREAM_TABLE'] = replacements['STREAM_TABLE'].replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + path, label) + '</th>')
    for key, alt in [
        ('window', 'The active conversation grows: first 10 apples, then add 3, then remove 2. Earlier updates remain visible.'),
        ('summaries', 'The old messages are replaced by a summary saying apples: 11. The next call uses the summary, not the original history.'),
        ('self-editing', 'The agent edits its live transcript: apples: 10 plus add 3 becomes apples: 13, with the processed batch removed.'),
        ('revised', 'A room check protects the next batch; code updates the state; the next call receives the pinned state and keeps space for more data.'),
        ('files-code', 'Batch records stay in files. A saved script updates a state file. The model receives the short result, apples: 11.'),
        ('files-editing', 'Saved records and code keep the prompt small. Editing the live conversation is available if needed; it was not needed in the 121K run.'),
        ('drop-thinking', 'Earlier reasoning contains apples: 11. Removing it without saving that total leaves the next call needing to reconstruct it.'),
        ('prefix', 'The first question computes the document cache. The second reuses the unchanged document and processes a new question.'),
        ('cleanup', 'Repeated progress lines and terminal formatting are reduced to one readable line with a repetition count.'),
        ('parking', 'An inactive numerical cache moves from GPU memory to disk, then back to active memory before the session resumes.'),
        ('offload', 'Part of the active cache stays in RAM or disk. Needed portions repeatedly move into GPU working space during generation.'),
    ]:
        replacements['DIAGRAM_' + key.upper().replace('-', '_')] = (
            '<figure class="ctx-diagram"><picture><source media="(max-width:600px)" width="360" height="698" srcset="images/context-how-' + key + '-mobile.png">'
            '<img src="@/images/context-how-' + key + '.png" width="1200" height="360" decoding="async" alt="' + escape(alt, quote=True) + '"></picture></figure>')

    registry = [
        ('context.html', 'Working past the context limit', 'Which ways of managing an AI’s memory actually helped?', None),
        ('context-questions.html', 'Longer context, plain answers', 'Common questions about memory, disk, speed and what “unlimited” can mean.', None),
        ('context-methods.html', 'Ways to work with more context', 'Compare what each approach keeps, what it costs and what the experiments showed.', None),
        ('context-files.html', 'Keep records in files', 'A long task can use a small working context.', 'context-methods.html'),
        ('context-cache.html', 'Read once, reuse the unchanged start', 'Prefix caching reduces the wait before an answer; long-context generation still has a cost.', 'context-methods.html'),
        ('context-memory.html', 'Can disk replace GPU memory?', 'Text files, saved numerical state and active offloading solve different problems.', 'context-methods.html'),
        ('context-clm.html', 'Let the model edit its own context', 'CLM, summaries and state files: how a long task can use a short working history.', 'context-methods.html'),
        ('context-results.html', 'Longer context: the measured results', 'Task outcomes, speed, recall and the source records behind the explanations.', None),
        ('context-large-window.html', 'Use a larger context window', 'Keep more conversation in view: how it works, its results and where it runs out.', 'context-methods.html'),
        ('context-summaries.html', 'Replace old history with a summary', 'How periodic summaries make room, what they preserve and what they can lose.', 'context-methods.html'),
        ('context-self-editing.html', 'Let the model edit its conversation', 'What CLM self-editing actually changes, with a simple example and measured results.', 'context-methods.html'),
        ('context-revised-editing.html', 'Self-editing with a protected state', 'How delivery checks and an explicit state change the self-editing approach.', 'context-methods.html'),
        ('context-files-code.html', 'Keep records in files and use code', 'How a plain file-using agent completed long ledgers with a small active context.', 'context-methods.html'),
        ('context-files-editing.html', 'Files with context editing available', 'Two tools with different jobs: files store records, while editing can shorten the live conversation.', 'context-methods.html'),
        ('context-drop-thinking.html', 'Remove earlier thinking', 'What gets removed, when it saves space and why one long-task run stalled.', 'context-methods.html'),
        ('context-prefix-cache.html', 'Reuse the reading with a prefix cache', 'How cached computation helps a repeated prompt start answering sooner.', 'context-methods.html'),
        ('context-cleanup.html', 'Clean up text before the model reads it', 'What simple CPU cleanup removes and why its measured savings were small.', 'context-methods.html'),
        ('context-cache-parking.html', 'Park an inactive cache on disk', 'Save computed state between sessions, then restore it before generating.', 'context-methods.html'),
        ('context-cache-offload.html', 'Offload an active cache to RAM or disk', 'How moving numerical state can trade speed for capacity, and why it differs from saving text.', 'context-methods.html'),
    ]
    for name, title, description, parent in registry:
        body = (ROOT / 'content' / name).read_text(encoding='utf-8')
        for key, value in replacements.items():
            body = body.replace('{{' + key + '}}', value)
        assert not re.search(r'\{\{[A-Z_0-9]+\}\}', body), name + ': unresolved template value'
        if name == 'context.html':
            for ident, target in [('window', 'context-large-window.html'), ('cache', 'context-prefix-cache.html'),
                                  ('self-editing', 'context-self-editing.html'), ('files', 'context-files-code.html'),
                                  ('cleaning', 'context-cleanup.html'), ('disk', 'context-cache-offload.html'),
                                  ('decisions', 'context-results.html')]:
                body = body.replace('href="@/' + target + '"', 'id="approach-' + ident + '" href="@/' + target + '"', 1)
        p = dict(title=title, description=description, styles=['results.css', 'context.css'], body=body)
        if '<figure class="ctx-diagram">' in body:
            opening = re.match(r'<p>(.*?)</p>\s*', body, flags=re.S)
            p['meta_description'] = description
            p['description'] = re.sub(r'<[^>]+>', '', opening[1])
            p['body'] = body[opening.end():]
        if name != 'context.html':
            p['journey'] = {'topic': {'href': 'context.html', 'label': 'Longer context overview'}}
            if parent:
                p['journey']['approach'] = {'href': parent, 'label': 'Compare approaches'}
            if name != 'context-results.html' and '@/context-results.html' not in p['body']:
                p['body'] += '<p class="next-links">' + link('@/context-results.html', 'Detailed results and original research') + '</p>'
        p['body'] = '<div class="ctx-page">' + p['body'] + '</div>'
        pages[name] = p

    o = d['one_step']
    body = ('<p>The test used 80 easy yes/no, multiple-choice, sentiment and routing questions. A 24-question subset was also tested with thinking enabled.</p>'
            + table(['Answer path', 'Correct', 'Median time'], [
                ['One token, thinking off', o['right_one_step'], o['median_seconds']['one_step']],
                ['Ordinary label, thinking off', o['right_decoding'], o['median_seconds']['decoding']],
                ['Think, then label (subset)', o['right_thinking'], o['median_seconds']['thinking']]], 'Easy label decisions', 'ctx-summary-table')
            + '<p>The one-step and ordinary decoding answers matched on 80/80 items. Both got 78/80 right. On the 24 items with thinking enabled, one-step and thinking answers matched on 24/24; both got 23/24 right.</p>'
            '<p>Median response time was about 0.09 seconds for either non-thinking path, versus 0.59 seconds with thinking. The roughly 6.5× ratio describes these easy-item timing samples; it is not a speedup over ordinary non-thinking decoding.</p>'
            '<p>The allowed labels began with distinct tokens. Restricting the first token ensures a valid label choice; it does not guarantee the choice is correct. All model layers still run. Hard decisions where reasoning changes the answer need a separate accuracy comparison.</p>'
            '<p>' + link('@/context-clm.html#thinking', 'Removing earlier reasoning is a different experiment') + ' · ' + link('@/output-results.html', 'Related: trained label outputs') + '</p>'
            '<details><summary>Original measurements and method</summary><p>Qwen3.8-27B on two B70 GPUs, one pass. The one-token path restricted output to the allowed labels.</p><p>' + link(raw('edge/choice.json'), 'Every item and timing') + ' · '
            + link(note('window'), 'Original protocol and result') + ' · ' + link('@/docs/one-step-decisions.md', 'Method note') + '</p></details>')
    pages['one-step-results.html'] = dict(title='A label in one step', description='On easy decisions, skipping reasoning saved time; shortening an already short label added little.', styles=['results.css', 'context.css'], body='<div class="ctx-page">' + body + '</div>')
    section(pages, 'decisions.html', 'one-step-link', '<h2>Short labels on a larger model</h2><p>One-step labels matched ordinary decoding on 80 easy items. On a 24-item subset, skipping reasoning kept the same answers; median response time was about 0.09 seconds without reasoning versus 0.59 with it.</p><p>' + link('@/one-step-results.html', 'One-step decision results') + '</p>')
    section(pages, 'results.html', 'context', '<h2>Working past the context limit</h2><p>A 121K-token ledger with under 9K active context, faster repeat questions, and the limits of disk-backed memory.</p><p>'
            + link('@/context.html', 'Plain-language overview') + ' · ' + link('@/context-results.html', 'Measured results') + ' · ' + link('@/context-questions.html', 'Common questions') + '</p>')
