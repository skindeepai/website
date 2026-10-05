"""Longer-context topic, its results page and the one-step decision result.

Every number comes from results/long-context/result.json. Rows still running there
render as "running". Called by organize_results.update before refresh_navigation.
"""
import json
import re
from html import escape as e
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = 'results/long-context/result.json'


def link(href, label):
    return '<a href="' + e(href, quote=True) + '">' + e(label) + '</a>'


def local(path, label):
    return link('@/' + path, label)


def refs(items):
    return '<p class="small">' + ' · '.join(items) + '</p>'


def table(headers, rows, label):
    def cell(value):
        return e(str(value)) if str(value) != '' else '–'
    return ('<div class="table-scroll" tabindex="0" role="region" aria-label="' + e(label, quote=True) + '"><table><thead><tr>'
            + ''.join('<th scope="col">' + e(h) + '</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(('<th scope="row">' + cell(v) + '</th>') if i == 0 else '<td>' + cell(v) + '</td>' for i, v in enumerate(row)) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def flow(items, caption):
    return '<figure class="method-figure"><ol class="method-flow">' + ''.join('<li>' + e(v) + '</li>' for v in items) + '</ol><figcaption>' + e(caption) + '</figcaption></figure>'


def section(pages, name, ident, html):
    body = re.sub(r'<section id="' + ident + r'">.*?</section>', '', pages[name].get('body', ''), flags=re.S)
    pages[name]['body'] = body + '<section id="' + ident + '">' + html + '</section>'


def update(pages):
    d = json.loads((ROOT / DATA).read_text(encoding='utf-8'))
    lab = d['lab']
    src = d['sources']

    def note(key, label):
        return link(lab + src[key], label)

    rows = d['comparison']['rows']
    running = lambda r: r['status'] != 'measured'
    value = lambda r, k: 'running' if running(r) else r[k]

    # ---------- Topic page: lay summary, example, headline, one short section per approach.
    headline = table(['Strategy', 'Files', 'Right of 24', 'Time', 'Most context held'],
                     [[r['strategy'], r['files'], value(r, 'right'), value(r, 'time'), value(r, 'peak')] for r in rows],
                     'Headline comparison on the running-ledger task')
    w = d['window']
    c = d['cache']
    body = flow(['20 batches of counter updates arrive, 121,000 tokens in all',
                 'The model keeps a small table of current values and drops each batch once it is folded in',
                 'At the end it reports the current value of 24 counters'],
                'The worked example behind the comparison below. A batch cannot be fetched a second time.')
    body += ('<p>A language model can only look at a limited amount of text at once. That limit is its context window. '
             'A long job, such as a big log or a long coding session, can produce more text than fits. Then the model either '
             'loses track, stops, or the text has to be managed.</p>'
             '<p>We tested ways to keep long jobs going on one machine: two Intel Arc Pro B70 cards running Qwen3.8-27B '
             '(FP8 weights, 16-bit cache). Nothing else was compressed.</p>')
    body += ('<h2>An example: keep a running ledger</h2><p>The model receives 20 batches of updates to counters. Some updates '
             'overwrite a value and some delete it. At the end it must give the current value of 24 counters. The whole stream '
             'is 121,000 tokens; the smaller budgets below allow 32,000.</p>' + headline
             + '<p class="small">One task family, one seed. “Running” rows are still being measured. '
             + local('context-results.html#comparison', 'Full table, written tokens and what went wrong') + '</p>')
    body += ('<h2>What we would do</h2><p><strong>Let the model keep its working data in files and keep its context small.</strong> '
             'With files allowed it got every answer right in under two minutes and never held more than 9,000 tokens. '
             'Where files are not possible, open the whole window with the exact cache as a safety net. Use in-context editing '
             'to keep the record of the model’s own steps small.</p>')
    body += '<h2>The approaches</h2>'
    approaches = [
        ('window', 'Open the whole window', [
            f'The model’s full window of {w["opens_tokens"]:,} tokens opens on two cards with nothing quantized, and the model reads and uses all of it.',
            'The cost is speed and precision: writing slows from 127 tokens a second at 8K to 26 at 250K, and from 120K up about 1 look-up in 40 returns a look-alike neighbour. It never invented a value.'],
         'docs/context-window.md', 'context-results.html#window'),
        ('cache', 'Reuse what was already read, exactly', [
            f'A new prefix cache keeps only what was made while reading a prompt, so each turn reads only what is new. A question over a cached 200K context starts in {c["cached_200k"][1]} instead of {c["cached_200k"][0]}.',
            f'Answers matched a server without the cache in {c["cases_identical"]} cases. The first, cold read is {c["cold_read_slower"]} slower. An edit in the middle resumes from the last kept state before it.'],
         'docs/context-prefix-cache.md', 'context-results.html#cache'),
        ('self-editing', 'Let the model edit its own context', [
            'The paper “Context Language Models” lets the model rewrite its own transcript. Here the model’s own edits lost nothing: it kept a small table, folded each batch in with a short script and deleted the raw batch.',
            'The paper’s agent still lost 5 of 24 values, all removed by its own rollback-and-retry after delivery. An improved agent and a stream larger than the window are running.',
            '“Unlimited” means an unlimited length of work, not unlimited detail at once: removed text is gone unless it was saved somewhere.'],
         'docs/context-management.md', 'context-results.html#comparison'),
        ('files', 'Keep the working data in files', [
            'With a shell and files allowed, the model kept its data on disk and its context under 9K tokens. Every answer was right, 14 to 22 times faster than keeping everything in the window or summarising.',
            'This works when code or search can do the reading. When the model itself must understand raw text, that text has to pass through the card at some point.'],
         'docs/context-management.md#files', 'context-results.html#comparison'),
        ('cleaning', 'Tidy the context on the CPU', [
            'A small cleaner running beside the model does not turn 32K into 100K. On 5.0M tokens of real agent sessions, strict no-loss cleaning freed 1.8%.',
            'Not re-sending the model’s old thinking frees about 10% of a typical call, but dropping it blindly left one run stuck.'],
         'docs/context-cleaning.md', 'context-results.html#cleaning'),
        ('disk', 'Keep the model’s cache on disk', [
            'Estimated, not built. The cache is exactly 64 KiB per token, so 200K tokens take about 13 GB. Streaming it from disk on every written token would slow writing to roughly 3-4 tokens a second, against about 35 in video memory.',
            'Parking a whole conversation’s cache on disk and restoring it in seconds is designed and lossless, but not built.'],
         'docs/context-on-disk.md', 'context-results.html#disk'),
    ]
    for ident, title, lines, doc, result in approaches:
        body += ('<section id="approach-' + ident + '"><h3>' + e(title) + '</h3>' + ''.join('<p>' + e(t) + '</p>' for t in lines)
                 + '<p class="next-links">' + local(result, 'Results') + ' · ' + local(doc, 'Method and raw data') + '</p></section>')
    body += ('<section id="approach-decisions"><h3>Short decisions do not need long thinking</h3><p>For a yes/no, a choice or a label, '
             'the model can answer in one step with thinking off. That result belongs to “Return a decision”.</p><p class="next-links">'
             + local('one-step-results.html', 'One-step decisions: results') + '</p></section>')
    body += ('<h2>Your questions</h2><ul class="page-links">'
             + ''.join('<li>' + local('research.html#' + i, q) + '</li>' for i, q in [
                 ('L06', 'Can disk make up for limited video memory?'),
                 ('L03', 'Can the context be edited surgically, without plain trimming or a full summary?'),
                 ('L01', 'How fast is it, and how much quality is lost?'),
                 ('L04', 'How is this different from an updated plan file?'),
                 ('L05', 'Can a small CPU cleaner turn 32K into 100K?')]) + '</ul>')
    body += ('<details class="plain-details"><summary>Limits, failures and sources</summary><p>One machine, mostly one seed, one task family, '
             'single-server timings, and the exact cache tested with one user. What did not work is listed with the results.</p>'
             + refs([local('context-results.html#failures', 'What did not work'), local('context-results.html#limits', 'Limits and what is not measured'),
                     local('docs/claims.md#longer-context', 'Claim ledger'), note('results', 'Lab summary (plain words)'),
                     note('review', 'Paper review')]) + '</details>')
    pages['context.html'] = dict(
        title='Working past the context limit',
        description='A model can only read so much text at once. Long jobs can still keep going without losing track.',
        eyebrow='APPROACH 05', status='Measured pilot', styles=['results.css'], body=body,
        meta_title='Longer context: big windows, exact caches and self-editing | SkinDeep',
        meta_description='Measured tests on Qwen3.8-27B: a 262K-token window, an exact prompt cache, self-editing context, files and CPU cleaning. Results, failures and raw data.')

    # ---------- Results page: every measured table, failures, limits, not measured.
    raw = 'results/long-context/'
    body = '<p class="study-sample">' + e(d['setup']) + ' Updated ' + e(d['updated']) + '.</p>'
    body += ('<section id="window"><h2>The whole window</h2><p>The window opens at ' + f'{w["opens_tokens"]:,}' + ' tokens. '
             'Read speed is how fast the prompt is taken in; first word is the wait before the answer starts; write speed is how fast the answer comes out.</p>'
             + table(w['columns'], w['rows'], 'Speed by context length')
             + '<p>Drafting gave exactly the no-drafting answer at every length. An earlier finding of empty answers above 212K was an artifact of sending '
             'the prompt as bare text; in chat form the model answers at 213K and 250K. These are single-server first looks, not package numbers.</p>'
             + refs([local('docs/context-window.md', 'Method and raw data'), note('window', 'Lab notes')]) + '</section>')
    r = d['recall']
    body += ('<section id="recall"><h2>Finding a buried code</h2><p>' + str(r['asked']) + ' codes asked in all. Each record differs from its neighbours only in its number.</p>'
             + table(r['columns'], r['rows'], 'Recall by context length')
             + '<p>Every miss was another record’s real code: the neighbour, or a number sharing most digits. The model never invented a value, '
             'and the start of a 250K context was recalled as well as the end.</p>'
             + refs([local(raw + 'recall-ordinary-words.log', 'Summaries: ordinary words'), local(raw + 'recall-wall-of-codes.log', 'Summaries: wall of codes')]) + '</section>')
    body += ('<section id="cache"><h2>The exact prefix cache</h2>'
             + table(['Case', 'Without reuse', 'With the exact cache'], [
                 ['Question over a 200K context', c['cached_200k'][0], c['cached_200k'][1]],
                 ['Repeated 30K prompt', c['repeat_30k'][0], c['repeat_30k'][1]],
                 ['30K prompt edited in the middle, first time', c['edit_30k_first'][0], c['edit_30k_first'][1]],
                 ['The same edit sent again', c['edit_30k_first'][0], c['edit_30k_again']]], 'Wait for the first word')
             + '<p>Identical to a server without the cache in ' + e(c['cases_identical']) + ' cases, including second and third turns of a conversation; '
             'a cold repeat at 120K matched the cached answer. Token ids were compared, up to 48 tokens per case.</p>'
             '<p>Costs: a cold read is ' + e(c['cold_read_slower']) + ' slower because it reads in fixed ' + str(c['block_tokens']) + '-token pieces; '
             'each kept state costs ' + e(c['state_memory']) + '. Tested with one user.</p>'
             + refs([local('docs/context-prefix-cache.md', 'Method and raw data'), local(raw + 'cache-probe.log', 'All 99 comparisons')]) + '</section>')
    body += ('<section id="comparison"><h2>Managing the context on a long task</h2><p>' + e(d['comparison']['task']) + ' One seed.</p>'
             + table(d['comparison']['columns'], [[r['strategy'], r['files'], r['budget']] + [value(r, k) for k in ['right', 'time', 'written', 'peak', 'note']] for r in rows],
                     'Context strategies on the running-ledger task')
             + '<ul><li><strong>Files win.</strong> Same answers, under 9K of context, 14 to 22 times faster than keeping everything or summarising.</li>'
             '<li><strong>Without files</strong>, keeping everything and summarising both got 24 of 24. Summarising was slower (41 against 26 minutes) because the cache makes a big context cheap to re-read and every summary breaks it.</li>'
             '<li><strong>The model’s own pruning lost nothing.</strong> The five wrong values were batches the paper’s harness rolled back after delivery. The overflow that set it off came from thinking: the first call wrote 14K tokens of planning, and old thinking is re-sent every call.</li>'
             '<li><strong>Dropping old thinking is not a safe switch.</strong> The model had been carrying its state in its thinking; without it, every call re-derived everything and hit the output cap.</li></ul>'
             + refs([local('docs/context-management.md', 'Method and raw data'), note('self_editing', 'Lab notes, call by call')]) + '</section>')
    cl = d['cleaning']
    body += ('<section id="cleaning"><h2>Cleaning on the CPU</h2><p>Measured on ' + e(cl['tokens']) + ' tokens of real agent sessions and on the model’s own runs.</p>'
             + table(cl['columns'], cl['rows'], 'Context freed by each cleaning rule')
             + '<p>A 32K window holds ' + e(cl['window_32k_holds']) + ', not 100K.</p>' + refs([local('docs/context-cleaning.md', 'Method and raw data')]) + '</section>')
    k = d['disk_estimate']
    body += ('<section id="disk"><h2>Cache on disk: an estimate</h2><p><strong>Estimated from measurements, not built.</strong></p>'
             + table(['Quantity', 'Value'], [['Cache per token', k['cache_per_token']], ['Cache for 200K tokens', k['cache_200k']], ['Drive read speed', k['drive_read']],
                                            ['Upload to the card', k['card_upload']], ['One written token, cache streamed from disk', k['step_time']],
                                            ['Write speed: from disk / in video memory', k['write_speed'][0] + ' / ' + k['write_speed'][1]],
                                            ['Total streamed to read a 200K prompt', k['prompt_200k_streamed']]], 'Disk cache estimate')
             + '<p>' + e(k['park_and_restore'][0].upper() + k['park_and_restore'][1:]) + '.</p>' + refs([local('docs/context-on-disk.md', 'The arithmetic')]) + '</section>')
    body += ('<section id="one-step"><h2>One-step decisions</h2><p>This result belongs to the decisions topic. '
             + local('one-step-results.html', 'Same labels as decoding, 6.5 times faster than thinking on easy items') + '.</p></section>')
    body += ('<section id="failures"><h2>What did not work</h2><ul>'
             '<li>The paper’s agent as shipped: its rollback-and-retry deleted delivered data, and its warnings fired when there was still room.</li>'
             '<li>The paper’s “suffix cache reuse”: it reuses cache computed for the text before an edit. Not exact, so rejected.</li>'
             '<li>A CPU-side cleaner: 1.8% without loss.</li>'
             '<li>Dropping old thinking as a blanket rule: one run stuck for 2.6 hours with no answer.</li></ul></section>')
    body += ('<section id="limits"><h2>Limits and what is not measured</h2><ul><li>One machine, single-server timings.</li>'
             '<li>Mostly one seed and one task family; the longest finished agent task is 121K tokens.</li>'
             '<li>The window that opens (262K) is larger than the window that is look-up exact (about 60K in the hardest test).</li>'
             '<li>The exact cache is tested with one user and compared by token ids, not scores.</li></ul>'
             '<p><strong>Not measured:</strong> several users sharing the exact cache; the one-card server; tasks other than the ledger; streams larger than the window (running); '
             'a cache kept on disk (estimate only); hard decisions where thinking changes the answer.</p></section>')
    pages['context-results.html'] = dict(
        title='Longer context: test results', description='Measured tables, failures and limits for each way of working past the context limit.',
        status='Measured pilot', styles=['results.css'], body=body)

    # ---------- The one-step decision result belongs to the decisions topic.
    o = d['one_step']
    body = ('<p class="study-sample">' + str(o['items']) + ' easy items (' + e(o['kinds']) + '). Qwen3.8-27B (FP8 weights, 16-bit cache) on two Intel Arc Pro B70 cards. One pass.</p>'
            + flow(['Thinking off', 'Allow only the label words as the first token', 'Return that label'], 'One model step. The output is always one of the allowed labels.')
            + table(['How the label is produced', 'Same label as normal decoding', 'Right', 'Median time per item'], [
                ['One step, labels only', o['same_as_decoding'], o['right_one_step'], o['median_seconds']['one_step']],
                ['Normal decoding, thinking off', '–', o['right_decoding'], o['median_seconds']['decoding']],
                ['With thinking (24 items)', '–', o['right_thinking'], o['median_seconds']['thinking']]], 'One-step decisions')
            + '<p><strong>Same answers, ' + e(o['speedup_vs_thinking'].replace('x', ' times')) + ' faster than thinking on easy items.</strong> The one-step label agreed with the thinking answer on '
            + e(o['same_as_thinking']) + '. With drafting, a short label already takes one step, so the saving is the thinking. The restriction adds a guarantee that the answer is a valid label.</p>'
            + '<p>Not measured: hard decisions where thinking changes the answer.</p>'
            + refs([local('results/long-context/one-step-choice-summary.json', 'Summary record'),
                    local('output-results.html', 'Related: reading two label scores on a small model'), local('context.html', 'Where this came from: longer context')]))
    pages['one-step-results.html'] = dict(
        title='A label in one step', description='For a yes/no, a choice or a label, switch thinking off and allow only the label words.',
        status='Measured pilot', styles=['results.css'], body=body)

    section(pages, 'decisions.html', 'one-step-link', '<h2>Skip the thinking for short decisions</h2><p>A large model that thinks before answering can return a label '
            'in one step instead: same answers on 80 easy items, 6.5 times faster.</p>' + refs([local('one-step-results.html', 'One-step decision results')]))
    section(pages, 'results.html', 'context', '<h2>Working past the context limit</h2><div class="topic-links">'
            + ''.join('<a href="@/' + p + '"><span><strong>' + e(t) + '</strong>' + e(s) + '</span></a>' for p, t, s in [
                ('context-results.html', 'How do long jobs keep going?', 'A 262K window, an exact cache, self-editing, files and CPU cleaning, with failures.'),
                ('one-step-results.html', 'Can a decision skip the thinking?', 'One-step labels on a large model, compared with decoding and thinking.')]) + '</div>')
