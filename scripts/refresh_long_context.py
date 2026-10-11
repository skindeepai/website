"""Render the context explainers from templates and the published measurement record."""
import hashlib
import json
import math
import re
from context_trial_import import CONFIG, load_published, read_json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = 'results/long-context/result.json'
RETENTION_DATA = ('https://github.com/steveseguin/b70-optimization-lab/tree/974df5792372b424d66d8225af171c4a94307e74/'
                  'experiments/qwen38-27b-b70/data/2026-10-05-context/retention')


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


def score(row):
    if row.get('void'):
        return 'Void (rule breach)'
    if row.get('no_answer'):
        return 'No answer'
    assert isinstance(row['correct'], int) and isinstance(row['asked'], int), row['id']
    assert 0 <= row['correct'] <= row['asked'] and row['asked'] > 0, row['id']
    return f'{row["correct"]}/{row["asked"]}'


def diagnostic_replacements(root=ROOT):
    """Render selected later diagnostics from portable, hash-bound audit snapshots."""
    selection = json.loads((root / 'results/long-context/diagnostics.json').read_text())
    assert selection['schema'] == 'context-diagnostic-selection.v1'
    assert re.fullmatch(r'[0-9a-f]{40}', selection['source_revision'])
    counts = {'semantic': 36, 'sparse': 4, 'history': 8, 'full_source': 2}
    assert set(selection['studies']) == set(counts)
    audits = {}
    sources = []
    labels = {'semantic': 'Short prose', 'sparse': 'Wide tables',
              'history': 'Historical questions', 'full_source': 'Complete source'}
    for name, count in counts.items():
        spec = selection['studies'][name]
        path = (root / spec['audit_path']).resolve()
        assert path.is_relative_to(root.resolve())
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec['audit_sha256'], name + ': audit hash mismatch'
        audit = json.loads(raw)
        assert audit['schema'] == spec['schema'] and audit['measurement_kind'] == 'model'
        assert audit['infrastructure_abort'] is False
        assert audit['planned_trials'] == audit['completed_trials'] == len(audit['trials']) == count
        assert not any(audit.get(k, 0) for k in ('failed_trials', 'incomplete_trials', 'unstarted_trials'))
        assert all(r['status'] == 'completed' for r in audit['trials'])
        audits[name] = audit
        base = selection['source_repository'] + '/blob/' + selection['source_revision'] + '/'
        sources.append(link(base + spec['lab_note_path'], labels[name]) + ' ('
                       + link('@/' + spec['audit_path'], 'saved audit') + ')')

    def fraction(rows, numerator='correct', denominator='asked'):
        for row in rows:
            assert type(row[numerator]) is int and type(row[denominator]) is int
            assert 0 <= row[numerator] <= row[denominator] and row[denominator] > 0
        return f'{sum(r[numerator] for r in rows)}/{sum(r[denominator] for r in rows)}'

    semantic = audits['semantic']['trials']
    arms = {arm: [r for r in semantic if r['arm'] == arm] for arm in ('summary', 'archive', 'quoted')}
    assert all(len(rows) == 12 and len({r['document_id'] for r in rows}) == 12 for rows in arms.values())
    assert all(r['cold_cache_known'] and r['cached_tokens'] == 0 for r in semantic)
    sparse = {(r['case_id'], r['arm']): r['native_audit'] for r in audits['sparse']['trials']}
    assert set(sparse) == {(case, arm) for case in ('sparse-n128-seed83', 'sparse-n128-seed97-dispatch')
                           for arm in ('archive', 'quoted')}
    assert all(r['cold_cost_interpretation'] for r in audits['sparse']['trials'])
    original = [sparse[('sparse-n128-seed83', arm)] for arm in ('archive', 'quoted')]
    transfer = [sparse[('sparse-n128-seed97-dispatch', arm)] for arm in ('archive', 'quoted')]
    history = audits['history']['trials']
    assert {(r['case_id'], r['condition']) for r in history} == {
        (case, condition) for case in ('t01-clinic', 't02-theatre') for condition in ('AS', 'QS', 'AH', 'QH')}
    assert all(r['cold_cost_interpretation'] for r in history)
    source_only = [r for r in history if r['retrieval_mode'] == 'source-only']
    historical = [r for r in history if r['retrieval_mode'] == 'history']
    assert len(source_only) == len(historical) == 4
    assert len({fraction([r['score']]) for r in source_only}) == 1
    assert len({r['score']['asked'] for r in historical}) == 1
    direct = audits['full_source']['trials']
    assert [r['case_id'] for r in direct] == ['t01-clinic', 't02-theatre']
    for row in direct:
        assert row['final_task_success'] and row['cost_eligible'] and row['cold_cost_interpretation']
        assert row['checkpoint_quality'] is None and row['event_quality'] is None
        assert math.isfinite(row['task_seconds']) and row['task_seconds'] > 0
        assert all(row[key] == h[key] for h in history if h['case_id'] == row['case_id']
                   for key in ('task_sha256', 'question_sha256')), 'Full-source and streaming tasks differ'
    checkpoints = lambda rows: fraction(rows, 'checkpoints_exact', 'checkpoints_checked')
    rows = [
        ['Short prose: summaries, saved tables, quoted changes',
         'Final answers: summaries ' + fraction(arms['summary']) + '; saved tables ' + fraction(arms['archive'])
         + '; quoted changes ' + fraction(arms['quoted']) + '.',
         'Saved states: tables ' + checkpoints(arms['archive']) + '; quoted changes ' + checkpoints(arms['quoted'])
         + '. Summary states were not observed.'],
        ['Wide numeric tables: original task and new seed/style',
         'Original task: both ' + fraction([original[0]]) + '. New task: both ' + fraction([transfer[0]]) + '.',
         'Original states: both ' + checkpoints([original[0]]) + '. New states: table '
         + checkpoints([transfer[0]]) + ', quoted changes ' + checkpoints([transfer[1]])
         + '. Correct final answers hid earlier table errors.'],
        ['Historical balances and changing owners: two documents',
         'Source retrieval: ' + fraction([source_only[0]['score']]) + ' in all four runs. Adding saved-state lookup: '
         + str(min(r['score']['correct'] for r in historical)) + '–'
         + str(max(r['score']['correct'] for r in historical)) + '/' + str(historical[0]['score']['asked']) + '.',
         'Saved states: ' + checkpoints([r['native_audit'] for r in history])
         + '. Wrong answers connected questions to initial ticket owners.'],
        ['Complete source in one request: the same two documents',
         'Clinic: ' + fraction([direct[0]['score']]) + ' in ' + f"{direct[0]['task_seconds']:.1f} s"
         + '. Theatre: ' + fraction([direct[1]['score']]) + ' in ' + f"{direct[1]['task_seconds']:.1f} s" + '.',
         'Final questions and source were available together. Intermediate states and extracted events were not measured.']]
    # These selected pairs support the word "both" above; never silently merge unequal scores.
    for pair in (original, transfer):
        assert fraction([pair[0]]) == fraction([pair[1]])
    assert checkpoints([original[0]]) == checkpoints([original[1]])
    return {'BOOKKEEPING_TABLE': table(['Test', 'Final answers', 'What the state checks show'], rows,
                                       'Later bookkeeping and historical-question diagnostics', 'ctx-comparison'),
            'BOOKKEEPING_SOURCES': ' · '.join(sources)}


FRACTION = re.compile(r'(\d+)/(\d+)')
DP_METHODS = {'window': 'Window-drop', 'compaction': 'Compaction', 'self_edit_read': 'Self-editing, read mode',
              'archive': 'Self-editing + archive and search', 'archive_rules': 'Archive and search + pinned rule lines',
              'archive_free': 'Archive and search with free-text notes', 'keeper': 'Keeper', 'keeper_proxy': 'Keeper as a proxy',
              'summary': 'Summarise at 75% of the window'}
DP_ORDER = ['keeper', 'keeper_proxy', 'archive', 'window', 'summary', 'compaction', 'self_edit_read', 'archive_rules', 'archive_free']


def fraction_parts(value):
    m = FRACTION.fullmatch(value)
    assert m, value
    c, n = int(m[1]), int(m[2])
    assert 0 <= c <= n and n > 0, value
    return c, n


def distance_replacements(dp):
    """Tables for the distance-probe page; every row's bins must add up to its totals."""
    assert dp['lab_commit'] and all(dp[k].startswith('https://github.com/') for k in ('note_url', 'data_url', 'charter_url'))
    rounds = {r['id']: r for r in dp['rounds']}
    assert set(rounds) == {'round1', 'round2', 'cap32-two-seeds', 'cap48-20261008'}
    assert all(dp['lab_commit'] in url for url in (dp['note_url'], dp['data_url'], dp['charter_url'], *dp['data_urls'].values()))
    for rnd in dp['rounds']:
        for r in rnd['rows']:
            assert r['method'] in DP_METHODS, r['id']
            bins = [fraction_parts(r[k]) for k in ('d1_4', 'd5_24', 'd25_99', 'd100_399') if k in r]
            types = [fraction_parts(r[k]) for k in ('current', 'old_value', 'who')]
            probes, final, total = (fraction_parts(r[k]) for k in ('probes', 'final', 'total'))
            for parts in (bins, types):
                assert tuple(map(sum, zip(*parts))) == probes, r['id'] + ': probe split does not add up'
            assert (probes[0] + final[0], probes[1] + final[1]) == total, r['id'] + ': total does not add up'
            for key in ('probes', 'final', 'total') if 'by_seed' in r else ():
                parts = [fraction_parts(x[key]) for x in r['by_seed']]
                assert tuple(map(sum, zip(*parts))) == fraction_parts(r[key]), r['id'] + ': seeds do not add up to the pooled ' + key

    def label(r):
        return DP_METHODS[r['method']] + ' (' + r['arm'] + ')'

    def final(r):
        return r['final'] + (' (' + r['final_note'] + ')' if r.get('final_note') else '')

    def rows(items, far=False):
        return [[label(r), r['cap'], str(r['seeds']), r['probes'], r['d1_4'], r['d5_24'], r['d25_99'],
                 *([r['d100_399']] if far else []), r['current'], r['old_value'], r['who'], final(r), r['total'], r['minutes']]
                for r in items]
    head = ['Method', 'Window', 'Seeds', 'Probes right', '1–4 back', '5–24 back', '25–99 back']
    tail = ['Current value', 'Old value', 'Who did it', 'Final questions', 'All', 'Minutes per run']
    by_method = lambda r: DP_ORDER.index(r['method'])
    pooled = sorted(rounds['cap32-two-seeds']['rows'], key=by_method)
    r2 = pooled + sorted((r for r in rounds['round2']['rows'] if r['cap'] == '16K'), key=by_method)
    seed_rows = [[label(r), str(x['seed']), x['probes'], x['final'] + (' (' + x['final_note'] + ')' if x.get('final_note') else ''),
                  x['total'], x['minutes']] for r in pooled for x in r['by_seed']]

    # A rule and a method stated once at the start: every row must add up to its 39 questions.
    rules = dp['rules']
    rule_rows = []
    for r in sorted(rules['rows'], key=by_method):
        assert r['method'] in DP_METHODS and r['final_confidential'] in ('blank', 'wrong', 'right'), r['id']
        parts = {k: fraction_parts(r[k]) for k in ('current', 'old_value', 'who', 'rule', 'house_method', 'final', 'total')}
        assert tuple(map(sum, zip(*(v for k, v in parts.items() if k != 'total')))) == parts['total'] and parts['total'][1] == 39, r['id']
        memory = tuple(map(sum, zip(*(parts[k] for k in ('current', 'old_value', 'who')))))
        confidential = {'blank': 'confidential one left blank', 'wrong': 'confidential one wrong', 'right': 'confidential one right'}
        rule_rows.append([label(r), f'{memory[0]}/{memory[1]}', r['rule'], r['house_method'],
                          r['final'] + ' (' + confidential[r['final_confidential']] + ')', r['total'], r['minutes']])
    base_rules = [fraction_parts(r['rule']) for r in rules['rows'] if r['method'] not in ('archive_rules', 'archive_free')]
    assert len(base_rules) == 4 and tuple(map(sum, zip(*base_rules))) == (0, 20)

    # The keeper on the same rule-and-method stream: every row must add up to its question count.
    keeper = dp['keeper']
    rules_by_id = {r['id']: r for r in rules['rows']}

    def rule_stream_parts(r, asked):
        assert r['method'] in DP_METHODS, r['id']
        parts = {k: fraction_parts(r[k]) for k in ('current', 'old_value', 'who', 'rule', 'house_method', 'final', 'total')}
        assert tuple(map(sum, zip(*(v for k, v in parts.items() if k != 'total')))) == parts['total'] == (parts['total'][0], asked), r['id']
        return parts

    assert [r['method'] for r in keeper['rows']] == ['keeper', 'keeper_proxy']
    assert [r['arm'] for r in keeper['rows']] == [r['arm'] for r in keeper['second_seed']] == ['N32j', 'X32b']
    first = {r['method']: r for r in keeper['rows']}
    second = {r['method']: r for r in keeper['second_seed']}
    assert set(second) == {'keeper', 'keeper_proxy'} and fraction_parts(second['keeper']['total'])[1] == 46
    proxy = keeper['proxy']
    before = {r['seed']: r for r in proxy['before_fixes']['rows']}
    assert set(before) == {0, 1} and all(r['arm'] == proxy['before_fixes']['arm'] == 'X32' and r['method'] == 'keeper_proxy' for r in before.values())
    compared = [rules_by_id[i] for i in keeper['compare_with']]
    assert [r['arm'] for r in compared] == ['B32iraR', 'W32'] and keeper['not_run']
    for r in keeper['rows'] + [before[0]] + compared:
        rule_stream_parts(r, 39)
    for r in (second['keeper_proxy'], before[1]):
        rule_stream_parts(r, 46)
    assert len(keeper['lessons']) == 3 and all(dp['lab_commit'] in proxy[k] for k in ('readme_url', 'code_url'))

    # Summarising at a threshold: one seed of the plain stream, beside the first-seed 32K runs of the other methods.
    sb = dp['summary_baseline']
    seed0 = {r['id']: r for r in rounds['round2']['rows'] if r['cap'] == '32K'}
    summary_runs = [sb['row']] + [seed0[i] for i in sb['compare_with']]
    for r in summary_runs:
        assert r['method'] in DP_METHODS and r['seeds'] == 1, r['id']
        types = tuple(map(sum, zip(*(fraction_parts(r[k]) for k in ('current', 'old_value', 'who')))))
        asked, last, total = (fraction_parts(r[k]) for k in ('probes', 'final', 'total'))
        assert types == asked and (asked[0] + last[0], asked[1] + last[1]) == total and total[1] == 29, r['id']
    summary_runs.sort(key=lambda r: -fraction_parts(r['total'])[0])
    summary_rows = [[label(r), r['total'], r['probes'], r['current'], r['old_value'], r['who'], final(r), r['minutes']] for r in summary_runs]

    # The 1M stream. The earlier run is a stopped run: its splits are checked against the rebuilt probe total and nothing else.
    m = dp['million']
    assert m['status'].startswith('Stopped run') and m['method'] in DP_METHODS and m['limit']
    probes = fraction_parts(m['probes'])
    for split in ('by_distance', 'by_type', 'by_third'):
        assert tuple(map(sum, zip(*map(fraction_parts, m[split].values())))) == probes, split + ' does not add up'
    assert round(100 * probes[0] / probes[1]) == m['probes_percent']
    # The keeper's run on the same stream finished and was graded, so its splits must also add up to its graded total.
    km = keeper['million']
    assert km['status'].startswith('Finished run') and km['method'] == 'keeper' and km['limit']
    k_probes, k_final, k_total = (fraction_parts(km[k]) for k in ('probes', 'final', 'total'))
    for split in ('by_distance', 'by_type', 'by_third'):
        assert list(km[split]) == list(m[split]), split + ': the two runs are split differently'
        assert tuple(map(sum, zip(*map(fraction_parts, km[split].values())))) == k_probes, 'keeper ' + split + ' does not add up'
        assert all(fraction_parts(km[split][k])[1] == fraction_parts(m[split][k])[1] for k in km[split]), split + ': different questions'
    assert (k_probes[0] + k_final[0], k_probes[1] + k_final[1]) == k_total
    assert round(100 * k_probes[0] / k_probes[1]) == km['probes_percent'] and round(100 * k_total[0] / k_total[1]) == km['total_percent']
    m_final = fraction_parts(m['final'])
    assert (probes[0] + m_final[0], probes[1] + m_final[1]) == fraction_parts(m['total']) and probes[1] + m_final[1] == k_total[1]
    k_cost = km['cost']
    assert k_cost['lm_calls'] == k_cost['agent_calls'] + k_cost['notes_calls'] and len(k_cost['minutes_per_100k']) == 9
    assert all(dp['lab_commit'] in url for url in km['evidence'].values())
    # The proxy's run on the same stream also finished and was graded: same questions, same checks.
    xm = proxy['million']
    assert xm['status'].startswith('Finished run') and xm['method'] == 'keeper_proxy' and xm['arm'] == first['keeper_proxy']['arm'] and xm['limit']
    x_probes, x_final, x_total = (fraction_parts(xm[k]) for k in ('probes', 'final', 'total'))
    for split in ('by_distance', 'by_type', 'by_third'):
        assert list(xm[split]) == list(m[split]), split + ': the proxy run is split differently'
        assert tuple(map(sum, zip(*map(fraction_parts, xm[split].values())))) == x_probes, 'proxy ' + split + ' does not add up'
        assert all(fraction_parts(xm[split][k])[1] == fraction_parts(m[split][k])[1] for k in xm[split]), split + ': different questions'
    assert (x_probes[0] + x_final[0], x_probes[1] + x_final[1]) == x_total and x_total[1] == k_total[1] and x_probes[1] == k_probes[1]
    assert round(100 * x_total[0] / x_total[1]) == xm['total_percent'] == round(xm['total_percent_one_decimal'])
    x_cost = xm['cost']
    assert len(x_cost['minutes_per_100k']) == 9 and all(dp['lab_commit'] in url for url in xm['evidence'].values())

    def run_cell(r):
        return r['total'] + ' in ' + str(r['minutes']) + ' min'
    first_asked, second_asked = fraction_parts(first['keeper']['total'])[1], fraction_parts(second['keeper']['total'])[1]
    keeper_rows = [
        [label(first['keeper']), run_cell(first['keeper']), run_cell(second['keeper']), km['total'] + ' in ' + str(k_cost['minutes']) + ' min'],
        [label(first['keeper_proxy']), run_cell(first['keeper_proxy']), run_cell(second['keeper_proxy']),
         xm['total'] + ' in ' + str(x_cost['minutes']) + ' min'],
        [DP_METHODS['keeper_proxy'] + ', before its fixes (' + before[0]['arm'] + ')', run_cell(before[0]), run_cell(before[1]), 'Not run']]
    for r in compared:
        keeper_rows.append([label(r), run_cell(r), 'Not run', m['total'] + ', a stopped run' if r['arm'] == m['arm'] else 'Not run'])
    names = {'d1_4': 'Asked 1–4 batches back', 'd5_24': 'Asked 5–24 back', 'd25_99': 'Asked 25–99 back',
             'd100_399': 'Asked 100–399 back', 'd400_plus': 'Asked 400 or more back', 'current': 'Current-value questions',
             'who': 'Who-did-it questions', 'house_method': 'House-method questions', 'rule': 'Standing-rule questions',
             'old_value': 'Old-value questions', 'probe_items_1_10': 'First third of the run (question sets 1–10)',
             'probe_items_11_20': 'Middle third (sets 11–20)', 'probe_items_21_29': 'Last third (sets 21–29)'}
    cost = m['cost']
    # One of the final questions asks for a confidential counter; the note beside each final score must say the same thing.
    rule_kept = {'right': 'Yes', 'null': 'No: answered null'}
    assert all(('right' in r['final_note']) == (r['final_confidential'] == 'right') for r in (km, xm, m)) and 'null' in m['final_note']
    million_rows = ([['All ' + str(k_total[1]) + ' questions', km['total'] + ' (' + str(km['total_percent']) + '%)',
                      xm['total'] + ' (' + str(xm['total_percent']) + '%)', m['total']],
                     ['All planted questions', km['probes'] + ' (' + str(km['probes_percent']) + '%)', xm['probes'],
                      m['probes'] + ' (' + str(m['probes_percent']) + '%)']]
                    + [[names[k], km[split][k], xm[split][k], v] for split in ('by_distance', 'by_type', 'by_third') for k, v in m[split].items()]
                    + [['Final questions', km['final'], xm['final'], m['final']],
                       ['Rule kept in the final questions', *(rule_kept[r['final_confidential']] for r in (km, xm, m))],
                       ['Minutes', str(k_cost['minutes']), str(x_cost['minutes']), str(cost['minutes'])],
                       ['Minutes per 100K tokens', str(k_cost['minutes_per_100k_mean']), str(x_cost['minutes_per_100k_mean']),
                        str(cost['minutes_per_100k_mean'])],
                       ['Model calls', f"{k_cost['lm_calls']:,} ({k_cost['agent_calls']:,} by the agent)",
                        f"{x_cost['agent_calls']:,} by the agent", f"{cost['lm_calls']:,}"]])
    segments = [str(x) for x in k_cost['minutes_per_100k']]
    x_segments = [str(x) for x in x_cost['minutes_per_100k']]
    costs = [[label(r), f"{r['tokens_written']:,}", f"{r['lm_calls']:,}", f"{r['peak_context']:,}",
              str(r.get('recalls', '–'))] for r in rounds['round2']['rows'] if r['cap'] == '32K']
    out = {
        'DP_TABLE_ROUND2': table(head + tail, rows(r2), 'Questions right by distance back and by type, 16K and 32K windows', 'ctx-comparison'),
        'DP_TABLE_ROUND1': table(head + tail, rows(rounds['round1']['rows']), 'First round: 16K window, seed 0, one GPU', 'ctx-comparison'),
        'DP_TABLE_48K': table(head + ['100–399 back'] + tail, rows(rounds['cap48-20261008']['rows'], True),
                              'Earlier run: 48K window over a 480K history', 'ctx-comparison'),
        'DP_COST_TABLE': table(['Method at 32K', 'Tokens written', 'Model calls', 'Peak live context', 'Archive searches'], costs,
                               'Cost of one 32K run', 'ctx-comparison'),
        'DP_TABLE_SEEDS': table(['Method at 32K', 'Seed', 'Probes right', 'Final questions', 'All', 'Minutes'], seed_rows,
                                'Each seed on its own at a 32K window', 'ctx-comparison'),
        'DP_TABLE_RULES': table(['Method at 32K', 'Values, old values, who', 'Standing rule', 'House method', 'Final questions', 'All 39',
                                 'Minutes'], rule_rows, 'A rule and a method stated once at the start: questions right', 'ctx-comparison'),
        'DP_TABLE_1M': table(['What was counted', label(km) + ': finished and graded', label(xm) + ': finished and graded',
                              label(m) + ': stopped run'], million_rows,
                             'One million tokens through a 32K window: the keeper, the keeper as a proxy and the earlier stopped run'),
        'DP_TABLE_KEEPER': table(['Method at 32K', 'First 240K stream, ' + str(first_asked) + ' questions',
                                  'Second 240K stream, ' + str(second_asked) + ' questions',
                                  '1,000,000-token stream, ' + str(k_total[1]) + ' questions'], keeper_rows,
                                 'The keeper, the keeper as a proxy and two earlier methods: questions right and minutes per run'),
        'DP_TABLE_SUMMARY': table(['Method at 32K', 'All 29', 'Probes right', 'Current value', 'Old value', 'Who did it', 'Final questions',
                                   'Minutes'], summary_rows, 'Summarising at a threshold against the other methods, first seed',
                                  'ctx-comparison ctx-fit'),
        'DP_K1M_SEGMENTS': escape(', '.join(segments[:-1]) + ' and ' + segments[-1]),
        'DP_K1M_GRADE': escape(km['evidence']['grade'], quote=True),
        'DP_K1M_DETAILS': escape(km['evidence']['details'], quote=True),
        'DP_K1M_NOTES': escape(km['evidence']['final_notes'], quote=True),
        'DP_X1M_SEGMENTS': escape(', '.join(x_segments[:-1]) + ' and ' + x_segments[-1]),
        'DP_X1M_GRADE': escape(xm['evidence']['grade'], quote=True),
        'DP_X1M_DETAILS': escape(xm['evidence']['details'], quote=True),
        'DP_X1M_COUNTERS': escape(xm['evidence']['proxy_counters'], quote=True),
        'DP_PROXY_README': escape(keeper['proxy']['readme_url'], quote=True),
        'DP_PROXY_CODE': escape(keeper['proxy']['code_url'], quote=True),
        'DP_KEEPER_DATA_URL': escape(dp['data_urls']['keeper_proxy_and_summary_baseline'], quote=True),
        'DP_1M_LIMIT': escape(m['limit']), 'DP_1M_UNANSWERED': escape(m['unanswered']),
        'DP_1M_MINUTES': str(cost['minutes']), 'DP_1M_PER_100K': str(cost['minutes_per_100k_mean']),
        'DP_1M_FIRST_200K': str(cost['first_200k_minutes']), 'DP_1M_LAST_200K': str(cost['last_200k_minutes']),
        'DP_1M_CALLS': f"{cost['lm_calls']:,}", 'DP_1M_WRITTEN': escape(cost['tokens_written']),
        'DP_1M_PROMPT': escape(cost['prompt_tokens']), 'DP_1M_CACHED': escape(cost['prompt_from_prefix_cache']),
        'DP_1M_SEARCHES': str(cost['archive_searches']), 'DP_1M_PEAK': escape(m['peak_live_context']),
        'DP_1M_GRADE': escape(m['evidence']['rebuilt_grade'], quote=True),
        'DP_1M_ANSWERS': escape(m['evidence']['rebuilt_answers'], quote=True),
        'DP_1M_AT_STOP': escape(m['evidence']['answers_file_at_stop'], quote=True),
        'DP_RULES_ANALYSIS': escape(rules['analysis_url'], quote=True),
        'DP_SEEDS_ANALYSIS': escape(rounds['cap32-two-seeds']['analysis_url'], quote=True),
        'DP_DATA2_URL': escape(dp['data_urls']['second_seed_and_rule_questions'], quote=True),
        'DP_DATA3_URL': escape(dp['data_urls']['rule_fix_and_million_token_run'], quote=True),
        'DP_SEEDS_COST': escape(rounds['cap32-two-seeds']['cost_per_run']),
        'DP_NOTE_URL': escape(dp['note_url'], quote=True), 'DP_DATA_URL': escape(dp['data_url'], quote=True),
        'DP_CHARTER_URL': escape(dp['charter_url'], quote=True),
        'DP_R1_ANALYSIS': escape(rounds['round1']['analysis_url'], quote=True),
        'DP_R2_ANALYSIS': escape(rounds['round2']['analysis_url'], quote=True),
        'DP_48K_NOTE': escape(rounds['cap48-20261008']['note']),
        'DP_VOID_NOTE': escape(next(r['note'] for r in rounds['round2']['rows'] if r['id'] == 'B16ira')),
    }
    return out


def landing_table(by_id, lme, dp):
    """One row per question the lab has asked so far, newest last; every number comes from the record."""
    lme_rows = {r['id']: r for r in lme['rows']}
    r2 = {r['arm']: r for r in next(x for x in dp['rounds'] if x['id'] == 'cap32-two-seeds')['rows']}
    rules = {r['arm']: r for r in dp['rules']['rows']}
    million = dp['million']
    keeper = dp['keeper']
    k0, x0 = keeper['rows']
    k1 = {r['method']: r for r in keeper['second_seed']}
    km = keeper['million']
    xm = keeper['proxy']['million']
    seed0 = {r['id']: r for r in next(x for x in dp['rounds'] if x['id'] == 'round2')['rows'] if r['cap'] == '32K'}
    c32 = dp['summary_baseline']['row']
    no_place = [fraction_parts(r['rule']) for r in dp['rules']['rows'] if r['method'] not in ('archive_rules', 'archive_free')]
    no_place = tuple(map(sum, zip(*no_place)))
    light = by_id['improved-light-121k-s0'], by_id['improved-light-121k-s1']
    rows = [
        ['A 121K ledger of counter updates', 'Files and code: ' + score(by_id['files-plain']) + ' in ' + by_id['files-plain']['time'],
         'Summaries: ' + score(by_id['summary']) + ' in ' + by_id['summary']['time'], 'context-files-code.html'],
        ['The same ledger, no files allowed', 'Self-editing, thinking only when needed: ' + score(light[0]) + ' and ' + score(light[1])
         + ' in ' + light[0]['time'] + ' and ' + light[1]['time'],
         'Original CLM agent: ' + score(by_id['paper']) + ' in ' + by_id['paper']['time'], 'context-revised-editing.html'],
        ['A million-token story read through a 32K budget', 'Quoted events: ' + score(by_id['quoted-1m']) + ' in ' + by_id['quoted-1m']['time'],
         'Read mode: ' + score(by_id['read-improved-1m']) + ' in ' + by_id['read-improved-1m']['time'], 'context-results.html#reading'],
        ['Questions about text that was dropped (119K)', 'Archive and search: ' + score(by_id['ret-archive-119k']) + ' in ' + by_id['ret-archive-119k']['time'],
         'Summaries: ' + score(by_id['ret-summary-119k']) + ' in ' + by_id['ret-summary-119k']['time'], 'context-retention.html'],
        ['A public chat-memory benchmark (LongMemEval, 56 questions)',
         'Whole history in one call: ' + f"{lme_rows['lme-whole']['right']}/{lme_rows['lme-whole']['asked']}"
         + '; archive and search at 32K: ' + f"{lme_rows['lme-archive']['right']}/{lme_rows['lme-archive']['asked']}",
         'Summaries: stopped at ' + f"{lme_rows['lme-summary']['right']}/{lme_rows['lme-summary']['asked']}", 'context-retention.html#longmemeval'],
        ['A 240K history behind a 32K window, questions planted at known distances (two seeds)',
         'Archive and search: ' + r2['B32ira']['total'] + ' right, ' + r2['B32ira']['minutes'].replace('-', '–') + ' min per run',
         'Window-drop: ' + r2['W32']['total'] + ', and ' + r2['W32']['d25_99'] + ' beyond 25 batches back; compaction: ' + r2['P32']['total'],
         'context-distance.html#results'],
        ['Summarising at a threshold, on the first of those two histories (one seed)',
         'Archive and search: ' + seed0['B32ira']['total'] + ' in ' + seed0['B32ira']['minutes'] + ' min',
         'Summarising at 75% of the window: ' + c32['total'] + ' in ' + c32['minutes'] + ' min; window-drop: '
         + seed0['W32']['total'] + ' in ' + seed0['W32']['minutes'] + ' min', 'context-distance.html#summary'],
        ['A rule and a method stated once at the start of a 240K history (two streams)',
         'A keeper that asks nothing of the agent: ' + k0['total'] + ' and ' + k1['keeper']['total'] + '; run as a proxy: '
         + x0['total'] + ' and ' + k1['keeper_proxy']['total'],
         'On the first stream, archive and search with pinned rule lines: ' + rules['B32iraR']['total'] + '; window-drop: '
         + rules['W32']['total'] + '. Four methods with no pinned place for a rule: ' + f'{no_place[0]}/{no_place[1]}' + ' rule questions',
         'context-distance.html#keeper'],
        ['A 1,000,000-token history through a 32K window (one stream, one seed)',
         'The keeper: ' + km['total'] + ' right in ' + str(km['cost']['minutes']) + ' min; run as a proxy: ' + xm['total'] + ' in '
         + str(xm['cost']['minutes']) + ' min. Both graded by the task’s own checker',
         'Archive and search with pinned rule lines: ' + million['total'] + ' in a stopped run that was not verified',
         'context-distance.html#million'],
    ]
    html = table(['What was tested', 'Best so far', 'Compared with'], [r[:3] for r in rows], 'What each long-context test found', 'ctx-summary-table')
    for r in rows:
        html = html.replace('<th scope="row">' + escape(r[0]) + '</th>', '<th scope="row">' + link('@/' + r[3], r[0]) + '</th>', 1)
    return html


def update(pages):
    d = json.loads((ROOT / DATA).read_text(encoding='utf-8'))
    d, trial_manifest = load_published(d)
    base = d['lab']
    note = lambda key: d.get('source_urls', {}).get(key, base + d['sources'][key])
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
    overview = table(['Approach', 'Correct', 'Time'], [[overview_labels[key][0], score(by_id[key]), by_id[key]['time']]
        for key in overview_order], 'Long-task approaches and results', 'ctx-summary-table')
    for label, path in overview_labels.values():
        overview = overview.replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + path, label) + '</th>')
    other = table(['Approach', 'Result'], [
        ['Prefix caching', 'Repeated 30K prompt: first token in 0.8 s instead of 11.4 s. Standard-check writing speed stayed near 89 tokens/s.'],
        ['Bigger active context', 'Observed writing speed: about 127 tokens/s at 8K, versus ' + str(round(decode[-1]['tokens_per_second'])) + ' at 250K, in separate prompt tests.'],
        ['CPU text cleanup', '1.83% fewer tokens with conservative cleanup.'],
        ['Active cache on disk', 'Not benchmarked. The 200K-token attention cache alone needs about 12.2 GiB.'],
        ['Reuse after an edit', 'The CLM paper reuses cached work after an edit, approximately. Here the exact cache re-reads from just before the edit.']],
        'Other context approaches and their findings', 'ctx-summary-table ctx-other-table')
    for label, path in [('Prefix caching', 'context-prefix-cache.html'), ('Bigger active context', 'context-large-window.html'),
                        ('CPU text cleanup', 'context-cleanup.html'), ('Active cache on disk', 'context-cache-offload.html'), ('Reuse after an edit', 'context-reuse.html')]:
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
                       [[r['strategy'], r['files'], r['budget'], score(r), r['time'], r['written']] for r in measured
                        if r['task_family'] == 'ledger' and r['input_tokens'] == 121000],
                       '121K-token ledger trials, including both large-window and revised-agent seeds', 'ctx-comparison')
    dp_keeper = d['distance_probes']['keeper']
    assert dp_keeper['second_seed'][0]['method'] == 'keeper'
    dp_c32 = d['distance_probes']['summary_baseline']['row']
    dp_w32 = next(r for x in d['distance_probes']['rounds'] if x['id'] == 'round2' for r in x['rows'] if r['id'] == 'W32')
    method_rows = [
        ['Bigger active window', 'Holds more text at once. The window opened to 262K positions; longer active context slowed generation.'],
        ['Exact prefix cache', 'Reuses the unchanged start. Repeat questions began sooner; writing stayed near 89 tokens/s in the standard check.'],
        ['Suffix cache reuse (CLM paper)', 'Also reuses cached text after an edit, approximately. Not used here; answers would depend on edit history.'],
        ['Files, no context editing', 'Code reads saved records and keeps the totals. 24/24 in 1.9 minutes on the 121K ledger.'],
        ['Files, editing available', 'Adds permission to rewrite the conversation. 24/24 in 1.9 minutes on the 121K ledger; no edit was needed.'],
        ['CLM self-editing', 'Rewrites selected parts of the working history. Original agent: 19/24 in 64 minutes; its harness lost five delivered batches.'],
        ['Periodic summaries', 'Replaces old history with a short account. It got 24/24 in 41 minutes on the 121K ledger; making summaries added work. '
         'Over a 240K history at a 32K window it was the slowest method and scored below simply dropping old turns ('
         + dp_c32['total'] + ' against ' + dp_w32['total'] + ', one seed).'],
        ['Revised self-editing', 'Protects incoming batches and pins the current state. 24/24 and 21/24 on two 121K runs; 24/24 on a 478K stream.'],
        ['Remove old thinking', 'Drops earlier reasoning from later calls. One ledger run repeatedly rebuilt its state and returned no answer after 2.6 hours.'],
        ['Park a cache on disk', 'Saves an inactive session for later restoration. Save/restore takes time; not benchmarked here.'],
        ['Stream active cache from disk/RAM', 'Trades repeated transfers or CPU work for capacity. Requires engine support; not benchmarked here.'],
        ['CPU text cleanup', 'Removes repetition and formatting noise. Conservative cleanup reduced tokens by 1.83%.'],
        ['Archive and search', 'Moves dropped text word for word to a searchable archive. 36/36 on the retention test in 6.1 minutes; the best score on both seeds over a 240K history at a 32K window (63/65 pooled). It needs a pinned place for rules to keep one.'],
        ['Compaction (stub old tool output)', 'Replaces old tool output with stubs. Cheap, but it lost “who did it” details and, on one of two seeds at 32K, reported six old values as current.'],
        ['Keeper (managed from outside the agent)', 'The agent manages nothing. Something outside it archives the oldest turns word for word, has the model rewrite a short notes block at each cut, '
         'and repeats the rules where questions arrive. ' + dp_keeper['rows'][0]['total'] + ' and ' + dp_keeper['second_seed'][0]['total']
         + ' on two 240K streams at a 32K window; ' + dp_keeper['million']['total'] + ' over 1,000,000 tokens (one stream).']]
    methods = table(['Approach', 'What changes, and the result'], method_rows, 'Context approaches compared', 'ctx-summary-table ctx-other-table')
    for label, path in [('Bigger active window', 'context-large-window.html'), ('Exact prefix cache', 'context-prefix-cache.html'), ('Suffix cache reuse (CLM paper)', 'context-reuse.html'),
                        ('Files, no context editing', 'context-files-code.html'), ('Files, editing available', 'context-files-editing.html'), ('CLM self-editing', 'context-self-editing.html'),
                        ('Periodic summaries', 'context-summaries.html'), ('Revised self-editing', 'context-revised-editing.html'), ('Remove old thinking', 'context-drop-thinking.html'),
                        ('Park a cache on disk', 'context-cache-parking.html'), ('Stream active cache from disk/RAM', 'context-cache-offload.html'), ('CPU text cleanup', 'context-cleanup.html'),
                        ('Archive and search', 'context-retention.html'), ('Compaction (stub old tool output)', 'context-distance.html'),
                        ('Keeper (managed from outside the agent)', 'context-distance.html#keeper')]:
        methods = methods.replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + path, label) + '</th>')
    chart = chart_image('context-decode-speed', 'Observed output rate: 127 tokens per second at 8K context, 99 at 30K, 70 at 60K, 49 at 120K, 40 at 160K, 35 at 200K, 31 at 230K and 27 at 250K.',
                        'Separate prompt tests: one reply each at 8K and 30K; medians of 20 replies at the longer lengths.', 936)
    source_list = '<ul class="ctx-source-list">'
    for key, title, description in [
        ('results', 'Research summary', 'Completed findings and pending work in the recorded research snapshot.'),
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
            [[overview_labels[key][0], score(by_id[key]), by_id[key]['time']] for key in ['paper', 'improved-121k-s0', 'improved-121k-s1']],
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
        'REVIEW_EVIDENCE': d['review_evidence_url'],
        'RAW_RECALL_PROSE': raw('recall/recall-prose.json'), 'RAW_RECALL_LEDGER': raw('recall/recall-ledger.json'),
        'RAW_CACHE': raw('prefixcache-exact-mtp/cache.json'), 'RAW_GATE': raw('pcgate/results.json'),
        'CONTEXT_README': note('scripts') + 'README.md',
        'PROMPT_LEDGER': note('scripts') + 'make_ledger_tasks.py#L39-L62',
        'PROMPT_STORAGE': note('scripts') + 'make_kvstream_tasks.py#L215-L229',
        'PROMPT_BASELINE': note('scripts') + 'clm_baselines.py#L54-L85',
        'PROMPT_CLM': 'https://github.com/facebookresearch/context-language-models/blob/18dc11115f50f261233c5bba7937834491e307e8/clm/clm_harness/clm_agent/prompts.yaml#L5-L51',
        'PROMPT_REVISED': 'https://github.com/steveseguin/b70-optimization-lab/blob/73a6693b275d70432086c5ea91fa36e4cec0c6c2/experiments/qwen38-27b-b70/scripts/context/clm_improved.py#L107-L128',
    }
    replacements.update(diagnostic_replacements())
    for key in ['results', 'window', 'cache_test', 'cache_rules', 'self_editing', 'cleaning', 'review', 'timing']:
        replacements['NOTE_' + key.upper()] = note(key)
    minutes = lambda seconds: f'{seconds / 60:.1f} min'
    reuse_rows = [[r[0], minutes(r[1]), minutes(r[2]), minutes(r[3]), f'{100 * r[3] / r[1]:.0f}%'] for r in d['timing_analysis']['rows']]
    for seed in (0, 1):
        light = by_id[f'improved-light-121k-s{seed}']
        reuse_rows.append([f'Revised self-editing, thinking only when needed, seed {seed}', light['time'], 'Not split', 'Not split', 'Not split'])
    replacements['REUSE_TIME_TABLE'] = table(['Run (121K ledger)', 'Total', 'Writing and calls', 'Reading', 'Reading share'], reuse_rows,
                                             'Where the time went in the agent runs', 'ctx-comparison')
    edit = d['edit_cost']
    cells = {(x['context'], x['position']): x for x in edit['cells']}
    def edit_cell(x):
        if x.get('status') != 'measured':
            return 'Being measured'
        return f"{x['first_edit_s']:g} s first, {x['repeat_s']:g} s repeated (cold {x['cold_s']:g} s)"
    replacements['EDIT_COST_TABLE'] = table(['Context', *(('Edit at ' if p.endswith('%') else 'Edit in the ') + p for p in edit['positions'])],
                                            [[c, *(edit_cell(cells.get((c, p), {})) for p in edit['positions'])] for c in edit['contexts']],
                                            'Wait before the first output token after one edit, by context length and edit position', 'ctx-comparison')

    retention = [('ret-archive-119k', 'Table plus archive and recall'), ('ret-table-119k', 'Table only'),
                 ('ret-summary-119k', 'Summarise at 75%'), ('ret-keep-119k', 'Keep everything in the window'),
                 ('ret-files-119k', 'No management, files allowed')]
    replacements['RETENTION_TABLE'] = table(['Approach', 'Files', 'Right', 'Time', 'Peak context', 'Tokens written'],
                                            [[label, by_id[key]['files'], score(by_id[key]), by_id[key]['time'], by_id[key]['peak'], by_id[key]['written']]
                                             for key, label in retention], 'Retention test: 36 questions, 12 of them about dropped text', 'ctx-comparison')
    for key, _ in retention:
        for field in ('right', 'time', 'written'):
            replacements['RET_' + key.split('-')[1].upper() + '_' + field.upper()] = escape(by_id[key][field])
    replacements['RETENTION_DATA'] = RETENTION_DATA
    lme = d['longmemeval']
    def lme_row(r):
        return [r['approach'], r['budget'], f"{r['right']} of {r['asked']}" + (' (stopped)' if r.get('stopped') else ''),
                f"{r['median_time_s'] / 60:.1f} min", f"{r['median_tokens_written']:,}"]
    replacements['LONGMEMEVAL_TABLE'] = table(['Approach', 'Working budget', 'Right', 'Median time per question', 'Median tokens written'],
                                              [lme_row(r) for r in lme['rows']],
                                              'LongMemEval, 56 questions with about 110K tokens of chat history each', 'ctx-comparison')
    m = lme['matched']
    replacements['LONGMEMEVAL_MATCHED_TABLE'] = table(['Approach', f"Right of {m['asked']}", 'Median time', 'Median tokens written'],
                                                      [[r['approach'], str(r['right']),
                                                        f"{r['median_time_s'] / 60:.1f} min" if 'median_time_s' in r else 'See above',
                                                        f"{r['median_tokens_written']:,}" if 'median_tokens_written' in r else 'See above'] for r in m['rows']],
                                                      f"The same {m['asked']} questions for all three approaches", 'ctx-comparison')
    for k in ('benchmark', 'judge', 'server', 'date', 'note_url', 'data_url'):
        replacements['LME_' + k.upper()] = escape(lme[k])
    for k in ('only_archive', 'only_summary', 'both_wrong', 'asked'):
        replacements['LME_' + k.upper()] = str(m[k])
    replacements['LME_STRATA'] = escape(m['strata'])
    for k, v in lme['where_archive_loses'].items():
        replacements['LME_LOSES_' + k.upper()] = escape(v)
    replacements['LME_STOPPED'] = escape(lme['rows'][2]['stopped'])
    replacements['LME_INTERRUPTED'] = escape(lme['interrupted'])
    replacements.update(distance_replacements(d['distance_probes']))
    replacements['LANDING_TABLE'] = landing_table(by_id, lme, d['distance_probes'])
    replacements['READING_TABLE'] = table(
        ['Trial', 'Files', 'Correct', 'Time', 'Tokens generated'],
        [[r['strategy'], r['files'], score(r), r['time'], r['written']]
         for r in measured if r['task_family'] == 'reading'],
        'Narrative reading trials, with each seed and question count shown separately', 'ctx-comparison')
    replacements['QUOTED_RETENTION_TABLE'] = table(
        ['Trial', 'Correct', 'Time', 'Tokens generated', 'Peak context'],
        [[r['strategy'], score(r), r['time'], r['written'], r['peak']]
         for r in measured if r['id'].startswith('quoted-ret-')],
        'Quoted-events retention trials, one row per seed', 'ctx-comparison')
    read_stream, quoted_stream = by_id['read-improved-480k'], by_id['quoted-480k']
    assert read_stream['task_fingerprint'] == quoted_stream['task_fingerprint'], 'Matched reading tasks differ'
    saved = 100 * (1 - quoted_stream['elapsed_seconds'] / read_stream['elapsed_seconds'])
    replacements['MATCHED_READING_SUMMARY'] = (
        '<p><strong>On the matched 480K narrative stream, read mode scored ' + score(read_stream)
        + ' and quoted events scored ' + score(quoted_stream) + '.</strong> Quoted events took '
        + quoted_stream['time'] + ' versus ' + read_stream['time'] + ' for read mode, about '
        + f'{saved:.0f}%' + ' less elapsed time in this one-seed comparison. Both agents used a 32K working budget. '
        + 'These measurements used prefix caching in both methods and different agent protocols. '
        + 'They describe one paired task; a cold-request or replicated speed comparison has not been established.</p>')
    read_million, quoted_million = by_id['read-improved-1m'], by_id['quoted-1m']
    assert read_million['task_fingerprint'] == quoted_million['task_fingerprint'], 'Million-token tasks differ'
    replacements['MATCHED_MILLION_SUMMARY'] = (
        '<p><strong>On the matched million-token narrative stream, quoted events scored ' + score(quoted_million)
        + ' in ' + quoted_million['time'] + '; read mode scored ' + score(read_million)
        + ' in ' + read_million['time'] + '.</strong> This is one completed pair on the same seed and task, '
        + 'with a 32K working budget. It does not establish reliability across other tasks or seeds.</p>')
    checked = [(label, r['reverified']) for label, r in (('480K stream', quoted_stream), ('million-token stream', quoted_million))
               if r.get('reverified')]
    if checked:
        replacements['MATCHED_MILLION_SUMMARY'] += (
            '<p><strong>Re-checked on ' + escape(checked[0][1]['date']) + ' with the fixed scorer:</strong> '
            + '; '.join(escape(label) + f' {v["correct"]}/{v["asked"]} in {v["elapsed_seconds"] / 60:.1f} min'
                        for label, v in checked)
            + '. A review had found three holes in the quoted-events checker; after the fix, both runs were repeated on the same task '
            + 'and gave the same scores. ' + escape(checked[0][1]['note']) + ' '
            + link('@/docs/context-reverify.md', 'What was repeated') + ' · ' + link(checked[0][1]['source'], 'Lab note') + '</p>')
    def attempt_rows(rows):
        return [[r['id'].split('__')[-1], r['task_family'].replace('_', ' ') + ', '
                 + (f'{r["input_tokens"] / 1000:.0f}K' if r.get('input_tokens') else 'size unknown')
                 + (f', density {r["density"]:g}' if r.get('density') else ''),
                 r['arm'], r['seed'],
                 (('Void; raw ' if r['void'] else '') + f'{r["correct"]}/{r["asked"]}')
                 if r['status'] == 'completed' else r['status'].replace('_', ' '),
                 minutes(r['elapsed_seconds']) if r.get('elapsed_seconds') is not None else 'Not recorded'] for r in rows]
    completed = [r for r in trial_manifest['rows'] if r['status'] == 'completed']
    unfinished = [r for r in trial_manifest['rows'] if r['status'] != 'completed']
    replacements['TRIAL_HISTORY'] = table(['Trial', 'Task', 'Arm', 'Seed', 'Correct', 'Time'],
        attempt_rows(completed), 'Completed trial history, including failed and void attempts', 'ctx-comparison')
    if unfinished:
        replacements['TRIAL_HISTORY'] += '<h4>Unfinished attempts</h4>' + table(
            ['Trial', 'Task', 'Arm', 'Seed', 'Status', 'Time'], attempt_rows(unfinished),
            'Unfinished attempts, excluded from measured results', 'ctx-comparison')
    replacements['LOCAL_TRIAL_MANIFEST'] = '@/' + d['canonical_trials']['manifest']
    trial_config = read_json(CONFIG)
    replacements['TRIAL_SOURCE_LINK'] = ''
    if trial_config.get('source_revision'):
        source_url = (trial_config['source_repository'] + '/blob/' + trial_config['source_revision']
                      + '/' + trial_config['source_path'].strip('/') + '/manifest.json')
        replacements['TRIAL_SOURCE_LINK'] = ' · ' + link(source_url, 'Pinned lab export')

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
        ('model-tools', 'The GPU model requests an apple total. The application runs CPU code that reads files on disk. Only the result, 11 apples, returns to the next model prompt; the archive stays outside it.'),
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
        ('context-model-tools.html', 'How the model uses files', 'Follow a tool request from the GPU model to disk and back, with the actual prompt sources.', 'context-methods.html'),
        ('context-clm-vs-files.html', 'CLM or files: what changes?', 'Compare live context editing, an automatically loaded state file and a separate archive.', 'context-methods.html'),
        ('context-model-choice.html', 'Is the 27B model enough?', 'What Qwen3.8-27B solved with tools, and how a stronger or trained model might help.', 'context-methods.html'),
        ('context-results.html', 'Longer context: the measured results', 'Task outcomes, speed, recall and the source records behind the explanations.', None),
        ('context-large-window.html', 'Use a larger context window', 'Keep more conversation in view: how it works, its results and where it runs out.', 'context-methods.html'),
        ('context-summaries.html', 'Replace old history with a summary', 'How periodic summaries make room, what they preserve and what they can lose.', 'context-methods.html'),
        ('context-self-editing.html', 'Let the model edit its conversation', 'What CLM self-editing actually changes, with a simple example and measured results.', 'context-methods.html'),
        ('context-revised-editing.html', 'Self-editing with a protected state', 'How delivery checks and an explicit state change the self-editing approach.', 'context-methods.html'),
        ('context-files-code.html', 'Keep records in files and use code', 'How a plain file-using agent completed long ledgers with a small active context.', 'context-methods.html'),
        ('context-files-editing.html', 'Files with context editing available', 'Two tools with different jobs: files store records, while editing can shorten the live conversation.', 'context-methods.html'),
        ('context-drop-thinking.html', 'Remove earlier thinking', 'What gets removed, when it saves space and why one long-task run stalled.', 'context-methods.html'),
        ('context-prefix-cache.html', 'Reuse the reading with a prefix cache', 'How cached computation helps a repeated prompt start answering sooner.', 'context-methods.html'),
        ('context-reuse.html', 'Changing the context without re-reading it', 'How the CLM paper’s suffix cache reuse works, why it is approximate, and how it compares with this lab’s exact cache.', 'context-methods.html'),
        ('context-cleanup.html', 'Clean up text before the model reads it', 'What simple CPU cleanup removes and why its measured savings were small.', 'context-methods.html'),
        ('context-cache-parking.html', 'Park an inactive cache on disk', 'Save computed state between sessions, then restore it before generating.', 'context-methods.html'),
        ('context-cache-offload.html', 'Offload an active cache to RAM or disk', 'How moving numerical state can trade speed for capacity, and why it differs from saving text.', 'context-methods.html'),
        ('context-retention.html', 'Remembering what was dropped', 'An archive of dropped text with a search tool answered questions about the past as well as summarising, in a quarter of the time.', 'context-methods.html'),
        ('context-distance.html', 'Past the window: what the agent still remembers', 'A history of 240K to 1,000,000 tokens behind a 16K or 32K live window, with questions planted at known distances back: which ways of managing the window still answer them, which still follow a rule, and a keeper that does the managing from outside the agent.', 'context-methods.html'),
        ('context-reproduce.html', 'How to reproduce the context tests', 'The tasks, graders, agent harness, comparison driver and probes, with what others can and cannot rerun.', None),
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
        if '<figure class="ctx-diagram">' in body or name in ('context-clm-vs-files.html', 'context-model-choice.html'):
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
