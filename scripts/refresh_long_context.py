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
LAB_COMMIT_13D = '283a90e374818ec3c0c84495ee9a8a6004856a29'
RETENTION_DATA = ('https://github.com/steveseguin/b70-optimization-lab/tree/974df5792372b424d66d8225af171c4a94307e74/'
                  'experiments/qwen38-27b-b70/data/2026-10-05-context/retention')


def link(href, label):
    return '<a href="' + escape(href, quote=True) + '">' + escape(label) + '</a>'


def table(headers, rows, label, extra=''):
    return ('<div class="table-scroll ctx-table ' + extra + '" tabindex="0" role="region" aria-label="' + escape(label, quote=True) + '"><table><thead><tr>'
            + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(('<th scope="row">' + escape(str(v)) + '</th>') if i == 0 else '<td>' + escape(str(v)) + '</td>' for i, v in enumerate(row)) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def html_table(headers, rows, label, extra=''):
    """A table whose cells are already HTML, for rows that carry evidence links."""
    return ('<div class="table-scroll ctx-table ' + extra + '" tabindex="0" role="region" aria-label="' + escape(label, quote=True) + '"><table><thead><tr>'
            + ''.join('<th scope="col">' + escape(h) + '</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(('<th scope="row">' + v + '</th>') if i == 0 else '<td>' + v + '</td>' for i, v in enumerate(row)) + '</tr>' for row in rows)
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
              'summary': 'Summarise at 75% of the window', 'keeper_inplace': 'Keeper with in-place notes',
              'retrieval': 'Retrieval only', 'keeper_inplace_retrieval': 'Keeper with in-place notes and retrieval',
              'keeper_proxy_inplace': 'Keeper as a proxy, with in-place notes'}
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
    asked_by_seed = {0: first_asked, 1: second_asked}

    # In-place notes: the same keeper with cheaper notes. Every run must add up and be compared on the same questions.
    ip = keeper['in_place']
    assert ip['method'] == 'keeper_inplace' and len(ip['checks']) == 3 and ip['not_run'] and ip['limits']
    for r in ip['rows']:
        assert r['method'] == 'keeper_inplace' and rule_stream_parts(r, asked_by_seed[r['seed']])
    ip_lead = {r['seed']: r for r in ip['rows'] if r['headline']}
    ip_later = [r for r in ip['rows'] if not r['headline']]
    assert set(ip_lead) == {0, 1} and len(ip_later) == 5
    fr = ip['full_rewrite']
    fr_rows = {r['seed']: r for r in fr['rows']}
    assert fr['arm'] == first['keeper']['arm'] and fr_rows[0]['total'] == first['keeper']['total'] and fr_rows[1]['total'] == second['keeper']['total']
    assert all(int(float(fr_rows[seed]['minutes'])) == int(r['minutes']) for seed, r in ((0, first['keeper']), (1, second['keeper'])))
    assert fr['million']['total'] == km['total'] and round(fr['million']['minutes']) == k_cost['minutes'] and fr['million']['lm_calls'] == k_cost['lm_calls']
    im = ip['million']
    assert im['status'].startswith('Finished run') and im['method'] == 'keeper_inplace' and im['limit']
    i_probes, i_final, i_total = (fraction_parts(im[k]) for k in ('probes', 'final', 'total'))
    for split in ('by_distance', 'by_type', 'by_third'):
        assert list(im[split]) == list(km[split]), split + ': the in-place run is split differently'
        assert tuple(map(sum, zip(*map(fraction_parts, im[split].values())))) == i_probes, 'in-place ' + split + ' does not add up'
        assert all(fraction_parts(im[split][k])[1] == fraction_parts(km[split][k])[1] for k in im[split]), split + ': different questions'
    assert (i_probes[0] + i_final[0], i_probes[1] + i_final[1]) == i_total and i_total[1] == k_total[1]
    assert round(100 * i_total[0] / i_total[1]) == im['total_percent'] and round(100 * i_probes[0] / i_probes[1]) == im['probes_percent']
    i_cost, i_notes = im['cost'], im['notes']
    assert i_cost['lm_calls'] == i_cost['agent_calls'] + i_cost['notes_calls'] and len(i_cost['minutes_per_100k']) == 9
    assert round(i_cost['minutes_one_decimal']) == i_cost['minutes'] and i_notes['cuts_in_place'] <= i_notes['cuts'] == i_notes['versions_audited']
    assert all(dp['lab_commit'] in url for url in (*im['evidence'].values(), *ip['evidence'].values()))
    # The headline rows are the version with every check, the one that also ran at 1M; its three runs must add up.
    ec = ip['every_check']
    assert all(r['arm'] == im['arm'] == ec['arm'] for r in ip_lead.values()) and ec['runs'] == 3
    assert ec['notes_wrong'] == [ip_lead[0]['notes_wrong'], ip_lead[1]['notes_wrong'], im['notes_wrong']] and ec['the_one_wrong_counter']
    checked_runs = [fraction_parts(x) for x in (ip_lead[0]['total'], ip_lead[1]['total'], im['total'])]
    assert tuple(map(sum, zip(*checked_runs))) == fraction_parts(ec['total'])
    rewrite_runs = [fraction_parts(x) for x in (fr_rows[0]['total'], fr_rows[1]['total'], km['total'])]
    assert tuple(map(sum, zip(*rewrite_runs))) == fraction_parts(ec['full_rewrite_total'])
    assert ec['minutes'] == [ip_lead[0]['minutes'], ip_lead[1]['minutes'], str(i_cost['minutes_one_decimal'])]
    assert ec['full_rewrite_minutes'] == [fr_rows[0]['minutes'], fr_rows[1]['minutes'], str(fr['million']['minutes'])]
    # The notes audit: one row per audited run; the in-place rows above must carry the same counts.
    audit = ip['audit']
    wrong = {(r['arm'], r['stream']): r['notes_wrong'] for r in audit['rows']}
    assert len(wrong) == len(audit['rows']) and all(dp['lab_commit'] in r['audit_url'] and isinstance(r['notes_wrong'], int)
                                                    and fraction_parts(r['total']) for r in audit['rows'])
    no_thinking = [r['notes_wrong'] for r in audit['rows'] if 'no thinking' in r['notes']]
    assert len(no_thinking) == 6 and (min(no_thinking), max(no_thinking)) == (2, 8)
    assert all(r['notes_wrong'] == wrong[(r['arm'], ('First' if r['seed'] == 0 else 'Second') + ' 240K')] for r in ip['rows'])
    assert all(fr_rows[seed]['notes_wrong'] == wrong[(fr['arm'], name + ' 240K')] == 0 for seed, name in ((0, 'First'), (1, 'Second')))
    assert wrong[(im['arm'], '1,000,000')] == im['notes_wrong'] == 0 and wrong[(fr['arm'], '1,000,000')] == fr['million']['notes_wrong']
    six = ip['six_items_per_notes_call']
    six_audit = next(r for r in audit['rows'] if r['arm'] == six['arm'])
    assert (six_audit['total'], six_audit['minutes'], six_audit['notes_wrong']) == (six['total'], six['minutes'], six['notes_wrong'])
    assert six['total'] == ip_lead[six['seed']]['total'] and six['reading'] and six['defect_found']

    # Retrieval alone on the same streams. Its 1M run is a stopped run: only the questions asked along the way are compared.
    ro = dp['retrieval_only']
    assert ro['method'] == 'retrieval' and ro['limits'] and ro['correction'] and len(ro['lessons_from_the_builds']) == 2
    for r in ro['rows']:
        assert r['method'] == 'retrieval' and r['build'] in ro['builds'] and rule_stream_parts(r, asked_by_seed[r['seed']])
    ro_by_build = {build: {r['seed']: r for r in ro['rows'] if r['build'] == build} for build in ro['builds']}
    assert set(ro_by_build['final']) == set(ro_by_build['second']) == {0, 1} and set(ro_by_build['first']) == {0}
    rm = ro['million']
    assert rm['status'].startswith('Stopped run') and rm['method'] == 'retrieval' and rm['limit']
    r_probes = fraction_parts(rm['probes'])
    assert r_probes[1] == i_probes[1] and round(100 * r_probes[0] / r_probes[1]) == rm['probes_percent']
    for split in ('by_stream_third', 'by_type', 'by_distance'):
        assert tuple(map(sum, zip(*map(fraction_parts, rm[split].values())))) == r_probes, 'retrieval ' + split + ' does not add up'
    for split in ('by_type', 'by_distance'):
        assert all(fraction_parts(rm[split][k])[1] == fraction_parts(im[split][k])[1] for k in im[split]), split + ': different questions'
    rk = rm['keeper_inplace_on_the_same_questions']
    assert rk['probes'] == im['probes'] and rk['final'] == im['final'] and list(rk['by_stream_third']) == list(rm['by_stream_third'])
    assert tuple(map(sum, zip(*map(fraction_parts, rk['by_stream_third'].values())))) == i_probes
    assert all(fraction_parts(rk['by_stream_third'][k])[1] == fraction_parts(v)[1] for k, v in rm['by_stream_third'].items())
    assert fraction_parts(rm['final']) == (0, i_final[1]) and rk['minutes'] == i_cost['minutes_one_decimal']
    hs = ro['hard_stream']
    # The harder stream: one seed, exploratory. Every graded run must add up, and its audit count must match the audit table.
    assert hs['status'].startswith('Exploratory') and hs['limits'] and hs['reading'] and hs['misses'] and len(hs['lessons']) == 4
    assert all(x['lesson'] and x['from'] and x['fix'] for x in hs['lessons'])
    hard = {r['arm']: r for r in hs['runs']}
    assert len(hard) == len(hs['runs']) and all(dp['lab_commit'] in r['evidence_url'] and r['what'] and r['status'] for r in hs['runs'])
    for r in hs['runs']:
        if r['total'] is None:
            assert r['status'].startswith('Stopped') and r['note']
            continue
        h_total, h_probes, h_final = (fraction_parts(r[k]) for k in ('total', 'probes', 'final'))
        assert (h_probes[0] + h_final[0], h_probes[1] + h_final[1]) == h_total and h_total[1] == 50, r['arm'] + ' does not add up'
        if r['notes_wrong'] is not None:
            assert wrong[(r['arm'], 'Harder 240K')] == r['notes_wrong'], r['arm'] + ': audit count differs'
    hard_graded = [r for r in hs['runs'] if r['status'] == 'Finished, graded']
    hard_with = [r for r in hard_graded if r['with_retrieval']]
    hard_without = [r for r in hard_graded if not r['with_retrieval']]
    hard_void = [r for r in hs['runs'] if r['status'] == 'Finished, void']
    assert len(hard_with) == 2 and len(hard_without) == 1 and len(hard_void) == 1 and not hard_void[0]['with_retrieval']
    # The claim on the page: both runs with retrieval scored above every run without it, void or not.
    assert min(fraction_parts(r['total'])[0] for r in hard_with) > max(fraction_parts(r['total'])[0] for r in hard_without + hard_void)
    ordinary = hs['on_the_second_ordinary_240k_stream']
    assert ordinary['seed'] == 1 and fraction_parts(ordinary['total']) == (second_asked, second_asked) and ordinary['arm'] in hard
    assert wrong[(ordinary['arm'], 'Second 240K')] == ordinary['notes_wrong'] == 0
    assert (ordinary['without_retrieval']['arm'], ordinary['without_retrieval']['total'], ordinary['without_retrieval']['minutes']) == (
        ip_lead[1]['arm'], ip_lead[1]['total'], ip_lead[1]['minutes'])
    assert all(dp['lab_commit'] in url for url in (*ro['evidence'].values(), *rm['evidence'].values(), *hs['evidence'].values(),
                                                   ro['coverage_before_the_runs']['url']))
    retrieval_million_cell = (rm['probes'] + ' of the questions asked along the way; the final ' + str(i_final[1])
                              + ' not answered. A stopped run')

    # The proxy with in-place notes (2026-10-12): five builds, three of them run at 1M. Every run must add up, the
    # headline rows are the current build, and the rows it is compared with must be the ones published above.
    pi = proxy['in_place']
    assert pi['method'] == 'keeper_proxy_inplace' and pi['limits'] and pi['not_run'] and len(pi['lessons']) == 9
    assert all(x['lesson'] and x['from'] and x['fix'] for x in pi['lessons']) and len(pi['not_caught_by_any_check']) == 2
    builds = {b['arm']: b for b in pi['builds']}
    assert len(builds) == len(pi['builds']) == 5 and all(b['label'] and b['what'] and b['outcome'] for b in pi['builds'])
    for r in pi['rows']:
        assert r['arm'] in builds and r['method'] == 'keeper_proxy_inplace' and rule_stream_parts(r, asked_by_seed[r['seed']])
        assert dp['lab_commit'] in r['audit_url'] and float(r['minutes']) > 0
    pi_rows = {(r['arm'], r['seed']): r for r in pi['rows']}
    assert len(pi_rows) == len(pi['rows']) == 10 and all((arm, seed) in pi_rows for arm in builds for seed in (0, 1))
    for r in pi['rows']:
        if r.get('same_run_as'):
            twin = next(x for x in pi['rows'] if x['id'] == r['same_run_as'])
            assert (twin['total'], twin['minutes'], twin['seed'], twin['notes_wrong']) == (r['total'], r['minutes'], r['seed'], r['notes_wrong'])
    pi_lead = {r['seed']: r for r in pi['rows'] if r['headline']}
    assert set(pi_lead) == {0, 1} and pi_lead[0]['arm'] == pi_lead[1]['arm'] == pi['builds'][-1]['arm']
    assert all(fraction_parts(pi_lead[seed]['total']) == (asked_by_seed[seed], asked_by_seed[seed]) and pi_lead[seed]['notes_wrong'] == 0
               and pi_lead[seed]['in_place_attempts_failed'] == 0 and pi_lead[seed]['notes_versions_audited'] == '17 of 17' for seed in (0, 1))
    pim = pi['million']
    pim_runs = pim['runs']
    assert [r['attempt'] for r in pim_runs] == [1, 2, 3] and [r['headline'] for r in pim_runs] == [False, False, True] and pim['limit']
    assert [b['arm'] for b in pi['builds'] if b['ran_at_1m']] == [r['arm'] for r in pim_runs]
    for r in pim_runs:
        assert r['status'].startswith('Finished run') and dp['lab_commit'] in r['audit_url'] and r['misses']
        p_probes, p_final, p_total = (fraction_parts(r[k]) for k in ('probes', 'final', 'total'))
        assert list(r['by_type']) == list(km['by_type']), r['arm'] + ': split differently'
        assert tuple(map(sum, zip(*map(fraction_parts, r['by_type'].values())))) == p_probes, r['arm'] + ' by type does not add up'
        assert all(fraction_parts(r['by_type'][k])[1] == fraction_parts(km['by_type'][k])[1] for k in km['by_type']), r['arm'] + ': different questions'
        assert (p_probes[0] + p_final[0], p_probes[1] + p_final[1]) == p_total and p_total[1] == k_total[1] and p_final[1] == k_final[1]
        assert r['blank'] + r['wrong'] == p_total[1] - p_total[0], r['arm'] + ': misses do not add up'
        assert r['bar_met'] == (p_total[0] >= pim['bar_minimum_right'] and r['minutes'] < pim['bar_under_minutes'])
        assert r['in_place_attempts_failed'] == 0 and r['notes_versions_audited'] == f"{r['cuts_in_place']} of {r['cuts_in_place']}"
    pim_lead = pim_runs[-1]
    assert pim_lead['arm'] == pi_lead[0]['arm'] and pim_lead['bar_met']
    pc, c240 = pim['compare'], pi['compare_at_240k']
    assert (pc['harness_keeper_in_place']['arm'], pc['harness_keeper_in_place']['total'], pc['harness_keeper_in_place']['minutes']) == (
        im['arm'], im['total'], i_cost['minutes_one_decimal'])
    assert (pc['proxy_full_rewrite']['arm'], pc['proxy_full_rewrite']['total'], round(pc['proxy_full_rewrite']['minutes'])) == (
        xm['arm'], xm['total'], x_cost['minutes'])
    assert (c240['harness_keeper_in_place']['arm'], c240['harness_keeper_in_place']['totals'], c240['harness_keeper_in_place']['minutes']) == (
        im['arm'], [ip_lead[0]['total'], ip_lead[1]['total']], [ip_lead[0]['minutes'], ip_lead[1]['minutes']])
    assert (c240['proxy_full_rewrite']['arm'], c240['proxy_full_rewrite']['totals']) == (
        first['keeper_proxy']['arm'], [first['keeper_proxy']['total'], second['keeper_proxy']['total']])
    assert [str(round(float(x))) for x in c240['proxy_full_rewrite']['minutes']] == [first['keeper_proxy']['minutes'], second['keeper_proxy']['minutes']]
    # Where the time of the current build's 1M run went: the parts must agree with the totals quoted beside them.
    tw = pim['where_the_time_went']
    assert tw['arm'] == pim_lead['arm'] and round(tw['run_seconds'] / 60, 1) == pim_lead['minutes']
    assert round(100 * (tw['server_reading_seconds'] + tw['server_writing_seconds']) / tw['run_seconds']) == tw['server_percent']
    assert sum(tw['written_thousands'].values()) == int(tw['tokens_written'].removesuffix('K'))
    assert tw['written_thousands']['notes'] == int(pim_lead['notes_tokens_written'].removesuffix('K'))
    assert 0 < tw['proxy_handling_seconds'] - tw['proxy_waiting_for_the_server_seconds'] < 30 and tw['proxy_own_time'].startswith('Under half a minute')
    assert all(dp['lab_commit'] in url for url in pi['evidence'].values())

    # The harness keeper with in-place notes on a second 1M stream: its parts, and the two streams together, must add up.
    sm = ip['second_million_stream']
    assert sm['status'].startswith('Finished run') and sm['method'] == 'keeper_inplace' and sm['seed'] == 1 and sm['limit'] and sm['fix']
    s_total, s_final = fraction_parts(sm['total']), fraction_parts(sm['final'])
    s_types = tuple(map(sum, zip(*map(fraction_parts, sm['by_type'].values()))))
    assert (s_types[0] + s_final[0], s_types[1] + s_final[1]) == s_total and sm['blank'] + sm['wrong'] == s_total[1] - s_total[0]
    assert round(100 * s_total[0] / s_total[1], 1) == sm['total_percent_one_decimal']
    assert (sm['first_stream']['arm'], sm['first_stream']['total'], sm['first_stream']['minutes']) == (im['arm'], im['total'], i_cost['minutes_one_decimal'])
    both = fraction_parts(sm['both_streams']['total'])
    assert both == (i_total[0] + s_total[0], i_total[1] + s_total[1])
    assert (round(100 * both[0] / both[1]), round(100 * both[0] / both[1], 1)) == (sm['both_streams']['percent'], sm['both_streams']['percent_one_decimal'])
    assert sm['notes_versions_audited'] == f"{sm['cuts_in_place']} of {sm['cuts_in_place']}" and sm['in_place_attempts_failed'] == 0
    assert all(dp['lab_commit'] in url for url in sm['evidence'].values())

    # The proxy with in-place notes on the same second stream (the evening of the same day): the same checks.
    ps = pi['second_million_stream']
    assert ps['status'].startswith('Finished run') and ps['method'] == 'keeper_proxy_inplace' and ps['seed'] == 1 and ps['limit']
    p2_total, p2_final = fraction_parts(ps['total']), fraction_parts(ps['final'])
    p2_types = tuple(map(sum, zip(*map(fraction_parts, ps['by_type'].values()))))
    assert (p2_types[0] + p2_final[0], p2_types[1] + p2_final[1]) == p2_total
    assert ps['blank'] + ps['wrong'] == p2_total[1] - p2_total[0]
    assert round(100 * p2_total[0] / p2_total[1], 1) == ps['total_percent_one_decimal']
    assert list(ps['by_type']) == list(sm['by_type']) and all(
        fraction_parts(ps['by_type'][k])[1] == fraction_parts(v)[1] for k, v in sm['by_type'].items()), 'the proxy was asked different questions'
    assert fraction_parts(ps['first_stream']['total']) == fraction_parts(pim_lead['total'])
    assert (ps['first_stream']['arm'], ps['first_stream']['minutes']) == (pim_lead['arm'], pim_lead['minutes'])
    p2_both = fraction_parts(ps['both_streams']['total'])
    assert p2_both == tuple(map(sum, zip(fraction_parts(pim_lead['total']), p2_total)))
    assert (round(100 * p2_both[0] / p2_both[1]), round(100 * p2_both[0] / p2_both[1], 1)) == (
        ps['both_streams']['percent'], ps['both_streams']['percent_one_decimal'])
    assert ps['notes_versions_audited'] == f"{ps['cuts_in_place']} of {ps['cuts_in_place']}" and ps['in_place_attempts_failed'] == 0
    assert ps['expected_before_the_run'] and all(LAB_COMMIT_13D in url for url in ps['evidence'].values())

    # A rule that names nothing it covers. Each run must add up twice: by question type, and as governed plus other questions.
    ur = ro['unnamed_rule']
    assert ur['limit'] and ur['stream_fault'] and ur['what_retrieval_answered'] and ur['what_the_keeper_did']
    governed = {}
    for who, method in (('keeper', 'keeper_inplace'), ('retrieval', 'retrieval')):
        assert [r['seed'] for r in ur[who]] == [0, 1]
        for r in ur[who]:
            parts = rule_stream_parts({**r, 'method': method, 'id': 'unnamed-rule-' + r['arm']}, asked_by_seed[r['seed']])
            kept, others = fraction_parts(r['governed_by_the_rule']), fraction_parts(r['other_questions'])
            assert (kept[0] + others[0], kept[1] + others[1]) == parts['total'], r['arm'] + ': governed and other questions do not add up'
        governed[who] = tuple(map(sum, zip(*(fraction_parts(r['governed_by_the_rule']) for r in ur[who]))))
        assert governed[who] == fraction_parts(ur['both_streams'][who])
    assert all(fraction_parts(a['governed_by_the_rule'])[1] == fraction_parts(b['governed_by_the_rule'])[1]
               for a, b in zip(ur['keeper'], ur['retrieval'])), 'the two methods were asked different governed questions'
    assert governed['keeper'][0] == governed['keeper'][1] and governed['retrieval'][0] == 0
    # The page says the keeper's only misses outside the governed questions are the tallies it answered with the word.
    assert sum(fraction_parts(r['other_questions'])[1] - fraction_parts(r['other_questions'])[0] for r in ur['keeper']) == ur['keeper_tallies_answered_with_the_word']
    assert sum(fraction_parts(r['house_method'])[1] - fraction_parts(r['house_method'])[0] for r in ur['keeper']) == ur['keeper_tallies_answered_with_the_word']
    assert all(dp['lab_commit'] in url for url in ur['evidence'].values())

    # The harder stream on a second seed: one graded run, with retrieval; its partner did not finish, so there is no comparison.
    h2 = hs['second_seed']
    assert h2['seed'] == 1 and h2['reading'] and h2['why_slow'] and len(h2['runs']) == 2 and not set(r['arm'] for r in h2['runs']) & set(hard)
    assert all(dp['lab_commit'] in r['evidence_url'] and r['what'] and r['note'] for r in h2['runs'])
    h2_graded = [r for r in h2['runs'] if r['status'] == 'Finished, graded']
    h2_stopped = [r for r in h2['runs'] if r['status'] != 'Finished, graded']
    assert len(h2_graded) == 1 and h2_graded[0]['with_retrieval'] and len(h2_stopped) == 1
    assert h2_stopped[0]['total'] is None and h2_stopped[0]['status'].startswith('Stopped') and not h2_stopped[0]['with_retrieval']
    h2_total, h2_probes, h2_final = (fraction_parts(h2_graded[0][k]) for k in ('total', 'probes', 'final'))
    assert (h2_probes[0] + h2_final[0], h2_probes[1] + h2_final[1]) == h2_total and h2_total[1] == h2['questions']
    assert all(dp['lab_commit'] in url for url in h2['evidence'].values())

    keeper_rows = [
        [label(first['keeper']), run_cell(first['keeper']), run_cell(second['keeper']), km['total'] + ' in ' + str(k_cost['minutes']) + ' min'],
        [DP_METHODS['keeper_inplace'], run_cell(ip_lead[0]), run_cell(ip_lead[1]), im['total'] + ' in ' + str(i_cost['minutes']) + ' min'],
        [DP_METHODS['keeper_proxy_inplace'], run_cell(pi_lead[0]), run_cell(pi_lead[1]),
         pim_lead['total'] + ' in ' + str(round(pim_lead['minutes'])) + ' min'],
        [label(first['keeper_proxy']), run_cell(first['keeper_proxy']), run_cell(second['keeper_proxy']),
         xm['total'] + ' in ' + str(x_cost['minutes']) + ' min'],
        [DP_METHODS['keeper_proxy'] + ', before its fixes (' + before[0]['arm'] + ')', run_cell(before[0]), run_cell(before[1]), 'Not run'],
        [DP_METHODS['retrieval'] + ', final build', run_cell(ro_by_build['final'][0]), run_cell(ro_by_build['final'][1]), retrieval_million_cell]]
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
    def timed(total, minutes):
        return total + ' in ' + str(minutes) + ' min'
    inplace_rows = [
        ['Full notes rewrite (the keeper above)', timed(fr_rows[0]['total'], fr_rows[0]['minutes']),
         timed(fr_rows[1]['total'], fr_rows[1]['minutes']), timed(km['total'], k_cost['minutes'])],
        ['In-place notes, with every check', run_cell(ip_lead[0]), run_cell(ip_lead[1]), timed(im['total'], i_cost['minutes'])],
        ['In-place notes, five earlier runs while the checks were added',
         '; '.join(run_cell(r) for r in ip_later if r['seed'] == 0), '; '.join(run_cell(r) for r in ip_later if r['seed'] == 1), 'Not run']]
    # The proxy with in-place notes beside the two forms it is measured against: the three rows of the headline.
    proxy_now_rows = [
        ['Proxy with in-place notes (today’s build)', run_cell(pi_lead[0]), run_cell(pi_lead[1]),
         timed(pim_lead['total'], round(pim_lead['minutes']))],
        ['Keeper inside the test harness, in-place notes', run_cell(ip_lead[0]), run_cell(ip_lead[1]), timed(im['total'], i_cost['minutes'])],
        ['Proxy, earlier form (rewrites all its notes at each cut)',
         *(timed(total, minutes) for total, minutes in zip(c240['proxy_full_rewrite']['totals'], c240['proxy_full_rewrite']['minutes'])),
         timed(xm['total'], x_cost['minutes'])]]
    pim_by_arm = {r['arm']: r for r in pim_runs}

    def build_cell(arm, seed):
        r = pi_rows[(arm, seed)]
        return escape(run_cell(r)) + (' (the same run as the first build’s)' if r.get('same_run_as') else '')
    proxy_run_rows = [
        [escape(b['label']), link(pi['evidence']['table_240k'], b['arm']), build_cell(b['arm'], 0), build_cell(b['arm'], 1),
         escape(timed(pim_by_arm[b['arm']]['total'], pim_by_arm[b['arm']]['minutes'])) if b['ran_at_1m'] else 'Not run',
         escape(b['outcome'])] for b in pi['builds']]
    unnamed_rule_rows = (
        [['Questions the rule governs, ' + name, k['governed_by_the_rule'], r['governed_by_the_rule']]
         for name, k, r in zip(('first stream', 'second stream'), ur['keeper'], ur['retrieval'])]
        + [['Questions the rule governs, both streams', ur['both_streams']['keeper'], ur['both_streams']['retrieval']]]
        + [['The other questions, ' + name, k['other_questions'], r['other_questions']]
           for name, k, r in zip(('first stream', 'second stream'), ur['keeper'], ur['retrieval'])]
        + [['Whole stream, ' + name, run_cell(k), run_cell(r)]
           for name, k, r in zip(('first', 'second'), ur['keeper'], ur['retrieval'])])
    inplace_million_rows = (
        [['All ' + str(i_total[1]) + ' questions', im['total'] + ' (' + str(im['total_percent']) + '%)',
          km['total'] + ' (' + str(km['total_percent']) + '%)']]
        + [[names[k], im[split][k], km[split][k]] for split in ('by_type', 'by_distance') for k in km[split]]
        + [['Final questions', im['final'], km['final']],
           ['Minutes', str(i_cost['minutes']), str(k_cost['minutes'])],
           ['Model calls (note-taking calls among them)', f"{i_cost['lm_calls']:,} ({i_cost['notes_calls']})",
            f"{k_cost['lm_calls']:,} ({k_cost['notes_calls']})"],
           ['Prompt tokens sent (reused from the cache)', i_cost['prompt_tokens'] + ' (' + i_cost['prompt_from_cache'] + ')',
            fr['million']['prompt_tokens'] + ' (' + fr['million']['prompt_from_cache'] + ')'],
           ['Tokens written by the note-taking calls', i_cost['notes_tokens_written'], fr['million']['notes_tokens_written']],
           ['Size of the notes, in tokens', f"{i_notes['min_tokens']:,} to {i_notes['max_tokens']:,}",
            f"{km['notes']['min_tokens']:,} to {km['notes']['max_tokens']:,}"],
           ['Counters ever wrong in the notes (audit)', str(im['notes_wrong']) + ' in ' + str(i_notes['versions_audited']) + ' versions',
            fr['million']['notes_wrong_note']]])
    audit_rows = [[escape(r['notes']), link(r['audit_url'], r['arm']), escape(r['stream']), escape(r['total']), escape(str(r['minutes'])),
                   str(r['notes_wrong']) + (' (' + escape(r['wrong_note']) + ')' if r.get('wrong_note') else '')] for r in audit['rows']]
    recount = audit.get('recount')
    recount_note = ''
    if recount:
        # Every recounted row must carry the new count, and each rose by one; the sentence names streams, not arm codes.
        assert all(wrong[(c['arm'], c['stream'])] == c['to'] == c['from'] + 1 for c in recount['changed'])
        per_stream = {}
        for c in recount['changed']:
            per_stream[c['stream']] = per_stream.get(c['stream'], 0) + 1
        places = ' and '.join(str(n) + ' on the ' + stream[0].lower() + stream[1:] + ' stream' for stream, n in per_stream.items())
        recount_note = ('<p>The audit was run again after a fault was found in the audit tool: it had skipped some versions of the notes. '
                        'In the recount ' + str(len(recount['changed'])) + ' counts in this table rose by one: ' + escape(places)
                        + '. The linked files show the recount.</p>')
    retrieval_rows = [
        [DP_METHODS['retrieval'] + ', final build', run_cell(ro_by_build['final'][0]), run_cell(ro_by_build['final'][1]), retrieval_million_cell],
        [DP_METHODS['keeper_inplace'], run_cell(ip_lead[0]), run_cell(ip_lead[1]),
         im['probes'] + ' of the questions asked along the way; ' + im['final'] + ' final'],
        [DP_METHODS['keeper_inplace_retrieval'], 'Not run', run_cell(ordinary), 'Not run'],
        [DP_METHODS['retrieval'] + ', first build', run_cell(ro_by_build['first'][0]),
         'Not run', 'Not run'],
        [DP_METHODS['retrieval'] + ', second build', run_cell(ro_by_build['second'][0]),
         run_cell(ro_by_build['second'][1]), 'Not run']]
    third_names = {'probe_items_1_9': 'First third of the stream (question sets 1–9)', 'probe_items_10_19': 'Middle third (sets 10–19)',
                   'probe_items_20_29': 'Last third (sets 20–29)'}
    retrieval_million_rows = (
        [['All ' + str(r_probes[1]) + ' questions asked along the way', rm['probes'] + ' (' + str(rm['probes_percent']) + '%)',
          im['probes'] + ' (' + str(im['probes_percent']) + '%)']]
        + [[third_names[k], v, rk['by_stream_third'][k]] for k, v in rm['by_stream_third'].items()]
        + [[names[k], rm[split][k], im[split][k]] for split in ('by_type', 'by_distance') for k in im[split]]
        + [['The ' + str(i_final[1]) + ' final questions', 'Not answered', im['final']],
           ['Minutes', str(rm['minutes_to_reach_the_final_item']) + ' to reach the final questions; stopped at '
            + str(rm['stopped_after_minutes']), str(i_cost['minutes_one_decimal']) + ' for the whole run']])
    build_names = {'first': 'First build', 'second': 'Second build', 'final': 'Final build'}
    stream_names = {0: 'First 240K', 1: 'Second 240K'}
    retrieval_run_rows = (
        [[build_names[r['build']], link(ro['evidence']['table_240k'], r['arm']), stream_names[r['seed']], escape(r['total']),
          escape(r['final']), escape(r['minutes'])] for r in ro['rows']]
        + [['Final build, 40 sentences per question', link(rm['evidence']['answers_scored'], rm['arm']), '1,000,000',
            escape(rm['probes']) + ' along the way', 'Not answered', 'Stopped at ' + str(rm['stopped_after_minutes'])],
           ['Final build', link(hs['evidence']['retrieval_answers_at_stop'], hs['retrieval']['arm']), 'Harder 240K',
            escape(hs['retrieval']['answered_right']) + ' when stopped', 'Not reached', 'Stopped at 20.5']])
    def hard_cell(r, key):
        return escape(r[key]) if r[key] is not None else 'Not graded'
    harder_run_rows = (
        [[escape(r['what']), link(r['evidence_url'], r['arm']), hard_cell(r, 'total') + (' raw, void' if r['status'] == 'Finished, void' else ''),
          hard_cell(r, 'final'), escape(r['minutes']) if r['minutes'] is not None else 'Stopped at 11',
          str(r['notes_wrong']) if r['notes_wrong'] is not None else 'Not reported', escape(r['note'])] for r in hs['runs']]
        + [['Retrieval only, final build', link(hs['evidence']['retrieval_answers_at_stop'], hs['retrieval']['arm']),
            escape(hs['retrieval']['answered_right']) + ' when stopped', 'Not reached', 'Stopped at 20.5', 'No notes',
            'Too slow to finish. The keeper with in-place notes had ' + escape(hs['retrieval']['keeper_inplace_at_that_point'])
            + ' at that point.']]
        + [[escape(r['what']), link(r['evidence_url'], r['arm']), hard_cell(r, 'total'), hard_cell(r, 'final'),
            escape(r['minutes']) if r['minutes'] is not None else 'Stopped at 22.7',
            str(r['notes_wrong']) if r['notes_wrong'] is not None else 'Not reported', escape(r['note'])] for r in h2['runs']])
    stream_heads = ['First 240K stream, ' + str(first_asked) + ' questions', 'Second 240K stream, ' + str(second_asked) + ' questions',
                    '1,000,000-token stream, ' + str(k_total[1]) + ' questions']
    segments = [str(x) for x in k_cost['minutes_per_100k']]
    i_segments = [str(x) for x in i_cost['minutes_per_100k']]
    x_segments = [str(x) for x in x_cost['minutes_per_100k']]
    costs = [[label(r), f"{r['tokens_written']:,}", f"{r['lm_calls']:,}", f"{r['peak_context']:,}",
              str(r.get('recalls', '–'))] for r in rounds['round2']['rows'] if r['cap'] == '32K']

    # The repair marathon on real code: 40 planted one-token defects and three standing instructions, one run per arm.
    rp = dp['repair']
    assert rp['n_bugs'] == 40 and rp['notices'] == 3 and rp['audit_asked'] == 7 and len(rp['rule_labels']) == 3
    assert all(rp['lab_commit'] in rp[k] for k in ('note_url', 'task_url', 'runs_url'))
    for s in rp['seeds']:
        for r in s['rows']:
            if 'ungraded' in r:
                assert r['fixed'] == r['clean'] == '—', r['arm']
                continue
            fixed, clean = fraction_parts(r['fixed']), fraction_parts(r['clean'])
            assert 0 <= clean[0] <= fixed[0] and fixed == (fixed[0], 40) and clean[1] == 40, r['arm']
            rules_kept = [fraction_parts(r[k]) for k in ('r1', 'r2', 'r3')]
            assert all(0 <= n <= m for n, m in rules_kept), r['arm']
            # clean counts tickets (a ticket can break several rules), the rule columns count applications: no shared total
            assert fixed[0] - clean[0] <= sum(m - n for n, m in rules_kept) or fixed[0] == clean[0], r['arm']
            if r.get('audit'):
                a = fraction_parts(r['audit'])
                assert 0 <= a[0] <= a[1] == rp['audit_asked'], r['arm']
    rp_s0, rp_s1x, rp_s2x = rp['seeds']
    assert rp_s2x['id'] == 's2x' and rp_s2x['variant'] and [r['arm'] for r in rp_s2x['rows']] == ['XR6', 'W32g']
    repair_head = ['Arm', 'Context managed by', 'Fixed', 'Clean', *rp['rule_labels']]

    def repair_rows(s):
        out_rows = []
        for r in s['rows']:
            cells = [r['arm'] + (' — ' + r['manages'] if r['manages'] else ''), r['fixed'], r['clean'], r['r1'], r['r2'], r['r3']]
            if s['audit_ticket']:
                cells.append(r['audit'] if r['audit'] is not None else '—')
            cells += [str(r['minutes']), r['calls'] + ((' (' + r['note'] + ')') if r.get('note') else '')]
            out_rows.append(cells)
        return out_rows

    repair_head_s1 = repair_head + ['Oldest fixes recalled (7 asked)', 'Minutes', 'Model calls']
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
        'DP_TABLE_INPLACE': table(['How the notes are taken', *stream_heads], inplace_rows,
                                  'In-place notes against a full notes rewrite: questions right and minutes per run'),
        'DP_TABLE_INPLACE_1M': table(['What was counted', 'In-place notes', 'Full notes rewrite'], inplace_million_rows,
                                     'One million tokens: the keeper with in-place notes and with a full notes rewrite'),
        'DP_TABLE_AUDIT': html_table(['How the notes were taken', 'Code in the evidence files', 'Stream', 'Questions right', 'Minutes',
                                      'Counters ever wrong in the notes'], audit_rows,
                                     'Every audited run: its score and the wrong values the audit found in its notes', 'ctx-comparison'),
        'DP_TABLE_RETRIEVAL': table(['Method at 32K', *stream_heads], retrieval_rows,
                                    'Retrieval alone against the keeper with in-place notes: questions right and minutes per run'),
        'DP_TABLE_RETRIEVAL_1M': table(['What was counted', 'Retrieval only: a stopped run', 'Keeper with in-place notes'], retrieval_million_rows,
                                       'One million tokens: retrieval alone and the keeper on the questions asked along the way'),
        'DP_TABLE_RETRIEVAL_RUNS': html_table(['Build', 'Code in the evidence files', 'Stream', 'Questions right', 'Final questions',
                                               'Minutes'], retrieval_run_rows, 'Every retrieval-only run', 'ctx-comparison'),
        'DP_TABLE_HARDER_RUNS': html_table(['Run', 'Code in the evidence files', 'Questions right', 'Final questions', 'Minutes',
                                            'Counters ever wrong in the notes', 'What happened'], harder_run_rows,
                                           'Every run on the harder stream', 'ctx-comparison'),
        'DP_C10B_LOG': escape(hs['evidence']['log'], quote=True),
        'DP_TABLE_PROXY_NOW': table(['How the notes are kept', *stream_heads], proxy_now_rows,
                                    'The proxy with in-place notes, the keeper inside the test harness and the proxy’s earlier form: questions right and minutes per run'),
        'DP_TABLE_PROXY_RUNS': html_table(['Build', 'Code in the evidence files', *stream_heads, 'What happened'], proxy_run_rows,
                                          'Every build of the proxy with in-place notes', 'ctx-comparison'),
        'DP_TABLE_UNNAMED_RULE': table(['What was counted', DP_METHODS['keeper_inplace'], DP_METHODS['retrieval']], unnamed_rule_rows,
                                       'A rule that names nothing: the keeper with in-place notes against retrieval alone'),
        'DP_REPAIR_TABLE_S0': table(repair_head + ['Minutes', 'Model calls'], repair_rows(rp_s0),
                                    'The repair marathon, first seed: 40 planted defects fixed, and clean of the three standing instructions', 'ctx-comparison ctx-fit'),
        'DP_REPAIR_TABLE_S1X': table(repair_head_s1, repair_rows(rp_s1x),
                                     'The repair marathon, second seed, with a final audit of the seven oldest fixes', 'ctx-comparison ctx-fit'),
        'DP_REPAIR_TABLE_S2X': table(repair_head_s1, repair_rows(rp_s2x),
                                     'The repair marathon without its trap tickets (the *_index functions and peekable left out), third seed, with the same final audit', 'ctx-comparison ctx-fit'),
        'DP_REPAIR_TASK_URL': escape(rp['task_url'], quote=True),
        'DP_REPAIR_RUNS_URL': escape(rp['runs_url'], quote=True),
        'DP_REPAIR_NOTE_URL': escape(rp['note_url'], quote=True),
        'DP_REPAIR_PENDING': escape(rp['pending']),
        'DP_REPAIR_LIMIT': escape(rp['limit']),
        'DP_C12_DATA_URL': escape(pi['evidence']['folder'], quote=True),
        'DP_C12_TABLE_240K': escape(pi['evidence']['table_240k'], quote=True),
        'DP_C12_TABLE_1M': escape(pi['evidence']['table_1m'], quote=True),
        'DP_C12_LOG_NIGHT': escape(pi['evidence']['log_first_night'], quote=True),
        'DP_C12_LOG_MORNING': escape(pi['evidence']['log_morning'], quote=True),
        'DP_C12_CLOSEOUT': escape(pi['evidence']['closeout_note'], quote=True),
        'DP_P1M_DETAILS': escape(pi['evidence']['details_1m'], quote=True),
        'DP_P1M_AUDIT': escape(pi['evidence']['notes_audit_1m'], quote=True),
        'DP_P1M_COUNTERS': escape(pi['evidence']['proxy_counters_1m'], quote=True),
        'DP_PROXY_AUDIT_CODE': escape(pi['evidence']['audit_code'], quote=True),
        'DP_S1M_DETAILS': escape(sm['evidence']['details'], quote=True),
        'DP_S1M_AUDIT': escape(sm['evidence']['audit'], quote=True),
        'DP_P1M2_LOG': escape(ps['evidence']['log'], quote=True),
        'DP_P1M2_AUDIT': escape(ps['evidence']['audit'], quote=True),
        'DP_P1M2_STATE': escape(ps['evidence']['state'], quote=True),
        'DP_UR_TABLE': escape(ur['evidence']['table'], quote=True),
        'DP_UR_TASK': escape(ur['evidence']['task_generator'], quote=True),
        'DP_H2_TABLE': escape(h2['evidence']['table'], quote=True),
        'DP_AUDIT_RECOUNT': recount_note,
        'DP_I1M_SEGMENTS': escape(', '.join(i_segments[:-1]) + ' and ' + i_segments[-1]),
        'DP_I1M_GRADE': escape(im['evidence']['details'], quote=True),
        'DP_I1M_NOTES': escape(im['evidence']['final_notes'], quote=True),
        'DP_I1M_COUNTERS': escape(im['evidence']['counters'], quote=True),
        'DP_I1M_AUDIT': escape(im['evidence']['audit'], quote=True),
        'DP_C10_DATA_URL': escape(dp['data_urls']['in_place_notes_audit_and_retrieval'], quote=True),
        'DP_C10_TABLE_240K': escape(ip['evidence']['table_240k'], quote=True),
        'DP_C10_TABLE_1M': escape(ip['evidence']['table_1m'], quote=True),
        'DP_C10_TABLE_HARD': escape(hs['evidence']['table'], quote=True),
        'DP_AUDIT_EXAMPLE': escape(ip['evidence']['audit_example'], quote=True),
        'DP_KEEPER_CODE': escape(ip['evidence']['keeper_code'], quote=True),
        'DP_AUDIT_CODE': escape(ip['evidence']['audit_code'], quote=True),
        'DP_RETRIEVAL_CODE': escape(ro['evidence']['retrieval_code'], quote=True),
        'DP_RETRIEVAL_COVERAGE': escape(ro['coverage_before_the_runs']['url'], quote=True),
        'DP_R1M_ANSWERS': escape(rm['evidence']['answers_scored'], quote=True),
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
    ip = keeper['in_place']
    im = ip['million']
    ip_lead = {r['seed']: r for r in ip['rows'] if r['headline']}
    ro = dp['retrieval_only']
    rb = {r['seed']: r for r in ro['rows'] if r['build'] == 'final'}
    rm = ro['million']
    hs = ro['hard_stream']
    hard_graded = [r for r in hs['runs'] if r['status'] == 'Finished, graded']
    hard_with = [r['total'] for r in hard_graded if r['with_retrieval']]
    hard_without = [r['total'] for r in hard_graded if not r['with_retrieval']]
    hard_second = next(r for r in hs['second_seed']['runs'] if r['status'] == 'Finished, graded')
    pi = keeper['proxy']['in_place']
    pi_lead = {r['seed']: r for r in pi['rows'] if r['headline']}
    pim_lead = next(r for r in pi['million']['runs'] if r['headline'])
    sm = ip['second_million_stream']
    ps = pi['second_million_stream']
    ur = ro['unnamed_rule']
    rp = dp['repair']
    rp0 = {r['arm']: r for r in rp['seeds'][0]['rows']}
    rp1 = {r['arm']: r for r in rp['seeds'][1]['rows']}
    rp2 = {r['arm']: r for r in rp['seeds'][2]['rows']}
    assert rp['seeds'][0]['id'] == 's0' and rp['seeds'][1]['id'] == 's1x' and rp['seeds'][2]['id'] == 's2x'
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
         'A keeper that asks nothing of the agent: ' + k0['total'] + ' and ' + k1['keeper']['total'] + '; run as a proxy with in-place notes: '
         + pi_lead[0]['total'] + ' and ' + pi_lead[1]['total'] + ' (' + x0['total'] + ' and ' + k1['keeper_proxy']['total']
         + ' in the proxy’s earlier form)',
         'On the first stream, archive and search with pinned rule lines: ' + rules['B32iraR']['total'] + '; window-drop: '
         + rules['W32']['total'] + '. Four methods with no pinned place for a rule: ' + f'{no_place[0]}/{no_place[1]}' + ' rule questions',
         'context-distance.html#keeper'],
        ['A 1,000,000-token history through a 32K window',
         'The keeper as a proxy, with in-place notes: ' + pim_lead['total'] + ' right in ' + str(round(pim_lead['minutes']))
         + ' min on one stream, and ' + ps['total'] + ' in ' + str(round(ps['minutes']))
         + ' min on a second. The keeper inside the test harness, with in-place notes: ' + im['total'] + ' in '
         + str(im['cost']['minutes']) + ' min on the first stream and ' + sm['total'] + ' in ' + str(round(sm['minutes']))
         + ' min on the second. All graded by the task’s own checker',
         'The keeper with a full notes rewrite: ' + km['total'] + ' in ' + str(km['cost']['minutes']) + ' min; the proxy’s earlier form: '
         + xm['total'] + ' in ' + str(xm['cost']['minutes']) + ' min. Archive and search with pinned rule lines: ' + million['total']
         + ' in a stopped run that was not verified',
         'context-distance.html#million'],
        ['Retrieval alone (“RAG”) against the keeper, on the same three histories',
         'The keeper with in-place notes: ' + ip_lead[0]['total'] + ' and ' + ip_lead[1]['total'] + ' at 240K; ' + im['probes']
         + ' of the questions asked along the 1,000,000-token history, and ' + im['final'] + ' final questions',
         'Retrieval only: ' + rb[0]['total'] + ' and ' + rb[1]['total'] + ' at 240K; ' + rm['probes']
         + ' along the 1,000,000-token history, where it did not answer the final questions (a stopped run)',
         'context-distance.html#retrieval'],
        ['A rule that names nothing it covers, on two 240K histories',
         'The keeper with in-place notes applied it every time: ' + ur['both_streams']['keeper'],
         'Retrieval only: ' + ur['both_streams']['retrieval'] + '. It gave the true number every time; the rule had left the window and no question leads a search back to it',
         'context-distance.html#unnamed-rule'],
        ['A harder 240K history that says “it” and makes some values depend on others (the comparison is one seed, exploratory)',
         'The keeper with in-place notes and retrieval together: ' + ' and '.join(hard_with) + ', and ' + hard_second['total']
         + ' on a second seed',
         'The keeper with notes only: ' + ' and '.join(hard_without) + '; retrieval only: stopped with ' + hs['retrieval']['answered_right']
         + ' right on the first three question sets',
         'context-distance.html#harder'],
        ['A real-code repair marathon: 40 planted defects and three standing instructions that arrive as notices (three seeds)',
         'The proxy between agent and server: ' + rp0['XR']['clean'] + ' and ' + rp1['XR3']['clean'] + ' clean; the audit of the oldest fixes '
         + rp1['XR3']['audit'] + ', and ' + str(rp1['XR4']['minutes']) + ' min at medium thinking effort',
         'Summaries: ' + rp0['C32']['clean'] + ' then ' + rp1['C32']['clean'] + ' (one summary came back as work, not a summary; audit '
         + rp1['C32']['audit'] + '); at the same medium effort summaries matched the proxy, ' + rp1['C32m']['clean'] + ' clean, audit '
         + rp1['C32m']['audit'] + ', ' + str(rp1['C32m']['minutes']) + ' min; window-drop: ' + rp0['W32']['clean'] + ' and '
         + rp1['W32']['clean'] + ' clean, and ' + rp2['W32g']['clean'] + ' without the task’s trap tickets',
         'context-distance.html#repair'],
    ]
    html = table(['What was tested', 'Best so far', 'Compared with'], [r[:3] for r in rows], 'What each long-context test found', 'ctx-summary-table')
    for r in rows:
        html = html.replace('<th scope="row">' + escape(r[0]) + '</th>', '<th scope="row">' + link('@/' + r[3], r[0]) + '</th>', 1)
    return html


def score_plot(title, rows, caption):
    """Accessible, responsive bars from the same records used by the evidence tables."""
    denominators = {fraction_parts(value)[1] for _, value, _, _ in rows}
    assert len(denominators) == 1, 'A plot must compare the same question set'
    total = denominators.pop()
    body = ''
    for label, value, timing, tone in rows:
        correct, asked = fraction_parts(value)
        assert 0 <= correct <= asked and tone in ('keeper', 'proxy', 'baseline')
        body += ('<div class="ctx-score-row"><div class="ctx-score-label">' + escape(label)
                 + '<span>' + escape(timing) + '</span></div><div class="ctx-score-track" aria-hidden="true">'
                 + '<span class="ctx-score-fill ctx-score-' + tone + '" style="width:'
                 + f'{100 * correct / asked:.4f}' + '%"></span></div>'
                 + '<strong>' + escape(value) + '</strong></div>')
    return ('<figure class="ctx-score-plot"><figcaption><strong>' + escape(title)
            + '</strong><span>Correct answers · scale 0–'
            + str(total) + ' · longer is better</span></figcaption>' + body
            + '<p class="ctx-table-note">' + escape(caption) + '</p></figure>')


def minutes_plot(title, rows, caption):
    """Bars of whole-run minutes on one scale, from the same records as the tables; rows are (label, score, minutes, tone)."""
    ceiling = math.ceil(max(float(value) for _, _, value, _ in rows) / 10) * 10
    body = ''
    for label, value, minutes, tone in rows:
        assert tone in ('keeper', 'proxy', 'baseline') and fraction_parts(value)
        body += ('<div class="ctx-score-row"><div class="ctx-score-label">' + escape(label)
                 + '<span>' + escape(value) + ' correct</span></div><div class="ctx-score-track" aria-hidden="true">'
                 + '<span class="ctx-score-fill ctx-score-' + tone + '" style="width:'
                 + f'{100 * float(minutes) / ceiling:.4f}' + '%"></span></div><strong>' + f'{float(minutes):.1f}' + ' min</strong></div>')
    return ('<figure class="ctx-score-plot"><figcaption><strong>' + escape(title)
            + '</strong><span>Minutes for the whole run · shorter is faster · scale 0–' + str(ceiling)
            + ' minutes</span></figcaption>' + body + '<p class="ctx-table-note">' + escape(caption) + '</p></figure>')


def bar_plot(title, scale, rows, caption):
    """Bars on one stated scale; rows are (label, sublabel, share of the scale from 0 to 1, value text, tone)."""
    body = ''
    for label, sublabel, share, value, tone in rows:
        assert 0 <= share <= 1 and tone in ('keeper', 'proxy', 'baseline')
        body += ('<div class="ctx-score-row"><div class="ctx-score-label">' + escape(label)
                 + '<span>' + escape(sublabel) + '</span></div><div class="ctx-score-track" aria-hidden="true">'
                 + '<span class="ctx-score-fill ctx-score-' + tone + '" style="width:' + f'{100 * share:.4f}'
                 + '%"></span></div><strong>' + escape(value) + '</strong></div>')
    return ('<figure class="ctx-score-plot"><figcaption><strong>' + escape(title) + '</strong><span>' + escape(scale)
            + '</span></figcaption>' + body + '<p class="ctx-table-note">' + escape(caption) + '</p></figure>')


def visual_replacements(by_id, dp):
    keeper = dp['keeper']
    k0, x0 = keeper['rows']
    rules = {r['arm']: r for r in dp['rules']['rows']}
    ip = keeper['in_place']
    im = ip['million']
    ip_lead = {r['seed']: r for r in ip['rows'] if r['headline']}
    fr = {r['seed']: r for r in ip['full_rewrite']['rows']}
    ro = dp['retrieval_only']
    rb0 = next(r for r in ro['rows'] if r['build'] == 'final' and r['seed'] == 0)
    rm = ro['million']
    pi = keeper['proxy']['in_place']
    pi_lead = {r['seed']: r for r in pi['rows'] if r['headline']}
    pim = pi['million']
    pim_lead = next(r for r in pim['runs'] if r['headline'])
    tw = pim['where_the_time_went']
    sm = ip['second_million_stream']
    ur = ro['unnamed_rule']
    short_rows = [(label, r['total'], r['minutes'] + ' min', tone) for label, r, tone in [
        ('Keeper', k0, 'keeper'), ('Keeper, in-place notes', ip_lead[0], 'keeper'),
        ('Keeper proxy, in-place notes', pi_lead[0], 'proxy'), ('Keeper proxy, earlier form', x0, 'proxy'),
        ('Retrieval only', rb0, 'baseline'),
        ('Agent-managed archive + pinned rules', rules['B32iraR'], 'baseline'),
        ('Window-drop', rules['W32'], 'baseline'), ('Tool-output compaction', rules['P32'], 'baseline')]]
    million_rows = [('Keeper proxy, in-place notes', pim_lead['total'], str(round(pim_lead['minutes'])) + ' min', 'proxy')] + [
        (label, r['total'], str(r['cost']['minutes']) + ' min', tone) for label, r, tone in [
            ('Keeper, in-place notes', im, 'keeper'), ('Keeper, full notes rewrite', keeper['million'], 'keeper'),
            ('Keeper proxy, full notes rewrite', keeper['proxy']['million'], 'proxy')]]
    # The headline: minutes for the same million-token stream, with the score beside each bar.
    proxy_now_rows = [
        ('Proxy with in-place notes (today’s build)', pim_lead['total'], pim_lead['minutes'], 'proxy'),
        ('Keeper inside the test harness, in-place notes', im['total'], im['cost']['minutes_one_decimal'], 'keeper'),
        ('Proxy, earlier form', keeper['proxy']['million']['total'], pim['compare']['proxy_full_rewrite']['minutes'], 'proxy')]
    tries = ['First try', 'Second try', 'Third try: today’s build']
    proxy_tries_rows = [(tries[r['attempt'] - 1], r['total'], r['minutes'], 'proxy') for r in pim['runs']]
    run_minutes = tw['run_seconds'] / 60
    proxy_time_rows = [
        ('The model server reading', tw['tokens_read_outside_the_cache'] + ' tokens', tw['server_reading_seconds'] / tw['run_seconds'],
         str(round(tw['server_reading_seconds'] / 60)) + ' min', 'proxy'),
        ('The model server writing', tw['tokens_written'] + ' tokens', tw['server_writing_seconds'] / tw['run_seconds'],
         str(round(tw['server_writing_seconds'] / 60)) + ' min', 'proxy')]
    busy_minutes = round((tw['server_reading_seconds'] + tw['server_writing_seconds']) / 60)
    written = tw['written_thousands']
    unnamed_rule_rows = [
        ('Keeper with in-place notes', ur['both_streams']['keeper'], 'the rule stayed in its notes', 'keeper'),
        ('Retrieval only', ur['both_streams']['retrieval'], 'no search led back to the rule', 'baseline')]
    ps = pi['second_million_stream']
    two_stream_rows = [
        (label + ' stream, ' + str(fraction_parts(total)[1]) + ' questions', f'{float(minutes):.1f}' + ' min',
         fraction_parts(total)[0] / fraction_parts(total)[1], total, tone)
        for label, total, minutes, tone in [
            ('Keeper, first', im['total'], im['cost']['minutes_one_decimal'], 'keeper'),
            ('Keeper, second', sm['total'], sm['minutes'], 'keeper'),
            ('Proxy, first', pim_lead['total'], pim_lead['minutes'], 'proxy'),
            ('Proxy, second', ps['total'], ps['minutes'], 'proxy')]]
    inplace_time_rows = [
        ('1,000,000 tokens, full notes rewrite', keeper['million']['total'], ip['full_rewrite']['million']['minutes'], 'baseline'),
        ('1,000,000 tokens, in-place notes', im['total'], im['cost']['minutes_one_decimal'], 'keeper'),
        ('First 240K stream, full notes rewrite', fr[0]['total'], fr[0]['minutes'], 'baseline'),
        ('First 240K stream, in-place notes', ip_lead[0]['total'], ip_lead[0]['minutes'], 'keeper'),
        ('Second 240K stream, full notes rewrite', fr[1]['total'], fr[1]['minutes'], 'baseline'),
        ('Second 240K stream, in-place notes', ip_lead[1]['total'], ip_lead[1]['minutes'], 'keeper')]
    hard = {r['arm']: r for r in ro['hard_stream']['runs']}
    harder_rows = [(label, hard[arm]['total'], hard[arm]['minutes'] + ' min' + extra, tone) for label, arm, extra, tone in [
        ('Notes and retrieval, first run', 'N32ur', '', 'keeper'),
        ('Notes and retrieval, after the fixes', 'N32sr', '', 'keeper'),
        ('Notes only, earlier version', 'N32z', '', 'baseline'),
        ('Notes only, voided run', 'N32s', ' · raw score', 'baseline')]]
    assert [hard[arm]['status'] for arm in ('N32ur', 'N32sr', 'N32z', 'N32s')] == ['Finished, graded'] * 3 + ['Finished, void']
    assert [hard[arm]['with_retrieval'] for arm in ('N32ur', 'N32sr', 'N32z', 'N32s')] == [True, True, False, False]
    retrieval_million_rows = [
        ('Keeper, in-place notes', im['probes'], str(im['cost']['minutes_one_decimal']) + ' min, whole run', 'keeper'),
        ('Retrieval only', rm['probes'], str(rm['minutes_to_reach_the_final_item']) + ' min to the final questions', 'baseline')]
    plain = {r['id']: r for r in next(x for x in dp['rounds'] if x['id'] == 'round2')['rows'] if r['cap'] == '32K'}
    summary = dp['summary_baseline']['row']
    summary_rows = [(label, r['total'], r['minutes'] + ' min', tone) for label, r, tone in [
        ('Archive + search', plain['B32ira'], 'keeper'), ('Window-drop', plain['W32'], 'baseline'),
        ('Summarise at 75%', summary, 'baseline'), ('Tool-output compaction', plain['P32'], 'baseline')]]
    retention_rows = [(label, score(by_id[key]), by_id[key]['time'], tone) for key, label, tone in [
        ('ret-archive-119k', 'State + archive + search', 'keeper'),
        ('ret-summary-119k', 'Summaries', 'baseline'), ('ret-table-119k', 'Current-state table only', 'baseline')]]
    compact = table(['Approach', 'What stays available', 'Correct'], [
        ['Keeper', 'Notes, protected rules, searchable originals', k0['total']],
        ['Keeper proxy', 'The same policy in a small program between client and server', pi_lead[0]['total']],
        ['Retrieval only (RAG)', 'Every sentence in a store; the best matches shown with each question; no notes', rb0['total']],
        ['Agent-managed archive', 'A state table, pinned rules, searchable originals', rules['B32iraR']['total']],
        ['Window-drop', 'Recent turns only', rules['W32']['total']],
        ['Tool-output compaction', 'Short stubs in place of old tool output', rules['P32']['total']],
    ], 'The same 240K rule-and-method stream, seed 0, 32K live context', 'ctx-quick-table')
    for label, target in [('Keeper', 'context-distance.html#keeper'), ('Keeper proxy', 'context-distance.html#proxy-now'),
                          ('Retrieval only (RAG)', 'context-distance.html#retrieval'),
                          ('Agent-managed archive', 'context-distance.html#rule'), ('Window-drop', 'context-distance.html#methods'),
                          ('Tool-output compaction', 'context-distance.html#methods')]:
        compact = compact.replace('<th scope="row">' + label + '</th>', '<th scope="row">' + link('@/' + target, label) + '</th>')
    def time_plot(title, choices, caption):
        rows = [(label, by_id[key]) for key, label in choices]
        values = [float(r['time'].removesuffix(' min')) for _, r in rows]
        ceiling = math.ceil(max(values) / 10) * 10
        bars = ''
        for (label, row), value in zip(rows, values):
            bars += ('<div class="ctx-score-row"><div class="ctx-score-label">' + escape(label)
                     + '<span>' + score(row) + ' correct</span></div><div class="ctx-score-track" aria-hidden="true">'
                     + '<span class="ctx-score-fill ctx-score-proxy" style="width:'
                     + f'{100 * value / ceiling:.4f}' + '%"></span></div><strong>' + f'{value:g}' + ' min</strong></div>')
        return ('<figure class="ctx-score-plot"><figcaption><strong>' + escape(title)
                + '</strong><span>Total elapsed time · shorter is faster · scale 0–' + str(ceiling)
                + ' minutes</span></figcaption>' + bars + '<p class="ctx-table-note">' + escape(caption) + '</p></figure>')

    return {
        'QUICK_COMPARISON': compact,
        'KEEPER_SCORE_PLOT': score_plot('Remembering facts and following the original rules', short_rows,
            'Same 240K-token stream, seed 0, 32K live budget. One run per method; total elapsed time. The keeper, retrieval and pinned-rule arms used a larger server ceiling, while keeping the same live budget. The in-place and retrieval rows were measured later on the same stream. Retrieval alone matches the keeper here because the rule names the counters it covers; a rule that names nothing was never found by retrieval alone, ' + ur['both_streams']['retrieval'].replace('/', ' of ') + ' on a later test.'),
        'MILLION_SCORE_PLOT': score_plot('One million tokens, a 32K working window', million_rows,
            'Same stream and seed: 203 questions during the task, plus 25 at the end. One run per bar; all four finished and were graded by the task’s own checker. The keeper runs inside the test harness. The proxy is a separate small program between an ordinary agent and the model server. In-place notes keep the same notes at lower cost: the model writes down only what changed, and nothing is read twice. On a second million-token stream the keeper with in-place notes answered ' + sm['total'].replace('/', ' of ') + ' and the proxy ' + ps['total'].replace('/', ' of ') + '. Qwen3.8-27B on two B70 GPUs; total elapsed time.'),
        'PROXY_NOW_PLOT': minutes_plot('One million tokens: the proxy has caught up with the lab version', proxy_now_rows,
            'Same 1,000,000-token stream and the same ' + str(fraction_parts(pim_lead['total'])[1]) + ' questions, 32K live window, one run per bar. The proxy serves an agent that sends its whole history and manages nothing. One model, one machine, one synthetic family of tasks.'),
        'PROXY_TRIES_PLOT': minutes_plot('Three tries on the million-token stream', proxy_tries_rows,
            'The proxy with in-place notes on the same stream, one run per bar, in the order they were made. The bar set before the first try was ' + str(pim['bar_minimum_right']) + ' or more right in under ' + str(pim['bar_under_minutes']) + ' minutes: the first try missed it on both counts. One more build, between the first and second tries, failed its check on the 240K streams and was not run at this size.'),
        'PROXY_TIME_PLOT': bar_plot('The model server’s share of the run', 'Minutes of the whole run · scale 0–' + str(round(run_minutes)) + ' minutes',
            proxy_time_rows,
            'The proxy’s million-token run with today’s build; minutes are rounded. Together the two bars are about ' + str(busy_minutes) + ' of the ' + str(round(run_minutes)) + ' minutes (' + str(tw['server_percent']) + '%). Reading: the stream once, and 1.3 million tokens more in turns re-read after each cut, search results and note-taking. Writing: ' + f"{written['notes']},000" + ' tokens of notes, ' + f"{written['the_agents_own_turns']},000" + ' in the agent’s own turns and ' + f"{written['search_rounds']},000" + ' while searching. From the server’s own counters.'),
        'UNNAMED_RULE_PLOT': score_plot('A rule that names nothing: how often was it applied?', unnamed_rule_rows,
            'The ' + str(fraction_parts(ur['both_streams']['keeper'])[1]) + ' questions on two 240K streams whose right answer is the word “overdrawn”. One run per stream for each method, 32K live window.'),
        'TWO_STREAM_PLOT': bar_plot('Two million-token streams: the keeper inside the harness, and the proxy', 'Share of the questions answered correctly · scale 0–100% · longer is better',
            two_stream_rows,
            'In-place notes, 32K live window, one run per bar, each graded by the task’s own checker. The two streams have different numbers of questions, so each bar is a share of its own stream. Together: the keeper '
            + sm['both_streams']['total'].replace('/', ' of ') + ' (' + str(sm['both_streams']['percent']) + '%), the proxy '
            + ps['both_streams']['total'].replace('/', ' of ') + ' (' + str(ps['both_streams']['percent_one_decimal'])
            + '%). The keeper’s second run used the note checks of the morning of 2026-10-12; the proxy’s second run, that evening, was the first text the two fitted check rules had not seen.'),
        'INPLACE_TIME_PLOT': minutes_plot('The same notes, with nothing read twice', inplace_time_rows,
            'Same streams and questions for both forms, 32K live window, one run per bar. All three in-place runs are the version with every check.'),
        'HARDER_SCORE_PLOT': score_plot('A harder stream: the keeper with and without retrieval', harder_rows,
            'Same harder 240K stream and seed, 32K live window, 50 questions, one run per bar. The 43 is a raw score: the grader voided that run because its answers file held one key that was not asked for. Retrieval alone is not a bar: it was stopped after 20.5 minutes with 10 of its first 15 answers right. One more notes-only run got stuck in a loop and scored 3; it is the first lesson below.'),
        'RETRIEVAL_SCORE_PLOT': score_plot('One million tokens: the 203 questions asked along the way', retrieval_million_rows,
            'Same stream and seed, 32K live window. The 25 final questions are left out: the keeper answered all 25, and the retrieval run answered none of them, a limit of that build. The retrieval run was stopped at 59 minutes and is scored from the answers it had written by then; it is not a finished, graded run.'),
        'SUMMARY_SCORE_PLOT': score_plot('When old text leaves the window', summary_rows,
            'Same plain 240K-token history, seed 0, 32K budget; 29 questions. This test has no standing-rule questions and is separate from the keeper comparison.'),
        'RETENTION_SCORE_PLOT': score_plot('Can it still answer about the past?', retention_rows,
            'Same 119K-token narrative history, seed 0, 32K budget; 36 final questions, including 12 about earlier text. Total elapsed time.'),
        'CLM_TIME_PLOT': time_plot('Less time maintaining the same ledger', [
            ('paper', 'Original self-editing'), ('summary', 'Summaries'),
            ('improved-121k-s0', 'First revision'), ('improved-light-121k-s0', 'Checked updates, less thinking'),
            ('files-plain', 'Files + code')],
            '121K ledger, seed 0; 24 final questions. Original self-editing lost five batches in a harness rollback. External data files were allowed only in the files arm. Different permissions and harnesses are part of this comparison.'),
        'QUOTED_TIME_PLOT': time_plot('Reading prose, then keeping the totals', [
            ('read-improved-1m', 'Read-mode self-editing'), ('quoted-1m', 'Quoted events + code')],
            'Same million-token narrative task, seed 0, 32K budget; 24 final questions. These are the original matched runs; fixed-checker reruns and intermediate-state checks are recorded in the detailed results.'),
    }


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
    dp_inplace = {r['seed']: r for r in dp_keeper['in_place']['rows'] if r['headline']}
    dp_retrieval = {r['seed']: r for r in d['distance_probes']['retrieval_only']['rows'] if r['build'] == 'final'}
    dp_hard = d['distance_probes']['retrieval_only']['hard_stream']
    dp_hard_graded = [r for r in dp_hard['runs'] if r['status'] == 'Finished, graded']
    dp_hard_second = next(r for r in dp_hard['second_seed']['runs'] if r['status'] == 'Finished, graded')
    dp_proxy_inplace = dp_keeper['proxy']['in_place']
    dp_proxy_lead = {r['seed']: r for r in dp_proxy_inplace['rows'] if r['headline']}
    dp_proxy_million = next(r for r in dp_proxy_inplace['million']['runs'] if r['headline'])
    dp_second_million = dp_keeper['in_place']['second_million_stream']
    dp_unnamed_rule = d['distance_probes']['retrieval_only']['unnamed_rule']
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
         + ' on two 240K streams at a 32K window; ' + dp_keeper['million']['total'] + ' over 1,000,000 tokens (one stream).'],
        ['Keeper with in-place notes', 'The same keeper with cheaper notes. The note-taking call reuses what the server has already read, the model '
         'writes only what changed, and the window is then emptied down to the task and the notes. ' + dp_inplace[0]['total'] + ' and '
         + dp_inplace[1]['total'] + ' on the two 240K streams in ' + dp_inplace[0]['minutes'] + ' and ' + dp_inplace[1]['minutes']
         + ' minutes; ' + dp_keeper['in_place']['million']['total'] + ' over 1,000,000 tokens in '
         + str(dp_keeper['in_place']['million']['cost']['minutes']) + ' minutes, against ' + str(dp_keeper['million']['cost']['minutes'])
         + ' for the full rewrite. On a second 1,000,000-token stream, ' + dp_second_million['total'] + ' in '
         + str(round(dp_second_million['minutes'])) + ' minutes.'],
        ['Keeper as a proxy, with in-place notes', 'The same notes kept by a small program in front of the model server, for an agent '
         'that sends its whole history and manages nothing. ' + dp_proxy_lead[0]['total'] + ' and ' + dp_proxy_lead[1]['total']
         + ' on the two 240K streams in ' + dp_proxy_lead[0]['minutes'] + ' and ' + dp_proxy_lead[1]['minutes'] + ' minutes; '
         + dp_proxy_million['total'] + ' over 1,000,000 tokens in ' + str(round(dp_proxy_million['minutes']))
         + ' minutes (one stream). Not yet run under a real coding or research agent.'],
        ['Retrieval only (RAG)', 'No notes. Every sentence goes into a store, and the best matches are shown when questions arrive. '
         + dp_retrieval[0]['total'] + ' and ' + dp_retrieval[1]['total'] + ' on the two 240K streams; over 1,000,000 tokens '
         + d['distance_probes']['retrieval_only']['million']['probes'] + ' of the questions asked along the way, and the final 25 not answered '
         '(a stopped run). A rule that names nothing it covers was never found: ' + dp_unnamed_rule['both_streams']['retrieval']
         + ', against ' + dp_unnamed_rule['both_streams']['keeper'] + ' for the keeper.'],
        ['Keeper with retrieval added', 'The keeper with in-place notes, and the best-matching stored sentences shown with each set of '
         'questions. On a harder 240K stream, ' + ' and '.join(r['total'] for r in dp_hard_graded if r['with_retrieval']) + ', against '
         + ' and '.join(r['total'] for r in dp_hard_graded if not r['with_retrieval']) + ' with notes only (one seed, exploratory); '
         + dp_hard_second['total'] + ' on a second seed, where the run without retrieval did not finish. On the '
         'second ordinary 240K stream, ' + dp_hard['on_the_second_ordinary_240k_stream']['total'] + ' in '
         + dp_hard['on_the_second_ordinary_240k_stream']['minutes'] + ' minutes. Not yet run over 1,000,000 tokens.']]
    methods = table(['Approach', 'What changes, and the result'], method_rows, 'Context approaches compared', 'ctx-summary-table ctx-other-table')
    for label, path in [('Bigger active window', 'context-large-window.html'), ('Exact prefix cache', 'context-prefix-cache.html'), ('Suffix cache reuse (CLM paper)', 'context-reuse.html'),
                        ('Files, no context editing', 'context-files-code.html'), ('Files, editing available', 'context-files-editing.html'), ('CLM self-editing', 'context-self-editing.html'),
                        ('Periodic summaries', 'context-summaries.html'), ('Revised self-editing', 'context-revised-editing.html'), ('Remove old thinking', 'context-drop-thinking.html'),
                        ('Park a cache on disk', 'context-cache-parking.html'), ('Stream active cache from disk/RAM', 'context-cache-offload.html'), ('CPU text cleanup', 'context-cleanup.html'),
                        ('Archive and search', 'context-retention.html'), ('Compaction (stub old tool output)', 'context-distance.html'),
                        ('Keeper (managed from outside the agent)', 'context-distance.html#keeper'),
                        ('Keeper with in-place notes', 'context-distance.html#faster'),
                        ('Keeper as a proxy, with in-place notes', 'context-distance.html#proxy-now'),
                        ('Retrieval only (RAG)', 'context-distance.html#retrieval'),
                        ('Keeper with retrieval added', 'context-distance.html#harder')]:
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
    replacements.update(visual_replacements(by_id, d['distance_probes']))
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
        ('context.html', 'Working past the context limit', 'Compare ways to preserve facts and keep long tasks moving.', None),
        ('context-questions.html', 'Longer context, plain answers', 'Common questions about memory, disk, speed and what “unlimited” can mean.', None),
        ('context-methods.html', 'Ways to work with more context', 'A visual guide to keeper, self-editing, files and quoted events.', None),
        ('context-files.html', 'Keep records in files', 'A long task can use a small working context.', 'context-methods.html'),
        ('context-cache.html', 'Read once, reuse the unchanged start', 'Prefix caching reduces the wait before an answer; long-context generation still has a cost.', 'context-methods.html'),
        ('context-memory.html', 'Disk and GPU memory', 'Text files, saved numerical state and active offloading solve different problems.', 'context-methods.html'),
        ('context-clm.html', 'Let the model edit its own context', 'How self-editing differs from keeper, summaries and state files.', 'context-methods.html'),
        ('context-model-tools.html', 'How the model uses files', 'Follow a tool request from the GPU model to disk and back, with the actual prompt sources.', 'context-methods.html'),
        ('context-clm-vs-files.html', 'CLM and external files', 'Compare live context editing, an automatically loaded state file and a separate archive.', 'context-methods.html'),
        ('context-model-choice.html', 'Choosing a context model', 'What Qwen3.8-27B solved with tools, and how a stronger or trained model might help.', 'context-methods.html'),
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
        ('context-distance.html', 'Past the window: what the agent still remembers', 'Facts, old values and standing rules across histories of up to a million tokens.', 'context-methods.html'),
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
