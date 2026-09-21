"""Retrospective PERSON-span coverage from frozen token predictions; no inference."""
import os
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '1'
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/privacy-spans/result.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze():
    prior = ROOT / 'results/practical-privacy'
    replay = ROOT / 'results/practical-replay/privacy'
    source = ROOT / 'experiments/.cache/practical/echr_test.json'
    protocol = read(prior / 'protocol.json')
    summary = read(replay / 'result.json')
    diagnostics = read(replay / 'diagnostics.json')
    expected = read(replay / 'records.json')
    comparison = read(replay / 'comparison.json')
    assert sha(source) == protocol['data_sha256']['test']
    assert comparison['exact_same_outcomes']
    assert sha(replay / 'numerical-predictions.npz') == comparison['numerical_sha256']
    assert summary == read(prior / 'result.json')
    assert [r['id'] for r in diagnostics] == protocol['ids']['test']
    assert [r['id'] for r in expected] == protocol['ids']['test']
    docs = {d['doc_id']: d for d in read(source)}
    keys = ['fully_marked', 'partly_marked', 'unmarked', 'beyond_prefix', 'crosses_prefix', 'no_token_overlap']
    totals = dict.fromkeys(keys, 0)
    rows = []
    token_counts = dict(tp=0, fn=0, fp=0)
    with np.load(replay / 'numerical-predictions.npz', allow_pickle=False) as saved:
        for record, original in zip(diagnostics, expected):
            doc = docs[record['id']]
            tokens = list(re.finditer(r'\w+|[^\w\s]', doc['text']))[:1200]
            assert tokens
            scores = saved[record['probability_array']][:, 1]
            assert len(scores) == len(tokens) == original['tokens']
            assert np.isfinite(scores).all()
            marked = scores >= summary['threshold']
            # Same occurrence offsets from different annotators count once.
            spans = sorted({(m['start_offset'], m['end_offset'])
                            for annotation in doc['annotations'].values()
                            for m in annotation['entity_mentions'] if m['entity_type'] == 'PERSON'})
            labels = np.array([any(t.start() < end and t.end() > start for start, end in spans) for t in tokens])
            assert np.array_equal(labels, saved[record['label_array']])
            measured = dict(tp=int((labels & marked).sum()), fn=int((labels & ~marked).sum()), fp=int((~labels & marked).sum()))
            assert all(measured[k] == original['metrics'][k] for k in measured)
            for key in measured:
                token_counts[key] += measured[key]
            counts = dict.fromkeys(keys, 0)
            occurrences = []
            for start, end in spans:
                assert 0 <= start < end <= len(doc['text'])
                indices = [i for i, token in enumerate(tokens) if token.start() < end and token.end() > start]
                if start >= tokens[-1].end():
                    status = 'beyond_prefix'
                elif end > tokens[-1].end():
                    status = 'crosses_prefix'
                elif not indices:
                    status = 'no_token_overlap'
                elif all(marked[indices]):
                    status = 'fully_marked'
                elif any(marked[indices]):
                    status = 'partly_marked'
                else:
                    status = 'unmarked'
                counts[status] += 1
                occurrences.append(dict(start=start, end=end, status=status, observed_tokens=len(indices), marked_tokens=int(marked[indices].sum())))
            for key in keys:
                totals[key] += counts[key]
            rows.append(dict(id=record['id'], person_occurrences=len(spans), counts=counts,
                             all_person_occurrences_marked=bool(spans) and counts['fully_marked'] == len(spans),
                             has_incompletely_marked_occurrence=counts['fully_marked'] < len(spans), occurrences=occurrences))
    assert all(token_counts[k] == summary['learned'][k] for k in token_counts)
    within = sum(totals[k] for k in ['fully_marked', 'partly_marked', 'unmarked'])
    dependencies = [source, prior / 'protocol.json', prior / 'result.json', replay / 'result.json',
                    replay / 'diagnostics.json', replay / 'records.json', replay / 'comparison.json', replay / 'numerical-predictions.npz']
    return dict(scope='Retrospective error analysis of frozen predictions, not new inference, new holdout, or changed masking.',
                source_sha256=sha(Path(__file__)),
                dependencies={p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies},
                definition='Distinct (start,end) PERSON occurrences unioned across annotators; overlapping nonidentical spans remain separate. Repeated names at different offsets remain separate. Fully marked means every regex token intersecting an entirely observed span has a positive prediction. Whitespace is not scored. This is token-mask coverage, not safe anonymization.',
                threshold=summary['threshold'], prefix_tokens=1200, documents=len(rows),
                person_occurrences=sum(totals.values()), within_prefix=within, counts=totals,
                fully_marked_within_prefix_rate=totals['fully_marked'] / within,
                documents_all_person_occurrences_marked=sum(r['all_person_occurrences_marked'] for r in rows),
                documents_with_incompletely_marked_occurrence=sum(r['has_incompletely_marked_occurrence'] for r in rows),
                documents_without_person_occurrences=sum(r['person_occurrences'] == 0 for r in rows),
                original_token_confusion_recomputed=token_counts, records=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Recompute and compare without overwriting evidence.')
    args = parser.parse_args()
    result = analyze()
    if args.check:
        assert read(OUT) == result, 'Saved analysis differs.'
    else:
        assert not OUT.exists(), 'Preserve completed result; use --check.'
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ['documents', 'person_occurrences', 'within_prefix', 'counts', 'fully_marked_within_prefix_rate', 'documents_with_incompletely_marked_occurrence']}))
