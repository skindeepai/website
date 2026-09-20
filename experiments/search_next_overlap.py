"""Post-hoc query-text overlap audit; preserve the sealed 200-query primary run."""
import search_next as study
import math
from pathlib import Path


def normalized(text):
    return ' '.join(text.casefold().split())


def main():
    import pyarrow.parquet as parquet
    queries = {r['_id']: r['text'] for r in parquet.read_table(study.base.CACHE / 'queries.parquet', use_threads=False).to_pylist()}
    old = study.read(study.base.OUT / 'protocol.json')
    learned = study.read(study.base.OUT / 'learned/protocol.json')
    records = study.read(study.OUT / 'predictions.json')
    partitions = {'fit': learned['train_ids'], 'development': learned['development_ids'], 'old_test': old['query_ids']}
    overlaps = []
    for row in records:
        for partition, ids in partitions.items():
            matches = [qid for qid in ids if normalized(queries[qid]) == normalized(row['text'])]
            if matches:
                overlaps.append({'id': row['id'], 'partition': partition, 'matches': matches, 'text': row['text']})
    excluded = sorted({r['id'] for r in overlaps})
    kept = [r for r in records if r['id'] not in excluded]
    metrics = {}
    for method in study.METHODS:
        metrics[method] = {key: sum(r['methods'][method][key] for r in kept) / len(kept) for key in ['hit@1', 'recall@5', 'ndcg@10', 'mrr@10']}
        metrics[method]['correct_first'] = sum(r['methods'][method]['hit@1'] for r in kept)
        metrics[method]['reranked_queries'] = sum(r['methods'][method]['reranked'] for r in kept)
    changes = {}
    for name, rows in [('primary', records), ('text_deduplicated', kept)]:
        gains = sum(r['methods']['fusion']['hit@1'] > r['methods']['bm25']['hit@1'] for r in rows)
        losses = sum(r['methods']['fusion']['hit@1'] < r['methods']['bm25']['hit@1'] for r in rows)
        n = gains + losses
        # Exact paired sign/McNemar binomial test, descriptive post-hoc (not multiplicity corrected).
        p = min(1., 2 * sum(math.comb(n, k) for k in range(min(gains, losses) + 1)) / 2 ** n)
        changes[name] = {'fusion_gains_vs_bm25': gains, 'fusion_losses_vs_bm25': losses, 'two_sided_exact_p': p,
                         'scope': 'Exploratory paired sign test after observing several method results; no multiple-comparison correction or population guarantee.'}
    study.write('text-overlap-audit.json', {'scope': 'Post-hoc sensitivity analysis after a reviewer identified duplicate claim text across official split IDs. No methods or settings changed.',
                                           'normalization': 'Unicode casefold, split whitespace and rejoin with single spaces. No semantic paraphrase matching.',
                                           'primary_queries': len(records), 'overlaps': overlaps, 'excluded_ids': excluded,
                                           'sensitivity_queries': len(kept), 'sensitivity_metrics': metrics, 'paired_changes': changes,
                                           'candidate_recall@20': sum(r['candidate_recall'] for r in kept) / len(kept),
                                           'queries_with_relevant_candidate': sum(r['candidate_has_relevant'] for r in kept),
                                           'predictions_sha256': study.base.digest(study.OUT / 'predictions.json'),
                                           'script_sha256': study.base.digest(Path(__file__)),
                                           'limits': 'IDs are disjoint but text need not be. Does not detect similar claims, shared papers, or model pretraining contamination. Original protocol/200 predictions unchanged. Timing table remains the original 200 queries.'})
    print('Overlaps:', overlaps)
    print('Sensitivity queries:', len(kept), 'metrics:', metrics)
    print('Paired:', changes)


if __name__ == '__main__':
    main()
