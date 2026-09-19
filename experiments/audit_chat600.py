"""Recompute the declared extra calibration check without changing the experiment."""
import json
from pathlib import Path
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/chat600'
def upper(k,n):return 1. if n==0 or k==n else float(beta.ppf(.95,k+1,n-k))

def main():
    result=json.loads((OUT/'result.json').read_text());rows=json.loads((OUT/'predictions.json').read_text())
    amendment=json.loads((OUT/'calibration-overlap-amendment.json').read_text());excluded=set(amendment['excluded_calibration_ids'])
    cal=[r for r in rows if r['split']=='calibration' and r['id'] not in excluded]
    n=len(cal);toxic=sum(r['label']==1 for r in cal)
    added=sum(r['full']==r['label'] and r['candidate']!=r['label'] for r in cal)
    missed=sum(r['label']==1 and r['full']==1 and r['candidate']==0 for r in cal)
    early=sum(r['depth']<24 for r in cal);correct=sum(r['full']==r['label'] for r in cal);caught=sum(r['label']==1 and r['full']==1 for r in cal)
    bounds={'added_errors':upper(added,n),'additional_missed_toxic':upper(missed,toxic)}
    checks={'added_errors':bounds['added_errors']<=.01,'additional_missed_toxic':bounds['additional_missed_toxic']<=.05,'coverage':early/n>=.05,'baseline_accuracy':correct/n>=.9,'baseline_toxic_recall':caught/toxic>=.8}
    output={'calibration_queries':n,'toxic':toxic,'excluded_ids':sorted(excluded),'added_errors':added,'additional_missed_toxic':missed,'early_count':early,
            'full_correct':correct,'full_toxic_caught':caught,'upper95':bounds,'checks':checks,'clean_guard_passed':all(checks.values()),
            'original_guard_passed':result['guard_passed'],'accepted':result['guard_passed'] and all(checks.values()),
            'method':'Additional requirement recorded before fitting outcomes: same policy and limits, original guard must also pass. All600testmessages retained.'}
    (OUT/'calibration-integrity.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8');print(json.dumps(output))

if __name__=='__main__':main()
