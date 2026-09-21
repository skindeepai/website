"""Summarize completed local checks without calling broader studies complete."""
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]

def main():
    r=read('results/preference-sampling/result.json')['records']
    lines=['# Preference follow-ups','',
        'These are completed local synthetic checks of specific gaps in the earlier pilots. They do not complete human preference, real-generator, privacy or application validation. No production sampling policy was changed.','',
        '## Does the actual browser sampler learn more efficiently?','',
        'The experiment uses the actual browser training core and extracted sampling function: 16 settings, 380 candidates and a shuffled batch of five predicted likes, four uncertain choices and three random choices. The UI refreshes its queue every six ratings, so only half of each 12-item batch is rated; the experiment reproduces that behavior.','',
        'Random and uncertainty controls receive the same candidate pools and consume the same random-number budgets. All policies start from the same six examples. We retrain after each rating and record 12, 24, 48, 72 and 96 ratings across five fixed seeds. Evaluation uses 100 separate uniform examples per seed.','',
        'Mean balanced accuracy at 96 ratings (average of positive recall and negative recall):','']
    policies=['random','uncertainty','browser_mixed'];scenarios=['linear','noisy_linear','two_modes']
    names={'random':'Random','uncertainty':'Uncertain examples','browser_mixed':'Browser mixture'}
    lines+=table(['Synthetic preference']+[names[p] for p in policies],[[s]+[f'{100*statistics.mean(v["balanced_accuracy"] for v in r if v["scenario"]==s and v["policy"]==p and v["ratings"]==96):.2f}%' for p in policies] for s in scenarios])
    lines+=['','The mixture is not consistently better. It shows more truly liked examples under the linear rules, but that is different from learning accurately. With noisy ratings, it performs worse than random selection. The linear model also struggles with two separated preferred regions, whatever the sampler.','',
        'Noisy training flips 15% of ratings independently; evaluation asks whether the clean underlying preference was recovered. The second nonlinear rule likes examples where |z0| > 0.55 and |z1| < 0.65. These authored rules and five seeds are not people or confidence intervals.','',
        '### Learning curves: every policy and checkpoint','']
    for scenario in scenarios:
        lines+=['','#### '+scenario,'',*table(['Ratings']+[names[p] for p in policies],[[n]+[f'{100*statistics.mean(v["balanced_accuracy"] for v in r if v["scenario"]==scenario and v["policy"]==p and v["ratings"]==n):.2f}%' for p in policies] for n in [12,24,48,72,96]])]
    lines+=['','[Protocol](../results/preference-sampling/protocol.json), [all checkpoint predictions and weights](../results/preference-sampling/result.json), [every rated example](../results/preference-sampling/runs.json), [independent evaluation fixtures](../results/preference-sampling/fixtures.json), [runner](../experiments/preference_sampling.cjs). A recorded first checkpoint at 90% is not an exact label requirement or a guarantee of maintaining that score.','',
        '## Can selecting a high-scoring candidate help?','']
    follow=read('results/preference-followups/result.json')['records'];selection=[v for v in follow if v['experiment']=='P05']
    lines+=table(['Same 64-candidate pool','Mean true utility'],[['Random candidate',f'{statistics.mean(v["random_utility"] for v in selection):.3f}'],['Highest predicted preference',f'{statistics.mean(v["reranked_utility"] for v in selection):.3f}']])
    improved=sum(v['reranked_utility']>v['random_utility'] for v in selection)
    lines+=['',f'Reranking improved {improved} of {len(selection)} seeds and worsened {sum(v["reranked_utility"]<v["random_utility"] for v in selection)}. Its worst utility change was {min(v["reranked_utility"]-v["random_utility"] for v in selection):.3f}; the average improvement is not a guarantee.']
    lines+=['','Higher utility is better. Each method sees the same 64 new candidates after 64 training labels; the first independently uniform candidate is the random control. This checks selection quality at equal candidate count, not image-generation time or human taste. The true utility rewards proximity to a declared center. All five seed outcomes and candidates are saved; the earlier optimization failure remains in the [original synthetic results](../results/synthetic/result.json).','',
        '## What about recurring contexts?','',
        'The supplied context alternates A, B, A, B, with opposite preferences and 24 new ratings per phase. We compare one combined-history model, a reset at each phase, and separate retained models for the known context IDs. The context is given to the system; it does not infer a mood.','']
    methods=['all_history','phase_reset','known_context']
    lines+=table(['Phase']+methods,[[str(phase)]+[f'{statistics.mean(v["correct"] for v in follow if v["experiment"]=="P07" and v["phase"]==phase and v["method"]==method):.1f} / 100' for method in methods] for phase in [1,2,3,4]])
    lines+=['',*table(['Return to A after final B','Correct / 100'],[[method,f'{statistics.mean(v["correct"] for v in follow if v["experiment"]=="P07_retention" and v["method"]==method):.1f}'] for method in methods]),'',
        'These are five-seed averages on the same 100 held-out points per seed. Context-specific models preserve earlier context data and use more model storage; this is an explicit design difference, not a matched-memory comparison. No real sessions, inferred context or human retention is measured.','',
        '## Do explicit constraints prevent a conflicting edit?','']
    constraints=[v for v in follow if v['experiment']=='P10']
    lines+=table(['Known requirement: z0 ≤ 0','Violations / 5'],[['Unconstrained preference maximum',sum(v['unconstrained_violation'] for v in constraints)],['Explicit coordinate cap',sum(v['constrained_violation'] for v in constraints)]])
    lines+=['','These cases deliberately train a preference that conflicts with the known requirement. The coordinate cap enforces the declared inequality by construction. It tests a mechanism, not a learned safety model, unseen constraints or real-world compliance. The earlier fixture had zero violations before either method and could not show this difference.','',
        '[Prospective protocol](../results/preference-followups/protocol.json), [all results](../results/preference-followups/result.json), [runner](../experiments/preference_followups.cjs). Both studies use one CPU compute thread and the existing browser training core. To reproduce, use a separate checkout; completed result files are protected from overwriting. Run each script with `--prepare` first, then without it.']
    lines+=['','## Review','',
        'A separate agent in this work session reconstructed all 22,500 sampler evaluation decisions, 225 checkpoints and 4,320 selected training examples, including the six-rating queue refresh. It also replayed all 85 candidate-selection, context and constraint records and verified the source/protocol hashes. The calculations matched. This is an internal code/artifact audit, not external replication or evidence about people.']
    (ROOT/'docs/preference-followups.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    status=read('content/research-evidence.json')
    ledger=['# Research evidence status','',
        'The website now answers these questions in the [FAQ](../research.html). This ledger preserves all 29 original protocol IDs and distinguishes completed local checks from broader unfinished studies. A negative result closes the recorded run, not the entire research question.','',
        'Completed work includes actual decision timing, model fallback, replay checks, coordinate pilots, maze actions and the new exact sampler / candidate / context / constraint checks. Human studies, privacy attacks, real preference generators, audio/video context, matched accelerator tests and end-to-end GUI tasks have not been completed. They require evidence that these local fixtures do not supply.','',
        *table(['ID','Evidence status','Observed','Still unestablished'],[[i,v['status'],v['observed']+(' '+' '.join(f'[Record {j+1}](../{n})' for j,n in enumerate(v['evidence'])) if v['evidence'] else ''),v['remaining']] for i,v in status.items()]),'',
        '[Detailed study designs](../EXPERIMENTS.md). The untested items are retained as scope limits, not advertised as forthcoming features or silently marked done.']
    # Later experiment lessons are maintained below the generated protocol table.
    status_path=ROOT/'docs/research-status.md'
    marker='## Additional checks and lessons'
    previous=status_path.read_text(encoding='utf-8') if status_path.exists() else ''
    if marker in previous:
        ledger+=['',marker+previous.split(marker,1)[1].rstrip()]
    status_path.write_text('\n'.join(ledger)+'\n',encoding='utf-8',newline='\n')
    print('Updated preference follow-ups and the 29-item evidence ledger.')

if __name__=='__main__':main()
