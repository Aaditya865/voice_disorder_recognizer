"""Build submission documentation from saved evidence, without fitting models."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts_v3'

def table(frame):
    headers=list(frame.columns)
    def value(x):
        if isinstance(x,float):return f'{x:.6f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(value(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)])

def build():
    release=json.loads((OUT/'release.json').read_text());summary=json.loads((OUT/'nested_summary.json').read_text())
    selection=release['selection'];metrics=pd.read_csv(OUT/'nested_metrics.csv');audit=pd.read_csv(OUT/'dataset_audit.csv')
    candidate=pd.read_csv(OUT/'candidate_comparison.csv');subgroups=pd.read_csv(OUT/'subgroup_metrics.csv')
    foldinfo=json.loads((OUT/'nested_folds.json').read_text())
    folds=pd.DataFrame([{'outer_fold':f['fold'],'chosen':f['selection']['winner']['id'],
        'members':', '.join(f['selection']['members']),'threshold':f['selection']['threshold']} for f in foldinfo])
    benchmark=json.loads((OUT/'runtime_benchmark.json').read_text()) if (OUT/'runtime_benchmark.json').exists() else {'status':'not yet verified'}
    lines=['# Voice Disorder Recognizer — two-source development experiment','',
        '**Academic project, not a medical diagnosis.** This revision has no genuinely untouched external test.',
        '','## Dataset and label audit','',table(audit),'',
        'One record per SVD participant. VOICED record IDs are participant proxies; cross-session identity cannot be independently reverified. '+
        'VOICED release labels give 57 healthy and 151 pathological records, differing from its website summary. '+
        'Headers and detailed metadata agree on binary status; detailed diagnosis wording can differ. '+
        'All 208 VOICED records, including the previously evaluated 42, are development data in this revision.',
        '','## Selection and evaluation protocol','',
        'Four outer participant-disjoint folds evaluate a procedure using three inner folds. Strata combine source and binary label. '+
        'Fit imputation, scaling, PCA/selection and balancing inside fitting partitions. Separate speakers support temperature calibration and CNN stopping. '+
        'Selection maximizes the mean of source-specific balanced accuracies. Within 0.01 of the best score, prefer GMM, MLP, CNN, then ensemble. '+
        'Outer folds are never used to choose the final artifact; repeat the inner procedure on all development data for final fitting.',
        '',f'Final chosen configuration: **{selection["winner"]["id"]}**. Members: {", ".join(selection["members"])}.',
        f'Comparison models: {selection["best_gmm"]} and {selection["best_dnn"]}.',
        '',table(folds),'',
        'The rows below assess family-selection procedures across outer folds. They do not independently test the final refitted artifact. '+
        'All scores are development estimates; dataset familiarity and source-specific biases remain. Bootstrap intervals resample participant proxies within source and class; they condition on saved predictions and do not measure uncertainty from retraining.',
        '','## Nested model comparison','']
    formatted=[]
    for _,r in metrics.iterrows():
        row={'Procedure':r.role,'Source':r.source,'N':int(r.n)}
        for k in ('balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc'):
            row[k]=f'{r[k+"_mean"]:.6f} +/- {r[k+"_std"]:.6f}'
        formatted.append(row)
    lines += [table(pd.DataFrame(formatted)),'','Means +/- sample standard deviations across four outer folds. Full precision: artifacts_v3/nested_metrics.csv.',
        '','## Pooled outer predictions and confidence intervals','']
    for source,m in summary.items():
        ci=m['confidence_intervals']['balanced_accuracy']
        lines += [f'### {source}','',f'Accuracy {m["accuracy"]:.6f}; balanced accuracy {m["balanced_accuracy"]:.6f} '+
            f'(bootstrap 95% interval {ci[0]:.6f}–{ci[1]:.6f}); macro-F1 {m["macro_f1"]:.6f}; AUC {m["roc_auc"]:.6f}.',
            f'Sensitivity {m["sensitivity"]:.6f}; specificity {m["specificity"]:.6f}; Brier {m["brier"]:.6f}; ECE {m["ece"]:.6f}.',
            f'Confusion counts: TN={m["tn"]}, FP={m["fp"]}, FN={m["fn"]}, TP={m["tp"]}.',
            f'Accepted coverage {m["coverage"]:.6f}; accepted count {m["accepted_count"]}; accepted accuracy {m["accepted_accuracy"]}.','',
            table(pd.read_csv(OUT/f'classification_{source}.csv').rename(columns={'Unnamed: 0':'Class'})),'',
            f'![{source} confusion](artifacts_v3/confusion_{source}.png)','']
    lines += ['## Candidate searches','',
        'Each stage used only its own development partitions. Scores here were used for selection and are not independent test results. '+
        'The final-development rows compare all candidate settings on identical inner folds.', '',
        table(candidate[candidate.stage=='final-development'][['id','source','score','balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc']]),'',
        '## Demographics, errors and interpretation','',table(subgroups),'',
        'Groups with fewer than 20 examples in either class are flagged inconclusive. Differences do not prove a causal demographic shortcut. '+
        'SVD healthy speakers are substantially younger than pathological speakers. Equal source/class weighting does not remove that confounding. '+
        'Inspect development_errors.csv for false positives/negatives; no examples were removed because a model predicted them incorrectly.',
        '','## Uncertainty policy','',
        f'Final threshold: {selection["threshold"]}. A value above 1 means all predictions are uncertain. '+
        'Threshold selection required 80% accepted accuracy, 20% coverage and 20 accepted examples in each source on inner predictions. '+
        'These are selection criteria, not a performance guarantee. Outer-fold coverage and accepted accuracy are reported above; the app uses the final development-selected threshold.',
        '','## Historical external result — preserved','',
        'SVD v2 was tested once on 42 reserved VOICED records: balanced accuracy 0.525, sensitivity 4/30, specificity 11/12. '+
        'It misclassified 26 pathological examples as healthy. Its 0.70 confidence rule accepted 31 cases, only 12 correctly. '+
        'That result is retained unchanged. Revised VOICED development scores cannot be described as an improvement on a fresh external test.',
        '','## Subtype scope','',
        'The unchanged v2 GMM ranks Functional dysphonia group, Laryngitis and Vocal fold paresis. '+
        'It was developed on 185, 73 and 141 SVD speakers respectively, excluding 153 mixed/other cases. '+
        'Its grouped-CV balanced accuracy was 0.489254 +/- 0.095303. No subtype passed the earlier confidence/coverage criteria; all rankings remain exploratory and uncertain.',
        '','## Runtime and software validation','',json.dumps(benchmark,indent=2),'',
        '## Limitations','',
        'This is an academic research demonstration, not clinical screening or diagnosis. There is no new independent final test. '+
        'The same known datasets informed earlier design decisions. Age/sex, microphone and source differences remain potential confounders. '+
        'A short sustained vowel does not uniquely identify an underlying disease. Subtype coverage is incomplete, and calibration may fail under domain shift. '+
        'The search is bounded and does not establish the globally best model. More diverse independently labeled speakers and prospective evaluation are required.',
        '','## Sources','',
        '- SVD: https://zenodo.org/records/16874898',
        '- VOICED: https://physionet.org/content/voiced/1.0.0/',
        '- Gupta et al. (2026), scoping review: https://doi.org/10.3389/fdgth.2026.1800132',
        '- Saved experiment configuration, source snapshots and predictions: artifacts_v3/.','']
    (ROOT/'V3_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':build()
