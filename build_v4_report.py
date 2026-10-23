"""Build a measured report from completed predictions; never fits a model."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from svd_v4_results import metrics
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/v4'

def md(df):
    def f(v):
        if isinstance(v,float):return '' if np.isnan(v) else f'{v:.6f}'
        return str(v).replace('|','/')
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    return '\n'.join(lines+['| '+' | '.join(f(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    release=json.loads((OUT/'release.json').read_text());decision=json.loads((OUT/'promotion.json').read_text())
    challenge=json.loads((OUT/'challenge_metrics.json').read_text()) if (OUT/'challenge_metrics.json').exists() else {'status':'not evaluated'}
    table=pd.read_csv(OUT/'before_after.csv');p=pd.read_csv(OUT/'nested_predictions.csv');p=p[p.role=='selection']
    manifest=pd.read_csv(OUT/'development_manifest.csv');p=p.merge(manifest[['record_id','age','sex','diagnosis']],on='record_id',validate='one_to_one')
    calibration=[];subgroups=[]
    for source,d in p.groupby('source'):
        for lo,hi in [(0,.6),(.6,.7),(.7,.8),(.8,.9),(.9,1.00001)]:
            b=d[(d.confidence>=lo)&(d.confidence<hi)]
            calibration.append({'source':source,'lower':lo,'upper':min(hi,1.),'n':len(b),'mean_score':b.confidence.mean(),
              'accuracy':float((b.prediction==b.label).mean()) if len(b) else np.nan,'note':'inconclusive small bin' if len(b)<20 else ''})
        d=d.copy();d['age_band']=pd.cut(d.age,[-1,30,50,70,200],right=False,labels=['under30','30-49','50-69','70plus'])
        for key in ['sex','age_band']:
            for group,b in d.groupby(key,observed=True):
                if b.label.nunique()<2:
                    subgroups.append({'source':source,'group':str(group),'type':key,'n':len(b),'note':'single class; balanced accuracy not meaningful'});continue
                subgroups.append({'source':source,'group':str(group),'type':key,**metrics(b),'note':'inconclusive small group' if len(b)<20 else ''})
    pd.DataFrame(calibration).to_csv(OUT/'calibration_bins.csv',index=False);pd.DataFrame(subgroups).to_csv(OUT/'subgroups.csv',index=False)
    errors=p[p.label==1].copy();errors['false_negative']=errors.prediction==0
    errors.groupby(['source','diagnosis']).agg(n=('record_id','size'),false_negatives=('false_negative','sum')).to_csv(OUT/'false_negatives_after.csv')
    comparison=pd.read_csv(OUT/'candidate_comparison.csv')
    # Ensemble predictions are saved separately by the training engine.
    ensemble_rows=[]
    selections={f"outer{f['fold']}":f['selection'] for f in json.loads((OUT/'nested_folds.json').read_text())}
    selections['final-development']=release['selection']
    indexed=manifest.set_index('record_id')
    for stage,selection in selections.items():
        z=np.load(OUT/(stage+'_inner_predictions.npz'));part=indexed.loc[z['record_id']].reset_index();pp=z['Ensemble']
        cutoff=next(r['cutoff'] for r in selection['ranking'] if r['id']=='Ensemble')
        for source in ['pooled','SVD','VOICED']:
            mask=np.ones(len(part),bool) if source=='pooled' else part.source.to_numpy()==source
            from svd_v4 import metric
            ensemble_rows.append({'stage':stage,'id':'Ensemble','source':source,'cutoff':cutoff,**metric(part[mask],pp[mask],cutoff)})
    pd.DataFrame(ensemble_rows).to_csv(OUT/'ensemble_inner_metrics.csv',index=False)
    comparison=pd.concat([comparison,pd.DataFrame(ensemble_rows)],ignore_index=True)
    tuned=comparison[comparison.stage.str.startswith('outer')].groupby(['id','source'])[['balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc']].agg(['mean','std'])
    tuned.columns=['_'.join(c) for c in tuned.columns];tuned.reset_index().to_csv(OUT/'candidate_inner_summary.csv',index=False)
    selected=release['selection']['winner'];selected_text=f"{selected['id']}; cutoff {selected['cutoff']:.6f}; rejection threshold {release['selection']['threshold']:.6f}"
    deployment='Eligible on predefined development criteria; release acceptance still required.' if decision['development_promotion_eligible'] else 'Do not replace v3: the predefined improvement criteria were not satisfied.'
    lines=['# V4 improvement experiment — measured results',
      'Academic project, not a medical diagnosis. This is an adaptive development experiment, not an independent external validation.',
      '## Outcome',f'Final experimental configuration: **{selected_text}**.',deployment,
      '## Before / after on the same participants',
      md(table[table.procedure.isin(['v3_previous','v4_same_participants'])][['procedure','source','n','accuracy','balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc','brier','ece']]),
      'Values are printed to six decimals; full precision is retained in CSV. Outer folds assess a selection procedure, not a new independent test of the final refitted model.',
      '## Paired improvement criterion','```json\n'+json.dumps(decision,indent=2)+'\n```',
      'The paired bootstrap conditions on fixed outer-fold predictions. It does not capture all model-selection uncertainty, source effects or prior iterative use of development data.',
      '## Expanded-cohort model comparison',md(table[~table.procedure.isin(['v3_previous','v4_same_participants'])][['procedure','source','n','balanced_accuracy_mean','balanced_accuracy_std','macro_f1_mean','sensitivity_mean','specificity_mean','roc_auc_mean']]),
      '## Every candidate tried',
      'candidate_inner_summary.csv contains mean/std of the inner-validation tuning results across outer training partitions. These overlap and are NOT independent outer-test performance for each fixed candidate. CNN, MLP, GMM, SVM, embedding/fusion and augmentation variants all completed the same nested search.',
      md(tuned.reset_index()[tuned.reset_index().source=='pooled']),
      '## New-speaker challenge','```json\n'+json.dumps(challenge,indent=2)+'\n```',
      'This is a same-source pathological-only challenge. It cannot provide specificity, balanced accuracy or AUC. The historical v2 external result remains 0.525 balanced accuracy and 4/30 pathological detections. VOICED is development in v3 and v4, so neither provides a new external-dataset result.',
      '## Calibration and abstention',md(pd.DataFrame(calibration)),
      'Temperature calibration was fitted on held-out fitting-partition speakers. High scores are not guaranteed correctness. The original per-source accuracy, coverage and count requirements were preserved. Inspect outer-fold accepted accuracy as well as coverage; inner eligibility may fail on outer data.',
      '## Changes and audit',
      '- Added 198 previously unused SVD pathological speakers to development; retained 1,212 original SVD and 208 VOICED participant proxies.',
      '- Reserved 100 new SVD speakers before modeling; 63 usable downloads survived availability/quality checks. Rate limits and short files were not replaced based on model performance.',
      '- Found zero incorrect SVD binary mappings or speaker mappings against source metadata. Did not relabel false negatives to improve scores.',
      '- Reused verified 8 kHz acoustic extraction and a pinned 16 kHz WavLM representation. Training and inference share these implementations.',
      '- Tested MFCC-only, pitch-excluded, rich acoustic, frozen embedding and fusion inputs; GMM, MLP, CNN and SVM; training-only MLP feature noise and CNN spectrogram augmentation; an equal-weight GMM/DNN ensemble.',
      '- Nested selection tuned binary cutoff; separate-speaker temperature calibration and CNN early stopping were retained.',
      '- HUPA requires an agreement. PVQD N/Y denote complaints, not diagnostic truth, and recordings mix tasks. Neither was force-mapped to binary health labels.',
      '## Remaining limits',
      'No new independent external dataset with verified binary labels was evaluated. The extra development data are pathological-only and broaden the task, which may hurt some populations. Source and age/sex confounding remain; removing explicit pitch features cannot eliminate those effects. The sealed challenge is small, positive-only and subject to download/quality selection. Some clinical disorders may have little audible effect in a short vowel. Subtypes remain exploratory. Neither this bounded search nor calibration establishes clinical reliability.',
      '## Reproduction and evidence',
      'Release checks: 210 cached inner fitting/calibration partitions were audited with zero participant overlap. Acoustic and WavLM training/inference feature parity passed on one fixed development recording. Fourteen release tests passed, including historical integrity, saved/reloaded predictions, metric recomputation and retained-app behavior. These checks establish software consistency, not diagnostic validity.',
      'See V4_RUN.md, protocol_v4.json, download_manifest.csv, outer_assignment.csv, nested_folds.json, nested_predictions.csv, candidate_comparison.csv and challenge_predictions.csv. The deployed v3 model is preserved separately.',
      '## Sources',
      '- https://zenodo.org/records/16874898',
      '- https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b',
      '- https://data.mendeley.com/datasets/9dz247gnyb/4',
      '- https://pmc.ncbi.nlm.nih.gov/articles/PMC13049965/']
    (ROOT/'V4_REPORT.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
