"""Report completed v4 results and open the sealed positive-only challenge once."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np,pandas as pd,joblib
from sklearn.metrics import confusion_matrix,balanced_accuracy_score,f1_score,classification_report
from threadpoolctl import threadpool_limits
import svd_v4 as v4
import svd_v3 as v3
ROOT=v4.ROOT;OUT=v4.OUT

def metrics(d):
    p=np.c_[1-d.p_pathological,d.p_pathological];m=v3.metric(d.label,p)
    y=d.label.to_numpy();pred=d.prediction.to_numpy();tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    m.update(accuracy=float((y==pred).mean()),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
      macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),sensitivity=float(tp/(tp+fn)) if tp+fn else None,
      specificity=float(tn/(tn+fp)) if tn+fp else None,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n=len(d))
    if 'accepted' in d:
        keep=d.accepted.to_numpy(bool);m.update(coverage=float(keep.mean()),accepted=int(keep.sum()),
            accepted_accuracy=float((y[keep]==pred[keep]).mean()) if keep.any() else None)
    return m

def report():
    release=json.loads((OUT/'release.json').read_text());p=pd.read_csv(OUT/'nested_predictions.csv')
    old=pd.read_csv(ROOT/'artifacts_v3/nested_predictions.csv');old=old[old.role=='selection'].copy()
    old['prediction']=(old.p_pathological>=.5).astype(int)
    now=p[(p.role=='selection')&(p.origin=='previous_development')].copy()
    merged=old.merge(now,on='record_id',suffixes=('_old','_new'),validate='one_to_one')
    assert len(merged)==1420 and (merged.fold_old==merged.fold_new).all()
    rows=[]
    for name,d in [('v3_previous',old),('v4_same_participants',now),('v4_all_development',p[p.role=='selection']),
                   ('v4_GMM',p[p.role=='GMM']),('v4_DNN',p[p.role=='DNN'])]:
        for source in ['pooled','SVD','VOICED']:
            part=d if source=='pooled' else d[d.source==source]
            row={'procedure':name,'source':source,**metrics(part)}
            fs=[metrics(g) for _,g in part.groupby('fold')]
            for k in ['balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc']:
                row[k+'_mean']=float(np.mean([f[k] for f in fs]));row[k+'_std']=float(np.std([f[k] for f in fs],ddof=1))
            rows.append(row)
    table=pd.DataFrame(rows);table.to_csv(OUT/'before_after.csv',index=False)
    # Paired, source/class-stratified participant bootstrap. Conditional on fixed OOF predictions.
    rng=np.random.default_rng(20261024);delta=[]
    groups=list(merged.groupby(['source_old','label_old']).indices.values())
    def equal_source_ba(df,which):
        return np.mean([balanced_accuracy_score(g.label_old,g['prediction_'+which]) for _,g in df.groupby('source_old')])
    for _ in range(2000):
        b=merged.iloc[np.concatenate([rng.choice(ids,len(ids),replace=True) for ids in groups])]
        delta.append(equal_source_ba(b,'new')-equal_source_ba(b,'old'))
    gain=float(equal_source_ba(merged,'new')-equal_source_ba(merged,'old'));ci=np.quantile(delta,[.025,.975]).tolist()
    source_changes={s:float(balanced_accuracy_score(g.label_old,g.prediction_new)-balanced_accuracy_score(g.label_old,g.prediction_old)) for s,g in merged.groupby('source_old')}
    eligible=gain>=.02 and ci[0]>0 and min(source_changes.values())>=-.01
    decision={'development_promotion_eligible':bool(eligible),'equal_source_ba_gain':gain,'paired_bootstrap_95_delta':ci,
      'source_changes':source_changes,'scope':'Adaptive development experiment; CI conditions on fixed predictions, not all tuning uncertainty',
      'selected':release['selection']['winner'],'threshold':release['selection']['threshold']}
    v3.dump(OUT/'promotion.json',decision)
    for source in ['SVD','VOICED']:
        d=now[now.source==source]
        pd.DataFrame(classification_report(d.label,d.prediction,labels=[0,1],target_names=v3.LABELS,output_dict=True,zero_division=0)).T.to_csv(OUT/f'classification_{source}.csv')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay.from_predictions(d.label,d.prediction,display_labels=v3.LABELS,ax=ax,colorbar=False)
        ax.set_title(source+' development: same participants');fig.tight_layout();fig.savefig(OUT/f'confusion_{source}.png',dpi=160);plt.close(fig)
    print(json.dumps(decision,indent=2),flush=True)

def embedding_model():
    local=ROOT.parents[1]/'work/v4_dependencies'
    if local.exists():sys.path.insert(0,str(local))
    import torch
    from transformers import WavLMModel
    torch.set_num_threads(2);m=WavLMModel.from_pretrained(OUT/'wavlm',local_files_only=True);m.eval();return m

def features(content,model=None):
    x,mel,h,meta=v3.extract(content,8000)
    if model is not None:
        import torch
        from svd_training import load_audio
        y,_=load_audio(content,v3.spec_for(16000));y=(y-y.mean())/np.sqrt(y.var()+1e-7)
        with torch.inference_mode():z=model(torch.tensor(y[None],dtype=torch.float32),output_hidden_states=True).hidden_states[6][0]
        x=np.r_[x,torch.cat([z.mean(0),z.std(0,unbiased=False)]).numpy()]
    return x,mel,h,meta

def challenge():
    if (OUT/'challenge_opened.json').exists():raise ValueError('Challenge already opened. Do not repeat selection using its results.')
    release=json.loads((OUT/'release.json').read_text());bundle=joblib.load(OUT/'binary.joblib')
    v3.dump(OUT/'challenge_opened.json',{'time':time.time(),'model_hash':v3.sha(OUT/'binary.joblib'),
       'selection':release['selection']['winner'],'scope':'New SVD speakers; pathological-only sensitivity challenge. Not external-dataset validation.'})
    old=joblib.load(ROOT/'artifacts_v3/binary.joblib');sel=bundle['selection'];names=sel['members']
    need=any(bundle['models'][n]['config']['subset'] in ['embedding','fusion'] for n in names)
    enc=embedding_model() if need else None
    records=pd.read_csv(OUT/'download_audit.csv').fillna('');records=records[(records.role=='sealed_pathological_challenge')&(records.error=='')]
    dev=pd.read_csv(OUT/'development_manifest.csv');known=set(dev.audio_hash_8000);assert not set(dev.speaker_id)&set(records.speaker_id)
    rows=[]
    for r in records.itertuples():
        content=Path(r.path).read_bytes();x,mel,h,_=features(content,enc)
        if h in known:
            rows.append({'record_id':r.record_id,'speaker_id':r.speaker_id,'error':'Analyzed audio duplicates development; excluded'});continue
        p=np.mean([v4.probability(bundle['models'][n],x[None],mel[None])[0] for n in names],axis=0)
        before=[]
        for n in old['selection']['members']:
            m=old['models'][n];xx,mm,_,_=v3.extract(content,m['config']['sr']);before.append(v3.probability(m,xx[None],mm[None])[0])
        q=np.mean(before,axis=0);pred=int(p[1]>=sel['winner']['cutoff'])
        rows.append({'record_id':r.record_id,'speaker_id':r.speaker_id,'label':1,'old_prediction':int(q.argmax()),'new_prediction':pred,
          'old_score':float(q[1]),'new_score':float(p[1]),'accepted':bool(p[pred]>=sel['threshold']),'error':''})
        pd.DataFrame(rows).to_csv(OUT/'challenge_predictions.csv',index=False)
    valid=pd.DataFrame(rows);valid=valid[valid.error==''];n=len(valid)
    # Wilson score interval for sensitivity, rather than pretending specificity is observed.
    def wilson(k,n):
        z=1.95996398454;ph=k/n;den=1+z*z/n;mid=(ph+z*z/(2*n))/den;half=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
        return [float(mid-half),float(mid+half)]
    result={'n_pathological':n,'n_healthy':0,'old_sensitivity':float(valid.old_prediction.mean()),'new_sensitivity':float(valid.new_prediction.mean()),
       'new_sensitivity_wilson95':wilson(int(valid.new_prediction.sum()),n),'accepted_count':int(valid.accepted.sum()),
       'balanced_accuracy':None,'specificity':None,'roc_auc':None,'limitations':'Same-source positive-only challenge, with download/quality exclusions; not a representative external binary test.'}
    v3.dump(OUT/'challenge_metrics.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['report','challenge']);a=p.parse_args()
    with threadpool_limits(limits=2):globals()[a.stage]()
