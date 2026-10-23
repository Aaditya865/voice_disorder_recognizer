"""Bounded improvement experiment; the published v3 release is never overwritten.

Commands: prepare, train, report, challenge. All main-cohort scores are development.
The pathology-only challenge is opened once, after configuration selection is frozen.
"""
from pathlib import Path
import argparse,json,time,hashlib,warnings,sys
import numpy as np,pandas as pd,joblib
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
import svd_v3 as v3
from svd_advanced import transform_fit,transform,raw_scores as original_scores

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/v4';SEED=20261024
RAW_FIT=v3.fit_raw;SPLITS=v3.splits

def configs():
    common=dict(features='rich',sr=8000)
    rows=[dict(id='GMM_rich4',family='GMM',subset='rich',components=4,covariance='diag',pca=20),
      dict(id='GMM_mfcc2',family='GMM',subset='mfcc',components=2,covariance='diag',pca=12),
      dict(id='MLP_base',family='MLP',subset='base',layers=[64,32],alpha=.1,select=0),
      dict(id='MLP_mfcc',family='MLP',subset='mfcc',layers=[32,16],alpha=1.,select=0),
      dict(id='MLP_no_pitch',family='MLP',subset='no_pitch',layers=[32,16],alpha=1.,select=0),
      dict(id='MLP_rich',family='MLP',subset='rich',layers=[32,16],alpha=1.,select=40),
      dict(id='MLP_noise',family='MLP',subset='base',layers=[64,32],alpha=1.,select=0,feature_noise=True),
      dict(id='MLP_svd_only',family='MLP',subset='base',layers=[64,32],alpha=.1,select=0,svd_only=True),
      dict(id='SVM_mfcc_C1',family='SVM',subset='mfcc',C=1.,pca=0),
      dict(id='SVM_rich_C10',family='SVM',subset='rich',C=10.,pca=0),
      dict(id='CNN_aug',family='CNN',subset='rich',width=16,dropout=.5,augment=True),
      dict(id='MLP_wavlm',family='MLP',subset='embedding',layers=[32,16],alpha=1.,select=0,pca=32),
      dict(id='MLP_fusion',family='MLP',subset='fusion',layers=[64,32],alpha=1.,select=0,pca=48),
      dict(id='SVM_wavlm',family='SVM',subset='embedding',C=1.,pca=48)]
    return [{**common,**r,'order':i} for i,r in enumerate(rows)]

def subset(x,c):
    kind=c['subset']
    return {'base':lambda:x[:,:42],'rich':lambda:x[:,:97],'mfcc':lambda:x[:,:26],
      'no_pitch':lambda:np.c_[x[:,:36],x[:,94:97]],'embedding':lambda:x[:,97:],
      'fusion':lambda:x}[kind]()

def fit_raw(x,mel,y,src,c,seed,stop=None,epochs=60):
    if c.get('svd_only'):
        mask=src=='SVD';x,mel,y,src=x[mask],mel[mask],y[mask],src[mask]
    zraw=subset(x,c)
    if c['family'] not in ['SVM'] and not c.get('feature_noise'):
        stop2=(subset(stop[0],c),*stop[1:]) if stop is not None else None
        return RAW_FIT(zraw,mel,y,src,c,seed,stop2,epochs)
    z,pre=transform_fit(zraw,y,c);w=v3.weights(y,src)
    bundle={'config':c,**pre,'classes':2,'temperature':1.}
    if c['family']=='SVM':
        model=SVC(C=c['C'],kernel='rbf',gamma='scale',probability=False,random_state=seed).fit(z,y,sample_weight=w)
    else:
        # Mild feature noise only on fitting participants; original transform is frozen first.
        rng=np.random.default_rng(seed);zz=np.r_[z,z+rng.normal(0,.1,z.shape)]
        model=MLPClassifier(hidden_layer_sizes=tuple(c['layers']),alpha=c['alpha'],max_iter=700,
            batch_size=32,random_state=seed,early_stopping=False,n_iter_no_change=25)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(zz,np.r_[y,y],sample_weight=np.r_[w,w]/2)
        bundle['warnings']=[str(a.message) for a in caught]
    bundle['model']=model;return bundle

def scores(bundle,x,mel):
    raw=subset(x,bundle['config'])
    if bundle['config']['family']=='SVM':
        d=bundle['model'].decision_function(transform(bundle,raw));return np.c_[-d/2,d/2]
    return original_scores(bundle,raw,mel)

def probability(bundle,x,mel):return softmax(scores(bundle,x,mel)/bundle['temperature'],axis=1)

def setup():
    v3.fit_raw=fit_raw;v3.raw_scores=scores
    v3.DEADLINE=time.time()+6*3600

def prepare():
    if (OUT/'protocol_v4.json').exists():raise ValueError('Protocol frozen')
    old=pd.read_csv(ROOT/'data/v3/featured_manifest.csv')
    fm=json.loads((ROOT/'artifacts_v3/feature_manifest.json').read_text())
    assert v3.sha(ROOT/'data/v3/features_8000.npz')==fm['arrays']['8000']['file_hash']
    new=pd.read_csv(OUT/'download_audit.csv').fillna('')
    new=new[(new.role=='development')&(new.error=='')].copy()
    meta=pd.read_csv(ROOT.parents[1]/'work/svd/voice_data.csv').fillna('')
    new['rid']=new.record_id.str.split(':').str[1].astype(int)
    new=new.merge(meta,left_on='rid',right_on='AufnahmeID',validate='one_to_one')
    new['age']=(pd.to_datetime(new.AufnahmeDatum)-pd.to_datetime(new.Geburtsdatum)).dt.days/365.2425
    new['sex']=new.Geschlecht.map({'w':'f','m':'m'}).fillna('unknown')
    old['origin']='previous_development';new['origin']='new_development'
    columns=['record_id','speaker_id','source','label','diagnosis','age','sex','path','raw_hash','origin']
    frame=pd.concat([old[columns],new[columns]],ignore_index=True)
    assert not frame.speaker_id.duplicated().any()
    embedding_ids=pd.read_csv(OUT/'embedding_order.csv').record_id.tolist()
    assert embedding_ids==frame.record_id.tolist()
    emb=np.load(OUT/'embeddings.npz')['x']
    xold=np.load(ROOT/'data/v3/features_8000.npz');xx=list(xold['x']);mm=list(xold['mel'])
    hashes=old.audio_hash_8000.tolist();excluded=[]
    for i,r in new.iterrows():
        x,m,h,_=v3.extract(v3.audio_bytes(r.path),8000)
        if h in hashes:excluded.append(r.record_id)
        xx.append(x);mm.append(m);hashes.append(h)
    keep=~frame.record_id.isin(excluded);frame['audio_hash_8000']=hashes
    frame=frame[keep].reset_index(drop=True)
    x=np.c_[np.stack(xx)[keep],emb[keep]];mel=np.stack(mm)[keep]
    frame.to_csv(OUT/'development_private.csv',index=False)
    frame.drop(columns='path').to_csv(OUT/'development_manifest.csv',index=False)
    np.savez_compressed(OUT/'features.npz',x=x,mel=mel)
    oldfolds=json.loads((ROOT/'artifacts_v3/nested_folds.json').read_text());assignment={}
    for f in oldfolds:
        for sid in f['validation']:assignment[sid]=f['fold']
    fresh=frame[frame.origin=='new_development'];rng=np.random.default_rng(SEED)
    for j,sid in enumerate(rng.permutation(fresh.speaker_id.to_numpy())):assignment[sid]=j%4
    frame['outer_fold']=frame.speaker_id.map(assignment);frame[['speaker_id','outer_fold']].to_csv(OUT/'outer_assignment.csv',index=False)
    protocol={'seed':SEED,'candidates':configs(),'source_hashes':{**v3.source_hashes(),'svd_v4.py':v3.sha(Path(__file__))},
        'features_hash':v3.sha(OUT/'features.npz'),'manifest_hash':v3.sha(OUT/'development_private.csv'),
        'duplicate_exclusions':excluded,'selection':'Equal-source balanced accuracy; cutoff selected inside inner validation; within .01 prefer GMM, MLP, SVM, CNN, ensemble',
        'promotion':'Paired bootstrap lower95 delta>0; mean-source BA gain>=.02; neither source decreases by>.01 on original cohort. Otherwise retain v3.',
        'challenge':'Downloaded usable subset of 100 preselected new pathological speakers; sensitivity only; never select models using it.',
        'uncertainty':'Original .6,.7,.8,.9 requirements unchanged: each source accuracy>=.8, coverage>=.2, accepted>=20',
        'development_only':True,'max_training_hours':6}
    protocol['embedding_hash']=v3.sha(OUT/'embeddings.npz')
    protocol['embedding_provenance']=json.loads((OUT/'wavlm_provenance.json').read_text())
    protocol['prior_release_hash']=v3.sha(ROOT/'artifacts_v3/binary.joblib')
    protocol['old_outer_predictions_hash']=v3.sha(ROOT/'artifacts_v3/nested_predictions.csv')
    v3.dump(OUT/'protocol_v4.json',protocol)
    print('prepared',len(frame),'new',int((frame.origin=='new_development').sum()),flush=True)

def metric(frame,p,cutoff=.5):
    y=frame.label.to_numpy(int);pred=(p[:,1]>=cutoff).astype(int);m=v3.metric(y,p)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    m.update(accuracy=float((y==pred).mean()),balanced_accuracy=float(balanced_accuracy_score(y,pred)),
      macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),sensitivity=float(tp/(tp+fn)) if tp+fn else None,
      specificity=float(tn/(tn+fp)) if tn+fp else None,tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))
    return m

def score(frame,p,cutoff):
    return float(np.mean([metric(frame[frame.source==s],p[frame.source.to_numpy()==s],cutoff)['balanced_accuracy'] for s in v3.SOURCES]))

def cutoff_for(frame,p):
    values=[(score(frame,p,t),t) for t in np.arange(.25,.751,.05)]
    return max(values,key=lambda a:(a[0],-abs(a[1]-.5)))[1]

def rejection(frame,p,cutoff):
    pred=(p[:,1]>=cutoff).astype(int);confidence=p[np.arange(len(p)),pred];curve=[]
    for threshold in [.6,.7,.8,.9]:
        row={'threshold':threshold,'eligible':True}
        for source in v3.SOURCES:
            m=frame.source.to_numpy()==source;accepted=m&(confidence>=threshold)
            acc=float((pred[accepted]==frame.label.to_numpy()[accepted]).mean()) if accepted.any() else None
            cov=float(accepted.sum()/m.sum());row[source]={'accepted':int(accepted.sum()),'coverage':cov,'accuracy':acc}
            row['eligible'] &= bool(accepted.sum()>=20 and cov>=.2 and acc is not None and acc>=.8)
        curve.append(row)
    return min((r['threshold'] for r in curve if r['eligible']),default=1.01),curve

def choose(rows):
    best=max(r['score'] for r in rows)
    return min([r for r in rows if r['score']>=best-.01],key=lambda r:({'GMM':0,'MLP':1,'SVM':2,'CNN':3,'Ensemble':4}[r['family']],r['order']))

def train():
    setup();protocol=json.loads((OUT/'protocol_v4.json').read_text())
    assert protocol['source_hashes']=={**v3.source_hashes(),'svd_v4.py':v3.sha(Path(__file__))}
    assert protocol['features_hash']==v3.sha(OUT/'features.npz')
    assert protocol['manifest_hash']==v3.sha(OUT/'development_private.csv')
    if (OUT/'release.json').exists():raise ValueError('Completed experiment frozen')
    frame=pd.read_csv(OUT/'development_private.csv');a=np.load(OUT/'features.npz');x,mel=a['x'],a['mel']
    clock=OUT/'clock.json'
    if not clock.exists():v3.dump(clock,{'started':time.time()})
    v3.DEADLINE=json.loads(clock.read_text())['started']+6*3600
    cache=OUT/'fit_cache';cache.mkdir(exist_ok=True);configurations=configs();comparisons=[];foldlog=[];predrows=[]
    def fit(ids,c,seed):
        v3.check_time();key=v3.key([protocol['source_hashes'],protocol['features_hash'],ids.tolist(),c,seed]);path=cache/(key+'.joblib')
        if path.exists():return joblib.load(path)
        model=v3.fit_calibrated(frame.iloc[ids].reset_index(drop=True),x[ids],mel[ids],c,seed)
        joblib.dump(model,path);return model
    def select(ids,seed,stage):
        part=frame.iloc[ids].reset_index(drop=True);folds=SPLITS(part,3,seed);ps={};rows=[]
        for c in configurations:
            p=np.zeros((len(ids),2))
            for k,(tr,va) in enumerate(folds):
                print('FIT',stage,c['id'],k,flush=True);m=fit(ids[tr],c,seed+k)
                assert not set(part.iloc[va].speaker_id)&set(m['partition_audit']['fitting']+m['partition_audit']['calibration'])
                p[va]=probability(m,x[ids[va]],mel[ids[va]])
            cut=cutoff_for(part,p);s=score(part,p,cut);ps[c['id']]=p
            rows.append({'id':c['id'],'family':c['family'],'order':c['order'],'score':s,'cutoff':float(cut)})
            for source in ['pooled',*v3.SOURCES]:
                mask=np.ones(len(part),bool) if source=='pooled' else part.source.to_numpy()==source
                comparisons.append({'stage':stage,'id':c['id'],'source':source,'cutoff':cut,**metric(part[mask],p[mask],cut)})
            print('SCORE',stage,c['id'],s,flush=True)
            pd.DataFrame(comparisons).to_csv(OUT/'candidate_comparison.csv',index=False)
        g=choose([r for r in rows if r['family']=='GMM']);d=choose([r for r in rows if r['family'] in ['MLP','CNN']])
        ep=(ps[g['id']]+ps[d['id']])/2;cut=cutoff_for(part,ep);ps['Ensemble']=ep
        rows.append({'id':'Ensemble','family':'Ensemble','order':len(rows),'score':score(part,ep,cut),'cutoff':float(cut)})
        best=choose(rows);members=[g['id'],d['id']] if best['id']=='Ensemble' else [best['id']]
        threshold,curve=rejection(part,ps[best['id']],best['cutoff'])
        np.savez_compressed(OUT/(stage+'_inner_predictions.npz'),record_id=part.record_id.to_numpy(str),**ps)
        return {'winner':best,'members':members,'best_gmm':g['id'],'best_dnn':d['id'],'threshold':threshold,'threshold_curve':curve,'ranking':rows,
            'inner_splits':[{'train':part.iloc[tr].speaker_id.tolist(),'validation':part.iloc[va].speaker_id.tolist()} for tr,va in folds]}
    assignments=pd.read_csv(OUT/'outer_assignment.csv');assert assignments.speaker_id.tolist()==frame.speaker_id.tolist()
    for fold in range(4):
        tr=np.flatnonzero(assignments.outer_fold.to_numpy()!=fold);va=np.flatnonzero(assignments.outer_fold.to_numpy()==fold)
        sel=select(tr,SEED+100*fold,'outer'+str(fold));names=set(sel['members']+[sel['best_gmm'],sel['best_dnn']])
        models={n:fit(tr,next(c for c in configurations if c['id']==n),SEED+1000+fold) for n in names}
        ps={n:probability(m,x[va],mel[va]) for n,m in models.items()};p=np.mean([ps[n] for n in sel['members']],axis=0)
        for role,pp in [('selection',p),('GMM',ps[sel['best_gmm']]),('DNN',ps[sel['best_dnn']])]:
            cut=sel['winner']['cutoff'] if role=='selection' else next(r['cutoff'] for r in sel['ranking'] if r['id']==sel['best_gmm' if role=='GMM' else 'best_dnn'])
            for i,j in enumerate(va):
                pred=int(pp[i,1]>=cut);conf=float(pp[i,pred]);predrows.append({'record_id':frame.iloc[j].record_id,'speaker_id':frame.iloc[j].speaker_id,
                  'source':frame.iloc[j].source,'origin':frame.iloc[j].origin,'label':int(frame.iloc[j].label),'fold':fold,'role':role,
                  'p_pathological':float(pp[i,1]),'prediction':pred,'cutoff':cut,'confidence':conf,'threshold':sel['threshold'],'accepted':bool(conf>=sel['threshold'])})
        foldlog.append({'fold':fold,'train':frame.iloc[tr].speaker_id.tolist(),'validation':frame.iloc[va].speaker_id.tolist(),'selection':sel,
          'model_partitions':{n:m['partition_audit'] for n,m in models.items()}})
        pd.DataFrame(predrows).to_csv(OUT/'nested_predictions.csv',index=False);v3.dump(OUT/'nested_folds.json',foldlog)
    final=select(np.arange(len(frame)),SEED+2000,'final-development');names=set(final['members']+[final['best_gmm'],final['best_dnn']])
    models={n:fit(np.arange(len(frame)),next(c for c in configurations if c['id']==n),SEED+3000) for n in names}
    joblib.dump({'models':models,'selection':final,'protocol':protocol,'labels':v3.LABELS},OUT/'binary.joblib')
    v3.dump(OUT/'release.json',{'selection':final,'model_hash':v3.sha(OUT/'binary.joblib'),'completed':True,'elapsed_seconds':time.time()-json.loads(clock.read_text())['started']})
    print('COMPLETE',final['winner'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train']);args=parser.parse_args()
    with threadpool_limits(limits=2):globals()[args.stage]()
