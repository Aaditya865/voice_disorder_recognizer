import numpy as np,pandas as pd
import svd_v4 as v4

def test_feature_subsets():
    x=np.arange(2*1633).reshape(2,1633)
    assert v4.subset(x,{'subset':'mfcc'}).shape==(2,26)
    assert v4.subset(x,{'subset':'embedding'}).shape==(2,1536)
    assert v4.subset(x,{'subset':'no_pitch'}).shape==(2,39)
    assert not np.isin(x[0,36:42],v4.subset(x,{'subset':'no_pitch'})[0]).any()

def test_cutoff_and_probability_metrics_are_separate():
    frame=pd.DataFrame({'label':[0,1,0,1],'source':['SVD','SVD','VOICED','VOICED']})
    p=np.array([[.8,.2],[.6,.4],[.7,.3],[.55,.45]])
    cut=v4.cutoff_for(frame,p)
    assert v4.score(frame,p,cut)==1
    m=v4.metric(frame,p,cut)
    assert m['tp']==2 and m['tn']==2
    assert m['brier']==v4.v3.metric(frame.label,p)['brier']

def test_rejection_not_relaxed():
    frame=pd.DataFrame({'label':[0,1]*30,'source':['SVD']*30+['VOICED']*30})
    p=np.array([[.95,.05] if y==0 else [.05,.95] for y in frame.label])
    assert v4.rejection(frame,p,.5)[0]==.6
    p[30:]=.5
    assert v4.rejection(frame,p,.5)[0]>1

def test_fit_calibration_disjoint_and_fitting_only_scaler(monkeypatch):
    import time
    monkeypatch.setattr(v4.v3,'fit_raw',v4.fit_raw)
    monkeypatch.setattr(v4.v3,'raw_scores',v4.scores)
    monkeypatch.setattr(v4.v3,'DEADLINE',time.time()+3600)
    rng=np.random.default_rng(9)
    f=pd.DataFrame({'speaker_id':[str(i) for i in range(80)],'label':[0,1]*40,'source':['SVD']*40+['VOICED']*40})
    x=rng.normal(size=(80,1633));mel=rng.normal(size=(80,40,128))
    c=next(c for c in v4.configs() if c['id']=='SVM_mfcc_C1')
    model=v4.v3.fit_calibrated(f,x,mel,c,42)
    fit=model['partition_audit']['fitting'];cal=model['partition_audit']['calibration']
    assert not set(fit)&set(cal)
    np.testing.assert_allclose(model['scaler'].mean_,x[np.array(fit,int),:26].mean(0))
    p=v4.probability(model,x,mel);np.testing.assert_allclose(p.sum(1),1)
