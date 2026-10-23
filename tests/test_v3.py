import numpy as np
import pandas as pd
import pytest
from svd_v3 import (extract,spec_for,weights,splits,fit_calibrated,probability,candidates,rejection,choose)
from test_svd_training import wav_bytes

@pytest.mark.parametrize('sr',[8000,16000])
def test_feature_variants(sr):
    x,m,h,meta=extract(wav_bytes(),sr)
    assert x.shape==(97,) and m.shape==(40,128)
    assert np.isfinite(m).all() and np.isfinite(x[-3:]).all()
    assert meta['analyzed_seconds']>=.5

def fixture_frame():
    return pd.DataFrame({'speaker_id':[str(i) for i in range(160)],
        'label':[i%2 for i in range(160)],'source':['SVD' if i<80 else 'VOICED' for i in range(160)]})

def test_nested_participants_and_fitting_preprocessing():
    frame=fixture_frame();rng=np.random.default_rng(4);x=rng.normal(size=(160,97));mel=np.zeros((160,40,128))
    a,b=splits(frame,4,1)[0];model=fit_calibrated(frame.iloc[a].reset_index(drop=True),x[a],mel[a],candidates()[0],1)
    audit=model['partition_audit'];fit=set(audit['fitting']);cal=set(audit['calibration'])
    assert not fit&cal and not (fit|cal)&set(frame.iloc[b].speaker_id)
    ids=frame.speaker_id.isin(fit).to_numpy()
    np.testing.assert_allclose(model['scaler'].mean_,x[ids,:42].mean(0))
    p=probability(model,x[b],mel[b]);np.testing.assert_allclose(p.sum(1),1)

def test_weights_balance_each_source_class():
    y=np.array([0]*30+[1]*10+[0]*8+[1]*12);s=np.array(['SVD']*40+['VOICED']*20)
    w=weights(y,s)
    for source in ('SVD','VOICED'):
        for label in (0,1):assert w[(s==source)&(y==label)].sum()==pytest.approx(15)

def test_rejection_requires_both_sources():
    f=fixture_frame();p=np.full((160,2),.01);p[np.arange(160),f.label]=.99
    threshold,_=rejection(f,p);assert threshold==.6
    p[80:]=p[80:,::-1];threshold,_=rejection(f,p);assert threshold==1.01

def test_complexity_tiebreak():
    rows=[dict(score=.7,family='GMM',order=0),dict(score=.709,family='CNN',order=1)]
    assert choose(rows)['family']=='GMM'
