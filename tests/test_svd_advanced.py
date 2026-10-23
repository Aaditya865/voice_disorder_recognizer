import numpy as np
import pandas as pd
import pytest
from svd_advanced import (DATA, OUT, SEED, LABELS, FEATURE_VERSION, groupsplit,
    fit_calibrated, predict, extract_advanced, candidates)
from test_svd_training import wav_bytes

def test_new_protocol_and_audit():
    dev=pd.read_csv(DATA/'development.csv').fillna('')
    assert dev.speaker_id.nunique()==len(dev)
    for row in dev[dev.subtype!=''].itertuples():
        labels={s.strip() for s in row.source_diagnoses.split(',')}
        assert labels<=LABELS[row.subtype]
    external=pd.read_csv(DATA/'external_test_frozen.csv')
    assert len(external)==42
    assert set(external.speaker_id)&set(dev.speaker_id.astype(str))==set()
    assert external.binary.value_counts().to_dict()=={1:30,0:12}

def test_calibration_and_scaler_are_inside_training_groups():
    rng=np.random.default_rng(3)
    groups=np.repeat(np.arange(60),2)
    y=(groups%2).astype(int)
    x=rng.normal(size=(len(y),97))
    mel=np.zeros((len(y),40,128),dtype=np.float32)
    a,b=groupsplit(y,groups,4,SEED)[0]
    model=fit_calibrated(x[a],mel[a],y[a],groups[a],candidates()[0],SEED)
    fitting=set(model['fitting_speakers']);cal=set(model['calibration_speakers'])
    assert not fitting&cal
    assert not (fitting|cal)&set(groups[b])
    selected=np.isin(groups,list(fitting))
    np.testing.assert_allclose(model['scaler'].mean_,x[selected,:42].mean(axis=0))
    p=predict(model,x[b],mel[b])
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1),1)

def test_advanced_feature_contract():
    x,mel,ahash=extract_advanced(wav_bytes())
    assert x.shape==(97,)
    assert mel.shape==(40,128)
    assert np.isfinite(mel).all()
    assert np.isfinite(x[-3:]).all()
    assert len(ahash)==64
