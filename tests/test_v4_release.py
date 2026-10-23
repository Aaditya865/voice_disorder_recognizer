"""Release integrity checks; completed experiment required."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import svd_v4 as v4
from svd_v4_results import metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/v4'
pytestmark=pytest.mark.skipif(not (OUT/'release.json').exists(),reason='Experiment not complete')

def check_partitions(a,allowed,forbidden=()):
    fit=set(a['fitting']);cal=set(a['calibration'])
    assert not fit&cal
    assert fit|cal <= set(allowed)
    assert not (fit|cal)&set(forbidden)
    if 'stopping_train' in a:
        tr=set(a['stopping_train']);va=set(a['stopping_validation'])
        assert not tr&va and tr|va <= fit

def test_protocol_and_history_unchanged():
    p=json.loads((OUT/'protocol_v4.json').read_text())
    for name,value in p['source_hashes'].items():assert v4.v3.sha(ROOT/name)==value
    assert v4.v3.sha(ROOT/'artifacts_v3/binary.joblib')==p['prior_release_hash']
    assert v4.v3.sha(ROOT/'artifacts_v3/nested_predictions.csv')==p['old_outer_predictions_hash']
    release=json.loads((OUT/'release.json').read_text())
    assert v4.v3.sha(OUT/'binary.joblib')==release['model_hash']

def test_every_outer_and_inner_partition():
    frame=pd.read_csv(OUT/'development_manifest.csv')
    assert not frame.speaker_id.duplicated().any()
    assert not frame.audio_hash_8000.duplicated().any()
    outer=json.loads((OUT/'nested_folds.json').read_text())
    assert len(outer)==4
    validation=[]
    for f in outer:
        tr=set(f['train']);va=set(f['validation']);validation.extend(va)
        assert not tr&va and tr|va==set(frame.speaker_id)
        for split in f['selection']['inner_splits']:
            a=set(split['train']);b=set(split['validation'])
            assert not a&b and a|b==tr
        for model in f['model_partitions'].values():check_partitions(model,tr,va)
    assert len(validation)==len(set(validation))==len(frame)
    bundle=joblib.load(OUT/'binary.joblib')
    for m in bundle['models'].values():check_partitions(m['partition_audit'],frame.speaker_id)

def test_metrics_recomputed_from_predictions():
    p=pd.read_csv(OUT/'nested_predictions.csv')
    table=pd.read_csv(OUT/'before_after.csv')
    for role,name in [('selection','v4_all_development'),('GMM','v4_GMM'),('DNN','v4_DNN')]:
        for source in ['pooled','SVD','VOICED']:
            d=p[(p.role==role)&((p.source==source) if source!='pooled' else True)]
            expected=table[(table.procedure==name)&(table.source==source)].iloc[0]
            for key,value in metrics(d).items():
                if value is not None:assert float(expected[key])==pytest.approx(value)

def test_saved_reloaded_predictions(tmp_path):
    bundle=joblib.load(OUT/'binary.joblib');path=tmp_path/'copy.joblib';joblib.dump(bundle,path)
    again=joblib.load(path);rng=np.random.default_rng(3)
    x=rng.normal(size=(3,1633));mel=rng.normal(size=(3,40,128))
    for name,m in bundle['models'].items():
        p=v4.probability(m,x,mel)
        np.testing.assert_allclose(p,v4.probability(again['models'][name],x,mel),rtol=1e-7,atol=1e-9)
        assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1)

def test_challenge_was_not_used_for_training():
    if not (OUT/'challenge_predictions.csv').exists():pytest.skip('Sealed challenge not opened')
    d=pd.read_csv(OUT/'challenge_predictions.csv').fillna('');d=d[d.error=='']
    dev=pd.read_csv(OUT/'development_manifest.csv')
    assert not set(d.speaker_id)&set(dev.speaker_id)
    assert not d.speaker_id.duplicated().any()
    m=json.loads((OUT/'challenge_metrics.json').read_text())
    assert m['n_pathological']==len(d) and m['n_healthy']==0
    assert m['new_sensitivity']==pytest.approx(d.new_prediction.mean())
    assert m['old_sensitivity']==pytest.approx(d.old_prediction.mean())
    assert m['balanced_accuracy'] is None and m['specificity'] is None
