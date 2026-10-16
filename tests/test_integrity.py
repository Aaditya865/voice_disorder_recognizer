"""Meaningful contracts: leakage, signal errors, saved-model consistency and test seals.

Synthetic arrays are used ONLY in these unit tests, never as experiment data.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from voice_review.features import preprocess, audio_fingerprint, extract, SR
from voice_review.models import Transforms, fit_model, score_model
from voice_review.metrics import metrics, confidence_intervals
from voice_review.data import assert_partition
from voice_review.experiment import evaluate_final

@pytest.mark.parametrize('audio,sr,match', [
    (np.zeros(24000),8000,'silent'),
    (np.ones(100),8000,'three seconds'),
    (np.full(24000,np.nan),8000,'finite'),
    (np.ones(24000),4000,'8000'),
])
def test_reject_invalid_audio(audio,sr,match):
    with pytest.raises(ValueError,match=match):
        preprocess(audio,sr)

def test_shared_crop_and_resampling():
    t = np.arange(5*16000)/16000
    x = .2*np.sin(2*np.pi*150*t)+.1
    result = preprocess(np.column_stack([x,x]),16000)
    assert result.shape == (24000,)
    assert abs(result.mean()) < 1e-10
    assert audio_fingerprint(result) == audio_fingerprint(result*2)

def test_acoustic_extraction_known_periodic_signal():
    x = .1*np.sin(2*np.pi*150*np.arange(24000)/8000)
    seq,q,status = extract(x)
    assert seq.shape == (13,297)
    assert np.isfinite(q).all(), status
    assert abs(q[4]-150)<1
    assert q[2]<.001

def test_scalers_and_imputation_use_training_only():
    rng = np.random.default_rng(1)
    seq = rng.normal(size=(12,13,20)).astype('float32')
    q = rng.normal(size=(12,6))
    q[:,0] = np.nan
    t = Transforms().fit(seq,q)
    expected_mean = t.seq_mean.copy()
    expected_medians = t.medians.copy()
    t.transform(seq+1e6,np.full_like(q,np.nan))
    np.testing.assert_array_equal(t.seq_mean,expected_mean)
    np.testing.assert_array_equal(t.medians,expected_medians)
    assert expected_medians[0] == 0
    assert np.isfinite(t.transform(seq,q)[2]).all()

@pytest.mark.parametrize('key',['speaker_id','audio_hash','source_sha256'])
def test_overlap_is_detected(key):
    frame = pd.DataFrame({'record_id':['a','b'],'speaker_id':['a','b'],'audio_hash':['x','y'],'source_sha256':['m','n']})
    frame.loc[1,key] = frame.loc[0,key]
    with pytest.raises(AssertionError,match=key):
        assert_partition(frame,['a'],['b'])

@pytest.mark.parametrize('family',['GMM','DNN'])
def test_reload_reproduces_scores(tmp_path,family):
    rng = np.random.default_rng(4)
    s = rng.normal(size=(20,13,30)).astype('float32')
    q = rng.normal(size=(20,6))
    y = np.array([0,1]*10)
    config = dict(family=family,features='combined',components=1,dropout=.2)
    model = fit_model(s,q,y,config,seed=42,fixed_epochs=2)
    path = tmp_path/'model.joblib'
    joblib.dump(model,path)
    np.testing.assert_allclose(score_model(model,s,q),score_model(joblib.load(path),s,q),atol=1e-7)

def test_metrics_counts_and_threshold():
    m = metrics(np.array([0,0,1,1]),np.array([-1,1,-1,1]))
    assert m['confusion_matrix'] == [[1,1],[1,1]]
    assert m['balanced_accuracy'] == .5
    assert m['macro_f1'] == .5
    assert metrics(np.array([0,1]),np.array([0,0]))['specificity'] == 0

def test_paired_bootstrap_identical_models_have_zero_difference():
    y = np.array([0,1]*10)
    s = np.linspace(-1,1,20)
    intervals = confidence_intervals(y,{'GMM':s,'DNN':s},repeats=50)
    assert intervals['DNN_minus_GMM_balanced_accuracy'] == [0.,0.]

def test_final_requires_explicit_gate(tmp_path):
    with pytest.raises(ValueError,match='confirm-final'):
        evaluate_final(tmp_path,confirm=False)

def test_final_cannot_be_repeated(tmp_path):
    (tmp_path/'results').mkdir()
    (tmp_path/'results/final_evaluation.lock').write_text('opened')
    with pytest.raises(RuntimeError,match='already started'):
        evaluate_final(tmp_path,confirm=True)

def test_frozen_artifact_mutation_blocks_test(tmp_path):
    (tmp_path/'models').mkdir()
    (tmp_path/'data').mkdir()
    (tmp_path/'data/splits.json').write_text('changed')
    (tmp_path/'models/frozen_experiment.json').write_text(json.dumps({'files':{'data/splits.json':'bad-hash'}}))
    with pytest.raises(RuntimeError,match='Frozen artifact changed'):
        evaluate_final(tmp_path,confirm=True)

def test_real_split_and_sealed_feature_cache():
    root = Path(__file__).resolve().parents[1]
    if not (root/'data/splits.json').exists():
        pytest.skip('Dataset not downloaded')
    frame = pd.read_csv(root/'data/manifest.csv')
    splits = json.loads((root/'data/splits.json').read_text())
    assert_partition(frame,splits['development'],splits['test'])
    seen = []
    for fold in splits['folds']:
        assert_partition(frame,fold['train'],fold['validation'])
        assert set(fold['train']+fold['validation']) == set(splits['development'])
        seen.extend(fold['validation'])
    assert sorted(seen) == sorted(splits['development'])
    if not (root/'results/final_evaluation.lock').exists():
        assert not any((root/'data/features'/f'{rid}.npz').exists() for rid in splits['test'])

def test_sealed_record_rejected_before_feature_extraction(monkeypatch):
    from voice_review.experiment import predict
    import voice_review.features as feature_module
    root = Path(__file__).resolve().parents[1]
    if not (root/'data/splits.json').exists() or (root/'results/final_evaluation.lock').exists():
        pytest.skip('No sealed dataset available')
    splits = json.loads((root/'data/splits.json').read_text())
    if not (root/'data/raw'/f'{splits["test"][0]}.hea').exists():
        pytest.skip('Download the original VOICED release for the reserved-source-audio check')
    def should_not_run(*args):
        raise AssertionError('Feature extraction must not run for a sealed demo input')
    monkeypatch.setattr(feature_module,'extract',should_not_run)
    with pytest.raises(ValueError,match='sealed test set'):
        predict(root/'data/raw'/f'{splits["test"][0]}.hea',root)

def test_final_evaluation_end_to_end_on_synthetic_fixture(tmp_path,monkeypatch):
    """Exercise the full final-evaluation path without reading the real test set."""
    import hashlib
    import voice_review.experiment as experiment
    for directory in ['data','models','results']:
        (tmp_path/directory).mkdir()
    ids = ['train0','train1','test0','test1','test2','test3']
    frame = pd.DataFrame({'record_id':ids,'speaker_id':ids,'audio_hash':ids,'source_sha256':ids,
                          'label':[0,1,0,0,1,1],'sex':['x']*6,'age':[30]*6,'diagnosis':['test fixture']*6})
    frame.to_csv(tmp_path/'data/manifest.csv',index=False)
    (tmp_path/'data/splits.json').write_text(json.dumps({'development':ids[:2],'test':ids[2:]}))
    rng = np.random.default_rng(11)
    seq = rng.normal(size=(20,13,30)).astype('float32')
    q = rng.normal(size=(20,6))
    y = np.array([0,1]*10)
    for family in ['GMM','DNN']:
        config = dict(family=family,features='combined',components=1,dropout=.2)
        joblib.dump(fit_model(seq,q,y,config,fixed_epochs=2),tmp_path/f'models/{family.lower()}.joblib')
    paths = [tmp_path/'data/splits.json',tmp_path/'data/manifest.csv',tmp_path/'models/gmm.joblib',tmp_path/'models/dnn.joblib']
    (tmp_path/'models/frozen_experiment.json').write_text(json.dumps({'files':{str(p.relative_to(tmp_path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}))
    monkeypatch.setattr(experiment,'features_for',lambda *args:(seq[:4],q[:4],[]))
    bootstrap = experiment.confidence_intervals
    monkeypatch.setattr(experiment,'confidence_intervals',lambda y,s:bootstrap(y,s,repeats=50))
    evaluate_final(tmp_path,confirm=True)
    result = json.loads((tmp_path/'results/final_metrics.json').read_text())
    pred = pd.read_csv(tmp_path/'results/final_predictions.csv')
    for family in ['GMM','DNN']:
        assert result['models'][family] == metrics(pred.label.to_numpy(),pred[family+'_score'].to_numpy())
    assert (tmp_path/'results/final_evaluation.lock').exists()
    with pytest.raises(RuntimeError,match='already started'):
        evaluate_final(tmp_path,confirm=True)
