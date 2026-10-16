import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from voice_review.web_inference import ModelService,read_upload
from voice_review.features import read_audio,preprocess
from voice_review.laryngeal import label_for,measure,NAMES

ROOT = Path(__file__).resolve().parents[1]


def encoded(seconds=4,sr=16000,format='WAV',silent=False):
    x = np.zeros(int(seconds*sr)) if silent else .2*np.sin(2*np.pi*137*np.arange(int(seconds*sr))/sr)
    b=io.BytesIO()
    sf.write(b,x,sr,format=format,subtype='MPEG_LAYER_III' if format=='MP3' else 'FLOAT')
    return b.getvalue()


@pytest.fixture(scope='module')
def service():
    return ModelService(ROOT)


@pytest.mark.parametrize('name,data,match',[
    ('a.txt',b'x','WAV or MP3'),('a.wav',b'','nonempty'),
    ('a.wav',b'broken','decoded'),('a.wav',encoded(seconds=1),'three seconds'),
    ('a.wav',encoded(silent=True),'silent'),
],ids=['wrong_type','empty','corrupt','too_short','silent'])
def test_friendly_input_errors(name,data,match):
    with pytest.raises(ValueError,match=match):
        read_upload(data,name)


def test_mp3_decodes_and_uses_canonical_feature_input():
    audio,metadata=read_upload(encoded(format='MP3'),'recording.mp3')
    assert audio.shape==(24000,)
    assert np.isfinite(audio).all()
    assert metadata['original_sample_rate']==16000


def test_long_audio_is_trimmed_instead_of_rejected():
    audio,metadata=read_upload(encoded(seconds=75,sr=8000),'long.wav')
    assert len(audio)==24000
    assert metadata['long_clip'] and metadata['original_seconds']==75


def test_no_model_files_load_during_requests(service,monkeypatch):
    def fail(*args,**kwargs):
        pytest.fail('Model file loaded during request')
    monkeypatch.setattr(joblib,'load',fail)
    for name in ['healthy_voice002.wav','pathological_voice003.wav']:
        path=ROOT/'samples'/name
        result=service.analyze(path.read_bytes(),name)
        assert set(result['models'])=={'GMM','DNN'}


def test_samples_match_saved_oof_and_exact_preprocessing(service):
    original=pd.read_csv(ROOT/'results/development_predictions.csv').set_index('record_id')
    subtype=pd.read_csv(ROOT/'results/laryngeal/predictions.csv').set_index('record_id')
    metadata=json.loads((ROOT/'samples/manifest.json').read_text())
    for sample in metadata:
        path=ROOT/'samples'/sample['filename']
        result=service.analyze(path.read_bytes(),path.name)
        rid=result['record_id']
        assert rid==sample['record_id'] and result['fold']==sample['validation_fold']
        source_path=ROOT/'data/raw'/f'{rid}.hea'
        if source_path.exists():
            raw,sr=read_audio(source_path)
            np.testing.assert_allclose(result['audio'],preprocess(raw,sr),atol=1e-12)
        for family in ['GMM','DNN']:
            assert result['models'][family]['raw_score']==pytest.approx(original.loc[rid,family+'_score'],abs=1e-4)
            assert result['models'][family]['three_class_result']==NAMES[int(subtype.loc[rid,family+'_prediction'])]


def test_reserved_audio_blocked_before_extraction(service,monkeypatch):
    import voice_review.web_inference as web
    rid=service.splits['test'][0]
    if not (ROOT/'data/raw'/f'{rid}.hea').exists():
        pytest.skip('Download the original VOICED release for the reserved-source-audio check')
    raw,sr=read_audio(ROOT/'data/raw'/f'{rid}.hea')
    crop=preprocess(raw,sr)
    b=io.BytesIO()
    sf.write(b,crop,8000,format='WAV',subtype='DOUBLE')
    def fail(*args):
        pytest.fail('Reserved feature extraction attempted')
    monkeypatch.setattr(web,'extract',fail)
    with pytest.raises(ValueError,match='reserved final-test'):
        service.analyze(b.getvalue(),'test.wav')


def test_multiclass_saved_metrics_and_group_definition():
    assert label_for('reflux laryngitis')==1
    assert label_for('hyperkinetic dysphonia (nodule)')==2
    pred=pd.read_csv(ROOT/'results/laryngeal/predictions.csv')
    saved=json.loads((ROOT/'results/laryngeal/metrics.json').read_text())
    for family in ['GMM','DNN']:
        values=pred[[f'{family}_score_{c}' for c in range(3)]].to_numpy()
        assert measure(pred.label,values)==saved['models'][family]


def test_app_renders_with_one_analysis_button_and_no_input_crash():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
    assert not app.exception
    assert len(app.button)==1
    assert app.button[0].label=='Analyze Voice' and app.button[0].disabled
    assert 'Academic project' in app.caption[0].value
