from pathlib import Path
import joblib
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from svd_advanced import extract_advanced,predict
from voice_review.advanced_web import AdvancedService,predict_voice
from test_svd_web import encoded

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def service(): return AdvancedService(ROOT)

def test_shared_pipeline_and_cached_network(service,monkeypatch):
    content=encoded()
    x,mel,_=extract_advanced(content)
    expected=predict(service.bundles['binary'],x[None],mel[None])[0]
    expected=expected/expected.sum()
    monkeypatch.setattr(joblib,'load',lambda *a,**k:pytest.fail('Reloaded a saved model during inference'))
    result=service.analyze(content,'voice.wav')
    assert result['pathological_probability']==pytest.approx(expected[1])
    assert len(result['top3'])==3
    assert sum(v['Conditional model score (%)'] for v in result['top3'])==pytest.approx(100)
    assert result['subtype_uncertain']
    assert service.bundles['binary']['members'][1]['_runtime'] is not None

@pytest.mark.parametrize('data,name',[(b'broken','x.wav'),(b'','x.wav'),(encoded(.1),'x.wav'),
    (encoded(silent=True),'x.wav'),(encoded(),'x.txt')],ids=['corrupt','empty','short','silent','extension'])
def test_invalid_audio(service,data,name):
    with pytest.raises(ValueError): service.analyze(data,name)

def test_long_and_compressed_audio(service):
    assert service.analyze(encoded(mp3=True),'x.mp3')['mp3']
    result=service.analyze(encoded(20),'x.wav')
    assert len(result['audio'])==80000
    assert result['original_seconds']==20

def test_uncertain_ui_and_model_cache(service):
    app=AppTest.from_file(str(ROOT/'app_svd_v2.py'),default_timeout=60).run()
    assert not app.exception
    assert app.button[0].disabled
    result=service.analyze(encoded(),'voice.wav')
    # UI branches use fixture state; these are not experimental predictions.
    result.update(result='Uncertain',binary_uncertain=True,leading_label='Pathological')
    app.session_state['v2_analysis']=result
    app.run()
    assert not app.exception
    assert len(app.metric)==2
    assert any('Uncertain' in x.value for x in app.subheader)
    assert any('Disorder type: uncertain' in x.value for x in app.markdown)
    assert any('external dataset' in x.value for x in app.warning)
