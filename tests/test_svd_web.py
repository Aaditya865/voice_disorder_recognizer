import io
from pathlib import Path
import joblib
import numpy as np
import pytest
import soundfile as sf
from streamlit.testing.v1 import AppTest
from svd_training import predict_voice
from voice_review.svd_web import SVDService

ROOT = Path(__file__).resolve().parents[1]

def encoded(seconds=1, silent=False, mp3=False):
    t = np.arange(int(seconds*16000))/16000
    x = np.zeros_like(t) if silent else .2*np.sin(2*np.pi*140*t)
    b = io.BytesIO()
    sf.write(b,x,16000,format='MP3' if mp3 else 'WAV',subtype='MPEG_LAYER_III' if mp3 else 'PCM_16')
    return b.getvalue()

@pytest.fixture(scope='module')
def service():
    return SVDService(ROOT)

def test_shared_features_and_no_reload(service,monkeypatch):
    data = encoded()
    expected = {f:predict_voice(data,bundle=b) for f,b in service.bundles.items()}
    monkeypatch.setattr(joblib,'load',lambda *a,**k:pytest.fail('Model reloaded during request'))
    result = service.analyze(data,'voice.wav')
    for f in expected:
        assert result['models'][f]['diagnosis'] == expected[f]['diagnosis']
        assert result['models'][f]['score']*100 == pytest.approx(expected[f]['confidence_percent'])
    assert len(result['models']['DNN']['top3']) == 3

@pytest.mark.parametrize('data,name',[(b'broken','x.wav'),(b'','x.wav'),(encoded(.1),'x.wav'),
    (encoded(silent=True),'x.wav'),(encoded(),'x.txt')], ids=['corrupt','empty','short','silent','extension'])
def test_bad_input(service,data,name):
    with pytest.raises(ValueError): service.analyze(data,name)

def test_mp3_and_long_audio(service):
    assert service.analyze(encoded(mp3=True),'x.mp3')['mp3']
    assert len(service.analyze(encoded(20),'x.wav')['audio']) == 5*16000

def test_app_start_and_result_render(service):
    app = AppTest.from_file(str(ROOT/'app_svd_v1.py'),default_timeout=60).run()
    assert not app.exception
    assert app.button[0].disabled
    result = service.analyze(encoded(),'x.wav')
    app.session_state['svd_analysis'] = result
    app.run()
    assert not app.exception
    assert len(app.metric) == 2
    assert any('Result:' in x.value for x in app.subheader)
