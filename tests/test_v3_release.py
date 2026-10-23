from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from svd_v3 import extract,probability,metric,sha
from svd_advanced import extract_advanced
from voice_review.release_web import ReleaseService,predict_voice
from test_svd_web import encoded

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'artifacts_v3/release.json').exists(),reason='Training not complete')

@pytest.fixture(scope='module')
def service():return ReleaseService(ROOT)

def test_saved_pipeline_and_no_reload(service,monkeypatch):
    content=encoded();b=service.bundle;ps={}
    for name in b['selection']['members']:
        m=b['models'][name];x,mel,_,_=extract(content,m['config']['sr'])
        ps[name]=probability(m,x[None],mel[None])[0]
    expected=np.mean(list(ps.values()),axis=0)
    monkeypatch.setattr(joblib,'load',lambda *a,**k:pytest.fail('Reloaded during inference'))
    actual=service.analyze(content,'voice.wav')
    assert actual['pathological_probability']==pytest.approx(expected[1])
    assert actual['subtype_uncertain'] and len(actual['models'])==2
    assert sum(r['Conditional model score (%)'] for r in actual['top3'])==pytest.approx(100)
    x,m,_,_=extract(content,16000);oldx,oldm,_=extract_advanced(content)
    np.testing.assert_allclose(x,oldx,equal_nan=True);np.testing.assert_allclose(m,oldm)

@pytest.mark.parametrize('data,name',[(b'broken','x.wav'),(b'','x.wav'),(encoded(.1),'x.wav'),
    (encoded(silent=True),'x.wav'),(encoded(),'x.txt')],ids=['corrupt','empty','short','silent','extension'])
def test_bad_input(service,data,name):
    with pytest.raises(ValueError):service.analyze(data,name)

def test_mp3_long_and_quality_warnings(service):
    assert service.analyze(encoded(mp3=True),'x.mp3')['mp3']
    result=service.analyze(encoded(20),'x.wav')
    assert result['metadata']['analyzed_seconds']==5
    assert result['original_seconds']==20
    assert isinstance(result['quality_warnings'],list)

def test_nested_metrics_and_history():
    artifact=ROOT/'artifacts_v3'
    p=pd.read_csv(artifact/'nested_predictions.csv');summary=json.loads((artifact/'nested_summary.json').read_text())
    for source in ('SVD','VOICED','pooled'):
        d=p[(p.role=='selection')&((p.source==source) if source!='pooled' else True)]
        assert not d.speaker_id.duplicated().any()
        m=metric(d.label.to_numpy(),np.c_[1-d.p_pathological,d.p_pathological])
        for k,v in m.items():assert v==pytest.approx(summary[source][k])
    for path,value in json.loads((artifact/'historical_v2_hashes.json').read_text()).items():assert sha(ROOT/path)==value
    for fold in json.loads((artifact/'nested_folds.json').read_text()):
        assert not set(fold['train'])&set(fold['validation'])
        for model in fold['model_partitions'].values():
            fit=set(model['fitting']);cal=set(model['calibration']);val=set(fold['validation'])
            assert not fit&cal and not (fit|cal)&val
            if 'stopping_train' in model:
                assert not set(model['stopping_train'])&set(model['stopping_validation'])
                assert set(model['stopping_train']+model['stopping_validation'])<=fit

def test_app_uncertainty_and_clear_input(service):
    app=AppTest.from_file(str(ROOT/'app_v3.py'),default_timeout=60).run()
    assert not app.exception and app.button[0].disabled
    result=service.analyze(encoded(),'voice.wav')
    result.update(result='Uncertain',binary_uncertain=True,leading_label='Pathological')
    app.session_state['v3_analysis']=result;app.run()
    assert not app.exception and len(app.metric)==2
    assert any('Uncertain' in s.value for s in app.subheader)
    assert any('specific cause unresolved' in m.value for m in app.markdown)
    app.session_state['v3_input_hash']='previous-file';app.run()
    assert not app.exception and not app.subheader
