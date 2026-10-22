import io
import joblib
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from svd_training import (FEATURE_NAMES, load_audio, extract_features, grouped_partition,
                          fit_bundle, probabilities, predict_voice, assert_disjoint)


def wav_bytes(seconds=1.0, silence=False):
    sr = 50000
    t = np.arange(int(seconds*sr))/sr
    x = np.zeros_like(t) if silence else .2*np.sin(2*np.pi*140*t)+.04*np.sin(2*np.pi*280*t)
    stream = io.BytesIO()
    sf.write(stream, x, sr, format='WAV', subtype='PCM_16')
    return stream.getvalue()


def test_upload_and_training_path_identical(tmp_path):
    raw = wav_bytes()
    path = tmp_path/'example.wav'
    path.write_bytes(raw)
    from_file, _ = extract_features(path)
    from_upload, meta = extract_features(io.BytesIO(raw))
    np.testing.assert_allclose(from_file, from_upload, equal_nan=True)
    assert len(from_file) == len(FEATURE_NAMES) == 42
    assert meta['original_sample_rate'] == 50000
    assert abs(from_file[36]-140) < 5
    audio, _ = load_audio(raw)
    assert len(audio) == 16000
    assert np.max(np.abs(audio)) == pytest.approx(1)


def test_bad_audio_rejected():
    with pytest.raises(ValueError, match='silent'):
        load_audio(wav_bytes(silence=True))
    with pytest.raises(ValueError, match='at least'):
        load_audio(wav_bytes(seconds=.1))
    with pytest.raises(ValueError, match='decode'):
        load_audio(b'not an audio file')


def test_speakers_disjoint_even_with_multiple_recordings():
    frame = pd.DataFrame({'speaker_id':np.repeat(np.arange(100),2),
                          'audio_hash':[str(i) for i in range(200)]})
    y = np.repeat(np.arange(100)%5,2)
    left, right = grouped_partition(frame,y,5,123)
    assert not set(frame.iloc[left].speaker_id)&set(frame.iloc[right].speaker_id)
    frame.loc[right[0],'audio_hash'] = frame.loc[left[0],'audio_hash']
    with pytest.raises(RuntimeError, match='Duplicate audio'):
        assert_disjoint(frame,left,right)


def test_bundle_reload_and_helper(tmp_path):
    rng = np.random.default_rng(123)
    features, _ = extract_features(wav_bytes())
    x = np.tile(np.nan_to_num(features),(40,1)) + rng.normal(0,.05,(40,42))
    y = np.arange(40)%2
    x[0,38] = np.nan
    encoder = LabelEncoder().fit(['Healthy','Laryngitis'])
    bundle = fit_bundle(x,y,encoder,{'family':'GMM','components':1},123)
    path = tmp_path/'model.joblib'
    joblib.dump(bundle,path)
    restored = joblib.load(path)
    np.testing.assert_allclose(probabilities(bundle,x), probabilities(restored,x))
    result = predict_voice(io.BytesIO(wav_bytes()), bundle=restored)
    assert sum(result['probabilities_percent'].values()) == pytest.approx(100)
    assert result['diagnosis'] in encoder.classes_
    assert result['probabilities_calibrated'] is False
