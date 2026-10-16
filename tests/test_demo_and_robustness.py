import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from voice_review.demo import uploaded_audio
from voice_review.experiment import predict
from voice_review.robustness import summarize_runs, run_robustness


def wav_bytes(seconds=3, rate=8000):
    data = .1*np.sin(2*np.pi*150*np.arange(int(seconds*rate))/rate)
    buf = io.BytesIO()
    sf.write(buf, data, rate, format='WAV', subtype='FLOAT')
    return buf.getvalue()


def test_upload_name_cannot_escape_temporary_directory_and_is_removed(tmp_path):
    with uploaded_audio('../../outside.wav', wav_bytes(), tmp_path) as path:
        assert path.resolve().is_relative_to((tmp_path/'.runtime/uploads').resolve())
        assert path.name == 'recording.wav'
        assert path.exists()
    assert not path.exists()
    assert not (tmp_path/'outside.wav').exists()


@pytest.mark.parametrize('name,contents,match', [
    ('audio.exe', b'x', 'WAV or FLAC'),
    ('audio.wav', b'', 'nonempty'),
    ('audio.wav', b'not an audio file', 'not readable'),
])
def test_upload_rejects_invalid_content(tmp_path, name, contents, match):
    with pytest.raises(ValueError, match=match):
        with uploaded_audio(name, contents, tmp_path):
            pytest.fail('Invalid file accepted')


def test_upload_duration_is_bounded_and_cleanup_runs(tmp_path):
    with pytest.raises(ValueError, match='60 seconds'):
        with uploaded_audio('long.wav', wav_bytes(seconds=61), tmp_path):
            pytest.fail('Long recording accepted')
    assert list((tmp_path/'.runtime/uploads').iterdir()) == []


def test_uploaded_example_has_identical_predictions_and_cleanup_on_error(tmp_path):
    root = Path(__file__).resolve().parents[1]
    example = root/'examples/voice001.wav'
    if not example.exists():
        pytest.skip('Development examples not generated')
    direct = predict(example, root)
    with uploaded_audio('chosen.wav', example.read_bytes(), tmp_path) as path:
        via_upload = predict(path, root)
    assert via_upload == direct
    with pytest.raises(ValueError, match='simulated failure'):
        with uploaded_audio('chosen.wav', example.read_bytes(), tmp_path) as failed_path:
            raise ValueError('simulated failure')
    assert not failed_path.exists()


def test_seed_summary_counts_label_changes_without_averaging_predictions():
    rows = []
    for seed, scores in [(1, [-1, 1]), (2, [1, 1])]:
        for rid, y, score in zip(['a','b'], [0,1], scores):
            rows.append(dict(model='DNN', seed=seed, record_id=rid, label=y,
                             score=score, prediction=int(score>=0)))
    summary, runs = summarize_runs(pd.DataFrame(rows))
    assert summary['DNN']['records_with_label_changes'] == 1
    assert summary['DNN']['balanced_accuracy_mean'] == .75
    assert len(runs) == 2


def test_seed_analysis_cannot_run_after_final_opening(tmp_path):
    (tmp_path/'results').mkdir()
    (tmp_path/'results/final_evaluation.lock').write_text('test opened')
    with pytest.raises(RuntimeError, match='closed'):
        run_robustness(tmp_path)


def test_real_seed_predictions_cover_only_development_and_match_summary():
    root = Path(__file__).resolve().parents[1]
    path = root/'results/robustness/predictions.csv'
    if not path.exists():
        pytest.skip('Optional seed analysis not run')
    frame = pd.read_csv(path)
    splits = json.loads((root/'data/splits.json').read_text())
    assert set(frame.record_id) == set(splits['development'])
    assert not set(frame.record_id) & set(splits['test'])
    for _, group in frame.groupby(['model','seed']):
        assert len(group) == len(splits['development'])
        assert not group.record_id.duplicated().any()
    summary, _ = summarize_runs(frame)
    saved = json.loads((root/'results/robustness/summary.json').read_text())
    assert summary == saved['summary']


def test_unseen_demo_uses_speaker_held_out_models_and_reproduces_oof():
    root = Path(__file__).resolve().parents[1]
    path = root/'demo_audio/voice003.wav'
    if not path.exists():
        pytest.skip('Unseen demo not generated')
    result = predict(path, root, validation=True)
    context = result['model_context']
    splits = json.loads((root/'data/splits.json').read_text())
    fold = splits['folds'][context['fold']]
    assert context['record_id'] in fold['validation']
    assert context['record_id'] not in fold['train']
    saved = pd.read_csv(root/'results/development_predictions.csv').set_index('record_id').loc[context['record_id']]
    for family in ['GMM','DNN']:
        assert result['models'][family]['raw_score'] == pytest.approx(saved[family+'_score'], abs=1e-4)
    refit = predict(path,root)
    assert 'included in full-model training' in refit['model_context']['notice']


def test_validation_mode_rejects_unknown_audio_before_extraction(tmp_path,monkeypatch):
    import voice_review.features as feature_module
    root = Path(__file__).resolve().parents[1]
    if not (root/'data/splits.json').exists():
        pytest.skip('Dataset not prepared')
    def forbidden(*args):
        pytest.fail('Unknown validation audio must not be extracted or predicted')
    monkeypatch.setattr(feature_module,'extract',forbidden)
    with uploaded_audio('unknown.wav',wav_bytes(),tmp_path) as path:
        with pytest.raises(ValueError,match='unchanged bundled development'):
            predict(path,root,validation=True)
