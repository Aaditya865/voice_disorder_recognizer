"""One preprocessing/extraction path shared by training and prediction."""
from pathlib import Path
from math import gcd
import hashlib
import warnings
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.signal import resample_poly
import soundfile as sf
import wfdb

SR = 8000
SAMPLES = 3 * SR
QUALITY_NAMES = ['cpps_db', 'hnr_db', 'jitter_local', 'shimmer_local', 'f0_median_hz', 'f0_iqr_hz']
SPEC = {
    'sample_rate': SR, 'seconds': 3, 'crop': 'central', 'dc_removal': True,
    'mfcc': {'count': 13, 'include_c0': True, 'n_fft': 256, 'win_length': 200,
             'hop_length': 80, 'n_mels': 26, 'fmin': 50, 'fmax': 3800,
             'center': False, 'window': 'hann'},
    'pitch_floor': 60, 'pitch_ceiling': 500,
    'quality_names': QUALITY_NAMES, 'version': 1,
    'cpps': 'Praat power cepstrogram; 60 Hz floor, 2 ms step, 4 kHz ceiling, 50 Hz preemphasis; smoothing 10 ms / 1 ms; 60-500 Hz peak search; robust straight trend',
}

def read_audio(path):
    path = Path(path)
    try:
        if path.suffix.lower() == '.hea':
            record = wfdb.rdrecord(str(path.with_suffix('')))
            x, sr = record.p_signal, int(record.fs)
        else:
            x, sr = sf.read(path, always_2d=True, dtype='float64')
    except Exception as exc:
        raise ValueError(f'Cannot read audio: {path.name}: {exc}') from exc
    return np.asarray(x, dtype=np.float64), sr

def preprocess(x, sr):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('Audio must contain finite mono or stereo samples.')
    if sr < SR:
        raise ValueError('Audio sample rate must be at least 8000 Hz.')
    if len(x) < 3 * sr:
        raise ValueError('Record at least three seconds of sustained /a/.')
    if sr != SR:
        divisor = gcd(int(sr), SR)
        x = resample_poly(x, SR // divisor, int(sr) // divisor)
    start = (len(x) - SAMPLES) // 2
    x = x[start:start + SAMPLES].copy()
    x -= x.mean()
    if np.sqrt(np.mean(x*x)) < 1e-7:
        raise ValueError('The central three seconds are silent or too quiet.')
    return x

def audio_fingerprint(x):
    # Detect identical crops even with a constant gain change.
    y = x / max(float(np.max(np.abs(x))), 1e-12)
    return hashlib.sha256(np.round(y * 100000).astype('<i4').tobytes()).hexdigest()

def extract(x):
    cfg = SPEC['mfcc'].copy()
    count = cfg.pop('count')
    cfg.pop('include_c0')
    mfcc = librosa.feature.mfcc(y=np.asarray(x, dtype=np.float64), sr=SR,
                                 n_mfcc=count, **cfg).astype(np.float32)
    sound = parselmouth.Sound(x, sampling_frequency=SR)
    q = np.full(6, np.nan, dtype=np.float64)
    errors = []
    def attempt(i, fn):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                value = float(fn())
            if np.isfinite(value):
                q[i] = value
            else:
                errors.append(f'{QUALITY_NAMES[i]}: undefined')
        except (parselmouth.PraatError, ValueError, RuntimeError) as exc:
            errors.append(f'{QUALITY_NAMES[i]}: {exc}')
    def cpps():
        pc = call(sound, 'To PowerCepstrogram', 60, .002, 4000, 50)
        return call(pc, 'Get CPPS', True, .01, .001, 60, 500, .05,
                    'Parabolic', .001, .05, 'Straight', 'Robust')
    attempt(0, cpps)
    attempt(1, lambda: call(sound.to_harmonicity_cc(time_step=.01, minimum_pitch=60), 'Get mean', 0, 0))
    try:
        pp = call(sound, 'To PointProcess (periodic, cc)', 60, 500)
        attempt(2, lambda: call(pp, 'Get jitter (local)', 0, 0, 1/500, 1/60, 1.3))
        attempt(3, lambda: call([sound, pp], 'Get shimmer (local)', 0, 0, 1/500, 1/60, 1.3, 1.6))
    except parselmouth.PraatError as exc:
        errors.append(f'Pulse detection: {exc}')
    pitch = sound.to_pitch_ac(time_step=.01, pitch_floor=60, pitch_ceiling=500)
    f0 = pitch.selected_array['frequency']
    voiced = f0[(f0 > 0) & np.isfinite(f0)]
    if len(voiced):
        q[4] = np.median(voiced)
        q[5] = np.percentile(voiced, 75) - np.percentile(voiced, 25)
    else:
        errors.append('No reliable pitch periods; pitch measurements missing.')
    if not np.isfinite(mfcc).all():
        raise ValueError('Nonfinite MFCC values.')
    return mfcc, q, {'measurement_failures': errors,
                     'missing_quality_features': [QUALITY_NAMES[i] for i in np.flatnonzero(~np.isfinite(q))],
                     'voiced_frame_fraction': float(len(voiced) / max(1, len(f0))),
                     'status': 'warning' if errors else 'usable',
                     'scope': 'Signal checks do not verify vowel identity or clinical recording suitability.'}

def extract_path(path):
    audio, sr = read_audio(path)
    x = preprocess(audio, sr)
    seq, quality, status = extract(x)
    return seq, quality, status, x
