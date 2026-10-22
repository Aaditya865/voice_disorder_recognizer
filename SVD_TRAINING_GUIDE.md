# Local SVD training and prediction

This is a separate, complete training script for your downloaded five-label SVD
subset. It leaves the earlier VOICED experiment and website models intact.
The DNN here is scikit-learn's MLPClassifier, as requested, rather than the
earlier temporal CNN. All code is in `svd_training.py`.

## Run on this computer (PowerShell)

```powershell
cd "C:\Users\adity\Documents\Codex\2026-10-15\i-x20\outputs\voice_disorder_recognizer"
& "..\..\work\venv\Scripts\python.exe" svd_training.py predict "C:\Users\adity\.gemini\antigravity\scratch\svd_voice_dataset\audio_wav\1-a_n.wav"
```

Training has already completed in `artifacts_svd_local`. To reproduce the same
fixed experiment in a separate directory:

```powershell
& "..\..\work\venv\Scripts\python.exe" svd_training.py train --output artifacts_svd_reproduction
```

Reproduction is not a new independent test. Do not change model settings based
on the existing test results and then claim an unbiased test on the same speakers.
The script refuses to reuse an output folder after its test evaluation starts.

## Fresh Python 3.12 environment

```powershell
py -3.12 -m venv .venv_svd
& ".\.venv_svd\Scripts\python.exe" -m pip install -r requirements-svd.txt
& ".\.venv_svd\Scripts\python.exe" svd_training.py train --audio-dir "C:\Users\adity\.gemini\antigravity\scratch\svd_voice_dataset\audio_wav" --labels "C:\Users\adity\.gemini\antigravity\scratch\svd_voice_dataset\recordings.csv" --output artifacts_svd_reproduction
```

## Features and methodological choices

- 16 kHz, DC removal, peak amplitude normalization; central five seconds at most.
  Clips between 0.5 and 5 seconds use their available length. Shorter clips are
  excluded with a reason. Nothing is padded or repeated to simulate longer audio.
- 42 features: 13 MFCC means and standard deviations (including the zeroth
  coefficient); mean/std of centroid, spectral bandwidth (spread), rolloff,
  RMS and ZCR; F0 median/std/IQR and voiced fraction; Praat HNR; frame-amplitude
  variation proxy. The proxy is not clinical cycle-to-cycle shimmer.
- Peak normalization removes absolute loudness information. RMS is therefore a
  relative measurement. Undefined pitch/HNR measurements remain missing until
  training-only median imputation; missing-value indicators are included.
- Outer fixed fold 0 of five-fold StratifiedGroupKFold approximates 80/20;
  inner fixed fold 0 of four-fold grouped splitting selects settings. Speaker
  IDs and analyzed-audio fingerprints are asserted disjoint. All recordings of
  a speaker with multiple diagnoses remain together.
- GMM: separate diagonal distribution per diagnosis, equal class priors, one
  versus two components. MLP: (64,32) versus (128,64) hidden layers, class-balanced
  sample weights, regularization. Built-in random early-stopping validation is
  disabled to prevent speaker leakage. Imputation/scaling are refit on training
  only for each candidate, then refit on all development data for final models.
- Both families are evaluated on the frozen test set once, after validation
  selects configurations and the exported family. Reports are recording-weighted;
  confidence intervals resample whole speakers to account for repeated recordings.
- No augmentation is used in this requested baseline. The sample count always
  refers to real supplied recordings, not synthetic examples.

## Results from the executed run

Input: 1,356 recordings, 1,236 unique speakers. Five short recordings excluded.
Usable: 1,351 recordings; 1,081 development and 270 test recordings.

| Model | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|
| GMM | 0.359499 | 0.341213 |
| MLP (DNN) | 0.334127 | 0.332724 |

MLP was selected before testing: validation balanced accuracy 0.302730 compared
with GMM 0.302476. The difference is tiny and does not establish superiority.
GMM's higher observed test score was not used to change the deployed selection.

These are modest five-class results. Functional and hyperfunctional dysphonia
are related labels, and similar acoustic characteristics occur across disorders.
The dataset is imbalanced, some people have different labels across sessions,
and voice quality alone cannot establish a diagnosis. No external dataset,
phone-microphone validation, calibrated confidence or clinical validation is
provided. Specific causes of confusion require additional error analysis.

## Saved outputs

`artifacts_svd_local/` contains:

- `svd_voice_model.joblib`: selected MLP, imputer, scaler, label encoder, feature
  specification, training identities, versions and provenance in one dictionary.
- `svd_gmm.joblib`, `svd_dnn.joblib`: both trained pipelines.
- `dataset_audit.csv`, `usable_class_counts.csv`, `excluded_files.csv`.
- `partition_manifest.csv`, `splits.json`, `split_counts.csv`.
- `validation_results.json`, `selection.json`.
- `test_metrics_gmm.json`, `test_metrics_dnn.json`: metrics and speaker-bootstrap
  confidence intervals; per-class precision, recall, F1 and support.
- `classification_report_*.csv`, `test_predictions_*.csv`,
  `confusion_matrix_*.png`, `model_comparison.csv`.
- `feature_cache/`: reusable deterministic extraction cache; not needed by the app.

## Streamlit integration (complete small example)

Put this in a separate `svd_demo.py` beside `svd_training.py`. Install Streamlit
in that environment if needed; the existing project environment already has it.

```python
from pathlib import Path
import joblib
import streamlit as st
from svd_training import predict_voice

ROOT = Path(__file__).resolve().parent
st.title("SVD voice disorder recognizer")
st.caption("Academic project, not a medical diagnosis.")

@st.cache_resource(max_entries=1)
def load_model():
    return joblib.load(ROOT / "artifacts_svd_local/svd_voice_model.joblib")

upload = st.file_uploader("Upload a sustained aa vowel", type=["wav"])
if st.button("Analyze Voice", disabled=upload is None):
    try:
        result = predict_voice(upload, bundle=load_model())
        st.subheader(result["diagnosis"])
        st.write("Model:", result["model"])
        st.write("Model scores (%):", result["probabilities_percent"])
        st.caption(result["notice"])
        if result["uncertain"]:
            st.warning("Uncertain, result not reliable.")
        st.audio(upload)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
```

Run with `python -m streamlit run svd_demo.py`. Percentage scores sum to 100,
but are uncalibrated model outputs, not medical probabilities. High scores may
still be wrong. The 60% warning threshold is an explicitly heuristic display rule.
Do not use training recordings to demonstrate claims of unseen-speaker accuracy.

Dataset: https://zenodo.org/records/16874898 (SVD). Labels are taken from the
user-supplied CSV; this run did not independently re-establish clinical diagnoses.
