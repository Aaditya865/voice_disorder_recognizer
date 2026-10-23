# Reproducing the two-source experiment

## Scope

All 1,420 examples are development data. Repeating these commands does not create an independent test.
Historical artifacts remain unchanged. Use a fresh output directory; never delete guards to tune against a claimed test set.

## Requirements and inputs

Use Python 3.12 and install `requirements-training.txt`. Obtain the SVD 50 kHz WAVs used by the cohort,
the SVD source metadata CSV, and the VOICED .hea/.dat/-info.txt files. Source links are in README.md.
The wrapper checks the frozen source audio and metadata hashes before proceeding; different conversions are a new experiment, not exact reproduction.
Full SVD/VOICED research audio is intentionally excluded from GitHub.

## Prepared laptop: fresh reproduction

Run from the project folder. This creates a new run; it does not replace website models.

```powershell
& '..\..\work\venv\Scripts\python.exe' reproduce_v3.py --svd-audio-root 'C:\Users\adity\.gemini\antigravity\scratch\svd_voice_dataset\audio_wav' --voiced-root 'data/raw' --source-meta '..\..\work\svd\voice_data.csv' --output '..\..\work\v3-reproduction'
```

On another machine use your `.venv` Python and replace the three input paths. Add `--prepare-only` to validate inputs and build the new workspace without training.
Inside that workspace, the stages are `python svd_v3.py features`, `python svd_v3.py train`, and `python svd_v3.py report`.
An interrupted fit can resume with the same train command before the fixed ten-hour deadline. Complete releases refuse retraining.
Cache keys include source hashes, feature hashes, participant IDs, configuration and seed.

## Prediction

```powershell
& '..\..\work\venv\Scripts\python.exe' predict_release.py 'demo_v3/SVD_healthy_2.wav'
```

Use an actual filename from demo_v3/manifest.csv if different. The helper `voice_review.release_web.predict_voice`
accepts a path or Streamlit UploadedFile. Pass a cached ReleaseService to avoid loading models for every call.

## Validation

```powershell
& '..\..\work\venv\Scripts\python.exe' -m pytest tests/test_v3.py tests/test_v3_release.py -q
```

Nested selection uses equal source importance, group-disjoint preprocessing/calibration/stopping and limited augmentation.
Model, input and code fingerprints are included in artifacts_v3/. See V3_REPORT.md for criteria, real results and limitations.
