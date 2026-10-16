# Reproduce and validate

Start with the environment setup in the root README. Run commands from the repository root. The examples below use Windows PowerShell; on Linux/macOS substitute `.venv/bin/python`.

## Run the website without training

```powershell
& '.\.venv\Scripts\python.exe' -m streamlit run app.py
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Some integrity checks require original source audio. They skip when it is absent. To restore the verified public release and run those checks:

```powershell
& '.\.venv\Scripts\python.exe' download_data.py --individual --s3
& '.\.venv\Scripts\python.exe' -m pytest -q
```

The downloader verifies release checksums. Raw audio and regenerated feature caches remain ignored by Git. A clean checkout includes all models, metadata and samples needed for website inference.

## Reproduce development in a separate experiment copy

Preserve this repository's saved models/results first. The following commands can regenerate outputs, so run them in a separate working copy when reproducing an experiment:

```powershell
& '.\.venv\Scripts\python.exe' download_data.py --individual --s3
& '.\.venv\Scripts\python.exe' run.py prepare
& '.\.venv\Scripts\python.exe' run.py train
& '.\.venv\Scripts\python.exe' run.py robustness
```

The original binary selection compares ten settings over three fixed development folds. Feature transformations, class weights and stopping decisions use training subsets. The same development folds selected settings, so development results have selection bias.

`python -m voice_review.laryngeal` implements the separate three-class experiment. Its guard refuses to overwrite existing subtype metrics. To reproduce it, use another experiment copy with the same metadata, splits and generated development features, but without models/laryngeal/ and results/laryngeal/ outputs. Do not remove original provenance in the submitted copy.

`models/frozen_experiment.json` covers the original scientific code, selected models, folds and lockfile. `models/laryngeal/provenance.json` separately covers the subtype artifacts. Existing tracked scientific files were copied byte-for-byte. The portable-checkout adaptations affect tests, documentation and launchers only.

## Final evaluation

The original 42-record test partition has not been evaluated. Do not run final evaluation during a demo. The original pipeline exposes `run.py freeze` and `run.py evaluate-final --confirm-final`; use them only after development is complete and the protocol is finalized. Final evaluation has an audit lock and must not be repeated for model selection. Source data and the required environment must first be restored. Three-class final evaluation is not implemented by that binary command.

## Environment records

Git attributes preserve the exact bytes of scientific source, models, metadata, results and the original environment lock, including their original line endings. This keeps frozen SHA-256 records valid after cloning on another platform.

The clean repository was checked before upload: **41 tests passed and two raw-source-audio checks skipped**. Restoring the original dataset enables those two additional checks.

- Python 3.12 was used locally.
- requirements.txt: pinned direct dependencies.
- requirements-lock.txt: original scientific training environment, retained for frozen integrity.
- requirements-web-lock.txt: complete later Windows website environment; includes Windows-specific packages.
- Saved joblib models must be loaded only from trusted project artifacts.
