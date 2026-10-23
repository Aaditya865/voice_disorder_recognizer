# Reproduce the separate V4 experiment

The working website is still launched with `app.py`. Do not replace its artifacts manually.

## Run the current website on this laptop

```powershell
cd 'C:\Users\adity\Documents\Codex\2026-10-15\i-x20\outputs\voice_disorder_recognizer'
& '..\..\work\venv\Scripts\python.exe' -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Launch before presenting and analyze one development example to warm imports and audio libraries. Example predictions are not an accuracy test.

## Rebuild in a new workspace

Use Python 3.12 and a separate environment on another machine. Installing the experimental dependencies is not required to run the retained v3 website.

```powershell
py -3.12 -m venv .venv-v4
& '.\.venv-v4\Scripts\python.exe' -m pip install -r requirements-v4.txt
```

On the prepared laptop, the exact input paths are:

```powershell
& '..\..\work\venv\Scripts\python.exe' reproduce_v4.py `
  --output 'C:\Users\adity\Documents\Codex\2026-10-15\i-x20\work\v4-reproduction-full' `
  --svd-audio-root 'C:\Users\adity\.gemini\antigravity\scratch\svd_voice_dataset\audio_wav' `
  --voiced-root '.\data\raw' `
  --source-meta 'C:\Users\adity\Documents\Codex\2026-10-15\i-x20\work\svd\voice_data.csv' `
  --extra-audio-root '.\experiments\v4\new_audio' `
  --wavlm-root '.\experiments\v4\wavlm'
```

Add `--prepare-only` for a quick source-hash/input check without training. Choose a fresh output directory on every reproduction. The script rebuilds original-cohort features, WavLM embeddings, new features, nested selection and the development report. It does not reopen the challenge automatically. The sealed challenge has a single evaluation marker; after it has been evaluated it must not be used for further selection.

On another machine replace the three data roots and metadata path. If `--wavlm-root` is omitted, the embedding stage downloads the pinned Microsoft model once. Install requirements-v4.txt in the Python environment used to run the command. The original laptop also has isolated optional packages in work/v4_dependencies; the wrapper passes that directory to child processes when available.

## Data acquisition

- Original SVD and new SVD /a/ recordings: https://zenodo.org/records/16874898 (CC BY 4.0). Use the official NSP-to-WAV conversion; exact hashes are required.
- VOICED: https://physionet.org/content/voiced/1.0.0/ . Retain the .hea, .dat and -info.txt files.
- Official SVD metadata: https://stimmdb.coli.uni-saarland.de/data/voice_data.csv . Keep this locally; full birth-date metadata is not included in the release package.
- Pinned WavLM revision: 4c66d4806a428f2e922ccfa1a962776e232d487b, https://huggingface.co/microsoft/wavlm-base-plus . The model is frozen, not fine-tuned.

The public package excludes raw training audio, embedding matrices and fitting caches. download_manifest.csv identifies successful/failed records. The experiment must use the frozen successful cohort; silently replacing missing recordings changes the experiment. Rate-limit failures are not evidence that the original data are unusable.

## Commands for an already prepared experiment

```powershell
python svd_v4.py prepare
python svd_v4.py train
python svd_v4_results.py report
python build_v4_report.py
python -m pytest tests/test_v4.py -q
```

`prepare` refuses an already-frozen protocol. `train` refuses a completed release and resumes cached fits only before its original six-hour deadline. Code, cohort, features and configuration hashes invalidate cached fits. Do not change the clock or protocol to extend an experiment silently.

Completed outputs include candidate_comparison.csv, candidate_inner_summary.csv, ensemble_inner_metrics.csv, nested_predictions.csv, before_after.csv, calibration_bins.csv, subgroup results and the separate challenge report. Candidate inner-validation scores are tuning diagnostics, not independent test scores for every architecture.
