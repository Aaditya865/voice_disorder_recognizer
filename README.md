# Voice Disorder Recognizer — current two-source release

Academic project, not a medical diagnosis.

The current app compares GMM and DNN binary voice classifiers. The selected configuration is **MLP_medium_8000**.
It accepts a sustained aa vowel, returns a research label or **Uncertain**, and offers three exploratory pathology-group scores.
The subtype model cannot reliably identify a specific disorder and always remains uncertain.

## Latest experiment: v4 was not promoted

A completed nested development experiment added 198 SVD speakers and compared 14 configurations plus an ensemble, including CNNs and frozen WavLM embeddings. Its equal-source balanced accuracy on the original participants fell by 0.03115444915458554. It failed the predefined promotion criteria, so **the website retains v3 and all its uncertainty safeguards**.

See [measured before/after results and limitations](V4_REPORT.md) and [complete experiment reproduction instructions](V4_RUN.md). The new-speaker challenge contains pathological recordings only; it is not an external binary accuracy test. Neither more data nor a larger model automatically improved this project.

## Run the prepared laptop

Double-click **OPEN_WEBSITE.cmd**, or run the commands in [RUN_WEBSITE.md](RUN_WEBSITE.md).
The supplied models load without training or internet. Use **app.py**, not streamlit_app.py, as the entrypoint.

## Install elsewhere

Use Python 3.12. From this folder:

```powershell
py -3.12 -m venv .venv
& ./.venv/Scripts/python.exe -m pip install -r requirements.txt
& ./.venv/Scripts/python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

CPU PyTorch is included in requirements.txt. Model files are version-sensitive: use the supplied versions.
The CPU-wheel requirement targets Windows/Linux; macOS installation has not been validated.

## Evidence and scope

- Development: 1,212 SVD speakers and 208 VOICED participant proxies; total 1,420.
- Four outer grouped folds evaluate a procedure that selects configurations using three inner grouped folds.
- SVD pooled outer balanced accuracy: 0.689509; VOICED: 0.627048.
- These are **development results**, not an untouched external test of the final fitted model.
- All previously evaluated VOICED recordings are now development data.
- The historical v2 external failure remains unchanged: balanced accuracy 0.525; 4/30 pathological cases detected.
- Age/sex and recording differences remain possible confounders. No accuracy percentage is guaranteed.

See [complete report](V3_REPORT.md), [training/reproduction guide](V3_TRAINING_GUIDE.md), and [demo script and questions](DEMO_AND_QA.md).
Full-precision metrics, predictions, source snapshots, hashes and audits are in artifacts_v3/.

## Main files

```text
app.py                         Current Streamlit entrypoint
voice_review/release_web.py    Shared inference service and predict_voice helper
svd_v3.py                     Audit, extraction, nested training and reporting
svd_training.py               Shared original audio/feature utilities
svd_advanced.py                Shared model utilities and historical v2 experiment
reproduce_v3.py               Reproduce in a fresh directory using acquired data
predict_release.py            Offline WAV/MP3 inference
artifacts_v3/                 Frozen current models, configuration and evidence
artifacts_svd_v2/             Preserved historical experiment and subtype model
demo_v3/                     Four labeled DEVELOPMENT demonstrations
requirements.txt             Website/runtime dependencies
requirements-training.txt    Additional training/test dependencies
```

Old notebooks, results/, app_voiced_legacy.py, app_svd_v1.py and app_svd_v2.py describe earlier experiments.
Do not mix their metrics or demonstration claims with the current release. README_HISTORY.md preserves earlier documentation.

## Data and reference

- SVD: https://zenodo.org/records/16874898 — Pützer and Barry, CC BY 4.0.
- SVD metadata: https://stimmdb.coli.uni-saarland.de/data/voice_data.csv
- VOICED: https://physionet.org/content/voiced/1.0.0/
- Gupta et al. (2026): https://doi.org/10.3389/fdgth.2026.1800132 — a scoping review, not the source of a winning model or target accuracy.

Full research audio is not bundled. Obtain it from the official sources and preserve dataset terms and attribution.

Current slides: [Review presentation](presentation/Voice_Disorder_Review_24_October.pptx), with [presenter notes](presentation/Presenter_Notes.md).
