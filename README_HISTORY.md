# Voice Disorder Recognizer — current SVD website

The main `app.py` now runs the trained five-class SVD GMM and DNN/MLP pipelines.
Supported labels: Healthy, Functional Dysphonia, Hyperfunctional Dysphonia,
Laryngitis and Vocal Fold Paresis. Laryngitis is a specific laryngeal label;
this is not comprehensive detection of all laryngeal disorders.

## Run

Use Python 3.12:

```sh
pip install -r requirements.txt
python -m streamlit run app.py
```

Models are loaded once from `artifacts_svd_local/`. WAV and MP3 uploads use the
same 16 kHz, normalized acoustic feature extraction and saved transforms as
training. MP3 compression effects are not externally validated. The app shows
Healthy/Pathological derived from the five-class decision, top-three scores,
plain-language descriptions, model comparison, uncertainty warnings and audio.
Scores are uncalibrated; a high score does not establish reliability.

## Measured results

1,356 supplied recordings; five short clips excluded. Development: 1,081
recordings / 987 speakers. Reserved test: 270 recordings / 246 speakers.
No speaker overlap. Labels come from the supplied CSV. Repeated speakers are
kept together, including speakers with different diagnoses across sessions.

| Model | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|
| GMM | 35.9499% | 0.341213 |
| DNN / MLP | 33.4127% | 0.332724 |

MLP remains primary because validation selected it before the test evaluation;
its validation advantage was very small. These are modest five-class results,
not the previous binary VOICED results and not clinical diagnostic accuracy.
Functional and hyperfunctional labels overlap conceptually and acoustically.
External validation and confidence calibration remain outstanding.

See [training and prediction instructions](SVD_TRAINING_GUIDE.md),
[full reports](artifacts_svd_local/), and [deployment instructions](docs/DEPLOYMENT.md).

SVD: https://zenodo.org/records/16874898 — Pützer and Barry; hosted by Essen
University Hospital; CC BY 4.0. No raw SVD audio is bundled in this repository.
The existing `samples/` files are legacy VOICED examples; their old expected
predictions and speaker-routing descriptions do not apply to this SVD app.

## Previous VOICED experiment (archived documentation below)

`app_voiced_legacy.py` preserves the old demo. Install `requirements-legacy.txt`
to reproduce that experiment. Its results describe different data and models.

---

<!-- website-update -->
# Website review update — 16 October 2026

The Streamlit website is implemented. Start with **OPEN_WEBSITE.cmd** or **RUN_WEBSITE.md**. **WEBSITE_IMPLEMENTATION.md** contains complete new source and packaging instructions. **DEMO_AND_QA.md** contains the review script. Existing binary models and partitions are preserved; the laryngitis-label experiment is separate and preliminary. See **LARYNGEAL_REPORT.md** for its weak per-class results. The final test remains sealed. The earlier project notes below describe the prior notebook milestone.

---

# Voice Disorder Recognizer

An academic comparison of a Gaussian Mixture Model and a compact convolutional DNN on VOICED sustained vowels. It is not a diagnostic tool.

## Start here for the review

1. Open **Review.html** (or double-click **OPEN_REVIEW.cmd**). This portable presentation contains actual development results, audio examples, methodology, limitations and charts. It works without an internet connection.
2. To run live inference, launch **OPEN_NOTEBOOK.cmd** and run the notebook cells in order. In section 3, choose an example and click **Analyze example**, or use **Choose audio** then **Analyze chosen file**. The notebook uses saved models; it does not retrain or open the final test set.
3. Read **REVIEW_TALK_TRACK.md** for a short presentation guide and **REPORT.md** for the academic write-up.

The HTML is a presentation of saved results. Live predictions run in the notebook or command line. Predictions in the HTML audio examples are out-of-fold. The notebook offers speaker-held-out fold models and full-development refits; it explicitly identifies which models are being used.

The live controls show both predictions, disagreement, the exact three-second crop, MFCCs, measurement failures and acoustic values. The file picker accepts mono/stereo WAV/FLAC, 3–60 seconds, at least 8 kHz, at most 10 MB. Files go only to your local Jupyter process. Temporary uploaded input files are removed after analysis, but displayed audio/results may be saved in notebook outputs: clear personal recordings before sharing. If controls are unavailable, restart JupyterLab after installing dependencies, run the cells again, or use the local-path alternative in section 4.

For an unseen-to-training demonstration, keep **Speaker-held-out demo** selected and use `demo_audio/voice003.wav`. The app verifies the recording fingerprint and routes it to its validation-fold models. This speaker was absent from fitting and early stopping, although validation informed model-setting selection. This is not an independent final test. For a new external recording, choose **Full-development models**. Those refitted models have seen the bundled examples. Command-line equivalent: `run.py predict --audio demo_audio/voice003.wav --validation-demo`. See `demo_audio/README.md` for provenance.

`ROBUSTNESS.md` reports all four development seed runs, including individual prediction instability. It does not replace the original selected models or their reported results.

## Local environment

The prepared environment is located two directories above this project at `work/venv`. In PowerShell, with this project as the current directory:

```powershell
$voicePython = (Resolve-Path '..\..\work\venv\Scripts\python.exe').Path
& $voicePython -m jupyterlab Voice_Disorder_Review.ipynb
```

For another computer, install Python 3.12, create a virtual environment and install dependencies. CPU-only PyTorch is sufficient:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-lock.txt` records exact versions actually used on this computer. The short requirements list is more portable; reproduce the locked environment when comparing results exactly. Python/library changes may affect results. The launchers also recognize a project-local `.venv`.

## Reproduce the development experiment

From the project directory, using the prepared environment:

```powershell
& $voicePython download_data.py --individual --s3
& $voicePython run.py prepare
& $voicePython run.py train
& $voicePython run.py robustness
& $voicePython run.py report
& $voicePython -m pytest tests -q
```

The downloader uses PhysioNet's documented public S3 mirror and verifies the supplied SHA-256 checksums. The archive path is available by omitting both options. No private credentials are needed. Attribution is required; see REFERENCES.txt and DATA_CARD.md.

The training command tests 10 configurations over three fixed participant-disjoint folds (two feature sets; three GMM component settings and two CNN dropout settings). Model selection uses mean fold balanced accuracy. It also creates out-of-fold predictions, feature ablations, permutation importance, bootstrap intervals and demographic/error analyses. Runtime depends on the CPU; feature extraction is cached with source and configuration hashes.

The optional robustness command repeats the selected settings at three additional fixed seeds, using the existing development folds. It saves its protocol before fitting, retains every run, and verifies that the original models are unchanged. Completed results are reused only if the input hashes match. If you revise the experiment, preserve the old robustness outputs before starting a separately identified analysis; do not silently overwrite earlier experiments. The seed spread describes training randomness, not population confidence intervals or external validation.

## Predict a recording

```powershell
& $voicePython run.py predict --audio 'C:\path\to\vowel.wav'
```

Input: sustained /a/, at least 3 seconds, WAV/FLAC or a WFDB `.hea` with its `.dat`, sample rate at least 8 kHz. Higher sample rates are resampled. Output: quality status, healthy/pathological prediction from each model, raw decision score and an uncalibrated model score. A recording's technical acceptability does not prove it contains the correct task. No clinical interpretation is provided. Scores are not calibrated medical probabilities. Raw scores are not comparable between model families.

## Final evaluation — keep sealed for tomorrow's review

When development is complete and the design is frozen:

```powershell
& $voicePython run.py freeze
& $voicePython run.py evaluate-final --confirm-final
```

This validates frozen hashes, creates an atomic audit lock, extracts the test features for the first time, evaluates both frozen models, calculates paired participant bootstrap intervals, saves predictions, and refreshes the report. The command refuses a second evaluation. Do not remove the lock to retry experiments or tune to the test set. If evaluation fails after opening the test set, retain the lock and investigate the failure before any recovery; no automatic retry is provided.

The freeze step records model, split, source-code and environment hashes. It is forbidden after final evaluation starts. Editing the inference code or frozen artifacts after freezing causes evaluation to refuse to run; freeze only after finishing development checks.

Development OOF intervals are approximate and condition on chosen models; they do not remove model-selection bias. The independent final test is small, and it cannot establish cross-dataset generalization.

## Files

- `voice_review/`: preprocessing, features, models, experiment, metrics and reporting.
- `data/manifest.csv`, `data/audit.json`, `data/splits.json`: record-level audit and fixed partitions.
- `data/raw/`: original verified public release files; `data/features/`: development-only feature cache until final evaluation.
- `models/`: refitted development models, validation-fold models, frozen experiment hashes.
- `results/`: measured predictions, searches, error analyses, intervals, figures and environment versions.
- `Voice_Disorder_Review.ipynb`: reusable live demonstration.
- `Review.html`, `REPORT.md`: generated presentation and academic report.
- `tests/`: synthetic unit-test fixtures and checks against the actual partitions. Synthetic data are never used for reported model results.

Saved joblib files are Python serialization artifacts: load only the project-generated files you trust, never models from arbitrary uploads.
