"""Benchmark and document the completed release. Activate only after tests pass."""
from pathlib import Path
import argparse,io,json,shutil,time
import numpy as np
import pandas as pd
from svd_v3 import audio_bytes,dump
from voice_review.release_web import ReleaseService,predict_voice
from build_v3_report import build

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts_v3'

def benchmark():
    frame=pd.read_csv(ROOT/'data/v3/featured_manifest.csv')
    selected=[]
    for source in ('SVD','VOICED'):
        for label in (0,1):
            selected.extend(frame[(frame.source==source)&(frame.label==label)].head(3 if source=='SVD' else 2).to_dict('records'))
    if len(selected)!=10:raise ValueError('Expected ten fixed development recordings')
    start=time.perf_counter();service=ReleaseService(ROOT);load_seconds=time.perf_counter()-start
    service.analyze(audio_bytes(selected[0]['path']),'warmup.wav')
    results=[]
    for row in selected:
        content=audio_bytes(row['path']);result=service.analyze(content,'recording.wav')
        repeated=predict_voice(io.BytesIO(content),service=service)
        np.testing.assert_allclose(result['pathological_probability'],repeated['pathological_probability'])
        results.append({'record_id':row['record_id'],'source':row['source'],'true_label':int(row['label']),
            'leading_label':result['leading_label'],'uncertain':result['binary_uncertain'],
            'seconds':result['elapsed_seconds'],'model_score':result['score']})
    times=[r['seconds'] for r in results]
    dump(OUT/'runtime_benchmark.json',{'scope':'Ten fixed development examples; not an accuracy evaluation',
        'model_load_seconds':load_seconds,'median_seconds':float(np.median(times)),
        'p95_seconds':float(np.percentile(times,95)),'max_seconds':max(times),
        'median_under_five_seconds':bool(np.median(times)<5),'offline_website_prediction_agreement':True,'recordings':results})
    samples=ROOT/'demo_v3';samples.mkdir(exist_ok=True)
    sample_rows=[]
    for source in ('SVD','VOICED'):
        for label in (0,1):
            row=next(r for r in selected if r['source']==source and r['label']==label)
            name=f"{source}_{'healthy' if label==0 else 'pathological'}_{row['record_id'].split(':')[1]}.wav"
            (samples/name).write_bytes(audio_bytes(row['path']))
            sample_rows.append({'file':name,'record_id':row['record_id'],'diagnosis':row['diagnosis'],'role':'development demonstration'})
    pd.DataFrame(sample_rows).to_csv(samples/'manifest.csv',index=False)
    (samples/'README.md').write_text('# Development demonstration recordings\n\nThese four files were selected by source and label order, not by correct predictions. '+
        'All are project development examples, not untouched tests. The supplied diagnosis describes the dataset label; a model may disagree. '+
        'See manifest.csv for identities. Do not claim that these files establish accuracy.\n\n'+
        'SVD: Pützer and Barry, https://zenodo.org/records/16874898 (CC BY 4.0). '+
        'VOICED: https://physionet.org/content/voiced/1.0.0/; retain its dataset attribution/license.\n',encoding='utf-8')
    print(json.dumps({'load_seconds':load_seconds,'median_seconds':float(np.median(times))},indent=2))

def docs():
    build();release=json.loads((OUT/'release.json').read_text());s=json.loads((OUT/'nested_summary.json').read_text());winner=release['selection']['winner']['id']
    readme=f'''# Voice Disorder Recognizer — current two-source release

Academic project, not a medical diagnosis.

The current app compares GMM and DNN binary voice classifiers. The selected configuration is **{winner}**.
It accepts a sustained aa vowel, returns a research label or **Uncertain**, and offers three exploratory pathology-group scores.
The subtype model cannot reliably identify a specific disorder and always remains uncertain.

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
- SVD pooled outer balanced accuracy: {s['SVD']['balanced_accuracy']:.6f}; VOICED: {s['VOICED']['balanced_accuracy']:.6f}.
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
'''
    current=ROOT/'README.md'
    if not (ROOT/'README_HISTORY.md').exists():shutil.copy2(current,ROOT/'README_HISTORY.md')
    current.write_text(readme,encoding='utf-8')
    run='''# Run the current website

Prepared laptop (no installation or training needed):

```powershell
Set-Location 'C:\\Users\\adity\\Documents\\Codex\\2026-10-15\\i-x20\\outputs\\voice_disorder_recognizer'
& '..\\..\\work\\venv\\Scripts\\python.exe' -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. Alternatively double-click OPEN_WEBSITE.cmd.
If the port is already occupied, use the existing server or stop it with Ctrl+C; do not start competing copies.

Upload one of the four WAV files in demo_v3/ and press Analyze Voice. These are development demonstrations, not unseen tests.
Use a sustained aa vowel. WAV/MP3, mono/stereo, 8–192 kHz, minimum 0.5 seconds, maximum upload 20 MB.
Only the central five seconds are analyzed. The selected model supplies its own 8/16 kHz feature specification.
The subtype model keeps its original 16 kHz pipeline. Scores do not establish medical probabilities.

The app caches models once, processes uploads in memory and does not save uploaded audio to disk.
Browser recording needs microphone permission; rely on supplied WAVs for the review.

To use the preserved local fallback: replace `app.py` in the launch command with `app_svd_v2.py`.
The fallback is an earlier research model with documented weak external performance; it must not be presented as the current experiment.

Installation on another Windows laptop is in README.md. Cloud availability is separate from local operation.
'''
    (ROOT/'RUN_WEBSITE.md').write_text(run,encoding='utf-8')
    guide='''# Reproducing the two-source experiment

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
& '..\\..\\work\\venv\\Scripts\\python.exe' reproduce_v3.py --svd-audio-root 'C:\\Users\\adity\\.gemini\\antigravity\\scratch\\svd_voice_dataset\\audio_wav' --voiced-root 'data/raw' --source-meta '..\\..\\work\\svd\\voice_data.csv' --output '..\\..\\work\\v3-reproduction'
```

On another machine use your `.venv` Python and replace the three input paths. Add `--prepare-only` to validate inputs and build the new workspace without training.
Inside that workspace, the stages are `python svd_v3.py features`, `python svd_v3.py train`, and `python svd_v3.py report`.
An interrupted fit can resume with the same train command before the fixed ten-hour deadline. Complete releases refuse retraining.
Cache keys include source hashes, feature hashes, participant IDs, configuration and seed.

## Prediction

```powershell
& '..\\..\\work\\venv\\Scripts\\python.exe' predict_release.py 'demo_v3/SVD_healthy_5.wav'
```

Use an actual filename from demo_v3/manifest.csv if different. The helper `voice_review.release_web.predict_voice`
accepts a path or Streamlit UploadedFile. Pass a cached ReleaseService to avoid loading models for every call.

## Validation

```powershell
& '..\\..\\work\\venv\\Scripts\\python.exe' -m pytest tests/test_v3.py tests/test_v3_release.py -q
```

Nested selection uses equal source importance, group-disjoint preprocessing/calibration/stopping and limited augmentation.
Model, input and code fingerprints are included in artifacts_v3/. See V3_REPORT.md for criteria, real results and limitations.
'''
    (ROOT/'V3_TRAINING_GUIDE.md').write_text(guide,encoding='utf-8')
    (ROOT/'docs/DEPLOYMENT.md').write_text('''# Deploy the current release

Streamlit Community Cloud: repository Aaditya865/voice_disorder_recognizer, branch main, entrypoint app.py, Python 3.12.
Install root requirements.txt, including CPU PyTorch. Keep artifacts_v3/, artifacts_svd_v2/, artifacts_svd_local/partition_manifest.csv,
svd_v3.py, svd_training.py, svd_advanced.py and voice_review/ in the deployment.
No training or download runs on startup. A successful GitHub push does not verify cloud deployment.
Check hosting logs and test a real upload after rebuilding. The local fallback is documented in RUN_WEBSITE.md.
Uploads are sent to the server and processed in memory; hosting-provider policies also apply.
''',encoding='utf-8')
    (ROOT/'PROJECT_STATUS.md').write_text('# Current release status\n\nThe two-source development experiment is complete. '+
        'See README.md and V3_REPORT.md for current scope and metrics. The earlier external test has been evaluated; it is no longer sealed. '+
        'Older notebooks and presentations describe historical experiments. Current public deployment must be verified separately.\n',encoding='utf-8')

def activate():
    checks=json.loads((OUT/'acceptance.json').read_text())
    if not checks.get('passed'):raise RuntimeError('Acceptance checks have not passed')
    shutil.copy2(ROOT/'app_v3.py',ROOT/'app.py')
    print('Activated verified v3 app.py; app_svd_v2.py remains available.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['benchmark','docs','activate']);args=parser.parse_args()
    {'benchmark':benchmark,'docs':docs,'activate':activate}[args.stage]()
