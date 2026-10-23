## 1. Voice Disorder Recognizer

This academic project compares Gaussian mixture and deep neural models for recognizing healthy versus pathological sustained vowels. We built a shared training and website pipeline and measured its limitations. The current revision has development validation only, not a new independent clinical test.

Sources: README.md; V3_REPORT.md

## 2. Research question and paper

The main task is binary classification of a sustained aa vowel. The three disorder groups are exploratory and always uncertain. Gupta and colleagues reviewed 80 multiclass studies; the paper motivates speaker separation, complementary voice-quality features and honest external evaluation. It does not prove that GMM or DNN is the best model.

Sources: V3_REPORT.md; Gupta et al. 2026: https://doi.org/10.3389/fdgth.2026.1800132

## 3. Two datasets, 1,420 development participants

SVD contributes 1,212 unique speaker IDs after auditing. VOICED contributes 208 release record IDs used as participant proxies; identity cannot be independently reverified. The file labels give 57 healthy and 151 pathological VOICED records, rather than its website summary. All previously evaluated VOICED cases now count as development data. SVD healthy and pathological speakers differ greatly in age, which remains a potential confounder.

Sources: artifacts_v3/dataset_audit.csv; artifacts_v3/voiced_label_audit.csv; https://zenodo.org/records/16874898; https://physionet.org/content/voiced/1.0.0/

## 4. Audio-to-prediction pipeline

The website reuses training feature extraction and the saved transforms. It accepts WAV or MP3, checks duration and silence, and analyzes at most the central five seconds. We compare 16 and 8 kHz variants. The 97 features include MFCC and delta statistics, CPPS, HNR, pitch, jitter, shimmer and spectral measures. The CNN additionally sees log-Mel spectrograms. Models load once; invalid inputs produce an error rather than a fabricated output.

Sources: svd_v3.py; voice_review/release_web.py; app.py

## 5. Model selection and leakage controls

We compare six configurations at each of two sampling rates: three GMMs, two MLPs and one CNN. Four outer participant-disjoint folds assess the entire selection procedure, with three inner folds choosing settings. Imputation, scaling and balancing occur only within fitting partitions. Separate participants support calibration and neural-network stopping. The selection objective gives SVD and VOICED equal importance. Scores within 0.01 prefer the simpler model. This reduces selection bias but does not erase prior knowledge of these datasets.

Sources: svd_v3.py; artifacts_v3/protocol.json; artifacts_v3/nested_folds.json

## 6. Nested development results

These numbers pool the outer held-out predictions of the selection procedure. They are not an untouched evaluation of the final refitted model. The complete report includes family comparisons, fold means and standard deviations, per-class results and participant bootstrap intervals. Do not compare these VOICED scores with the old external test as though both were independent tests.

Sources: artifacts_v3/nested_metrics.csv; artifacts_v3/nested_summary.json; V3_REPORT.md

## 7. Errors and the preserved external failure

The new source-specific confusion counts come from nested development evaluation. Earlier, the SVD-only v2 model was evaluated once on 42 reserved VOICED cases. It detected only four of thirty pathological recordings, giving balanced accuracy 0.525. We preserved that result. The current experiment uses VOICED as development, so its new scores cannot establish repaired external generalization.

Sources: artifacts_v3/nested_summary.json; artifacts_svd_v2/external_binary_test_metrics.json

## 8. Uncertainty and limits

We tested confidence thresholds using inner validation, requiring minimum accuracy and coverage in each source. If none qualifies, every binary output is uncertain. Even a qualifying rule is not a clinical guarantee. We report outer-fold acceptance coverage and errors. All subtype predictions remain uncertain because the earlier subtype experiment did not support reliable identification. Unknown recording conditions, age and sex differences, short vowels and incomplete labels remain limitations.

Sources: artifacts_v3/release.json; artifacts_v3/nested_summary.json; voice_review/release_web.py

## 9. Demonstration and next steps

For the demonstration, start the local app, choose a supplied development example and click Analyze Voice. Explain the binary result, uncertainty, two model scores and exploratory ranking. The files were chosen by source/label order, not by prediction correctness. Show a malformed or very short recording only if time allows. The local application does not need internet after installation. Next work requires representative new speakers and a genuinely untouched external cohort, not repeated tuning on the same demo files.

Sources: demo_v3/manifest.csv; RUN_WEBSITE.md; artifacts_v3/runtime_benchmark.json; DEMO_AND_QA.md