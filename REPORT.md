# Voice Disorder Recognizer — academic report

## Status
Working research prototype. Development experiment completed; final held-out evaluation is pending unless results/final_metrics.json exists. No clinical efficacy claim is made.

## Research question
How do a class-conditional GMM and a compact convolutional DNN compare for healthy-versus-pathological classification of sustained /a/ recordings under participant-disjoint validation?

## Reference-paper relationship
Gupta et al. (2026) review 80 multiclass studies; binary-only studies were excluded. MFCCs appeared in 43 studies and voice-quality features in 19. The authors recommend clear labels, speaker separation, balanced accuracy, interpretability, and cross-database testing. Their reported 39–99% range is not an achievable target or a model result reproduced here. The review does not establish a best model. It contains conflicting summaries of test sizes and framework use; these are not resolved by assumption.

## Dataset and identity
VOICED 1.0.0: 208 records audited, 208 retained. Original diagnoses and exclusions are in data/manifest.csv.

PhysioNet describes one acquired vowel recording per subject. Release record IDs are used as participant proxies; no independent cross-session person identifier is provided. Exact/gain-normalized duplicate crops are checked, but identity cannot be independently reverified.

### Dataset count discrepancy
The release files contain 57 healthy and 151 pathological labels, in agreement between .hea and -info.txt. The website reports 58 and 150. File-level diagnoses are retained; no label is changed to force agreement with the summary.

## Methods
Central three-second crop, DC removal, mono 8 kHz. MFCCs: 13 coefficients including C0, 25 ms Hann window, 10 ms hop, 256-point FFT, 26 mel bands, 50–3800 Hz, no centered padding. Voice quality: CPPS, HNR, local jitter and shimmer, median F0 and IQR, with a 60–500 Hz pitch search. Undefined measurements are imputed using training medians, with six explicit missingness indicators. No amplitude normalization, denoising, or augmentation is added. Dataset-provider filtering is inherited.

GMM: separate diagonal mixtures, equal priors, covariance regularization 0.01, five initializations, 1/2/4 components. CNN: Conv1D 13→16 (kernel 5)→32 (kernel 3), ReLU, global average pooling, optional 12-value quality/missingness branch, 16-unit dense layer, dropout 0.2/0.4, binary logit. AdamW 0.001, weight decay 0.0001, batch 16, up to 100 epochs, patience 12.

The development portion uses three fixed stratified grouped folds. All transforms and weights are fitted inside training partitions. CNN early stopping uses a further 20% training-fold subset; the fold model is then freshly refitted on all fold-training participants for the selected epoch count. Final development refits use the median selected epoch count. Both models classify at raw score zero; no threshold is optimized on test data.

## Model selection
Settings and MFCC-only versus combined features are selected by mean fold balanced accuracy. Exact ties prefer fewer input features, fewer mixture components, then stronger dropout. This selection uses the displayed development folds, introducing selection optimism. The reserved final test is independent of tuning. This is internal model selection plus a held-out test, not nested cross-validation.

## Preliminary development results
| Model | Balanced accuracy | Macro-F1 | Sensitivity | Specificity | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| GMM | 61.6% | 0.511 | 38.8% | 84.4% | 0.620 | 51.2% |
| DNN | 57.5% | 0.561 | 66.1% | 48.9% | 0.566 | 61.4% |

Selected configurations:
- GMM: GMM_combined_1; mean fold balanced accuracy 61.7%.
- DNN: DNN_combined_0.4; mean fold balanced accuracy 57.6%.

Intervals and paired differences are in results/development_intervals.json. They resample participants and condition on the fitted models; they do not account for hyperparameter-selection or training uncertainty. Inspect the confusion-matrix counts, not accuracy alone.

## Explainability and errors
Feature-family ablations appear in results/search.json. Grouped permutation importance uses ten independent speaker permutations per validation fold, preserving within-sequence structure. Development error lists and demographic subgroup results are saved separately. Negative permutation drops and weak contributions must be retained, not hidden.

## Limitations
Small class sizes; one public cohort; no independently verifiable cross-session participant IDs; possible label noise and selection bias; fixed pitch settings; signal checks cannot recognize the vowel; weak generalization evidence; uncalibrated model outputs. GMM and CNN use different pooling, so observed differences cannot be attributed solely to model family. External validation, connected speech, and disorder-type classification are deferred.

## Final-test procedure
After the review, freeze the experiment and invoke the explicit one-time evaluation command in README.md. The command checks artifact hashes and refuses repeated evaluation. Do not retrain or select settings after inspecting the final test. A failed evaluation leaves an audit lock for investigation.

## Training variability follow-up

DNN predictions change for some participants across training seeds, so a single-run result is fragile. The selected one-component GMM is effectively deterministic on fixed data; its zero seed spread is expected and does not imply robust generalization. The same development folds selected the configurations, so selection bias remains. Seed standard deviations are descriptive, not confidence intervals. No best seed is selected and no demonstration model is replaced.

| Model | Mean balanced accuracy | Seed SD | Seed range | Records whose label changes |
|---|---:|---:|---:|---:|
| GMM | 61.6% | 0.0 pp | 61.6%–61.6% | 0/166 |
| DNN | 56.7% | 2.6 pp | 53.2%–59.4% | 73/166 |

### Protocol

Four runs: original seed 2026, plus 12026, 22026, 32026. Fold seed = base seed + fold index. Architecture, feature set, decision threshold and outer partitions stay fixed. Initialization, minibatch order and training-only stopping subset vary. This mixes those sources of training variability; it does not isolate initialization alone. The protocol and source hashes were saved before fitting the extra runs. All extra fits use development speakers only.

### Every run

| Model | Seed | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|
| DNN | 2026 | 57.5% | 66.1% | 48.9% |
| DNN | 12026 | 59.4% | 74.4% | 44.4% |
| DNN | 22026 | 53.2% | 75.2% | 31.1% |
| DNN | 32026 | 56.9% | 47.1% | 66.7% |
| GMM | 2026 | 61.6% | 38.8% | 84.4% |
| GMM | 12026 | 61.6% | 38.8% | 84.4% |
| GMM | 22026 | 61.6% | 38.8% | 84.4% |
| GMM | 32026 | 61.6% | 38.8% | 84.4% |

### Original model error overlap

Both correct: 62 · Both wrong: 41 · GMM correct only: 23 · DNN correct only: 40.

Agreement does not imply correctness. These counts describe original out-of-fold predictions, not the full-development refits used in the live demo. Per-record scores for all seeds are preserved in results/robustness/predictions.csv. The final test remains independent of this analysis.


## References
Gupta R, Jin CT, Gunjawate DR, Nguyen DD, Stasak B, Chacon AM, Madill C. (2026).
Voice disorders classification using machine learning: a scoping review. Frontiers in Digital Health 8:1800132.
https://doi.org/10.3389/fdgth.2026.1800132

Cesari U, De Pietro G, Marciano E, Niri C, Sannino G, Verde L. (2018).
A new database of healthy and pathological voices. Computers & Electrical Engineering 68:310–321.
https://doi.org/10.1016/j.compeleceng.2018.04.008

VOICED Database, version 1.0.0. https://doi.org/10.13026/C25Q2N
https://physionet.org/content/voiced/1.0.0/ — Open Data Commons Attribution License v1.0.
License: https://opendatacommons.org/licenses/by/1-0/

Pollard TJ et al. (2026). PhysioNet as a global platform for biomedical research. Nature Health.
https://doi.org/10.1038/s44360-026-00096-z
