# V4 audit and predeclared experiment

The working v3 website and its historical artifacts remain frozen. This directory is a separate experiment.

## Data findings

- Existing development: 1,212 SVD speakers and 208 VOICED participant proxies.
- Rechecked SVD binary labels and participant mappings against official source metadata: zero discrepancies.
- Existing 8 kHz features: 1,420 x 97, 14 missing measurements, no infinities, no constant columns. Missingness uses fitting-only imputation.
- Errors must not be used to relabel participants. Several diagnoses are mixed and a normal-sounding vowel does not negate the recorded diagnosis.
- Found 445 previously unused SVD pathological speakers with normal /a/ archive entries; no new healthy speakers with the required available task.
- Preselected 100 new speakers for a sealed sensitivity-only challenge before modeling. Download/quality checks yielded 63 usable challenge and 198 usable development recordings. HTTP 429/500 and three short recordings are retained in download_audit.csv; no performance-based replacement occurred.
- New development participants receive fixed outer-fold assignments. Original participants retain their v3 outer folds for paired comparison.
- HUPA requires an access request/data-use agreement; no request was submitted on the user's behalf.
- PVQD v4 metadata and official overview were inspected. N/Y mean absence/presence of a reported complaint, not a verified diagnosis. Missing diagnoses are not healthy labels. Files mix vowels and sentences; no unverified automatic segmentation or label mapping is used. Excluded from this deadline's binary experiment.

## Experiment

Four outer participant-disjoint folds, three inner folds. All datasets already seen remain development. Models select using equal-source balanced accuracy. A binary cutoff is chosen from 0.25 to 0.75 in 0.05 steps using inner predictions. Temperature calibration uses a separate subset of the fitting partition. CNN early stopping has a further disjoint subset. This is a bounded search, not a claim of globally best accuracy.

Compare GMM, MLP/DNN, compact CNN, SVM benchmarks, frozen WavLM layer-6 mean/std embeddings and acoustic/embedding fusion. MLP feature noise and CNN spectrogram augmentation occur only during fitting. SVD-only fitting is a candidate, but its calibration and selection still use both development sources. Therefore it is not a leave-dataset-out test.

Rejection requirements remain unchanged: thresholds 0.60, 0.70, 0.80, 0.90; each source needs >=80% accepted accuracy, >=20% coverage and >=20 accepted examples in inner validation. Otherwise every binary result remains uncertain.

Promotion requires equal-source balanced accuracy gain >=0.02, a positive lower paired-bootstrap confidence bound, and no source falling by more than 0.01, assessed on the same original participants. This is an operational development criterion, not proof of external generalization. The final pathological-only challenge is evaluated once after freezing selection; it cannot establish specificity, AUC, balanced accuracy or clinical utility.

## Sources

- https://zenodo.org/records/16874898
- https://stimmdb.coli.uni-saarland.de/data/voice_data.csv
- https://physionet.org/content/voiced/1.0.0/
- https://huggingface.co/microsoft/wavlm-base-plus
- https://data.mendeley.com/datasets/9dz247gnyb/4
- https://pmc.ncbi.nlm.nih.gov/articles/PMC13049965/
