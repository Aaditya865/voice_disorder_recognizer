# V4 improvement experiment — measured results

Academic project, not a medical diagnosis. This is an adaptive development experiment, not an independent external validation.

## Outcome

Final experimental configuration: **MLP_noise; cutoff 0.450000; rejection threshold 0.800000**.

Do not replace v3: the predefined improvement criteria were not satisfied.

## Before / after on the same participants

| procedure | source | n | accuracy | balanced_accuracy | macro_f1 | sensitivity | specificity | roc_auc | brier | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v3_previous | pooled | 1420 | 0.686620 | 0.685846 | 0.684328 | 0.607397 | 0.764296 | 0.741254 | 0.203804 | 0.046200 |
| v3_previous | SVD | 1212 | 0.697195 | 0.689509 | 0.690443 | 0.603261 | 0.775758 | 0.744483 | 0.200450 | 0.056096 |
| v3_previous | VOICED | 208 | 0.625000 | 0.627048 | 0.593383 | 0.622517 | 0.631579 | 0.661555 | 0.223349 | 0.175947 |
| v4_same_participants | pooled | 1420 | 0.679577 | 0.678401 | 0.674404 | 0.559033 | 0.797768 | 0.762691 | 0.196516 | 0.023964 |
| v4_same_participants | SVD | 1212 | 0.696370 | 0.684305 | 0.684192 | 0.548913 | 0.819697 | 0.765130 | 0.193501 | 0.035505 |
| v4_same_participants | VOICED | 208 | 0.581731 | 0.569943 | 0.545132 | 0.596026 | 0.543860 | 0.650517 | 0.214083 | 0.129242 |

Values are printed to six decimals; full precision is retained in CSV. Outer folds assess a selection procedure, not a new independent test of the final refitted model.

## Paired improvement criterion

```json
{
  "development_promotion_eligible": false,
  "equal_source_ba_gain": -0.03115444915458554,
  "paired_bootstrap_95_delta": [
    -0.06589836027447553,
    0.003563055105321428
  ],
  "source_changes": {
    "SVD": -0.005204216073781298,
    "VOICED": -0.05710468223538967
  },
  "scope": "Adaptive development experiment; CI conditions on fixed predictions, not all tuning uncertainty",
  "selected": {
    "id": "MLP_noise",
    "family": "MLP",
    "order": 6,
    "score": 0.6827908731793361,
    "cutoff": 0.44999999999999996
  },
  "threshold": 0.8
}
```

The paired bootstrap conditions on fixed outer-fold predictions. It does not capture all model-selection uncertainty, source effects or prior iterative use of development data.

## Expanded-cohort model comparison

| procedure | source | n | balanced_accuracy_mean | balanced_accuracy_std | macro_f1_mean | sensitivity_mean | specificity_mean | roc_auc_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v4_all_development | pooled | 1618 | 0.676461 | 0.033860 | 0.657530 | 0.555094 | 0.797827 | 0.758909 |
| v4_all_development | SVD | 1410 | 0.683219 | 0.029499 | 0.668322 | 0.546742 | 0.819697 | 0.762792 |
| v4_all_development | VOICED | 208 | 0.571578 | 0.048132 | 0.526515 | 0.596728 | 0.546429 | 0.656246 |
| v4_GMM | pooled | 1618 | 0.664272 | 0.037881 | 0.652158 | 0.578280 | 0.750264 | 0.739268 |
| v4_GMM | SVD | 1410 | 0.683879 | 0.031729 | 0.676113 | 0.602607 | 0.765152 | 0.762355 |
| v4_GMM | VOICED | 208 | 0.518811 | 0.065989 | 0.465835 | 0.457859 | 0.579762 | 0.566828 |
| v4_DNN | pooled | 1618 | 0.676461 | 0.033860 | 0.657530 | 0.555094 | 0.797827 | 0.758909 |
| v4_DNN | SVD | 1410 | 0.683219 | 0.029499 | 0.668322 | 0.546742 | 0.819697 | 0.762792 |
| v4_DNN | VOICED | 208 | 0.571578 | 0.048132 | 0.526515 | 0.596728 | 0.546429 | 0.656246 |

## Every candidate tried

candidate_inner_summary.csv contains mean/std of the inner-validation tuning results across outer training partitions. These overlap and are NOT independent outer-test performance for each fixed candidate. CNN, MLP, GMM, SVM, embedding/fusion and augmentation variants all completed the same nested search.

| id | source | balanced_accuracy_mean | balanced_accuracy_std | macro_f1_mean | macro_f1_std | sensitivity_mean | sensitivity_std | specificity_mean | specificity_std | roc_auc_mean | roc_auc_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN_aug | pooled | 0.681756 | 0.014636 | 0.657798 | 0.030272 | 0.514164 | 0.094197 | 0.849347 | 0.067650 | 0.745215 | 0.005916 |
| Ensemble | pooled | 0.700629 | 0.005474 | 0.689976 | 0.009500 | 0.612326 | 0.051427 | 0.788931 | 0.047547 | 0.755206 | 0.010757 |
| GMM_mfcc2 | pooled | 0.639217 | 0.030620 | 0.619519 | 0.053714 | 0.517878 | 0.144131 | 0.760555 | 0.089391 | 0.685682 | 0.016180 |
| GMM_rich4 | pooled | 0.665282 | 0.018982 | 0.651799 | 0.021680 | 0.554166 | 0.048773 | 0.776398 | 0.034209 | 0.708170 | 0.023917 |
| MLP_base | pooled | 0.654513 | 0.006729 | 0.638217 | 0.004846 | 0.525700 | 0.042133 | 0.783325 | 0.052198 | 0.700731 | 0.010363 |
| MLP_fusion | pooled | 0.663610 | 0.026215 | 0.639855 | 0.048266 | 0.515454 | 0.143870 | 0.811767 | 0.100747 | 0.734632 | 0.013053 |
| MLP_mfcc | pooled | 0.667377 | 0.005187 | 0.650253 | 0.010006 | 0.532790 | 0.047735 | 0.801965 | 0.045098 | 0.728499 | 0.018260 |
| MLP_no_pitch | pooled | 0.693094 | 0.011898 | 0.679770 | 0.016468 | 0.581912 | 0.045525 | 0.804277 | 0.025456 | 0.756527 | 0.014008 |
| MLP_noise | pooled | 0.688016 | 0.020540 | 0.668363 | 0.037793 | 0.552721 | 0.122897 | 0.823310 | 0.083493 | 0.754399 | 0.011119 |
| MLP_rich | pooled | 0.670682 | 0.012219 | 0.646820 | 0.029301 | 0.511576 | 0.114939 | 0.829787 | 0.096379 | 0.723746 | 0.007603 |
| MLP_svd_only | pooled | 0.656793 | 0.026978 | 0.629223 | 0.046578 | 0.479051 | 0.120166 | 0.834535 | 0.071076 | 0.716924 | 0.011573 |
| MLP_wavlm | pooled | 0.656964 | 0.021378 | 0.631577 | 0.039538 | 0.505433 | 0.164179 | 0.808496 | 0.122238 | 0.726780 | 0.012621 |
| SVM_mfcc_C1 | pooled | 0.683377 | 0.010586 | 0.677375 | 0.013045 | 0.677296 | 0.116845 | 0.689459 | 0.128670 | 0.736877 | 0.008876 |
| SVM_rich_C10 | pooled | 0.660537 | 0.013534 | 0.643582 | 0.023715 | 0.561849 | 0.152996 | 0.759224 | 0.138457 | 0.726092 | 0.016969 |
| SVM_wavlm | pooled | 0.657663 | 0.009166 | 0.645702 | 0.021183 | 0.584987 | 0.125954 | 0.730338 | 0.109126 | 0.716083 | 0.014449 |

## New-speaker challenge

```json
{
  "n_pathological": 63,
  "n_healthy": 0,
  "old_sensitivity": 0.6825396825396826,
  "new_sensitivity": 0.6984126984126984,
  "new_sensitivity_wilson95": [
    0.5763962768644076,
    0.7976230883781127
  ],
  "accepted_count": 24,
  "balanced_accuracy": null,
  "specificity": null,
  "roc_auc": null,
  "limitations": "Same-source positive-only challenge, with download/quality exclusions; not a representative external binary test."
}
```

This is a same-source pathological-only challenge. It cannot provide specificity, balanced accuracy or AUC. The historical v2 external result remains 0.525 balanced accuracy and 4/30 pathological detections. VOICED is development in v3 and v4, so neither provides a new external-dataset result.

## Calibration and abstention

| source | lower | upper | n | mean_score | accuracy | note |
| --- | --- | --- | --- | --- | --- | --- |
| SVD | 0.000000 | 0.600000 | 344 | 0.527740 | 0.476744 |  |
| SVD | 0.600000 | 0.700000 | 383 | 0.649593 | 0.634465 |  |
| SVD | 0.700000 | 0.800000 | 448 | 0.745395 | 0.727679 |  |
| SVD | 0.800000 | 0.900000 | 162 | 0.840032 | 0.901235 |  |
| SVD | 0.900000 | 1.000000 | 73 | 0.948382 | 0.986301 |  |
| VOICED | 0.000000 | 0.600000 | 66 | 0.522413 | 0.348485 |  |
| VOICED | 0.600000 | 0.700000 | 49 | 0.655060 | 0.530612 |  |
| VOICED | 0.700000 | 0.800000 | 44 | 0.741457 | 0.636364 |  |
| VOICED | 0.800000 | 0.900000 | 33 | 0.850603 | 0.848485 |  |
| VOICED | 0.900000 | 1.000000 | 16 | 0.955008 | 1.000000 | inconclusive small bin |

Temperature calibration was fitted on held-out fitting-partition speakers. High scores are not guaranteed correctness. The original per-source accuracy, coverage and count requirements were preserved. Inspect outer-fold accepted accuracy as well as coverage; inner eligibility may fail on outer data.

## Changes and audit

- Added 198 previously unused SVD pathological speakers to development; retained 1,212 original SVD and 208 VOICED participant proxies.

- Reserved 100 new SVD speakers before modeling; 63 usable downloads survived availability/quality checks. Rate limits and short files were not replaced based on model performance.

- Found zero incorrect SVD binary mappings or speaker mappings against source metadata. Did not relabel false negatives to improve scores.

- Reused verified 8 kHz acoustic extraction and a pinned 16 kHz WavLM representation. Training and inference share these implementations.

- Tested MFCC-only, pitch-excluded, rich acoustic, frozen embedding and fusion inputs; GMM, MLP, CNN and SVM; training-only MLP feature noise and CNN spectrogram augmentation; an equal-weight GMM/DNN ensemble.

- Nested selection tuned binary cutoff; separate-speaker temperature calibration and CNN early stopping were retained.

- HUPA requires an agreement. PVQD N/Y denote complaints, not diagnostic truth, and recordings mix tasks. Neither was force-mapped to binary health labels.

## Remaining limits

No new independent external dataset with verified binary labels was evaluated. The extra development data are pathological-only and broaden the task, which may hurt some populations. Source and age/sex confounding remain; removing explicit pitch features cannot eliminate those effects. The sealed challenge is small, positive-only and subject to download/quality selection. Some clinical disorders may have little audible effect in a short vowel. Subtypes remain exploratory. Neither this bounded search nor calibration establishes clinical reliability.

## Reproduction and evidence

Release checks: 210 cached inner fitting/calibration partitions were audited with zero participant overlap. Acoustic and WavLM training/inference feature parity passed on one fixed development recording. Fourteen release tests passed, including historical integrity, saved/reloaded predictions, metric recomputation and retained-app behavior. These checks establish software consistency, not diagnostic validity.

See V4_RUN.md, protocol_v4.json, download_manifest.csv, outer_assignment.csv, nested_folds.json, nested_predictions.csv, candidate_comparison.csv and challenge_predictions.csv. The deployed v3 model is preserved separately.

## Sources

- https://zenodo.org/records/16874898

- https://huggingface.co/microsoft/wavlm-base-plus/tree/4c66d4806a428f2e922ccfa1a962776e232d487b

- https://data.mendeley.com/datasets/9dz247gnyb/4

- https://pmc.ncbi.nlm.nih.gov/articles/PMC13049965/
