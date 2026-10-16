# Development robustness analysis

DNN predictions change for some participants across training seeds, so a single-run result is fragile. The selected one-component GMM is effectively deterministic on fixed data; its zero seed spread is expected and does not imply robust generalization. The same development folds selected the configurations, so selection bias remains. Seed standard deviations are descriptive, not confidence intervals. No best seed is selected and no demonstration model is replaced.

| Model | Mean balanced accuracy | Seed SD | Seed range | Records whose label changes |
|---|---:|---:|---:|---:|
| GMM | 61.6% | 0.0 pp | 61.6%–61.6% | 0/166 |
| DNN | 56.7% | 2.6 pp | 53.2%–59.4% | 73/166 |

## Protocol

Four runs: original seed 2026, plus 12026, 22026, 32026. Fold seed = base seed + fold index. Architecture, feature set, decision threshold and outer partitions stay fixed. Initialization, minibatch order and training-only stopping subset vary. This mixes those sources of training variability; it does not isolate initialization alone. The protocol and source hashes were saved before fitting the extra runs. All extra fits use development speakers only.

## Every run

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

## Original model error overlap

Both correct: 62 · Both wrong: 41 · GMM correct only: 23 · DNN correct only: 40.

Agreement does not imply correctness. These counts describe original out-of-fold predictions, not the full-development refits used in the live demo. Per-record scores for all seeds are preserved in results/robustness/predictions.csv. The final test remains independent of this analysis.
