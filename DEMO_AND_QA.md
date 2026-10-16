# Two-minute review demo and reviewer questions

## Before the review
Launch the app once, keep samples/ open, and use the saved WAVs. Keep the architecture slide and Review.html ready offline. Predictions are actual model outputs and may be wrong; do not describe a high model score as validated confidence.

## Two-minute script
| Time | Click / action | Say |
|---|---|---|
| 0:00–0:20 | Show title and upload box | “This academic prototype compares GMM and a small convolutional DNN on sustained vowel recordings. It is not a medical diagnosis.” |
| 0:20–0:45 | Upload healthy_voice002.wav; Analyze Voice | “We use the same central three seconds, MFCCs, voice-quality features and saved transforms as training. Both models predict healthy for this example.” |
| 0:45–1:05 | Play the analyzed segment; point to comparison and score | “The models are cached, so inference is fast. These are uncalibrated model scores. The sample is recognized and routed to a validation model that excluded its speaker from fitting.” |
| 1:05–1:30 | Replace with pathological_voice003.wav; Analyze Voice | “Both binary models predict pathological. The dataset diagnosis is a nodule; the laryngitis suggestion here is wrong. This shows why the subtype output is experimental.” |
| 1:30–1:50 | Replace with laryngeal_voice008.wav; Analyze Voice | “The dataset labels this reflux laryngitis. DNN predicts pathological with the laryngitis group; GMM misses it. We show disagreements and failures rather than hiding them.” |
| 1:50–2:00 | Show architecture/results slide | “Binary development balanced accuracy is 61.6% for GMM and 57.5% for DNN. Subtype results are weaker. The 42-record final test remains reserved.” |

## Likely reviewer questions
1. **Why GMM and DNN?** GMM is a compact generative baseline for a small dataset. The convolutional DNN learns temporal MFCC patterns. Both satisfy the assignment's permitted model families.
2. **What dataset did you actually train on?** VOICED 1.0.0: 208 recordings. The supplied diagnosis metadata gives 57 healthy and 151 pathological, differing by one from the website summary. Development uses 166; 42 are reserved for final evaluation.
3. **What does speaker independence mean here?** All recordings with a speaker ID belong to one partition; duplicates are checked. Record IDs are participant proxies in this release, not independently verified identities. Demo samples use fold models that excluded those IDs.
4. **What features are used?** Thirteen MFCCs and CPPS, HNR, local jitter, local shimmer, median pitch and pitch IQR. GMM uses MFCC means/standard deviations; DNN uses MFCC sequences. Each uses saved training-fitted transforms and missing-value handling.
5. **Is the displayed percentage confidence?** No. It is an uncalibrated score derived from the binary decision output. GMM and DNN scores are not directly comparable medical probabilities and can be high when wrong.
6. **Does it reliably detect laryngeal disease?** No. The exploratory group contains 39 laryngitis-labelled records, only 32 in development. Other pathology can also involve the larynx. Three-class laryngitis precision is 32.4% for GMM and 27.3% for DNN.
7. **How did you avoid leakage?** Fixed speaker-proxy partitions, train-only transforms and class weights, internal stopping subsets, separate fold models and an untouched final test. Development folds selected the binary settings, so the reported estimates have selection bias.
8. **Why balanced accuracy?** Healthy/pathological counts are imbalanced. Balanced accuracy averages class recall. We also retain per-class precision, recall, F1 and confusion counts, including the website's binary-plus-subtype decision rule.
9. **How does the reference paper support this?** Gupta et al. (2026), DOI 10.3389/fdgth.2026.1800132, is a scoping review of multiclass studies. It motivates transparent labels, speaker separation and evaluation. It does not establish that our model pair is best, and its study accuracies are not our results.
10. **What are the limitations and next steps?** Small single-source data, identity proxies, weak subtype precision, unstable DNN results, uncalibrated scores and no external or phone-audio validation. Next: complete the frozen final evaluation, larger/external data, reliable identity metadata and calibration on independent development data.

## One-slide summary
Architecture_and_Results.pptx is editable. Architecture_and_Results.png is the offline image version. The table shows pooled development validation, not test accuracy. LARYNGEAL_REPORT.md contains every source diagnosis count, per-class metrics and updated confusion matrices.
