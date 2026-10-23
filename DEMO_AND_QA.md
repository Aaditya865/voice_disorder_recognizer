# Current release: two-minute demonstration

Launch OPEN_WEBSITE.cmd before the review. Keep demo_v3/ and the presentation open. These examples were used in development; their predictions are not new accuracy measurements. The app intentionally returns Uncertain for every binary prediction because no threshold qualified on both sources.

| Time | Action | Explanation |
|---|---|---|
| 0:00–0:20 | Show title and disclaimer | We compare GMM and DNN for healthy versus pathological sustained vowels. This is an academic research demonstration. |
| 0:20–0:45 | Upload demo_v3/SVD_healthy_2.wav; Analyze Voice | The saved feature extraction, transforms and calibration match training. This is a development demonstration. |
| 0:45–1:05 | Point to Uncertain and model scores | The leading label is a research prediction. No threshold met our requirements in both datasets, so the software does not assert a reliable classification. |
| 1:05–1:25 | Play audio; show waveform and GMM/DNN comparison | Model loading is cached. Median warm inference across ten fixed examples was 0.36 seconds on this laptop. Models can disagree or both be wrong. |
| 1:25–1:45 | Replace with demo_v3/SVD_pathological_105.wav; Analyze Voice | Replacement clears the old output. Subtype rankings are exploratory; the specific cause remains unresolved. |
| 1:45–2:00 | Show results slide | Nested development balanced accuracy was 0.6895 on SVD and 0.6270 on VOICED. We preserve the earlier external failure and do not claim new external validation. |

## Ten likely questions

1. **Why GMM and DNN?** GMM models class-conditional feature distributions; neural networks learn nonlinear boundaries. Both satisfy the assignment. We compared MLP and CNN variants rather than assuming complexity wins.
2. **Which model won?** The 8 kHz MLP with hidden layers (64,32), selected by equal-source development balanced accuracy and the predeclared simplicity rule. GMM remains available for comparison.
3. **How much data?** 1,212 SVD speakers and 208 VOICED participant proxies: 717 healthy and 703 pathological records. VOICED identity relies on published record correspondence, not independent identity verification.
4. **How did you prevent leakage?** Participant-disjoint outer/inner folds and separate calibration/stopping participants; learned transforms and balancing use fitting data only. Audit checks covered 190 cached fits.
5. **What do the numbers measure?** Outer-fold predictions evaluate the selection procedure on development data. They do not test the final refitted model on a new population.
6. **Why not 99%?** Acoustically overlapping conditions, confounded populations and source differences limit generalization. We report measurements, not a promised target.
7. **Why does everything say Uncertain?** None of 0.60/0.70/0.80/0.90 achieved the required accepted accuracy, coverage and sample count in both sources during final selection. We retained the agreed rejection policy.
8. **Can it diagnose laryngitis or paralysis?** No. The unchanged subtype model ranks three supported groups conditionally and remains uncertain; many possible causes are outside its labels.
9. **What happened on external data?** Historical v2 detected only 4 of 30 pathological test voices, with balanced accuracy 0.525. Those records are now development data, so revised results are not a fresh external test.
10. **What improves reliability next?** Representative new speakers, label/identity verification, age/sex/source controls, and a genuinely untouched external evaluation. More tuning alone does not establish reliability.

## Fallback

Use the installed local app without internet. Keep the slides and V3_REPORT.md ready if hardware fails. app_svd_v2.py preserves the historical app, but its outputs and metrics must be labeled as historical. Never describe selected successful examples as unseen tests.
