# VOICED data card

Source: https://physionet.org/content/voiced/1.0.0/
Version: 1.0.0. DOI: https://doi.org/10.13026/C25Q2N
License: Open Data Commons Attribution License v1.0, https://opendatacommons.org/licenses/by/1-0/.
Retain this attribution and REFERENCES.txt when sharing results or redistributing the public dataset.

The provider describes 208 clinically verified voice recordings, sustained /a/, approximately five seconds, sampled at 8 kHz using a Samsung Galaxy S4 in controlled recording conditions. Data were already filtered by the provider. Participants were adults, with the provider describing exclusions. See the source for collection details rather than assuming this cohort represents a general population.

## Observed metadata discrepancy

The website summary says **58 healthy / 150 pathological**. The downloaded release contains **57 healthy / 151 pathological** according to both the `.hea` diagnosis fields and the `-info.txt` diagnosis fields. This implementation retains file-level labels and does not relabel a recording to force agreement with the website. The reason for the mismatch is unknown.

## Identity and labels

The documentation describes one acquired sample per subject. We use release record IDs as participant proxies, with exact and gain-normalized central-crop duplicate checks. There is no independent persistent patient identifier to reverify identity across sessions. Accordingly, speaker independence is conditional on the provider's description.

Healthy = the explicit healthy/normal label. Pathological = the supplied pathological diagnoses. Original diagnosis strings remain in the manifest. Unknown or missing metadata must not be silently invented. Multiple detailed diagnoses are not merged into new clinical disorder categories. Age, sex, diagnosis text, questionnaires and file names are not model inputs.

## Splits and exclusions

Audit all source files before partitioning. Reject unreadable, nonfinite, silent or shorter-than-three-second audio; log exclusions. Retain one copy of identical audio; exclude conflicting-label duplicate groups. Use seed 2026 for a stratified 20% holdout and three stratified grouped development folds. All sessions/segments from a participant must remain together if the dataset is extended.

## Known limits

Small cohort and small healthy test count; label consistency cannot be independently clinically assessed; recording and demographic confounding; one vocal task and device protocol; no external validation. Audio is not an anatomical examination. A binary prediction cannot identify a specific disorder or exclude disease.
