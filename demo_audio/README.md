# Unseen-to-training audio demonstration

Play or select **voice003.wav** in notebook section 3, keeping **Speaker-held-out demo** selected. Click **Analyze example**, or choose this file through the file picker and click **Analyze chosen file**.

This is a real VOICED sustained-vowel recording with dataset label **pathological**, selected by record ID before examining correctness. It is a central three-second excerpt, mono 8 kHz, stored as float WAV to preserve the input fingerprint.

The demonstration routes the audio to the original validation-fold GMM and DNN. This speaker was excluded from those models' training and early stopping. Their results reproduce the saved out-of-fold predictions. Model settings were selected using development validation, so this is **unseen during fitting, not a completely independent final test**.

Selecting Full-development models uses refits that have seen this recording. Do not describe predictions from that mode as unseen. The 42 final-test participants remain sealed.

Source: VOICED 1.0.0, https://physionet.org/content/voiced/1.0.0/ . Open Data Commons Attribution License v1.0. See REFERENCES.txt for dataset citation.
