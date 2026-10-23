# Deploy the current release

Streamlit Community Cloud: repository Aaditya865/voice_disorder_recognizer, branch main, entrypoint app.py, Python 3.12.
Install root requirements.txt, including CPU PyTorch. Keep artifacts_v3/, artifacts_svd_v2/, artifacts_svd_local/partition_manifest.csv,
svd_v3.py, svd_training.py, svd_advanced.py and voice_review/ in the deployment.
No training or download runs on startup. A successful GitHub push does not verify cloud deployment.
Check hosting logs and test a real upload after rebuilding. The local fallback is documented in RUN_WEBSITE.md.
Uploads are sent to the server and processed in memory; hosting-provider policies also apply.
