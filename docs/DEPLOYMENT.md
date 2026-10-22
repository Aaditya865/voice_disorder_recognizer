# Deploy the SVD website

On Streamlit Community Cloud, select repository
`Aaditya865/voice_disorder_recognizer`, branch `main`, entrypoint `app.py`,
and Python 3.12. Install dependencies from the root `requirements.txt`.
This active app uses scikit-learn and does not require PyTorch.

Keep `svd_training.py`, `voice_review/svd_web.py`, `artifacts_svd_local/`,
`.streamlit/config.toml`, and `app.py` available. Models and preprocessing
are loaded once. No training or dataset download runs on website startup.

If already connected to main, a pushed commit should trigger rebuilding.
Inspect the hosting logs if it fails; reboot the app to clear a stale model
cache after an update. A GitHub push alone is not proof of successful hosting.

Local fallback: `python -m streamlit run app.py`, then open
http://localhost:8501. Uploaded audio is transmitted to the server, processed
in memory and not written to disk by the application. Host policies also apply.

The old VOICED app is available as `app_voiced_legacy.py` and requires
`requirements-legacy.txt`. Do not mix its metrics with the SVD results.
