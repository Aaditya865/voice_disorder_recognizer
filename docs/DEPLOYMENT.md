# Optional deployment

The verified demo runs locally. No hosted deployment is claimed.

Streamlit Community Cloud instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app

1. Connect this GitHub repository to your Streamlit Community Cloud account.
2. Choose app.py as the entrypoint and Python 3.12.
3. Keep models/, voice_review/, data/manifest.csv, data/splits.json, results/selection.json, results/laryngeal/confusion_matrices.png, samples/, .streamlit/config.toml and requirements.txt available.
4. For a Linux CPU-only dependency build, use a deployment branch with `--extra-index-url https://download.pytorch.org/whl/cpu` and `torch==2.14.1+cpu` in requirements.txt. Confirm dependency availability and resource limits on the selected host.
5. Test sample uploads and errors after deployment before sharing the URL.

Cloud hosting has not been verified. On a hosted instance, uploaded audio is transmitted to that host. The app processes uploads in memory and does not write them to files; hosting policies and logging remain the host's responsibility.

For the review, keep a local fallback: install the environment, run `python -m streamlit run app.py`, and open http://localhost:8501. Review.html and Architecture_and_Results.png are offline presentations of saved results if live inference cannot run.
