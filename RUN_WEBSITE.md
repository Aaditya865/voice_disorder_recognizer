# Run the current website

Prepared laptop (no installation or training needed):

```powershell
Set-Location 'C:\Users\adity\Documents\Codex\2026-10-15\i-x20\outputs\voice_disorder_recognizer'
& '..\..\work\venv\Scripts\python.exe' -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. Alternatively double-click OPEN_WEBSITE.cmd.
If the port is already occupied, use the existing server or stop it with Ctrl+C; do not start competing copies.

Upload one of the four WAV files in demo_v3/ and press Analyze Voice. These are development demonstrations, not unseen tests.
Use a sustained aa vowel. WAV/MP3, mono/stereo, 8–192 kHz, minimum 0.5 seconds, maximum upload 20 MB.
Only the central five seconds are analyzed. The selected model supplies its own 8/16 kHz feature specification.
The subtype model keeps its original 16 kHz pipeline. Scores do not establish medical probabilities.

The app caches models once, processes uploads in memory and does not save uploaded audio to disk.
Browser recording needs microphone permission; rely on supplied WAVs for the review.

To use the preserved local fallback: replace `app.py` in the launch command with `app_svd_v2.py`.
The fallback is an earlier research model with documented weak external performance; it must not be presented as the current experiment.

Installation on another Windows laptop is in README.md. Cloud availability is separate from local operation.
