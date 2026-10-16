"""Local notebook controls using the same guarded inference path as the CLI."""
from contextlib import contextmanager
from pathlib import Path
import html
import json
import tempfile
import time
import numpy as np
import pandas as pd
import soundfile as sf
from .data import ROOT
from .experiment import inspect_recording
from .features import SR, QUALITY_NAMES

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@contextmanager
def uploaded_audio(name, content, root=ROOT):
    """Bound and validate input; never trust an upload name as a filesystem path."""
    suffix = Path(name).suffix.lower()
    if suffix not in {'.wav', '.flac'}:
        raise ValueError('Choose a WAV or FLAC recording.')
    if not 0 < len(content) <= MAX_UPLOAD_BYTES:
        raise ValueError('Choose a nonempty file of at most 10 MB.')
    directory = root/'.runtime/uploads'
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='voice-', dir=directory) as folder:
        path = Path(folder)/('recording'+suffix)
        path.write_bytes(content)
        try:
            meta = sf.info(path)
        except RuntimeError as exc:
            raise ValueError('This file is not readable WAV/FLAC audio.') from exc
        if meta.format not in {'WAV', 'WAVEX', 'RF64', 'FLAC'}:
            raise ValueError('The file contents must be WAV or FLAC audio.')
        if meta.duration > 60 or meta.channels > 2:
            raise ValueError('Use a mono/stereo recording no longer than 60 seconds.')
        yield path


def show_recording(path, root=ROOT, title=None, validation=False):
    from IPython.display import display, HTML, Audio
    import matplotlib.pyplot as plt
    started = time.perf_counter()
    result, audio, seq, quality = inspect_recording(path, root, validation=validation)
    elapsed = time.perf_counter() - started
    name = html.escape(title or Path(path).name)
    cards = []
    for family, values in result['models'].items():
        cards.append(f'<div style="flex:1;min-width:170px;border:1px solid #cbd5e1;border-radius:12px;padding:18px;background:#f8fafc;color:#172033">'
                     f'<b>{family}</b><div style="font-size:24px;margin:8px 0">{values["prediction"].title()}</div>'
                     f'<small>Raw {values["score_type"]}: {values["raw_score"]:.3f}<br>Decision threshold: 0</small></div>')
    disagreement = len({v['prediction'] for v in result['models'].values()}) > 1
    notice = ('Models disagree on this recording.' if disagreement else 'Models agree on this recording.')
    context = result['model_context']
    display(HTML(f'<h3>{name}</h3><p><b>{html.escape(context["mode"].title())}</b><br>{html.escape(context["notice"])}</p><div style="display:flex;flex-wrap:wrap;gap:12px">'
                 + ''.join(cards) + f'</div><p><b>{notice}</b> Agreement does not establish correctness.</p>'
                 '<p>Academic predictions only. Raw scores are not probabilities and cannot be compared across models.</p>'))
    display(Audio(data=audio, rate=SR, normalize=False))
    status = result['quality']
    print(f'Central 3-second crop · 8 kHz · analysis {elapsed:.2f} s')
    print('Technical checks: ' + ('measurements need attention' if status['missing_quality_features'] else 'completed'))
    print(f'Pitch detected in {status["voiced_frame_fraction"]:.0%} of analysis frames. This does not verify vowel identity.')
    for failure in status['measurement_failures']:
        print('Measurement note:', failure)
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), constrained_layout=True)
    axes[0].plot(np.arange(len(audio))/SR, audio, color='#0f766e', linewidth=.6)
    axes[0].set(xlabel='Time (seconds)', ylabel='Amplitude', title='Exact audio crop supplied to both pipelines')
    axes[1].imshow(seq, origin='lower', aspect='auto', extent=[0,3,-.5,12.5], cmap='viridis')
    axes[1].set(xlabel='Time (seconds)', ylabel='MFCC index', title='13 MFCCs · 297 analysis frames')
    display(fig)
    plt.close(fig)
    labels = ['CPPS (dB)', 'HNR (dB)', 'Local jitter (%)', 'Local shimmer (%)', 'Median pitch (Hz)', 'Pitch IQR (Hz)']
    shown = quality.copy()
    shown[2:4] *= 100
    display(pd.DataFrame({'Measurement': labels, 'Value': [f'{v:.3f}' if np.isfinite(v) else 'Missing; training-median imputation' for v in shown]}))
    return result


def make_demo(root=ROOT):
    import ipywidgets as widgets
    from IPython.display import display
    examples = sorted((root/'demo_audio').glob('*.wav')) + sorted((root/'examples').glob('*.wav'))
    manifest = pd.read_csv(root/'data/manifest.csv').set_index('record_id')
    options = [(f'{p.stem} · {"healthy" if manifest.loc[p.stem,"label"] == 0 else "pathological"} dataset label', str(p)) for p in examples]
    chooser = widgets.Dropdown(options=options, description='Example:', layout=widgets.Layout(width='95%'))
    mode = widgets.ToggleButtons(options=[('Speaker-held-out demo','validation'), ('Full-development models','refit')],
                                 value='validation', description='Models:')
    example_button = widgets.Button(description='Analyze example', button_style='primary', icon='play')
    upload = widgets.FileUpload(accept='.wav,.flac', multiple=False, description='Choose audio')
    upload_button = widgets.Button(description='Analyze chosen file', icon='wave-square')
    clear = widgets.Button(description='Clear results', icon='eraser')
    output = widgets.Output()

    def guarded(action):
        with output:
            output.clear_output(wait=True)
            example_button.disabled = upload_button.disabled = True
            try:
                action()
            except (ValueError, OSError) as exc:
                print('Recording not accepted:', exc)
            except Exception as exc:
                print(f'Analysis failed ({type(exc).__name__}): {exc}. No prediction is available.')
            finally:
                example_button.disabled = upload_button.disabled = False

    def example_click(_):
        if chooser.value:
            guarded(lambda: show_recording(Path(chooser.value), root, validation=mode.value=='validation'))

    def upload_click(_):
        def action():
            if not upload.value:
                raise ValueError('Choose an audio file first.')
            item = upload.value[0]
            name, content = item['name'], bytes(item['content'])
            upload.value = ()  # Do not retain audio bytes in saved widget state.
            with uploaded_audio(name, content, root) as path:
                show_recording(path, root, title=name, validation=mode.value=='validation')
        guarded(action)

    def clear_click(_):
        upload.value = ()
        output.clear_output()

    example_button.on_click(example_click)
    upload_button.on_click(upload_click)
    clear.on_click(clear_click)
    view = widgets.VBox([
        widgets.HTML('<b>Choose a development example or your own sustained /a/ recording.</b>'
                     '<p>WAV/FLAC · 3–60 seconds · mono/stereo · at least 8 kHz · maximum 10 MB.</p>'
                     '<p>Files are sent to your local Jupyter process. Temporary input files are deleted after analysis; '
                     'displayed audio and results may be saved in the notebook. Clear outputs before sharing personal recordings.</p>'),
        widgets.HTML('<p><b>Speaker-held-out demo</b> uses each bundled recording\'s validation-fold models, trained without that speaker. '
                     'For a new external recording, select <b>Full-development models</b>.</p>'),
        mode, chooser, example_button, widgets.HBox([upload, upload_button, clear], layout=widgets.Layout(flex_flow='row wrap')), output])
    return view
