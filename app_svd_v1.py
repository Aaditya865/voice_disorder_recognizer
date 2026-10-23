"""Streamlit website for the evaluated five-class SVD models."""
from pathlib import Path
import hashlib
import logging
import numpy as np
import pandas as pd
import streamlit as st
from voice_review.svd_web import SVDService

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Voice Disorder Recognizer', page_icon=':material/graphic_eq:', layout='centered')
st.title('Voice Disorder Recognizer')
st.write('Upload a sustained aa vowel to compare five-class GMM and DNN predictions.')
st.caption('Academic project, not a medical diagnosis.')

@st.cache_resource(max_entries=1, show_spinner='Loading SVD models...')
def load_models():
    return SVDService(ROOT)

uploaded = st.file_uploader('Upload your voice recording', type=['wav','mp3'], max_upload_size=20,
    help='A steady 1–3 second aa vowel is recommended. Minimum 0.5 seconds; at most the central 5 seconds are analyzed.', key='voice_file')
with st.expander('Or record audio'):
    recorded = st.audio_input('Record a steady aa vowel', sample_rate=16000, key='voice_mic')
source = uploaded if uploaded is not None else recorded
both = uploaded is not None and recorded is not None
content = source.getvalue() if source is not None else None
fingerprint = hashlib.sha256(content).hexdigest() if content else None
if st.session_state.get('svd_input_hash') != fingerprint or both:
    st.session_state.pop('svd_analysis', None)
    st.session_state['svd_input_hash'] = fingerprint
if both:
    st.info('Clear the upload or microphone recording to analyze one recording.')
try:
    service = load_models()
except Exception:
    logging.exception('SVD model loading failed')
    st.error('The SVD models could not be loaded. Check the supplied model files and requirements.')
    st.stop()

if st.button('Analyze Voice', type='primary', width='stretch', disabled=source is None or both):
    st.session_state.pop('svd_analysis', None)
    try:
        with st.spinner('Analyzing your voice...'):
            st.session_state['svd_analysis'] = service.analyze(content, source.name)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception('SVD inference failed')
        st.error('Unable to analyze this audio. Try a clear WAV recording of a steady vowel.')

result = st.session_state.get('svd_analysis')
if result is not None:
    main = result['models'][result['selected']]
    st.subheader(f'Result: {main["result"]}')
    st.write(f'**Predicted diagnosis label:** {main["diagnosis"]}')
    st.write(main['description'])
    st.caption(f'Primary model: {result["selected"]} (MLP) — selected on development validation before final testing.')
    st.progress(main['score'], text=f'Model confidence score: {main["score"]:.1%} — uncalibrated')
    st.caption('Percentages are model scores, not medical probabilities. Even a high score can be incorrect.')
    if main['score'] < .6:
        st.warning('Uncertain, result not reliable. The 60% warning threshold is a heuristic.')
    if result['models']['GMM']['diagnosis'] != result['models']['DNN']['diagnosis']:
        st.warning('GMM and DNN disagree on the diagnosis label. Treat this result as uncertain.')
    st.write('**Top 3 labels — primary model**')
    st.dataframe(pd.DataFrame(main['top3']), hide_index=True,
        column_config={'Model score (%)': st.column_config.NumberColumn(format='%.2f%%')})
    for column, (family, prediction) in zip(st.columns(2), result['models'].items()):
        with column:
            st.metric(family if family=='GMM' else 'DNN / MLP', prediction['result'])
            st.write(prediction['diagnosis'])
            st.caption(f'Uncalibrated score: {prediction["score"]:.1%}')
    st.caption(result['context'])
    if result['metadata']['measurement_warnings']:
        st.warning('Some voice measurements were undefined. The saved training-only imputer was applied.')
    if result['mp3']:
        st.caption('MP3 compression effects were not validated in the WAV test set.')
    st.audio(result['audio'], sample_rate=16000, alt='Audio segment analyzed by both models')
    st.line_chart(pd.DataFrame({'Time (s)': np.arange(len(result['audio']))[::16]/16000,
        'Amplitude': result['audio'][::16]}), x='Time (s)', y='Amplitude', height=180,
        color='#0f766e', alt='Waveform of analyzed audio')
    st.caption(f'Analyzed {result["metadata"]["analyzed_seconds"]:.2f} seconds at 16 kHz in {result["elapsed_seconds"]:.2f} seconds. Uploaded audio is processed in memory and is not written to disk by this app.')

with st.expander('Models, supported labels and measured results'):
    st.write('Supported labels: ' + ', '.join(service.labels) + '.')
    st.write('Healthy/Pathological is derived from the five-class prediction. It is not a separately evaluated binary model. Laryngitis is the explicit laryngeal-inflammation label; other listed disorders also involve the larynx.')
    st.write('Development: 1,081 recordings from 987 speakers. Reserved test: 270 recordings from 246 different speakers. Five short recordings excluded.')
    st.dataframe(service.metrics[['model','test_balanced_accuracy','test_macro_f1']], hide_index=True)
    st.caption('Metric values above are fractions. Five-class balanced accuracy: GMM 35.95%, DNN 33.41%. Performance is modest. MLP was selected narrowly on validation; test scores did not change that selection.')
    st.write('Features: MFCC, spectral and energy statistics, pitch, Praat HNR and a frame-amplitude variation proxy. The proxy is not clinical shimmer. Both models reuse saved training-only preprocessing.')
    st.caption('Phone microphones and new recording conditions have not been externally validated.')
    for family in ('gmm','dnn'):
        st.image(str(ROOT/f'artifacts_svd_local/confusion_matrix_{family}.png'),
            caption=f'{family.upper()}: reserved-speaker test', alt=f'{family.upper()} test confusion matrix')
