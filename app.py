"""Run with: python -m streamlit run app.py"""
from pathlib import Path
import hashlib
import logging
import numpy as np
import pandas as pd
import streamlit as st
from voice_review.web_inference import ModelService

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Voice Disorder Recognizer',page_icon=':material/graphic_eq:',layout='centered')
st.title('Voice Disorder Recognizer')
st.write('Upload a sustained “aa” vowel to compare GMM and DNN voice predictions.')
st.caption('Academic project, not a medical diagnosis.')


@st.cache_resource(max_entries=1,show_spinner='Loading trained models…')
def load_models():
    return ModelService(ROOT)


uploaded = st.file_uploader('Upload your voice recording',type=['wav','mp3'],
                            max_upload_size=20,help='At least 3 seconds. The central 3 seconds are analyzed.',key='voice_file')
with st.expander('Or record audio'):
    recorded = st.audio_input('Record a steady “aa” for at least 3 seconds',sample_rate=16000,key='voice_mic')

source = uploaded if uploaded is not None else recorded
both = uploaded is not None and recorded is not None
content = source.getvalue() if source is not None else None
name = source.name if source is not None else ''
fingerprint = hashlib.sha256(content).hexdigest() if content else None
if st.session_state.get('input_hash') != fingerprint or both:
    st.session_state.pop('analysis',None)
    st.session_state['input_hash'] = fingerprint
if both:
    st.info('Remove the uploaded file or clear the microphone recording, then analyze one recording.')

try:
    service = load_models()
except Exception:
    logging.exception('Model initialization failed')
    st.error('Models could not be loaded. Launch the app from the complete project folder with its supplied environment.')
    st.stop()

if st.button('Analyze Voice',type='primary',width='stretch',disabled=source is None or both):
    st.session_state.pop('analysis',None)
    try:
        with st.spinner('Analyzing your voice…'):
            st.session_state['analysis'] = service.analyze(content,name)
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception('Voice analysis failed')
        st.error('This recording could not be analyzed. Try a clear WAV recording of a steady vowel.')

result = st.session_state.get('analysis')
if result is not None:
    main = result['models']['GMM']
    st.subheader(f'Result: {main["result"]}')
    st.write('**Primary model: GMM** · chosen from the existing binary development comparison.')
    st.write(f'**Disorder group:** {main["subtype"]}')
    st.caption('Subtype suggestions are experimental and have frequent false positives. “Laryngeal” covers laryngitis-labelled recordings only; other pathology may also involve the larynx.')
    st.progress(main['model_score'],text=f'Model score: {main["model_score"]:.1%} — uncalibrated')
    st.caption('Validated confidence is unavailable. This score is not a medical probability and can be high for an incorrect prediction.')
    columns = st.columns(2)
    for column,(family,prediction) in zip(columns,result['models'].items()):
        with column:
            st.metric(family,prediction['result'])
            st.caption(f'Model score {prediction["model_score"]:.1%} · uncalibrated')
            st.write(prediction['subtype'])
    if result['models']['GMM']['result'] != result['models']['DNN']['result']:
        st.warning('The two binary models disagree. Neither result establishes a diagnosis.')
    st.caption(result['context'])
    if result['input']['trimmed']:
        st.caption(f'Original duration: {result["input"]["original_seconds"]:.1f} s. Analyzed the central 3 seconds at 8 kHz.')
    if result['quality']['missing_quality_features']:
        st.warning('Some acoustic measurements were undefined. The saved training-time imputation was applied.')
    st.caption('Playback: the three-second segment analyzed from your upload (converted for browser playback only).')
    st.audio(result['audio'],sample_rate=8000,alt='Analyzed segment of uploaded voice recording')
    waveform = pd.DataFrame({'Time (s)':np.arange(len(result['audio']))[::8]/8000,
                             'Amplitude':result['audio'][::8]})
    st.line_chart(waveform,x='Time (s)',y='Amplitude',height=180,color='#0f766e',
                  alt='Waveform of the central three seconds used for prediction')
    st.caption(f'Analysis completed in {result["elapsed_seconds"]:.2f} seconds. Audio stays in this local session and is not written to disk.')

with st.expander('About the results and demo samples'):
    st.write('Use the three WAV files in samples/. Recognized dataset recordings automatically use models trained without that speaker. New recordings use the saved full-development models.')
    st.write('Binary development balanced accuracy: GMM 61.6%; DNN 57.5%. The same folds selected settings. Final-test performance remains unmeasured.')
    st.write('The subtype extension compares Healthy, Laryngeal (laryngitis-labelled), and Other pathology. It does not identify all laryngeal disorders or replace an examination.')
    st.write('Exploratory three-class balanced accuracy: GMM 51.5%; DNN 38.4%. Laryngitis-group precision: GMM 32.4%; DNN 27.3%. These models frequently assign the wrong subtype.')
    st.image(str(ROOT/'results/laryngeal/confusion_matrices.png'),caption='Auxiliary three-class development results; the website first applies its binary decision.')
    st.caption('The waveform is displayed at reduced resolution. No noise removal or pitch-changing processing is applied. MP3 compression and phone microphones introduce unvalidated differences from the training data.')
