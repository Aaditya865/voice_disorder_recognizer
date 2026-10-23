"""Streamlit demonstration of the frozen SVD v2 models."""
from pathlib import Path
import hashlib
import logging
import numpy as np
import pandas as pd
import streamlit as st
from voice_review.advanced_web import AdvancedService

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Voice Disorder Recognizer',page_icon=':material/graphic_eq:',layout='centered')
st.title('Voice Disorder Recognizer')
st.write('Analyze a sustained aa vowel with a GMM + convolutional DNN ensemble.')
st.caption('Academic project, not a medical diagnosis.')
st.warning('Research demonstration: performance did not transfer reliably to an external dataset. Even high-scoring results can be wrong; do not use this app to establish or rule out a disorder.')

@st.cache_resource(max_entries=1,show_spinner='Loading trained models...')
def load_models():
    return AdvancedService(ROOT)

uploaded=st.file_uploader('Upload your voice recording',type=['wav','mp3'],max_upload_size=20,key='voice_file',
    help='Use a steady 1–3 second aa vowel. Minimum 0.5 seconds; at most the central 5 seconds are analyzed.')
with st.expander('Or record audio'):
    recorded=st.audio_input('Record a steady aa vowel',sample_rate=16000,key='voice_mic')
source=uploaded if uploaded is not None else recorded
both=uploaded is not None and recorded is not None
content=source.getvalue() if source is not None else None
fingerprint=hashlib.sha256(content).hexdigest() if content else None
if st.session_state.get('v2_input_hash')!=fingerprint or both:
    st.session_state.pop('v2_analysis',None)
    st.session_state['v2_input_hash']=fingerprint
if both:st.info('Clear the upload or microphone recording to analyze one recording.')
try:
    service=load_models()
except Exception:
    logging.exception('Model loading failed')
    st.error('The trained models could not be loaded. Check the model files and install the supplied requirements.')
    st.stop()

if st.button('Analyze Voice',type='primary',width='stretch',disabled=source is None or both):
    st.session_state.pop('v2_analysis',None)
    try:
        with st.spinner('Analyzing your voice...'):
            st.session_state['v2_analysis']=service.analyze(content,source.name)
    except ValueError as exc:st.error(str(exc))
    except Exception:
        logging.exception('Inference failed')
        st.error('Unable to analyze this audio. Try a clear WAV recording of a steady vowel.')

result=st.session_state.get('v2_analysis')
if result is not None:
    st.subheader(f'Research result: {result["result"]}')
    if result['binary_uncertain']:
        st.warning('Uncertain, result not reliable. The leading model label is '+result['leading_label']+'.')
    st.progress(result['score'],text=f'Leading binary model score: {result["score"]:.1%}')
    st.caption(f'GMM + CNN ensemble; temperature-calibrated on development speakers. Below {result["threshold"]:.0%}, the result is uncertain. Calibration did not transfer reliably to external data; scores are not medical probabilities.')
    if result['leading_label']=='Pathological':
        st.write('**Disorder type: uncertain — specific cause unresolved.**' if result['subtype_uncertain']
            else '**Disorder group:** '+result['subtype_label'])
        st.write('**Top 3 supported disorder groups — exploratory only**')
        st.dataframe(pd.DataFrame(result['top3']),hide_index=True,column_config={
            'Conditional model score (%)':st.column_config.NumberColumn(format='%.2f%%')})
        st.caption('These scores are conditional on one of these three groups. Other disorders and comorbid conditions are not represented; no subtype was reliable enough for confident identification.')
        st.write('Leading candidate: '+result['subtype_label']+'. '+result['subtype_description'])
    else:
        st.write('A healthy model label does not rule out a disorder. No disorder type is assigned.')
        with st.expander('Exploratory disorder-group scores'):
            st.dataframe(pd.DataFrame(result['top3']),hide_index=True)
            st.caption('Conditional scores across three pathology groups only; these are not diagnoses or the probability that a healthy voice has these conditions.')
    for column,member in zip(st.columns(len(result['models'])),result['models']):
        with column:
            st.metric(member['model'],member['label'])
            st.caption(f'Model score: {member["score"]:.1%}')
    if len({m['label'] for m in result['models']})>1:
        st.info('The component models disagree. Interpret the ensemble result cautiously.')
    st.caption(result['context'])
    if result['missing_features']:st.warning('Some voice measurements were undefined; the saved training-only imputer was applied.')
    if result['mp3']:st.caption('MP3 compression effects have not been separately validated.')
    st.audio(result['audio'],sample_rate=16000,alt='Analyzed audio segment')
    st.line_chart(pd.DataFrame({'Time (s)':np.arange(len(result['audio']))[::16]/16000,
        'Amplitude':result['audio'][::16]}),x='Time (s)',y='Amplitude',height=180,color='#0f766e',alt='Analyzed voice waveform')
    st.caption(f'Analyzed {result["metadata"]["analyzed_seconds"]:.2f} seconds at 16 kHz in {result["elapsed_seconds"]:.2f} seconds. Uploads are processed in memory, not saved to disk by this app.')

with st.expander('Measured results and limitations'):
    st.write('Development: 1,212 SVD speakers (660 healthy, 552 pathological). Four-fold speaker-disjoint model selection; the old SVD test is now development data.')
    st.write('Subtype development: 185 functional-group, 73 laryngitis and 141 vocal-fold-paresis speakers. Mixed or other diagnoses contribute only to binary training.')
    table=service.comparison[['task','model','balanced_accuracy_mean','balanced_accuracy_std','macro_f1_mean','roc_auc_mean']]
    st.dataframe(table,hide_index=True)
    st.caption('Fractions; development CV is used for selection and is optimistic as a final performance estimate. Subtype identification has no independent final test.')
    external=service.external
    st.write('**One-time external binary test: 42 reserved VOICED recordings (12 healthy, 30 pathological).**')
    st.write(f'Balanced accuracy: {external["balanced_accuracy"]:.4f}; macro-F1: {external["macro_f1"]:.4f}; sensitivity: {external["sensitivity"]:.4f}; specificity: {external["specificity"]:.4f}; ROC-AUC: {external["roc_auc"]:.4f}.')
    ci=external['balanced_accuracy_95_ci']
    st.caption(f'Participant-level bootstrap 95% interval for balanced accuracy: {ci[0]:.4f}–{ci[1]:.4f}. This small cross-database test does not establish performance on phone recordings or clinical reliability.')
    st.image(str(service.directory/'external_binary_test_confusion.png'),caption='External binary test — all 42 cases, including uncertain cases')
    st.image(str(service.directory/'subtype_cv_confusion.png'),caption='Subtype development CV — exploratory, selection-biased')
    st.write('Inputs: 97 acoustic features, including MFCCs and deltas, CPPS, HNR, pitch, jitter, shimmer and spectral statistics; the CNN additionally uses log-Mel spectrograms. All learned preprocessing is fitted inside training partitions.')
