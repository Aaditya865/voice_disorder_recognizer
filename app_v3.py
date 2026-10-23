"""Academic voice recognizer: frozen two-source development experiment."""
from pathlib import Path
import hashlib,logging
import numpy as np
import pandas as pd
import streamlit as st
from voice_review.release_web import ReleaseService

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Voice Disorder Recognizer',page_icon=':material/graphic_eq:',layout='centered')
st.title('Voice Disorder Recognizer')
st.write('A sustained aa vowel, analyzed by GMM and DNN research models.')
st.caption('Academic project, not a medical diagnosis.')
st.warning('This revision has development validation only. Performance on a genuinely new population is unverified; even high-scoring predictions can be wrong.')

@st.cache_resource(max_entries=1,show_spinner='Loading saved models...')
def load_models(release_hash):return ReleaseService(ROOT)

uploaded=st.file_uploader('Upload your voice recording',type=['wav','mp3'],max_upload_size=20,key='voice_file',
    help='Record a steady aa vowel, ideally 1–3 seconds. Minimum 0.5 seconds; at most the central 5 seconds are analyzed.')
with st.expander('Or record audio'):
    recorded=st.audio_input('Record a steady aa vowel',sample_rate=16000,key='voice_mic')
source=uploaded if uploaded is not None else recorded
both=uploaded is not None and recorded is not None
content=source.getvalue() if source is not None else None
fingerprint=hashlib.sha256(content).hexdigest() if content else None
if st.session_state.get('v3_input_hash')!=fingerprint or both:
    st.session_state.pop('v3_analysis',None);st.session_state['v3_input_hash']=fingerprint
if both:st.info('Clear the upload or microphone recording to analyze one input.')
try:
    release_hash=hashlib.sha256((ROOT/'artifacts_v3/release.json').read_bytes()).hexdigest()
    service=load_models(release_hash)
except Exception:
    logging.exception('Release loading failed')
    st.error('The saved release could not be loaded. Check the model files and install the supplied requirements. No prediction was generated.')
    st.stop()

if st.button('Analyze Voice',type='primary',width='stretch',disabled=source is None or both):
    st.session_state.pop('v3_analysis',None)
    try:
        with st.spinner('Analyzing the recording...'):
            st.session_state['v3_analysis']=service.analyze(content,source.name)
    except ValueError as exc:st.error(str(exc))
    except Exception:
        logging.exception('Inference failed');st.error('Unable to analyze this recording. Try a clear WAV of a steady aa vowel.')

result=st.session_state.get('v3_analysis')
if result is not None:
    st.subheader('Research result: '+result['result'])
    if result['binary_uncertain']:st.warning('Uncertain, result not reliable. Leading model label: '+result['leading_label']+'.')
    st.progress(result['score'],text=f'Leading binary model score: {result["score"]:.1%}')
    st.caption('Selected model: '+result['selected_model']+' | Version: '+result['model_version'])
    if result['threshold']>1:
        st.caption('No development threshold met the required accuracy and coverage in both datasets. All outputs remain uncertain.')
    else:
        st.caption(f'The development rejection threshold is {result["threshold"]:.0%}. Calibrated model scores are not medical probabilities and do not guarantee correctness.')
    for warning in result['quality_warnings']:st.warning(warning)
    if result['leading_label']=='Pathological':
        st.write('**Disorder type: uncertain — specific cause unresolved.**')
        st.write('**Top 3 supported groups — exploratory only**')
        st.dataframe(pd.DataFrame(result['top3']),hide_index=True,alt='Exploratory pathology group rankings',
            column_config={'Conditional model score (%)':st.column_config.NumberColumn(format='%.2f%%')})
        st.write('Leading candidate: '+result['subtype_label']+'. '+result['subtype_description'])
        st.caption('The unchanged subtype model covers only three groups. These scores assume one of those groups; other conditions cannot be ruled out.')
    else:
        st.write('A healthy model label does not rule out a disorder. No disorder type is assigned.')
        with st.expander('Exploratory pathology-group scores'):
            st.dataframe(pd.DataFrame(result['top3']),hide_index=True,alt='Conditional subtype scores')
            st.caption('Not the probability that a healthy voice has a disease. Subtype predictions are always uncertain.')
    for col,member in zip(st.columns(2),result['models']):
        with col:
            st.metric(member['model'],member['label']);st.caption(f'Model score: {member["score"]:.1%}')
    if len({m['label'] for m in result['models']})>1:st.info('The GMM and DNN disagree; interpret the result cautiously.')
    st.caption(result['context'])
    if result['mp3']:st.caption('MP3 decoding is supported; compression effects have not been independently validated.')
    st.audio(result['audio'],sample_rate=result['sample_rate'],alt='Analyzed audio segment')
    step=max(1,result['sample_rate']//1000)
    st.line_chart(pd.DataFrame({'Time (s)':np.arange(len(result['audio']))[::step]/result['sample_rate'],
        'Amplitude':result['audio'][::step]}),x='Time (s)',y='Amplitude',height=180,color='#0f766e',alt='Analyzed audio waveform')
    st.caption(f'Analyzed {result["metadata"]["analyzed_seconds"]:.2f} seconds; waveform at {result["sample_rate"]} Hz. Inference took {result["elapsed_seconds"]:.2f} seconds. Uploads remain in memory and are not saved to disk by this app.')

with st.expander('Measured results, scope and limitations'):
    st.write('Development: 1,212 SVD participants and 208 VOICED participant proxies. All previously evaluated recordings are development data for this revision.')
    st.write('Four outer participant-disjoint folds assess a model-selection procedure using three inner folds. No new untouched external evaluation is available.')
    st.dataframe(service.metrics[['role','source','n','balanced_accuracy_mean','balanced_accuracy_std',
        'macro_f1_mean','sensitivity_mean','specificity_mean','roc_auc_mean']],hide_index=True,alt='Nested development model comparison')
    st.caption('Values are fractions; mean and standard deviation are across four outer folds. GMM/DNN rows evaluate family-selection procedures, not a test of the final refitted artifact.')
    for source in ('SVD','VOICED'):
        m=service.summary[source];ci=m['confidence_intervals']['balanced_accuracy']
        st.write(f'{source}: pooled balanced accuracy {m["balanced_accuracy"]:.4f}; bootstrap 95% interval {ci[0]:.4f}–{ci[1]:.4f}; accepted coverage {m["coverage"]:.1%}.')
        st.image(str(service.directory/f'confusion_{source}.png'),caption=f'{source}: nested development predictions, including uncertain cases',alt=f'{source} development confusion matrix')
    st.write('Historical v2 external test: balanced accuracy 0.525; only 4/30 pathological recordings detected. That result remains preserved. VOICED is now development data, so revised VOICED scores are not a new external-test improvement.')
    st.caption('Age/sex imbalance, source differences, small subgroups and the vowel-only task limit interpretation. The subtype model remains exploratory. Neither models nor thresholds establish clinical reliability.')
