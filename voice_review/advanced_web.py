"""Cached, bounded inference using the exact SVD v2 training features."""
import io
import json
from pathlib import Path
import threading
import time
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from threadpoolctl import threadpool_limits
from svd_training import SPEC, load_audio
from svd_advanced import FEATURE_VERSION, extract_advanced, predict, cnn_network, digest

DESCRIPTIONS = {
    'Functional dysphonia group': 'A broad group involving difficulties in voice production, including excessive muscular effort. Voices may sound strained, weak or inconsistent; these sounds are not unique to this group.',
    'Laryngitis': 'Inflammation of the larynx (voice box). It can produce hoarseness or roughness, but audio alone cannot establish inflammation or its cause.',
    'Vocal fold paresis': 'Reduced movement of a vocal fold. The voice may sound weak or breathy; audio alone cannot confirm reduced movement.',
}

def warm(bundle):
    """Construct each neural network once; weights remain in memory thereafter."""
    if 'members' in bundle:
        for member in bundle['members']: warm(member)
    elif bundle['config']['family']=='CNN':
        import torch
        torch.set_num_threads(2)
        dim=bundle['weights']['head.0.weight'].shape[1]-bundle['config']['width']*8
        net=cnn_network(bundle['config']['width'],bundle['config']['dropout'],dim,bundle['classes'])
        net.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in bundle['weights'].items()})
        net.eval()
        bundle['_runtime']=net

def decode_upload(content,filename):
    if Path(filename).suffix.lower() not in {'.wav','.mp3'}:
        raise ValueError('Please upload a WAV or MP3 file.')
    if not 0<len(content)<=20*1024*1024:
        raise ValueError('Choose a nonempty file smaller than 20 MB.')
    try:
        with sf.SoundFile(io.BytesIO(content)) as source:
            if source.format not in {'WAV','WAVEX','RF64','MP3'}:
                raise ValueError('The contents must be WAV or MP3 audio.')
            if source.channels not in (1,2) or not 8000<=source.samplerate<=192000:
                raise ValueError('Use mono/stereo audio sampled between 8 and 192 kHz.')
            duration=source.frames/source.samplerate
            if duration<SPEC['min_seconds']:
                raise ValueError('Record at least half a second of a steady aa vowel; 1–3 seconds is recommended.')
            mp3=source.format=='MP3'
            if mp3:
                count=min(source.frames,int(SPEC['max_seconds']*source.samplerate))
                source.seek((source.frames-count)//2)
                data=source.read(count,dtype='float64',always_2d=True)
                encoded=io.BytesIO()
                sf.write(encoded,data,source.samplerate,format='WAV',subtype='DOUBLE')
                content=encoded.getvalue()
    except (RuntimeError,OSError) as exc:
        raise ValueError('Unable to decode this recording. Export it as WAV and try again.') from exc
    return content,mp3,duration

class AdvancedService:
    def __init__(self,root):
        self.root=Path(root)
        self.directory=self.root/'artifacts_svd_v2'
        expected=json.loads((self.directory/'model_hashes.json').read_text())
        for filename,sha in expected.items():
            if digest(self.directory/filename)!=sha: raise ValueError('Saved model integrity check failed.')
        self.bundles={task:joblib.load(self.directory/f'{task}.joblib') for task in ('binary','subtype')}
        for bundle in self.bundles.values():
            if bundle['feature_version']!=FEATURE_VERSION:raise ValueError('Feature version mismatch.')
            warm(bundle)
        self.lock=threading.RLock()
        self.comparison=pd.read_csv(self.directory/'comparison.csv')
        self.external=json.loads((self.directory/'external_binary_test_metrics.json').read_text())
        self.known=set(self.bundles['binary']['development_hashes'])
        old=self.root/'artifacts_svd_local/partition_manifest.csv'
        if old.exists():self.known.update(pd.read_csv(old).audio_hash)
        self.external_hashes=set(pd.read_csv(self.directory/'external_binary_test_predictions.csv').audio_hash)

    def analyze(self,content,filename):
        start=time.perf_counter()
        content,mp3,duration=decode_upload(content,filename)
        with self.lock,threadpool_limits(limits=2):
            rich,mel,ahash=extract_advanced(content)
            audio,metadata=load_audio(content)
            x,m=rich[None],mel[None]
            binary=self.bundles['binary'];subtype=self.bundles['subtype']
            p=np.asarray(predict(binary,x,m)[0],dtype=float);p/=p.sum()
            q=np.asarray(predict(subtype,x,m)[0],dtype=float);q/=q.sum()
            index=int(p.argmax());score=float(p[index])
            threshold=binary['selection']['uncertainty_threshold']
            uncertain=score<threshold
            members=[]
            for member in binary.get('members',[binary]):
                mp=predict(member,x,m)[0]
                members.append({'model':member['config']['id'],'label':binary['labels'][int(mp.argmax())],
                    'score':float(mp.max())})
        order=np.argsort(-q)
        subtype_uncertain=float(q.max())<subtype['selection']['uncertainty_threshold']
        if ahash in self.known:
            context='Previously used during project development. This is not an independent unseen test.'
        elif ahash in self.external_hashes:
            context='External evaluation recording: excluded from training, but its final test result has already been measured.'
        else:
            context='Unrecognized recording. Speaker identity and recording conditions cannot be verified from the upload.'
        return {'result':'Uncertain' if uncertain else binary['labels'][index],
            'leading_label':binary['labels'][index],'binary_uncertain':uncertain,'score':score,
            'pathological_probability':float(p[1]),'threshold':threshold,'models':members,
            'subtype_uncertain':subtype_uncertain,'subtype_label':subtype['labels'][int(order[0])],
            'subtype_description':DESCRIPTIONS[subtype['labels'][int(order[0])]],
            'top3':[{'Disorder group':subtype['labels'][int(i)],'Conditional model score (%)':float(q[i]*100)} for i in order[:3]],
            'audio':audio,'metadata':metadata,'missing_features':int(np.isnan(rich).sum()),
            'context':context,'mp3':mp3,'original_seconds':duration,'elapsed_seconds':time.perf_counter()-start}

def predict_voice(audio_file,service=None):
    """Accept a path or Streamlit UploadedFile. Reuse a cached service in a website."""
    if service is None: service=AdvancedService(Path(__file__).resolve().parents[1])
    if hasattr(audio_file,'getvalue'):
        return service.analyze(audio_file.getvalue(),getattr(audio_file,'name','recording.wav'))
    path=Path(audio_file)
    return service.analyze(path.read_bytes(),path.name)
