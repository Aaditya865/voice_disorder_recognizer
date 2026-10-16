"""Cached model resources and bounded audio input for the Streamlit website."""
from pathlib import Path
import io
import json
import threading
import time
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.special import expit
from .data import ROOT, assert_partition
from .features import preprocess, extract, audio_fingerprint
from .models import VoiceCNN
from .laryngeal import network, NAMES

MAX_BYTES = 20*1024*1024


def read_upload(content, filename):
    if Path(filename).suffix.lower() not in {'.wav','.mp3'}:
        raise ValueError('Please choose a WAV or MP3 audio file.')
    if not 0 < len(content) <= MAX_BYTES:
        raise ValueError('Please choose a nonempty audio file smaller than 20 MB.')
    try:
        with sf.SoundFile(io.BytesIO(content)) as source:
            if source.format not in {'WAV','WAVEX','RF64','MP3'}:
                raise ValueError('The file contents must be WAV or MP3 audio.')
            if source.channels not in {1,2}:
                raise ValueError('Please export your recording as mono or stereo audio.')
            sr = source.samplerate
            if not 8000 <= sr <= 192000:
                raise ValueError('Use audio sampled between 8 kHz and 192 kHz.')
            duration = source.frames/sr
            if duration < 3:
                raise ValueError('Please record at least three seconds of a steady “aa” vowel.')
            # Bound decoded memory even for long, highly compressed MP3 files.
            count = min(source.frames,60*sr)
            source.seek(max(0,(source.frames-count)//2))
            audio = source.read(count,dtype='float64',always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise ValueError('This audio could not be decoded. Try exporting it as WAV.') from exc
    return preprocess(audio,sr), {'original_seconds':duration,'original_sample_rate':sr,
                                  'trimmed':duration>3,'long_clip':duration>60}


class ModelService:
    """One shared read-only model pool; no model files are loaded per prediction."""
    def __init__(self,root=ROOT):
        self.root = Path(root)
        self.lock = threading.RLock()
        torch.set_num_threads(2)
        self.manifest = pd.read_csv(self.root/'data/manifest.csv')
        self.splits = json.loads((self.root/'data/splits.json').read_text())
        selected = json.loads((self.root/'results/selection.json').read_text())
        assert_partition(self.manifest,self.splits['development'],self.splits['test'])
        self.hash_to_id = dict(zip(self.manifest.audio_hash,self.manifest.record_id))
        self.pool = {}
        for family in ['GMM','DNN']:
            self._load(('binary',family,None),self.root/f'models/{family.lower()}.joblib')
            self._load(('subtype',family,None),self.root/f'models/laryngeal/{family.lower()}.joblib')
            for fold,part in enumerate(self.splits['folds']):
                assert_partition(self.manifest,part['train'],part['validation'])
                self._load(('binary',family,fold),self.root/f'models/validation/{selected[family]["config_id"]}_fold{fold}.joblib')
                self._load(('subtype',family,fold),self.root/f'models/laryngeal/{family.lower()}_fold{fold}.joblib')

    def _load(self,key,path):
        bundle = joblib.load(path)
        if key[1]=='DNN':
            model = network() if key[0]=='subtype' else VoiceCNN(bundle['config']['dropout'],12 if bundle['config']['features']=='combined' else 0)
            model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in bundle['weights'].items()})
            model.eval()
            bundle['_runtime_network'] = model
        self.pool[key] = bundle

    def _score(self,key,seq,quality):
        bundle = self.pool[key]
        s,q,tab = bundle['transform'].transform(seq,quality)
        if key[1]=='GMM':
            values = np.column_stack([m.score_samples(tab) for m in bundle['models']])
            return values if key[0]=='subtype' else values[:,1]-values[:,0]
        with torch.no_grad():
            return bundle['_runtime_network'](torch.from_numpy(s),torch.from_numpy(q)).numpy()

    def analyze(self,content,filename):
        started = time.perf_counter()
        with self.lock:
            audio,meta = read_upload(content,filename)
            rid = self.hash_to_id.get(audio_fingerprint(audio))
            if rid in self.splits['test']:
                raise ValueError('This is a reserved final-test recording. Please use a supplied demo sample or your own audio.')
            fold = None
            context = 'New or unrecognized recording. Real-world performance has not been validated.'
            if rid in self.splits['development']:
                fold = next(i for i,p in enumerate(self.splits['folds']) if rid in p['validation'])
                if rid in self.splits['folds'][fold]['train']:
                    raise RuntimeError('Speaker isolation check failed.')
                for family in ['GMM','DNN']:
                    if rid in self.pool[('subtype',family,fold)]['training_ids']:
                        raise RuntimeError('Subtype speaker isolation check failed.')
                context = 'Development validation example: this speaker was excluded from these models’ fitting. Validation informed model settings; this is not a final independent test.'
            seq,q,quality = extract(audio)
            if not np.isfinite(q).any():
                raise ValueError('Voice-quality measurements could not be extracted. Try a clearer, steady vowel recording.')
            results = {}
            for family in ['GMM','DNN']:
                raw = float(self._score(('binary',family,fold),seq[None],q[None])[0])
                subtype_values = self._score(('subtype',family,fold),seq[None],q[None])[0]
                if not np.isfinite(raw) or not np.isfinite(subtype_values).all():
                    raise ValueError('A reliable numerical result could not be computed for this audio.')
                pathological = raw>=0
                subtype = int(np.argmax(subtype_values))
                result_label = 'Pathological' if pathological else 'Healthy'
                type_label = ('LARYNGEAL — laryngitis-labelled group' if subtype==1 else
                              'Other pathology — specific disorder unresolved') if pathological and subtype else 'Not resolved'
                if not pathological:
                    type_label = 'Not assigned by the binary result'
                results[family] = {'result':result_label,'raw_score':raw,
                    'model_score':float(expit(abs(raw))), 'subtype':type_label,
                    'three_class_result':NAMES[subtype]}
        return {'models':results,'audio':audio,'quality':quality,'input':meta,'context':context,
                'record_id':rid,'fold':fold,'elapsed_seconds':time.perf_counter()-started}
