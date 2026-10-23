"""V3 inference service. No fitting or dataset downloads during inference."""
import io,json,threading,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from threadpoolctl import threadpool_limits
from svd_training import load_audio
from svd_advanced import extract_advanced,predict as subtype_predict
from svd_v3 import VERSION,LABELS,extract,probability,sha,source_hashes,spec_for
from .advanced_web import decode_upload,warm,DESCRIPTIONS

class ReleaseService:
    def __init__(self,root):
        self.root=Path(root);self.directory=self.root/'artifacts_v3'
        self.release=json.loads((self.directory/'release.json').read_text())
        if self.release['status']!='complete':raise ValueError('Incomplete experiment cannot be deployed')
        if sha(self.directory/'binary.joblib')!=self.release['model_sha256']:raise ValueError('Model checksum mismatch')
        if self.release['source_hashes']!=source_hashes():raise ValueError('Feature/source version mismatch')
        self.bundle=joblib.load(self.directory/'binary.joblib')
        for sr,spec in self.bundle['feature_specs'].items():
            if spec!=spec_for(int(sr)):raise ValueError('Feature specification mismatch')
        self.subtype=joblib.load(self.root/'artifacts_svd_v2/subtype.joblib')
        expected=json.loads((self.root/'artifacts_svd_v2/model_hashes.json').read_text())
        if sha(self.root/'artifacts_svd_v2/subtype.joblib')!=expected['subtype.joblib']:raise ValueError('Subtype checksum mismatch')
        for model in self.bundle['models'].values():warm(model)
        self.lock=threading.RLock();self.labels=LABELS;self.version=VERSION
        self.metrics=pd.read_csv(self.directory/'nested_metrics.csv')
        self.summary=json.loads((self.directory/'nested_summary.json').read_text())
        self.historical=json.loads((self.root/'artifacts_svd_v2/external_binary_test_metrics.json').read_text())
        self.known={sr:set(v) for sr,v in self.bundle['known_hashes'].items()}
        # Include recordings from the historical SVD project, even if not retained now.
        old=self.root/'artifacts_svd_local/partition_manifest.csv'
        if old.exists():self.known['16000'].update(pd.read_csv(old).audio_hash)

    def analyze(self,content,filename):
        start=time.perf_counter();content,mp3,duration=decode_upload(content,filename)
        warnings=[]
        with sf.SoundFile(io.BytesIO(content)) as file:
            count=min(file.frames,int(5*file.samplerate));file.seek((file.frames-count)//2)
            raw=file.read(count,dtype='float64',always_2d=True)
            if np.mean(np.abs(raw)>=.999)>=.01:warnings.append('Possible clipping: at least 1% of raw samples are near full scale.')
        with self.lock,threadpool_limits(limits=2):
            rates={m['config']['sr'] for m in self.bundle['models'].values()}|{16000}
            features={sr:extract(content,sr) for sr in rates}
            probabilities={name:probability(model,features[model['config']['sr']][0][None],
                features[model['config']['sr']][1][None])[0] for name,model in self.bundle['models'].items()}
            selection=self.bundle['selection'];p=np.mean([probabilities[n] for n in selection['members']],axis=0)
            p=p/p.sum();index=int(p.argmax());score=float(p[index]);threshold=selection['threshold']
            rich16,mel16,_,_=features[16000]
            # The unchanged subtype bundle requires its original v2 feature contract.
            original_rich,original_mel,_=extract_advanced(content)
            q=subtype_predict(self.subtype,original_rich[None],original_mel[None])[0]
            q=np.asarray(q,float);q/=q.sum();order=np.argsort(-q)
            display_sr=max(self.bundle['models'][n]['config']['sr'] for n in selection['members'])
            audio,metadata=load_audio(content,spec_for(display_sr))
            missing=sum(int(np.isnan(features[sr][0]).sum()) for sr in rates)
            if missing:warnings.append('Some measurements were undefined; saved training-only imputers were applied.')
            if not np.isfinite(rich16[39]) or rich16[39]<.3:
                warnings.append('Limited detected voicing. Confirm that this is a steady aa vowel; this check does not indicate health.')
        known=any(h in self.known[str(sr)] for sr,(_,_,h,_) in features.items())
        models=[]
        for name in (selection['best_gmm'],selection['best_dnn']):
            pp=probabilities[name];models.append({'model':name,'label':LABELS[int(pp.argmax())],'score':float(pp.max())})
        return dict(result='Uncertain' if score<threshold else LABELS[index],leading_label=LABELS[index],
            binary_uncertain=score<threshold,score=score,pathological_probability=float(p[1]),threshold=threshold,
            selected_model=selection['winner']['id'],model_version=VERSION,models=models,
            subtype_uncertain=True,subtype_label=self.subtype['labels'][int(order[0])],
            subtype_description=DESCRIPTIONS[self.subtype['labels'][int(order[0])]],
            top3=[{'Disorder group':self.subtype['labels'][int(i)],'Conditional model score (%)':float(q[i]*100)} for i in order],
            audio=audio,sample_rate=display_sr,metadata=metadata,missing_features=missing,quality_warnings=warnings,
            context='Known project development recording; not an independent unseen test.' if known else
                'Unrecognized recording. Speaker identity, correct vocal task and recording conditions cannot be verified.',
            mp3=mp3,original_seconds=duration,elapsed_seconds=time.perf_counter()-start)

def predict_voice(audio_file,service=None):
    service=service or ReleaseService(Path(__file__).resolve().parents[1])
    if hasattr(audio_file,'getvalue'):return service.analyze(audio_file.getvalue(),getattr(audio_file,'name','voice.wav'))
    path=Path(audio_file);return service.analyze(path.read_bytes(),path.name)
