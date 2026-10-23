"""Optional research inference for the separate experiment; does not change app.py."""
from pathlib import Path
import argparse,json,time
import joblib
import numpy as np
import svd_v4 as v4
from svd_v4_results import features,embedding_model

class ExperimentalPredictor:
    def __init__(self):
        self.release=json.loads((v4.OUT/'release.json').read_text())
        if v4.v3.sha(v4.OUT/'binary.joblib')!=self.release['model_hash']:
            raise ValueError('Experimental model hash mismatch')
        self.bundle=joblib.load(v4.OUT/'binary.joblib')
        for name,value in self.bundle['protocol']['source_hashes'].items():
            if v4.v3.sha(v4.ROOT/name)!=value:raise ValueError('Feature/model source changed: '+name)
        need=any(m['config']['subset'] in ['embedding','fusion'] for m in self.bundle['models'].values())
        self.encoder=embedding_model() if need else None

    def predict_voice(self,audio_file):
        start=time.perf_counter()
        if isinstance(audio_file,(str,Path)):content=Path(audio_file).read_bytes()
        elif hasattr(audio_file,'getvalue'):content=audio_file.getvalue()
        elif hasattr(audio_file,'read'):content=audio_file.read()
        else:content=audio_file
        if not content or len(content)>20*1024*1024:raise ValueError('Upload a nonempty WAV/MP3 smaller than 20 MB.')
        x,mel,_,meta=features(content,self.encoder)
        allp={n:v4.probability(m,x[None],mel[None])[0] for n,m in self.bundle['models'].items()}
        sel=self.bundle['selection'];p=np.mean([allp[n] for n in sel['members']],axis=0)
        pred=int(p[1]>=sel['winner']['cutoff']);score=float(p[pred]);uncertain=score<sel['threshold']
        return {'result':'Uncertain' if uncertain else self.bundle['labels'][pred],
            'leading_label':self.bundle['labels'][pred], 'score':score,
            'probabilities':dict(zip(self.bundle['labels'],map(float,p))),
            'model':sel['winner']['id'],'model_version':self.release['model_hash'][:12],
            'threshold':sel['threshold'],'decision_cutoff':sel['winner']['cutoff'],
            'components':{n:list(map(float,pp)) for n,pp in allp.items()},
            'elapsed_seconds':time.perf_counter()-start,'quality_metadata':meta,
            'scope':'Experimental development model. Academic project, not a medical diagnosis.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('audio');args=parser.parse_args()
    print(json.dumps(ExperimentalPredictor().predict_voice(args.audio),indent=2,default=str))
