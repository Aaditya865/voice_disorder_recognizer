"""Pinned, frozen WavLM embeddings; no fitting on clinical audio.

Install requirements-v4.txt first. Run from a reproduced workspace after obtaining
the exact cohort audio. Cache keys include audio bytes, extractor and model hashes.
"""
from pathlib import Path
import json,sys,os
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'experiments/v4'
local=ROOT.parents[1]/'work/v4_dependencies'
if local.exists():sys.path.insert(0,str(local))
os.environ['HF_HUB_DISABLE_XET']='1'
import numpy as np,pandas as pd,torch
from transformers import WavLMModel
from svd_v3 import audio_bytes,spec_for,sha,key,source_hashes,dump
from svd_training import load_audio
REVISION='4c66d4806a428f2e922ccfa1a962776e232d487b'

def main():
    torch.set_num_threads(2);directory=OUT/'wavlm'
    if directory.exists():model=WavLMModel.from_pretrained(directory,local_files_only=True)
    else:
        model=WavLMModel.from_pretrained('microsoft/wavlm-base-plus',revision=REVISION)
        model.save_pretrained(directory,safe_serialization=True)
    model.eval();old=pd.read_csv(ROOT/'data/v3/featured_manifest.csv')
    extra=pd.read_csv(OUT/'download_audit.csv').fillna('');extra=extra[(extra.role=='development')&(extra.error=='')]
    frame=pd.concat([old,extra],ignore_index=True);rows=[];cache=OUT/'verified_embedding_cache';cache.mkdir(exist_ok=True)
    specification={'revision':REVISION,'source_hashes':source_hashes(),'extractor':sha(__file__),
        'model_hashes':{p.name:sha(p) for p in directory.glob('*') if p.is_file()},
        'layer':6,'pooling':'mean/std unbiased=False','normalization':'zero mean/unit variance +1e-7','sampling_rate':16000}
    for i,r in frame.iterrows():
        path=Path(r.path);actual=key([sha(path),sha(path.with_suffix('.dat'))]) if path.suffix=='.hea' else sha(path)
        if actual!=r.raw_hash:raise ValueError('Audio changed: '+r.record_id)
        ident=key([actual,specification]);dest=cache/(ident+'.npz')
        if not dest.exists():
            x,_=load_audio(audio_bytes(path),spec_for(16000));x=(x-x.mean())/np.sqrt(x.var()+1e-7)
            with torch.inference_mode():h=model(torch.tensor(x[None],dtype=torch.float32),output_hidden_states=True).hidden_states[6][0]
            v=torch.cat([h.mean(0),h.std(0,unbiased=False)]).numpy();np.savez_compressed(dest,x=v)
        rows.append(np.load(dest)['x'])
        if (i+1)%100==0:print('Embeddings',i+1,len(frame),flush=True)
    np.savez_compressed(OUT/'embeddings.npz',x=np.stack(rows));frame[['record_id','speaker_id']].to_csv(OUT/'embedding_order.csv',index=False)
    dump(OUT/'wavlm_provenance.json',specification)

if __name__=='__main__':main()
