"""Audit all cached fitting partitions and verify saved feature/inference parity."""
import json
import joblib
import numpy as np
import pandas as pd
import svd_v4 as v4
from svd_v4_results import features,embedding_model

def main():
    out=v4.OUT;p=json.loads((out/'protocol_v4.json').read_text())
    frame=pd.read_csv(out/'development_private.csv');x=np.load(out/'features.npz')
    assignment=pd.read_csv(out/'outer_assignment.csv');count=0
    def verify(ids,c,seed,forbidden):
        nonlocal count
        key=v4.v3.key([p['source_hashes'],p['features_hash'],ids.tolist(),c,seed])
        m=joblib.load(out/'fit_cache'/(key+'.joblib'));a=m['partition_audit']
        fit=set(a['fitting']);cal=set(a['calibration']);allowed=set(frame.iloc[ids].speaker_id)
        assert not fit&cal and fit|cal<=allowed and not (fit|cal)&set(forbidden)
        if 'stopping_train' in a:
            tr=set(a['stopping_train']);va=set(a['stopping_validation'])
            assert not tr&va and tr|va<=fit
        count+=1
    for fold in range(5):
        ids=np.arange(len(frame)) if fold==4 else np.flatnonzero(assignment.outer_fold.to_numpy()!=fold)
        outerval=[] if fold==4 else frame[assignment.outer_fold==fold].speaker_id.tolist()
        seed=v4.SEED+2000 if fold==4 else v4.SEED+100*fold
        part=frame.iloc[ids].reset_index(drop=True)
        for c in v4.configs():
            for k,(tr,va) in enumerate(v4.SPLITS(part,3,seed)):
                verify(ids[tr],c,seed+k,outerval+part.iloc[va].speaker_id.tolist())
    # Verify acoustic and frozen embedding extraction against cached training input.
    enc=embedding_model();row=frame.iloc[0]
    xx,mm,_,_=features(v4.v3.audio_bytes(row.path),enc)
    np.testing.assert_allclose(xx,x['x'][0],rtol=1e-5,atol=1e-5,equal_nan=True)
    np.testing.assert_allclose(mm,x['mel'][0],rtol=1e-6,atol=1e-6)
    result={'inner_fit_partitions_audited':count,'speaker_overlap':0,
      'training_inference_feature_parity':'passed (acoustics and WavLM, one fixed development recording)',
      'scope':'Partition and numerical checks; does not establish clinical validity.'}
    v4.v3.dump(out/'acceptance.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
