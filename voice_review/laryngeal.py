"""Exploratory label-based three-class extension; preserves the binary experiment."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from .data import ROOT, SEED, assert_partition, features_for, save_json
from .models import Transforms, VoiceCNN, seed_all

NAMES = ['Healthy', 'Laryngeal (laryngitis-labelled)', 'Other pathology']


def label_for(diagnosis):
    text = diagnosis.strip().lower()
    return 0 if text in {'healthy','normal','healthy voice','normal voice'} else 1 if 'laryngitis' in text else 2


def network(dropout=.4):
    model = VoiceCNN(dropout, 12)
    model.head[-1] = nn.Linear(16, 3)
    return model


def measure(y, scores):
    pred = np.asarray(scores).argmax(axis=1)
    return {'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0,1,2]).tolist(),
            'report': classification_report(y, pred, labels=[0,1,2], target_names=NAMES,
                                            output_dict=True, zero_division=0)}


def _cnn(seq, quality, y, seed, epochs=100, stop=None):
    seed_all(seed)
    transform = Transforms(True).fit(seq, quality)
    s,q,_ = transform.transform(seq, quality)
    sx,qx,target = torch.from_numpy(s),torch.from_numpy(q),torch.tensor(y,dtype=torch.long)
    model = network()
    counts = np.bincount(y, minlength=3)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(len(y)/(3*counts),dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    stopping = None
    if stop is not None:
        ss,qq,_ = transform.transform(stop[0],stop[1])
        stopping = (torch.from_numpy(ss),torch.from_numpy(qq),torch.tensor(stop[2],dtype=torch.long))
    best, best_epoch, stale = float('inf'),1,0
    for epoch in range(1,epochs+1):
        model.train()
        for idx in torch.randperm(len(y)).split(16):
            optimizer.zero_grad()
            loss = criterion(model(sx[idx],qx[idx]),target[idx])
            loss.backward()
            optimizer.step()
        if stopping is not None:
            model.eval()
            with torch.no_grad():
                loss = float(criterion(model(stopping[0],stopping[1]),stopping[2]))
            if loss < best-1e-4:
                best,best_epoch,stale = loss,epoch,0
            else:
                stale += 1
            if stale >= 12:
                break
    return {'family':'DNN','task':'laryngitis_three_class','transform':transform,
            'weights':{k:v.detach().numpy() for k,v in model.state_dict().items()},
            'epochs':epochs if stop is None else best_epoch,'seed':seed,'labels':NAMES}


def fit(seq, quality, y, family, seed=SEED, epochs=None):
    if set(np.asarray(y).tolist()) != {0,1,2}:
        raise ValueError('Every training partition must contain all three classes.')
    if family == 'GMM':
        transform = Transforms(True).fit(seq,quality)
        tab = transform.transform(seq,quality)[2]
        models = []
        for label in range(3):
            model = GaussianMixture(n_components=1,covariance_type='diag',reg_covar=.01,
                                    n_init=5,max_iter=300,random_state=seed).fit(tab[y==label])
            if not model.converged_:
                raise RuntimeError('Three-class GMM did not converge.')
            models.append(model)
        return {'family':'GMM','task':'laryngitis_three_class','transform':transform,
                'models':models,'seed':seed,'labels':NAMES}
    if epochs is None:
        tr,va = train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed)
        epochs = _cnn(seq[tr],quality[tr],y[tr],seed,stop=(seq[va],quality[va],y[va]))['epochs']
    return _cnn(seq,quality,y,seed,epochs=int(epochs))


def scores(bundle, seq, quality):
    s,q,tab = bundle['transform'].transform(seq,quality)
    if bundle['family']=='GMM':
        return np.column_stack([m.score_samples(tab) for m in bundle['models']])
    model = network()
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in bundle['weights'].items()})
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(s),torch.from_numpy(q)).numpy()


def run(root=ROOT):
    if (root/'results/final_evaluation.lock').exists():
        raise RuntimeError('Final test has been opened; development is closed.')
    out,modeldir = root/'results/laryngeal',root/'models/laryngeal'
    out.mkdir(exist_ok=True)
    modeldir.mkdir(exist_ok=True)
    if (out/'metrics.json').exists():
        raise RuntimeError('This extension already ran. Preserve its outputs before a separately identified revision.')
    frame = pd.read_csv(root/'data/manifest.csv')
    splits = json.loads((root/'data/splits.json').read_text())
    frame['subtype'] = frame.diagnosis.map(label_for)
    ids = splits['development']
    assert_partition(frame,ids,splits['test'])
    dev = frame.set_index('record_id').loc[ids]
    if dev.speaker_id.duplicated().any():
        raise ValueError('One retained recording per participant is required for internal stopping.')
    audit = frame.groupby('diagnosis').agg(speaker_proxies=('speaker_id','nunique'),records=('record_id','size')).reset_index()
    audit['development'] = audit.diagnosis.map(dev.groupby('diagnosis').speaker_id.nunique()).fillna(0).astype(int)
    audit['reserved_test'] = audit.speaker_proxies-audit.development
    audit['group'] = audit.diagnosis.map(lambda d:NAMES[label_for(d)])
    audit.to_csv(out/'diagnosis_audit.csv',index=False)
    fold_counts = []
    for fold,part in enumerate(splits['folds']):
        assert_partition(frame,part['train'],part['validation'])
        counts = {side:frame[frame.record_id.isin(part[side])].subtype.value_counts().reindex([0,1,2],fill_value=0).tolist()
                  for side in ['train','validation']}
        fold_counts.append({'fold':fold,**counts})
    if dev.subtype.value_counts().min()<20 or any(min(c['train'])<12 or min(c['validation'])<5 for c in fold_counts):
        raise ValueError('Insufficient development support for this exploratory three-class protocol.')
    original = [root/'models/gmm.joblib',root/'models/dnn.joblib',root/'data/splits.json']
    signatures = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in original}
    save_json(out/'protocol.json',{'labels':NAMES,'mapping':'Exact healthy labels; any diagnosis containing laryngitis; all other pathology.',
        'limitations':'Laryngeal here means laryngitis-labelled only. Other pathology can also involve the larynx. Counts use record IDs as participant proxies.',
        'development_counts':dev.subtype.value_counts().sort_index().tolist(),'fold_counts':fold_counts,
        'settings':'GMM 1 diagonal component/class; CNN 16/32 conv filters, 16-unit dense, 3 logits, dropout .4. Fixed before this run; no subtype tuning.',
        'early_stopping':'20% stratified subset inside training fold; weighted cross-entropy, patience 12; fresh full-fold refit.',
        'seed':SEED,'original_files':signatures})
    seq,q,_ = features_for(ids,frame,root)
    y = dev.subtype.to_numpy()
    index = {rid:i for i,rid in enumerate(ids)}
    predictions = pd.DataFrame({'record_id':ids,'label':y})
    results,fold_results = {},[]
    for family in ['GMM','DNN']:
        oof = np.full((len(ids),3),np.nan)
        epochs = []
        for fold,part in enumerate(splits['folds']):
            tr,va = [np.array([index[r] for r in part[side]]) for side in ['train','validation']]
            bundle = fit(seq[tr],q[tr],y[tr],family,SEED+fold)
            oof[va] = scores(bundle,seq[va],q[va])
            bundle['training_ids'],bundle['validation_ids'] = part['train'],part['validation']
            joblib.dump(bundle,modeldir/f'{family.lower()}_fold{fold}.joblib')
            if 'epochs' in bundle:
                epochs.append(bundle['epochs'])
            measured = measure(y[va],oof[va])
            fold_results.append({'family':family,'fold':fold,'epochs':bundle.get('epochs'),'metrics':measured})
            print(f'{family} subtype fold {fold+1}: balanced accuracy {measured["balanced_accuracy"]:.3f}',flush=True)
        if not np.isfinite(oof).all():
            raise AssertionError('Every development record must have a prediction.')
        results[family] = measure(y,oof)
        predictions[family+'_prediction'] = oof.argmax(axis=1)
        for c in range(3):
            predictions[f'{family}_score_{c}'] = oof[:,c]
        bundle = fit(seq,q,y,family,SEED,epochs=int(np.median(epochs)) if epochs else None)
        bundle['training_ids'] = ids
        joblib.dump(bundle,modeldir/f'{family.lower()}.joblib')
        np.testing.assert_allclose(scores(bundle,seq[:3],q[:3]),
            scores(joblib.load(modeldir/f'{family.lower()}.joblib'),seq[:3],q[:3]),atol=1e-7)
    predictions.to_csv(out/'predictions.csv',index=False)
    save_json(out/'fold_metrics.json',fold_results)
    save_json(out/'metrics.json',{'status':'EXPLORATORY DEVELOPMENT VALIDATION; no final test opened',
                               'labels':NAMES,'models':results,'n':len(y)})
    for name,digest in signatures.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    paths = list(modeldir.glob('*.joblib'))+[root/'voice_review/laryngeal.py',root/'voice_review/features.py',root/'voice_review/models.py',root/'data/splits.json',out/'protocol.json']
    save_json(modeldir/'provenance.json',{'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})


if __name__=='__main__':
    run()
