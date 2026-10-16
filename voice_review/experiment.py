"""Reproducible development experiment; final evaluation is an explicit one-time action."""
from pathlib import Path
import hashlib
import json
import platform
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from .data import ROOT, SEED, audit, make_splits, features_for, save_json, assert_partition
from .features import SPEC, QUALITY_NAMES, extract_path
from .models import fit_model, score_model
from .metrics import metrics, confidence_intervals

def prepare(root=ROOT):
    if (root / 'results/final_evaluation.lock').exists():
        raise RuntimeError('Final test has been opened; preserve this experiment unchanged.')
    frame = audit(root)
    splits = make_splits(frame, root)
    save_json(root / 'data/feature_spec.json', SPEC)
    ids = splits['development']
    seq, q, logs = features_for(ids, frame, root)
    save_json(root / 'results/feature_quality_development.json', logs)
    print(f'Audited {len(frame)} records; {len(ids)} development, {len(splits["test"])} sealed test. Features: {seq.shape}.', flush=True)
    if not np.isfinite(q[:,0]).any():
        raise RuntimeError('CPPS failed for every recording; inspect feature extraction before continuing.')
    pd.DataFrame(q, columns=QUALITY_NAMES, index=ids).rename_axis('record_id').to_csv(root / 'results/quality_features_development.csv')

def config_id(config):
    setting = config.get('components', config.get('dropout'))
    return f'{config["family"]}_{config["features"]}_{setting}'

def train(root=ROOT):
    if (root / 'results/final_evaluation.lock').exists():
        raise RuntimeError('Test evaluation already started. No further tuning permitted in this experiment.')
    frame = pd.read_csv(root / 'data/manifest.csv')
    splits = json.loads((root / 'data/splits.json').read_text())
    assert_partition(frame, splits['development'], splits['test'])
    ids = splits['development']
    labels = frame.set_index('record_id').loc[ids, 'label'].to_numpy()
    seq, q, _ = features_for(ids, frame, root)
    index = {rid: i for i, rid in enumerate(ids)}
    modeldir = root / 'models'
    modeldir.mkdir(exist_ok=True)
    cvdir = modeldir / 'validation'
    cvdir.mkdir(exist_ok=True)
    configs = []
    for feature_set in ['mfcc_only', 'combined']:
        configs += [dict(family='GMM', features=feature_set, components=n) for n in [1,2,4]]
        configs += [dict(family='DNN', features=feature_set, dropout=d) for d in [.2,.4]]
    results, oof, records = [], {}, []
    for config in configs:
        cid = config_id(config)
        scores = np.full(len(ids), np.nan)
        fold_scores, epochs = [], []
        for fold, part in enumerate(splits['folds']):
            assert_partition(frame, part['train'], part['validation'])
            train_idx = np.array([index[r] for r in part['train']])
            valid_idx = np.array([index[r] for r in part['validation']])
            bundle = fit_model(seq[train_idx], q[train_idx], labels[train_idx], config, seed=SEED + fold)
            score = score_model(bundle, seq[valid_idx], q[valid_idx])
            scores[valid_idx] = score
            result = metrics(labels[valid_idx], score)
            fold_scores.append(result['balanced_accuracy'])
            if 'epochs' in bundle:
                epochs.append(bundle['epochs'])
            records.append({'config_id': cid, 'fold': fold, 'config': config, 'metrics': result,
                            'epochs': bundle.get('epochs')})
            joblib.dump(bundle, cvdir / f'{cid}_fold{fold}.joblib')
            print(f'{cid} fold {fold+1}/3: balanced accuracy {result["balanced_accuracy"]:.3f}', flush=True)
        if not np.isfinite(scores).all():
            raise AssertionError('Every development participant must have exactly one validation prediction.')
        oof[cid] = scores
        results.append({'config_id': cid, **config, 'mean_fold_balanced_accuracy': float(np.mean(fold_scores)),
                        'std_fold_balanced_accuracy': float(np.std(fold_scores)),
                        'refit_epochs': int(np.median(epochs)) if epochs else None,
                        'pooled_oof_metrics': metrics(labels, scores)})
    selected = {}
    predictions = pd.DataFrame({'record_id': ids, 'label': labels})
    for family in ['GMM', 'DNN']:
        options = [r for r in results if r['family'] == family]
        # Fewer features, fewer GMM components, then stronger DNN regularization break exact ties.
        options.sort(key=lambda r: (-r['mean_fold_balanced_accuracy'], r['features']=='combined',
                                    r.get('components', 0), -r.get('dropout',0)))
        best = options[0]
        config = next(c for c in configs if config_id(c) == best['config_id'])
        selected[family] = best
        bundle = fit_model(seq, q, labels, config, seed=SEED, fixed_epochs=best['refit_epochs'])
        joblib.dump(bundle, modeldir / f'{family.lower()}.joblib')
        predictions[family + '_score'] = oof[best['config_id']]
        predictions[family + '_prediction'] = (oof[best['config_id']] >= 0).astype(int)
        # Verify serialization reproduces values, using development inputs only.
        before = score_model(bundle, seq[:5], q[:5])
        after = score_model(joblib.load(modeldir / f'{family.lower()}.joblib'), seq[:5], q[:5])
        np.testing.assert_allclose(before, after, atol=1e-7)
    predictions.to_csv(root / 'results/development_predictions.csv', index=False)
    save_json(root / 'results/search.json', {'candidates': results, 'folds': records})
    save_json(root / 'results/selection.json', selected)
    save_json(root / 'results/development_metrics.json', {
        'status': 'PRELIMINARY DEVELOPMENT VALIDATION; hyperparameters selected on these folds; not an unbiased final estimate',
        'models': {m: metrics(labels, predictions[m+'_score'].to_numpy()) for m in selected},
        'majority_baseline': metrics(labels, np.ones(len(labels))),
        'test_status': 'sealed', 'development_speakers': len(ids), 'sealed_test_speakers': len(splits['test'])})
    save_json(root / 'results/development_intervals.json', confidence_intervals(labels,
              {m: predictions[m+'_score'].to_numpy() for m in selected}))
    (root / 'requirements-lock.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    save_json(root / 'results/runtime.json', {'python': sys.version, 'platform': platform.platform(), 'seed': SEED})
    explain(root)
    freeze(root)
    print('Development experiment complete. Final test remains sealed.', flush=True)

def explain(root=ROOT):
    frame = pd.read_csv(root / 'data/manifest.csv')
    splits = json.loads((root / 'data/splits.json').read_text())
    selected = json.loads((root / 'results/selection.json').read_text())
    importance = []
    # Permute entire feature groups between validation speakers, preserving each
    # sequence's time structure and each quality group's within-speaker correlations.
    for family, chosen in selected.items():
        for fold, part in enumerate(splits['folds']):
            seq, q, _ = features_for(part['validation'], frame, root)
            y = frame.set_index('record_id').loc[part['validation'], 'label'].to_numpy()
            bundle = joblib.load(root / f'models/validation/{chosen["config_id"]}_fold{fold}.joblib')
            baseline = metrics(y, score_model(bundle, seq, q))['balanced_accuracy']
            for group in ['MFCC', 'Voice quality']:
                if group == 'Voice quality' and chosen['features'] != 'combined':
                    continue
                drops = []
                rng = np.random.default_rng(SEED + fold)
                for _ in range(10):
                    order = rng.permutation(len(y))
                    score = score_model(bundle, seq[order] if group=='MFCC' else seq,
                                        q[order] if group=='Voice quality' else q)
                    drops.append(baseline - metrics(y, score)['balanced_accuracy'])
                importance.append({'model': family, 'fold': fold, 'feature_group': group,
                                   'mean_balanced_accuracy_drop': float(np.mean(drops)),
                                   'std_drop': float(np.std(drops))})
    pd.DataFrame(importance).to_csv(root/'results/permutation_importance.csv', index=False)
    pred = pd.read_csv(root/'results/development_predictions.csv').merge(frame[['record_id','sex','age','diagnosis']], on='record_id')
    pred['age'] = pd.to_numeric(pred.age, errors='coerce')
    pred['age_group'] = pd.cut(pred.age, [-np.inf,39,59,np.inf], labels=['under40','40to59','60plus']).astype(str)
    subgroups = []
    for family in ['GMM','DNN']:
        errors = pred[pred[family+'_prediction'] != pred.label]
        errors.to_csv(root/f'results/{family.lower()}_development_errors.csv', index=False)
        for column in ['sex','age_group']:
            for name, group in pred.groupby(column):
                subgroups.append({'model': family, 'variable': column, 'group': str(name),
                                  'healthy': int((group.label==0).sum()), 'pathological': int((group.label==1).sum()),
                                  'metrics': metrics(group.label.to_numpy(), group[family+'_score'].to_numpy()),
                                  'interpretation': 'descriptive only; small subgroups, not a fairness certification'})
    save_json(root/'results/subgroups.json', subgroups)

def evaluate_final(root=ROOT, confirm=False):
    if not confirm:
        raise ValueError('Explicit --confirm-final is required to open the sealed test set.')
    marker = root / 'results/final_evaluation.lock'
    if marker.exists():
        raise RuntimeError('Final evaluation was already started; refusing a repeated test evaluation.')
    lock = json.loads((root/'models/frozen_experiment.json').read_text())
    for name, expected in lock['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen artifact changed: {name}')
    frame = pd.read_csv(root/'data/manifest.csv')
    splits = json.loads((root/'data/splits.json').read_text())
    assert_partition(frame, splits['development'], splits['test'])
    # Atomic marker precedes reading test features. A failure keeps the marker for audit.
    with marker.open('x', encoding='utf-8') as out:
        out.write('Test opened; model selection must not continue.\n')
    ids = splits['test']
    seq, q, logs = features_for(ids, frame, root)
    y = frame.set_index('record_id').loc[ids, 'label'].to_numpy()
    predictions = pd.DataFrame({'record_id':ids, 'label':y})
    all_scores = {}
    for family in ['GMM','DNN']:
        bundle = joblib.load(root/f'models/{family.lower()}.joblib')
        all_scores[family] = score_model(bundle, seq, q)
        predictions[family+'_score'] = all_scores[family]
        predictions[family+'_prediction'] = (all_scores[family]>=0).astype(int)
    predictions.to_csv(root/'results/final_predictions.csv', index=False)
    annotated = predictions.merge(frame[['record_id','sex','age','diagnosis']],on='record_id')
    for family in ['GMM','DNN']:
        annotated[annotated[family+'_prediction'] != annotated.label].to_csv(root/f'results/{family.lower()}_final_errors.csv',index=False)
    save_json(root/'results/final_metrics.json', {'status':'FINAL HELD-OUT TEST', 'n':len(y),
              'models':{m:metrics(y,s) for m,s in all_scores.items()},
              'intervals':confidence_intervals(y, all_scores), 'quality':logs})
    print('Final evaluation completed and saved. Do not tune against these results.', flush=True)

def inspect_recording(path, root=ROOT, validation=False):
    """Return the prediction and exact model inputs for the local demonstration."""
    # Prevent using a released dataset test file as a demo before final evaluation.
    from .features import audio_fingerprint, read_audio, preprocess, extract
    audio, sr = read_audio(path)
    x = preprocess(audio, sr)
    manifest_path = root/'data/manifest.csv'
    split_path = root/'data/splits.json'
    fingerprint = audio_fingerprint(x)
    model_paths = {family: root/f'models/{family.lower()}.joblib' for family in ['GMM','DNN']}
    context = {'mode': 'full-development refit', 'record_id': None,
               'notice': 'New or unrecognized input; source generalization is unvalidated.'}
    if manifest_path.exists() and split_path.exists():
        manifest = pd.read_csv(manifest_path)
        splits = json.loads(split_path.read_text())
        matching = manifest[manifest.audio_hash == fingerprint]
        if len(matching) == 1:
            rid = matching.iloc[0].record_id
            context['record_id'] = rid
            if rid in splits['development']:
                context['notice'] = 'Known development recording included in full-model training; demonstration only.'
        if validation:
            if len(matching) != 1 or matching.iloc[0].record_id not in splits['development']:
                raise ValueError('Speaker-held-out demo requires an unchanged bundled development recording. Use full-development models for a new external recording.')
            rid = matching.iloc[0].record_id
            folds = [(i, p) for i, p in enumerate(splits['folds']) if rid in p['validation']]
            if len(folds) != 1:
                raise ValueError('A unique held-out validation fold could not be verified.')
            fold, part = folds[0]
            assert_partition(manifest, part['train'], part['validation'])
            selected = json.loads((root/'results/selection.json').read_text())
            model_paths = {family: root/f'models/validation/{c["config_id"]}_fold{fold}.joblib'
                           for family,c in selected.items()}
            context = {'mode': 'speaker-held-out development validation', 'record_id': rid,
                       'fold': fold, 'training_records': len(part['train']),
                       'notice': 'This speaker was excluded from these models\' fitting and early stopping. Validation still informed configuration selection; not an independent final test.'}
    elif validation:
        raise ValueError('Verified development splits are required for a speaker-held-out demo.')
    if manifest_path.exists() and split_path.exists() and not (root/'results/final_evaluation.lock').exists():
        manifest = pd.read_csv(manifest_path)
        splits = json.loads(split_path.read_text())
        sealed_hashes = set(manifest[manifest.record_id.isin(splits['test'])].audio_hash)
        if fingerprint in sealed_hashes:
            raise ValueError('This recording belongs to the sealed test set. Choose a development example.')
    seq, q, status = extract(x)
    results = {}
    for family in ['GMM','DNN']:
        bundle = joblib.load(model_paths[family])
        score = float(score_model(bundle, seq[None], q[None])[0])
        results[family] = {'prediction':'pathological' if score>=0 else 'healthy',
                           'raw_score':score, 'score_type':'log likelihood ratio' if family=='GMM' else 'logit',
                           'uncalibrated_model_score':float(expit(score))}
    result = {'quality':status, 'models':results, 'model_context': context,
              'notice':'Academic research only. Scores are uncalibrated model outputs, not medical probabilities.'}
    return result, x, seq, q


def predict(path, root=ROOT, validation=False):
    return inspect_recording(path, root, validation=validation)[0]

def freeze(root=ROOT):
    if (root/'results/final_evaluation.lock').exists():
        raise RuntimeError('Cannot refreeze after final test evaluation has started.')
    selected = json.loads((root/'results/selection.json').read_text())
    files = [root/'data/splits.json', root/'data/manifest.csv', root/'data/feature_spec.json',
             root/'models/gmm.joblib', root/'models/dnn.joblib', root/'results/selection.json',
             root/'requirements-lock.txt'] + sorted((root/'voice_review').glob('*.py'))
    files += [root/f'models/validation/{c["config_id"]}_fold{fold}.joblib'
              for c in selected.values() for fold in range(3)]
    lock = {'selection':selected, 'files':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    save_json(root/'models/frozen_experiment.json',lock)
