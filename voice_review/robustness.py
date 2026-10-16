"""Fixed development-only seed sensitivity; never selects or replaces a model."""
import hashlib
import json
import numpy as np
import pandas as pd
from .data import ROOT, SEED, save_json, features_for, assert_partition
from .models import fit_model, score_model
from .metrics import metrics

REPEAT_SEEDS = (12026, 22026, 32026)


def summarize_runs(predictions):
    summaries = {}
    runs = []
    for (family, seed), part in predictions.groupby(['model', 'seed']):
        runs.append({'model': family, 'seed': int(seed),
                     **metrics(part.label.to_numpy(), part.score.to_numpy())})
    for family, part in predictions.groupby('model'):
        values = [r['balanced_accuracy'] for r in runs if r['model'] == family]
        votes = part.pivot(index='record_id', columns='seed', values='prediction')
        if votes.isna().any().any():
            raise ValueError('Each seed must predict the same development records.')
        unstable = votes.nunique(axis=1) > 1
        summaries[family] = {
            'runs': len(values), 'balanced_accuracy_mean': float(np.mean(values)),
            'balanced_accuracy_sd': float(np.std(values, ddof=1)),
            'balanced_accuracy_min': float(np.min(values)),
            'balanced_accuracy_max': float(np.max(values)),
            'records_with_label_changes': int(unstable.sum()),
            'records': len(votes),
        }
    return summaries, runs


def run_robustness(root=ROOT):
    if (root/'results/final_evaluation.lock').exists():
        raise RuntimeError('Development analysis is closed after final-test opening.')
    out = root/'results/robustness'
    out.mkdir(exist_ok=True)
    paths = [root/'data/splits.json', root/'results/selection.json',
             root/'results/development_predictions.csv', root/'models/gmm.joblib',
             root/'models/dnn.joblib', root/'voice_review/models.py',
             root/'voice_review/robustness.py']
    signature = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    protocol = {
        'purpose': 'Training randomness sensitivity, not additional model selection',
        'original_seed': SEED, 'repeat_seeds': list(REPEAT_SEEDS),
        'fold_seed_rule': 'base seed + zero-based fold index',
        'fixed': 'Original development folds, selected architecture/features, threshold zero',
        'varied': 'Initialization, minibatch order and internal early-stopping subset; not outer partitions',
        'reporting': 'Show every run. Do not choose the best seed or change saved demonstration models.',
        'interpretation': 'Four seed runs are descriptive; standard deviations are not confidence intervals. Selection bias remains.',
        'files': signature,
    }
    protocol_path = out/'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError('Robustness inputs changed; preserve the existing analysis before revising it.')
    else:
        save_json(protocol_path, protocol)  # Written before any additional fits.
    if (out/'summary.json').exists():
        print('Seed analysis already complete; retained all original outputs.', flush=True)
        return
    frame = pd.read_csv(root/'data/manifest.csv')
    splits = json.loads((root/'data/splits.json').read_text())
    selected = json.loads((root/'results/selection.json').read_text())
    ids = splits['development']
    assert_partition(frame, ids, splits['test'])
    if frame.set_index('record_id').loc[ids].speaker_id.duplicated().any():
        raise ValueError('Internal stopping assumes one retained recording per participant.')
    seq, q, _ = features_for(ids, frame, root)
    y = frame.set_index('record_id').loc[ids, 'label'].to_numpy()
    index = {rid: i for i, rid in enumerate(ids)}
    original = pd.read_csv(root/'results/development_predictions.csv').set_index('record_id').loc[ids]
    rows, fold_rows = [], []
    for family, chosen in selected.items():
        config = {k: chosen[k] for k in ['family', 'features', 'components', 'dropout'] if k in chosen}
        for seed in (SEED,) + REPEAT_SEEDS:
            if seed == SEED:
                scores = original[family+'_score'].to_numpy()
            else:
                scores = np.full(len(ids), np.nan)
                for fold, part in enumerate(splits['folds']):
                    assert_partition(frame, part['train'], part['validation'])
                    if set(part['train'] + part['validation']) != set(ids):
                        raise ValueError('Fold includes non-development records or omits records.')
                    tr = np.array([index[r] for r in part['train']])
                    va = np.array([index[r] for r in part['validation']])
                    model = fit_model(seq[tr], q[tr], y[tr], config, seed=seed+fold)
                    scores[va] = score_model(model, seq[va], q[va])
                    measured = metrics(y[va], scores[va])
                    fold_rows.append({'model': family, 'seed': seed, 'fold': fold,
                                      'epochs': model.get('epochs'), 'metrics': measured})
                    print(f'{family} seed {seed}, fold {fold+1}: BA {measured["balanced_accuracy"]:.3f}', flush=True)
            if not np.isfinite(scores).all():
                raise ValueError('Incomplete seed predictions.')
            rows.extend({'record_id': rid, 'label': int(label), 'model': family, 'seed': seed,
                         'score': float(score), 'prediction': int(score >= 0)}
                        for rid, label, score in zip(ids, y, scores))
    predictions = pd.DataFrame(rows)
    summary, runs = summarize_runs(predictions)
    predictions.to_csv(out/'predictions.csv', index=False)
    save_json(out/'folds.json', fold_rows)
    save_json(out/'summary.json', {'status': 'DEVELOPMENT ONLY; no seed selection',
                                  'summary': summary, 'runs': runs,
                                  'limitations': protocol['interpretation']})
    # Confirm analysis did not replace any original model, configuration or predictions.
    for name, expected in signature.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f'Original artifact changed during robustness analysis: {name}')
    print('Seed sensitivity completed; original models retained and final test sealed.', flush=True)
