import numpy as np
from sklearn.metrics import roc_auc_score

def metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    pred = (scores >= 0).astype(int)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        'balanced_accuracy': (sensitivity + specificity)/2 if sensitivity is not None and specificity is not None else None,
        'macro_f1': .5 * (2*tp/max(1,2*tp+fp+fn) + 2*tn/max(1,2*tn+fp+fn)),
        'sensitivity': sensitivity, 'specificity': specificity,
        'roc_auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        'accuracy': float(np.mean(y == pred)), 'n': len(y),
        'confusion_matrix': [[tn, fp], [fn, tp]],
    }

def confidence_intervals(y, scores_by_model, repeats=2000, seed=2026):
    """Paired participant bootstrap. One retained recording per participant."""
    rng = np.random.default_rng(seed)
    values = {name: {k: [] for k in ['balanced_accuracy','macro_f1','sensitivity','specificity','roc_auc','accuracy']}
              for name in scores_by_model}
    differences = []
    for _ in range(repeats):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        sample = {}
        for name, scores in scores_by_model.items():
            sample[name] = metrics(y[idx], scores[idx])
            for key in values[name]:
                values[name][key].append(sample[name][key])
        if 'DNN' in sample and 'GMM' in sample:
            differences.append(sample['DNN']['balanced_accuracy'] - sample['GMM']['balanced_accuracy'])
    def interval(vals):
        return [float(v) for v in np.percentile(vals, [2.5,97.5])]
    return {'method': 'paired participant percentile bootstrap; conditional on fitted models; not training uncertainty',
            'requested_resamples': repeats, 'accepted_resamples': len(next(iter(values.values()))['accuracy']),
            'models': {n: {k: interval(v) for k,v in d.items()} for n,d in values.items()},
            'DNN_minus_GMM_balanced_accuracy': interval(differences) if differences else None}
