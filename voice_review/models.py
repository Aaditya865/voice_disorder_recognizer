"""Fold-local transforms, class-conditional GMMs, and a compact temporal CNN."""
import copy
import random
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

class Transforms:
    def __init__(self, combined=True):
        self.combined = combined

    def fit(self, seq, quality):
        self.seq_mean = seq.mean(axis=(0, 2), keepdims=True)
        self.seq_std = np.maximum(seq.std(axis=(0, 2), keepdims=True), 1e-6)
        self.medians = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.
                                 for c in quality.T])
        filled = np.where(np.isfinite(quality), quality, self.medians)
        self.q_scaler = StandardScaler().fit(filled)
        self.tab_scaler = StandardScaler().fit(self._tab_raw(seq, quality))
        return self

    def q(self, quality):
        if not self.combined:
            return np.zeros((len(quality), 0), dtype=np.float32)
        missing = ~np.isfinite(quality)
        filled = np.where(missing, self.medians, quality)
        return np.concatenate([self.q_scaler.transform(filled), missing.astype(float)], axis=1).astype(np.float32)

    def _tab_raw(self, seq, quality):
        return np.concatenate([seq.mean(axis=2), seq.std(axis=2), self.q(quality)], axis=1)

    def transform(self, seq, quality):
        sequence = ((seq - self.seq_mean) / self.seq_std).astype(np.float32)
        tab = self.tab_scaler.transform(self._tab_raw(seq, quality))
        q = self.q(quality)
        if not all(np.isfinite(x).all() for x in [sequence, q, tab]):
            raise ValueError('Model inputs must be finite.')
        return sequence, q, tab

class VoiceCNN(nn.Module):
    def __init__(self, dropout=.2, quality_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(13, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(32 + quality_dim, 16), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, sequence, quality):
        return self.head(torch.cat([self.encoder(sequence).squeeze(-1), quality], dim=1)).squeeze(-1)

def _train_cnn(seq, quality, y, config, seed, stop=None, epochs=100):
    seed_all(seed)
    transform = Transforms(config['features'] == 'combined').fit(seq, quality)
    s, q, _ = transform.transform(seq, quality)
    sx, qx = torch.from_numpy(s), torch.from_numpy(q)
    target = torch.tensor(y, dtype=torch.float32)
    model = VoiceCNN(config['dropout'], q.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    # Derived only from this fit's training labels.
    weight = float(np.sum(y == 0) / np.sum(y == 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    stop_data = None
    if stop is not None:
        ss, qq, _ = transform.transform(stop[0], stop[1])
        stop_data = (torch.from_numpy(ss), torch.from_numpy(qq), torch.tensor(stop[2], dtype=torch.float32))
    best_loss, best_epoch, stale = float('inf'), 1, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for idx in torch.randperm(len(y)).split(16):
            optimizer.zero_grad()
            loss = criterion(model(sx[idx], qx[idx]), target[idx])
            loss.backward()
            optimizer.step()
        if stop_data is not None:
            model.eval()
            with torch.no_grad():
                loss = float(criterion(model(stop_data[0], stop_data[1]), stop_data[2]))
            if loss < best_loss - 1e-4:
                best_loss, best_epoch, stale = loss, epoch, 0
            else:
                stale += 1
            if stale >= 12:
                break
    model.eval()
    return {'family': 'DNN', 'config': config, 'transform': transform,
            'weights': {k: v.detach().numpy() for k, v in model.state_dict().items()},
            'epochs': epochs if stop is None else best_epoch, 'seed': seed}

def fit_model(seq, quality, y, config, seed=2026, fixed_epochs=None):
    y = np.asarray(y)
    if set(y.tolist()) != {0, 1}:
        raise ValueError('Both classes are required in every training partition.')
    if config['family'] == 'GMM':
        t = Transforms(config['features'] == 'combined').fit(seq, quality)
        tab = t.transform(seq, quality)[2]
        models = []
        for label in [0, 1]:
            model = GaussianMixture(n_components=config['components'], covariance_type='diag',
                                    reg_covar=.01, n_init=5, max_iter=300, random_state=seed)
            model.fit(tab[y == label])
            if not model.converged_:
                raise RuntimeError('GMM failed to converge; inspect the fit rather than reporting it.')
            models.append(model)
        return {'family': 'GMM', 'config': config, 'transform': t, 'models': models, 'seed': seed}
    if fixed_epochs is None:
        # Caller supplies one retained record per participant. This subset is inside the
        # training fold; the evaluation fold is never used for early stopping.
        train, stop = train_test_split(np.arange(len(y)), test_size=.2, stratify=y, random_state=seed)
        trial = _train_cnn(seq[train], quality[train], y[train], config, seed,
                           stop=(seq[stop], quality[stop], y[stop]))
        fixed_epochs = trial['epochs']
    # Refit from fresh initialization on all fold-training speakers for selected epochs.
    return _train_cnn(seq, quality, y, config, seed, epochs=int(fixed_epochs))

def score_model(bundle, seq, quality):
    s, q, tab = bundle['transform'].transform(seq, quality)
    if bundle['family'] == 'GMM':
        return bundle['models'][1].score_samples(tab) - bundle['models'][0].score_samples(tab)
    model = VoiceCNN(bundle['config']['dropout'], q.shape[1])
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in bundle['weights'].items()})
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(s), torch.from_numpy(q)).numpy()
    return scores.astype(float)
