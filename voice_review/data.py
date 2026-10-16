from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedGroupKFold
from .features import read_audio, preprocess, audio_fingerprint, extract_path, SPEC, QUALITY_NAMES

SEED = 2026
ROOT = Path(__file__).resolve().parents[1]

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

def audit(root=ROOT):
    raw = root / 'data/raw'
    rows = []
    for header in sorted(raw.glob('voice*.hea')):
        info = (raw / (header.stem + '-info.txt')).read_text(encoding='utf-8', errors='replace')
        def field(key):
            match = re.search(r'^\s*' + re.escape(key) + r':\s*(.*)$', info, re.MULTILINE | re.IGNORECASE)
            return match.group(1).strip() if match else ''
        diagnosis = field('Diagnosis').strip().lower()
        if not diagnosis:
            raise ValueError(f'Missing diagnosis: {header.name}')
        # Explicit healthy labels; preserve every other supplied diagnosis in the manifest.
        healthy = diagnosis in {'healthy', 'normal', 'healthy voice', 'normal voice'}
        header_match = re.search(r'<diagnoses>:\s*(.*?)\s*<medications>', header.read_text(), re.IGNORECASE)
        header_diagnosis = header_match.group(1).strip().lower() if header_match else ''
        if header_diagnosis and (header_diagnosis == 'healthy') != healthy:
            raise ValueError(f'Conflicting binary labels between header and info file: {header.stem}')
        row = dict(record_id=header.stem, speaker_id=header.stem, diagnosis=diagnosis,
                   label=int(not healthy), sex=field('Gender'), age=field('Age'),
                   source=str(header.relative_to(root)), included=True, exclusion_reason='')
        row['source_sha256'] = hashlib.sha256(header.with_suffix('.dat').read_bytes()).hexdigest()
        row['header_sha256'] = hashlib.sha256(header.read_bytes()).hexdigest()
        try:
            x, sr = read_audio(header)
            row.update(sample_rate=sr, seconds=len(x)/sr, channels=x.shape[1])
            central = preprocess(x, sr)
            row['audio_hash'] = audio_fingerprint(central)
        except ValueError as exc:
            row.update(included=False, exclusion_reason=str(exc), audio_hash='')
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError('No dataset found. Run download_data.py first.')
    # Duplicate groups with conflicting labels are excluded; otherwise keep one record.
    for _, group in frame[frame.included].groupby('audio_hash'):
        if len(group) > 1:
            drop = group.index if group.label.nunique() > 1 else group.index[1:]
            frame.loc[drop, 'included'] = False
            frame.loc[drop, 'exclusion_reason'] = 'Duplicate audio; conflicting labels' if group.label.nunique() > 1 else 'Duplicate audio'
    if not (frame.label == 0).any():
        raise ValueError(f'No recognized healthy labels. Audit diagnoses: {sorted(frame.diagnosis.unique())}')
    (root / 'data').mkdir(exist_ok=True)
    frame.to_csv(root / 'data/manifest.csv', index=False)
    save_json(root / 'data/audit.json', {
        'records': len(frame), 'included': int(frame.included.sum()),
        'label_counts_all': {str(k): int(v) for k,v in frame.label.value_counts().items()},
        'diagnoses': {str(k): int(v) for k,v in frame.diagnosis.value_counts().items()},
        'exclusions': frame[~frame.included].fillna('').to_dict('records'),
        'identity_basis': 'PhysioNet describes one acquired vowel recording per subject. Release record IDs are used as participant proxies; no independent cross-session person identifier is provided. Exact/gain-normalized duplicate crops are checked, but identity cannot be independently reverified.',
        'label_mapping': '0 = healthy/normal; 1 = all supplied pathological diagnoses. Original strings retained.',
        'source': 'https://physionet.org/content/voiced/1.0.0/',
        'website_counts': {'healthy': 58, 'pathological': 150},
        'count_discrepancy': 'The release files contain 57 healthy and 151 pathological labels, in agreement between .hea and -info.txt. The website reports 58 and 150. File-level diagnoses are retained; no label is changed to force agreement with the summary.',
    })
    return frame

def make_splits(frame, root=ROOT):
    frame = frame[frame.included].reset_index(drop=True)
    path = root / 'data/splits.json'
    if path.exists():
        splits = json.loads(path.read_text())
        if sorted(splits['development'] + splits['test']) != sorted(frame.record_id.tolist()):
            raise ValueError('Dataset changed; refusing to silently change frozen splits.')
        assert_partition(frame, splits['development'], splits['test'])
        return splits
    if frame.speaker_id.duplicated().any():
        raise ValueError('Expected one retained recording per speaker; grouped holdout design must be revised.')
    tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(frame, frame.label))
    dev = frame.iloc[tr].sort_values('record_id').reset_index(drop=True)
    folds = []
    for train, valid in StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(dev, dev.label, dev.speaker_id):
        folds.append({'train': dev.iloc[train].record_id.tolist(), 'validation': dev.iloc[valid].record_id.tolist()})
    result = {'seed': SEED, 'development': dev.record_id.tolist(), 'test': frame.iloc[te].record_id.tolist(),
              'folds': folds, 'test_status': 'sealed_until_explicit_final_evaluation'}
    assert_partition(frame, result['development'], result['test'])
    for fold in folds:
        assert_partition(frame, fold['train'], fold['validation'])
    save_json(path, result)
    return result

def assert_partition(frame, train_ids, valid_ids):
    if set(train_ids) & set(valid_ids):
        raise AssertionError('Record overlap')
    left = frame[frame.record_id.isin(train_ids)]
    right = frame[frame.record_id.isin(valid_ids)]
    for key in ['speaker_id', 'audio_hash', 'source_sha256']:
        if set(left[key]) & set(right[key]):
            raise AssertionError(f'{key} overlap')

def features_for(ids, frame, root=ROOT):
    cache = root / 'data/features'
    cache.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(SPEC, sort_keys=True).encode()).hexdigest()
    sequences, qualities, logs = [], [], []
    indexed = frame.set_index('record_id')
    for rid in ids:
        row = indexed.loc[rid]
        source = root / row.source
        if hashlib.sha256(source.with_suffix('.dat').read_bytes()).hexdigest() != row.source_sha256:
            raise ValueError(f'Audio source changed since audit: {rid}')
        if hashlib.sha256(source.read_bytes()).hexdigest() != row.header_sha256:
            raise ValueError(f'Audio header changed since audit: {rid}')
        path = cache / (rid + '.npz')
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                if str(z['spec']) != signature or str(z['source_sha256']) != row.source_sha256:
                    raise ValueError('Cached features do not match source/specification.')
                seq, q = z['sequence'], z['quality']
            status = json.loads(path.with_suffix('.json').read_text())
        else:
            seq, q, status, _ = extract_path(root / row.source)
            np.savez_compressed(path, sequence=seq, quality=q, spec=signature, source_sha256=row.source_sha256)
            save_json(path.with_suffix('.json'), status)
        sequences.append(seq)
        qualities.append(q)
        logs.append({'record_id': rid, **status})
    return np.stack(sequences), np.stack(qualities), logs
