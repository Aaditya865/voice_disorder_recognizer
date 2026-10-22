"""SVD inference adapter sharing the trained feature pipeline."""
import hashlib
import io
import json
from pathlib import Path
import threading
import time
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from svd_training import SPEC, FEATURE_NAMES, extract_features, load_audio, probabilities

DESCRIPTIONS = {
 'Healthy': 'The model assigned the dataset’s healthy label. This does not rule out a voice disorder.',
 'Functional Dysphonia': 'Difficulties in how the voice is produced. It may sound strained, weak or inconsistent; these qualities are not unique to this label.',
 'Hyperfunctional Dysphonia': 'A label associated with excessive effort during voice production. The voice may sound tight or strained and overlap with other conditions.',
 'Laryngitis': 'Inflammation of the larynx (voice box). The voice may sound hoarse or rough; audio alone cannot establish the cause.',
 'Vocal Fold Paresis': 'Reduced movement of a vocal fold. The voice may sound weak or breathy; confirmation requires clinical assessment.',
}

class SVDService:
    def __init__(self, root):
        self.directory = Path(root)/'artifacts_svd_local'
        self.lock = threading.RLock()
        self.bundles = {f: joblib.load(self.directory/f'svd_{f.lower()}.joblib') for f in ('GMM','DNN')}
        self.selected = json.loads((self.directory/'selection.json').read_text())['selected_family']
        for bundle in self.bundles.values():
            if bundle['feature_spec'] != SPEC or bundle['feature_names'] != FEATURE_NAMES:
                raise ValueError('Saved feature specification mismatch.')
        self.labels = self.bundles['DNN']['label_encoder'].classes_.tolist()
        if self.bundles['GMM']['label_encoder'].classes_.tolist() != self.labels:
            raise ValueError('Model label encoders disagree.')
        manifest = pd.read_csv(self.directory/'partition_manifest.csv', dtype=str)
        self.known = dict(zip(manifest.audio_hash, manifest.partition))
        self.metrics = pd.read_csv(self.directory/'model_comparison.csv')

    def analyze(self, content, filename):
        start = time.perf_counter()
        if Path(filename).suffix.lower() not in {'.wav','.mp3'}:
            raise ValueError('Please upload a WAV or MP3 file.')
        if not 0 < len(content) <= 20*1024*1024:
            raise ValueError('Choose a nonempty file smaller than 20 MB.')
        try:
            with sf.SoundFile(io.BytesIO(content)) as source:
                if source.format not in {'WAV','WAVEX','RF64','MP3'}:
                    raise ValueError('The contents must be WAV or MP3 audio.')
                if source.channels not in (1,2) or not 8000 <= source.samplerate <= 192000:
                    raise ValueError('Use mono/stereo audio sampled between 8 and 192 kHz.')
                duration = source.frames/source.samplerate
                if duration < SPEC['min_seconds']:
                    raise ValueError('Record at least half a second of a steady aa vowel; 1–3 seconds is recommended.')
                is_mp3 = source.format == 'MP3'
                if is_mp3:
                    count = min(source.frames, int(SPEC['max_seconds']*source.samplerate))
                    source.seek((source.frames-count)//2)
                    decoded = source.read(count, dtype='float64', always_2d=True)
                    converted = io.BytesIO()
                    sf.write(converted, decoded, source.samplerate, format='WAV', subtype='DOUBLE')
                    content = converted.getvalue()
        except (RuntimeError, OSError) as exc:
            raise ValueError('Unable to decode this recording. Export it as WAV and try again.') from exc
        with self.lock:
            features, metadata = extract_features(content)
            audio, _ = load_audio(content)
            fingerprint = hashlib.sha256(np.round(audio*1e6).astype('<i4').tobytes()).hexdigest()
            partition = self.known.get(fingerprint)
            context = {'development': 'Known development recording: this speaker was included in model fitting. This is not an unseen demonstration.',
                       'test': 'Known reserved-test recording: excluded from model fitting; the final test has already been evaluated.'}.get(
                           partition, 'Unrecognized recording. Speaker identity and recording conditions cannot be verified from the upload.')
            predictions = {}
            for family, bundle in self.bundles.items():
                p = probabilities(bundle, features[None])[0]
                order = np.argsort(-p)
                label = self.labels[int(order[0])]
                predictions[family] = {'diagnosis': label, 'result': 'Healthy' if label=='Healthy' else 'Pathological',
                    'score': float(p[order[0]]), 'description': DESCRIPTIONS[label],
                    'top3': [{'Diagnosis': self.labels[int(i)], 'Model score (%)': float(p[i]*100)} for i in order[:3]]}
        return {'models': predictions, 'selected': self.selected, 'audio': audio, 'context': context,
                'metadata': metadata, 'mp3': is_mp3, 'original_seconds': duration,
                'elapsed_seconds': time.perf_counter()-start}
