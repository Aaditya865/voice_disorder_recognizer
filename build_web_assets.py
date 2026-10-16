"""Create sample recordings and review reports from saved measured outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from voice_review.data import ROOT, save_json
from voice_review.features import read_audio, preprocess, audio_fingerprint
from voice_review.laryngeal import label_for, NAMES
from voice_review.web_inference import ModelService


def main():
    root = ROOT
    output = root/'results/laryngeal'
    info = json.loads((output/'metrics.json').read_text())
    audit = pd.read_csv(output/'diagnosis_audit.csv')
    frame = pd.read_csv(root/'data/manifest.csv')
    splits = json.loads((root/'data/splits.json').read_text())
    dev = frame[frame.record_id.isin(splits['development'])].copy()
    dev['subtype'] = dev.diagnosis.map(label_for)
    sampledir = root/'samples'
    sampledir.mkdir(exist_ok=True)
    samples = []
    service = ModelService(root)
    for label,name in [(0,'healthy'),(2,'pathological'),(1,'laryngeal')]:
        row = (dev[dev.record_id=='voice003'] if label==2 else dev[dev.subtype==label].sort_values('record_id')).iloc[0]
        audio,sr = read_audio(root/row.source)
        crop = preprocess(audio,sr)
        path = sampledir/f'{name}_{row.record_id}.wav'
        sf.write(path,crop,8000,subtype='DOUBLE')
        prediction = service.analyze(path.read_bytes(),path.name)
        assert prediction['record_id']==row.record_id and prediction['fold'] is not None
        samples.append({'filename':path.name,'record_id':row.record_id,'dataset_diagnosis':row.diagnosis,
                        'dataset_group':NAMES[label],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'audio_hash':audio_fingerprint(crop),'validation_fold':prediction['fold'],
                        'measured_predictions':prediction['models'],'analysis_seconds':prediction['elapsed_seconds']})
    save_json(sampledir/'manifest.json',samples)
    (sampledir/'README.md').write_text('# Demo audio\n\nThese are real VOICED development recordings. Healthy and laryngitis examples use the lowest development record ID in their group. The pathological example reuses voice003, the already supplied unseen-to-training demonstration. File names describe dataset labels, not guaranteed predictions. Examples are illustrative, not a performance estimate. All clips are sustained /a/, central three seconds, mono 8 kHz. Double-precision WAV preserves the original feature input.\n\nThe website recognizes the audio fingerprint and automatically uses speaker-held-out validation models. The speaker is excluded from fitting and early stopping; model-setting selection still used development validation. Do not convert these files to MP3 for an unseen-speaker claim: compression changes the fingerprint and signal.\n\nSource: VOICED 1.0.0, https://physionet.org/content/voiced/1.0.0/ . Open Data Commons Attribution License v1.0. See REFERENCES.txt.\n',encoding='utf-8')
    fig,axes = plt.subplots(1,2,figsize=(11,4.4),constrained_layout=True)
    for ax,family in zip(axes,['GMM','DNN']):
        cm = np.array(info['models'][family]['confusion_matrix'])
        ax.imshow(cm,cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18,color='white' if cm[i,j]>cm.max()*.6 else '#172033')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['Healthy','Laryngitis group','Other pathology'],
               yticklabels=['Healthy','Laryngitis group','Other pathology'],xlabel='Predicted group',ylabel='Dataset group',
               title=f'{family}: exploratory three-class validation')
        ax.tick_params(axis='x',rotation=18)
    fig.savefig(output/'confusion_matrices.png',dpi=160)
    plt.close(fig)
    subtype = pd.read_csv(output/'predictions.csv')
    binary = pd.read_csv(root/'results/development_predictions.csv')
    merged = subtype.merge(binary[['record_id','GMM_prediction','DNN_prediction']],on='record_id',suffixes=('_subtype','_binary'))
    combined = {}
    for family in ['GMM','DNN']:
        b,s = merged[family+'_prediction_binary'].to_numpy(),merged[family+'_prediction_subtype'].to_numpy()
        display_group = np.where(b==0,0,np.where(s==0,3,s))
        combined[family] = {'confusion_matrix':confusion_matrix(merged.label,display_group,labels=[0,1,2,3]).tolist()[:3],
            'rows':NAMES,'columns':NAMES+['Unresolved pathological'],
            'per_class':classification_report(merged.label,display_group,labels=[0,1,2],target_names=NAMES,output_dict=True,zero_division=0)}
    save_json(output/'website_decision_metrics.json',combined)
    lines = ['# Laryngeal label audit and exploratory extension',
      '\n## Decision\nThe release has 38 reflux-laryngitis records and one hypokinetic-dysphonia record explicitly labelled laryngitis. There are 32 in development and seven reserved for final testing. This permits a small exploratory label-group experiment, not reliable clinical laryngeal detection. We do not infer anatomical absence from other diagnosis strings. Other pathology may also involve the larynx. Independent identity remains based on record IDs as participant proxies.',
      '\n## Exact diagnosis audit\n| Source diagnosis | Speaker proxies | Development | Reserved test | Assigned group |\n|---|---:|---:|---:|---|']
    for r in audit.itertuples():
        lines.append(f'| {r.diagnosis} | {r.speaker_proxies} | {r.development} | {r.reserved_test} | {r.group} |')
    lines += ['\n## Protocol\nAll original development/test IDs and three validation folds remain fixed. Labels: Healthy; Laryngeal (laryngitis-labelled); Other pathology. No new model-family search: one diagonal GMM component per class and the existing small CNN with a three-logit head and dropout 0.4. CNN uses training-only inverse-frequency class weights, an internal stopping subset and fresh fold refits. Features and transforms reuse the original implementations. Three-class models are saved separately and do not replace the original binary models. The architecture was informed by prior development work; these are preliminary development estimates. Test audio/features remain sealed.',
      '\n## Three-class model results\n| Model | Group | Precision | Recall | F1 | Support |\n|---|---|---:|---:|---:|---:|']
    for family,r in info['models'].items():
        for name in NAMES:
            m = r['report'][name]
            lines.append(f'| {family} | {name} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["f1-score"]:.3f} | {int(m["support"])} |')
    for family,r in info['models'].items():
        lines.append(f'\n{family}: balanced accuracy {r["balanced_accuracy"]:.1%}; macro-F1 {r["report"]["macro avg"]["f1-score"]:.3f}.\n')
    lines += ['\n![Three-class confusion matrices](results/laryngeal/confusion_matrices.png)',
      '\n## Interpretation\nLaryngitis-group precision is only 32.4% for GMM and 27.3% for DNN. Most positive subtype assignments are therefore wrong on this development comparison. Do not present a subtype suggestion as a diagnosis. Group counts do not establish clinical adequacy. The scope excludes all other separately named disorders and broad laryngeal detection.',
      '\n## Website decision metrics\nThe website retains the binary decision, then adds an exploratory subtype only for a pathological result. A conflicting healthy subtype prediction becomes Unresolved. These combined outputs differ from the auxiliary three-class model above. The following confusion counts evaluate exactly that displayed decision rule. Rows: actual Healthy / Laryngitis group / Other pathology. Columns: predicted Healthy / Laryngitis group / Other pathology / Unresolved pathological.']
    for family,r in combined.items():
        lines.append(f'\n### {family}\n\n| Actual | Healthy | Laryngitis group | Other pathology | Unresolved |\n|---|---:|---:|---:|---:|')
        for name,counts in zip(NAMES,r['confusion_matrix']):
            lines.append('| '+name+' | '+' | '.join(map(str,counts))+' |')
        lines.append('\n| Group | Precision | Recall | F1 |\n|---|---:|---:|---:|')
        for name in NAMES:
            m=r['per_class'][name]
            lines.append(f'| {name} | {m["precision"]:.3f} | {m["recall"]:.3f} | {m["f1-score"]:.3f} |')
    (root/'LARYNGEAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(samples,indent=2))


if __name__=='__main__':
    main()
