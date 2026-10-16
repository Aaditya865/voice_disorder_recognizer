"""Generate a portable, self-contained review and a reusable notebook from real results."""
from pathlib import Path
import base64
import html
import io
import json
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import nbformat as nbf
from .data import ROOT
from .features import extract_path

REFERENCES = '''Gupta R, Jin CT, Gunjawate DR, Nguyen DD, Stasak B, Chacon AM, Madill C. (2026).
Voice disorders classification using machine learning: a scoping review. Frontiers in Digital Health 8:1800132.
https://doi.org/10.3389/fdgth.2026.1800132

Cesari U, De Pietro G, Marciano E, Niri C, Sannino G, Verde L. (2018).
A new database of healthy and pathological voices. Computers & Electrical Engineering 68:310–321.
https://doi.org/10.1016/j.compeleceng.2018.04.008

VOICED Database, version 1.0.0. https://doi.org/10.13026/C25Q2N
https://physionet.org/content/voiced/1.0.0/ — Open Data Commons Attribution License v1.0.
License: https://opendatacommons.org/licenses/by/1-0/

Pollard TJ et al. (2026). PhysioNet as a global platform for biomedical research. Nature Health.
https://doi.org/10.1038/s44360-026-00096-z
'''

def figure_uri(fig, path):
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()

def pct(v):
    return f'{100*v:.1f}%' if v is not None else 'Not estimable'

def build_report(root=ROOT):
    results = root/'results'
    figures = results/'figures'
    figures.mkdir(exist_ok=True)
    info = json.loads((results/'development_metrics.json').read_text())
    audit = json.loads((root/'data/audit.json').read_text())
    selected = json.loads((results/'selection.json').read_text())
    search = json.loads((results/'search.json').read_text())
    intervals = json.loads((results/'development_intervals.json').read_text())
    predictions = pd.read_csv(results/'development_predictions.csv')
    manifest = pd.read_csv(root/'data/manifest.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = {'GMM':'#0f766e','DNN':'#4f46e5'}
    fig, axes = plt.subplots(1,2,figsize=(9,3.6))
    for ax, family in zip(axes,['GMM','DNN']):
        cm = np.asarray(info['models'][family]['confusion_matrix'])
        ax.imshow(cm, cmap='Blues', vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18,color='white' if cm[i,j]>cm.max()/2 else '#172033')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Healthy','Pathological'],yticklabels=['Healthy','Pathological'],
               xlabel='Predicted label',ylabel='Dataset label',title=family+' · development validation')
    fig.tight_layout()
    cm_uri = figure_uri(fig,figures/'confusion_matrices.png')
    fig, ax = plt.subplots(figsize=(6,4))
    for family in ['GMM','DNN']:
        fpr,tpr,_ = roc_curve(predictions.label,predictions[family+'_score'])
        ax.plot(fpr,tpr,color=colors[family],lw=2,label=f'{family} · AUC {info["models"][family]["roc_auc"]:.3f}')
    ax.plot([0,1],[0,1],'--',color='#94a3b8')
    ax.set(xlabel='False-positive rate',ylabel='True-positive rate',title='Development ROC curves',xlim=(0,1),ylim=(0,1))
    ax.legend(frameon=False)
    roc_uri = figure_uri(fig,figures/'roc_curves.png')
    imp = pd.read_csv(results/'permutation_importance.csv')
    summary = imp.groupby(['model','feature_group']).mean(numeric_only=True).reset_index()
    fig, ax = plt.subplots(figsize=(7,3.4))
    ax.barh([r.model+' · '+r.feature_group for r in summary.itertuples()], summary.mean_balanced_accuracy_drop,
            color=[colors[m] for m in summary.model])
    ax.axvline(0,color='#94a3b8',lw=1)
    ax.set(xlabel='Mean decrease in balanced accuracy after permutation',title='What the selected models use')
    importance_uri = figure_uri(fig,figures/'permutation_importance.png')
    rows = []
    for family in ['GMM','DNN']:
        m = info['models'][family]
        lo,hi = intervals['models'][family]['balanced_accuracy']
        rows.append(f'<tr><th>{family}</th><td><b>{pct(m["balanced_accuracy"])}</b><small>{pct(lo)}–{pct(hi)} bootstrap interval</small></td>'
                    f'<td>{m["macro_f1"]:.3f}</td><td>{pct(m["sensitivity"])}</td><td>{pct(m["specificity"])}</td><td>{m["roc_auc"]:.3f}</td><td>{pct(m["accuracy"])}</td></tr>')
    baseline = info['majority_baseline']
    rows.append(f'<tr class="muted"><th>Always pathological</th><td>{pct(baseline["balanced_accuracy"])}</td><td>{baseline["macro_f1"]:.3f}</td><td>100.0%</td><td>0.0%</td><td>0.500</td><td>{pct(baseline["accuracy"])}</td></tr>')
    audio_cards = []
    examples = root/'examples'
    examples.mkdir(exist_ok=True)
    for label in [0,1]:
        rid = sorted(predictions[predictions.label==label].record_id)[0]
        row = manifest.set_index('record_id').loc[rid]
        _, _, status, audio = extract_path(root/row.source)
        # Float WAV retains the canonical crop for identical inference and seal checks.
        path = examples/f'{rid}.wav'
        sf.write(path,audio,8000,subtype='FLOAT')
        buffer = io.BytesIO()
        sf.write(buffer,audio,8000,format='WAV',subtype='PCM_16')
        uri = 'data:audio/wav;base64,'+base64.b64encode(buffer.getvalue()).decode()
        p = predictions.set_index('record_id').loc[rid]
        result_text = ' · '.join(f'{m}: {"pathological" if p[m+"_prediction"] else "healthy"}' for m in ['GMM','DNN'])
        audio_cards.append(f'<article class="sample"><span class="eyebrow">DEVELOPMENT EXAMPLE / {rid}</span><h3>{"Healthy" if label==0 else "Pathological"} dataset label</h3><audio controls preload="none" src="{uri}"></audio><p>{result_text}</p><small>Predictions shown are out-of-fold. Example selected by ID, not correctness. Audio is a three-second excerpt.</small></article>')
    quality = json.loads((results/'feature_quality_development.json').read_text())
    missing_count = sum(bool(x['missing_quality_features']) for x in quality)
    candidates = []
    for c in search['candidates']:
        setting = f'{c["components"]} components' if c['family']=='GMM' else f'dropout {c["dropout"]}'
        candidates.append(f'<tr><td>{c["family"]}</td><td>{c["features"].replace("_"," ")}</td><td>{setting}</td><td>{pct(c["mean_fold_balanced_accuracy"])}</td></tr>')
    final_path = results/'final_metrics.json'
    final_html = '<p class="callout">The final test set is sealed. No held-out test performance has been measured. All numbers on this page are development results.</p>'
    if final_path.exists():
        final = json.loads(final_path.read_text())
        final_predictions = pd.read_csv(results/'final_predictions.csv')
        fig, axes = plt.subplots(1,3,figsize=(13,3.7))
        for ax, family in zip(axes[:2],['GMM','DNN']):
            cm = np.asarray(final['models'][family]['confusion_matrix'])
            ax.imshow(cm,cmap='Blues',vmin=0)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=17)
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Healthy','Pathological'],
                   yticklabels=['Healthy','Pathological'],xlabel='Prediction',ylabel='Dataset label',title=family+' final test')
            fpr,tpr,_ = roc_curve(final_predictions.label,final_predictions[family+'_score'])
            axes[2].plot(fpr,tpr,label=family,color=colors[family])
        axes[2].plot([0,1],[0,1],'--',color='#94a3b8')
        axes[2].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Final test ROC')
        axes[2].legend()
        fig.tight_layout()
        final_uri = figure_uri(fig,figures/'final_evaluation.png')
        final_rows = []
        for m,v in final['models'].items():
            lo,hi = final['intervals']['models'][m]['balanced_accuracy']
            final_rows.append(f'<tr><th>{m}</th><td>{pct(v["balanced_accuracy"])}<small>{pct(lo)}–{pct(hi)}</small></td><td>{v["macro_f1"]:.3f}</td><td>{pct(v["sensitivity"])}</td><td>{pct(v["specificity"])}</td><td>{v["roc_auc"]:.3f}</td><td>{pct(v["accuracy"])}</td></tr>')
        final_html = '<section class="panel"><h2>Final held-out evaluation completed</h2><p>Frozen models, one independent test evaluation. Small-sample uncertainty remains.</p><div class="table-wrap"><table><thead><tr><th>Model</th><th>Balanced accuracy / 95% interval</th><th>Macro-F1</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th><th>Accuracy</th></tr></thead><tbody>' + ''.join(final_rows) + f'</tbody></table></div><img alt="Final test confusion matrices and ROC curves" src="{final_uri}"></section>'
    status = 'TEST SEALED' if not final_path.exists() else 'FINAL TEST COMPLETE'
    title = 'Voice Disorder<br><em>Recognizer</em>'
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Voice Disorder Recognizer · Project review</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--teal:#0f766e;--line:#dce5ea;--paper:#f4f7fa}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);font:16px/1.65 "Segoe UI",Arial,sans-serif;color:var(--ink)}}main{{max-width:1120px;margin:auto;padding:40px 32px 70px}}nav{{display:flex;justify-content:space-between;align-items:center;padding-bottom:30px;font-size:13px;font-weight:700;letter-spacing:1px}}.badge{{background:#e0f2ee;color:#075e55;padding:7px 12px;border-radius:20px;font-size:11px}}header{{background:#102b3b;color:white;padding:48px;border-radius:20px;position:relative;overflow:hidden}}header:after{{content:"∿ ∿ ∿";position:absolute;right:30px;top:40px;font-size:100px;color:#48bdad;opacity:.2;letter-spacing:-8px}}.eyebrow{{font-size:11px;letter-spacing:2px;font-weight:700;color:#31998c}}header .eyebrow{{color:#7ee0cf}}h1{{font-size:52px;line-height:1.08;letter-spacing:-2px;margin:18px 0}}h1 em{{font-style:normal;color:#7ee0cf}}header p{{max-width:650px;color:#c4d6df}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:22px 0}}.stat,.panel,.sample{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px}}.stat strong{{display:block;font-size:32px;line-height:1.2}}.stat span{{color:var(--muted);font-size:13px}}h2{{font-size:24px;letter-spacing:-.5px;margin:0 0 8px}}h3{{font-size:18px}}section{{margin-top:28px}}.muted,small{{color:var(--muted)}}small{{display:block;font-size:12px}}.callout{{padding:16px 20px;background:#fff6e5;border-left:4px solid #d19b2d;border-radius:5px;font-size:14px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:13px 10px;text-align:left;border-bottom:1px solid var(--line)}}thead th{{font-size:11px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted)}}.table-wrap{{overflow:auto}}.twocol{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img{{width:100%;height:auto}}.flow{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:16px 0}}.flow span{{padding:10px 13px;border:1px solid var(--line);border-radius:8px;background:white;font-size:13px}}.flow b{{color:var(--teal)}}audio{{width:100%;margin:12px 0}}li{{margin:8px 0}}a{{color:var(--teal)}}details summary{{cursor:pointer;font-weight:600;padding:8px 0}}footer{{margin-top:40px;border-top:1px solid var(--line);padding-top:20px;color:var(--muted);font-size:12px}}.pill{{font-size:11px;border-radius:5px;padding:4px 8px;background:#edf2f7;display:inline-block;margin-right:6px}}@media(max-width:760px){{main{{padding:20px 16px}}header{{padding:28px}}h1{{font-size:38px}}.grid,.twocol{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{padding:0}}header{{print-color-adjust:exact}}section{{break-inside:avoid}}}}
</style></head><body><main><nav><span>VOICE LAB / ACADEMIC PROJECT</span><span class="badge">{status}</span></nav><header><span class="eyebrow">GMM + CONVOLUTIONAL DNN</span><h1>{title}</h1><p>A reproducible comparison of two machine-learning pipelines for healthy versus pathological sustained vowels. Built with real VOICED recordings.</p><span class="pill" style="color:#172033">RESEARCH PROTOTYPE</span><span class="pill" style="color:#172033">NOT A CLINICAL TOOL</span></header>
<div class="grid"><div class="stat"><strong>{audit['records']}</strong><span>VOICED recordings audited</span></div><div class="stat"><strong>{info['development_speakers']}</strong><span>Development participants · 3 validation folds</span></div><div class="stat"><strong>{info['sealed_test_speakers']}</strong><span>Reserved final-test participants</span></div></div>
{final_html}
<section class="panel"><span class="eyebrow">01 / EXPERIMENT</span><h2>From recording to prediction</h2><div class="flow"><span>Audit + participant split</span><b>→</b><span>Central 3 s /a/</span><b>→</b><span>MFCC + voice quality</span><b>→</b><span>GMM / DNN</span><b>→</b><span>Validation</span></div><div class="twocol"><div><h3>Statistical baseline</h3><p>Class-conditional diagonal GMMs on pooled MFCCs, with optional voice-quality measurements. Equal class priors; 1, 2 and 4 components tested.</p></div><div><h3>Temporal neural model</h3><p>Two convolution layers, global-average pooling and a compact classifier, with an optional voice-quality branch. Training-only class weights and early stopping.</p></div></div><p class="muted">The selected feature set for each model is shown below. This compares complete pipelines with different pooling, not architecture alone.</p></section>
<section class="panel"><span class="eyebrow">02 / PRELIMINARY FINDINGS</span><h2>Development validation results</h2><p class="muted">Each participant is predicted by a fold model trained without that participant. The same folds selected settings, so these estimates can still be optimistic.</p><div class="table-wrap"><table><thead><tr><th>Model</th><th>Balanced accuracy</th><th>Macro-F1</th><th>Sensitivity</th><th>Specificity</th><th>ROC-AUC</th><th>Accuracy</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p><small>Bootstrap intervals are approximate, conditional on fitted fold models, and do not account for model-selection or training uncertainty. The majority rule illustrates why accuracy alone is insufficient.</small></p><img alt="Development confusion matrices with counts" src="{cm_uri}"></section>
<section class="twocol">{''.join(audio_cards)}</section>
<section class="twocol"><div class="panel"><h2>Ranking performance</h2><img alt="ROC curves on development validation predictions" src="{roc_uri}"></div><div class="panel"><h2>Feature-group sensitivity</h2><img alt="Grouped validation permutation importance" src="{importance_uri}"><small>Ten permutations per fold. Correlated features and distribution changes limit interpretation; this is not causal evidence.</small></div></section>
<section class="panel"><span class="eyebrow">03 / VALIDITY</span><h2>What the experiment can and cannot establish</h2><ul><li>Participant and duplicate hashes are disjoint across development and test partitions.</li><li>Record IDs are participant proxies based on the dataset documentation; cross-session identity cannot be independently reverified.</li><li>Scaling, imputation and class weights are fitted inside training partitions. The test set is never used for tuning.</li><li>{missing_count} development recordings have at least one missing voice-quality measurement. Missing values use training-only medians and indicators.</li><li>Only one small, controlled dataset is used. Results do not establish performance on phones, connected speech, unseen sites, or specific disorder types.</li><li>The reference is a multiclass scoping review, not a model benchmark. This binary project adopts its evaluation recommendations without claiming to reproduce its results.</li></ul><details><summary>View the complete model search</summary><div class="table-wrap"><table><thead><tr><th>Family</th><th>Features</th><th>Setting</th><th>Mean fold balanced accuracy</th></tr></thead><tbody>{''.join(candidates)}</tbody></table></div></details></section>
<section class="panel"><span class="eyebrow">04 / SUBMISSION</span><h2>Review now. Final evaluation when frozen.</h2><p>The working notebook and saved models support prediction on suitable recordings. Final evaluation is a separate, one-time command after the experiment is frozen.</p><p><a href="Voice_Disorder_Review.ipynb">Open notebook file</a> · <a href="REPORT.md">Read academic report</a> · <a href="README.md">Setup and reproduction instructions</a></p></section>
<footer><b>Sources</b><br>Gupta et al. (2026), <a href="https://doi.org/10.3389/fdgth.2026.1800132">Voice disorders classification using machine learning: a scoping review</a>.<br>Cesari et al. (2018), <a href="https://doi.org/10.1016/j.compeleceng.2018.04.008">A new database of healthy and pathological voices</a>.<br><a href="https://physionet.org/content/voiced/1.0.0/">VOICED 1.0.0 / PhysioNet</a> · Open Data Commons Attribution License v1.0.<br>Generated exclusively from saved experiment outputs. No synthetic data are used in the reported experiment.</footer></main></body></html>'''
    discrepancy = '<p class="callout"><b>Dataset audit finding:</b> file-level labels are 57 healthy / 151 pathological in both metadata formats. The website summary reports 58 / 150. We preserve the supplied file-level labels; the discrepancy is unresolved.</p>'
    document = document.replace('<section class="panel"><span class="eyebrow">01 / EXPERIMENT</span>',discrepancy+'<section class="panel"><span class="eyebrow">01 / EXPERIMENT</span>')
    selected_text = 'Selected settings: ' + '; '.join(f'{m}: {r["config_id"].replace("_"," ")}' for m,r in selected.items()) + '.'
    document = document.replace('The selected feature set for each model is shown below.',html.escape(selected_text))
    robustness_html = build_robustness_report(root)
    document = document.replace('<section class="panel"><span class="eyebrow">03 / VALIDITY</span>', robustness_html + '<section class="panel"><span class="eyebrow">03 / VALIDITY</span>')
    (root/'Review.html').write_text(document,encoding='utf-8')
    (root/'REFERENCES.txt').write_text(REFERENCES,encoding='utf-8')
    build_notebook(root)
    build_academic_report(root, info, audit, selected, intervals)
    print('Created Review.html, Voice_Disorder_Review.ipynb and REPORT.md from measured development results.',flush=True)

def build_notebook(root):
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md('# Voice Disorder Recognizer\n### GMM + convolutional DNN · academic review\n\nReal VOICED recordings. **Development validation results; final test is sealed unless explicitly evaluated.**\n\nThis is an academic prototype, not a medical diagnostic tool. Run cells in order. No cell here trains models or opens the test set.'),
        code('from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Audio, Image\nfrom voice_review.experiment import predict\nROOT = Path.cwd()\nassert (ROOT / "models/gmm.joblib").exists(), "Open this notebook from the project folder."\ninfo = json.loads((ROOT / "results/development_metrics.json").read_text())\naudit = json.loads((ROOT / "data/audit.json").read_text())\nprint(f"{audit[\'records\']} records | {info[\'development_speakers\']} development | {info[\'sealed_test_speakers\']} reserved test")'),
        md('## 1. Method\nAudit identities and labels → freeze speaker partitions → central three-second crop → MFCC and voice-quality features → fold-local transforms → GMM / DNN → grouped validation.\n\nGMM pools MFCCs; the DNN uses their time sequence. The same source recordings and feature families are compared. Metadata and diagnosis strings are never predictors.'),
        code('selection = json.loads((ROOT / "results/selection.json").read_text())\ndisplay(pd.DataFrame([{k:v for k,v in row.items() if k != "pooled_oof_metrics"} for row in selection.values()]))'),
        md('## 2. Preliminary validation\nThese folds also selected settings; performance is not an unbiased final estimate. Balanced accuracy gives equal weight to healthy and pathological recall.'),
        code('display(pd.DataFrame(info["models"]).T.drop(columns=["confusion_matrix"]))\ndisplay(Image(filename=str(ROOT / "results/figures/confusion_matrices.png")))\ndisplay(Image(filename=str(ROOT / "results/figures/roc_curves.png")))'),
        md('## 3. Working audio prediction\nRun this cell to activate the controls. Keep **Speaker-held-out demo** selected, choose an example and click **Analyze example**. The default voice003.wav is real audio from a speaker excluded from those fold models\' fitting and early stopping.\n\nThese are **development validation** predictions: settings were selected using validation results, so this is not a completely independent final test. For your own new WAV/FLAC, choose **Full-development models**, then **Choose audio** and **Analyze chosen file**. Full-development models have seen the bundled examples. Scores are uncalibrated, not medical probabilities.'),
        code('from voice_review.demo import make_demo, show_recording\ndemo = make_demo(ROOT)\ndisplay(demo)'),
        md('## 4. Optional local-path alternative\nIf the controls are unavailable, set `audio_path` below to a local WAV/FLAC containing at least three seconds of sustained /a/. Displayed audio and results can be saved in notebook outputs; clear personal recordings before sharing.'),
        code('audio_path = None  # Example: ROOT / "examples" / "voice001.wav"\nif audio_path is not None:\n    try:\n        result = show_recording(audio_path, ROOT)\n    except ValueError as exc:\n        print("Recording not accepted:", exc)\nelse:\n    print("Use the controls above, or set audio_path for the local-path alternative.")'),
        md('## 5. Explainability and limitations\nPermutation analysis measures prediction sensitivity to feature groups, not causal disease mechanisms. Small subgroups do not support fairness claims. One controlled dataset cannot demonstrate generalization to real-world clinical recordings.'),
        code('display(Image(filename=str(ROOT / "results/figures/permutation_importance.png")))\nprint(audit["identity_basis"])\nprint("Final test:", "evaluated" if (ROOT / "results/final_metrics.json").exists() else "sealed")'),
        md('## 6. Training stability\nFour fixed seed runs on the original development folds. Initialization, minibatch order and the internal stopping subset vary; the architecture and outer folds stay fixed. Every run is shown, and the original saved models are retained. This does not measure uncertainty across new datasets.'),
        code('stability_path = ROOT / "results/robustness/summary.json"\nif stability_path.exists():\n    stability = json.loads(stability_path.read_text())\n    display(pd.DataFrame(stability["summary"]).T)\n    display(Image(filename=str(ROOT / "results/figures/seed_stability.png")))\nelse:\n    print("Optional development seed analysis has not been run.")'),
        md('## 7. References\n' + REFERENCES.replace('\n\n','\n\n')),
    ]
    nb.metadata['kernelspec'] = {'display_name':'Python 3 (Voice project)', 'language':'python', 'name':'python3'}
    nb.metadata['language_info'] = {'name':'python','version':'3.12'}
    nbf.write(nb,root/'Voice_Disorder_Review.ipynb')


def build_robustness_report(root):
    path = root/'results/robustness/summary.json'
    if not path.exists():
        return ''
    result = json.loads(path.read_text())
    runs = pd.DataFrame(result['runs'])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for family, color in [('GMM','#0f766e'),('DNN','#4f46e5')]:
        group = runs[runs.model==family].sort_values('seed')
        ax.plot(range(len(group)), 100*group.balanced_accuracy, 'o-', color=color, label=family)
    seeds = sorted(runs.seed.unique())
    ax.set(xticks=range(len(seeds)), xticklabels=[str(s) for s in seeds], xlabel='Base random seed',
           ylabel='Balanced accuracy (%)', ylim=(45,70), title='Training stability · fixed development folds')
    ax.axhline(50, linestyle='--', color='#94a3b8', label='Majority-rule baseline')
    ax.legend(frameon=False)
    uri = figure_uri(fig, root/'results/figures/seed_stability.png')
    rows, markdown = [], []
    for family in ['GMM','DNN']:
        r = result['summary'][family]
        cells = [family, pct(r['balanced_accuracy_mean']),
                 f'{100*r["balanced_accuracy_sd"]:.1f} pp',
                 f'{pct(r["balanced_accuracy_min"])}–{pct(r["balanced_accuracy_max"])}',
                 f'{r["records_with_label_changes"]}/{r["records"]}']
        rows.append('<tr>'+''.join(f'<td>{v}</td>' for v in cells)+'</tr>')
        markdown.append('| '+' | '.join(cells)+' |')
    pred = pd.read_csv(root/'results/development_predictions.csv')
    gmm_wrong = pred.GMM_prediction != pred.label
    dnn_wrong = pred.DNN_prediction != pred.label
    groups = {
        'Both correct': int((~gmm_wrong & ~dnn_wrong).sum()),
        'Both wrong': int((gmm_wrong & dnn_wrong).sum()),
        'GMM correct only': int((~gmm_wrong & dnn_wrong).sum()),
        'DNN correct only': int((gmm_wrong & ~dnn_wrong).sum()),
    }
    agreement = ' · '.join(f'{name}: {count}' for name,count in groups.items())
    interpretation = ('DNN predictions change for some participants across training seeds, so a single-run result is fragile. '
                      'The selected one-component GMM is effectively deterministic on fixed data; its zero seed spread is expected '
                      'and does not imply robust generalization. The same development folds selected the configurations, so selection bias remains. '
                      'Seed standard deviations are descriptive, not confidence intervals. No best seed is selected and no demonstration model is replaced.')
    protocol = json.loads((root/'results/robustness/protocol.json').read_text())
    text = '# Development robustness analysis\n\n'+interpretation+'\n\n'
    text += '| Model | Mean balanced accuracy | Seed SD | Seed range | Records whose label changes |\n|---|---:|---:|---:|---:|\n'+'\n'.join(markdown)
    text += '\n\n## Protocol\n\nFour runs: original seed '+str(protocol['original_seed'])+', plus '+', '.join(map(str,protocol['repeat_seeds']))+'. Fold seed = base seed + fold index. Architecture, feature set, decision threshold and outer partitions stay fixed. Initialization, minibatch order and training-only stopping subset vary. This mixes those sources of training variability; it does not isolate initialization alone. The protocol and source hashes were saved before fitting the extra runs. All extra fits use development speakers only.\n\n'
    text += '## Every run\n\n| Model | Seed | Balanced accuracy | Sensitivity | Specificity |\n|---|---:|---:|---:|---:|\n'
    for r in result['runs']:
        text += f'| {r["model"]} | {r["seed"]} | {pct(r["balanced_accuracy"])} | {pct(r["sensitivity"])} | {pct(r["specificity"])} |\n'
    text += '\n## Original model error overlap\n\n'+agreement+'.\n\nAgreement does not imply correctness. These counts describe original out-of-fold predictions, not the full-development refits used in the live demo. Per-record scores for all seeds are preserved in results/robustness/predictions.csv. The final test remains independent of this analysis.\n'
    (root/'ROBUSTNESS.md').write_text(text,encoding='utf-8')
    return '<section class="panel"><span class="eyebrow">TRAINING STABILITY</span><h2>How much does a training run matter?</h2><p>'+interpretation+'</p><div class="table-wrap"><table><thead><tr><th>Model</th><th>Mean BA</th><th>Seed SD</th><th>Range</th><th>Label changes</th></tr></thead><tbody>'+''.join(rows)+f'</tbody></table></div><img alt="All four seed runs on fixed development folds" src="{uri}"><p><b>Original prediction overlap:</b> {agreement}.</p><p><a href="ROBUSTNESS.md">Full development robustness analysis</a></p></section>'

def build_academic_report(root, info, audit, selected, intervals):
    lines = ['# Voice Disorder Recognizer — academic report',
        '\n## Status\nWorking research prototype. Development experiment completed; final held-out evaluation is pending unless results/final_metrics.json exists. No clinical efficacy claim is made.',
        '\n## Research question\nHow do a class-conditional GMM and a compact convolutional DNN compare for healthy-versus-pathological classification of sustained /a/ recordings under participant-disjoint validation?',
        '\n## Reference-paper relationship\nGupta et al. (2026) review 80 multiclass studies; binary-only studies were excluded. MFCCs appeared in 43 studies and voice-quality features in 19. The authors recommend clear labels, speaker separation, balanced accuracy, interpretability, and cross-database testing. Their reported 39–99% range is not an achievable target or a model result reproduced here. The review does not establish a best model. It contains conflicting summaries of test sizes and framework use; these are not resolved by assumption.',
        f'\n## Dataset and identity\nVOICED 1.0.0: {audit["records"]} records audited, {audit["included"]} retained. Original diagnoses and exclusions are in data/manifest.csv.\n\n{audit["identity_basis"]}',
        '\n## Methods\nCentral three-second crop, DC removal, mono 8 kHz. MFCCs: 13 coefficients including C0, 25 ms Hann window, 10 ms hop, 256-point FFT, 26 mel bands, 50–3800 Hz, no centered padding. Voice quality: CPPS, HNR, local jitter and shimmer, median F0 and IQR, with a 60–500 Hz pitch search. Undefined measurements are imputed using training medians, with six explicit missingness indicators. No amplitude normalization, denoising, or augmentation is added. Dataset-provider filtering is inherited.\n\nGMM: separate diagonal mixtures, equal priors, covariance regularization 0.01, five initializations, 1/2/4 components. CNN: Conv1D 13→16 (kernel 5)→32 (kernel 3), ReLU, global average pooling, optional 12-value quality/missingness branch, 16-unit dense layer, dropout 0.2/0.4, binary logit. AdamW 0.001, weight decay 0.0001, batch 16, up to 100 epochs, patience 12.\n\nThe development portion uses three fixed stratified grouped folds. All transforms and weights are fitted inside training partitions. CNN early stopping uses a further 20% training-fold subset; the fold model is then freshly refitted on all fold-training participants for the selected epoch count. Final development refits use the median selected epoch count. Both models classify at raw score zero; no threshold is optimized on test data.',
        '\n## Model selection\nSettings and MFCC-only versus combined features are selected by mean fold balanced accuracy. Exact ties prefer fewer input features, fewer mixture components, then stronger dropout. This selection uses the displayed development folds, introducing selection optimism. The reserved final test is independent of tuning. This is internal model selection plus a held-out test, not nested cross-validation.',
        '\n## Preliminary development results\n| Model | Balanced accuracy | Macro-F1 | Sensitivity | Specificity | ROC-AUC | Accuracy |\n|---|---:|---:|---:|---:|---:|---:|']
    for family,m in info['models'].items():
        lines.append(f'| {family} | {pct(m["balanced_accuracy"])} | {m["macro_f1"]:.3f} | {pct(m["sensitivity"])} | {pct(m["specificity"])} | {m["roc_auc"]:.3f} | {pct(m["accuracy"])} |')
    lines += ['\nSelected configurations:']
    for family,c in selected.items():
        lines.append(f'- {family}: {c["config_id"]}; mean fold balanced accuracy {pct(c["mean_fold_balanced_accuracy"])}.')
    lines += ['\nIntervals and paired differences are in results/development_intervals.json. They resample participants and condition on the fitted models; they do not account for hyperparameter-selection or training uncertainty. Inspect the confusion-matrix counts, not accuracy alone.',
        '\n## Explainability and errors\nFeature-family ablations appear in results/search.json. Grouped permutation importance uses ten independent speaker permutations per validation fold, preserving within-sequence structure. Development error lists and demographic subgroup results are saved separately. Negative permutation drops and weak contributions must be retained, not hidden.',
        '\n## Limitations\nSmall class sizes; one public cohort; no independently verifiable cross-session participant IDs; possible label noise and selection bias; fixed pitch settings; signal checks cannot recognize the vowel; weak generalization evidence; uncalibrated model outputs. GMM and CNN use different pooling, so observed differences cannot be attributed solely to model family. External validation, connected speech, and disorder-type classification are deferred.',
        '\n## Final-test procedure\nAfter the review, freeze the experiment and invoke the explicit one-time evaluation command in README.md. The command checks artifact hashes and refuses repeated evaluation. Do not retrain or select settings after inspecting the final test. A failed evaluation leaves an audit lock for investigation.',
        '\n## References\n'+REFERENCES]
    final_path = root/'results/final_metrics.json'
    if final_path.exists():
        final = json.loads(final_path.read_text())
        lines.insert(2, '\n## Completed held-out results\n' + '\n'.join(f'- {m}: balanced accuracy {pct(r["balanced_accuracy"])}; macro-F1 {r["macro_f1"]:.3f}.' for m,r in final['models'].items()))
    lines.insert(5, '\n### Dataset count discrepancy\n'+audit['count_discrepancy'])
    robustness = root/'ROBUSTNESS.md'
    if robustness.exists():
        lines.insert(-1, '\n## Training variability follow-up\n'+robustness.read_text(encoding='utf-8').replace('# Development robustness analysis\n','').replace('## ', '### '))
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
