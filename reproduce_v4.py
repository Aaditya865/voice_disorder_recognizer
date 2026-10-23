"""Rebuild v4 in a fresh workspace, without changing the deployed release."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
import pandas as pd
from svd_v3 import sha
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--svd-audio-root',type=Path,required=True);p.add_argument('--voiced-root',type=Path,required=True)
    p.add_argument('--source-meta',type=Path,required=True);p.add_argument('--extra-audio-root',type=Path,required=True)
    p.add_argument('--wavlm-root',type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    workspace=a.output.resolve()
    if workspace.exists():raise ValueError('Use a new output workspace; releases are never overwritten.')
    root=workspace/'outputs/voice_disorder_recognizer';root.parent.mkdir(parents=True)
    subprocess.run([sys.executable,str(ROOT/'reproduce_v3.py'),'--output',str(root),'--svd-audio-root',str(a.svd_audio_root.resolve()),
      '--voiced-root',str(a.voiced_root.resolve()),'--source-meta',str(a.source_meta.resolve()),'--prepare-only'],check=True)
    metadata=workspace/'work/svd';metadata.mkdir(parents=True);shutil.copy2(a.source_meta,metadata/'voice_data.csv')
    for name in ['svd_v4.py','svd_v4_results.py','v4_embeddings.py','build_v4_report.py']:shutil.copy2(ROOT/name,root/name)
    out=root/'experiments/v4';out.mkdir(parents=True)
    manifest=pd.read_csv(ROOT/'experiments/v4/download_manifest.csv').fillna('')
    for i,r in manifest.iterrows():
        if r.error:continue
        path=(a.extra_audio_root/(r.record_id.split(':')[1]+'-a_n.wav')).resolve()
        if sha(path)!=r.raw_hash:raise ValueError('New speaker audio hash mismatch: '+r.record_id)
        manifest.loc[i,'path']=str(path)
    manifest.to_csv(out/'download_audit.csv',index=False)
    for name in ['new_speaker_partitions.csv','AUDIT.md']:
        if (ROOT/'experiments/v4'/name).exists():shutil.copy2(ROOT/'experiments/v4'/name,out/name)
    for name in ['nested_folds.json','nested_predictions.csv','binary.joblib']:
        shutil.copy2(ROOT/'artifacts_v3'/name,root/'artifacts_v3'/name)
    if a.prepare_only:print('Verified input hashes and prepared:',root);return
    if a.wavlm_root:shutil.copytree(a.wavlm_root,out/'wavlm')
    env=os.environ.copy()
    optional=ROOT.parents[1]/'work/v4_dependencies'
    if optional.exists():env['PYTHONPATH']=str(optional)+os.pathsep+env.get('PYTHONPATH','')
    def run(script,*args):subprocess.run([sys.executable,'-u',str(root/script),*args],cwd=root,check=True,env=env)
    run('svd_v3.py','features');run('v4_embeddings.py');run('svd_v4.py','prepare');run('svd_v4.py','train');run('svd_v4_results.py','report')
    run('build_v4_report.py')
    print('Development reproduction complete. Challenge is not rerun automatically:',root)

if __name__=='__main__':main()
