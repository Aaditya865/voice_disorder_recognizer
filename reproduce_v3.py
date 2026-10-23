"""Reproduce the frozen cohort in a NEW directory using locally acquired data.

Does not overwrite the distributed release. Each invocation needs a new output
directory. Source WAVs must match the recorded hashes; no silent substitution.
"""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
import pandas as pd
from svd_v3 import sha,key

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--svd-audio-root',type=Path,required=True)
    p.add_argument('--voiced-root',type=Path,required=True)
    p.add_argument('--source-meta',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();dest=args.output.resolve()
    if dest.exists():raise ValueError('Output already exists; select a fresh reproduction directory.')
    cohort=pd.read_csv(ROOT/'artifacts_v3/manifest.csv').fillna('')
    svd=[];voiced=[];copies=[]
    for r in cohort.itertuples():
        ident=r.record_id.split(':',1)[1]
        if r.source=='SVD':
            path=(args.svd_audio_root/f'{ident}-a_n.wav').resolve()
            if sha(path)!=r.raw_hash:raise ValueError('SVD audio bytes differ from the experiment: '+str(path))
            svd.append(dict(record_id=int(ident),speaker_id=int(r.speaker_id.split(':',1)[1]),
                binary=int(r.label),source_diagnoses=r.diagnosis,path=str(path)))
        else:
            path=args.voiced_root/(ident+'.hea')
            if key([sha(path),sha(path.with_suffix('.dat'))])!=r.raw_hash:raise ValueError('VOICED bytes differ: '+ident)
            voiced.append(dict(record_id=ident,speaker_id=ident,label=int(r.label),diagnosis=r.diagnosis,age=r.age,sex=r.sex))
            copies.extend([path,path.with_suffix('.dat'),path.with_name(ident+'-info.txt')])
    expected=json.loads((ROOT/'artifacts_v3/protocol.json').read_text())['source_metadata_hash']
    if sha(args.source_meta)!=expected:raise ValueError('Source metadata hash differs from the frozen experiment.')
    dest.mkdir(parents=True)
    for name in ('svd_v3.py','svd_advanced.py','svd_training.py'):shutil.copy2(ROOT/name,dest/name)
    shutil.copytree(ROOT/'artifacts_svd_v2',dest/'artifacts_svd_v2')
    (dest/'data/svd_v2').mkdir(parents=True);(dest/'data/raw').mkdir()
    pd.DataFrame(svd).to_csv(dest/'data/svd_v2/development_features.csv',index=False)
    pd.DataFrame(voiced).to_csv(dest/'data/manifest.csv',index=False)
    for path in copies:shutil.copy2(path,dest/'data/raw'/path.name)
    def run(stage,extra=None):
        subprocess.run([sys.executable,'-u',str(dest/'svd_v3.py'),stage,*(extra or [])],check=True,cwd=dest)
    run('prepare',['--source-meta',str(args.source_meta.resolve())])
    if not args.prepare_only:
        for stage in ('features','train','report'):run(stage)
    print('Reproduction directory:',dest)
    print('This repeats development evaluation; it does not create an untouched external test.')

if __name__=='__main__':main()
