"""Execute the review notebook with this Python environment and local runtime state."""
import os
import json
from pathlib import Path
import sys
import asyncio

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

root = Path(__file__).resolve().parent
runtime = root / '.runtime'
for name, folder in [('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','runtime'),('JUPYTER_CONFIG_DIR','config')]:
    path = runtime / folder
    path.mkdir(parents=True,exist_ok=True)
    os.environ[name] = str(path)
kernels = runtime/'jupyter/kernels/python3'
kernels.mkdir(parents=True,exist_ok=True)
(kernels/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Python 3 (Voice project)','language':'python'}),encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(runtime/'jupyter')
# Kernel connection files and caches stay in the project workspace.
os.environ['TEMP'] = str(runtime/'runtime')
os.environ['TMP'] = str(runtime/'runtime')
import nbformat
from nbclient import NotebookClient

if __name__ == '__main__':
    path = root/'Voice_Disorder_Review.ipynb'
    notebook = nbformat.read(path,as_version=4)
    NotebookClient(notebook,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
    nbformat.write(notebook,path)
    print('Notebook executed successfully; all outputs saved.',flush=True)
