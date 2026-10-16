"""Download only the public VOICED release; verify its supplied SHA-256 hashes."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
import sys

ROOT = Path(__file__).resolve().parent
URL = 'https://physionet.org/content/voiced/get-zip/1.0.0/'

def main():
    dest = ROOT / 'data' / 'raw'
    dest.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'data' / 'voiced-1.0.0.zip'
    if '--individual' in sys.argv:
        base = ('https://physionet-open.s3.amazonaws.com/voiced/1.0.0/' if '--s3' in sys.argv
                else 'https://physionet.org/files/voiced/1.0.0/')
        for name in ['RECORDS', 'SHA256SUMS.txt']:
            with urllib.request.urlopen(base + name, timeout=60) as response:
                (dest / name).write_bytes(response.read())
        names = []
        for record in (dest / 'RECORDS').read_text().splitlines():
            names.extend([record + '.dat', record + '.hea', record + '-info.txt'])
        def fetch(name):
            path = dest / name
            if path.exists():
                return
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(base + name, timeout=60) as response:
                        payload = response.read()
                    path.write_bytes(payload)
                    return
                except Exception:
                    if attempt == 2:
                        raise
        print(f'Downloading {len(names)} WFDB and metadata files...', flush=True)
        with ThreadPoolExecutor(max_workers=8) as pool:
            for n, _ in enumerate(pool.map(fetch, names), 1):
                if n % 100 == 0:
                    print(f'{n}/{len(names)} files', flush=True)
        verify(dest, base)
        return
    if not archive.exists():
        tmp = archive.with_suffix('.partial')
        print('Downloading the public VOICED release...', flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response, tmp.open('wb') as out:
            shutil.copyfileobj(response, out)
        tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            name = Path(member.filename).name
            if not name or member.is_dir():
                continue
            # Flat allowlist: do not extract arbitrary archive paths.
            if name in {'RECORDS', 'SHA256SUMS.txt', 'LICENSE.txt', 'LICENSE'} or name.endswith(('.hea', '.dat', '-info.txt')):
                (dest / name).write_bytes(z.read(member))
    verify(dest)

def verify(dest, source=URL):
    checks = {}
    for line in (dest / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = dest / Path(name.lstrip('*')).name
        if path.is_file():
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != digest:
                raise RuntimeError(f'Checksum mismatch: {path.name}')
            checks[path.name] = actual
    (ROOT / 'data' / 'download_provenance.json').write_text(json.dumps({
        'source': source, 'release': '1.0.0', 'verified_files': checks,
        'license': 'Open Data Commons Attribution License v1.0',
        'documentation': 'https://physionet.org/content/voiced/1.0.0/'
    }, indent=2), encoding='utf-8')
    print(f'Verified {len(checks)} release files.', flush=True)

if __name__ == '__main__':
    main()
