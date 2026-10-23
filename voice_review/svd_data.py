"""Reproducible SVD audit and selective, CRC-checked official archive download."""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import io
import json
import re
import struct
import time
import unicodedata
import zipfile
import zlib
import numpy as np
import pandas as pd
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/svd"
RESULTS = ROOT / "results/svd"
SEED = 20261020
RECORD_URL = "https://zenodo.org/api/records/16874898"
META_URL = "https://stimmdb.coli.uni-saarland.de/data/voice_data.csv"
GROUPS = {
    "Functional / hyperfunctional dysphonia": {"Funktionelle Dysphonie", "Hyperfunktionelle Dysphonie"},
    "Vocal fold paresis": {"Rekurrensparese"},
    "Laryngitis": {"Laryngitis"},
    "Reinke's edema": {"Reinke Ödem"},
    "Vocal fold polyp": {"Stimmlippenpolyp"},
    "Leukoplakia": {"Leukoplakie"},
}
CANDIDATES = {
    **GROUPS,
    "Vocal fold nodules": {"Phonationsknötchen"},
    "Spasmodic dysphonia": {"Spasmodische Dysphonie"},
    "GERD (not an LPR diagnosis)": {"GERD"},
    "Laryngeal neoplasm": {"Stimmlippenkarzinom", "Kehlkopftumor", "Epiglottiskarzinom", "Carcinoma in situ"},
}

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

def get(url, **kwargs):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=90, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

class RemoteZip(io.RawIOBase):
    """Seekable HTTP byte ranges; never silently download a full archive."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos
    def read(self, n=-1):
        n = min(n if n >= 0 else self.size, self.size - self.pos)
        if n <= 0: return b""
        r = get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos+n-1}"})
        if r.status_code != 206 or len(r.content) != n:
            raise ValueError("Server did not honor the bounded range download.")
        self.pos += n
        return r.content

def source_metadata():
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not (DATA / "source.csv").exists():
        (DATA / "source.csv").write_bytes(get(META_URL).content)
    if not (DATA / "zenodo.json").exists():
        (DATA / "zenodo.json").write_bytes(get(RECORD_URL).content)
    record = json.loads((DATA / "zenodo.json").read_text(encoding="utf-8"))
    archive = next(x for x in record["files"] if x["key"] == "data.zip")
    if not (DATA / "archive_index.json").exists():
        with zipfile.ZipFile(RemoteZip(archive["links"]["self"], archive["size"])) as z:
            dump(DATA / "archive_index.json", [
                dict(name=x.filename, size=x.file_size, compressed=x.compress_size,
                     offset=x.header_offset, method=x.compress_type, crc=x.CRC)
                for x in z.infolist()])
    index = json.loads((DATA / "archive_index.json").read_text())
    audio = {int(re.search(r"/(\d+)-a_n.nsp$", x["name"]).group(1)): x for x in index
             if re.search(r"/(\d+)-a_n.nsp$", x["name"])}
    df = pd.read_csv(DATA / "source.csv").fillna("")
    df = df.rename(columns={"AufnahmeID": "record_id", "SprecherID": "speaker_id",
                            "Geschlecht": "sex", "AufnahmeTyp": "status", "Pathologien": "diagnoses"})
    df["diagnoses"] = df.diagnoses.map(lambda s: unicodedata.normalize("NFC", s))
    df["labels"] = df.diagnoses.map(lambda s: {t.strip() for t in s.split(",") if t.strip()})
    df["has_audio"] = df.record_id.isin(audio)
    return df, audio, archive

def audit():
    df, audio, archive = source_metadata()
    available = df[df.has_audio].copy()
    rows = []
    for label in sorted(set().union(*available.labels)):
        part = available[available.labels.map(lambda a: label in a)]
        rows.append(dict(diagnosis=label, recordings=len(part), unique_speakers=part.speaker_id.nunique()))
    healthy = available[available.status == "n"]
    rows.append(dict(diagnosis="Healthy", recordings=len(healthy), unique_speakers=healthy.speaker_id.nunique()))
    raw = pd.DataFrame(rows).sort_values(["unique_speakers", "diagnosis"], ascending=[False, True])
    raw["frequency_rank"] = raw.unique_speakers.rank(method="min", ascending=False).astype(int)
    raw.to_csv(RESULTS / "diagnosis_audit.csv", index=False)
    candidate = []
    for name, labels in CANDIDATES.items():
        part = available[available.labels.map(lambda a: bool(a & labels))]
        candidate.append(dict(diagnosis=name, source_labels="; ".join(sorted(labels)),
                              recordings=len(part), unique_speakers=part.speaker_id.nunique()))
    cand = pd.DataFrame(candidate).sort_values("unique_speakers", ascending=False)
    cand["frequency_rank"] = cand.unique_speakers.rank(method="min", ascending=False).astype(int)
    cand.to_csv(RESULTS / "candidate_audit.csv", index=False)
    # Do not assign an arbitrary dominant disease to a multi-diagnosis session.
    def label_row(row):
        if row.status == "n" and not row.labels: return "Healthy"
        if row.status != "p": return "EXCLUDE: inconsistent status"
        matches = [name for name, labels in GROUPS.items() if row.labels and row.labels <= labels]
        return matches[0] if len(matches) == 1 else "Other pathology"
    available["class_name"] = available.apply(label_row, axis=1)
    # Healthy/pathological histories from the same speaker are excluded as ambiguous.
    mixed = set(available.groupby("speaker_id").status.nunique().loc[lambda s: s > 1].index)
    available.loc[available.speaker_id.isin(mixed), "class_name"] = "EXCLUDE: mixed healthy/pathological history"
    exclusions = available[available.class_name.str.startswith("EXCLUDE")]
    exclusions.drop(columns="labels").to_csv(RESULTS / "metadata_exclusions.csv", index=False)
    available = available[~available.class_name.str.startswith("EXCLUDE")].copy()
    # First session per speaker is specified before examining acoustic/model performance.
    available = available.sort_values("record_id").drop_duplicates("speaker_id").reset_index(drop=True)
    counts = available.groupby("class_name").agg(recordings=("record_id", "size"), unique_speakers=("speaker_id", "nunique"))
    weak = set(counts[counts.unique_speakers < 20].index) - {"Healthy", "Other pathology"}
    available.loc[available.class_name.isin(weak), "class_name"] = "Other pathology"
    available.drop(columns="labels").to_csv(DATA / "candidates.csv", index=False)
    counts = available.groupby("class_name").agg(recordings=("record_id", "size"), unique_speakers=("speaker_id", "nunique"))
    counts.to_csv(RESULTS / "eligible_before_audio.csv")
    dump(RESULTS / "source_audit.json", {
        "metadata_sessions": len(df), "metadata_unique_speakers": int(df.speaker_id.nunique()),
        "normal_pitch_a_sessions": len(df[df.has_audio]), "mixed_history_speakers_excluded": len(mixed),
        "one_session_per_speaker_candidates": len(available), "weak_classes_merged": sorted(weak),
        "metadata_sha256": hashlib.sha256((DATA / "source.csv").read_bytes()).hexdigest(),
        "source": META_URL, "archive": RECORD_URL, "license": record_license(),
        "policy": "One earliest available normal-pitch /a/ session per speaker. Mixed diagnosis sessions go to Other pathology. Mixed healthy/pathological histories excluded."})
    print(counts.to_string(), flush=True)
    return available, audio, archive

def record_license():
    return json.loads((DATA / "zenodo.json").read_text())["metadata"].get("license")

def decode_nsp(data):
    """Read mono DS16 chunked PCM, checking the HEDR rate and sample count.

    Format reference: FFmpeg libavformat/nspdec.c. No EGG channels accepted.
    """
    if data[:8] != b"FORMDS16": raise ValueError("Unsupported NSP container")
    pos, sr, expected = 12, None, None
    while pos + 8 <= len(data):
        kind, size = data[pos:pos+4], struct.unpack_from("<I", data, pos+4)[0]
        pos += 8
        chunk = data[pos:pos+size]
        if len(chunk) != size: raise ValueError("Truncated NSP chunk")
        if kind == b"HEDR":
            if size < 32: raise ValueError("Invalid NSP header")
            sr, expected = struct.unpack_from("<II", chunk, 20)
        if kind in (b"SDA_", b"SD_A"):
            if sr is None or not 8000 <= sr <= 192000 or size % 2:
                raise ValueError("Invalid mono PCM metadata")
            pcm = np.frombuffer(chunk, dtype="<i2").copy()
            if expected != len(pcm): raise ValueError("NSP sample count mismatch")
            return pcm, sr
        pos += size + (size % 2)
    raise ValueError("No supported mono audio chunk")

def download_one(row, info, url):
    path = DATA / "audio" / f"{int(row.record_id)}-a_n.wav"
    try:
        if not path.exists():
            end = info["offset"] + info["compressed"] + 1023
            r = get(url, headers={"Range": f'bytes={info["offset"]}-{end}'})
            if r.status_code != 206 or r.content[:4] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP range response")
            n, extra = struct.unpack_from("<HH", r.content, 26)
            body = r.content[30+n+extra:30+n+extra+info["compressed"]]
            content = zlib.decompress(body, -15) if info["method"] == 8 else body
            if len(content) != info["size"] or zlib.crc32(content) != info["crc"]:
                raise ValueError("ZIP CRC/size mismatch")
            pcm, sr = decode_nsp(content)
            sf.write(path, pcm, sr, subtype="PCM_16")
        samples, sr = sf.read(path)
        duration = len(samples)/sr
        if duration < 1: raise ValueError("Shorter than one second")
        if not np.isfinite(samples).all() or np.sqrt(np.mean(samples**2)) < 1e-7:
            raise ValueError("Silent or nonfinite")
        return dict(record_id=int(row.record_id), speaker_id=int(row.speaker_id),
                    class_name=row.class_name, diagnoses=row.diagnoses, sex=row.sex,
                    duration=duration, sample_rate=sr, path=path.relative_to(ROOT).as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), error="")
    except Exception as exc:
        return dict(record_id=int(row.record_id), speaker_id=int(row.speaker_id),
                    class_name=row.class_name, error=str(exc))

def download():
    candidates, audio, archive = audit()
    (DATA / "audio").mkdir(exist_ok=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download_one, row, audio[row.record_id], archive["links"]["self"])
                   for row in candidates.itertuples()]
        for future in cf.as_completed(futures):
            rows.append(future.result())
            if len(rows) % 100 == 0:
                print(f"Downloaded/checked {len(rows)}/{len(candidates)}; failures {sum(bool(r['error']) for r in rows)}", flush=True)
    table = pd.DataFrame(rows).sort_values("record_id")
    table.to_csv(DATA / "download_audit.csv", index=False)
    valid = table[table.error == ""].copy()
    # Exact decoded duplicates cannot appear as independent speakers.
    valid = valid.drop_duplicates("sha256")
    counts = valid.groupby("class_name").speaker_id.nunique()
    weak = set(counts[counts < 20].index) - {"Healthy", "Other pathology"}
    valid.loc[valid.class_name.isin(weak), "class_name"] = "Other pathology"
    assert not valid.speaker_id.duplicated().any()
    valid.to_csv(DATA / "manifest.csv", index=False)
    final = valid.groupby("class_name").agg(recordings=("record_id", "size"), unique_speakers=("speaker_id", "nunique"))
    final.to_csv(RESULTS / "class_audit.csv")
    print(final.to_string(), flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "download"])
    args = parser.parse_args()
    audit() if args.stage == "audit" else download()
