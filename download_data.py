"""Download DAIC-WOZ sessions, keep only the modalities we use, delete the zips.

    python download_data.py --labels-only            # just split CSVs
    python download_data.py --limit 10               # first 10 sessions (quick trial)
    python download_data.py                          # everything (~85 GB transferred, ~1 GB kept)

Each session zip is ~300-980 MB, so it is streamed to a temp file, the needed members are
extracted, and the zip is removed. Re-running skips sessions that are already complete.
Use --user/--password if the server asks for basic-auth credentials.
Note: DAIC-WOZ is distributed under a USC licence agreement - make sure yours covers your use.
"""
import argparse
import os
import tempfile
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

from config import BASE_URL, KEEP, LABELS, RAW, SPLIT_CSVS, EXTRA_FILES
from data_utils import load_split


def get(session, url, dest, desc=None):
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def fetch_labels(session):
    for name in list(SPLIT_CSVS.values()) + EXTRA_FILES:
        dest = LABELS / name
        if not dest.exists():
            print("downloading", name)
            get(session, BASE_URL + name, dest)


def complete(pid):
    d = RAW / str(pid)
    return all((d / f"{pid}_{k}").exists() for k in KEEP)


def fetch_session(session, pid):
    if complete(pid):
        return "skip"
    out = RAW / str(pid)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, f"{pid}_P.zip")
        get(session, f"{BASE_URL}{pid}_P.zip", zpath, desc=f"{pid}_P.zip")
        with zipfile.ZipFile(zpath) as z:
            for member in z.namelist():
                base = os.path.basename(member)
                if base.endswith(KEEP):
                    with z.open(member) as src, open(out / base, "wb") as dst:
                        dst.write(src.read())
    return "ok" if complete(pid) else "partial"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", type=int, nargs="*", default=None)
    ap.add_argument("--user")
    ap.add_argument("--password")
    a = ap.parse_args()

    s = requests.Session()
    if a.user:
        s.auth = (a.user, a.password or "")
    fetch_labels(s)
    if a.labels_only:
        return

    ids = a.ids or sorted(pd.concat([load_split(k) for k in SPLIT_CSVS]).Participant_ID.unique())
    if a.limit:
        ids = ids[: a.limit]
    for pid in tqdm(ids, desc="sessions"):
        try:
            status = fetch_session(s, pid)
            if status == "partial":
                tqdm.write(f"[warn] {pid}: some expected files missing from zip")
        except Exception as e:  # keep going; re-run to retry
            tqdm.write(f"[error] {pid}: {e}")


if __name__ == "__main__":
    main()
