"""Turn raw sessions into cached feature tables + per-utterance arrays for the deep model.

    python build_features.py [--embed auto|st|hash]
Outputs (data/cache): text_ling.csv, text_emb.csv, audio.csv, visual.csv, texts.csv, utt/<pid>.npz, meta.json
"""
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import features as F
from config import CACHE, SPLIT_CSVS
from data_utils import load_split, session_file

REQUIRED = {"text": ["TRANSCRIPT.csv"], "audio": ["COVAREP.csv", "FORMANT.csv"],
            "visual": ["CLNF_AUs.txt", "CLNF_gaze.txt", "CLNF_pose.txt"]}


def has(pid, mod):
    return all(session_file(pid, s).exists() for s in REQUIRED[mod])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="auto", choices=["auto", "st", "hash"])
    a = ap.parse_args()

    ids = sorted(pd.concat([load_split(k) for k in SPLIT_CSVS]).Participant_ID.unique())
    ids = [i for i in ids if has(i, "text")]
    print(f"{len(ids)} sessions with transcripts")
    (CACHE / "utt").mkdir(exist_ok=True)

    ling, emb, aud, vis, texts = {}, {}, {}, {}, {}
    backend = None
    for pid in tqdm(ids, desc="features"):
        try:
            df = F.read_transcript(session_file(pid, "TRANSCRIPT.csv"))
            p = F.participant_utterances(df)
            segs = list(zip(p.start.astype(float), p.stop.astype(float)))
            if len(p) == 0:
                tqdm.write(f"[warn] {pid}: no participant speech, skipped")
                continue
            ling[pid] = F.text_features(df)
            texts[pid] = " ".join(p.text)
            E, backend = F.embed(list(p.text), a.embed)
            emb[pid] = E.mean(0)

            utt = {"emb": E.astype(np.float32)}
            if has(pid, "audio"):
                aud[pid] = F.audio_features(pid, segs)
                utt["audio"] = F.utterance_audio(F.load_covarep(pid), segs)
            if has(pid, "visual"):
                vis[pid] = F.visual_features(pid, segs)
                utt["visual"] = F.utterance_au(pid, segs)
            np.savez_compressed(CACHE / "utt" / f"{pid}.npz", **utt)
        except Exception as e:
            tqdm.write(f"[error] {pid}: {type(e).__name__}: {e}")

    def save(d, name):
        df = pd.DataFrame.from_dict(d, orient="index")
        df.index.name = "Participant_ID"
        df.to_csv(CACHE / name)
        print(f"{name}: {df.shape}")

    save(ling, "text_ling.csv")
    save({k: dict(enumerate(v)) for k, v in emb.items()}, "text_emb.csv")
    save(aud, "audio.csv")
    save(vis, "visual.csv")
    pd.Series(texts, name="text").rename_axis("Participant_ID").to_csv(CACHE / "texts.csv")
    (CACHE / "meta.json").write_text(json.dumps({"embed_backend": backend}))


if __name__ == "__main__":
    main()
