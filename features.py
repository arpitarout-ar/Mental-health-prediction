"""Feature extraction for the three DAIC-WOZ modalities.

Everything is computed on *participant speech only* (Ellie's prompts are excluded: her follow-up
questions depend on earlier answers and leak label information).
"""
import re
import zlib

import numpy as np
import pandas as pd

from data_utils import session_file

TAGS = re.compile(r"<[^>]*>|\[[^\]]*\]")
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}
NEG = set("""sad depressed depression anxious anxiety angry hate lonely alone tired bad worse worst hard difficult stress
stressed hurt pain afraid scared worry worried cry crying empty lost hopeless guilty ashamed failure terrible awful
miserable nothing never nobody unhappy sick fear struggle struggling upset mad annoyed trouble problem problems
divorce death died""".split())
POS = set("""happy good great love loved enjoy enjoyed fun wonderful excited amazing nice best better fine glad relaxed proud
fantastic awesome grateful laugh friends family beautiful favorite comfortable easy""".split())
ABSOLUTIST = set("""always never completely totally entire everything nothing constantly absolutely all every whole definitely
nobody everyone anything""".split())
FILLERS = {"um", "uh", "umm", "uhh", "er", "erm", "hmm", "mm"}

# ----------------------------------------------------------------------------- transcript


def read_transcript(path):
    df = pd.read_csv(path, sep="\t")
    if df.shape[1] == 1:
        df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"start_time": "start", "stop_time": "stop"})
    df["speaker"] = df["speaker"].astype(str).str.strip().str.lower()
    df["value"] = df["value"].fillna("").astype(str)
    return df[["start", "stop", "speaker", "value"]].sort_values("start").reset_index(drop=True)


def clean(s):
    s = TAGS.sub(" ", s.lower()).replace("scrubbed_entry", " ").replace("_", " ")
    s = re.sub(r"\bx{2,}\b", " ", s)
    s = re.sub(r"[^a-z' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def participant_utterances(df):
    p = df[df.speaker == "participant"].copy()
    p["text"] = p["value"].map(clean)
    return p[p.text.str.len() > 0].reset_index(drop=True)


def segments(df):
    p = participant_utterances(df)
    return list(zip(p.start.astype(float), p.stop.astype(float)))


def response_latencies(df):
    lat, last_ellie_stop, prev = [], None, None
    for r in df.itertuples():
        if r.speaker == "ellie":
            last_ellie_stop = r.stop
        elif r.speaker == "participant" and prev == "ellie" and last_ellie_stop is not None:
            d = r.start - last_ellie_stop
            if 0 <= d <= 30:
                lat.append(d)
        prev = r.speaker
    return lat


def text_features(df):
    p = participant_utterances(df)
    toks = [t.split() for t in p.text]
    lens = np.array([len(t) for t in toks]) if toks else np.array([0])
    words = [w for t in toks for w in t]
    n = max(len(words), 1)
    dur = float((p.stop - p.start).sum()) if len(p) else np.nan
    lat = response_latencies(df)
    raw = " ".join(df[df.speaker == "participant"].value.str.lower())
    frac = lambda lex: sum(w in lex for w in words) / n
    return {
        "n_utterances": len(toks),
        "n_words": len(words),
        "words_per_utt_mean": lens.mean(),
        "words_per_utt_std": lens.std(),
        "short_utt_ratio": float((lens <= 3).mean()),
        "type_token_ratio": len(set(words)) / n,
        "speaking_time": dur,
        "speech_rate_wps": len(words) / dur if dur and dur > 0 else np.nan,
        "latency_mean": float(np.mean(lat)) if lat else np.nan,
        "latency_median": float(np.median(lat)) if lat else np.nan,
        "first_person_ratio": frac(FIRST_PERSON),
        "negative_ratio": frac(NEG),
        "positive_ratio": frac(POS),
        "absolutist_ratio": frac(ABSOLUTIST),
        "filler_ratio": frac(FILLERS),
        "laughter_per_100w": 100 * raw.count("laughter") / n,
        "dont_know_per_100w": 100 * (raw.count("don't know") + raw.count("dont know")) / n,
    }


# ----------------------------------------------------------------------------- embeddings


def hash_embed(texts, dim=256):
    """Deterministic signed bag-of-words hashing; offline fallback for sentence embeddings."""
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.split():
            h = zlib.crc32(w.encode())
            X[i, h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    return X


_ST = None
_WARNED = False


def embed(texts, backend="auto"):
    """Returns (embeddings, backend_used). backend: auto | st | hash."""
    global _ST, _WARNED
    if not texts:
        return np.zeros((0, 256), dtype=np.float32), "hash"
    if backend in ("auto", "st"):
        try:
            if _ST is None:
                from sentence_transformers import SentenceTransformer
                _ST = SentenceTransformer("all-MiniLM-L6-v2")
            return _ST.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True), "st"
        except Exception as e:
            if backend == "st":
                raise
            if not _WARNED:
                print(f"[info] sentence-transformers unavailable ({type(e).__name__}); using hashed embeddings")
                _WARNED = True
    return hash_embed(texts), "hash"


# ----------------------------------------------------------------------------- audio


def frame_mask(n, segs, fs):
    m = np.zeros(n, bool)
    for s, e in segs:
        a, b = max(int(s * fs), 0), min(int(e * fs), n)
        if b > a:
            m[a:b] = True
    return m


def _num(path, header=None):
    return pd.read_csv(path, header=header, skipinitialspace=True).apply(pd.to_numeric, errors="coerce")


def _stats(X, prefix):
    if len(X) == 0:
        mu = sd = np.full(X.shape[1], np.nan)
    else:
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    out = {f"{prefix}{i}_mean": v for i, v in enumerate(mu)}
    out.update({f"{prefix}{i}_std": v for i, v in enumerate(sd)})
    return out


def load_covarep(pid):
    cov = np.array(_num(session_file(pid, "COVAREP.csv")), dtype=float)
    cov[~np.isfinite(cov)] = np.nan
    return cov  # 100 Hz; col0 = F0, col1 = VUV flag


def audio_features(pid, segs):
    cov = load_covarep(pid)
    pm = frame_mask(len(cov), segs, 100)
    voiced = pm & (cov[:, 1] == 1)
    feats = {"voiced_ratio": voiced.sum() / max(pm.sum(), 1)}
    feats.update(_stats(np.delete(cov[voiced], 1, axis=1), "covarep"))
    fm = np.array(_num(session_file(pid, "FORMANT.csv")), dtype=float)
    fm[~np.isfinite(fm)] = np.nan
    v = np.zeros(len(fm), bool)
    k = min(len(fm), len(voiced))
    v[:k] = voiced[:k]
    feats.update(_stats(fm[v], "formant"))
    return feats


def utterance_audio(cov, segs):
    """Per-utterance mean of voiced COVAREP frames (73 dims, NaN->0)."""
    out = np.zeros((len(segs), cov.shape[1] - 1), np.float32)
    for i, (s, e) in enumerate(segs):
        a, b = max(int(s * 100), 0), min(int(e * 100), len(cov))
        seg = cov[a:b]
        seg = seg[seg[:, 1] == 1] if len(seg) else seg
        if len(seg):
            out[i] = np.nan_to_num(np.nanmean(np.delete(seg, 1, axis=1), 0))
    return out


# ----------------------------------------------------------------------------- visual


def read_clnf(pid, kind):
    df = pd.read_csv(session_file(pid, f"CLNF_{kind}.txt"), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def _ok(df, conf=0.8):
    ok = np.ones(len(df), bool)
    if "success" in df:
        ok &= df["success"].to_numpy() == 1
    if "confidence" in df:
        ok &= df["confidence"].to_numpy(float) > conf
    return ok


def time_mask(ts, segs):
    m = np.zeros(len(ts), bool)
    for s, e in segs:
        m[np.searchsorted(ts, s):np.searchsorted(ts, e)] = True
    return m


def au_columns(df):
    return [c for c in df.columns if c.startswith("AU") and c.endswith("_r")]


def visual_features(pid, segs):
    feats = {}
    for tag in ("AUs", "gaze", "pose"):
        df = read_clnf(pid, tag)
        ts = df["timestamp"].to_numpy(float)
        m = _ok(df) & time_mask(ts, segs)
        cols = au_columns(df) if tag == "AUs" else [c for c in df.columns if c not in ("frame", "timestamp", "confidence", "success")]
        X = df.loc[m, cols].to_numpy(float)
        for j, c in enumerate(cols):
            feats[f"{tag}_{c}_mean"] = X[:, j].mean() if len(X) else np.nan
            feats[f"{tag}_{c}_std"] = X[:, j].std() if len(X) else np.nan
            if tag == "pose":
                feats[f"{tag}_{c}_speed"] = np.abs(np.diff(X[:, j])).mean() if len(X) > 1 else np.nan
    return feats


def utterance_au(pid, segs):
    df = read_clnf(pid, "AUs")
    cols = au_columns(df)
    ts = df["timestamp"].to_numpy(float)
    ok = _ok(df)
    X = df[cols].to_numpy(float)
    out = np.zeros((len(segs), len(cols)), np.float32)
    for i, (s, e) in enumerate(segs):
        a, b = np.searchsorted(ts, s), np.searchsorted(ts, e)
        sel = ok[a:b]
        if sel.any():
            out[i] = np.nan_to_num(X[a:b][sel].mean(0))
    return out
