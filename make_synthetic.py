"""Generate a tiny FAKE dataset with the same file layout as DAIC-WOZ, for smoke-testing the pipeline only.

    DAICWOZ_DATA=/tmp/syn/data DAICWOZ_OUT=/tmp/syn/out python make_synthetic.py
The numbers produced from it are meaningless - it exists so you can verify the code runs end-to-end.
"""
import numpy as np
import pandas as pd

from config import LABELS, RAW, SPLIT_CSVS

rng = np.random.default_rng(0)
NEUTRAL = "work family weekend friends travel food music movie school city sleep morning today really pretty good".split()
SAD = "tired alone nothing never hard sad worried empty hopeless problem hate".split()
HAPPY = "love great fun enjoy happy wonderful excited friends".split()
AUS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU09_r", "AU10_r", "AU12_r"]


def sentence(dep):
    n = rng.integers(2, 9) if dep else rng.integers(5, 20)
    pool = NEUTRAL + (SAD * 2 if dep else HAPPY * 2)
    return "i " + " ".join(rng.choice(pool, n)) if rng.random() < 0.5 + 0.2 * dep else " ".join(rng.choice(pool, n))


def make_session(pid, dep, dur=90):
    d = RAW / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    rows, t = [], 0.0
    while t < dur:
        rows.append((t, t + 1.5, "Ellie", "how are you doing today")); t += 1.5 + rng.uniform(0.2, 1.5 + 2 * dep)
        L = rng.uniform(1, 3) if dep else rng.uniform(2, 7)
        rows.append((t, t + L, "Participant", sentence(dep))); t += L + 0.3
    pd.DataFrame(rows, columns=["start_time", "stop_time", "speaker", "value"]).to_csv(
        d / f"{pid}_TRANSCRIPT.csv", sep="\t", index=False)
    n = int(dur * 100)
    cov = rng.normal(0, 1, (n, 74)) + dep * 0.25
    cov[:, 1] = (rng.random(n) < 0.6).astype(float)
    pd.DataFrame(cov).to_csv(d / f"{pid}_COVAREP.csv", header=False, index=False)
    pd.DataFrame(rng.normal(500, 100, (n, 5))).to_csv(d / f"{pid}_FORMANT.csv", header=False, index=False)
    m = int(dur * 30)
    base = {"frame": np.arange(m), "timestamp": np.arange(m) / 30, "confidence": 0.95, "success": 1}
    au = pd.DataFrame({**base, **{c: np.abs(rng.normal(0.5 - 0.15 * dep, 0.3, m)) for c in AUS}})
    gaze = pd.DataFrame({**base, **{c: rng.normal(0, 0.1 + 0.05 * dep, m) for c in ["x0", "y0", "z0", "x1", "y1", "z1"]}})
    pose = pd.DataFrame({**base, **{c: np.cumsum(rng.normal(0, 0.2 - 0.05 * dep, m)) for c in ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]}})
    for name, df in (("AUs", au), ("gaze", gaze), ("pose", pose)):
        df.columns = [("" if i == 0 else " ") + c for i, c in enumerate(df.columns)]   # real files have padded headers
        df.to_csv(d / f"{pid}_CLNF_{name}.txt", index=False)


def main():
    sizes = {"train": 44, "dev": 16, "test": 18}
    pid = 300
    for split, n in sizes.items():
        recs = []
        for _ in range(n):
            dep = int(rng.random() < 0.33)
            score = int(np.clip(rng.normal(14 if dep else 4, 3), 0, 24))
            make_session(pid, dep)
            recs.append({"Participant_ID": pid, "PHQ8_Binary": int(score >= 10), "PHQ8_Score": score, "Gender": int(rng.random() < .5)})
            pid += 1
        df = pd.DataFrame(recs)
        if split == "test":
            df = df.rename(columns={"PHQ8_Binary": "PHQ_Binary", "PHQ8_Score": "PHQ_Score"})
        df.to_csv(LABELS / SPLIT_CSVS[split], index=False)
    print("synthetic data written")


if __name__ == "__main__":
    main()
