"""Label loading + session-path helpers."""
import pandas as pd
from config import RAW, LABELS, SPLIT_CSVS


def session_file(pid, suffix):
    return RAW / str(pid) / f"{pid}_{suffix}"


def load_split(name):
    path = LABELS / SPLIT_CSVS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python download_data.py --labels-only` first")
    df = pd.read_csv(path)
    # the full test file uses PHQ_Binary / PHQ_Score; train/dev use PHQ8_*
    df = df.rename(columns={"PHQ_Binary": "PHQ8_Binary", "PHQ_Score": "PHQ8_Score"})
    missing = {"Participant_ID", "PHQ8_Binary", "PHQ8_Score"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {missing}; found {list(df.columns)}")
    df["Participant_ID"] = df["Participant_ID"].astype(int)
    df["split"] = name
    return df


def load_labels():
    """All labelled sessions as one frame indexed by Participant_ID."""
    return pd.concat([load_split(s) for s in SPLIT_CSVS], ignore_index=True).set_index("Participant_ID")


def available(ids, suffix="TRANSCRIPT.csv"):
    return [i for i in ids if session_file(i, suffix).exists()]
