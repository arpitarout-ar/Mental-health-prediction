"""Central paths and constants. Override locations with DAICWOZ_DATA / DAICWOZ_OUT env vars."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("DAICWOZ_DATA", ROOT / "data"))
RAW = DATA / "raw"          # extracted per-session files: raw/<pid>/<pid>_TRANSCRIPT.csv ...
LABELS = DATA / "labels"    # split CSVs
CACHE = DATA / "cache"      # engineered features
OUT = Path(os.environ.get("DAICWOZ_OUT", ROOT / "outputs"))
MODELS = OUT / "models"
for _p in (RAW, LABELS, CACHE, OUT, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"

# Only these files are pulled out of each session zip (the .wav and HOG files are huge and unused here).
KEEP = ("TRANSCRIPT.csv", "COVAREP.csv", "FORMANT.csv",
        "CLNF_AUs.txt", "CLNF_gaze.txt", "CLNF_pose.txt")

SPLIT_CSVS = {
    "train": "train_split_Depression_AVEC2017.csv",
    "dev": "dev_split_Depression_AVEC2017.csv",
    "test": "full_test_split.csv",          # test split *with* labels
}
EXTRA_FILES = ["test_split_Depression_AVEC2017.csv", "DAICWOZDepression_Documentation_AVEC2017.pdf"]

SEED = 42
