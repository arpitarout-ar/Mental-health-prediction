"""Exploratory analysis: label stats and which features correlate with PHQ-8.

    python analysis.py   ->  outputs/label_distribution.png, outputs/feature_correlations.csv, outputs/top_features.png
"""
import matplotlib
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CACHE, OUT
from data_utils import load_labels


def main():
    lab = load_labels()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for s, g in lab.groupby("split"):
        ax[0].hist(g.PHQ8_Score, bins=range(0, 26, 2), alpha=0.5, label=f"{s} (n={len(g)})")
    ax[0].set_xlabel("PHQ-8 score"); ax[0].legend(); ax[0].set_title("Score distribution")
    if "Gender" in lab:
        lab.groupby("Gender").PHQ8_Binary.mean().rename({0: "female", 1: "male"}).plot.bar(ax=ax[1], rot=0)
        ax[1].set_ylabel("share depressed"); ax[1].set_title("Prevalence by gender")
    plt.tight_layout(); plt.savefig(OUT / "label_distribution.png", dpi=150); plt.close()
    print(lab.groupby("split").PHQ8_Binary.agg(["count", "mean"]).rename(columns={"mean": "share_depressed"}))

    rows = []
    for name in ("text_ling", "audio", "visual"):
        p = CACHE / f"{name}.csv"
        if not p.exists():
            continue
        X = pd.read_csv(p, index_col=0).join(lab.PHQ8_Score, how="inner")
        for c in X.columns.drop("PHQ8_Score"):
            ok = X[[c, "PHQ8_Score"]].dropna()
            if len(ok) > 10 and ok[c].nunique() > 1:
                r, pv = spearmanr(ok[c], ok.PHQ8_Score)
                rows.append({"modality": name, "feature": c, "spearman_r": r, "p_value": pv, "n": len(ok)})
    df = pd.DataFrame(rows).sort_values("p_value")
    df.to_csv(OUT / "feature_correlations.csv", index=False)
    top = df.head(15).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top.modality + ": " + top.feature, top.spearman_r)
    plt.xlabel("Spearman r with PHQ-8 score"); plt.title("Top-15 features (uncorrected p)")
    plt.tight_layout(); plt.savefig(OUT / "top_features.png", dpi=150); plt.close()
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
