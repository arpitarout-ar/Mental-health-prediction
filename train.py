"""Train & evaluate unimodal, early-fusion and late-fusion depression models on DAIC-WOZ.

Protocol (per model)
  1. tune on TRAIN with 5-fold CV (macro-F1 / MAE), pick decision threshold from out-of-fold probabilities
  2. evaluate on the official DEV split
  3. refit on TRAIN+DEV with the same hyper-parameters, re-pick threshold, evaluate once on the TEST split
Targets: PHQ-8 binary (>=10) as classification, PHQ-8 total score (0-24) as regression.

    python train.py
Outputs -> outputs/: results.csv, test_predictions.csv, f1_comparison.png, roc_test.png, confusion_best.png,
                     models/text_models.joblib (used by app.py)
"""
import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

import models as M
from config import CACHE, MODELS, OUT, SEED
from data_utils import load_labels

TABULAR = {"Text (linguistic)": "text_ling", "Text (embeddings)": "text_emb", "Audio": "audio", "Visual": "visual"}


def load_tables():
    t = {}
    for key, f in [("text_ling", "text_ling.csv"), ("text_emb", "text_emb.csv"),
                   ("audio", "audio.csv"), ("visual", "visual.csv")]:
        p = CACHE / f
        if p.exists():
            df = pd.read_csv(p, index_col=0)
            if df.shape[1] > 0:
                t[key] = df
    t["tfidf"] = pd.read_csv(CACHE / "texts.csv", index_col=0)["text"]
    return t


def X_for(X, ids):
    X = X.reindex(ids)
    return X.fillna("").tolist() if isinstance(X, pd.Series) else X.to_numpy(float)


def cv_splits(y, n=5):
    n = int(min(n, np.bincount(y).min()))
    return list(StratifiedKFold(n, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))


def fit_stage(factory, X, lab, tr, te):
    ytr = lab.loc[tr, "PHQ8_Binary"].to_numpy(int)
    str_ = lab.loc[tr, "PHQ8_Score"].to_numpy(float)
    Xtr, Xte = X_for(X, tr), X_for(X, te)
    cv = cv_splits(ytr)

    pipe, grid = factory("clf")
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1).fit(Xtr, ytr)
    oof = cross_val_predict(clone(gs.best_estimator_), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]

    rpipe, rgrid = factory("reg")
    gr = GridSearchCV(rpipe, rgrid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1).fit(Xtr, str_)
    return {"p": gs.best_estimator_.predict_proba(Xte)[:, 1], "s": gr.predict(Xte), "oof": oof,
            "thr": M.best_threshold(ytr, oof), "cv_f1": gs.best_score_, "params": gs.best_params_,
            "est": gs.best_estimator_, "reg_est": gr.best_estimator_}


def run_model(factory, X, lab, ids):
    dev = fit_stage(factory, X, lab, ids["train"], ids["dev"])
    test = fit_stage(factory, X, lab, ids["train"] + ids["dev"], ids["test"])
    return {"dev": dev, "test": test}


def late_fusion(results, names, lab, ids):
    out = {}
    for stage in ("dev", "test"):
        rs = [results[n][stage] for n in names]
        out[stage] = {"p": np.mean([r["p"] for r in rs], 0), "s": np.mean([r["s"] for r in rs], 0),
                      "oof": np.mean([r["oof"] for r in rs], 0), "cv_f1": np.nan, "params": {"members": names}}
        fit_ids = ids["train"] if stage == "dev" else ids["train"] + ids["dev"]
        out[stage]["thr"] = M.best_threshold(lab.loc[fit_ids, "PHQ8_Binary"].to_numpy(int), out[stage]["oof"])
    return out


def evaluate(results, lab, ids):
    rows, preds = [], pd.DataFrame({"Participant_ID": ids["test"],
                                    "PHQ8_Binary": lab.loc[ids["test"], "PHQ8_Binary"].values,
                                    "PHQ8_Score": lab.loc[ids["test"], "PHQ8_Score"].values})
    for name, r in results.items():
        for split, stage in (("dev", "dev"), ("test", "test")):
            s = r[stage]
            y = lab.loc[ids[split], "PHQ8_Binary"].to_numpy(int)
            sc = lab.loc[ids[split], "PHQ8_Score"].to_numpy(float)
            row = {"model": name, "split": split, "n": len(y), **M.cls_metrics(y, s["p"], s["thr"]),
                   **M.reg_metrics(sc, s["s"]), "threshold": s["thr"], "cv_macro_f1_train": s["cv_f1"],
                   "params": json.dumps(s["params"], default=str)}
            row["f1_ci_low"], row["f1_ci_high"] = M.bootstrap_ci(y, s["p"], s["thr"])
            rows.append(row)
        preds[f"p_{name}"] = r["test"]["p"]
        preds[f"phq_{name}"] = np.clip(r["test"]["s"], 0, 24)
    return pd.DataFrame(rows), preds


def plots(res, results, lab, ids):
    piv = res.pivot(index="model", columns="split", values="macro_f1").loc[list(results)]
    ax = piv.plot.barh(figsize=(8, 0.6 * len(piv) + 2))
    ax.set_xlabel("Macro-F1"); ax.set_title("Depression detection (PHQ-8 >= 10)"); ax.axvline(0.5, ls=":", c="grey")
    plt.tight_layout(); plt.savefig(OUT / "f1_comparison.png", dpi=150); plt.close()

    y = lab.loc[ids["test"], "PHQ8_Binary"].to_numpy(int)
    plt.figure(figsize=(5.5, 5))
    for n, r in results.items():
        fpr, tpr, _ = roc_curve(y, r["test"]["p"])
        plt.plot(fpr, tpr, label=n)
    plt.plot([0, 1], [0, 1], "k:"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC - test split")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "roc_test.png", dpi=150); plt.close()

    best = res[res.split == "dev"].sort_values("macro_f1", ascending=False).model.iloc[0]
    cm = M.confusion(y, results[best]["test"]["p"], results[best]["test"]["thr"])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1], ["Healthy", "Depressed"]); ax.set_yticks([0, 1], ["Healthy", "Depressed"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{best}\n(selected on dev, shown on test)", fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "confusion_best.png", dpi=150); plt.close()
    return best


def main():
    tables = load_tables()
    lab = load_labels()
    have = set(tables["tfidf"].index)
    ids = {s: [i for i in lab.index[lab.split == s] if i in have] for s in ("train", "dev", "test")}
    print({k: len(v) for k, v in ids.items()}, "sessions with features")

    specs = {"TF-IDF text": (tables["tfidf"], M.tfidf)}
    for name, key in TABULAR.items():
        if key in tables:
            specs[name] = (tables[key], M.tabular)
    if sum(k in tables for k in TABULAR.values()) > 1:
        fused = pd.concat([tables[k].add_prefix(f"{k}__") for k in TABULAR.values() if k in tables], axis=1)
        specs["Early fusion (all)"] = (fused, M.tabular)

    results = {}
    for name, (X, factory) in specs.items():
        print("training", name)
        results[name] = run_model(factory, X, lab, ids)
    members = [n for n in TABULAR if n in results]
    if len(members) > 1:
        results["Late fusion (prob. avg)"] = late_fusion(results, members, lab, ids)

    res, preds = evaluate(results, lab, ids)
    res.to_csv(OUT / "results.csv", index=False)
    preds.to_csv(OUT / "test_predictions.csv", index=False)
    best = plots(res, results, lab, ids)

    if "text_ling" in tables and "text_emb" in tables:  # for the demo app
        meta = json.loads((CACHE / "meta.json").read_text())
        joblib.dump({"ling": results["Text (linguistic)"]["test"]["est"], "emb": results["Text (embeddings)"]["test"]["est"],
                     "thr": float(np.mean([results["Text (linguistic)"]["test"]["thr"], results["Text (embeddings)"]["test"]["thr"]])),
                     "ling_cols": list(tables["text_ling"].columns), "embed_backend": meta["embed_backend"]},
                    MODELS / "text_models.joblib")

    show = res[["model", "split", "macro_f1", "f1_depressed", "auc", "mae", "rmse"]].round(3)
    print("\n", show.to_string(index=False))
    print(f"\nBest on dev: {best}.  Outputs in {OUT}")


if __name__ == "__main__":
    main()
