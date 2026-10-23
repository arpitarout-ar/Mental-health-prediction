"""Pipelines, metrics and helpers shared by train.py."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SafeKBest(BaseEstimator, TransformerMixin):
    """SelectKBest that tolerates k > n_features; k=None keeps everything."""

    def __init__(self, k=30, task="clf"):
        self.k, self.task = k, task

    def fit(self, X, y):
        k = X.shape[1] if self.k is None else min(self.k, X.shape[1])
        self.sel_ = SelectKBest(f_classif if self.task == "clf" else f_regression, k=k).fit(X, y)
        return self

    def transform(self, X):
        return self.sel_.transform(X)


def tabular(task):
    est = (("clf", LogisticRegression(class_weight="balanced", max_iter=5000)) if task == "clf"
           else ("reg", Ridge()))
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("sel", SafeKBest(task=task)), est])
    grid = {"sel__k": [10, 30, None]}
    grid.update({"clf__C": [0.01, 0.1, 1.0]} if task == "clf" else {"reg__alpha": [10, 100, 1000]})
    return pipe, grid


def tfidf(task):
    est = (("clf", LogisticRegression(class_weight="balanced", max_iter=5000)) if task == "clf"
           else ("reg", Ridge()))
    pipe = Pipeline([("tf", TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True)), est])
    grid = {"clf__C": [0.1, 1.0, 10.0]} if task == "clf" else {"reg__alpha": [0.3, 1.0, 3.0]}
    return pipe, grid


def best_threshold(y, p):
    ts = np.linspace(0.2, 0.8, 61)
    return float(ts[np.argmax([f1_score(y, p >= t, average="macro") for t in ts])])


def cls_metrics(y, p, thr=0.5):
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    return {"macro_f1": f1_score(y, pred, average="macro"), "f1_depressed": f1_score(y, pred, pos_label=1),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "accuracy": accuracy_score(y, pred), "auc": roc_auc_score(y, p) if len(set(y)) > 1 else np.nan}


def reg_metrics(y, yhat):
    yhat = np.clip(yhat, 0, 24)
    return {"mae": mean_absolute_error(y, yhat), "rmse": mean_squared_error(y, yhat) ** 0.5}


def bootstrap_ci(y, p, thr, n=300, seed=0):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), (np.asarray(p) >= thr).astype(int)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            vals.append(f1_score(y[i], pred[i], average="macro"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def confusion(y, p, thr):
    return confusion_matrix(y, (np.asarray(p) >= thr).astype(int), labels=[0, 1])
