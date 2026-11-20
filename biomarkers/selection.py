"""Cross-validated classification and stability-based gene ranking.

All preprocessing (variance filter, scaling) is fitted inside each training
fold, so held-out samples never influence feature selection or scaling.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _l1_logistic_regression(seed):
    """L1-penalised logistic regression across scikit-learn versions."""
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    l1 = {"l1_ratio": 1.0} if (major, minor) >= (1, 8) else {"penalty": "l1"}
    return LogisticRegression(solver="liblinear", C=0.5, max_iter=5000, random_state=seed, **l1)


def make_classifiers(seed=0):
    return {
        "l1_logreg": _l1_logistic_regression(seed),
        "random_forest": RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
    }


def top_variance_features(X_train, n):
    """Indices of the n most variable features, computed on training data only."""
    n = min(n, X_train.shape[1])
    return np.argsort(X_train.var(axis=0))[::-1][:n]


def feature_importance(model):
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        return np.abs(clf.coef_).ravel()
    return clf.feature_importances_


@dataclass
class Result:
    metrics: pd.DataFrame       # one row per (repeat, fold)
    ranking: pd.DataFrame       # one row per feature


def evaluate(X, y, estimator, n_splits=3, n_repeats=10, top_k=20, prefilter=2000, seed=0):
    """Repeated stratified CV; returns fold metrics and a stability ranking.

    For every fold the model is fitted on the training part, scored on the
    held-out part, and its top_k features (by |coefficient| or impurity
    importance) are recorded. `selection_frequency` is the fraction of folds in
    which a feature made the top_k, which is a measure of how stable the selection is.
    """
    X_values = X.to_numpy(dtype=float)
    names = np.asarray(X.columns)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    rows, counts, importance_sum = [], pd.Series(0.0, index=names), pd.Series(0.0, index=names)
    n_folds = 0
    for fold, (train, test) in enumerate(cv.split(X_values, y)):
        keep = top_variance_features(X_values[train], prefilter)
        model = Pipeline([("scale", StandardScaler()), ("clf", clone(estimator))])
        model.fit(X_values[train][:, keep], y[train])

        proba = model.predict_proba(X_values[test][:, keep])[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "fold": fold,
            "accuracy": accuracy_score(y[test], pred),
            "balanced_accuracy": balanced_accuracy_score(y[test], pred),
            "roc_auc": roc_auc_score(y[test], proba) if len(np.unique(y[test])) == 2 else np.nan,
        })

        imp = feature_importance(model)
        kept_names = names[keep]
        importance_sum[kept_names] += imp
        top = kept_names[np.argsort(imp)[::-1][:top_k]]
        counts[top] += 1
        n_folds += 1

    ranking = pd.DataFrame({
        "selection_frequency": counts / n_folds,
        "mean_importance": importance_sum / n_folds,
    }).sort_values(["selection_frequency", "mean_importance"], ascending=False)
    ranking = ranking[ranking["selection_frequency"] > 0]
    return Result(metrics=pd.DataFrame(rows), ranking=ranking)


def summarize_metrics(metrics):
    return metrics[["accuracy", "balanced_accuracy", "roc_auc"]].agg(["mean", "std"]).T
