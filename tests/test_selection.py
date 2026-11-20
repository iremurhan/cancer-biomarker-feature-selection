"""Synthetic-data tests: a planted signal must be found and ranked first."""
import numpy as np
import pandas as pd

from biomarkers.data import load_cumida
from biomarkers.selection import evaluate, make_classifiers, top_variance_features


def _synthetic(n=30, p=300, informative=("g0", "g1", "g2"), seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"g{i}" for i in range(p)])
    for g in informative:
        X[g] += 4.0 * y
    return X, y


def test_planted_genes_are_most_stable():
    X, y = _synthetic()
    for clf in make_classifiers().values():
        result = evaluate(X, y, clf, n_splits=3, n_repeats=3, top_k=5, prefilter=300)
        assert set(result.ranking.index[:3]) == {"g0", "g1", "g2"}
        assert result.metrics["balanced_accuracy"].mean() > 0.9


def test_variance_filter_uses_only_given_rows():
    X = np.zeros((4, 3))
    X[:, 0] = [1, -1, 0, 0]      # varies only in the first two rows
    X[:, 2] = [0, 0, 5, -5]      # varies only in the last two rows
    assert top_variance_features(X[:2], 1).tolist() == [0]
    assert top_variance_features(X[2:], 1).tolist() == [2]


def test_load_cumida(tmp_path):
    path = tmp_path / "toy.csv"
    pd.DataFrame({"samples": [1, 2, 3, 4], "type": ["normal", "tumoral", "normal", "tumoral"],
                  "p1": [0.1, 0.9, 0.2, 0.8], "p2": [1, 1, 1, 1]}).to_csv(path, index=False)
    X, y, classes = load_cumida(path)
    assert list(X.columns) == ["p1", "p2"]
    assert y.tolist() == [0, 1, 0, 1]
    assert classes == ["normal", "tumoral"]
