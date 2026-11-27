"""Checks that run before the championship, so a too-good holdout gets a second look.

* Leakage: a column that predicts the target almost perfectly on its own is usually known only after the outcome
  (a cancellation date, a chargeback flag). A shallow tree on that one column, scored on rows it didn't see, finds it.
* Duplicates: holdout rows that also appear, column for column, in the context make the holdout score optimistic.

Both only warn: the column might be legitimate, and you decide with `drop=[...]`.
"""

from __future__ import annotations

import numpy as np

LEAK_SCORE = 0.98  # ROC AUC (binary), balanced accuracy (multiclass) or R² (regression) of one column alone
DUPLICATE_SHARE = 0.10  # share of holdout rows with an identical row in the context
MAX_ROWS = 5_000


def _filled(col: np.ndarray) -> np.ndarray:
    """Missing values as their own value below everything else, so a tree can split on "missing"."""
    col = np.asarray(col, dtype=np.float64)
    bad = ~np.isfinite(col)
    if bad.all():
        return np.zeros((len(col), 1))
    low = np.nanmin(np.where(bad, np.nan, col)) - 1.0
    return np.where(bad, low, col).reshape(-1, 1)


def leaky_columns(X: np.ndarray, y: np.ndarray, names: list[str], task: str, seed: int = 0) -> list[dict]:
    """Columns that alone score at least LEAK_SCORE on rows the one-column model didn't see, best first."""
    from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    n = len(y)
    if n < 40 or X.shape[1] == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:MAX_ROWS]
    cut = int(len(idx) * 0.7)
    tr, te = idx[:cut], idx[cut:]
    regression = task == "regression"
    if not regression and (len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2):
        return []
    out = []
    for j, name in enumerate(names):
        col = _filled(X[:, j])
        try:
            if regression:
                m = DecisionTreeRegressor(max_leaf_nodes=32, min_samples_leaf=5, random_state=seed)
                m.fit(col[tr], y[tr])
                score, label = float(r2_score(y[te], m.predict(col[te]))), "R²"
            else:
                m = DecisionTreeClassifier(max_leaf_nodes=32, min_samples_leaf=5, random_state=seed)
                m.fit(col[tr], y[tr])
                if task == "binary":
                    score, label = float(roc_auc_score(y[te], m.predict_proba(col[te])[:, 1])), "ROC AUC"
                else:
                    score, label = float(balanced_accuracy_score(y[te], m.predict(col[te]))), "balanced accuracy"
        except ValueError:
            continue
        if score >= LEAK_SCORE:
            out.append({"code": "possible_leakage", "column": name, "score": round(score, 4), "metric": label,
                        "message": f"'{name}' alone predicts the target with {label} {score:.3f}. If it's only known "
                                   f"after the outcome, the holdout looks better than production will: drop it "
                                   f"with drop=['{name}']."})
    return sorted(out, key=lambda w: -w["score"])


def duplicate_share(X_ctx: np.ndarray, X_hold: np.ndarray) -> float:
    """Share of holdout rows whose features match some context row exactly (missing values count as equal)."""
    if len(X_hold) == 0 or len(X_ctx) == 0:
        return 0.0

    def keys(X: np.ndarray) -> set | list:
        Z = np.where(np.isnan(X), np.inf, X).astype(np.float64)
        return [r.tobytes() for r in np.ascontiguousarray(Z)]

    seen = set(keys(X_ctx))
    return float(np.mean([k in seen for k in keys(X_hold)]))


def duplicate_warning(share: float) -> dict | None:
    if share < DUPLICATE_SHARE:
        return None
    return {"code": "holdout_duplicates", "share": round(share, 4),
            "message": f"{share:.0%} of the holdout rows also appear, column for column, in the context, so the holdout "
                       f"score is optimistic. Drop exact duplicates, or check with zg.cross_validate(..., groups=...)."}
