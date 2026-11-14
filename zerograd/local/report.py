"""The classification report (precision, recall, F1 and support per class) from a confusion matrix."""

from __future__ import annotations

import numpy as np
import pandas as pd


def from_confusion(cm: np.ndarray, classes: list) -> pd.DataFrame:
    """scikit-learn's classification_report as a DataFrame. Rows: each class, then accuracy, macro avg, weighted avg.

    `cm[i, j]` counts rows of class i predicted as class j.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    total = support.sum()
    rows = {str(c): [p, r, f, int(s)] for c, p, r, f, s in zip(classes, precision, recall, f1, support)}
    acc = tp.sum() / total if total else 0.0
    w = support / total if total else np.zeros_like(support)
    rows["accuracy"] = [np.nan, np.nan, acc, int(total)]
    rows["macro avg"] = [precision.mean(), recall.mean(), f1.mean(), int(total)]
    rows["weighted avg"] = [(precision * w).sum(), (recall * w).sum(), (f1 * w).sum(), int(total)]
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["precision", "recall", "f1-score", "support"])
    df["support"] = df["support"].astype(int)
    return df


def binary_confusion(counts: dict, threshold: float) -> np.ndarray:
    """The 2x2 confusion matrix at the saved cut nearest to `threshold` (cuts every 0.01 on the holdout)."""
    t = np.asarray(counts["t"], dtype=np.float64)
    i = int(np.argmin(np.abs(t - threshold)))
    tp, fp, fn, tn = (float(np.asarray(counts[k])[i]) for k in ("tp", "fp", "fn", "tn"))
    return np.array([[tn, fp], [fn, tp]])
