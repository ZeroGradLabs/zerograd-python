"""Pick the score cut of a binary capability for what the business wants, not for 0.5.

    cap.optimize_threshold("cost", cost_fp=1, cost_fn=20)      # missing a fraud costs 20 times a false alarm
    cap.optimize_threshold("recall", min_precision=0.8)        # catch the most while 8 in 10 alerts are right

Read off the championship holdout (confusion counts at every cut 0.01 to 0.99, saved with the capability),
or off rows you pass with the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._errors import Invalid

OBJECTIVES = ("f1", "cost", "youden", "recall", "precision")
CUTS = np.round(np.arange(1, 100) / 100, 2)


@dataclass
class ThresholdResult:
    threshold: float
    objective: str
    metrics: dict
    table: pd.DataFrame  # one row per cut: tp, fp, fn, tn, precision, recall, f1, fpr, cost, flagged
    source: str

    def __repr__(self) -> str:
        m = self.metrics
        return (f"<ThresholdResult · {self.objective} · threshold {self.threshold:.2f} · precision {m['precision']:.3f} "
                f"· recall {m['recall']:.3f} · flags {m['flagged']:.1%} of rows · from the {self.source}>")


def table(counts: dict, cost_fp: float, cost_fn: float) -> pd.DataFrame:
    df = pd.DataFrame({k: np.asarray(counts[k], dtype=np.float64) for k in ("tp", "fp", "fn", "tn")},
                      index=pd.Index(np.asarray(counts["t"], dtype=np.float64), name="threshold"))
    n = df.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        df["precision"] = np.where(df.tp + df.fp > 0, df.tp / (df.tp + df.fp), 1.0)
        df["recall"] = np.where(df.tp + df.fn > 0, df.tp / (df.tp + df.fn), 0.0)
        df["f1"] = np.where(df.precision + df.recall > 0,
                            2 * df.precision * df.recall / (df.precision + df.recall), 0.0)
        df["fpr"] = np.where(df.fp + df.tn > 0, df.fp / (df.fp + df.tn), 0.0)
    df["cost"] = cost_fp * df.fp + cost_fn * df.fn
    df["flagged"] = (df.tp + df.fp) / n
    return df


def counts_from_scores(y: np.ndarray, score: np.ndarray) -> dict:
    hit = np.asarray(score)[None, :] >= CUTS[:, None]
    pos = np.asarray(y).astype(bool)[None, :]
    return {"t": CUTS.tolist(), "tp": (hit & pos).sum(axis=1), "fp": (hit & ~pos).sum(axis=1),
            "fn": (~hit & pos).sum(axis=1), "tn": (~hit & ~pos).sum(axis=1)}


def optimize(objective: str, *, counts: dict | None = None, y: np.ndarray | None = None,
             score: np.ndarray | None = None, cost_fp: float = 1.0, cost_fn: float = 1.0,
             min_precision: float | None = None, min_recall: float | None = None,
             source: str = "holdout") -> ThresholdResult:
    if objective not in OBJECTIVES:
        raise Invalid(f"Unknown objective '{objective}'. Use one of: {', '.join(OBJECTIVES)}.", code="invalid_objective")
    if cost_fp < 0 or cost_fn < 0 or (objective == "cost" and cost_fp + cost_fn == 0):
        raise Invalid("cost_fp and cost_fn are non-negative, and not both zero.", code="invalid_objective")
    if objective == "recall" and min_precision is None:
        raise Invalid('objective="recall" needs min_precision (e.g. 0.8).', code="invalid_objective")
    if objective == "precision" and min_recall is None:
        raise Invalid('objective="precision" needs min_recall (e.g. 0.5).', code="invalid_objective")
    if counts is None:
        assert y is not None and score is not None
        if not 0 < int(np.sum(y)) < len(y):
            raise Invalid("The rows need both classes to pick a threshold.", code="invalid_threshold")
        counts = counts_from_scores(y, score)
    df = table(counts, cost_fp, cost_fn)
    ok = pd.Series(True, index=df.index)
    if min_precision is not None:
        ok &= df.precision >= min_precision
    if min_recall is not None:
        ok &= df.recall >= min_recall
    if not ok.any():
        raise Invalid("No threshold meets the constraint on these rows; loosen min_precision or min_recall.",
                      code="invalid_threshold")
    cand = df[ok]
    key = {"f1": cand.f1, "cost": -cand.cost, "youden": cand.recall - cand.fpr, "recall": cand.recall,
           "precision": cand.precision}[objective]
    best = float(key.idxmax())  # the lowest cut among ties: idxmax returns the first
    row = df.loc[best]
    metrics = {k: float(row[k]) for k in ("precision", "recall", "f1", "fpr", "cost", "flagged")}
    return ThresholdResult(threshold=best, objective=objective, metrics=metrics, table=df, source=source)
