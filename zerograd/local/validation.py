"""Cross-validation: every model on k folds instead of one holdout, split at random, by group or by time.

A single holdout answers "which model is best on these rows". Folds say how much that answer moves, and the
split says whether it holds where the model will be used:

- random (default): stratified k-fold.
- by group (`groups="customer_id"`): every row of a group lands on the same side, so a model is never scored
  on a customer it already saw. Use it when rows repeat per customer, store, patient or device.
- by time (`time="date"`): train on the past, score the next block, walk forward. Use it when the model will
  predict the future; a random split leaks it.

The group and time columns are left out of the features. Each fold's context is a stratified sample of its
training rows (up to the same row limit as the championship), and every test row is scored.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..models import ModelSpec
from . import championship as champ
from ._errors import Invalid
from .preprocess import Preprocessor


def binary_extras(y: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    """KS (largest gap between the score distributions of the two classes) and lift in the top 10% by score."""
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return {"ks": None, "lift_at_10": None}
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, score)
    top = np.argsort(-score, kind="stable")[: max(1, int(round(0.1 * len(y))))]
    return {"ks": float(np.max(tpr - fpr)), "lift_at_10": float(y[top].mean() / y.mean())}


def splits(y: np.ndarray, task: str, folds: int, *, groups: np.ndarray | None = None,
           times: np.ndarray | None = None, gap: int = 0, seed: int = 0) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """(how, [(train_idx, test_idx), ...])."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

    if folds < 2:
        raise Invalid("folds is at least 2.", code="invalid_folds")
    if groups is not None and times is not None:
        raise Invalid("Split by group or by time, not both.", code="invalid_split")
    s = champ.strata(y, task)
    if times is not None:
        order = np.argsort(times, kind="stable")
        tss = TimeSeriesSplit(n_splits=folds, gap=gap)
        return "time", [(order[tr], order[te]) for tr, te in tss.split(order)]
    if groups is not None:
        n_groups = len(pd.unique(groups))
        if n_groups < folds:
            raise Invalid(f"{n_groups} groups; at least {folds} are needed for {folds} folds.", code="invalid_folds")
        if task == champ.REGRESSION:
            gkf = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        else:
            gkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        return "group", list(gkf.split(np.zeros(len(y)), s, groups))
    smallest = int(np.bincount(s).min()) if len(s) else 0
    if smallest < folds:
        raise Invalid(f"The rarest class has {smallest} rows; {folds} folds need at least {folds}.",
                      code="invalid_folds")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return "random", list(skf.split(np.zeros(len(y)), s))


@dataclass
class CVResult:
    """Every model on every fold. `.summary` is the mean and spread per model; `.folds` has one row per fold."""

    folds: pd.DataFrame
    metric: str
    split: str
    task: str
    details: dict = field(default_factory=dict)

    @property
    def summary(self) -> pd.DataFrame:
        ok = self.folds[self.folds["error"].isna()] if "error" in self.folds else self.folds
        metric_cols = [c for c in ok.columns if c in champ.METRICS or c in ("ks", "lift_at_10")]
        g = ok.groupby("model", sort=False)
        out = g[metric_cols].mean()
        out[f"{self.metric}_std"] = g[self.metric].std(ddof=1)
        out["folds"] = g.size()
        out["seconds"] = g["seconds"].sum().round(2)
        higher = champ.METRICS[self.metric][1]
        first = [self.metric, f"{self.metric}_std", *[c for c in ("ks", "lift_at_10") if c in out]]
        out = out[first + [c for c in out.columns if c not in first]]
        out = out.sort_values(self.metric, ascending=not higher)
        out.attrs.update(metric=self.metric, split=self.split)
        return out

    @property
    def best(self) -> str:
        """The model with the best mean, baselines aside."""
        s = self.summary
        return str(next(m for m in s.index if not str(m).startswith("baseline:")))

    def __repr__(self) -> str:
        return (f"<CVResult · {self.split} split · {self.folds['fold'].nunique()} folds · {self.metric} · "
                f"best {self.best}>\n{self.summary.round(4).to_string()}")


def run_cv(pre: Preprocessor, df: pd.DataFrame, specs: list[ModelSpec], *, folds: int, groups: np.ndarray | None,
           times: np.ndarray | None, gap: int, metric: str, max_context_rows: int, chunk: int, seed: int,
           on_fold=None) -> CVResult:
    task = pre.task
    regression = task == champ.REGRESSION
    X, y = pre.transform(df), pre.encode_target(df[pre.target])
    n_classes = len(pre.classes)
    how, parts = splits(y, task, folds, groups=groups, times=times, gap=gap, seed=seed)
    rows: list[dict[str, Any]] = []
    for k, (tr, te) in enumerate(parts):
        ctx = tr[champ.stratified_sample(champ.strata(y[tr], task), max_context_rows, seed=seed + k)]
        base = {"fold": k, "train_rows": int(len(tr)), "context_rows": int(len(ctx)), "test_rows": int(len(te))}
        for spec in specs:
            if on_fold:
                on_fold(k, len(parts), spec)
            t0 = _time.monotonic()
            row = {"model": spec.name, **base}
            try:
                if regression:
                    fitted = spec.fit_regressor(X[ctx], y[ctx], serving=True)
                    pred = champ.predict_in_chunks(fitted, X[te], chunk, task)
                    row.update(champ.regression_metrics(y[te], pred))
                else:
                    fitted = spec.fit(X[ctx], y[ctx], n_classes, serving=True)
                    proba = champ.predict_in_chunks(fitted, X[te], chunk)
                    row.update(champ.metrics(y[te], proba, n_classes))
                    if n_classes == 2:
                        row.update(binary_extras(y[te], proba[:, 1]))
                row["error"] = None
            except Exception as exc:  # one model failing on one fold is a result, not a crash
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            row["seconds"] = round(_time.monotonic() - t0, 3)
            rows.append(row)
        for b in champ.baselines(y[ctx], y[te], n_classes, task):
            if b.model not in ("baseline:majority", "baseline:mean"):
                continue
            row = {"model": b.model, **base, **b.metrics, "error": None, "seconds": 0.0}
            if n_classes == 2:
                row.update({"ks": 0.0, "lift_at_10": 1.0})
            rows.append(row)
    frame = pd.DataFrame(rows)
    models = frame[~frame["model"].str.startswith("baseline:")]
    if models["error"].notna().all():
        raise Invalid("Every model failed on every fold: " + "; ".join(sorted(set(models["error"].dropna()))[:3]),
                      code="all_models_failed")
    return CVResult(folds=frame, metric=metric, split=how, task=task)
