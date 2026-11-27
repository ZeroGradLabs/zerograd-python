"""The championship: compare every model on the same holdout and keep the best.

Same procedure as the ZeroGrad platform, so a local result and a cloud result are comparable.

1. Hold out 20% of the labeled rows (stratified, fixed seed), capped by a stratified sample so huge
   histories don't turn evaluation into hours of scoring. Every holdout row is scored.
2. Build the context from the rest: all of it, or a stratified sample when it is bigger than the limit.
3. For each model: fit on the context, score the whole holdout, compute the task's metrics and the
   curves the dashboard plots (confusion matrix; ROC and precision-recall for binary tasks).
4. With two or more models that finished, add their equal-weight average (the ensemble) as one more candidate.
5. Champion = best value of the capability's selection metric (default ROC AUC, or RMSE for regression;
   ties: lower log loss). Regression splits and samples are stratified on quantile bins of the target.
6. The capability keeps a fresh context drawn from *all* labeled rows, so production predictions see
   the most history the limit allows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..models import FittedModel, ModelSpec
from ._errors import Invalid
from .preprocess import Preprocessor

Progress = Callable[..., None]  # (stage, fraction, message, eta_s=None)
# Seconds a model is expected to take on (context rows, holdout rows, features); and a hook to learn from what it took.
Estimate = Callable[[str, int, int, int], float]
Observe = Callable[[str, int, int, int, float], None]


REGRESSION = "regression"


def strata(y: np.ndarray, task: str | None = None, bins: int = 10) -> np.ndarray:
    """What to stratify on: the labels themselves, or for regression the target's quantile bin (up to `bins`)."""
    if task != REGRESSION:
        return y
    y = np.asarray(y, dtype=np.float64)
    if len(y) == 0:
        return np.zeros(0, dtype=np.int64)
    edges = np.unique(np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, y, side="right").astype(np.int64)


def stratified_sample(y: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Indices of a stratified sample of size n (every class keeps at least one row)."""
    if n >= len(y):
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.floor(counts / len(y) * n).astype(int))
    # Hand out the rounding remainder to the largest classes.
    while quota.sum() < n:
        room = counts - quota
        quota[np.argmax(room)] += 1
    while quota.sum() > n:
        quota[np.argmax(quota)] -= 1
    picked = []
    for cls, q in zip(classes, quota):
        idx = np.flatnonzero(y == cls)
        picked.append(rng.choice(idx, size=min(q, len(idx)), replace=False))
    return np.sort(np.concatenate(picked))


def holdout_split(y: np.ndarray, fraction: float, seed: int = 0, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, holdout_idx), stratified when every class has at least 2 rows.

    With max_rows, a bigger holdout is cut down to a stratified sample of that size (the rows left out
    are used by neither side, so the context still never sees a holdout row's neighbours twice)."""
    train, hold = _holdout_split(y, fraction, seed)
    if max_rows and len(hold) > max_rows:
        hold = hold[stratified_sample(y[hold], max_rows, seed=seed + 7)]
    return train, hold


def _holdout_split(y: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        perm = rng.permutation(len(y))
        cut = max(1, int(round(len(y) * fraction)))
        return np.sort(perm[cut:]), np.sort(perm[:cut])
    hold = []
    for cls in classes:
        idx = rng.permutation(np.flatnonzero(y == cls))
        k = max(1, int(round(len(idx) * fraction)))
        hold.append(idx[:k])
    hold_idx = np.sort(np.concatenate(hold))
    mask = np.ones(len(y), dtype=bool)
    mask[hold_idx] = False
    return np.flatnonzero(mask), hold_idx


def predict_in_chunks(model: FittedModel, X: np.ndarray, chunk: int, task: str | None = None) -> np.ndarray:
    """Class probabilities (n, k), or for regression one prediction per row (n,)."""
    if task == REGRESSION:
        parts = [model.predict(X[i : i + chunk]) for i in range(0, len(X), chunk)]
        return np.concatenate(parts).astype(np.float64) if parts else np.zeros(0)
    parts = [model.predict_proba(X[i : i + chunk]) for i in range(0, len(X), chunk)]
    return np.vstack(parts) if parts else np.zeros((0, 0))


# name -> (label, higher is better, tasks it applies to)
METRICS: dict[str, tuple[str, bool, tuple[str, ...]]] = {
    "roc_auc": ("ROC AUC", True, ("binary", "multiclass")),
    "pr_auc": ("PR AUC", True, ("binary",)),
    "log_loss": ("Log loss", False, ("binary", "multiclass")),
    "accuracy": ("Accuracy", True, ("binary", "multiclass")),
    "balanced_accuracy": ("Balanced accuracy", True, ("binary", "multiclass")),
    "f1": ("F1", True, ("binary",)),
    "precision": ("Precision", True, ("binary",)),
    "recall": ("Recall", True, ("binary",)),
    "f1_macro": ("F1 (macro)", True, ("multiclass",)),
    "rmse": ("RMSE", False, (REGRESSION,)),
    "mae": ("MAE", False, (REGRESSION,)),
    "r2": ("R²", True, (REGRESSION,)),
}
DEFAULT_METRIC = "roc_auc"
REGRESSION_METRIC = "rmse"


def default_metric(task: str | None = None) -> str:
    return REGRESSION_METRIC if task == REGRESSION else DEFAULT_METRIC


def check_metric(metric: str | None, task: str | None = None) -> str:
    """The metric's name, checked; None gives the task's default. With `task`, it must apply to that task."""
    metric = metric or default_metric(task)
    if metric not in METRICS:
        raise Invalid(f"Unknown metric '{metric}'. Use one of: {', '.join(METRICS)}.", code="invalid_metric")
    if task is not None and task not in METRICS[metric][2]:
        fits = [m for m, (_, _, tasks) in METRICS.items() if task in tasks]
        raise Invalid(f"Metric '{metric}' does not apply to a {task} task. Use one of: {', '.join(fits)}.",
                      code="invalid_metric")
    return metric


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else None,
    }


def regression_plots(y: np.ndarray, pred: np.ndarray, n: int = 300, bins: int = 20) -> dict:
    """Predicted against actual (a sample of up to n points) and a histogram of the errors."""
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if len(y) == 0:
        return {}
    keep = np.sort(np.random.default_rng(0).choice(len(y), size=min(n, len(y)), replace=False))
    err = pred - y
    counts, edges = np.histogram(err, bins=bins)
    return {
        "scatter": {"actual": [round(float(v), 6) for v in y[keep]],
                    "predicted": [round(float(v), 6) for v in pred[keep]]},
        "residuals": {"edges": [round(float(v), 6) for v in edges], "count": counts.tolist()},
    }


def _clean(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-7, 1)
    return proba / proba.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, proba: np.ndarray, n_classes: int) -> dict[str, float | None]:
    proba = _clean(proba)
    pred = proba.argmax(axis=1)
    out: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
    try:
        if n_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=list(range(n_classes))))
    except ValueError:
        out["roc_auc"] = None  # a class is missing from the holdout
    out["log_loss"] = float(log_loss(y, proba, labels=list(range(n_classes))))
    if n_classes == 2:
        out["pr_auc"] = float(average_precision_score(y, proba[:, 1])) if y.any() else None
        out["f1"] = float(f1_score(y, pred, zero_division=0))
        out["precision"] = float(precision_score(y, pred, zero_division=0))
        out["recall"] = float(recall_score(y, pred, zero_division=0))
    else:
        out["f1_macro"] = float(f1_score(y, pred, average="macro", zero_division=0))
    return out


def _thin(xs: np.ndarray, ys: np.ndarray, n: int = 120) -> tuple[list[float], list[float]]:
    """Keep about n points of a curve (always the ends) so it stays small in JSON."""
    if len(xs) > n:
        keep = np.unique(np.linspace(0, len(xs) - 1, n).round().astype(int))
        xs, ys = xs[keep], ys[keep]
    return [round(float(v), 4) for v in xs], [round(float(v), 4) for v in ys]


def plots(y: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    """What the dashboard draws: confusion matrix (rows = actual, columns = predicted) and, for binary, ROC and PR."""
    proba = _clean(proba)
    out: dict = {"confusion": confusion_matrix(y, proba.argmax(axis=1), labels=list(range(n_classes))).tolist()}
    if n_classes == 2 and 0 < y.sum() < len(y):
        fpr, tpr, _ = roc_curve(y, proba[:, 1])
        out["roc"] = dict(zip(("fpr", "tpr"), _thin(fpr, tpr)))
        precision, recall, _ = precision_recall_curve(y, proba[:, 1])
        r, p = _thin(recall[::-1], precision[::-1])
        out["pr"] = {"recall": r, "precision": p, "base_rate": round(float(y.mean()), 4)}
        # confusion counts at every threshold 0.01..0.99, for the threshold picker
        t = np.round(np.arange(1, 100) / 100, 2)
        hit = proba[:, 1][None, :] >= t[:, None]
        pos = y.astype(bool)[None, :]
        out["thresholds"] = {
            "t": t.tolist(),
            "tp": (hit & pos).sum(axis=1).tolist(),
            "fp": (hit & ~pos).sum(axis=1).tolist(),
            "fn": (~hit & pos).sum(axis=1).tolist(),
            "tn": (~hit & ~pos).sum(axis=1).tolist(),
        }
        # cumulative gains: review the top x% by score, catch y% of the positives
        order = np.argsort(-proba[:, 1], kind="stable")
        caught = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
        share = np.linspace(0, 1, 51)
        idx = np.round(share * len(y)).astype(int)
        out["gains"] = {"share": share.round(3).tolist(), "captured": caught[idx].round(4).tolist()}
        # calibration: predicted probability against how often it happened, in 10 equal-width bins
        bins = np.minimum((proba[:, 1] * 10).astype(int), 9)
        pred, obs, cnt = [], [], []
        for b in range(10):
            m = bins == b
            if m.sum():
                pred.append(round(float(proba[m, 1].mean()), 4))
                obs.append(round(float(y[m].mean()), 4))
                cnt.append(int(m.sum()))
        out["calibration"] = {"predicted": pred, "observed": obs, "count": cnt}
    return out


@dataclass
class Candidate:
    model: str
    display: str
    metrics: dict[str, float | None]
    seconds: float
    error: str | None = None
    plots: dict | None = None
    output: np.ndarray | None = None  # holdout probabilities (or predictions); kept in memory, never saved

    def to_dict(self) -> dict:
        return {"model": self.model, "display": self.display, **self.metrics, "seconds": round(self.seconds, 2),
                "error": self.error, "plots": self.plots}


def score(metrics_: dict, metric: str) -> float | None:
    """The metric's value, signed so that bigger is always better. ROC AUC falls back to accuracy when undefined."""
    v = metrics_.get(metric)
    if v is None and metric == "roc_auc":
        v = metrics_.get("accuracy")
    if v is None:
        return None
    return v if METRICS[metric][1] else -v


def rank_key(metric: str = DEFAULT_METRIC) -> Callable[[Candidate], tuple]:
    def key(c: Candidate) -> tuple:
        s = score(c.metrics, metric)
        ll = c.metrics.get("log_loss")
        single = not c.model.startswith("ensemble:")  # a tie goes to the single model: it costs one model, not all
        return (c.error is None, s is not None, s if s is not None else 0.0, single, -(ll if ll is not None else 1e9))

    return key


_rank = rank_key()


SCORE_STEP = 500  # rows per progress tick while scoring the holdout


def evaluate(
    spec: ModelSpec,
    X_ctx: np.ndarray,
    y_ctx: np.ndarray,
    X_hold: np.ndarray,
    y_hold: np.ndarray,
    n_classes: int,
    chunk: int,
    on_rows: Callable[[int, int], None] | None = None,
    task: str | None = None,
) -> Candidate:
    """Fit on the context, then score the holdout in steps so progress can be told in rows.

    The context is encoded once and cached (serving mode), so scoring in small steps costs about
    the same as one big predict: each step only runs its own rows against the cached context.
    """
    t0 = time.monotonic()
    regression = task == REGRESSION
    try:
        if regression:
            fitted = spec.fit_regressor(X_ctx, y_ctx, serving=True)
        else:
            if spec.max_classes is not None and n_classes > spec.max_classes:
                raise ValueError(f"{spec.display} supports up to {spec.max_classes} classes; this task has {n_classes}")
            fitted = spec.fit(X_ctx, y_ctx, n_classes, serving=True)
        step = max(1, min(chunk, SCORE_STEP))
        parts, done, total = [], 0, len(X_hold)
        if on_rows:
            on_rows(0, total)
        for i in range(0, total, step):
            parts.append(fitted.predict(X_hold[i : i + step]) if regression else fitted.predict_proba(X_hold[i : i + step]))
            done = min(total, i + step)
            if on_rows:
                on_rows(done, total)
        if regression:
            pred = np.concatenate(parts) if parts else np.zeros(0)
            if not np.all(np.isfinite(pred)):
                raise ValueError("the model returned NaN or infinite predictions")
            return Candidate(spec.name, spec.display, regression_metrics(y_hold, pred), time.monotonic() - t0,
                             plots=regression_plots(y_hold, pred), output=pred)
        proba = np.vstack(parts) if parts else np.zeros((0, n_classes))
        m = metrics(y_hold, proba, n_classes)
        return Candidate(spec.name, spec.display, m, time.monotonic() - t0, plots=plots(y_hold, proba, n_classes),
                         output=proba)
    except Exception as exc:  # one model failing must not sink the championship
        empty = (dict.fromkeys(("rmse", "mae", "r2")) if regression
                 else {"roc_auc": None, "accuracy": None, "log_loss": None})
        return Candidate(spec.name, spec.display, empty, time.monotonic() - t0, error=f"{type(exc).__name__}: {exc}")


def ensemble_candidate(members: list[Candidate], y_hold: np.ndarray, n_classes: int, task: str | None) -> Candidate:
    """Every model that finished, averaged with equal weights and scored on the same holdout.

    Nothing is fitted on the holdout (no weights to tune), so its score is as honest as each member's."""
    from ..models.ensemble import ensemble_name

    name = ensemble_name([c.model for c in members])
    display = "Ensemble (" + " + ".join(c.display for c in members) + ")"
    out = np.mean([c.output for c in members], axis=0)
    seconds = sum(c.seconds for c in members)
    if task == REGRESSION:
        return Candidate(name, display, regression_metrics(y_hold, out), seconds,
                         plots=regression_plots(y_hold, out), output=out)
    return Candidate(name, display, metrics(y_hold, out, n_classes), seconds, plots=plots(y_hold, out, n_classes),
                     output=out)


@dataclass
class ChampionshipResult:
    champion: str
    candidates: list[Candidate]
    rows: dict[str, int]
    context_X: np.ndarray
    context_y: np.ndarray
    baselines: list[Candidate] | None = None


def baselines(y_ctx: np.ndarray, y_hold: np.ndarray, n_classes: int, task: str | None = None) -> list[Candidate]:
    """What you get without a model, scored on the same holdout: random scores, and always the most common class.

    Regression: always the context's mean, and always its median."""
    if task == REGRESSION:
        out = []
        for name, display, value in (("baseline:mean", "Always the average", float(np.mean(y_ctx))),
                                     ("baseline:median", "Always the median", float(np.median(y_ctx)))):
            pred = np.full(len(y_hold), value)
            out.append(Candidate(name, display, regression_metrics(y_hold, pred), 0.0,
                                 plots=regression_plots(y_hold, pred)))
        return out
    prior = np.bincount(y_ctx, minlength=n_classes).astype(float)
    prior = (prior + 1e-9) / prior.sum()
    rng = np.random.default_rng(0)
    rand = rng.dirichlet(np.ones(n_classes), size=len(y_hold))
    const = np.tile(prior, (len(y_hold), 1))
    out = []
    for name, display, proba in (("baseline:random", "Coin flip (random scores)", rand),
                                 ("baseline:majority", "Always the most common class", const)):
        out.append(Candidate(name, display, metrics(y_hold, proba, n_classes), 0.0, plots=plots(y_hold, proba, n_classes)))
    return out


def evaluate_all(models: list[ModelSpec], X_ctx, y_ctx, X_hold, y_hold, n_classes: int, chunk: int, progress: Progress,
                 *, start: float, span: float, estimate: Estimate | None = None, observe: Observe | None = None,
                 task: str | None = None) -> list[Candidate]:
    """Score every model on the holdout, reporting progress with an estimate of the time left.

    The estimate starts from `estimate` (learned from earlier runs) and is rescaled by how this run is going:
    once a model finishes, the remaining ones are assumed to be off by the same factor.
    """
    n_ctx, n_hold, n_feat = len(y_ctx), len(y_hold), int(X_ctx.shape[1]) if X_ctx.ndim > 1 else 1
    expected = [estimate(m.name, n_ctx, n_hold, n_feat) if estimate else None for m in models]
    scale = 1.0
    out: list[Candidate] = []
    n = len(models)
    for i, spec in enumerate(models):
        later = [e for e in expected[i + 1 :] if e is not None]
        later_s = sum(later) * scale if len(later) == n - i - 1 else None
        head = f"Running inference with {spec.display} ({i + 1} of {n})"
        t_model = time.monotonic()
        t_rows: list[float] = []  # when scoring started, once the context is encoded

        def on_rows(done: int, total: int, i=i, head=head, t_model=t_model, t_rows=t_rows, later_s=later_s) -> None:
            now_ = time.monotonic()
            if done == 0:
                t_rows.append(now_)
                msg, frac = f"{head} · loading {n_ctx:,} rows as context", 0.0
                mine = expected[i] * scale if expected[i] else None  # noqa: B023 (read live on purpose)
            else:
                rate = (now_ - t_rows[0]) / done
                mine = rate * (total - done)  # measured, not guessed
                msg, frac = f"{head} · {done:,} / {total:,} holdout rows", done / max(1, total)
            eta = round(mine + later_s) if mine is not None and later_s is not None else None
            progress("evaluate", start + span * (i + frac) / n, msg, eta)

        if expected[i] is not None and later_s is not None:
            progress("evaluate", start + span * i / n, head, round(expected[i] * scale + later_s))
        else:
            progress("evaluate", start + span * i / n, head, None)
        c = evaluate(spec, X_ctx, y_ctx, X_hold, y_hold, n_classes, chunk, on_rows, task=task)
        out.append(c)
        if c.error is None and c.seconds:
            if observe:
                observe(spec.name, n_ctx, n_hold, n_feat, c.seconds)
            if expected[i]:
                # blend: one slow model says something about this machine, not everything
                scale = 0.5 * scale + 0.5 * (c.seconds / expected[i])
    return out


def run_championship(
    pre: Preprocessor,
    df: pd.DataFrame,
    models: list[ModelSpec],
    *,
    max_context_rows: int,
    holdout_fraction: float,
    chunk: int,
    progress: Progress,
    max_holdout_rows: int | None = None,
    metric: str | None = None,
    estimate: Estimate | None = None,
    observe: Observe | None = None,
    ensemble: bool = True,
) -> ChampionshipResult:
    if not models:
        raise Invalid("No models to run. Install some with pip install zerograd or pass models=[...].", code="no_models")
    task = pre.task
    metric = metric or default_metric(task)
    X = pre.transform(df)
    y = pre.encode_target(df[pre.target])
    n_classes = len(pre.classes)
    s = strata(y, task)

    train_idx, hold_idx = holdout_split(s, holdout_fraction, max_rows=max_holdout_rows)
    ctx_idx = train_idx[stratified_sample(s[train_idx], max_context_rows)]
    progress("split", 0.15, f"Context {len(ctx_idx):,} rows · holdout {len(hold_idx):,} rows (all scored)")

    candidates = evaluate_all(models, X[ctx_idx], y[ctx_idx], X[hold_idx], y[hold_idx], n_classes, chunk, progress,
                              start=0.15, span=0.75, estimate=estimate, observe=observe, task=task)

    ok = [c for c in candidates if c.error is None]
    if ensemble and len(ok) >= 2:
        candidates.append(ensemble_candidate(ok, y[hold_idx], n_classes, task))
        ok = [c for c in candidates if c.error is None]
    if not ok:
        raise Invalid(
            "Every model failed on this data: " + "; ".join(f"{c.display}: {c.error}" for c in candidates),
            code="all_models_failed",
        )
    champion = max(candidates, key=rank_key(metric))
    n_ok = sum(1 for c in ok if not c.model.startswith("ensemble:"))
    progress("select", 0.92, f"Compared {n_ok} model{'s' if n_ok > 1 else ''} on the same {len(hold_idx):,} holdout rows")

    final_idx = stratified_sample(s, max_context_rows, seed=1)
    return ChampionshipResult(
        champion=champion.model,
        candidates=candidates,
        baselines=baselines(y[ctx_idx], y[hold_idx], n_classes, task),
        rows={
            "labeled": int(len(y)),
            "evaluation_context": int(len(ctx_idx)),
            "holdout": int(len(hold_idx)),
            "production_context": int(len(final_idx)),
        },
        context_X=X[final_idx],
        context_y=y[final_idx],
    )
