"""The chain rule, in context: density, imputation and sampling from any tabular classifier.

    p(x1, ..., xd) = p(x1) · p(x2 | x1) · ... · p(xd | x1..xd-1)

Every factor is a classification question ("which code does column j take, given the columns before it?"),
which TabPFN v2 or TabICLv2 answer in one forward pass with the reference rows as context. Averaging a few
column orders makes the estimate stable. This is the method of the TabPFN paper (Hollmann et al., Nature 2025),
written here for any classifier in the registry, so no extra package or telemetry is needed.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..models import ModelSpec
from .table import Codes

EPS = 1e-6


def sample_rows(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.arange(n) if n <= k else np.sort(rng.choice(n, k, replace=False))


def conditional(spec: ModelSpec, X_ctx: np.ndarray, y_ctx: np.ndarray, n_codes: int, X_query: np.ndarray,
                chunk: int = 10_000) -> np.ndarray:
    """P(code | features) for every query row, shape (n_query, n_codes). No features: the smoothed marginal."""
    counts = np.bincount(y_ctx, minlength=n_codes).astype(np.float64) + 0.5
    marginal = counts / counts.sum()
    if X_ctx.shape[1] == 0 or len(np.unique(y_ctx)) < 2:
        return np.tile(marginal, (len(X_query), 1))
    model = spec.fit(X_ctx, y_ctx, n_codes)
    parts = [np.asarray(model.predict_proba(X_query[i:i + chunk]), dtype=np.float64)
             for i in range(0, len(X_query), chunk)]
    p = np.vstack(parts) if parts else np.zeros((0, n_codes))
    p = np.clip(np.nan_to_num(p, nan=0.0), 0.0, None) + EPS
    return p / p.sum(axis=1, keepdims=True)


def log_density(spec: ModelSpec, X_ctx: np.ndarray, X_query: np.ndarray, codes: Codes, *, n_orders: int = 3,
                max_context: int = 2000, seed: int = 0,
                on_step: Callable[[], None] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """log p(x) per query row, and -log p of each column given the others before it (averaged over orders)."""
    rng = np.random.default_rng(seed)
    d = X_query.shape[1]
    per_order = np.zeros((n_orders, len(X_query)))
    surprise = np.zeros((len(X_query), d))
    for o in range(n_orders):
        order = rng.permutation(d)
        for pos, j in enumerate(order):
            y_ctx = codes.column(X_ctx, j)
            y_q = codes.column(X_query, j)
            keep = np.flatnonzero(y_ctx >= 0)
            keep = keep[sample_rows(len(keep), max_context, rng)]
            prev = order[:pos]
            p = conditional(spec, X_ctx[np.ix_(keep, prev)], y_ctx[keep], codes.n_codes[j], X_query[:, prev])
            known = y_q >= 0  # a missing value that never happened in the reference adds nothing
            lp = np.zeros(len(X_query))
            lp[known] = np.log(p[np.flatnonzero(known), y_q[known]])
            per_order[o] += lp
            surprise[:, j] += -lp / n_orders
            if on_step:
                on_step()
    # average the densities (not the logs) across orders, in log space
    m = per_order.max(axis=0)
    return m + np.log(np.exp(per_order - m).mean(axis=0)), surprise


def sample(spec: ModelSpec, X_ref: np.ndarray, codes: Codes, n: int, *, temperature: float = 1.0,
           max_context: int = 2000, seed: int = 0, on_step: Callable[[], None] | None = None) -> np.ndarray:
    """n new rows, one column at a time: draw a code from P(code | columns so far), then a real value in it."""
    rng = np.random.default_rng(seed)
    d = X_ref.shape[1]
    out = np.full((n, d), np.nan, dtype=np.float32)
    order = rng.permutation(d)
    for pos, j in enumerate(order):
        y_ref = codes.column(X_ref, j)
        keep = np.flatnonzero(y_ref >= 0)
        keep = keep[sample_rows(len(keep), max_context, rng)]
        prev = order[:pos]
        p = conditional(spec, X_ref[np.ix_(keep, prev)], y_ref[keep], codes.n_codes[j], out[:, prev])
        if temperature != 1.0:
            p = p ** (1.0 / max(temperature, 1e-3))
            p /= p.sum(axis=1, keepdims=True)
        drawn = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
        out[:, j] = [codes.sample_value(j, int(c), rng) for c in drawn]
        if on_step:
            on_step()
    return out


def fill(spec: ModelSpec, X: np.ndarray, kinds: list[str], targets: np.ndarray, *, max_context: int = 2000,
         seed: int = 0, on_step: Callable[[], None] | None = None) -> np.ndarray:
    """Fill the cells where `targets` is True, each column predicted from all the others.

    Numbers use the model's regressor when it has one (else the expected value over quantile bins), categories
    and booleans the most likely class.
    """
    rng = np.random.default_rng(seed)
    X = X.astype(np.float32, copy=True)
    filled = X.copy()
    for j in np.flatnonzero(targets.any(axis=0)):
        others = [c for c in range(X.shape[1]) if c != j]
        ctx = np.flatnonzero(~np.isnan(X[:, j]) & ~targets[:, j])
        if len(ctx) == 0:
            continue
        ctx = ctx[sample_rows(len(ctx), max_context, rng)]
        rows = np.flatnonzero(targets[:, j])
        Xc, Xq, yc = X[np.ix_(ctx, others)], X[np.ix_(rows, others)], X[ctx, j].astype(np.float64)
        numeric = kinds[j] in ("numeric", "datetime")
        if not others or np.unique(yc).size == 1:  # nothing to condition on, or one value: the mean or the mode
            filled[rows, j] = yc.mean() if numeric else np.bincount(yc.astype(np.int64)).argmax()
        elif numeric:
            if spec.regress_fn is not None:
                filled[rows, j] = spec.fit_regressor(Xc, yc).predict(Xq)
            else:
                codes = Codes.fit(yc[:, None], ["numeric"])
                p = conditional(spec, Xc, codes.column(yc[:, None], 0), codes.n_codes[0], Xq)
                mids = np.array([np.median(v) if len(v) else np.nan for v in codes.values[0]])
                mids = np.where(np.isnan(mids), np.nanmean(yc), mids)
                filled[rows, j] = p @ mids
        else:
            codes = Codes.fit(yc[:, None], [kinds[j]])
            p = conditional(spec, Xc, codes.column(yc[:, None], 0), codes.n_codes[0], Xq)
            back: dict[int, float] = {}
            for cat, code in sorted((codes.maps[0] or {}).items(), key=lambda kv: -np.sum(yc == kv[0])):
                back.setdefault(code, float(cat))  # the "other" code fills with its most common category
            mode = float(np.bincount(yc.astype(np.int64)).argmax())
            filled[rows, j] = [back.get(int(c), mode) for c in p.argmax(axis=1)]
        if on_step:
            on_step()
    return filled
