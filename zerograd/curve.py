"""Context curves: how much labeled history a model needs, and what each extra row costs in speed.

Tabular foundation models read their context (your labeled rows) on every prediction, so a bigger context
usually means a better score and a slower answer. A context curve fits the same model on growing stratified
samples of the context, scores each on the same holdout, and times it. The recommended size is the smallest
one whose score is within `tolerance` of the best.

    curve = zg.context_curve(train, "churned")            # local: any table, any installed model
    curve = cap.context_curve(test)                       # a capability: its champion, its own context
    curve.recommended                                     # e.g. 2000
    curve.plot()                                          # score and latency, recommended size marked
    cap.update(context_rows=curve.recommended)            # serve with the smaller context
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import pandas as pd

DEFAULT_SIZES = (250, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
DEFAULT_TOLERANCE = 0.005  # absolute, for scores between 0 and 1
RELATIVE_TOLERANCE = 0.005  # errors in the target's own unit (RMSE, MAE): within 0.5% of the best

# name -> (label, higher is better); the same names as the championship metrics
_LABELS = {
    "roc_auc": ("ROC AUC", True), "pr_auc": ("PR AUC", True), "log_loss": ("Log loss", False),
    "accuracy": ("Accuracy", True), "balanced_accuracy": ("Balanced accuracy", True), "f1": ("F1", True),
    "precision": ("Precision", True), "recall": ("Recall", True), "f1_macro": ("F1 (macro)", True),
    "rmse": ("RMSE", False), "mae": ("MAE", False), "r2": ("R²", True),
}
_RELATIVE = {"rmse", "mae"}  # unbounded, in the target's unit: tolerance is a share of the best value
_ACCENT = "#6c63ff"  # ZeroGrad violet
_MUTED = "#8a8f87"


def default_sizes(available: int, cap: int | None = None) -> list[int]:
    """The usual ladder (250, 500, 1K, 2K, 5K, ...) up to what is available, ending at the full context."""
    top = min(available, cap) if cap else available
    sizes = [s for s in DEFAULT_SIZES if s < top]
    return sizes + [top]


def check_sizes(sizes: Iterable[int] | None, available: int, cap: int | None = None) -> list[int]:
    if sizes is None:
        return default_sizes(available, cap)
    out = sorted({int(s) for s in sizes})
    if not out or out[0] < 2:
        raise ValueError("sizes are context row counts, each at least 2.")
    too_big = [s for s in out if s > available]
    if too_big:
        raise ValueError(f"Only {available:,} labeled rows are available for the context; "
                         f"drop {', '.join(f'{s:,}' for s in too_big)} from sizes.")
    return out


def recommend(values: Iterable[float | None], sizes: Iterable[int], *, higher_is_better: bool = True,
              tolerance: float = DEFAULT_TOLERANCE, relative: bool = False) -> int | None:
    """The smallest size whose value is within `tolerance` of the best one (absolute, or a share of the best)."""
    pairs = [(int(s), float(v)) for s, v in zip(sizes, values) if v is not None and not _isnan(v)]
    if not pairs:
        return None
    best = max(v for _, v in pairs) if higher_is_better else min(v for _, v in pairs)
    room = tolerance * abs(best) if relative else tolerance
    ok = [s for s, v in pairs if (best - v if higher_is_better else v - best) <= room + 1e-12]
    return min(ok)


class ContextCurve(pd.DataFrame):
    """One row per context size (the index, `rows`): the score on the holdout and the time it took.

    A regular DataFrame, plus `recommended`, `plot()` and a notebook view that marks the recommended size.
    """

    _metadata = ["metric", "tolerance", "model"]

    @property
    def _constructor(self):  # slices and arithmetic give plain DataFrames
        return pd.DataFrame

    @classmethod
    def build(cls, rows: list[dict], *, metric: str, tolerance: float | None = None,
              model: str | None = None) -> "ContextCurve":
        """`tolerance` None: 0.005 absolute, or 0.5% of the best for RMSE and MAE."""
        if tolerance is None:
            tolerance = RELATIVE_TOLERANCE if metric in _RELATIVE else DEFAULT_TOLERANCE
        df = pd.DataFrame(rows)
        if len(df):
            df = df.set_index("rows").sort_index()
        out = cls(df)
        out.metric, out.tolerance, out.model = metric, tolerance, model
        if len(out) and metric in out:
            rec = out.recommended
            out["recommended"] = out.index == rec
        return out

    # ------------------------------------------------------------------ reading it
    @property
    def metric_label(self) -> str:
        return _LABELS.get(self.metric, (self.metric, True))[0]

    @property
    def higher_is_better(self) -> bool:
        return _LABELS.get(self.metric, (self.metric, True))[1]

    @property
    def relative(self) -> bool:
        """True when `tolerance` is a share of the best value (RMSE, MAE) rather than an absolute difference."""
        return self.metric in _RELATIVE

    @property
    def recommended(self) -> int | None:
        """The smallest context within `tolerance` of the best score."""
        if self.metric not in self:
            return None
        rec = recommend(self[self.metric].tolist(), self.index.tolist(), higher_is_better=self.higher_is_better,
                        tolerance=self.tolerance, relative=self.relative)
        return int(rec) if rec is not None else None

    def summary(self) -> str:
        rec = self.recommended
        if rec is None:
            return "No size produced a score."
        best_rows = self[self.metric].idxmax() if self.higher_is_better else self[self.metric].idxmin()
        best, got = self.loc[best_rows, self.metric], self.loc[rec, self.metric]
        within = f"{self.tolerance:.1%}".replace(".0%", "%") if self.relative else f"{self.tolerance:g}"
        text = (f"{rec:,} rows of context score {self.metric_label} {_fmt(got)}, "
                f"within {within} of the best ({_fmt(best)} with {int(best_rows):,} rows)")
        if "ms_per_row" in self and rec != best_rows:
            fast, slow = self.loc[rec, "ms_per_row"], self.loc[best_rows, "ms_per_row"]
            if fast > 0 and slow > 0:
                text += f", and predict {slow / fast:.1f}× faster ({fast:.2f} vs {slow:.2f} ms per row)"
        return text + "."

    # ------------------------------------------------------------------ showing it
    def plot(self, ax: Any = None, *, title: str | None = None, **kwargs: Any):
        """Score (left axis) and prediction latency (right axis) against context rows; the pick is marked.

        Needs matplotlib (`pip install matplotlib`, already in Colab and Jupyter setups). Returns the axes.
        Any other keyword (y=, kind=, ylim=, ...) draws pandas' regular DataFrame.plot instead.
        """
        if kwargs:
            return pd.DataFrame(self).plot(ax=ax, title=title, **kwargs)
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("ContextCurve.plot needs matplotlib: pip install matplotlib") from exc

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.2))
        x = self.index.to_numpy()
        y = self[self.metric].to_numpy(dtype=float)
        ax.plot(x, y, color=_ACCENT, marker="o", lw=2.2, ms=6, label=self.metric_label, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _short(v)))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        ax.set_xlabel("Rows of context")
        ax.set_ylabel(self.metric_label, color=_ACCENT)
        ax.tick_params(axis="y", colors=_ACCENT)
        for side in ("top",):
            ax.spines[side].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25)

        handles = ax.get_lines()[:1]
        if "ms_per_row" in self and self["ms_per_row"].notna().any():
            ax2 = ax.twinx()
            ax2.plot(x, self["ms_per_row"].to_numpy(dtype=float), color=_MUTED, marker="s", ms=4, lw=1.6, ls="--",
                     label="Prediction time (ms per row)")
            ax2.set_ylabel("ms per row", color=_MUTED)
            ax2.tick_params(axis="y", colors=_MUTED)
            ax2.spines["top"].set_visible(False)
            ax2.set_ylim(bottom=0)
            handles = handles + ax2.get_lines()[:1]

        rec = self.recommended
        if rec is not None:
            ry = float(self.loc[rec, self.metric])
            ax.axvline(rec, color=_ACCENT, alpha=0.35, lw=1, ls=":")
            ax.scatter([rec], [ry], s=170, facecolors="none", edgecolors=_ACCENT, lw=2, zorder=4)
            ax.annotate(f"recommended: {rec:,} rows", (rec, ry), xytext=(8, -18), textcoords="offset points",
                        color=_ACCENT, fontsize=9, fontweight="bold")
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
        model = f" · {self.model}" if self.model else ""
        ax.set_title(title or f"Context size vs {self.metric_label} and speed{model}", fontsize=11, loc="left")
        return ax

    def _repr_html_(self) -> str:
        df = pd.DataFrame(self)
        rec = self.recommended
        fmt = {c: "{:.3f}" for c in df.columns if c in _LABELS or c == "accuracy"}
        fmt.update({c: "{:.2f}" for c in ("fit_seconds", "predict_seconds", "seconds", "ms_per_row") if c in df})
        head = f"<p style='margin:4px 0;color:{_MUTED}'>{_escape(self.summary())}</p>" if rec is not None else ""
        try:
            styler = df.style.format(fmt, na_rep="—")
        except (AttributeError, ImportError):  # pandas' Styler needs jinja2
            return head + df.to_html(float_format=lambda v: f"{v:.3f}", na_rep="—")
        if rec is not None:
            styler = styler.apply(lambda r: [f"background-color: {_ACCENT}22; font-weight: 600" if r.name == rec
                                             else "" for _ in r], axis=1)
        return head + styler.to_html()


def _fmt(v: float) -> str:
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def _short(v: float) -> str:
    v = int(round(v))
    if v >= 1000 and v % 1000 == 0:
        return f"{v // 1000}K"
    if v >= 1000:
        return f"{v / 1000:.1f}K"
    return str(v)


def _isnan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ms_per_row(seconds: float, n: int) -> float:
    return float(seconds * 1000 / max(1, n))


__all__ = ["ContextCurve", "recommend", "default_sizes", "check_sizes", "DEFAULT_TOLERANCE", "RELATIVE_TOLERANCE"]
