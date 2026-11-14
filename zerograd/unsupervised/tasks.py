"""Anomaly detection, imputation, synthetic data and clustering, on your machine, with foundation models.

    import zerograd
    zg = zerograd.ZeroGrad()
    found = zg.anomalies(df)                 # every row scored, the unusual ones flagged, with the columns why
    filled = zg.impute(df)                   # a championship on cells we hide, then the champion fills the gaps
    fake = zg.synthesize(df, n=1_000)        # new rows, scored by how hard they are to tell from yours
    groups = zg.cluster(df)                  # segments, with what sets each one apart

Each task compares the foundation models with the classic scikit-learn method on the same data whenever there
is something to measure, and says which one won.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .._ui import Console
from ..errors import ValidationError
from ..models import ModelSpec, all_models, get_model
from . import chain
from .table import Codes, Table

ANOMALY_DEFAULTS = ("tabpfn-v2", "fomo-0d", "tabicl-v2")  # the first available scores when there are no labels
BASELINES = {
    "anomaly_detection": ("isolation-forest", "local-outlier-factor", "knn"),
    "imputation": ("mean-mode", "knn-imputer", "iterative-imputer"),
    "synthetic_data": ("independent-columns",),
    "clustering": ("kmeans", "gaussian-mixture"),
}


# ---------------------------------------------------------------- picking models
def candidates(task: str, models: Iterable[str] | None, noncommercial: Iterable[str] = ()) -> list[ModelSpec]:
    """The foundation models that run `task` here. models=None: every commercial one installed for it."""
    if models is None:
        chosen = [m for m in all_models() if m.commercial and task in m.sdk_tasks and m.available]
        chosen += [get_model(n) for n in noncommercial if task in get_model(n).sdk_tasks]
        return chosen
    out = []
    for name in models:
        if name in sum(BASELINES.values(), ()):
            continue
        spec = get_model(name)
        if not spec.commercial and name not in set(noncommercial):
            raise PermissionError(f"'{name}' is non-commercial: enable it with noncommercial=['{name}'].")
        if task not in spec.sdk_tasks:
            raise ValidationError(f"{spec.display} does not run {task.replace('_', ' ')} in this version of the SDK.")
        if not spec.available:
            raise ValidationError(f"{spec.display} is not installed: {spec.install}")
        out.append(spec)
    return out


def _baselines(task: str, models: Iterable[str] | None) -> list[str]:
    names = BASELINES[task]
    return list(names) if models is None else [m for m in models if m in names]


def _attribution(specs: list[ModelSpec]) -> str | None:
    marks = sorted({s.attribution for s in specs if s.attribution})
    return " · ".join(marks) or None


def _read(data: Any) -> pd.DataFrame:
    from .._remote import to_frame

    return to_frame(data)


def _standardized(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), np.where(np.isnan(med), 0.0, med), X)
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def _folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


class _Result:
    task = ""
    title = ""

    def _summary(self) -> list[str]:
        return []

    def __repr__(self) -> str:
        lines = [f"<{self.title}>"] + [f"  {line}" for line in self._summary()]
        ev = getattr(self, "evaluation", None)
        if ev is not None and len(ev):
            lines += ["", ev.to_string()]
        if getattr(self, "attribution", None):
            lines += ["", self.attribution]
        return "\n".join(lines)


# ---------------------------------------------------------------- anomaly detection
@dataclass
class AnomalyResult(_Result):
    """`scores`: one row per input row, same index: anomaly_score (higher = more unusual), percentile, is_anomaly
    and why (the columns that make it unusual, for the foundation models that can say)."""

    scores: pd.DataFrame
    model: str
    data: pd.DataFrame
    evaluation: pd.DataFrame | None = None
    attribution: str | None = None
    seconds: float = 0.0
    title = "Anomaly detection"

    def top(self, n: int = 10) -> pd.DataFrame:
        """The n most unusual rows, with their values and why."""
        idx = self.scores.sort_values("anomaly_score", ascending=False).index[:n]
        return self.scores.loc[idx, ["anomaly_score", "why"]].join(self.data.loc[idx])

    @property
    def anomalies(self) -> pd.DataFrame:
        return self.data.loc[self.scores["is_anomaly"]]

    def _summary(self) -> list[str]:
        return [f"model: {self.model} · {int(self.scores['is_anomaly'].sum()):,} of {len(self.scores):,} rows flagged"]


def _anomaly_scores(name: str, spec: ModelSpec | None, X_ctx: np.ndarray, X_q: np.ndarray, kinds: list[str],
                    codes: Codes | None, n_orders: int, max_context: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if spec is not None and spec.anomaly_fn is not None:
        return np.asarray(spec.anomaly_fn(X_ctx, X_q), dtype=np.float64), None
    if spec is not None:
        logp, surprise = chain.log_density(spec, X_ctx, X_q, codes, n_orders=n_orders, max_context=max_context, seed=seed)
        return -logp, surprise
    both = _standardized(np.vstack([X_ctx, X_q]))
    ctx, q = both[: len(X_ctx)], both[len(X_ctx):]
    if name == "isolation-forest":
        from sklearn.ensemble import IsolationForest

        return -IsolationForest(random_state=seed).fit(ctx).score_samples(q), None
    if name == "local-outlier-factor":
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(n_neighbors=min(20, len(ctx) - 1), novelty=True).fit(ctx)
        return -lof.score_samples(q), None
    if name == "knn":
        from sklearn.neighbors import NearestNeighbors

        dist, _ = NearestNeighbors(n_neighbors=min(5, len(ctx))).fit(ctx).kneighbors(q)
        return dist.mean(axis=1), None
    raise ValidationError(f"Unknown anomaly method '{name}'.")


def anomalies(data: Any, *, labels: Any = None, reference: Any = None, models: Iterable[str] | None = None,
              noncommercial: Iterable[str] = (), contamination: float = 0.05, drop: Iterable[str] | None = None,
              n_orders: int = 3, folds: int = 2, trim: bool = True, max_context_rows: int = 2000, seed: int = 0,
              progress: bool = True) -> AnomalyResult:
    """Score every row of `data` by how unusual it is. No labels needed.

    `reference`: rows known to be normal (say, last month); without it each row is scored against the other rows
    (cross-fitted in `folds`, so no row sees itself). `labels`: a column name or array marking known anomalies (1)
    for a championship between the foundation models and Isolation Forest, LOF and kNN; without labels the first
    available of TabPFN v2, FoMo-0D and TabICLv2 scores. `contamination`: the share of rows flagged. `trim`: score
    twice, the second time leaving out of the context the 2 × contamination most unusual rows of the first pass,
    so anomalies do not vouch for each other (every method gets the same treatment).
    """
    ui = Console(progress)
    t0 = time.monotonic()
    df = _read(data)
    y = None
    if isinstance(labels, str):
        if labels not in df.columns:
            raise ValidationError(f"Label column '{labels}' is not in the data.")
        y = df[labels].to_numpy()
        df = df.drop(columns=[labels])
    elif labels is not None:
        y = np.asarray(labels)
    if y is not None:
        y = (pd.Series(y).astype(float).fillna(0).to_numpy() > 0).astype(int)
        if len(y) != len(df) or y.sum() == 0 or y.sum() == len(y):
            raise ValidationError("labels needs one 0/1 value per row, with both values present.")
    if not 0 < contamination < 0.5:
        raise ValidationError("contamination is the share of rows to flag, between 0 and 0.5.")

    table = Table.fit(df, drop=list(drop or []))
    X = table.transform(df)
    ref = table.transform(_read(reference)) if reference is not None else None
    specs = candidates("anomaly_detection", models, noncommercial)
    if y is None and models is None:  # no labels: one model scores, the first one available
        specs = sorted(specs, key=lambda s: ANOMALY_DEFAULTS.index(s.name) if s.name in ANOMALY_DEFAULTS else 99)[:1]
    methods: list[tuple[str, ModelSpec | None]] = [(s.name, s) for s in specs]
    if y is not None or models is not None:  # labels: the classic methods compete too; or you named some
        methods += [(b, None) for b in _baselines("anomaly_detection", models)]
    if not methods:  # no foundation model installed
        methods = [("isolation-forest", None)]
    ui.done(f"{len(df):,} rows · {len(table.columns)} columns used"
            + (ui.dim(f" · dropped {', '.join(d['name'] for d in table.dropped)}") if table.dropped else ""))

    results: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    errors: dict[str, str] = {}
    for name, spec in methods:
        label = spec.display if spec else name
        try:
            with ui.spinner(f"{label} · scoring {len(df):,} rows", done=f"{label} scored"):
                codes = Codes.fit(ref if ref is not None else X, table.kinds) if spec and spec.anomaly_fn is None else None
                if ref is not None:
                    results[name] = _anomaly_scores(name, spec, ref, X, table.kinds, codes, n_orders,
                                                    max_context_rows, seed)
                else:
                    # Round 1 scores every row against the others. The context then holds the anomalies too, so
                    # round 2 leaves the rows round 1 found most unusual out of the context and scores again.
                    trusted = np.ones(len(X), dtype=bool)
                    for _ in range(2 if trim else 1):
                        score, why = np.zeros(len(X)), (np.zeros(X.shape) if spec and spec.anomaly_fn is None else None)
                        for part in _folds(len(X), max(2, folds), seed):
                            ctx = np.setdiff1d(np.flatnonzero(trusted), part)
                            s, w = _anomaly_scores(name, spec, X[ctx], X[part], table.kinds, codes, n_orders,
                                                   max_context_rows, seed)
                            score[part] = s
                            if why is not None and w is not None:
                                why[part] = w
                        trusted = score <= np.nanquantile(score, 1 - min(0.5, 2 * contamination))
                    results[name] = (score, why)
        except Exception as e:  # one method failing never sinks the others
            errors[name] = f"{type(e).__name__}: {e}"
            ui.say(f"  {ui.warn('!')} {label} did not finish: {str(e)[:80]}")
    if not results:
        raise ValidationError(f"No method finished: {errors}")

    evaluation = None
    champion = next(iter(results))
    if y is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score

        rows = []
        for name, (score, _) in results.items():
            s = np.nan_to_num(score, nan=np.nanmedian(score))
            rows.append({"model": name, "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
                         "kind": "foundation model" if name not in BASELINES["anomaly_detection"] else "classic"})
        rows += [{"model": n, "error": e} for n, e in errors.items()]
        evaluation = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)
        champion = str(evaluation.index[0])
        ui.podium([(r, evaluation.loc[r, "roc_auc"], None) for r in evaluation.index if r in results],
                  "ROC AUC", _attribution(specs), baseline=0.5, baseline_label="random order")

    score, surprise = results[champion]
    pct = pd.Series(score).rank(pct=True).to_numpy() * 100
    flag = pct > 100 * (1 - contamination)
    why: list[str | None] = [None] * len(score)
    if surprise is not None:
        rel = surprise - np.median(surprise, axis=0)  # against how surprising the column usually is
        for i in range(len(score)):
            top = [table.names[j] for j in np.argsort(-rel[i])[:3] if rel[i, j] > 0.5]
            why[i] = ", ".join(top) or None
    out = pd.DataFrame({"anomaly_score": score, "percentile": pct.round(1), "is_anomaly": flag, "why": why},
                       index=df.index)
    used = [get_model(champion)] if champion not in BASELINES["anomaly_detection"] else []
    res = AnomalyResult(scores=out, model=champion, data=df, evaluation=evaluation, attribution=_attribution(used),
                        seconds=time.monotonic() - t0)
    ui.say(f"{ui.accent('◆')} {int(flag.sum()):,} unusual rows flagged by {champion} "
           + ui.dim(f"· top {contamination:.0%} · .top() shows them with why"))
    return res


# ---------------------------------------------------------------- imputation
@dataclass
class ImputeResult(_Result):
    """`data`: your table with the gaps filled; `filled`: True where a value was filled in."""

    data: pd.DataFrame
    filled: pd.DataFrame
    model: str
    evaluation: pd.DataFrame
    attribution: str | None = None
    seconds: float = 0.0
    title = "Imputation"

    def _summary(self) -> list[str]:
        return [f"champion: {self.model} · {int(self.filled.to_numpy().sum()):,} cells filled"]


def _impute_baseline(name: str, X: np.ndarray, kinds: list[str], targets: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer, KNNImputer

    Xm = X.astype(np.float64).copy()
    Xm[targets] = np.nan
    out = Xm.copy()
    discrete = [j for j, k in enumerate(kinds) if k in ("categorical", "boolean")]
    if name == "mean-mode":
        for j in range(X.shape[1]):
            col = Xm[:, j]
            obs = col[~np.isnan(col)]
            if len(obs):
                val = np.bincount(obs.astype(int)).argmax() if j in discrete else obs.mean()
                out[np.isnan(col), j] = val
        return out
    keep = [j for j in range(X.shape[1]) if not np.isnan(Xm[:, j]).all()]
    imp = KNNImputer(n_neighbors=5) if name == "knn-imputer" else IterativeImputer(random_state=seed, max_iter=10)
    out[:, keep] = imp.fit_transform(Xm[:, keep])
    for j in discrete:
        out[:, j] = np.round(out[:, j])
    return out


def _impute_error(X: np.ndarray, filled: np.ndarray, kinds: list[str], hidden: np.ndarray) -> dict[str, float]:
    """Numbers: RMSE over the column's spread (0 = perfect, 1 = as good as guessing the mean). Categories: error rate."""
    per = {}
    for j in np.flatnonzero(hidden.any(axis=0)):
        t, p = X[hidden[:, j], j].astype(np.float64), filled[hidden[:, j], j].astype(np.float64)
        if kinds[j] in ("categorical", "boolean"):
            per[j] = float(np.mean(np.round(p) != t))
        else:
            sd = np.nanstd(X[:, j]) or 1.0
            per[j] = float(np.sqrt(np.nanmean((p - t) ** 2)) / sd)
    return per


def impute(data: Any, *, models: Iterable[str] | None = None, noncommercial: Iterable[str] = (),
           drop: Iterable[str] | None = None, hide_fraction: float = 0.1, max_hidden_per_column: int = 300,
           max_context_rows: int = 2000, seed: int = 0, progress: bool = True) -> ImputeResult:
    """Fill the missing values of `data`.

    First a championship: we hide `hide_fraction` of the known values in every column with gaps, each model fills
    them, and we score how close it got (numbers: RMSE over the column's spread; categories: error rate), against
    mean/mode, KNN and iterative imputation. Then the champion fills the real gaps.
    """
    ui = Console(progress)
    t0 = time.monotonic()
    df = _read(data)
    table = Table.fit(df, drop=list(drop or []))
    X = table.transform(df)
    gaps = np.isnan(X)
    if not gaps.any():
        raise ValidationError("No missing values in the columns used: nothing to fill.", code="nothing_to_impute")
    rng = np.random.default_rng(seed)
    hidden = np.zeros_like(gaps)
    for j in np.flatnonzero(gaps.any(axis=0)):
        known = np.flatnonzero(~gaps[:, j])
        k = min(max_hidden_per_column, max(1, int(round(hide_fraction * len(known)))))
        if len(known) > k + 5:
            hidden[rng.choice(known, k, replace=False), j] = True
    if not hidden.any():
        raise ValidationError("Too few known values to measure an imputation.", code="too_few_rows")
    ui.done(f"{len(df):,} rows · {int(gaps.sum()):,} missing cells in {int(gaps.any(axis=0).sum())} columns · "
            f"hiding {int(hidden.sum()):,} known cells to score the models")

    specs = candidates("imputation", models, noncommercial)
    methods: list[tuple[str, ModelSpec | None]] = [(s.name, s) for s in specs]
    methods += [(b, None) for b in _baselines("imputation", models)]
    Xh = X.copy()
    Xh[hidden] = np.nan
    rows, errors = [], {}
    for name, spec in methods:
        label = spec.display if spec else name
        try:
            with ui.spinner(f"{label} · filling {int(hidden.sum()):,} hidden cells", done=f"{label} done"):
                filled = (chain.fill(spec, Xh, table.kinds, hidden, max_context=max_context_rows, seed=seed) if spec
                          else _impute_baseline(name, X, table.kinds, hidden, seed))
            per = _impute_error(X, filled, table.kinds, hidden)
            rows.append({"model": name, "error": float(np.mean(list(per.values()))),
                         **{f"error:{table.names[j]}": v for j, v in per.items()},
                         "kind": "foundation model" if spec else "classic"})
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            ui.say(f"  {ui.warn('!')} {label} did not finish: {str(e)[:80]}")
    if not rows:
        raise ValidationError(f"No method finished: {errors}")
    evaluation = pd.DataFrame(rows).set_index("model").sort_values("error")
    champion = str(evaluation.index[0])
    ui.podium([(r, evaluation.loc[r, "error"], None) for r in evaluation.index], "imputation error",
              _attribution(specs), higher_is_better=False)

    spec = get_model(champion) if champion not in BASELINES["imputation"] else None
    with ui.spinner(f"{champion} · filling {int(gaps.sum()):,} missing cells", done="Gaps filled"):
        filled = (chain.fill(spec, X, table.kinds, gaps, max_context=max_context_rows, seed=seed) if spec
                  else _impute_baseline(champion, X, table.kinds, gaps, seed))
    out = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for j, name in enumerate(table.names):
        rows_j = gaps[:, j]
        if rows_j.any():
            values = table.decode_column(j, filled[rows_j, j]).to_numpy()
            if out[name].dtype.kind in "iu":
                out[name] = out[name].astype("float64")
            elif out[name].dtype.kind == "b":
                out[name] = out[name].astype(object)
            out.loc[rows_j, name] = values
            mask.loc[rows_j, name] = True
    return ImputeResult(data=out, filled=mask, model=champion, evaluation=evaluation,
                        attribution=_attribution([spec] if spec else []), seconds=time.monotonic() - t0)


# ---------------------------------------------------------------- synthetic data
@dataclass
class SyntheticResult(_Result):
    """`data`: the new rows. evaluation: how hard each generator's rows are to tell apart from yours
    (detection AUC: 0.5 = indistinguishable, 1 = obvious) and how many are exact copies of a real row."""

    data: pd.DataFrame
    model: str
    evaluation: pd.DataFrame
    attribution: str | None = None
    seconds: float = 0.0
    title = "Synthetic data"

    def _summary(self) -> list[str]:
        return [f"generator: {self.model} · {len(self.data):,} rows"]


def _detection_auc(real: np.ndarray, fake: np.ndarray, seed: int) -> float:
    """A classifier tries to tell real rows from synthetic ones (classifier two-sample test)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    n = min(len(real), len(fake), 5000)
    rng = np.random.default_rng(seed)
    a = real[rng.choice(len(real), n, replace=False)]
    b = fake[rng.choice(len(fake), n, replace=False)]
    Xy = np.vstack([a, b])
    lab = np.r_[np.zeros(n), np.ones(n)]
    auc = cross_val_score(HistGradientBoostingClassifier(max_iter=100, random_state=seed), Xy, lab, cv=3,
                          scoring="roc_auc")
    return float(np.mean(auc))


def synthesize(data: Any, n: int | None = None, *, models: Iterable[str] | None = None,
               noncommercial: Iterable[str] = (), drop: Iterable[str] | None = None, temperature: float = 1.0,
               max_context_rows: int = 2000, seed: int = 0, progress: bool = True) -> SyntheticResult:
    """`n` new rows (default: as many as `data`) with the same columns and relations.

    Each foundation model generates, a classifier tries to tell its rows from yours, and the hardest to tell
    apart wins (detection AUC closest to 0.5), against sampling every column on its own. Values are drawn from
    real values in the predicted range, so types and formats stay yours. Synthetic is not anonymous: check
    `exact_copies` and drop identifying columns before sharing.
    """
    ui = Console(progress)
    t0 = time.monotonic()
    df = _read(data)
    n = int(n or len(df))
    table = Table.fit(df, drop=list(drop or []))
    X = table.transform(df)
    codes = Codes.fit(X, table.kinds)
    specs = candidates("synthetic_data", models, noncommercial)
    methods: list[tuple[str, ModelSpec | None]] = [(s.name, s) for s in specs]
    methods += [(b, None) for b in _baselines("synthetic_data", models)]
    real_keys = {tuple(np.round(r, 6)) for r in np.nan_to_num(X, nan=-9e9)}
    rows, made, errors = [], {}, {}
    for name, spec in methods:
        label = spec.display if spec else name
        try:
            with ui.spinner(f"{label} · generating {n:,} rows", done=f"{label} done"):
                if spec:
                    fake = chain.sample(spec, X, codes, n, temperature=temperature, max_context=max_context_rows,
                                        seed=seed)
                else:
                    rng = np.random.default_rng(seed)
                    fake = np.column_stack([X[rng.integers(len(X), size=n), j] for j in range(X.shape[1])])
            made[name] = fake
            copies = np.mean([tuple(np.round(r, 6)) in real_keys for r in np.nan_to_num(fake, nan=-9e9)])
            rows.append({"model": name, "detection_auc": _detection_auc(X, fake, seed), "exact_copies": float(copies),
                         "kind": "foundation model" if spec else "classic"})
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            ui.say(f"  {ui.warn('!')} {label} did not finish: {str(e)[:80]}")
    if not rows:
        raise ValidationError(f"No generator finished: {errors}")
    evaluation = pd.DataFrame(rows).set_index("model").sort_values("detection_auc")
    champion = str(evaluation.index[0])
    ui.podium([(r, evaluation.loc[r, "detection_auc"], None) for r in evaluation.index],
              "detection AUC (0.5 = indistinguishable)", _attribution(specs), higher_is_better=False)
    spec = get_model(champion) if champion not in BASELINES["synthetic_data"] else None
    return SyntheticResult(data=table.decode(made[champion]), model=champion, evaluation=evaluation,
                           attribution=_attribution([spec] if spec else []), seconds=time.monotonic() - t0)


# ---------------------------------------------------------------- clustering
@dataclass
class ClusterResult(_Result):
    """`labels`: the segment of each row (same index as your data); `profiles`: size and what sets each apart."""

    labels: pd.Series
    k: int
    model: str
    profiles: pd.DataFrame
    evaluation: pd.DataFrame
    attribution: str | None = None
    seconds: float = 0.0
    title = "Clustering"

    def _summary(self) -> list[str]:
        return [f"model: {self.model} · {self.k} segments", "", self.profiles.to_string()]


def _pick_k(E: np.ndarray, k: int | None, max_k: int, seed: int) -> tuple[np.ndarray, int]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if k is not None:
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(E), k
    best = (-2.0, None, 2)
    sample = np.random.default_rng(seed).choice(len(E), min(len(E), 2000), replace=False)
    for kk in range(2, min(max_k, len(E) - 1) + 1):
        lab = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(E)
        if len(np.unique(lab[sample])) < 2:
            continue
        s = silhouette_score(E[sample], lab[sample])
        if s > best[0]:
            best = (s, lab, kk)
    if best[1] is None:
        return np.zeros(len(E), dtype=int), 1
    return best[1], best[2]


def _profiles(df: pd.DataFrame, table: Table, X: np.ndarray, lab: np.ndarray) -> pd.DataFrame:
    Z = _standardized(X)
    rows = []
    for c in np.unique(lab):
        m = lab == c
        diff = Z[m].mean(axis=0) - Z.mean(axis=0)
        notes = []
        for j in np.argsort(-np.abs(diff))[:3]:
            if abs(diff[j]) < 0.3:
                continue
            col = table.columns[j]
            if col.kind == "categorical":
                top = df.loc[m, col.name].astype(str).value_counts(normalize=True)
                notes.append(f"{col.name} mostly {top.index[0]} ({top.iloc[0]:.0%})")
            else:
                notes.append(f"{col.name} {'higher' if diff[j] > 0 else 'lower'} ({diff[j]:+.1f} sd)")
        rows.append({"segment": int(c), "rows": int(m.sum()), "share": round(float(m.mean()), 3),
                     "what sets it apart": "; ".join(notes) or "close to the average"})
    return pd.DataFrame(rows).set_index("segment")


def cluster(data: Any, k: int | None = None, *, models: Iterable[str] | None = None, labels: Any = None,
            noncommercial: Iterable[str] = (), drop: Iterable[str] | None = None, max_k: int = 10, seed: int = 0,
            progress: bool = True) -> ClusterResult:
    """Group the rows of `data` into `k` segments (default: the k between 2 and `max_k` that separates best).

    ZEUS embeds the rows zero-shot and k-means groups them; k-means and a Gaussian mixture on your standardized
    columns compete. With `labels` (known groups, to check) the best adjusted Rand index wins; without, the
    foundation model wins when it ran, since its pretraining is what makes it better than k-means on raw columns.
    """
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    from sklearn.neighbors import KNeighborsClassifier

    ui = Console(progress)
    t0 = time.monotonic()
    df = _read(data)
    truth = None
    if isinstance(labels, str):
        truth = df[labels].to_numpy()
        df = df.drop(columns=[labels])
    elif labels is not None:
        truth = np.asarray(labels)
    table = Table.fit(df, drop=list(drop or []))
    X = table.transform(df)
    Z = _standardized(X)
    specs = candidates("clustering", models, noncommercial)
    methods: list[tuple[str, ModelSpec | None]] = [(s.name, s) for s in specs]
    methods += [(b, None) for b in _baselines("clustering", models)]
    rng = np.random.default_rng(seed)
    found, rows, errors = {}, [], {}
    for name, spec in methods:
        label = spec.display if spec else name
        try:
            with ui.spinner(f"{label} · grouping {len(df):,} rows", done=f"{label} done"):
                if spec:
                    cap = int(spec.extra.get("max_rows", 2000))
                    part = np.sort(rng.choice(len(X), cap, replace=False)) if len(X) > cap else np.arange(len(X))
                    lab_part, kk = _pick_k(spec.embed_fn(X[part]), k, max_k, seed)
                    lab = (lab_part if len(part) == len(X) else
                           KNeighborsClassifier(n_neighbors=5).fit(Z[part], lab_part).predict(Z))
                    lab[part] = lab_part
                elif name == "kmeans":
                    lab, kk = _pick_k(Z, k, max_k, seed)
                else:
                    kk = k or _pick_k(Z, None, max_k, seed)[1]
                    lab = GaussianMixture(n_components=kk, random_state=seed).fit_predict(Z)
            sample = rng.choice(len(Z), min(len(Z), 2000), replace=False)
            row = {"model": name, "k": int(kk), "kind": "foundation model" if spec else "classic",
                   "silhouette": (float(silhouette_score(Z[sample], lab[sample]))
                                  if len(np.unique(lab[sample])) > 1 else None)}
            if truth is not None:
                row["adjusted_rand"] = float(adjusted_rand_score(truth, lab))
            rows.append(row)
            found[name] = (lab, int(kk))
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
            ui.say(f"  {ui.warn('!')} {label} did not finish: {str(e)[:80]}")
    if not rows:
        raise ValidationError(f"No method finished: {errors}")
    evaluation = pd.DataFrame(rows).set_index("model")
    if truth is not None:
        evaluation = evaluation.sort_values("adjusted_rand", ascending=False)
        champion = str(evaluation.index[0])
        ui.podium([(r, evaluation.loc[r, "adjusted_rand"], None) for r in evaluation.index], "adjusted Rand index",
                  _attribution(specs))
    else:
        fms = [r for r in evaluation.index if evaluation.loc[r, "kind"] == "foundation model"]
        champion = fms[0] if fms else str(evaluation.index[0])
    lab, kk = found[champion]
    spec = get_model(champion) if champion not in BASELINES["clustering"] else None
    res = ClusterResult(labels=pd.Series(lab, index=df.index, name="segment"), k=kk, model=champion,
                        profiles=_profiles(df, table, X, lab), evaluation=evaluation,
                        attribution=_attribution([spec] if spec else []), seconds=time.monotonic() - t0)
    ui.say(f"{ui.accent('◆')} {kk} segments by {champion} " + ui.dim("· .profiles says what sets each apart"))
    return res


__all__ = ["anomalies", "impute", "synthesize", "cluster", "AnomalyResult", "ImputeResult", "SyntheticResult",
           "ClusterResult", "candidates"]
