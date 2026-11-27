"""The local engine: the championship and predictions on your own machine. Nothing leaves it.

    from zerograd import ZeroGrad
    zg = ZeroGrad()                                   # local: no API key, no account
    cap = zg.create("titanic", examples=train_df, target="Survived")
    cap.evaluation                                    # every model on the same holdout; the champion is marked
    cap.predict(new_df)
    cap.save("titanic.zg"); cap = zerograd.load("titanic.zg")
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .._ui import Console, error_summary, fmt_value, phrase
from .._version import __version__
from ..curve import ContextCurve, check_sizes
from ..curve import ms_per_row as _ms_per_row
from ..errors import ValidationError
from ..models import FittedModel, ModelSpec, all_models, get_model, resolve_models
from . import championship as champ
from .preprocess import Preprocessor

FORMAT = "zerograd.local/1"
_UNSET: Any = object()


class LocalZeroGrad:
    """Runs the championship and the champion's predictions in this process.

    `models` picks which models compete (default: every commercial model that is installed).
    `noncommercial` names non-commercial models to add, for research and evaluation only; each shows
    its license notice and downloads its weights from the author's repository with your own account.
    """

    remote = False

    def __init__(self, *, models: Iterable[str] | None = None, noncommercial: Iterable[str] = (),
                 progress: bool = True, max_context_rows: int = 10_000, max_holdout_rows: int = 20_000,
                 holdout_fraction: float = 0.2, max_columns: int = 100, max_classes: int = 10, min_rows: int = 20,
                 predict_chunk_rows: int = 10_000):
        self._models = list(models) if models is not None else None
        self._noncommercial = list(noncommercial) if not isinstance(noncommercial, bool) else noncommercial
        self.specs = resolve_models(self._models, self._noncommercial)  # validates names, shows license notices
        self.progress = progress
        self.ui = Console(progress)
        self.ui.greet("local", _models_line(self.specs))
        self.limits = dict(max_context_rows=max_context_rows, max_holdout_rows=max_holdout_rows,
                           holdout_fraction=holdout_fraction, max_columns=max_columns, max_classes=max_classes,
                           min_rows=min_rows, predict_chunk_rows=predict_chunk_rows)

    def __repr__(self) -> str:
        return f"<ZeroGrad local · {', '.join(s.name for s in self.specs) or 'no models installed'}>"

    def models(self) -> pd.DataFrame:
        """Every known model with its license, and whether it is installed and selected here."""
        from ..models import list_models

        df = list_models().to_frame()
        df["selected"] = df.index.isin([s.name for s in self.specs])
        return df

    def create(self, name: str, examples: Any, target: str, *, drop: Iterable[str] | None = None,
               metric: str | None = None, task: str | None = None) -> "LocalCapability":
        """Run the championship on labeled history and return the capability with its champion.

        `examples` is a DataFrame or a CSV/Parquet path. `task` is "classification" or "regression"; left out,
        a numeric target with decimals or more than `max_classes` distinct values is regression.
        `metric` picks the champion. Classification: roc_auc (default), pr_auc, log_loss, accuracy,
        balanced_accuracy, f1, precision or recall (binary), f1_macro (multiclass). Regression: rmse (default),
        mae or r2.
        """
        from .._remote import to_frame

        if isinstance(examples, str) and (examples.startswith("gs://") or examples.startswith("upl_")):
            raise ValidationError("Bucket URIs and upload ids need the platform: use ZeroGrad(remote=True).")
        if metric is not None:
            champ.check_metric(metric)
        df = to_frame(examples)
        lim = self.limits
        pre, kept = Preprocessor.fit(df, target, drop=list(drop or []), max_columns=lim["max_columns"],
                                     max_classes=lim["max_classes"], min_rows=lim["min_rows"], task=task)
        metric = champ.check_metric(metric, pre.task)
        ui = Console(self.progress)
        specs = self._for_task(pre.task, ui)
        ui.done(f"{len(kept):,} labeled rows · {len(pre.columns)} columns used"
                + (f" · {_describe_target(pre)}" if pre.regression else "")
                + (ui.dim(f" · dropped {', '.join(d['name'] for d in pre.dropped)}") if pre.dropped else ""))
        _prefetch(specs, ui, pre.task)
        ui.say(f"{ui.accent('◆')} Championship · {len(specs)} model{'s' if len(specs) != 1 else ''} · "
               f"{ui.dim(phrase('championship'))}")
        live = _Live(ui)
        t0 = time.monotonic()
        try:
            result = champ.run_championship(
                pre, kept, specs, max_context_rows=lim["max_context_rows"],
                holdout_fraction=lim["holdout_fraction"], chunk=lim["predict_chunk_rows"], progress=live.progress,
                max_holdout_rows=lim["max_holdout_rows"], metric=metric,
            )
        finally:
            live.close()
        result.rows["context_limit"] = int(lim["max_context_rows"])
        cap = LocalCapability(name=name, pre=pre, result=result, metric=metric, seconds=time.monotonic() - t0,
                              noncommercial=[s.name for s in specs if not s.commercial],
                              chunk=lim["predict_chunk_rows"], progress=self.progress)
        ranked = sorted(result.candidates, key=champ.rank_key(metric), reverse=True)
        base_name, base_label = ("baseline:mean", "always the average") if pre.regression else (
            "baseline:majority", "majority class")
        base = next((b for b in result.baselines or [] if b.model == base_name), None)
        label, higher, _ = champ.METRICS[metric]
        ui.podium([(c.display, c.metrics.get(metric), c.error) for c in ranked], label,
                  cap.attribution, baseline=base.metrics.get(metric) if base else None,
                  higher_is_better=higher, baseline_label=base_label)
        ui.say(f"  {ui.dim(phrase('champion'))}")
        return cap

    # ------------------------------------------------------------------ tasks with no target
    def anomalies(self, data: Any, *, labels: Any = None, reference: Any = None,
                  models: Iterable[str] | None = None, contamination: float = 0.05, **options: Any):
        """Score every row by how unusual it is, with the columns why. See zerograd.unsupervised.anomalies."""
        from ..unsupervised import anomalies

        return anomalies(data, labels=labels, reference=reference, models=models, contamination=contamination,
                         noncommercial=self._nc_names(), progress=self.progress, **options)

    def impute(self, data: Any, *, models: Iterable[str] | None = None, **options: Any):
        """Fill missing values with the champion of a championship on hidden cells. See zerograd.unsupervised.impute."""
        from ..unsupervised import impute

        return impute(data, models=models, noncommercial=self._nc_names(), progress=self.progress, **options)

    def synthesize(self, data: Any, n: int | None = None, *, models: Iterable[str] | None = None, **options: Any):
        """New rows like yours, scored by how hard they are to tell apart. See zerograd.unsupervised.synthesize."""
        from ..unsupervised import synthesize

        return synthesize(data, n, models=models, noncommercial=self._nc_names(), progress=self.progress, **options)

    def cluster(self, data: Any, k: int | None = None, *, models: Iterable[str] | None = None, **options: Any):
        """Segments, and what sets each apart. See zerograd.unsupervised.cluster."""
        from ..unsupervised import cluster

        return cluster(data, k, models=models, noncommercial=self._nc_names(), progress=self.progress, **options)

    def _nc_names(self) -> list[str]:
        return [] if isinstance(self._noncommercial, bool) else list(self._noncommercial)

    def cross_validate(self, examples: Any, target: str, *, folds: int = 5, groups: str | None = None,
                       time: str | None = None, gap: int = 0, metric: str | None = None,
                       drop: Iterable[str] | None = None, task: str | None = None, seed: int = 0):
        """Every selected model on `folds` folds: the mean and spread of each metric, next to a baseline.

        Split at random (stratified), by group (`groups="customer_id"`: a group never sits on both sides) or by
        time (`time="date"`: train on the past, score the next block; `gap` rows left out in between). The group
        and time columns are not used as features. Binary tasks add KS and lift in the top 10%.
        Returns a CVResult: `.summary` per model, `.folds` per fold, `.best`.
        """
        from .._remote import to_frame
        from .validation import run_cv

        df = to_frame(examples)
        for col, what in ((groups, "groups"), (time, "time")):
            if col is not None and col not in df.columns:
                raise ValidationError(f"{what}='{col}' is not a column of the data.", code="missing_columns")
        if metric is not None:
            champ.check_metric(metric)
        lim = self.limits
        extra = [c for c in (groups, time) if c is not None]
        pre, kept = Preprocessor.fit(df, target, drop=list(drop or []) + extra, max_columns=lim["max_columns"],
                                     max_classes=lim["max_classes"], min_rows=lim["min_rows"], task=task)
        metric = champ.check_metric(metric, pre.task)
        g = kept[groups].astype(str).to_numpy() if groups else None
        t = None
        if time:
            ts = pd.to_datetime(kept[time], errors="coerce", utc=True) if not pd.api.types.is_numeric_dtype(
                kept[time]) else pd.to_numeric(kept[time], errors="coerce")
            if ts.isna().any():
                raise ValidationError(f"time='{time}' has {int(ts.isna().sum())} values that are not dates or numbers.",
                                      code="invalid_time")
            t = ts.to_numpy()
        ui = Console(self.progress)
        specs = self._for_task(pre.task, ui)
        _prefetch(specs, ui, pre.task)
        how = "by group" if groups else "by time" if time else "stratified"
        ui.say(f"{ui.accent('◆')} Cross-validation · {folds} folds {how} · {len(specs)} "
               f"model{'s' if len(specs) != 1 else ''} · {len(kept):,} rows")
        state: dict[str, Any] = {}

        def on_fold(k: int, n: int, spec: ModelSpec) -> None:
            if state.get("k") != k:
                if state.get("cm"):
                    state["cm"].__exit__(None, None, None)
                    ui.done(f"Fold {state['k'] + 1} of {n}")
                state["k"], state["cm"] = k, ui.spinner(f"Fold {k + 1} of {n}")
                state["sp"] = state["cm"].__enter__()
            state["sp"].update(suffix=ui.dim(f" · {spec.display}"))

        try:
            res = run_cv(pre, kept, specs, folds=folds, groups=g, times=t, gap=gap, metric=metric,
                         max_context_rows=lim["max_context_rows"], chunk=lim["predict_chunk_rows"], seed=seed,
                         on_fold=on_fold)
        finally:
            if state.get("cm"):
                state["cm"].__exit__(None, None, None)
        label = champ.METRICS[metric][0]
        summ = res.summary
        ui.podium([(m, summ.loc[m, metric], None) for m in summ.index if not str(m).startswith("baseline:")],
                  f"mean {label}", None, higher_is_better=champ.METRICS[metric][1])
        return res

    def context_curve(self, examples: Any, target: str, sizes: Iterable[int] | None = None, test: Any = None, *,
                      model: str | None = None, metric: str | None = None, seed: int = 0,
                      tolerance: float | None = None, drop: Iterable[str] | None = None,
                      task: str | None = None) -> ContextCurve:
        """Fit one model on growing samples of the context and score each on the same holdout, with timings.

        Finds the trade-off between speed and quality: the recommended size is the smallest context whose
        `metric` is within `tolerance` of the best. `sizes` defaults to 250, 500, 1K, 2K, 5K, 10K, ... up to
        every training row; `test` defaults to a stratified 20% of `examples` that no context ever sees.
        `model` defaults to the first selected model. Returns a ContextCurve (a DataFrame with `.plot()`).
        `task` works as in `create`. `tolerance` defaults to 0.005 (absolute), or 0.5% of the best for RMSE and MAE.
        """
        from .._remote import to_frame

        if metric is not None:
            champ.check_metric(metric)
        df = to_frame(examples)
        lim = self.limits
        pre, kept = Preprocessor.fit(df, target, drop=list(drop or []), max_columns=lim["max_columns"],
                                     max_classes=lim["max_classes"], min_rows=lim["min_rows"], task=task)
        metric = champ.check_metric(metric, pre.task)
        X, y = pre.transform(kept), pre.encode_target(kept[target])
        if test is None:
            train_idx, hold_idx = champ.holdout_split(champ.strata(y, pre.task), lim["holdout_fraction"], seed=seed,
                                                      max_rows=lim["max_holdout_rows"])
            Xc, yc, Xt, yt = X[train_idx], y[train_idx], X[hold_idx], y[hold_idx]
        else:
            t = to_frame(test)
            if target not in t.columns:
                raise ValidationError(f"test needs the target column '{target}' to be scored.", code="missing_target")
            t = t[t[target].notna()]
            Xc, yc, Xt, yt = X, y, pre.transform(t), pre.encode_target(t[target])
        spec = self._pick(model, pre.task)
        ui = Console(self.progress)
        _prefetch([spec], ui, pre.task)
        return _context_curve(spec, Xc, yc, Xt, yt, len(pre.classes), check_sizes(sizes, len(yc)), metric=metric,
                              seed=seed, tolerance=tolerance, chunk=lim["predict_chunk_rows"], ui=ui, task=pre.task)

    def _pick(self, model: str | None, task: str | None = None) -> ModelSpec:
        if model is None:
            specs = [s for s in self.specs if task is None or s.runs(task)]
            if not specs:
                raise ValidationError("No installed model runs this task yet: pip install zerograd", code="no_models")
            return specs[0]
        spec = next((s for s in self.specs if model in (s.name, s.display)), None)
        if spec is None:
            spec = resolve_models([model], self._noncommercial)[0]
        if task is not None and not spec.runs(task):
            raise ValidationError(f"{spec.display} does not run {task} in this version of the SDK.",
                                  code="unsupported_task")
        return spec

    def _for_task(self, task: str, ui: Console) -> list[ModelSpec]:
        """The selected models that run this task; the others sit out with a note."""
        specs = [s for s in self.specs if s.runs(task)]
        out = [s for s in self.specs if not s.runs(task)]
        if out:
            sits = "it sits" if len(out) == 1 else "they sit"
            ui.say(ui.dim(f"  {', '.join(s.display for s in out)}: {task} is not in this version of the SDK yet, "
                          f"so {sits} out"))
        if not specs:
            raise ValidationError(f"None of the selected models runs {task} in this version of the SDK. Models that "
                                  f"do: {', '.join(m.name for m in all_models() if m.runs(task) and m.commercial)}.",
                                  code="unsupported_task")
        return specs


class LocalCapability:
    """A champion and its context, ready to predict here. Save it with `save`, open it with `zerograd.load`."""

    def __init__(self, *, name: str, pre: Preprocessor, result: champ.ChampionshipResult | None = None,
                 metric: str = champ.DEFAULT_METRIC, seconds: float = 0.0, noncommercial: list[str] | None = None,
                 chunk: int = 10_000, progress: bool = True, _state: dict | None = None):
        self.name = name
        self.pre = pre
        self.metric = metric
        self.threshold: float | None = None
        self._chunk = chunk
        self._progress = progress
        self._fitted: FittedModel | None = None
        if _state is not None:  # loaded from disk
            self.__dict__.update(_state)
            return
        assert result is not None
        self.champion = result.champion
        self.candidates = [c.to_dict() for c in result.candidates]
        self.baseline_rows = [c.to_dict() for c in result.baselines or []]
        self.rows = result.rows
        self.context_X = result.context_X
        self.context_y = result.context_y
        self.seconds = seconds
        self.noncommercial = noncommercial or []
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ------------------------------------------------------------------ what it is
    def __repr__(self) -> str:
        return f"<LocalCapability {self.name} · {self.pre.task} · champion {self.champion_display}>"

    @property
    def spec(self) -> ModelSpec:
        return get_model(self.champion)

    @property
    def champion_display(self) -> str:
        return next((c["display"] for c in self.candidates if c["model"] == self.champion), self.champion)

    @property
    def classes(self) -> list:
        """The labels, in the order of the proba_ columns. Empty for regression."""
        return list(self.pre.classes)

    @property
    def task(self) -> str:
        return self.pre.task

    @property
    def expected_columns(self) -> list[str]:
        return self.pre.feature_names

    @property
    def columns(self) -> dict:
        """Columns used (with their kind) and dropped (with the reason)."""
        return self.pre.describe()

    @property
    def attribution(self) -> str | None:
        """The attribution the champion's license asks you to show, if any (e.g. Built with PriorLabs-TabPFN)."""
        try:
            return self.spec.attribution
        except ValueError:
            return None

    @property
    def errors(self) -> dict[str, str]:
        """The whole error message of each model that did not finish, by model name."""
        return {c["model"]: c["error"] for c in self.candidates if c.get("error")}

    @property
    def evaluation(self) -> pd.DataFrame:
        """The championship: every model scored on the same holdout."""
        df = pd.DataFrame(self.candidates)
        df["champion"] = df["model"] == self.champion
        metric_cols = [m for m, (_, _, tasks) in champ.METRICS.items() if self.pre.task in tasks and m in df]
        cols = [c for c in ("display", *metric_cols, "seconds", "champion", "error") if c in df]
        df = df[cols].rename(columns={"display": "model"}).set_index("model")
        if "error" in df and df["error"].isna().all():
            df = df.drop(columns="error")
        elif "error" in df:  # the first line, so a table stays readable; the whole message is in `errors`
            df["error"] = df["error"].map(lambda e: error_summary(e) if isinstance(e, str) else e)
        df.attrs["rows"] = self.rows
        df.attrs["metric"] = self.metric
        return df

    @property
    def baselines(self) -> pd.DataFrame:
        """What you get without a model on the same holdout: random scores and always the majority class
        (regression: always the average, always the median)."""
        df = pd.DataFrame(self.baseline_rows)
        keep = [c for c in ("display", "roc_auc", "accuracy", "log_loss", "rmse", "mae", "r2") if c in df]
        return df[keep].rename(columns={"display": "baseline"}).set_index("baseline") if len(df) else df

    def context_curve(self, test: Any, sizes: Iterable[int] | None = None, *, metric: str | None = None,
                      seed: int = 0, tolerance: float | None = None) -> ContextCurve:
        """The champion on growing samples of this capability's own context, scored on `test` (with the target).

        Use it to serve with a smaller, faster context: `cap.update(context_rows=curve.recommended)`.
        """
        from .._remote import to_frame

        metric = champ.check_metric(metric or self.metric, self.pre.task)
        t = to_frame(test)
        target = self.pre.target
        if target not in t.columns:
            raise ValidationError(f"test needs the target column '{target}' to be scored.", code="missing_target")
        t = t[t[target].notna()]
        ui = Console(self._progress)
        _prefetch([self.spec], ui, self.task)
        return _context_curve(self.spec, self.context_X, self.context_y, self.pre.transform(t),
                              self.pre.encode_target(t[target]), len(self.classes),
                              check_sizes(sizes, len(self.context_y)), metric=metric, seed=seed, tolerance=tolerance,
                              chunk=self._chunk, ui=ui, task=self.task)

    def update(self, *, threshold: Any = _UNSET, context_rows: int | None = None) -> "LocalCapability":
        """Binary tasks: score >= threshold predicts the positive class. None resets to 0.5.

        `context_rows` keeps a stratified sample of that many context rows (smaller is faster; see
        `context_curve`). It only shrinks: the rows left out are gone from this capability.
        """
        if context_rows is not None:
            n = int(context_rows)
            if n < 2:
                raise ValidationError("context_rows is a row count, at least 2.", code="invalid_context_rows")
            if n < len(self.context_y):
                keep = champ.stratified_sample(champ.strata(self.context_y, self.task), n, seed=0)
                self.context_X, self.context_y = self.context_X[keep], self.context_y[keep]
                self._fitted = None
            if threshold is _UNSET:
                return self
        return self._set_threshold(None if threshold is _UNSET else threshold)

    def add_examples(self, examples: Any, *, max_context_rows: int | None = None) -> "LocalCapability":
        """Add labeled rows to the context, with no training and no new championship.

        The next prediction already reads them. New rows always stay; when the context would pass
        `max_context_rows` (default: its size limit when the capability was created, 10,000), the older rows are
        sampled down, stratified, to make room. The champion and `evaluation` don't change: to re-elect the champion
        on fresh history, run `zg.create` again. Rows without the target are skipped.
        """
        from .._remote import to_frame

        df = to_frame(examples)
        target = self.pre.target
        if target not in df.columns:
            raise ValidationError(f"The new examples need the target column '{target}'.", code="missing_target")
        df = df[df[target].notna()]
        if df.empty:
            return self
        if not self.pre.regression:
            known = {str(c) for c in self.classes}
            unseen = sorted({str(v) for v in df[target].unique()} - known)
            if unseen:
                raise ValidationError(
                    f"'{target}' has values this capability has never seen: {', '.join(unseen[:5])}. It predicts "
                    f"{', '.join(map(str, self.classes))}; to add a class, create the capability again.",
                    code="unseen_classes")
        try:
            X_new, y_new = self.pre.transform(df), self.pre.encode_target(df[target])
        except (KeyError, ValueError) as exc:
            raise ValidationError(f"Could not read the new examples' '{target}' values ({exc}).",
                                  code="invalid_target") from None
        limit = int(max_context_rows or self.rows.get("context_limit") or max(10_000, len(self.context_y)))
        if len(y_new) >= limit:  # only the newest rows fit
            X_new, y_new = X_new[-limit:], y_new[-limit:]
            keep = np.zeros(0, dtype=np.int64)
        else:
            room = limit - len(y_new)
            keep = (np.arange(len(self.context_y)) if len(self.context_y) <= room
                    else champ.stratified_sample(champ.strata(self.context_y, self.task), room, seed=0))
        self.context_X = np.concatenate([self.context_X[keep], X_new.astype(self.context_X.dtype)])
        self.context_y = np.concatenate([self.context_y[keep], y_new.astype(self.context_y.dtype)])
        self.rows = {**self.rows, "labeled": int(self.rows.get("labeled", 0)) + int(len(df)),
                     "production_context": int(len(self.context_y)),
                     "added": int(self.rows.get("added", 0)) + int(len(df))}
        self._fitted = None
        return self

    def _set_threshold(self, threshold: float | None) -> "LocalCapability":
        if threshold is not None:
            if self.pre.task != "binary":
                raise ValidationError("A threshold only applies to binary capabilities.", code="invalid_threshold")
            if not 0 < float(threshold) < 1:
                raise ValidationError("threshold is a number between 0 and 1.", code="invalid_threshold")
        self.threshold = threshold
        return self

    # ------------------------------------------------------------------ threshold and explanations
    def optimize_threshold(self, objective: str = "f1", *, cost_fp: float = 1.0, cost_fn: float = 1.0,
                           min_precision: float | None = None, min_recall: float | None = None,
                           data: Any = None, apply: bool = True):
        """Binary tasks: the score cut that is best for `objective`, read off the championship holdout.

        objective: "f1"; "cost" (fewest cost_fp * false positives + cost_fn * false negatives); "youden"
        (recall minus false positive rate); "recall" (most recall with precision >= min_precision); or "precision"
        (most precision with recall >= min_recall). `data` (with the target) scores those rows instead.
        `apply=True` makes it the capability's threshold. Returns a ThresholdResult with the table behind it.
        """
        from .threshold import optimize

        if self.pre.task != "binary":
            raise ValidationError("A threshold only applies to binary capabilities.", code="invalid_threshold")
        if data is not None:
            y, score = self._labeled(data)
            res = optimize(objective, y=y, score=score, cost_fp=cost_fp, cost_fn=cost_fn,
                           min_precision=min_precision, min_recall=min_recall, source=f"{len(y):,} rows you passed")
        else:
            champion = next((c for c in self.candidates if c["model"] == self.champion), {})
            counts = (champion.get("plots") or {}).get("thresholds")
            if not counts:
                raise ValidationError("This capability has no holdout counts to read; pass data= with the target.",
                                      code="invalid_threshold")
            res = optimize(objective, counts=counts, cost_fp=cost_fp, cost_fn=cost_fn, min_precision=min_precision,
                           min_recall=min_recall, source=f"championship holdout ({self.rows.get('holdout', 0):,} rows)")
        if apply:
            self._set_threshold(res.threshold)
        return res

    def classification_report(self, data: Any = None, *, threshold: float | None = None) -> pd.DataFrame:
        """Precision, recall, F1 and support per class, plus accuracy, macro and weighted averages, like
        scikit-learn's classification_report, as a DataFrame.

        Read off the championship holdout, or off `data` (rows with the target) if you pass it. Binary tasks cut
        the score at `threshold`, else at the capability's threshold (`optimize_threshold`), else at 0.5.
        """
        from sklearn.metrics import confusion_matrix

        from .report import binary_confusion, from_confusion

        if self.pre.regression:
            raise ValidationError("A classification report needs a classification capability; this one is a "
                                  "regression.", code="unsupported_task")
        cut = threshold if threshold is not None else (self.threshold if self.threshold is not None else 0.5)
        binary = self.pre.task == "binary"
        if data is not None:
            y, X = self._labeled(data, raw=True)
            proba = champ.predict_in_chunks(self._model(), X, self._chunk)
            pred = (proba[:, 1] >= cut).astype(int) if binary else proba.argmax(axis=1)
            cm = confusion_matrix(y, pred, labels=list(range(len(self.classes))))
            source = f"{len(y):,} rows you passed"
        else:
            plots = next((c for c in self.candidates if c["model"] == self.champion), {}).get("plots") or {}
            if binary and plots.get("thresholds"):
                cm = binary_confusion(plots["thresholds"], cut)
            elif not binary and plots.get("confusion"):
                cm = np.asarray(plots["confusion"])
            else:
                raise ValidationError("This capability has no holdout counts to read; pass data= with the target.",
                                      code="missing_holdout")
            source = f"championship holdout ({self.rows.get('holdout', 0):,} rows)"
        df = from_confusion(cm, self.classes)
        df.attrs.update({"source": source, "threshold": cut if binary else None, "model": self.champion_display})
        return df

    def importance(self, data: Any, *, metric: str | None = None, repeats: int = 3, max_rows: int = 2000,
                   seed: int = 0) -> pd.DataFrame:
        """How much each column matters: shuffle it in `data` (rows with the target) and measure how much the
        champion's `metric` gets worse. Bigger is more important; near zero, the column could go.

        Pass rows the model has not seen as context (a test set), or the numbers come out optimistic. One
        prediction pass per column and repeat, on up to `max_rows` rows.
        """
        metric = champ.check_metric(metric or self.metric, self.pre.task)
        y, X = self._labeled(data, raw=True)
        rng = np.random.default_rng(seed)
        if len(y) > max_rows:
            keep = champ.stratified_sample(champ.strata(y, self.task), max_rows, seed=seed)
            X, y = X[keep], y[keep]
        model = self._model()

        def value(Xs: np.ndarray) -> float:
            out = champ.predict_in_chunks(model, Xs, self._chunk, self.task)
            m = (champ.regression_metrics(y, out) if self.pre.regression else champ.metrics(y, out, len(self.classes)))
            return champ.score(m, metric) or 0.0  # signed so that bigger is better

        ui = Console(self._progress)
        names = self.pre.feature_names
        rows = []
        with ui.spinner(f"Importance · {len(names)} columns × {repeats} · {len(y):,} rows",
                        done=f"Importance of {len(names)} columns") as sp:
            base = value(X)
            for j, name in enumerate(names):
                sp.update(suffix=ui.dim(f" · {name}"))
                drops = []
                for _ in range(repeats):
                    Xs = X.copy()
                    Xs[:, j] = Xs[rng.permutation(len(Xs)), j]
                    drops.append(base - value(Xs))
                rows.append({"column": name, "importance": float(np.mean(drops)),
                             "std": float(np.std(drops, ddof=1)) if repeats > 1 else 0.0})
        out = pd.DataFrame(rows).set_index("column").sort_values("importance", ascending=False)
        out.attrs.update(metric=metric, baseline=abs(base), rows=int(len(y)))
        return out

    def _labeled(self, data: Any, raw: bool = False):
        """(y, score) for binary, or with raw=True (y, X): rows of `data` that have the target."""
        from .._remote import to_frame

        t = to_frame(data)
        target = self.pre.target
        if target not in t.columns:
            raise ValidationError(f"data needs the target column '{target}'.", code="missing_target")
        t = t[t[target].notna()]
        y, X = self.pre.encode_target(t[target]), self.pre.transform(t)
        if raw:
            return y, X
        return y, champ.predict_in_chunks(self._model(), X, self._chunk)[:, 1]

    # ------------------------------------------------------------------ predict
    def predict(self, data: Any, *, threshold: float | None = None) -> pd.DataFrame | dict:
        """prediction, score (binary) and proba_<class>, aligned to the input. A dict (one row) returns a dict.

        Regression: one `prediction` column with the predicted number (a dict: {"prediction": 123.4})."""
        from .._remote import to_frame

        one = isinstance(data, dict)
        df = to_frame(data)
        X = self.pre.transform(df)
        model = self._model()
        if self.pre.regression:
            if threshold is not None:
                raise ValidationError("A threshold only applies to binary capabilities.", code="invalid_threshold")
            if len(X) >= 1000:
                ui = Console(self._progress)
                with ui.spinner(f"Scoring {len(X):,} rows with {self.champion_display} · {ui.dim(phrase('predict'))}",
                                done=f"Scored {len(X):,} rows"):
                    pred = champ.predict_in_chunks(model, X, self._chunk, self.task)
            else:
                pred = champ.predict_in_chunks(model, X, self._chunk, self.task) if len(X) else np.zeros(0)
            if one:
                return {"prediction": float(pred[0])}
            return pd.DataFrame({"prediction": pred}, index=df.index)
        if len(X) >= 1000:
            ui = Console(self._progress)
            with ui.spinner(f"Scoring {len(X):,} rows with {self.champion_display} · {ui.dim(phrase('predict'))}",
                            done=f"Scored {len(X):,} rows"):
                proba = champ.predict_in_chunks(model, X, self._chunk)
        else:
            proba = champ.predict_in_chunks(model, X, self._chunk) if len(X) else np.zeros((0, len(self.classes)))
        t = threshold if threshold is not None else self.threshold
        if self.pre.task == "binary":
            cut = 0.5 if t is None else float(t)
            idx = (proba[:, 1] >= cut).astype(int)
        else:
            idx = proba.argmax(axis=1)
        out = pd.DataFrame({"prediction": [self.classes[i] for i in idx]}, index=df.index)
        if self.pre.task == "binary":
            out["score"] = proba[:, 1]
        for j, cls in enumerate(self.classes):
            out[f"proba_{cls}"] = proba[:, j]
        if one:
            label = out["prediction"].iloc[0]
            rec: dict[str, Any] = {"prediction": label.item() if isinstance(label, np.generic) else label}
            if "score" in out:
                rec["score"] = float(out["score"].iloc[0])
            rec["proba"] = {str(c): float(out[f"proba_{c}"].iloc[0]) for c in self.classes}
            return rec
        return out

    def _model(self) -> FittedModel:
        if self._fitted is None:
            spec = self.spec
            ui = Console(self._progress)
            _prefetch([spec], ui, self.task)
            with ui.spinner(f"Warming up {spec.display} with {len(self.context_y):,} rows of context · "
                            f"{ui.dim(phrase('fit'))}", done=f"{spec.display} ready"):
                if self.pre.regression:
                    self._fitted = spec.fit_regressor(self.context_X, self.context_y, serving=True)
                else:
                    self._fitted = spec.fit(self.context_X, self.context_y, len(self.classes), serving=True)
        return self._fitted

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Write the capability (preprocessing, evaluation and context rows; no model weights) to one file.

        The file holds a sample of your labeled rows as the model's context: treat it like the data itself.
        """
        path = Path(path)
        meta = {
            "format": FORMAT, "zerograd_version": __version__, "name": self.name, "created_at": self.created_at,
            "preprocessor": self.pre.to_dict(), "metric": self.metric, "threshold": self.threshold,
            "champion": self.champion, "candidates": self.candidates, "baselines": self.baseline_rows,
            "rows": self.rows, "seconds": self.seconds, "noncommercial": self.noncommercial,
        }
        buf = io.BytesIO()
        np.savez_compressed(buf, X=self.context_X, y=self.context_y)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("capability.json", json.dumps(meta, indent=1, default=_json_default))
            z.writestr("context.npz", buf.getvalue())
        return path


def load(path: str | Path, *, noncommercial: Iterable[str] = (), progress: bool = True) -> LocalCapability:
    """Open a capability written by `LocalCapability.save`.

    A capability whose champion is non-commercial only opens with that model named in `noncommercial`.
    """
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read("capability.json"))
        ctx = np.load(io.BytesIO(z.read("context.npz")))
        X, y = ctx["X"], ctx["y"]
    if meta.get("format") != FORMAT:
        raise ValidationError(f"{path} is not a ZeroGrad local capability file.", code="invalid_file")
    resolve_models([meta["champion"]], list(noncommercial))  # same license gate as create
    state = {
        "champion": meta["champion"], "candidates": meta["candidates"], "baseline_rows": meta.get("baselines", []),
        "rows": meta["rows"], "context_X": X, "context_y": y, "seconds": meta.get("seconds", 0.0),
        "noncommercial": meta.get("noncommercial", []), "created_at": meta.get("created_at"),
        "threshold": meta.get("threshold"), "metric": meta.get("metric", champ.DEFAULT_METRIC),
    }
    return LocalCapability(name=meta["name"], pre=Preprocessor.from_dict(meta["preprocessor"]),
                           progress=progress, _state=state)


# ---------------------------------------------------------------------- helpers
def _context_curve(spec: ModelSpec, Xc: np.ndarray, yc: np.ndarray, Xt: np.ndarray, yt: np.ndarray, n_classes: int,
                   sizes: list[int], *, metric: str, seed: int, tolerance: float | None, chunk: int, ui: Console,
                   task: str | None = None) -> ContextCurve:
    """Fit `spec` on a stratified sample of each size, score it on (Xt, yt) and time the fit and the predictions."""
    label = champ.METRICS[metric][0]
    regression = task == champ.REGRESSION
    strata = champ.strata(yc, task)
    ui.say(f"{ui.accent('◆')} Context curve · {spec.display} · {len(sizes)} sizes · {len(yt):,} test rows")
    rows = []
    for size in sizes:
        idx = champ.stratified_sample(strata, size, seed=seed)
        row: dict[str, Any] = {"rows": int(len(idx))}
        with ui.spinner(f"{len(idx):,} rows of context") as sp:
            try:
                t0 = time.monotonic()
                fitted = (spec.fit_regressor(Xc[idx], yc[idx], serving=False) if regression
                          else spec.fit(Xc[idx], yc[idx], n_classes, serving=False))
                t1 = time.monotonic()
                out = champ.predict_in_chunks(fitted, Xt, chunk, task)
                t2 = time.monotonic()
                m = champ.regression_metrics(yt, out) if regression else champ.metrics(yt, out, n_classes)
                row.update({metric: m.get(metric), "fit_seconds": t1 - t0, "predict_seconds": t2 - t1,
                            "ms_per_row": _ms_per_row(t2 - t1, len(yt)), "error": None})
            except Exception as exc:  # a size that fails (e.g. out of memory) is a result too
                row.update({metric: None, "error": f"{type(exc).__name__}: {exc}"[:200]})
            secs = sp.elapsed
        v = row.get(metric)
        ui.done(f"{len(idx):,} rows · " + (f"{label} {fmt_value(v)}" if v is not None else ui.warn("failed"))
                + ui.dim(f" · {secs:.1f}s"))
        rows.append(row)
    if all(r.get("error") is None for r in rows):
        for r in rows:
            r.pop("error", None)
    curve = ContextCurve.build(rows, metric=metric, tolerance=tolerance, model=spec.display)
    if curve.recommended is not None:
        ui.say(f"  {ui.good('→')} {curve.summary()}")
    return curve


class _Live:
    """Turns the championship's progress callback into a spinner per model and a ✓ line when it finishes."""

    def __init__(self, ui: Console):
        self.ui = ui
        self.head = ""
        self.cm = None
        self.sp = None

    def progress(self, stage: str, fraction: float, message: str, eta: float | None = None) -> None:
        if stage == "split":
            self.ui.done(message.replace(" (all scored)", self.ui.dim(" (every row scored)")))
            return
        if stage != "evaluate":
            return
        head, _, tail = message.partition(" · ")
        if head != self.head:
            self._finish()
            self.head = head
            label = head.replace("Running inference with ", "")
            self.cm = self.ui.spinner(label)
            self.sp = self.cm.__enter__()
        if self.sp is not None:
            if "/" in tail and "holdout rows" in tail:
                done, total = (int(x.replace(",", "")) for x in tail.split(" holdout")[0].split(" / "))
                self.sp.update(suffix=self.ui.bar("", done / max(1, total), f"{done:,}/{total:,}"))
            elif tail:
                self.sp.update(suffix=self.ui.dim(f" · {tail}"))

    def _finish(self) -> None:
        if self.cm is not None:
            secs = self.sp.elapsed
            self.cm.__exit__(None, None, None)
            label = self.head.replace("Running inference with ", "")
            self.ui.done(f"{label} {self.ui.dim(f'{secs:.1f}s')}")
            self.cm = self.sp = None

    def close(self) -> None:
        self._finish()


def _models_line(specs: list[ModelSpec]) -> str:
    """The greeting: "ready" only when the weights are on disk too. Nothing downloads here."""
    if not specs:
        return "no models installed yet: pip install zerograd"
    from ..weights import missing_files

    names = ", ".join(s.display for s in specs)
    n = f"{len(specs)} model{'s' if len(specs) != 1 else ''}"
    if any(missing_files(s) for s in specs):
        return f"{n} installed: {names} · weights download on first use"
    return f"{n} ready: {names}"


def _prefetch(specs: list[ModelSpec], ui: Console, task: str = "classification") -> None:
    """Download missing weights up front, with a clear line, so they never fight with a spinner."""
    from ..weights import hub_file, missing_files

    for spec in specs:
        todo = missing_files(spec, task)
        if not todo:
            continue
        ui.say(f"{ui.accent('↓')} {spec.display}: fetching weights from huggingface.co/{todo[0][0]} "
               f"{ui.dim('(first time only) · ' + phrase('download'))}")
        for repo, filename in todo:
            try:
                hub_file(repo, filename, model=spec.name)
            except Exception as exc:
                ui.say(ui.warn(f"  could not download {filename}: {type(exc).__name__}"))
                break


def _describe_target(pre: Preprocessor) -> str:
    st = pre.target_stats
    if not st:
        return "regression"
    return f"regression · {pre.target} from {fmt_value(st['min'])} to {fmt_value(st['max'])}"


def _json_default(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"Not JSON serialisable: {type(v).__name__}")
