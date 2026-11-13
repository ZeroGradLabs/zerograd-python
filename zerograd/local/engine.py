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

from .._ui import Console, phrase
from .._version import __version__
from ..errors import ValidationError
from ..models import FittedModel, ModelSpec, get_model, resolve_models
from . import championship as champ
from .preprocess import Preprocessor

FORMAT = "zerograd.local/1"


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
        names = ", ".join(s.display for s in self.specs)
        self.ui.greet("local", f"{len(self.specs)} model{'s' if len(self.specs) != 1 else ''} ready: {names}" if self.specs
                      else 'no models installed yet: pip install "zerograd[local]"')
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
               metric: str | None = None) -> "LocalCapability":
        """Run the championship on labeled history and return the capability with its champion.

        `examples` is a DataFrame or a CSV/Parquet path. `metric` picks the champion: roc_auc (default), pr_auc,
        log_loss, accuracy, balanced_accuracy, f1, precision or recall (binary), f1_macro (multiclass).
        """
        from .._remote import to_frame

        if isinstance(examples, str) and (examples.startswith("gs://") or examples.startswith("upl_")):
            raise ValidationError("Bucket URIs and upload ids need the platform: use ZeroGrad(remote=True).")
        metric = champ.check_metric(metric)
        df = to_frame(examples)
        lim = self.limits
        pre, kept = Preprocessor.fit(df, target, drop=list(drop or []), max_columns=lim["max_columns"],
                                     max_classes=lim["max_classes"], min_rows=lim["min_rows"])
        if metric not in [m for m, (_, _, tasks) in champ.METRICS.items() if pre.task in tasks]:
            raise ValidationError(f"Metric '{metric}' does not apply to a {pre.task} task.", code="invalid_metric")
        ui = Console(self.progress)
        ui.done(f"{len(kept):,} labeled rows · {len(pre.columns)} columns used"
                + (ui.dim(f" · dropped {', '.join(d['name'] for d in pre.dropped)}") if pre.dropped else ""))
        _prefetch(self.specs, ui)
        ui.say(f"{ui.accent('◆')} Championship · {len(self.specs)} model{'s' if len(self.specs) != 1 else ''} · "
               f"{ui.dim(phrase('championship'))}")
        live = _Live(ui)
        t0 = time.monotonic()
        try:
            result = champ.run_championship(
                pre, kept, self.specs, max_context_rows=lim["max_context_rows"],
                holdout_fraction=lim["holdout_fraction"], chunk=lim["predict_chunk_rows"], progress=live.progress,
                max_holdout_rows=lim["max_holdout_rows"], metric=metric,
            )
        finally:
            live.close()
        cap = LocalCapability(name=name, pre=pre, result=result, metric=metric, seconds=time.monotonic() - t0,
                              noncommercial=[s.name for s in self.specs if not s.commercial],
                              chunk=lim["predict_chunk_rows"], progress=self.progress)
        ranked = sorted(result.candidates, key=champ.rank_key(metric), reverse=True)
        majority = next((b for b in result.baselines or [] if b.model == "baseline:majority"), None)
        ui.podium([(c.display, c.metrics.get(metric), c.error) for c in ranked], champ.METRICS[metric][0],
                  cap.attribution, baseline=majority.metrics.get(metric) if majority else None)
        ui.say(f"  {ui.dim(phrase('champion'))}")
        return cap


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
    def evaluation(self) -> pd.DataFrame:
        """The championship: every model scored on the same holdout."""
        df = pd.DataFrame(self.candidates)
        df["champion"] = df["model"] == self.champion
        metric_cols = [m for m, (_, _, tasks) in champ.METRICS.items() if self.pre.task in tasks and m in df]
        cols = [c for c in ("display", *metric_cols, "seconds", "champion", "error") if c in df]
        df = df[cols].rename(columns={"display": "model"}).set_index("model")
        if "error" in df and df["error"].isna().all():
            df = df.drop(columns="error")
        df.attrs["rows"] = self.rows
        df.attrs["metric"] = self.metric
        return df

    @property
    def baselines(self) -> pd.DataFrame:
        """What you get without a model on the same holdout: random scores and always the majority class."""
        df = pd.DataFrame(self.baseline_rows)
        keep = [c for c in ("display", "roc_auc", "accuracy", "log_loss") if c in df]
        return df[keep].rename(columns={"display": "baseline"}).set_index("baseline") if len(df) else df

    def update(self, *, threshold: float | None = None) -> "LocalCapability":
        """Binary tasks: score >= threshold predicts the positive class. None resets to 0.5."""
        if threshold is not None:
            if self.pre.task != "binary":
                raise ValidationError("A threshold only applies to binary capabilities.", code="invalid_threshold")
            if not 0 < float(threshold) < 1:
                raise ValidationError("threshold is a number between 0 and 1.", code="invalid_threshold")
        self.threshold = threshold
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, data: Any, *, threshold: float | None = None) -> pd.DataFrame | dict:
        """prediction, score (binary) and proba_<class>, aligned to the input. A dict (one row) returns a dict."""
        from .._remote import to_frame

        one = isinstance(data, dict)
        df = to_frame(data)
        X = self.pre.transform(df)
        model = self._model()
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
            _prefetch([spec], ui)
            with ui.spinner(f"Warming up {spec.display} with {len(self.context_y):,} rows of context · "
                            f"{ui.dim(phrase('fit'))}", done=f"{spec.display} ready"):
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


def _prefetch(specs: list[ModelSpec], ui: Console) -> None:
    """Download missing weights up front, with a clear line, so they never fight with a spinner."""
    for spec in specs:
        if spec.weights_fn is None:
            continue
        try:
            files = spec.weights_fn()
            from huggingface_hub import hf_hub_download, try_to_load_from_cache

            todo = [(r, f) for r, f in files if not isinstance(try_to_load_from_cache(r, f), str)]
        except Exception:
            continue  # the model reports its own error when it runs
        if not todo:
            continue
        ui.say(f"{ui.accent('↓')} {spec.display}: fetching weights from huggingface.co/{todo[0][0]} "
               f"{ui.dim('(first time only) · ' + phrase('download'))}")
        for repo, filename in todo:
            try:
                hf_hub_download(repo_id=repo, filename=filename)
            except Exception as exc:
                ui.say(ui.warn(f"  could not download {filename}: {type(exc).__name__}"))
                break


def _json_default(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(f"Not JSON serialisable: {type(v).__name__}")
