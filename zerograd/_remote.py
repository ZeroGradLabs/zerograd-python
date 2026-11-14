from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any, Iterable, Union

import httpx
import pandas as pd

from ._http import Transport
from ._progress import Bar
from .curve import ContextCurve
from .errors import RunFailedError, ValidationError, ZeroGradError
from .privacy import Vault, drop_sensitive

DEFAULT_BASE_URL = "https://api.zerograd.dev"
UPLOAD_THRESHOLD = 10 * 1024 * 1024  # above this, files go through an upload session
INLINE_ROWS = 5_000
CHUNK = 8 * 1024 * 1024  # a multiple of 256 KiB, as Cloud Storage requires

Table = Union[pd.DataFrame, str, os.PathLike, dict, list]


class RemoteZeroGrad:
    """Client for the ZeroGrad platform (remote=True). Built by ``ZeroGrad(api_key, remote=True)``.

    >>> client = ZeroGrad(remote=True)            # reads ZEROGRAD_API_KEY and ZEROGRAD_BASE_URL
    >>> cap = client.create(name="titanic-survival", examples=train_df, target="Survived")
    >>> cap.predict(new_df)
    """

    def __init__(self, api_key: str | None = None, *, base_url: str | None = None, timeout: float = 120.0,
                 progress: bool = True, max_retries: int = 4, http_client: httpx.Client | None = None):
        api_key = api_key or os.environ.get("ZEROGRAD_API_KEY")
        if not api_key:
            raise ZeroGradError("No API key. Pass api_key=... or set ZEROGRAD_API_KEY.", code="missing_api_key")
        self.base_url = (base_url or os.environ.get("ZEROGRAD_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.progress = progress
        self._t = Transport(self.base_url, api_key, timeout, http_client, max_retries)
        from urllib.parse import urlparse

        from ._ui import Console

        Console(progress).greet("remote", f"connected to {urlparse(self.base_url).netloc or self.base_url}")

    remote = True

    def __repr__(self) -> str:
        return f"<ZeroGrad remote {self.base_url}>"

    # ------------------------------------------------------------------ capabilities
    def _local_only(self, task: str):
        raise ValidationError(
            f"{task} runs locally in this version: use ZeroGrad(remote=False). On the platform it is part of ZeroGrad "
            "Labs; ask us for access.", code="local_only")

    def anomalies(self, *args, **kwargs):
        self._local_only("Anomaly detection")

    def impute(self, *args, **kwargs):
        self._local_only("Imputation")

    def synthesize(self, *args, **kwargs):
        self._local_only("Synthetic data")

    def cross_validate(self, *args, **kwargs):
        self._local_only("Cross-validation")

    def cluster(self, *args, **kwargs):
        self._local_only("Clustering")

    def create(self, name: str, examples: Table, target: str, *, drop: Iterable[str] | None = None,
               metric: str | None = None, wait: bool = True, sensitive: str = "drop", keep: Iterable[str] = (),
               vault: Vault | None = None, task: str | None = None) -> "Capability | Run":
        """Create a capability (or a new version of `name`) from labeled history and run the championship.

        `examples` is a DataFrame, a CSV/Parquet path, a gs:// URI or an upload id.
        `metric` picks the champion: roc_auc (default), pr_auc, log_loss, accuracy, balanced_accuracy,
        f1, precision or recall (binary), f1_macro (multiclass). `task="regression"` runs locally only, for now.
        Waits with a progress bar and returns the Capability; with wait=False returns the Run.

        Privacy, on your machine before upload (local tables only, not gs:// or upload ids):
        `sensitive="drop"` (default) drops columns that look like personal data with a SensitiveDataWarning;
        "warn" only warns, "keep" skips the check; `keep=[...]` exempts columns. `vault=Vault(key)` sends
        keyed tokens instead of your column names, categories, numbers and labels.
        """
        if task not in (None, "classification"):
            raise ValidationError(
                f"The platform runs classification in this version; {task} runs locally: ZeroGrad(remote=False)"
                f".create(..., task={task!r}).", code="unsupported_task")
        drop = list(drop or [])
        examples = _protect(examples, sensitive, keep=[target, *keep], vault=vault, target=target)
        if vault is not None:
            drop = [vault.column(c) for c in drop]
            target = vault.column(target)
        source, payload, n_rows = self._prepare(examples, "examples")
        if source:
            key = "upload_id" if source.startswith("upl_") else "source"
            body = _clean({"name": name, "target": target, key: source, "drop": drop, "metric": metric})
            started = self._t.request("POST", "/v1/capabilities", json=body)
        else:
            if self.progress:
                _say(f"Uploading {n_rows:,} rows · {_size(len(payload[1]))}")
            fields = _clean({"name": name, "target": target, "drop": ",".join(drop), "metric": metric})
            started = self._t.request("POST", "/v1/capabilities", data=fields, files={"examples": payload})
        run = Run(self, started["run_id"])
        if not wait:
            return run
        run.wait()
        cap = self.capability(started["capability_id"])
        if self.progress:
            best = next((c for c in cap.evaluation_rows if c["model"] == cap.champion), {})
            metric = f"ROC AUC {best['roc_auc']:.3f}" if best.get("roc_auc") is not None else f"accuracy {best.get('accuracy', 0):.3f}"
            _say(f"✓ Champion: {cap.champion_display} ({metric}) · capability {cap.id} v{cap.version}")
        return cap

    def capability(self, name_or_id: str) -> "Capability":
        return Capability(self, self._t.request("GET", f"/v1/capabilities/{name_or_id}"))

    def capabilities(self) -> list["Capability"]:
        return [Capability(self, c) for c in self._t.request("GET", "/v1/capabilities")["capabilities"]]

    def predict(self, capability: str, input: Table, *, version: int | None = None,
                threshold: float | None = None, sensitive: str = "drop", keep: Iterable[str] = (),
                vault: Vault | None = None) -> pd.DataFrame | dict:
        """One-off prediction. With a dict (one row) returns a dict; with a table returns a DataFrame."""
        cap = Capability(self, {"id": capability, "name": capability})
        if isinstance(input, dict):
            row = _protect(pd.DataFrame([input]), sensitive, keep=keep, vault=vault).iloc[0].to_dict()
            row = {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}  # NaN -> null
            r = self._t.request("POST", f"/v1/capabilities/{capability}/predict", json={"rows": [row]},
                                params=_clean({"version": version, "threshold": threshold}))
            out = {**r["predictions"][0], "version": r["version"]}
            return vault.decode(out) if vault is not None else out
        return cap.predict(input, version=version, threshold=threshold, sensitive=sensitive, keep=keep, vault=vault)

    def run(self, run_id: str) -> "Run":
        return Run(self, run_id)

    def models(self) -> pd.DataFrame:
        return pd.DataFrame(self._t.request("GET", "/v1/models")["models"])

    def usage(self) -> dict:
        return self._t.request("GET", "/v1/usage")

    def context_curve(self, examples: pd.DataFrame, target: str, sizes: Iterable[int], test: pd.DataFrame,
                      *, name: str = "context-curve", seed: int = 0) -> ContextCurve:
        """How quality grows with history: a championship per size, each scored on the same `test` rows.

        Creates a temporary capability per size and deletes it afterwards. Returns a ContextCurve: one row per
        size, indexed by rows, with the champion, ROC AUC and accuracy measured here on `test`, plus
        `recommended` and `plot()`.
        """
        y_true = test[target]
        features = test.drop(columns=[target])
        rows = []
        quiet = RemoteZeroGrad.__new__(RemoteZeroGrad)
        quiet.__dict__.update(self.__dict__, progress=False)
        for size in sizes:
            sample = _stratified(examples, target, size, seed)
            cap_name = f"{name}-{size}"
            if self.progress:
                _say(f"… {size} rows of history")
            cap = quiet.create(cap_name, sample, target)
            try:
                pred = cap.predict(features)
                rows.append({"rows": size, "champion": cap.champion_display,
                             "roc_auc": _roc_auc(y_true, pred, cap.classes), "accuracy": _accuracy(y_true, pred)})
            finally:
                cap.delete()
        if self.progress:
            _say("✓ " + " · ".join(f"{r['rows']} rows: AUC {r['roc_auc']:.3f}" for r in rows if r["roc_auc"] is not None))
        return ContextCurve.build(rows, metric="roc_auc")

    # ------------------------------------------------------------------ uploads
    def upload(self, data: bytes, filename: str) -> str:
        """Send a file of any size through a resumable upload session. Returns the upload id."""
        session = self._t.request("POST", "/v1/uploads", json={"filename": filename, "size": len(data)})
        url, total = session["url"], len(data)
        bar = Bar(self.progress)
        sent = 0
        while sent < total:
            chunk = data[sent : sent + CHUNK]
            end = sent + len(chunk) - 1
            resp = self._t.put_chunk(url, chunk, {"Content-Range": f"bytes {sent}-{end}/{total}"})
            if resp.status_code not in (200, 201, 308):
                raise ZeroGradError(f"Upload failed with HTTP {resp.status_code}: {resp.text[:200]}")
            sent = end + 1
            bar.update({"status": "running", "progress": sent / total, "message": f"Uploading {filename} · {_size(total)}"})
        bar.done(f"✓ Uploaded {filename} · {_size(total)}" if self.progress else None)
        return session["upload_id"]

    def _prepare(self, data: Table, field: str) -> tuple[str | None, tuple[str, bytes] | None, int]:
        """-> (source, None, 0) for gs:// and upload ids; otherwise (upload_id or None, (filename, bytes), rows)."""
        if isinstance(data, str) and (data.startswith("gs://") or data.startswith("upl_")):
            return data, None, 0
        df = to_frame(data)
        payload = _encode(df)
        if len(payload[1]) > UPLOAD_THRESHOLD:
            return self.upload(payload[1], payload[0]), None, len(df)
        return None, payload, len(df)


class Run:
    """An async job: create, predict (batch), tournament or batch pipeline."""

    def __init__(self, client: RemoteZeroGrad, run_id: str, data: dict | None = None):
        self._c = client
        self.id = run_id
        self.data = data or {}

    def refresh(self) -> "Run":
        self.data = self._c._t.request("GET", f"/v1/runs/{self.id}")
        return self

    @property
    def status(self) -> str:
        return self.data.get("status") or self.refresh().data["status"]

    @property
    def progress(self) -> float:
        return float(self.refresh().data.get("progress") or 0)

    @property
    def result(self) -> Any:
        return self.data.get("result")

    def wait(self, timeout: float | None = None, poll: float = 0.5) -> "Run":
        """Poll until the run ends. Raises RunFailedError if it fails or is canceled."""
        bar = Bar(self._c.progress)
        start = time.monotonic()
        while True:
            self.refresh()
            bar.update(self.data)
            if self.data["status"] in ("succeeded", "failed", "canceled"):
                break
            if timeout is not None and time.monotonic() - start > timeout:
                bar.done()
                raise TimeoutError(f"Run {self.id} still {self.data['status']} after {timeout}s")
            time.sleep(poll)
            poll = min(poll * 1.25, 3.0)
        bar.done()
        if self.data["status"] != "succeeded":
            err = self.data.get("error") or {}
            raise RunFailedError(err.get("message") or f"Run {self.id} {self.data['status']}", run=self, code=err.get("error"))
        return self

    def cancel(self) -> "Run":
        self.data = self._c._t.request("POST", f"/v1/runs/{self.id}/cancel")
        return self

    def output(self) -> pd.DataFrame:
        resp = self._c._t.request("GET", f"/v1/runs/{self.id}/output", raw=True)
        return pd.read_parquet(io.BytesIO(resp.content))

    def __repr__(self) -> str:
        d = self.data
        if not d:
            return f"<Run {self.id}>"
        return f"<Run {self.id} {d.get('kind')} {d.get('status')} {float(d.get('progress') or 0):.0%} · {d.get('message', '')}>"


class Capability:
    """One prediction you need, with its versions, evaluation and pipelines."""

    def __init__(self, client: RemoteZeroGrad, data: dict):
        self._c = client
        self.data = data
        self.id = data["id"]
        self.name = data.get("name", self.id)
        self.pipelines = Pipelines(client, self)

    def __repr__(self) -> str:
        return f"<Capability {self.name} ({self.id}) v{self.version} · {self.data.get('status')} · champion {self.data.get('champion')}>"

    def refresh(self) -> "Capability":
        self.data = self._c._t.request("GET", f"/v1/capabilities/{self.id}")
        return self

    @property
    def version(self) -> int | None:
        return self.data.get("current_version")

    @property
    def champion(self) -> str | None:
        return self.data.get("champion")

    @property
    def champion_display(self) -> str | None:
        for v in self.data.get("versions", []):
            if v["version"] == self.version:
                return v["champion_display"]
        return self.champion

    @property
    def classes(self) -> list:
        return self.data.get("classes") or []

    @property
    def expected_columns(self) -> list[str]:
        return self.data.get("expected_columns") or []

    @property
    def evaluation_rows(self) -> list[dict]:
        return self._c._t.request("GET", f"/v1/capabilities/{self.id}/evaluation")["candidates"]

    @property
    def evaluation(self) -> pd.DataFrame:
        """The championship: every model scored on the same holdout."""
        ev = self._c._t.request("GET", f"/v1/capabilities/{self.id}/evaluation")
        df = pd.DataFrame(ev["candidates"])
        df["champion"] = df["model"] == ev["champion"]
        metric_cols = [m["name"] for m in ev.get("metrics", [])] or ["roc_auc", "accuracy", "log_loss"]
        cols = [c for c in ("display", *metric_cols, "seconds", "champion", "error") if c in df]
        df = df[cols].rename(columns={"display": "model"}).set_index("model")
        if "error" in df and df["error"].isna().all():
            df = df.drop(columns="error")
        df.attrs["rows"] = ev["rows"]
        df.attrs["metric"] = ev.get("metric", "roc_auc")
        return df

    @property
    def threshold(self) -> float | None:
        """Binary capabilities: score >= threshold predicts the positive class. None means 0.5."""
        return self.data.get("threshold")

    @property
    def metric(self) -> str:
        return self.data.get("metric") or "roc_auc"

    def update(self, **settings: Any) -> "Capability":
        """Change `threshold` (binary; None resets to 0.5), `metric` (used by the next tournament) or `maturity_days`
        (how long outcomes take to be known, for `performance`). No retraining."""
        self.data = self._c._t.request("PATCH", f"/v1/capabilities/{self.id}", json=settings)
        return self

    def predict(self, data: Table, *, version: int | None = None, threshold: float | None = None,
                id_column: str | None = None, wait: bool = True, sensitive: str = "drop", keep: Iterable[str] = (),
                vault: Vault | None = None) -> "pd.DataFrame | Run":
        """Predict rows. Returns a DataFrame (prediction_id, prediction, score for binary, proba_<class>) aligned to the input.

        For binary tasks `prediction` is score >= threshold (this call's, else the capability's, else 0.5);
        the probabilities are always there if you'd rather pick the cut-off yourself.

        `prediction_id` is what you send back with `send_outcomes`. With `id_column` (one of your columns, like
        "loan_id") its values are kept as each prediction's reference, so you can send outcomes by your own id.
        That column is sent as is: it skips the sensitive-data check and the vault.

        Up to 5,000 rows answer in the same request; bigger inputs become a batch run the SDK waits for.
        `data` can also be a gs:// URI or an upload id, which always runs as a batch.
        `sensitive`, `keep` and `vault` work as in `ZeroGrad.create`; with a vault the result comes back decoded.
        """
        params = _clean({"version": version, "threshold": threshold, "id_column": id_column})
        if isinstance(data, str) and (data.startswith("gs://") or data.startswith("upl_")):
            key = "upload_id" if data.startswith("upl_") else "source"
            started = self._c._t.request("POST", f"/v1/capabilities/{self.id}/predict", json={key: data}, params=params)
            run = Run(self._c, started["run_id"])
            return run.wait().output() if wait else run
        frame = to_frame(data)
        if id_column is not None and id_column not in frame.columns:
            raise ValidationError(f"id_column '{id_column}' is not one of the columns.")
        df = _protect(frame.drop(columns=[id_column]) if id_column else frame, sensitive, keep=keep, vault=vault)
        if id_column:
            df = df.assign(**{id_column: frame[id_column].to_numpy()})
        filename, payload = _encode(df)
        if len(df) <= INLINE_ROWS and len(payload) <= UPLOAD_THRESHOLD:
            r = self._c._t.request("POST", f"/v1/capabilities/{self.id}/predict", files={"rows": (filename, payload)},
                                   params={**params, "mode": "inline"})
            out = _records_to_frame(r["predictions"], r["classes"], df.index)
            return vault.decode(out) if vault is not None else out
        upl = self._c.upload(payload, filename)
        started = self._c._t.request("POST", f"/v1/capabilities/{self.id}/predict", json={"upload_id": upl}, params=params)
        run = Run(self._c, started["run_id"])
        if not wait:
            return run
        out = run.wait().output()
        out.index = df.index
        return vault.decode(out) if vault is not None else out

    def send_outcomes(self, outcomes: Table, *, vault: Vault | None = None) -> dict:
        """Tell the platform what actually happened to past predictions.

        Each row has `prediction_id` or `reference` (your `id_column` value), plus `outcome` (one of the classes),
        `decision` (what you did with the prediction, free text) or both, and optionally `observed_at`.
        Sending the same id again updates it. Returns how many rows matched a prediction.
        With the vault the capability was created with, outcomes go out as the same label tokens.
        """
        if isinstance(outcomes, dict):
            outcomes = [outcomes]
        if isinstance(outcomes, list):
            rows = [{k: _jsonable(v) for k, v in r.items()} for r in outcomes]
            if vault is not None:
                rows = [{**r, "outcome": _vault_label(vault, r["outcome"])} if r.get("outcome") is not None else r for r in rows]
            return self._c._t.request("POST", f"/v1/capabilities/{self.id}/outcomes", json={"outcomes": rows})
        df = to_frame(outcomes).reset_index(drop=True)
        if vault is not None and "outcome" in df:
            df["outcome"] = df["outcome"].map(lambda v: None if pd.isna(v) else _vault_label(vault, v))
        filename, payload = _encode(df)
        return self._c._t.request("POST", f"/v1/capabilities/{self.id}/outcomes", files={"outcomes": (filename, payload)})

    def performance(self, cohort: str = "week", *, version: int | None = None, include_test: bool = False) -> pd.DataFrame:
        """Production metrics from the outcomes you sent, one row per cohort of prediction date.

        Only mature cohorts (older than the capability's `maturity_days`, set with `update(maturity_days=...)`)
        get metrics. `attrs` holds `overall`, `by_version` (each version against its holdout score) and `totals`.
        """
        r = self._c._t.request("GET", f"/v1/capabilities/{self.id}/performance",
                               params=_clean({"cohort": cohort, "version": version, "include_test": include_test or None}))
        rows = [{**{k: v for k, v in c.items() if k != "metrics"}, **(c.get("metrics") or {})} for c in r["cohorts"]]
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.set_index("start")
        for key in ("overall", "by_version", "totals", "maturity_days", "metric", "holdout"):
            df.attrs[key] = r.get(key)
        return df

    def delete(self) -> dict:
        """Delete the capability and the history stored with it."""
        out = self._c._t.request("DELETE", f"/v1/capabilities/{self.id}")
        if self._c.progress:
            _say(f"✓ Deleted {self.name}")
        return out


class Pipelines:
    def __init__(self, client: RemoteZeroGrad, cap: Capability):
        self._c = client
        self._cap = cap

    def create(self, kind: str, *, source: str | None = None, output: str | None = None, schedule: str | None = None,
               promote_if: str | None = None, name: str | None = None) -> "Pipeline":
        """kind="tournament" re-elects the champion on fresh history; kind="batch" scores files from a bucket."""
        body = _clean({"kind": kind, "source": source, "output": output, "schedule": schedule,
                       "promote_if": promote_if, "name": name})
        return Pipeline(self._c, self._c._t.request("POST", f"/v1/capabilities/{self._cap.id}/pipelines", json=body))

    def list(self) -> list["Pipeline"]:
        data = self._c._t.request("GET", f"/v1/capabilities/{self._cap.id}/pipelines")["pipelines"]
        return [Pipeline(self._c, p) for p in data]


class Pipeline:
    def __init__(self, client: RemoteZeroGrad, data: dict):
        self._c = client
        self.data = data
        self.id = data["id"]

    def __repr__(self) -> str:
        sched = f" · {self.data['schedule']}" if self.data.get("schedule") else ""
        return f"<Pipeline {self.data.get('name')} ({self.id}) {self.data.get('kind')}{sched}>"

    def run(self, wait: bool = True) -> Run:
        started = self._c._t.request("POST", f"/v1/pipelines/{self.id}/runs")
        run = Run(self._c, started["run_id"])
        return run.wait() if wait else run

    def delete(self) -> dict:
        return self._c._t.request("DELETE", f"/v1/pipelines/{self.id}")


# ---------------------------------------------------------------------- helpers
def to_frame(data: Table) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        return pd.DataFrame([data])
    if isinstance(data, list):
        return pd.DataFrame.from_records(data)
    path = Path(data)
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _encode(df: pd.DataFrame) -> tuple[str, bytes]:
    buf = io.BytesIO()
    try:
        df.reset_index(drop=True).to_parquet(buf, index=False)
        return "data.parquet", buf.getvalue()
    except Exception:  # mixed-type object columns: CSV always works
        return "data.csv", df.to_csv(index=False).encode()


def _vault_label(vault: Vault, value: Any) -> str:
    from .privacy import _canon

    tokens = {_canon(v): tok for tok, v in vault._labels.items()}
    if _canon(value) not in tokens:
        raise ValidationError(f"The vault doesn't know the label {value!r}. Use the vault (or vault.load) the capability was created with.")
    return tokens[_canon(value)]


def _jsonable(v: Any) -> Any:
    if v is None or (isinstance(v, float) and v != v):
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "item"):  # numpy scalars
        return v.item()
    return v


def _records_to_frame(records: list[dict], classes: list, index: pd.Index) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    if records and "prediction_id" in records[0]:
        out["prediction_id"] = [r["prediction_id"] for r in records]
    if records and "reference" in records[0]:
        out["reference"] = [r["reference"] for r in records]
    out["prediction"] = [r["prediction"] for r in records]
    if records and "score" in records[0]:
        out["score"] = [r["score"] for r in records]
    for cls in classes:
        out[f"proba_{cls}"] = [r["proba"][str(cls)] for r in records]
    return out


def _stratified(df: pd.DataFrame, target: str, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    frac = n / len(df)
    parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby(target, observed=True)]
    return pd.concat(parts).sample(frac=1, random_state=seed)


def _norm(v: Any) -> str:
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return str(v)


def _accuracy(y_true: pd.Series, pred: pd.DataFrame) -> float:
    return float((y_true.map(_norm).to_numpy() == pred["prediction"].map(_norm).to_numpy()).mean())


def _roc_auc(y_true: pd.Series, pred: pd.DataFrame, classes: list) -> float | None:
    """Binary ROC AUC via the rank formula (no scikit-learn needed)."""
    if "score" not in pred or len(classes) != 2:
        return None
    pos = _norm(classes[1])
    y = (y_true.map(_norm) == pos).to_numpy()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = pd.Series(pred["score"].to_numpy()).rank().to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}" if n < 10 else f"{n:.0f} {unit}"
        n /= 1024
    return f"{n} B"


def _say(text: str) -> None:
    import sys

    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def _protect(data, sensitive: str, *, keep: Iterable[str] = (), vault: Vault | None = None, target: str | None = None):
    """Drop personal data and pseudonymise, on this machine, before anything is sent. Remote sources pass through."""
    if sensitive not in ("drop", "warn", "keep"):
        raise ValueError("sensitive must be 'drop', 'warn' or 'keep'.")
    if isinstance(data, str) and (data.startswith("gs://") or data.startswith("upl_")):
        if vault is not None:
            raise ValueError("A vault needs the rows here to encode them; pass a DataFrame or a local file, not gs:// or an upload id.")
        return data
    if sensitive == "keep" and vault is None:
        return data
    df = to_frame(data)
    if sensitive != "keep":
        if sensitive == "drop":
            df = drop_sensitive(df, keep=keep, _stacklevel=5)
        else:
            import warnings

            from .privacy import SensitiveDataWarning, scan
            found = scan(df, ignore=keep)
            if found:
                warnings.warn("Looks like personal data: " + "; ".join(map(str, found)), SensitiveDataWarning, stacklevel=4)
    return vault.encode(df, target=target) if vault is not None else df
