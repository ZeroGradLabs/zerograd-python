"""An MCP server, so an agent (Claude Code, Cursor, Codex...) can use ZeroGrad on your machine.

    pip install "zerograd[mcp]"
    claude mcp add zerograd -- zerograd-mcp          # or: uvx --from "zerograd[mcp]" zerograd-mcp

Runs locally over stdio: your tables never leave the machine. The agent passes file paths (CSV or Parquet), not
rows, so a big table never goes through its context; predictions are written to a file next to the input and only
a summary and a preview come back. Capabilities are saved to ``~/.zerograd/capabilities`` (``ZEROGRAD_HOME`` to
move it) and stay there between sessions.

The tools are plain methods of :class:`Workbench`, so they can be tested and reused without MCP.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import ValidationError

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
EXT = ".zgcap"
PREVIEW_ROWS = 20
INLINE_ROWS = 100  # rows an agent may pass inline to predict; anything bigger goes in a file
TARGET_WORDS = re.compile(r"(^|_)(target|label|class|y|outcome|churn(ed)?|default(ed)?|fraud|survived|converted|"
                          r"approved|response|is_[a-z]+|has_[a-z]+)($|_)", re.I)

INSTRUCTIONS = """\
ZeroGrad makes predictions over tables (CSV or Parquet) with tabular foundation models: no training and no
hyperparameters. Use it when there is labeled history (rows where a column you want to predict is known) and new
rows to predict that column for: churn, fraud, default, conversion, a price, a demand...

Workflow: inspect_data (columns, a likely target, personal data to drop) -> create_capability (scores every model
on a holdout and keeps the best one, the champion) -> evaluate_capability -> predict. explain shows which columns
matter; optimize_threshold picks the score cut of a yes/no capability from the cost of each mistake.

When the tools do not cover something (anomalies, imputation, cross-validation by time, scikit-learn,
privacy...), read_docs has the SDK docs and API reference: write a short Python script with `import zerograd`.
Pass file paths, not rows. Everything runs on this machine: nothing is uploaded.
If create_capability returns warnings (a column that alone predicts the target, duplicated rows), tell the user before
quoting any score. Quote the champion's metric next to the baseline when you report a result, and show the
attribution line when a tool returns one (some model licenses ask for it)."""


def _home() -> Path:
    return Path(os.environ.get("ZEROGRAD_HOME") or Path.home() / ".zerograd").expanduser()


def _records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows: NaN becomes null and numpy types become Python ones."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _num(x: Any) -> Any:
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _path(p: str) -> Path:
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise ValidationError(f"File not found: {path}", code="file_not_found")
    return path


def _read(p: str) -> pd.DataFrame:
    path = _path(p)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


class Workbench:
    """What the MCP tools do, over the local engine and the capabilities saved in ``home``."""

    def __init__(self, home: str | Path | None = None, *, models: list[str] | None = None):
        self.home = Path(home).expanduser() if home else _home()
        self.models = models
        self._open: dict[str, Any] = {}

    # ------------------------------------------------------------------ capabilities on disk
    @property
    def folder(self) -> Path:
        d = self.home / "capabilities"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _file(self, name: str) -> Path:
        if not NAME.match(name or ""):
            raise ValidationError("A capability name is 1 to 64 letters, digits, '-', '_' or '.', like "
                                  "'churn' or 'loan-default'.", code="invalid_name")
        return self.folder / f"{name}{EXT}"

    def _cap(self, name: str):
        from .local import load

        if name not in self._open:
            f = self._file(name)
            if not f.exists():
                known = ", ".join(sorted(p.stem for p in self.folder.glob(f"*{EXT}"))) or "none yet"
                raise ValidationError(f"No capability named '{name}'. Capabilities: {known}. "
                                      "Create one with create_capability.", code="not_found")
            self._open[name] = load(f, progress=False)
        return self._open[name]

    def _engine(self, models: list[str] | None = None):
        from .local import LocalZeroGrad

        return LocalZeroGrad(models=models or self.models, progress=False)

    # ------------------------------------------------------------------ tools
    def inspect_data(self, path: str, target: str | None = None) -> dict:
        from .privacy import scan

        df = _read(path)
        cols = []
        for c in df.columns:
            s = df[c]
            example = s.dropna().iloc[0] if s.notna().any() else None
            cols.append({"name": str(c), "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                         "distinct": int(s.nunique(dropna=True)), "example": _num(example) if example is not None
                         else None})
        n = len(df)

        def looks_like_target(c: dict) -> bool:
            return 2 <= c["distinct"] <= 10 and c["missing"] < n

        suggested = [c["name"] for c in cols if TARGET_WORDS.search(c["name"]) and looks_like_target(c)]
        if not suggested and cols and looks_like_target(cols[-1]):
            suggested = [cols[-1]["name"]]
        out: dict[str, Any] = {
            "path": str(_path(path)), "rows": n, "columns": cols, "suggested_targets": suggested,
            "personal_data": [{"column": f.column, "kind": f.kind, "found_by": f.why} for f in scan(df)],
            "preview": _records(df.head(5)),
        }
        if target is not None:
            if target not in df.columns:
                raise ValidationError(f"No column '{target}' in {path}.", code="missing_target")
            y = df[target]
            out["target"] = {"name": target, "labeled_rows": int(y.notna().sum()),
                             "unlabeled_rows": int(y.isna().sum()),
                             "values": {str(k): int(v) for k, v in y.value_counts().head(10).items()}}
        out["hint"] = ("Drop the personal_data columns (drop=[...]) unless they are needed. Rows with an empty "
                       "target are not used to learn; predict them after create_capability.")
        return out

    def create_capability(self, name: str, path: str, target: str, drop: list[str] | None = None,
                          metric: str | None = None, task: str | None = None,
                          models: list[str] | None = None) -> dict:
        file = self._file(name)
        replaced = file.exists()
        df = _read(path)
        if target not in df.columns:
            raise ValidationError(f"No column '{target}' in {path}. Columns: {', '.join(map(str, df.columns))}.",
                                  code="missing_target")
        cap = self._engine(models).create(name, df, target, drop=drop, metric=metric, task=task)
        cap.save(file)
        self._open[name] = cap
        return {**self._summary(cap), "saved_to": str(file), "replaced": replaced}

    def list_capabilities(self) -> dict:
        caps = []
        for f in sorted(self.folder.glob(f"*{EXT}")):
            try:
                cap = self._cap(f.stem)
            except Exception as exc:  # a file from another version, or damaged: list it, do not fail
                caps.append({"name": f.stem, "error": str(exc)})
                continue
            caps.append({"name": cap.name, "task": cap.task, "target": cap.pre.target,
                         "champion": cap.champion_display, "metric": cap.metric, "score": self._score(cap),
                         "created_at": cap.created_at})
        return {"folder": str(self.folder), "capabilities": caps}

    def evaluate_capability(self, name: str, path: str | None = None) -> dict:
        cap = self._cap(name)
        out = self._summary(cap)
        if cap.task != "regression":
            data = _read(path) if path else None
            report = cap.classification_report(data)
            out["classification_report"] = [{"label": str(i), **{k: _num(v) for k, v in r.items()}}
                                            for i, r in report.iterrows()]
            out["report_source"] = report.attrs.get("source")
        return out

    def predict(self, name: str, path: str | None = None, rows: list[dict] | None = None,
                output_path: str | None = None, threshold: float | None = None) -> dict:
        cap = self._cap(name)
        if (path is None) == (rows is None):
            raise ValidationError("Pass either path (a CSV or Parquet file) or rows (a few records), not both.",
                                  code="missing_rows")
        if rows is not None:
            if len(rows) > INLINE_ROWS:
                raise ValidationError(f"Up to {INLINE_ROWS} rows inline; write bigger tables to a file and pass "
                                      "its path.", code="too_many_rows")
            df = pd.DataFrame.from_records(rows)
        else:
            df = _read(path)
        missing = [c for c in cap.expected_columns if c not in df.columns]
        if len(missing) == len(cap.expected_columns):
            raise ValidationError(f"None of the columns this capability uses are in the data. It expects: "
                                  f"{', '.join(cap.expected_columns)}.", code="missing_columns")
        # A missing column is predicted as a missing value, so "what if" rows with partial facts still work.
        pred = cap.predict(df.assign(**{c: np.nan for c in missing}) if missing else df, threshold=threshold)
        out: dict[str, Any] = {"capability": name, "rows": len(pred), "champion": cap.champion_display}
        if missing:
            out["missing_columns"] = missing  # predicted as missing values; quality may drop
        if cap.task == "regression":
            p = pred["prediction"]
            out["summary"] = {"mean": _num(p.mean()), "min": _num(p.min()), "max": _num(p.max())}
        else:
            out["summary"] = {"predicted": {str(k): int(v) for k, v in pred["prediction"].value_counts().items()}}
            if "score" in pred:
                out["summary"]["mean_score"] = _num(pred["score"].mean())
                out["threshold"] = cap.threshold if threshold is None else threshold
        if rows is not None and output_path is None:
            out["predictions"] = _records(pred)
        else:
            src = _path(path) if path else None
            dest = Path(output_path).expanduser() if output_path else src.with_name(f"{src.stem}.predictions.csv")
            full = pd.concat([df.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
            if dest.suffix.lower() in (".parquet", ".pq"):
                full.to_parquet(dest, index=False)
            else:
                full.to_csv(dest, index=False)
            out["output_path"] = str(dest)
            out["preview"] = _records(full.head(PREVIEW_ROWS))
        if cap.attribution:
            out["attribution"] = cap.attribution
        return out

    def explain(self, name: str, path: str, repeats: int = 3) -> dict:
        cap = self._cap(name)
        imp = cap.importance(_read(path), repeats=repeats)
        rows = [{"column": str(i), **{k: _num(v) for k, v in r.items()}} for i, r in imp.iterrows()]
        return {"capability": name, "metric": imp.attrs.get("metric", cap.metric), "importance": rows,
                "how": "Each column is shuffled and the drop in the metric is measured: bigger matters more, "
                       "near zero could go. Use rows the capability did not learn from, or it reads optimistic."}

    def optimize_threshold(self, name: str, objective: str = "cost", cost_fp: float = 1.0, cost_fn: float = 1.0,
                           min_precision: float | None = None, min_recall: float | None = None,
                           apply: bool = False) -> dict:
        cap = self._cap(name)
        res = cap.optimize_threshold(objective, cost_fp=cost_fp, cost_fn=cost_fn, min_precision=min_precision,
                                     min_recall=min_recall, apply=apply)
        if apply:
            cap.save(self._file(name))
        return {"capability": name, "threshold": res.threshold, "objective": res.objective,
                "metrics": {k: _num(v) for k, v in res.metrics.items()}, "source": res.source, "applied": apply,
                "meaning": f"A row is predicted positive ({cap.classes[1]!s}) when its score is at least "
                           f"{res.threshold:.2f}."}

    def list_models(self) -> dict:
        from .models import list_models

        keep = ("name", "display", "license", "commercial", "installed", "tasks", "install")
        return {"models": [{k: _num(r.get(k)) if not isinstance(r.get(k), (list, tuple)) else list(r[k])
                            for k in keep if k in r} for r in list_models()],
                "note": "create_capability runs the installed commercial models. Non-commercial models are off here."}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _score(cap) -> Any:
        champ = next((c for c in cap.candidates if c["model"] == cap.champion), {})
        return _num(champ.get(cap.metric))

    def _summary(self, cap) -> dict:
        ev = cap.evaluation.reset_index()
        out: dict[str, Any] = {
            "capability": cap.name, "task": cap.task, "target": cap.pre.target, "champion": cap.champion_display,
            "metric": cap.metric, "score": self._score(cap), "rows": {k: _num(v) for k, v in (cap.rows or {}).items()},
            "leaderboard": _records(ev), "baselines": _records(cap.baselines.reset_index()) if len(cap.baselines)
            else [], "columns": cap.columns,
        }
        if cap.task != "regression":
            out["classes"] = [str(c) for c in cap.classes]
        if cap.task == "binary":
            out["threshold"] = cap.threshold
        if cap.errors:
            out["errors"] = cap.errors
        if getattr(cap, "warnings", None):
            out["warnings"] = [w["message"] for w in cap.warnings]
        if cap.attribution:
            out["attribution"] = cap.attribution
        return out


# ---------------------------------------------------------------------- docs, offline
DOC_FILES = ("README.md", "MODELS.md")
API = ("zerograd:ZeroGrad", "zerograd.local:LocalZeroGrad", "zerograd.local:LocalCapability",
       "zerograd._remote:RemoteZeroGrad", "zerograd._remote:Capability", "zerograd:load", "zerograd:list_models",
       "zerograd:install_model", "zerograd:download", "zerograd:scan", "zerograd:drop_sensitive", "zerograd:Vault",
       "zerograd:ZeroGradClassifier", "zerograd:ZeroGradRegressor")
DOC_CHARS = 6000  # per section returned, so one answer stays small


def _doc_file(name: str) -> str | None:
    here = Path(__file__).resolve().parent
    for p in (here / "_docs" / name, here.parent / name):  # installed wheel, then a source checkout
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


def _markdown_sections(source: str, text: str) -> list[tuple[str, str]]:
    out, title, buf = [], source, []
    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            if "".join(buf).strip():
                out.append((title, "\n".join(buf).strip()))
            title, buf = line.lstrip("#").strip(), [line]
        else:
            buf.append(line)
    if "".join(buf).strip():
        out.append((title, "\n".join(buf).strip()))
    return out


def _api_sections() -> list[tuple[str, str]]:
    import importlib
    import inspect

    out = []
    for ref in API:
        mod, attr = ref.split(":")
        try:
            obj = getattr(importlib.import_module(mod), attr)
        except Exception:
            continue
        name = f"{mod.split('.')[0]}.{attr}" if mod == "zerograd" else attr

        def sig(f) -> str:
            try:
                return str(inspect.signature(f))
            except (TypeError, ValueError):
                return "(...)"

        if inspect.isclass(obj):
            out.append((f"API: {name}", f"class {name}{sig(obj)}\n\n{inspect.getdoc(obj) or ''}"))
            for m, f in inspect.getmembers(obj):
                if m.startswith("_") or not (inspect.isfunction(f) or isinstance(f, property)):
                    continue
                fn = f.fget if isinstance(f, property) else f
                head = f"{name}.{m}" + ("" if isinstance(f, property) else sig(fn))
                out.append((f"API: {name}.{m}", f"{head}\n\n{inspect.getdoc(fn) or ''}"))
        else:
            out.append((f"API: {name}", f"{name}{sig(obj)}\n\n{inspect.getdoc(obj) or ''}"))
    return out


def doc_sections() -> list[tuple[str, str]]:
    """(title, text) for each section of the README and MODELS.md, and for each public function and method."""
    secs: list[tuple[str, str]] = []
    for f in DOC_FILES:
        text = _doc_file(f)
        if text:
            secs += _markdown_sections(f, text)
    return secs + _api_sections()


def read_docs(query: str | None = None, sections: list[tuple[str, str]] | None = None) -> dict:
    """No query: the list of topics. A topic title: that section. Anything else: the best matching sections."""
    secs = sections if sections is not None else doc_sections()
    online = "https://zerograd.dev/docs"
    if not query or not query.strip():
        return {"topics": [t for t, _ in secs], "online": online,
                "hint": "Call read_docs with a topic, or with words like 'threshold cost' or 'regression metric'."}
    q = query.strip().lower()
    exact = [(t, x) for t, x in secs if t.lower() == q or t.lower() == f"api: {q}"]
    if exact:
        hits = exact
    else:
        words = [w for w in re.findall(r"[a-z0-9_]+", q) if len(w) > 1]
        scored = []
        for t, x in secs:
            tl, xl = t.lower(), x.lower()
            score = sum(3 * tl.count(w) + xl.count(w) for w in words)
            if all(w in tl or w in xl for w in words):
                score *= 2
            if score:
                scored.append((score, t, x))
        hits = [(t, x) for _, t, x in sorted(scored, key=lambda s: -s[0])[:3]]
    return {"query": query, "online": online,
            "results": [{"topic": t, "text": x if len(x) <= DOC_CHARS else x[:DOC_CHARS] + "\n[...]"}
                        for t, x in hits],
            **({} if hits else {"hint": "Nothing matched. Call read_docs with no query for the list of topics."})}


# ---------------------------------------------------------------------- MCP
def build_server(bench: Workbench | None = None):
    """The MCP server with every Workbench tool. Needs the `mcp` package: pip install "zerograd[mcp]"."""
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError:  # mcp 1.x
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer
        except ImportError as exc:
            raise SystemExit('The ZeroGrad MCP server needs the mcp package: pip install "zerograd[mcp]"') from exc
    import anyio

    from ._version import __version__

    bench = bench or Workbench()
    kw: dict[str, Any] = {"instructions": INSTRUCTIONS}
    try:
        server = MCPServer(name="zerograd", version=__version__, website_url="https://zerograd.dev", **kw)
    except TypeError:
        server = MCPServer("zerograd", **kw)

    async def run(fn, *args):
        # The engine is blocking (a championship can take minutes): keep the event loop free.
        return await anyio.to_thread.run_sync(lambda: fn(*args))

    @server.tool(title="Inspect a table")
    async def inspect_data(path: str, target: str | None = None) -> dict:
        """Look at a CSV or Parquet file before modeling: rows, each column (type, missing, distinct values,
        an example), columns that look like a target, columns that look like personal data, and the first rows.
        Pass `target` to also see how many rows are labeled and the class balance."""
        return await run(bench.inspect_data, path, target)

    @server.tool(title="Create a capability")
    async def create_capability(name: str, path: str, target: str, drop: list[str] | None = None,
                                metric: str | None = None, task: str | None = None,
                                models: list[str] | None = None) -> dict:
        """Learn to predict column `target` from the labeled rows of a CSV or Parquet file. Every installed model
        is scored on the same holdout and the best one (the champion) is kept, with a leaderboard and baselines.
        The capability is saved under `name` (replacing one with that name) and used by predict.

        drop: columns to leave out (ids, personal data, anything known only after the fact).
        metric: what picks the champion. Classification: roc_auc (default), pr_auc, log_loss, accuracy,
        balanced_accuracy, f1, precision, recall, f1_macro. Regression: rmse (default), mae, r2.
        task: "classification" or "regression"; left out, it is picked from the target.
        Can take a few minutes on a CPU with big tables."""
        return await run(bench.create_capability, name, path, target, drop, metric, task, models)

    @server.tool(title="List capabilities")
    async def list_capabilities() -> dict:
        """The capabilities saved on this machine: name, task, target, champion and its score."""
        return await run(bench.list_capabilities)

    @server.tool(title="Evaluate a capability")
    async def evaluate_capability(name: str, path: str | None = None) -> dict:
        """How good a capability is: every model's metrics on the holdout, the baselines (what you get with no
        model), columns used and dropped and, for classification, precision and recall per class. Pass `path`
        (rows with the target, ideally not used to create it) to compute the report on those rows instead."""
        return await run(bench.evaluate_capability, name, path)

    @server.tool(title="Predict")
    async def predict(name: str, path: str | None = None, rows: list[dict] | None = None,
                      output_path: str | None = None, threshold: float | None = None) -> dict:
        """Predict with a capability. Pass `path` (CSV or Parquet): predictions are written to `output_path`
        (default: <input>.predictions.csv next to it, with the input columns plus prediction, score and
        proba_<class>) and a summary and preview come back. Or pass `rows`, up to 100 records, to get the
        predictions back directly. `threshold` overrides the score cut of a yes/no capability."""
        return await run(bench.predict, name, path, rows, output_path, threshold)

    @server.tool(title="Explain a capability")
    async def explain(name: str, path: str, repeats: int = 3) -> dict:
        """Which columns matter, by permutation importance: each column is shuffled in `path` (rows with the
        target) and the drop in the capability's metric is measured."""
        return await run(bench.explain, name, path, repeats)

    @server.tool(title="Pick a threshold")
    async def optimize_threshold(name: str, objective: str = "cost", cost_fp: float = 1.0, cost_fn: float = 1.0,
                                 min_precision: float | None = None, min_recall: float | None = None,
                                 apply: bool = False) -> dict:
        """For yes/no capabilities: the score cut that is best for the business, read off the holdout.
        objective "cost" uses cost_fp (a false alarm) and cost_fn (a miss), e.g. cost_fn=20 if missing a fraud
        costs 20 false alarms; "f1"; "youden"; "recall" (most recall with precision >= min_precision);
        "precision" (most precision with recall >= min_recall). apply=true saves it as the capability's cut."""
        return await run(bench.optimize_threshold, name, objective, cost_fp, cost_fn, min_precision, min_recall,
                         apply)

    @server.tool(title="List models")
    async def list_models() -> dict:
        """The tabular foundation models ZeroGrad knows: license, commercial use, whether each is installed, and
        the pip command to install it."""
        return await run(bench.list_models)

    @server.tool(name="read_docs", title="Read the docs")
    async def read_docs_tool(query: str | None = None) -> dict:
        """The ZeroGrad docs and API reference of the installed version, offline. Use it when a tool does not
        cover what you need, or to write Python with the SDK (`import zerograd`) yourself: anomalies, imputation,
        cross-validation by time, context curves, scikit-learn, privacy, the platform. No query lists the topics;
        a topic returns that section; other words return the best matches."""
        return await run(read_docs, query)

    @server.prompt(title="Predict a column of a table")
    def predict_a_column(path: str, target: str) -> str:
        """The recommended workflow, from a file to predictions."""
        return (f"Use the ZeroGrad tools to predict '{target}' in {path}. 1) inspect_data with target='{target}'; "
                "drop ids and personal data. 2) create_capability with a short name. 3) Report the champion's "
                "metric next to the baseline, in plain words. 4) If it is a yes/no target, ask what a false alarm "
                "and a miss cost, and run optimize_threshold. 5) predict the rows with no target (or a new file) "
                "and say where the predictions were written.")

    return server


def main() -> None:
    os.environ.setdefault("ZEROGRAD_QUIET", "1")  # stdout is the protocol; keep the SDK's chatter out of the way
    build_server().run()


if __name__ == "__main__":
    main()
