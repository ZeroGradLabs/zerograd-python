"""What is there: every model and every task, as a table that reads well in a notebook and in a terminal.

    import zerograd
    zerograd.list_models()                           # all of them
    zerograd.list_models(task="classification", commercial=True)
    zerograd.list_tasks()
    zerograd.list_models().to_frame()                # a DataFrame; .to_dicts() for plain dicts

The catalog reads the same registry the license guards use, so what it says is what the SDK enforces.
"""

from __future__ import annotations

import html
from typing import Any, Iterator

from .registry import TASKS, ModelSpec, all_models, get_model

_MODEL_COLUMNS = ("name", "display", "family", "commercial", "tasks", "installed", "remote", "license",
                  "license_url", "weights", "license_acceptance", "attribution", "install", "notes")


def _row(spec: ModelSpec) -> dict[str, Any]:
    reason = spec.unavailable_reason()
    return {
        "name": spec.name,
        "display": spec.display,
        "family": spec.family,
        "commercial": spec.commercial,
        "tasks": list(spec.tasks),
        "installed": reason is None,
        "unavailable_reason": reason,
        "remote": spec.on_platform,
        "default": spec.commercial and spec.default,
        "license": spec.license,
        "license_url": spec.license_url,
        "weights": spec.source,
        "license_acceptance": spec.acceptance,
        "attribution": spec.attribution,
        "install": spec.install,
        "notes": spec.notes,
    }


class _Table:
    """A list of dict rows with a DataFrame view and notebook/terminal renderings."""

    rows: list[dict[str, Any]]
    key: str

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int | str) -> dict[str, Any]:
        if isinstance(item, str):
            for r in self.rows:
                if r[self.key] == item:
                    return r
            raise KeyError(item)
        return self.rows[item]

    def names(self) -> list[str]:
        return [r[self.key] for r in self.rows]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.rows]

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows).set_index(self.key) if self.rows else pd.DataFrame()


class ModelCatalog(_Table):
    key = "name"

    def __init__(self, rows: list[dict[str, Any]], title: str):
        self.rows = rows
        self.title = title

    def __repr__(self) -> str:
        if not self.rows:
            return f"{self.title}: no models match."
        head = ("model", "use", "tasks", "installed", "remote", "license")
        lines = [(r["name"], "commercial" if r["commercial"] else "NON-COMMERCIAL", ", ".join(r["tasks"]),
                  "yes" if r["installed"] else "no", "yes" if r["remote"] else "no", r["license"]) for r in self.rows]
        widths = [max(len(str(x)) for x in col) for col in zip(head, *lines)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        out = [self.title, fmt.format(*head), fmt.format(*("-" * w for w in widths))]
        out += [fmt.format(*line) for line in lines]
        if any(not r["commercial"] for r in self.rows):
            out.append("\nNon-commercial models are off by default: research and evaluation only, never with remote=True.")
        return "\n".join(out)

    def _repr_html_(self) -> str:
        if not self.rows:
            return f"<p><b>{html.escape(self.title)}</b>: no models match.</p>"
        body = []
        for r in self.rows:
            use = (_badge("Commercial use", "#0f7b3f", "#e3f5ea") if r["commercial"]
                   else _badge("Non-commercial", "#9a5b00", "#fff1d6"))
            tasks = " ".join(_badge(t, "#334", "#eef0f4" if TASKS.get(t, {}).get("in_sdk") else "#f7f7f9",
                                    dim=not TASKS.get(t, {}).get("in_sdk")) for t in r["tasks"])
            installed = "✓" if r["installed"] else f'<span title="{html.escape(r["unavailable_reason"] or "")}">—</span>'
            remote = "✓" if r["remote"] else "—"
            install = "" if r["installed"] else (
                f'<code style="font-size:11px;white-space:normal;word-break:break-word">{html.escape(r["install"])}</code>')
            extra = []
            if r["license_acceptance"]:
                extra.append(f'Accept first: {html.escape(r["license_acceptance"])}')
            if r["attribution"]:
                extra.append(f'Show: “{html.escape(r["attribution"])}”')
            if r["notes"]:
                extra.append(html.escape(r["notes"]))
            body.append(
                "<tr>"
                f'<td><b>{html.escape(r["display"])}</b><br><code style="font-size:11px">{html.escape(r["name"])}</code></td>'
                f"<td>{use}</td><td>{tasks}</td>"
                f'<td style="text-align:center;max-width:240px">{installed}<br>{install}</td>'
                f'<td style="text-align:center">{remote}</td>'
                f'<td><a href="{html.escape(r["license_url"])}" target="_blank">{html.escape(r["license"])}</a>'
                f'<br><span style="font-size:11px;opacity:.75">Weights: {html.escape(r["weights"])}</span></td>'
                f'<td style="font-size:11px;opacity:.85">{"<br>".join(extra)}</td>'
                "</tr>"
            )
        foot = ""
        if any(not r["commercial"] for r in self.rows):
            foot = ('<p style="font-size:12px;opacity:.8">Non-commercial models are off by default, for research and '
                    "evaluation only, and never run with <code>remote=True</code>. Enable one by name: "
                    "<code>ZeroGrad(noncommercial=[\"tabpfn-2.5\"])</code>. You are responsible for its license.</p>")
        return (
            f'<div><p style="margin:4px 0"><b>{html.escape(self.title)}</b> · {len(self.rows)} models · '
            "faded task = the model can do it, the SDK does not run it yet</p>"
            '<table style="border-collapse:collapse;font-size:13px">'
            "<thead><tr><th style='text-align:left'>Model</th><th style='text-align:left'>Use</th>"
            "<th style='text-align:left'>Tasks</th><th>Installed</th><th>remote=True</th>"
            "<th style='text-align:left'>License · weights</th><th style='text-align:left'>Notes</th></tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>{foot}</div>"
        )


class TaskCatalog(_Table):
    key = "task"

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __repr__(self) -> str:
        head = ("task", "in SDK", "models (commercial / non-commercial)", "description")
        lines = [(r["task"], "yes" if r["in_sdk"] else "not yet",
                  f'{len(r["commercial_models"])} / {len(r["noncommercial_models"])}', r["description"]) for r in self.rows]
        widths = [max(len(str(x)) for x in col) for col in zip(head, *lines)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        return "\n".join(["Tasks", fmt.format(*head), fmt.format(*("-" * w for w in widths)),
                          *(fmt.format(*line) for line in lines)])

    def _repr_html_(self) -> str:
        body = []
        for r in self.rows:
            status = (_badge("In the SDK", "#0f7b3f", "#e3f5ea") if r["in_sdk"]
                      else _badge("Not in the SDK yet", "#556", "#f0f1f4"))
            com = ", ".join(html.escape(n) for n in r["commercial_models"]) or "—"
            nc = ", ".join(html.escape(n) for n in r["noncommercial_models"]) or "—"
            body.append(f"<tr><td><b>{html.escape(r['display'])}</b><br><code style='font-size:11px'>{r['task']}</code></td>"
                        f"<td>{status}</td><td>{html.escape(r['description'])}</td>"
                        f"<td style='font-size:12px'>{com}</td><td style='font-size:12px'>{nc}</td></tr>")
        return ("<div><p style='margin:4px 0'><b>Tasks</b></p><table style='border-collapse:collapse;font-size:13px'>"
                "<thead><tr><th style='text-align:left'>Task</th><th style='text-align:left'>Status</th>"
                "<th style='text-align:left'>What it does</th><th style='text-align:left'>Commercial models</th>"
                "<th style='text-align:left'>Non-commercial models</th></tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table></div>")


def _badge(text: str, fg: str, bg: str, dim: bool = False) -> str:
    style = f"display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;color:{fg};background:{bg};white-space:nowrap"
    if dim:
        style += ";opacity:.55"
    return f'<span style="{style}">{html.escape(text)}</span>'


def list_models(task: str | None = None, *, commercial: bool | None = None, installed: bool | None = None,
                remote: bool | None = None) -> ModelCatalog:
    """Every model the SDK knows, filtered by task, commercial use, installed here, or available with remote=True.

    Returns a ModelCatalog: iterate it for dicts, `.to_frame()` for a DataFrame, `catalog["tabpfn-v2"]` for one row.
    """
    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Tasks: {', '.join(TASKS)}.")
    specs = sorted(all_models(), key=lambda m: (not m.commercial, not m.default, not m.on_platform, m.family, m.name))
    rows = [_row(s) for s in specs]
    if task is not None:
        rows = [r for r in rows if task in r["tasks"]]
    if commercial is not None:
        rows = [r for r in rows if r["commercial"] is commercial]
    if installed is not None:
        rows = [r for r in rows if r["installed"] is installed]
    if remote is not None:
        rows = [r for r in rows if r["remote"] is remote]
    bits = [f"task={task}" if task else None,
            None if commercial is None else ("commercial" if commercial else "non-commercial"),
            None if installed is None else ("installed" if installed else "not installed"),
            None if remote is None else ("remote" if remote else "local only")]
    title = "Models" + (f" ({', '.join(b for b in bits if b)})" if any(bits) else "")
    return ModelCatalog(rows, title)


def list_tasks() -> TaskCatalog:
    """Every task, whether this version of the SDK runs it, and which models do it."""
    rows = []
    for name, t in TASKS.items():
        models = [m for m in all_models() if name in m.tasks]
        rows.append({
            "task": name, "display": t["display"], "in_sdk": t["in_sdk"], "description": t["description"],
            "commercial_models": [m.name for m in models if m.commercial],
            "noncommercial_models": [m.name for m in models if not m.commercial],
        })
    return TaskCatalog(rows)


def model_info(name: str) -> dict[str, Any]:
    """One model's catalog row."""
    return _row(get_model(name))
