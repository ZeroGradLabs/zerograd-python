"""Model weights on your disk: what ZeroGrad downloaded, downloading on purpose, and removing it.

Nothing downloads when you import zerograd or build ``ZeroGrad()``. A model's weights download the first time it
runs (or with ``zerograd.download``) into the cache its authors use: the Hugging Face cache for most models
(``HF_HOME``), ``~/.cache/zerograd`` for ZEUS. ZeroGrad keeps a list of the files it downloaded itself, in
``~/.cache/zerograd/downloads.json``, so ``remove_weights`` only deletes those and never a file another project
put in your Hugging Face cache.

    zerograd.download(["tabpfn-v2"])         # fetch now, e.g. before a demo without wifi
    zerograd.cached_weights()                 # what is on disk, how big, and who downloaded it
    zerograd.remove_weights()                 # delete what ZeroGrad downloaded
    zerograd.config(auto_remove_weights=True) # or ZEROGRAD_AUTO_REMOVE_WEIGHTS=1: delete this session's downloads at exit
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd

    from .models.registry import ModelSpec

_TRUE = {"1", "true", "yes", "on"}
_settings: dict[str, bool] = {}
_session: list[dict] = []  # files downloaded by this process
_lock = threading.Lock()
_atexit_registered = False


# ---------------------------------------------------------------- settings
def config(*, auto_remove_weights: bool | None = None) -> dict:
    """Process-wide settings. Returns them all.

    `auto_remove_weights=True` deletes, when the Python process ends (the script finishes or the notebook kernel
    shuts down or restarts), the weights ZeroGrad downloaded during it. Weights that were already on disk stay.
    The environment variable ZEROGRAD_AUTO_REMOVE_WEIGHTS=1 sets the same default.
    """
    if auto_remove_weights is not None:
        _settings["auto_remove_weights"] = bool(auto_remove_weights)
    return {"auto_remove_weights": _auto_remove()}


def _auto_remove() -> bool:
    if "auto_remove_weights" in _settings:
        return _settings["auto_remove_weights"]
    return os.environ.get("ZEROGRAD_AUTO_REMOVE_WEIGHTS", "").strip().lower() in _TRUE


# ---------------------------------------------------------------- the list of what ZeroGrad downloaded
def cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "zerograd"


def _manifest_path() -> Path:
    return cache_dir() / "downloads.json"


def _read() -> list[dict]:
    try:
        return list(json.loads(_manifest_path().read_text()).get("files", []))
    except (OSError, ValueError):
        return []


def _write(files: list[dict]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"files": files}, indent=1))
    tmp.replace(path)


def record(model: str, path: str | Path, *, repo: str | None = None, filename: str | None = None) -> None:
    """Note that ZeroGrad downloaded `path` for `model` (only call it for a file that was not on disk before)."""
    path = Path(path)
    entry = {"model": model, "repo": repo, "file": filename or path.name, "path": str(path),
             "bytes": _size(path), "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with _lock:
        files = [f for f in _read() if f["path"] != entry["path"]]
        _write([*files, entry])
        _session.append(entry)
        _arm_atexit()


def adopt(model: str, path: str | Path) -> None:
    """Put a file ZeroGrad keeps in its own cache folder on the list, if it is not there yet (not a new download)."""
    path = Path(path)
    with _lock:
        files = _read()
        if any(f["path"] == str(path) for f in files):
            return
        _write([*files, {"model": model, "repo": None, "file": path.name, "path": str(path), "bytes": _size(path),
                         "downloaded_at": None}])


def _arm_atexit() -> None:
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(_at_exit)
        _atexit_registered = True


def _at_exit() -> None:
    if _auto_remove() and _session:
        _remove_entries(list(_session))


# ---------------------------------------------------------------- downloading
def hub_file(repo: str, filename: str, *, model: str | None = None) -> str:
    """hf_hub_download, noting the file when ZeroGrad is the one that downloads it."""
    from huggingface_hub import hf_hub_download

    cached = _hub_cached(repo, filename)
    if cached:
        return cached
    path = hf_hub_download(repo_id=repo, filename=filename)
    record(model or repo, path, repo=repo, filename=filename)
    return path


def _hub_cached(repo: str, filename: str) -> str | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo, filename)
    except Exception:
        return None
    return hit if isinstance(hit, str) else None


def missing_files(spec: "ModelSpec", task: str = "classification") -> list[tuple[str, str]]:
    """The (repo, file) pairs of `spec` that are not on disk yet."""
    try:
        files = spec.weights_for("regression" if task == "regression" else "classification")
    except Exception:
        return []
    return [(r, f) for r, f in files if not _hub_cached(r, f)]


def ensure(spec: "ModelSpec", task: str = "classification") -> None:
    """Download what `spec` needs for `task` and is missing, quietly. A model that fails reports its own error."""
    for repo, filename in missing_files(spec, task):
        try:
            hub_file(repo, filename, model=spec.name)
        except Exception:
            return
    fetch = spec.extra.get("fetch")  # weights outside Hugging Face (ZEUS): the function downloads and records them
    if fetch is not None:
        try:
            fetch()
        except Exception:
            return


def download(models: Iterable[str] | str | None = None, *, task: str | None = None,
             progress: bool = True) -> "pd.DataFrame":
    """Download weights now instead of on first use. Returns `cached_weights()` for those models.

    `models`: names (see `zerograd.list_models()`); default, every default commercial model that is installed.
    `task`: "classification" or "regression"; default, both, where the model has separate weights.
    """
    from ._ui import Console, phrase
    from .models.registry import get_model, resolve_models

    if isinstance(models, str):
        models = [models]
    specs = [get_model(m) for m in models] if models is not None else resolve_models(None)
    tasks = [task] if task else ["classification", "regression"]
    ui = Console(progress)
    for spec in specs:
        reason = spec.unavailable_reason()
        if reason:
            ui.say(ui.warn(f"{spec.display}: {reason} ({spec.install})"))
            continue
        todo = list(dict.fromkeys(pair for t in tasks if _runs(spec, t) for pair in missing_files(spec, t)))
        fetch = spec.extra.get("fetch")
        if not todo and fetch is None:
            ui.done(f"{spec.display}: already on disk")
            continue
        if todo:
            ui.say(f"{ui.accent('↓')} {spec.display}: fetching weights from huggingface.co/{todo[0][0]} "
                   f"{ui.dim(phrase('download'))}")
            for repo, filename in todo:
                hub_file(repo, filename, model=spec.name)
        if fetch is not None:
            fetch()
        ui.done(f"{spec.display}: on disk")
    return cached_weights([s.name for s in specs])


def _runs(spec: "ModelSpec", task: str) -> bool:
    return spec.runs(task) or (task == "classification" and spec.weights_fn is not None)


# ---------------------------------------------------------------- what is on disk
def cached_weights(models: Iterable[str] | str | None = None) -> "pd.DataFrame":
    """Weights on disk for the known models: model, file, MB, path, and whether ZeroGrad downloaded them.

    Files another tool put in the cache show `downloaded_by_zerograd=False`; `remove_weights` leaves them alone.
    """
    import pandas as pd

    from .models.registry import _REGISTRY

    if isinstance(models, str):
        models = [models]
    wanted = set(models) if models is not None else None
    ours = {f["path"]: f for f in _read() if Path(f["path"]).exists()}
    rows, seen = [], set()
    for spec in _REGISTRY.values():
        if wanted is not None and spec.name not in wanted:
            continue
        pairs = []
        for fn in (spec.weights_fn, spec.regression_weights_fn):
            try:
                pairs += fn() if fn is not None else []
            except Exception:
                pass
        for repo, filename in dict.fromkeys(pairs):
            path = _hub_cached(repo, filename)
            if path and path not in seen:
                seen.add(path)
                rows.append(_row(spec.name, filename, path, path in ours))
    for path, f in ours.items():  # ZEUS and anything else recorded outside Hugging Face
        if path not in seen and (wanted is None or f["model"] in wanted):
            rows.append(_row(f["model"], f["file"], path, True))
    df = pd.DataFrame(rows, columns=["model", "file", "mb", "downloaded_by_zerograd", "path"])
    return df.sort_values(["model", "file"]).reset_index(drop=True)


def _row(model: str, filename: str, path: str, ours: bool) -> dict:
    return {"model": model, "file": filename, "mb": round(_size(Path(path)) / 1e6, 1),
            "downloaded_by_zerograd": ours, "path": path}


# ---------------------------------------------------------------- removing
def remove_weights(models: Iterable[str] | str | None = None, *, progress: bool = True) -> dict:
    """Delete the weights ZeroGrad downloaded (all of them, or only for `models`). Returns files and MB freed.

    Only files on ZeroGrad's own list are deleted; weights you or another library downloaded stay. The next
    run that needs a deleted model downloads it again.
    """
    from ._ui import Console

    if isinstance(models, str):
        models = [models]
    wanted = set(models) if models is not None else None
    entries = [f for f in _read() if wanted is None or f["model"] in wanted]
    freed = _remove_entries(entries)
    if progress:
        Console(True).done(f"Removed {freed['files']} file{'s' if freed['files'] != 1 else ''} · "
                           f"{freed['mb']:,.1f} MB freed")
    return freed


def _remove_entries(entries: list[dict]) -> dict:
    files, freed = 0, 0
    gone = set()
    for f in entries:
        path = Path(f["path"])
        size = _size(path)
        if _delete(path):
            files += 1
            freed += size
        gone.add(f["path"])
    with _lock:
        _write([f for f in _read() if f["path"] not in gone])
        _session[:] = [f for f in _session if f["path"] not in gone]
    return {"files": files, "mb": round(freed / 1e6, 1)}


def _delete(path: Path) -> bool:
    """Delete one file. In the Hugging Face cache, the snapshot link and its blob, then empty folders of the repo."""
    if not path.exists() and not path.is_symlink():
        return False
    blob = path.resolve() if path.is_symlink() else None
    path.unlink(missing_ok=True)
    if blob is not None and blob.exists():
        blob.unlink(missing_ok=True)
    repo_dir = next((p for p in path.parents if p.name.startswith("models--")), None)
    if repo_dir is not None and repo_dir.exists():
        if not any(p.is_file() and not p.is_symlink() for p in (repo_dir / "blobs").glob("*")):
            shutil.rmtree(repo_dir, ignore_errors=True)  # nothing left of this repo: drop refs and empty snapshots
        else:
            for d in sorted((repo_dir / "snapshots").glob("**/"), key=lambda p: len(p.parts), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
    return True


def _size(path: Path) -> int:
    try:
        return path.stat().st_size  # follows the snapshot link to the blob
    except OSError:
        return 0
