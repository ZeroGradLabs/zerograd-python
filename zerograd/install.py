"""Install a model's own package from a fixed recipe, so you don't have to pip install or git clone it by hand.

    zerograd.install_model("tabdpt")                               # commercial, optional
    zerograd.install_model("limix-2", accept_license=True)         # non-commercial: you accept its license first
    zerograd.install_model("exaone-tabular", dry_run=True)         # only show the commands

It installs into the Python that runs it (``sys.executable -m pip``), so it works in a notebook kernel, Colab
included. The commands come from the recipes in this file and nothing else. Installing does not turn a
non-commercial model on: it still runs only when you name it in ``noncommercial=[...]``.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ._ui import Console


@dataclass(frozen=True)
class Recipe:
    pip: tuple[str, ...] = ()        # requirements, passed as is to pip install
    git: str | None = None           # a repository cloned (depth 1) into ~/.cache/zerograd/src/<model>, then pip install -e
    python: tuple[int, int] | None = None  # the oldest Python the model's package supports
    upgrade: bool = False            # pip install --upgrade (the model needs a newer version of a package you have)
    note: str = ""                   # shown after installing


# The same requirements as the extras in pyproject.toml (tests/test_install.py checks they agree).
RECIPES: dict[str, Recipe] = {
    "mitra-v2": Recipe(pip=("autogluon.tabular[mitra]>=1.6",)),
    "mitra": Recipe(pip=("autogluon.tabular[mitra]>=1.6",)),
    "tabdpt": Recipe(pip=("tabdpt>=1.3",)),
    **{name: Recipe(pip=("structured-data-models>=0.3",), python=(3, 11))
       for name in ("kumo-tabular", "kumo-tabular-medium", "kumo-tabular-small")},
    "tabfm": Recipe(pip=("tabfm[pytorch]",)),
    "exaone-tabular": Recipe(pip=("exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git",)),
    "limix-2": Recipe(git="https://github.com/limix-ldm-ai/LimiX", python=(3, 12)),
    **{name: Recipe(pip=("tabpfn",), upgrade=True,
                    note="The weights are gated: sign in to Prior Labs and accept the license when the first run asks.")
       for name in ("tabpfn-2.5", "tabpfn-2.6", "tabpfn-3", "tabpfn-3.5")},
}


def recipe(model: str) -> Recipe | None:
    return RECIPES.get(model)


def install_model(model: str, *, accept_license: bool = False, dry_run: bool = False, quiet: bool = False,
                  progress: bool = True) -> dict:
    """Install the package a model needs, from its fixed recipe, into this Python.

    `accept_license=True` is required for non-commercial models: read the license (the link is in the error and in
    `zerograd.model_info(name)`) before you pass it. `dry_run=True` returns the commands without running them.
    `quiet=True` hides pip's output. Returns {"model", "commands", "status"}; status is "installed",
    "already installed", "restart" (installed, but this Python needs a restart to import it) or "dry run".
    """
    from .models.registry import get_model

    spec = get_model(model)
    ui = Console(progress)
    rec = RECIPES.get(spec.name)
    if rec is None:
        if spec.available:
            ui.done(f"{spec.display} is already installed (it comes with {spec.install})")
            return {"model": spec.name, "commands": [], "status": "already installed"}
        raise ValueError(f"{spec.display} has no install recipe. Install it with: {spec.install}")

    if not spec.commercial and not accept_license:
        raise PermissionError(
            f"{spec.display} is non-commercial ({spec.license}): research and evaluation only. Read its license at "
            f"{spec.license_url} and, if you accept it, call zerograd.install_model({spec.name!r}, "
            "accept_license=True). Installing does not turn it on: name it in noncommercial=[...] to use it."
        )
    if rec.python and tuple(sys.version_info[:2]) < rec.python:
        need, have = ".".join(map(str, rec.python)), ".".join(map(str, sys.version_info[:2]))
        raise RuntimeError(f"{spec.display} needs Python {need} or newer; this is Python {have}.")
    if spec.available:
        ui.done(f"{spec.display} is already installed")
        return {"model": spec.name, "commands": [], "status": "already installed"}

    commands = _commands(spec.name, rec)
    if dry_run:
        return {"model": spec.name, "commands": commands, "status": "dry run"}
    if rec.git and not shutil.which("git"):
        raise RuntimeError(f"{spec.display} installs from {rec.git} and needs git, which is not on this machine.")

    ui.say(f"{ui.accent('↓')} Installing {spec.display} {ui.dim('(' + spec.license + ')')}")
    for cmd in commands:
        if cmd[0] == "git" and Path(cmd[-1]).exists():
            continue  # cloned on an earlier run: pip install -e picks it up as it is
        ui.say(ui.dim("  $ " + " ".join(_show(c) for c in cmd)))
        out = subprocess.run(cmd, stdout=subprocess.DEVNULL if quiet else None,  # noqa: S603 (fixed recipe)
                             stderr=subprocess.STDOUT if quiet else None, check=False)
        if out.returncode != 0:
            raise RuntimeError(f"Installing {spec.display} failed ({' '.join(cmd[:4])} … exited with "
                               f"{out.returncode}). Run it with quiet=False to see why.")

    importlib.invalidate_caches()
    status = "installed" if spec.available else "restart"
    if status == "installed":
        ui.done(f"{spec.display} installed")
    else:
        ui.say(ui.warn(f"{spec.display} installed. Restart the kernel (or Python) so it can be imported."))
    if rec.note:
        ui.say(ui.dim("  " + rec.note))
    if not spec.commercial:
        ui.say(ui.dim(f"  Non-commercial: use it with ZeroGrad(noncommercial=[{spec.name!r}])."))
    return {"model": spec.name, "commands": commands, "status": status}


def _commands(name: str, rec: Recipe) -> list[list[str]]:
    pip = [sys.executable, "-m", "pip", "install"]
    if rec.upgrade:
        pip.append("--upgrade")
    cmds = []
    if rec.pip:
        cmds.append([*pip, *rec.pip])
    if rec.git:
        dest = src_dir() / name
        cmds.append(["git", "clone", "--depth", "1", rec.git, str(dest)])
        cmds.append([*pip, "-e", str(dest)])
    return cmds


def src_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "zerograd" / "src"


def _show(arg: str) -> str:
    return f'"{arg}"' if any(ch in arg for ch in " []<>@") else arg
