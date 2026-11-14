"""zerograd.install_model: fixed recipes, license acceptance, Python check, and the commands it runs (nothing runs)."""

import logging
import re
import sys
from pathlib import Path

import pytest

import zerograd
from zerograd import install as inst
from zerograd.models import ModelSpec, get_model, register_model, unregister_model


def test_every_recipe_names_a_model_and_matches_the_extras():
    for name in inst.RECIPES:
        get_model(name)  # raises for an unknown name
    toml = Path(__file__).parent.parent.joinpath("pyproject.toml").read_text()  # no tomllib on Python 3.10

    def extra(name):
        return re.findall(r'"([^"]+)"', re.search(rf"^{name} = \[(.*)\]$", toml, re.M).group(1))

    assert list(inst.RECIPES["mitra-v2"].pip) == extra("mitra")
    assert list(inst.RECIPES["mitra"].pip) == extra("mitra")
    assert list(inst.RECIPES["tabdpt"].pip) == extra("tabdpt")


def test_noncommercial_needs_the_license_accepted():
    with pytest.raises(PermissionError, match="accept_license=True"):
        zerograd.install_model("tabfm", dry_run=True)
    out = zerograd.install_model("tabfm", accept_license=True, dry_run=True, progress=False)
    if out["status"] == "dry run":
        assert out["commands"] == [[sys.executable, "-m", "pip", "install", "tabfm[pytorch]"]]
    else:
        assert out["status"] == "already installed"


def test_git_recipe_clones_then_installs_editable(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(inst.sys, "version_info", (3, 12, 0))
    import dataclasses

    from zerograd.models import registry

    missing = dataclasses.replace(get_model("limix-2"), check_fn=lambda: "not installed")
    monkeypatch.setitem(registry._REGISTRY, "limix-2", missing)
    out = zerograd.install_model("limix-2", accept_license=True, dry_run=True)
    dest = str(tmp_path / "zerograd" / "src" / "limix-2")
    assert out["commands"] == [["git", "clone", "--depth", "1", "https://github.com/limix-ldm-ai/LimiX", dest],
                               [sys.executable, "-m", "pip", "install", "-e", dest]]


def test_python_too_old_is_refused(monkeypatch):
    monkeypatch.setattr(inst.sys, "version_info", (3, 11, 9))
    with pytest.raises(RuntimeError, match="Python 3.12"):
        zerograd.install_model("limix-2", accept_license=True, dry_run=True)


@pytest.fixture
def fake_model(monkeypatch):
    state = {"installed": False}
    spec = ModelSpec(name="test-install", display="Test install", family="test", license="MIT",
                     license_url="https://example.com", source="tests", commercial=True, install="-",
                     fit_fn=lambda *a: None, check_fn=lambda: None if state["installed"] else "missing", default=False)
    register_model(spec, replace=True)
    monkeypatch.setitem(inst.RECIPES, "test-install", inst.Recipe(pip=("some-package==1.0",)))
    ran = []

    def run(cmd, **kw):
        ran.append(cmd)
        state["installed"] = True
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(inst.subprocess, "run", run)
    yield ran, state
    unregister_model("test-install")


def test_installs_into_this_python(fake_model):
    ran, _ = fake_model
    out = zerograd.install_model("test-install", progress=False)
    assert ran == [[sys.executable, "-m", "pip", "install", "some-package==1.0"]]
    assert out["status"] == "installed"
    assert zerograd.install_model("test-install", progress=False)["status"] == "already installed"
    assert len(ran) == 1


def test_a_failed_install_says_so(fake_model, monkeypatch):
    monkeypatch.setattr(inst.subprocess, "run", lambda cmd, **kw: type("Failed", (), {"returncode": 1})())
    with pytest.raises(RuntimeError, match="failed"):
        zerograd.install_model("test-install", progress=False)


def test_no_recipe_points_to_pip():
    with pytest.raises(ValueError, match="no install recipe"):
        unavailable = ModelSpec(name="test-norecipe", display="No recipe", family="test", license="MIT",
                                license_url="https://example.com", source="tests", commercial=True,
                                install="pip install something", fit_fn=lambda *a: None,
                                check_fn=lambda: "missing", default=False)
        register_model(unavailable, replace=True)
        try:
            zerograd.install_model("test-norecipe")
        finally:
            unregister_model("test-norecipe")


def test_hf_token_notice_is_hidden_during_downloads(monkeypatch, caplog):
    from zerograd.weights import quiet_hub

    monkeypatch.delenv("ZEROGRAD_SHOW_HF_WARNINGS", raising=False)
    log = logging.getLogger("huggingface_hub.utils._http")
    with caplog.at_level(logging.WARNING), quiet_hub():
        log.warning("Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN.")
        log.error("a real error")
    assert "unauthenticated" not in caplog.text and "a real error" in caplog.text
    assert logging.getLogger("huggingface_hub").level == logging.NOTSET  # restored
