"""Weights on disk: what ZeroGrad downloads, lists and removes, the greeting, and TabPFN's CPU limit (no downloads)."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import zerograd
from zerograd import weights
from zerograd.models import ModelSpec, register_model, unregister_model


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hugging Face cache (snapshot links to blobs, like the real one) and a ZeroGrad cache in tmp_path."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("ZEROGRAD_AUTO_REMOVE_WEIGHTS", raising=False)
    monkeypatch.setattr(weights, "_settings", {})
    monkeypatch.setattr(weights, "_session", [])
    root = tmp_path / "hf"
    calls = []

    def snapshot(repo, filename):
        return root / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc" / filename

    def put(repo, filename, payload=b"w" * 1000):
        repo_dir = root / f"models--{repo.replace('/', '--')}"
        blob = repo_dir / "blobs" / f"{abs(hash((repo, filename)))}"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(payload)
        link = snapshot(repo, filename)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(blob)
        return str(link)

    def cached(repo, filename):
        p = snapshot(repo, filename)
        return str(p) if p.exists() else None

    def download(repo_id, filename):
        calls.append((repo_id, filename))
        return put(repo_id, filename)

    monkeypatch.setattr(weights, "_hub_cached", cached)
    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    spec = ModelSpec(name="test-weights", display="Test weights", family="test", license="MIT",
                     license_url="https://example.com", source="tests", commercial=True, install="-",
                     fit_fn=lambda X, y, n, s: None, check_fn=lambda: None, default=False,
                     weights_fn=lambda: [("org/model", "clf.ckpt")])
    register_model(spec, replace=True)
    yield types.SimpleNamespace(root=root, put=put, calls=calls, spec=spec)
    unregister_model("test-weights")


def test_download_records_and_remove_deletes_only_ours(hub):
    other = hub.put("someone/else", "theirs.ckpt")          # already in the cache, from another project
    table = zerograd.download(["test-weights"], progress=False)
    assert hub.calls == [("org/model", "clf.ckpt")]
    assert table["downloaded_by_zerograd"].tolist() == [True] and table["mb"].iloc[0] >= 0
    zerograd.download(["test-weights"], progress=False)      # on disk: no second download
    assert len(hub.calls) == 1

    freed = zerograd.remove_weights(progress=False)
    assert freed["files"] == 1
    assert not (hub.root / "models--org--model").exists()    # snapshot, blob and the empty repo folder
    assert Path(other).exists()
    assert zerograd.cached_weights(["test-weights"]).empty


def test_weights_already_in_the_cache_are_never_removed(hub):
    hub.put("org/model", "clf.ckpt")                         # the user had it before ZeroGrad ran
    weights.ensure(hub.spec)
    assert hub.calls == []
    assert zerograd.cached_weights(["test-weights"])["downloaded_by_zerograd"].tolist() == [False]
    assert zerograd.remove_weights(progress=False)["files"] == 0
    assert (hub.root / "models--org--model" / "snapshots" / "abc" / "clf.ckpt").exists()


def test_fit_downloads_through_the_list(hub):
    hub.spec.fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]), 2)
    assert hub.calls == [("org/model", "clf.ckpt")]
    assert [f["model"] for f in weights._read()] == ["test-weights"]


def test_auto_remove_deletes_this_sessions_downloads_at_exit(hub, monkeypatch):
    keep = hub.put("org/kept", "old.ckpt")
    weights.record("test-weights", keep)                     # downloaded by ZeroGrad in an earlier session
    monkeypatch.setattr(weights, "_session", [])
    weights.ensure(hub.spec)                                 # downloaded now
    weights._at_exit()                                       # off by default: nothing goes
    assert (hub.root / "models--org--model").exists()

    monkeypatch.setenv("ZEROGRAD_AUTO_REMOVE_WEIGHTS", "1")
    assert zerograd.config()["auto_remove_weights"] is True
    weights._at_exit()
    assert not (hub.root / "models--org--model").exists()
    assert Path(keep).exists()                               # only this session's downloads


def test_config_overrides_the_environment(hub, monkeypatch):
    monkeypatch.setenv("ZEROGRAD_AUTO_REMOVE_WEIGHTS", "1")
    assert zerograd.config(auto_remove_weights=False) == {"auto_remove_weights": False}


def test_greeting_says_installed_until_the_weights_are_on_disk(hub):
    from zerograd.local.engine import _models_line

    assert _models_line([hub.spec]) == "1 model installed: Test weights · weights download on first use"
    hub.put("org/model", "clf.ckpt")
    assert _models_line([hub.spec]) == "1 model ready: Test weights"


def test_tabpfn_v2_runs_on_cpu_past_1000_rows(monkeypatch):
    """tabpfn reads TABPFN_ALLOW_CPU_LARGE_DATASET only when first imported, so the adapter must pass the keyword."""
    from zerograd.models import commercial

    seen = {}

    class Fake:
        @classmethod
        def create_default_for_version(cls, version, **kwargs):
            seen.update(kwargs)
            return cls()

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

        def predict(self, X):
            return np.zeros(len(X))

    tabpfn = types.ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = tabpfn.TabPFNRegressor = Fake
    constants = types.ModuleType("tabpfn.constants")
    constants.ModelVersion = types.SimpleNamespace(V2="v2")
    monkeypatch.setitem(sys.modules, "tabpfn", tabpfn)
    monkeypatch.setitem(sys.modules, "tabpfn.constants", constants)
    monkeypatch.setattr(commercial, "tabpfn_v2_path", lambda kind="classifier": "tabpfn-v2-classifier.ckpt")

    commercial._tabpfn_fit(np.zeros((1500, 2)), np.zeros(1500, dtype=int), 2, False)
    assert seen["ignore_pretraining_limits"] is True
    seen.clear()
    commercial._tabpfn_regress(np.zeros((1500, 2)), np.zeros(1500), False)
    assert seen["ignore_pretraining_limits"] is True


def test_errors_read_whole():
    from zerograd._ui import error_summary

    err = ("RuntimeError: Running on CPU with more than 1000 samples is not allowed by default due to slow "
           "performance.\nTo override this behavior, set the environment variable ...")
    assert error_summary(err) == ("RuntimeError: Running on CPU with more than 1000 samples is not allowed by "
                                  "default due to slow performance.")
    assert error_summary("x" * 500).endswith("…") and len(error_summary("x" * 500)) == 300
