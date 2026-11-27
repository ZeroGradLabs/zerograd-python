"""Inpriora adapter wiring with a stand-in package: checkpoint folder, seed, device, class alignment."""

import sys
import types

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from zerograd.models import commercial, get_model


def test_inpriora_adapter_passes_the_checkpoint_folder(monkeypatch, tmp_path):
    seen = {}

    def fake(kind, base):
        class Est:
            def __init__(self, model_path=None, device=None, random_seed=None):
                seen[kind] = (model_path, device, random_seed)
                self.m = base()

            def fit(self, X, y):
                self.m.fit(X, y)
                self.classes_ = getattr(self.m, "classes_", None)
                return self

            def predict_proba(self, X):
                return self.m.predict_proba(X)

            def predict(self, X):
                return self.m.predict(X)

        return Est

    Clf, Reg = fake("clf", LogisticRegression), fake("reg", LinearRegression)

    monkeypatch.setitem(sys.modules, "inpriora", types.SimpleNamespace(InprioraClassifier=Clf, InprioraRegressor=Reg))
    monkeypatch.setattr(commercial, "_inpriora_dir", lambda kind: str(tmp_path / kind))

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int) * 2  # classes 0 and 2 of 3
    proba = get_model("inpriora").fit_fn(X, y, 3, False).predict_proba(X)
    assert proba.shape == (80, 3) and np.all(proba[:, 1] == 0)
    assert seen["clf"][0].endswith("classifier") and seen["clf"][2] == 0
    pred = get_model("inpriora").regress_fn(X, X[:, 0] * 2, False).predict(X)
    assert pred.shape == (80,) and seen["reg"][0].endswith("regressor")
    assert get_model("inpriora").weights_for("regression") == [
        ("ai-for-data/inpriora", "regressor/config.json"), ("ai-for-data/inpriora", "regressor/model.safetensors")]
