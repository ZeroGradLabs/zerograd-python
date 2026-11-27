"""Kumo-Tabular adapter with random weights: shapes, class order, regression. Skipped without structured-data-models."""

import numpy as np
import pytest

pytest.importorskip("sdm")

from zerograd.models import commercial, get_model  # noqa: E402


@pytest.fixture
def random_kumo(monkeypatch):
    def model(size, task):
        import torch
        from sdm.models import KumoTabular

        return KumoTabular(task=task, size="small", pretrained=False, device="cpu").eval(), torch

    monkeypatch.setattr(commercial, "_kumo_model", model)
    monkeypatch.setattr(commercial, "KUMO_ESTIMATORS", 2)


def _data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    X[3, 1] = np.nan
    return X


def test_kumo_classifier_returns_every_class_in_order(random_kumo):
    X = _data()
    y = np.where(X[:, 0] > 0.5, 3, np.where(X[:, 0] > -0.5, 1, 0))  # class 2 never appears in the context
    fitted = get_model("kumo-tabular").fit_fn(X[:100], y[:100], 4, False)
    proba = fitted.predict_proba(X[100:])
    assert proba.shape == (20, 4)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-4)
    assert np.all(proba[:, 2] == 0)


def test_kumo_regressor_returns_one_number_per_row(random_kumo):
    X = _data()
    fitted = get_model("kumo-tabular-small").regress_fn(X[:100], X[:100, 0] * 3.0, False)
    pred = fitted.predict(X[100:])
    assert pred.shape == (20,) and np.isfinite(pred).all()


def test_kumo_specs():
    for name in ("kumo-tabular", "kumo-tabular-medium", "kumo-tabular-small"):
        spec = get_model(name)
        assert spec.commercial and spec.runs("classification") and spec.runs("regression")
        assert spec.default == (name == "kumo-tabular")
        assert ("nvidia/Kumo-Tabular", "LICENSE") in spec.weights_for("regression")
