"""Uncertainty in the podium: bootstrap intervals and technical ties with the champion."""

import numpy as np
import pytest
from conftest import houses_like, titanic_like

from zerograd import ZeroGrad
from zerograd.local import championship as champ
from zerograd.local.championship import Candidate, add_noise, scorer


@pytest.fixture
def zg(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    return ZeroGrad(models=["test-logreg", "test-prior"], progress=False)


def _proba(p1):
    return np.column_stack([1 - p1, p1])


@pytest.mark.parametrize("metric", ["roc_auc", "accuracy", "log_loss", "f1", "pr_auc", "precision", "recall",
                                    "balanced_accuracy"])
def test_metric_value_matches_the_full_metrics(metric):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba = _proba(np.clip(0.3 * y + rng.random(300) * 0.7, 0, 1))
    assert scorer(metric, y, proba, 2)(np.ones(300)) == pytest.approx(champ.metrics(y, proba, 2)[metric])


@pytest.mark.parametrize("metric", ["rmse", "mae", "r2"])
def test_metric_value_matches_for_regression(metric):
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    pred = y + rng.normal(scale=0.5, size=200)
    got = scorer(metric, y, pred, 0, "regression")(np.ones(200))
    assert got == pytest.approx(champ.regression_metrics(y, pred)[metric])


@pytest.mark.parametrize("n_classes", [2, 3])
def test_weights_are_the_same_as_a_resample(n_classes):
    rng = np.random.default_rng(2)
    y = rng.integers(0, n_classes, 250)
    proba = rng.dirichlet(np.ones(n_classes), 250)
    proba[np.arange(250), y] += 0.2
    proba = np.round(proba / proba.sum(1, keepdims=True), 2)  # rounding makes ties, which must count as half
    idx = rng.integers(0, 250, 250)
    w = np.bincount(idx, minlength=250).astype(float)
    extra = ("f1", "precision", "recall", "pr_auc") if n_classes == 2 else ("f1_macro",)
    for metric in ("roc_auc", "log_loss", "accuracy", "balanced_accuracy", *extra):
        assert scorer(metric, y, proba, n_classes)(w) == pytest.approx(
            champ.metrics(y[idx], proba[idx], n_classes)[metric]), metric


def _cand(name, proba, y):
    return Candidate(name, name, champ.metrics(y, proba, 2), 0.0, output=proba)


def test_same_model_twice_is_a_tie_and_a_clear_gap_is_not():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = _proba(np.clip(0.6 * y + rng.random(400) * 0.4, 0, 1))
    twin = _proba(np.clip(good[:, 1] + rng.normal(scale=0.01, size=400), 0, 1))
    bad = _proba(rng.random(400))
    cands = [_cand("good", good, y), _cand("twin", twin, y), _cand("bad", bad, y)]
    add_noise(cands, cands[0], y, "roc_auc", 2)
    good_c, twin_c, bad_c = cands
    assert good_c.tied is None and good_c.ci is not None and good_c.ci[0] <= good_c.metrics["roc_auc"] <= good_c.ci[1]
    assert twin_c.tied is True
    assert bad_c.tied is False


def test_lower_is_better_metrics_compare_the_right_way():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    good = y + rng.normal(scale=0.1, size=300)
    bad = y + rng.normal(scale=2.0, size=300)
    cands = [Candidate(n, n, champ.regression_metrics(y, p), 0.0, output=p) for n, p in (("good", good), ("bad", bad))]
    add_noise(cands, cands[0], y, "rmse", 0, "regression")
    assert cands[1].tied is False and cands[1].ci[0] > cands[0].ci[1]


def test_evaluation_has_intervals_and_survives_save_load(zg, tmp_path):
    cap = zg.create("titanic", examples=titanic_like(400), target="Survived")
    ev = cap.evaluation
    assert {"ci_low", "ci_high", "tied"} <= set(ev.columns)
    champion = ev[ev["champion"]].iloc[0]
    assert champion["ci_low"] <= champion["roc_auc"] <= champion["ci_high"]
    assert ev.loc["test-prior", "tied"] in (True, False)
    path = cap.save(tmp_path / "t.zgcap")
    import zerograd

    again = zerograd.load(path)
    assert again.evaluation["ci_low"].tolist() == ev["ci_low"].tolist()


def test_podium_says_technical_tie(toy_models, monkeypatch, capsys):
    from zerograd.models import get_model, register_model, unregister_model

    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    twin = get_model("test-logreg")
    from dataclasses import replace

    register_model(replace(twin, name="test-logreg-twin", display="Twin"), replace=True)
    try:
        zg = ZeroGrad(models=["test-logreg", "test-logreg-twin", "test-prior"], progress=True)
        cap = zg.create("titanic", examples=titanic_like(400), target="Survived")
        err = capsys.readouterr().err
        assert "Technical tie" in err and "≈ tied" in err
        assert cap.evaluation.loc["Twin", "tied"] or cap.evaluation.loc["test-logreg", "tied"]
    finally:
        unregister_model("test-logreg-twin")


def test_regression_capability_gets_intervals(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    zg = ZeroGrad(models=["test-logreg", "test-prior"], progress=False)
    cap = zg.create("houses", examples=houses_like(300), target="price")
    ev = cap.evaluation
    assert ev["ci_low"].notna().all() and (ev["ci_low"] <= ev["rmse"]).all()
