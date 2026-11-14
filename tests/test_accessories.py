"""scikit-learn estimators, cross-validation by group and time, permutation importance, threshold by objective."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import zerograd
from zerograd import ZeroGradClassifier, ZeroGradRegressor
from zerograd.local.threshold import optimize


def credit(n=400, seed=0):
    """Default follows debt ratio; `noise` is pure noise; rows repeat per customer over time."""
    rng = np.random.default_rng(seed)
    ratio = rng.uniform(0, 1, n)
    df = pd.DataFrame({
        "customer": rng.integers(0, 40, n),
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "ratio": ratio,
        "noise": rng.normal(size=n),
        "plan": rng.choice(["basic", "plus"], n),
    })
    df["default"] = (ratio + rng.normal(0, 0.15, n) > 0.7).astype(int)
    df["amount"] = 1000 * ratio + rng.normal(0, 50, n)
    return df


# ---------------------------------------------------------------------- scikit-learn estimators
def test_classifier_works_with_sklearn(toy_models):
    df = credit()
    X, y = df[["ratio", "noise", "plan"]], df["default"].map({0: "ok", 1: "late"})
    clf = ZeroGradClassifier(model="test-logreg")
    assert clone(clf).get_params()["model"] == "test-logreg"
    scores = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
    assert scores.mean() > 0.85
    clf.fit(X, y)
    assert list(clf.classes_) == ["late", "ok"]
    proba = clf.predict_proba(X.iloc[:5])
    assert proba.shape == (5, 2) and np.allclose(proba.sum(axis=1), 1)
    assert set(clf.predict(X)) <= {"late", "ok"}
    assert list(clf.feature_names_in_) == ["ratio", "noise", "plan"]


def test_classifier_on_numpy_in_a_pipeline_and_grid_search(toy_models):
    df = credit()
    X, y = df[["ratio", "noise"]].to_numpy(), df["default"].to_numpy()
    pipe = make_pipeline(FunctionTransformer(), ZeroGradClassifier(model="test-logreg"))
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.8
    grid = GridSearchCV(ZeroGradClassifier(), {"model": ["test-logreg", "test-prior"]}, cv=3, scoring="roc_auc")
    grid.fit(X, y)
    assert grid.best_params_["model"] == "test-logreg"


def test_regressor_and_context_cap(toy_models):
    df = credit()
    reg = ZeroGradRegressor(model="test-logreg", max_context_rows=100).fit(df[["ratio", "noise"]], df["amount"])
    assert reg.context_rows_ == 100
    assert reg.score(df[["ratio", "noise"]], df["amount"]) > 0.9
    with pytest.raises(zerograd.ValidationError, match="columns"):
        reg.predict(np.zeros((3, 5)))


def test_estimator_refuses_a_model_without_the_task(toy_models):
    with pytest.raises(zerograd.ValidationError, match="regression"):
        ZeroGradRegressor(model="test-clf-only").fit(np.random.rand(30, 2), np.random.rand(30))


# ---------------------------------------------------------------------- cross-validation
def _zg(toy_models):
    return zerograd.ZeroGrad(models=["test-logreg", "test-prior"], progress=False)


def test_cross_validate_random(toy_models):
    res = _zg(toy_models).cross_validate(credit(), "default", folds=4)
    assert res.split == "random" and res.folds["fold"].nunique() == 4
    s = res.summary
    assert res.best == "test-logreg" and s.loc["test-logreg", "roc_auc"] > 0.85
    assert {"ks", "lift_at_10", "roc_auc_std", "folds"} <= set(s.columns)
    assert "baseline:majority" in s.index and s.loc["baseline:majority", "ks"] == 0
    assert "customer" not in res.folds.columns


def test_cross_validate_by_group_keeps_groups_apart(toy_models, monkeypatch):
    from zerograd.local import validation

    df = credit()
    seen = {}
    real = validation.splits

    def spy(*a, **k):
        how, parts = real(*a, **k)
        seen["parts"], seen["groups"] = parts, k["groups"]
        return how, parts

    monkeypatch.setattr(validation, "splits", spy)
    res = _zg(toy_models).cross_validate(df, "default", folds=5, groups="customer")
    assert res.split == "group"
    g = seen["groups"]
    for tr, te in seen["parts"]:
        assert not set(g[tr]) & set(g[te])


def test_cross_validate_by_time_trains_on_the_past(toy_models, monkeypatch):
    from zerograd.local import validation

    df = credit()
    seen = {}
    real = validation.splits

    def spy(*a, **k):
        how, parts = real(*a, **k)
        seen["parts"], seen["times"] = parts, k["times"]
        return how, parts

    monkeypatch.setattr(validation, "splits", spy)
    res = _zg(toy_models).cross_validate(df, "amount", folds=3, time="date")
    assert res.split == "time" and res.metric == "rmse" and res.best == "test-logreg"
    t = seen["times"]
    for tr, te in seen["parts"]:
        assert t[tr].max() <= t[te].min()


def test_cross_validate_checks_inputs(toy_models):
    zg = _zg(toy_models)
    with pytest.raises(zerograd.ValidationError, match="not a column"):
        zg.cross_validate(credit(), "default", groups="nope")
    with pytest.raises(zerograd.ValidationError, match="not both"):
        zg.cross_validate(credit(), "default", groups="customer", time="date")
    with pytest.raises(zerograd.ValidationError, match="locally"):
        zerograd.ZeroGrad(api_key="zg_test", remote=True).cross_validate(credit(), "default")


# ---------------------------------------------------------------------- threshold and importance
@pytest.fixture
def cap(toy_models):
    df = credit(800)
    return _zg(toy_models).create("credit", df.drop(columns=["amount", "customer", "date"]), "default"), df


def test_optimize_threshold_by_cost_moves_the_cut(cap):
    cap, df = cap
    cheap_miss = cap.optimize_threshold("cost", cost_fp=10, cost_fn=1, apply=False)
    dear_miss = cap.optimize_threshold("cost", cost_fp=1, cost_fn=10)
    assert dear_miss.threshold < cheap_miss.threshold
    assert cap.threshold == dear_miss.threshold
    assert "championship holdout" in dear_miss.source and len(dear_miss.table) == 99
    rec = cap.optimize_threshold("recall", min_precision=0.7, data=df, apply=False)
    assert rec.metrics["precision"] >= 0.7 and "rows you passed" in rec.source
    assert 0 < cap.optimize_threshold("f1").threshold < 1
    # survives save and load
    cap2 = zerograd.load(cap.save(__import__("tempfile").mktemp(suffix=".zg")), progress=False)
    assert cap2.optimize_threshold("youden").threshold > 0


def test_optimize_threshold_checks():
    counts = {"t": [0.25, 0.5, 0.75], "tp": [9, 6, 2], "fp": [6, 2, 0], "fn": [1, 4, 8], "tn": [4, 8, 10]}
    assert optimize("f1", counts=counts).threshold == 0.25
    assert optimize("precision", counts=counts, min_recall=0.5).threshold == 0.5
    with pytest.raises(zerograd.ValidationError, match="min_precision"):
        optimize("recall", counts=counts)
    with pytest.raises(zerograd.ValidationError, match="No threshold"):
        optimize("recall", counts=counts, min_precision=1.1)
    with pytest.raises(zerograd.ValidationError, match="Unknown objective"):
        optimize("money", counts=counts)


def test_importance_ranks_the_signal_first(cap):
    cap, df = cap
    imp = cap.importance(credit(300, seed=5), repeats=2)
    assert imp.index[0] == "ratio" and imp.loc["ratio", "importance"] > 0.1
    assert abs(imp.loc["noise", "importance"]) < 0.05
    assert imp.attrs["metric"] == "roc_auc"


def test_threshold_only_for_binary(toy_models):
    df = credit()
    reg = _zg(toy_models).create("amount", df[["ratio", "noise", "amount"]], "amount")
    with pytest.raises(zerograd.ValidationError, match="binary"):
        reg.optimize_threshold()
    imp = reg.importance(df[["ratio", "noise", "amount"]], repeats=1)
    assert imp.index[0] == "ratio" and imp.attrs["metric"] == "rmse"
