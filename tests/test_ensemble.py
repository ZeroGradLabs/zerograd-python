import numpy as np
import pytest
from conftest import houses_like, titanic_like
from sklearn.linear_model import LogisticRegression

import zerograd
from zerograd import ZeroGrad
from zerograd.models import get_model, resolve_models, unregister_model


class OneColumn:
    """A logistic regression that only reads column j: two of them on different columns complement each other."""

    def __init__(self, j):
        self.j = j

    def fit(self, X, y):
        self.m = LogisticRegression().fit(np.nan_to_num(X[:, [self.j]]), y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(np.nan_to_num(X[:, [self.j]]))


@pytest.fixture
def halves(monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    # titanic_like columns after PassengerId is dropped: Pclass, Sex, Age, Fare
    zerograd.register_estimator("only-pclass", lambda: OneColumn(0), display="Pclass only", default=False)
    zerograd.register_estimator("only-sex", lambda: OneColumn(1), display="Sex only", default=False)
    yield ["only-pclass", "only-sex"]
    unregister_model("only-pclass")
    unregister_model("only-sex")


def test_the_ensemble_competes_and_can_win(halves, tmp_path):
    zg = ZeroGrad(models=halves, progress=False)
    df = titanic_like(600)
    cap = zg.create("t", examples=df.iloc[:500], target="Survived")
    ev = cap.evaluation
    name = "Ensemble (Pclass only + Sex only)"
    assert name in ev.index and cap.champion == "ensemble:only-pclass+only-sex"
    assert ev.loc[name, "roc_auc"] > ev.loc[["Pclass only", "Sex only"], "roc_auc"].max()
    assert cap.champion_display == name

    test = df.iloc[500:].drop(columns="Survived")
    score = cap.predict(test)["score"].to_numpy()
    members = [get_model(n).fit(cap.context_X, cap.context_y, 2) for n in halves]
    expected = np.mean([m.predict_proba(cap.pre.transform(test))[:, 1] for m in members], axis=0)
    np.testing.assert_allclose(score, expected, rtol=1e-6)

    again = zerograd.load(cap.save(tmp_path / "t.zg"), progress=False)
    np.testing.assert_allclose(again.predict(test)["score"].to_numpy(), score, rtol=1e-6)


def test_a_tie_goes_to_the_single_model(toy_models, monkeypatch):
    # averaging with a constant prior keeps the ranking, so the ROC AUC ties: the single model wins
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    cap = ZeroGrad(models=["test-logreg", "test-prior"], progress=False).create("t", examples=titanic_like(300),
                                                                                 target="Survived")
    assert cap.champion == "test-logreg"
    assert "Ensemble (test-logreg + test-prior)" in cap.evaluation.index


def test_ensemble_false_and_a_single_model_leave_it_out(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    zg = ZeroGrad(models=["test-logreg", "test-prior"], progress=False)
    cap = zg.create("t", examples=titanic_like(200), target="Survived", ensemble=False)
    assert not any(m.startswith("Ensemble") for m in cap.evaluation.index)
    solo = ZeroGrad(models=["test-logreg"], progress=False).create("t", examples=titanic_like(200), target="Survived")
    assert list(solo.evaluation.index) == ["test-logreg"]


def test_regression_ensemble(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    cap = ZeroGrad(models=["test-logreg", "test-prior"], progress=False).create("h", examples=houses_like(300),
                                                                                 target="price")
    ev = cap.evaluation
    assert "Ensemble (test-logreg + test-prior)" in ev.index and ev["rmse"].notna().all()


def test_members_keep_their_license_gates(toy_models):
    spec = get_model("ensemble:test-logreg+test-nc")
    assert spec.commercial is False and spec.tasks == ("classification",)
    with pytest.raises(PermissionError, match="test-nc"):
        resolve_models(["ensemble:test-logreg+test-nc"])
    with pytest.warns(zerograd.NonCommercialModelWarning):
        resolve_models(["ensemble:test-logreg+test-nc"], noncommercial=["test-nc"])
    with pytest.raises(ValueError, match="two or more"):
        get_model("ensemble:test-logreg")


def test_attribution_comes_from_the_members(toy_models):
    from zerograd.models import ModelSpec, register_model

    spec = get_model("test-logreg")
    register_model(ModelSpec(**{**spec.__dict__, "name": "test-attr", "attribution": "Built with X"}), replace=True)
    try:
        assert get_model("ensemble:test-logreg+test-attr").attribution == "Built with X"
    finally:
        unregister_model("test-attr")
