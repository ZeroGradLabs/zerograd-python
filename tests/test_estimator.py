import numpy as np
import pytest
from conftest import houses_like, titanic_like
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

import zerograd
from zerograd import ZeroGrad
from zerograd.models import get_model, unregister_model


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    names = []
    yield names
    for n in names:
        unregister_model(n)


def test_a_scikit_learn_pair_covers_both_tasks(clean, toy_models):
    clean.append("hgb")
    spec = zerograd.register_estimator("hgb", HistGradientBoostingClassifier(max_iter=30),
                                       regressor=HistGradientBoostingRegressor(max_iter=30))
    assert spec.tasks == ("classification", "regression") and spec.display == "HistGradientBoostingClassifier"
    assert zerograd.list_models()["hgb"]["runs_in_sdk"] == ["classification", "regression"]

    zg = ZeroGrad(models=["test-prior", "hgb"], progress=False)
    cap = zg.create("titanic", examples=titanic_like(200), target="Survived")
    assert cap.champion == "hgb"
    assert cap.predict(titanic_like(5, seed=3)).shape[0] == 5

    reg = zg.create("houses", examples=houses_like(200), target="price")
    assert reg.task == "regression" and "HistGradientBoostingRegressor" not in reg.errors


def test_nan_is_filled_for_estimators_that_reject_it(clean, toy_models):
    clean.append("lr")
    zerograd.register_estimator("lr", LogisticRegression(max_iter=500))  # LogisticRegression rejects NaN
    df = titanic_like(200)
    assert df["Age"].isna().any()
    cap = ZeroGrad(models=["lr"], progress=False).create("t", examples=df, target="Survived")
    assert cap.errors == {} and cap.evaluation.loc["LogisticRegression", "roc_auc"] > 0.6


def test_any_class_with_fit_and_predict_proba_works(clean, toy_models):
    class Centroid:  # not scikit-learn: no clone, no tags
        def fit(self, X, y):
            X = np.nan_to_num(X)
            self.classes_ = np.unique(y)
            self.c = np.stack([X[y == k].mean(axis=0) for k in self.classes_])
            return self

        def predict_proba(self, X):
            d = -((np.nan_to_num(X)[:, None, :] - self.c[None]) ** 2).sum(axis=2)
            e = np.exp(d - d.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

    clean.extend(["centroid", "factory"])
    zerograd.register_estimator("centroid", Centroid, display="Centroid")   # a class
    zerograd.register_estimator("factory", lambda: Centroid(), default=False)  # a factory
    assert get_model("factory").default is False
    cap = ZeroGrad(models=["centroid", "factory"], progress=False).create("t", examples=titanic_like(150),
                                                                           target="Survived")
    assert cap.errors == {}
    proba = cap.predict(titanic_like(4, seed=1))
    assert np.allclose(proba[[c for c in proba if c.startswith("proba_")]].sum(axis=1), 1.0)


def test_the_instance_is_never_trained_in_place(clean, toy_models):
    clean.append("hgb")
    est = HistGradientBoostingClassifier(max_iter=10)
    zerograd.register_estimator("hgb", est)
    ZeroGrad(models=["hgb"], progress=False).create("t", examples=titanic_like(120), target="Survived")
    assert not hasattr(est, "classes_")


def test_save_load_and_the_sklearn_wrapper(clean, toy_models, tmp_path):
    clean.append("hgb")
    zerograd.register_estimator("hgb", HistGradientBoostingClassifier(max_iter=20))
    df = titanic_like(200)
    cap = ZeroGrad(models=["hgb"], progress=False).create("t", examples=df, target="Survived")
    path = cap.save(tmp_path / "t.zg")
    again = zerograd.load(path, progress=False)
    test = titanic_like(10, seed=5)
    assert np.allclose(again.predict(test)["score"], cap.predict(test)["score"])

    clf = zerograd.ZeroGradClassifier(model="hgb").fit(df.drop(columns="Survived"), df["Survived"])
    assert clf.predict_proba(test.drop(columns="Survived")).shape == (10, 2)


def test_errors_are_clear(clean, toy_models):
    class NoProba:
        def fit(self, X, y):
            return self

    with pytest.raises(TypeError, match="predict_proba"):
        zerograd.register_estimator("x", NoProba())
    with pytest.raises(ValueError, match="already registered"):
        zerograd.register_estimator("test-logreg", LogisticRegression())
    with pytest.raises(ValueError, match="missing"):
        zerograd.register_estimator("x", LogisticRegression(), missing="drop")
    with pytest.raises(ValueError, match="Pass an estimator"):
        zerograd.register_estimator("x")
