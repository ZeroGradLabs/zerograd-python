import numpy as np
import pandas as pd
import pytest
from conftest import houses_like, titanic_like

import zerograd
from zerograd import ValidationError, ZeroGrad
from zerograd.local.championship import strata
from zerograd.local.preprocess import Preprocessor


@pytest.fixture
def zg(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    return ZeroGrad(models=["test-logreg", "test-prior"], progress=False)


def test_numeric_target_with_many_values_is_regression():
    pre, _ = Preprocessor.fit(houses_like(100), "price")
    assert pre.task == "regression" and pre.classes == [] and pre.regression
    assert pre.target_stats["min"] < pre.target_stats["mean"] < pre.target_stats["max"]
    assert pre.encode_target(pd.Series(["12.5", 3])).tolist() == [12.5, 3.0]
    again = Preprocessor.from_dict(pre.to_dict())
    assert again.task == "regression" and again.target_stats == pre.target_stats


def test_detection_keeps_labels_as_classification():
    df = titanic_like(100)
    assert Preprocessor.fit(df, "Survived")[0].task == "binary"
    assert Preprocessor.fit(df, "Pclass")[0].task == "multiclass"  # 3 integer values: classes
    df["ratio"] = np.where(df["Survived"] == 1, 0.5, 1.5)  # two values, but with decimals: a number
    assert Preprocessor.fit(df, "ratio")[0].task == "regression"
    assert Preprocessor.fit(df, "Pclass", task="regression")[0].task == "regression"


def test_task_errors():
    df = houses_like(100)
    with pytest.raises(ValidationError, match="task='regression'") as e:
        Preprocessor.fit(df, "price", task="classification")
    assert e.value.code == "too_many_classes"
    with pytest.raises(ValidationError) as e:
        Preprocessor.fit(df, "hood", task="regression")
    assert e.value.code == "non_numeric_target"
    with pytest.raises(ValidationError) as e:
        Preprocessor.fit(df, "price", task="forecast")
    assert e.value.code == "invalid_task"


def test_regression_championship(zg, capsys):
    zg.progress = True
    df = houses_like(500)
    cap = zg.create("prices", examples=df.iloc[:400], target="price")
    err = capsys.readouterr().err
    assert "Champion: test-logreg · RMSE" in err and "always the average" in err and "regression · price" in err
    assert cap.task == "regression" and cap.metric == "rmse" and cap.classes == []
    ev = cap.evaluation
    assert list(ev.columns[:3]) == ["rmse", "mae", "r2"]
    assert ev.loc["test-logreg", "champion"] and ev.loc["test-logreg", "rmse"] < ev.loc["test-prior", "rmse"]
    assert ev.loc["test-logreg", "r2"] > 0.8
    assert set(cap.baselines.index) == {"Always the average", "Always the median"}
    assert cap.baselines.loc["Always the average", "rmse"] > ev.loc["test-logreg", "rmse"]
    plots = next(c for c in cap.candidates if c["model"] == "test-logreg")["plots"]
    assert len(plots["scatter"]["actual"]) == len(plots["scatter"]["predicted"]) > 0
    assert sum(plots["residuals"]["count"]) == cap.rows["holdout"]

    new = df.iloc[400:].drop(columns="price")
    preds = cap.predict(new)
    assert list(preds.columns) == ["prediction"] and (preds.index == new.index).all()
    assert preds["prediction"].dtype == np.float64
    one = cap.predict(new.iloc[0].to_dict())
    assert set(one) == {"prediction"} and isinstance(one["prediction"], float)
    with pytest.raises(ValidationError):
        cap.predict(new, threshold=0.5)
    with pytest.raises(ValidationError):
        cap.update(threshold=0.5)


def test_regression_metrics_pick_the_champion(zg):
    cap = zg.create("prices", examples=houses_like(300), target="price", metric="r2")
    assert cap.metric == "r2" and cap.champion == "test-logreg"
    with pytest.raises(ValidationError) as e:
        zg.create("prices", examples=houses_like(300), target="price", metric="roc_auc")
    assert e.value.code == "invalid_metric" and "rmse" in str(e.value)
    with pytest.raises(ValidationError):
        zg.create("t", examples=titanic_like(200), target="Survived", metric="rmse")


def test_models_without_regression_sit_out(toy_models, monkeypatch, capsys):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    zg = ZeroGrad(models=["test-logreg", "test-clf-only"])
    cap = zg.create("prices", examples=houses_like(200), target="price")
    assert "test-clf-only: regression is not in this version of the SDK yet" in capsys.readouterr().err
    assert [c["model"] for c in cap.candidates] == ["test-logreg"]
    only = ZeroGrad(models=["test-clf-only"], progress=False)
    with pytest.raises(ValidationError) as e:
        only.create("prices", examples=houses_like(200), target="price")
    assert e.value.code == "unsupported_task"


def test_save_load_and_smaller_context(zg, tmp_path):
    cap = zg.create("prices", examples=houses_like(300), target="price")
    new = houses_like(30, seed=4).drop(columns="price")
    before = cap.predict(new)
    again = zerograd.load(cap.save(tmp_path / "p.zg"), progress=False)
    assert again.task == "regression" and again.metric == "rmse"
    pd.testing.assert_frame_equal(before, again.predict(new))
    again.update(context_rows=50)
    assert len(again.context_y) == 50 and again.context_y.dtype == np.float64
    assert again.predict(new)["prediction"].notna().all()


def test_regression_context_curve(zg):
    df = houses_like(800)
    curve = zg.context_curve(df, "price", sizes=[30, 100, 400])
    assert curve.metric == "rmse" and not curve.higher_is_better and curve.relative
    assert curve.tolerance == 0.005 and list(curve.index) == [30, 100, 400]
    assert curve.recommended in (100, 400) and "within 0.5% of the best" in curve.summary()
    cap = zg.create("prices", examples=df.iloc[:600], target="price")
    c2 = cap.context_curve(df.iloc[600:], sizes=[50, 200], metric="mae")
    assert c2.metric == "mae" and c2["mae"].notna().all()


def test_strata_bins_the_target():
    y = np.arange(100, dtype=float)
    s = strata(y, "regression")
    assert s.min() == 0 and s.max() == 9 and np.bincount(s).tolist() == [10] * 10
    assert strata(np.array([1, 0, 1]), "binary").tolist() == [1, 0, 1]


def test_remote_refuses_regression(monkeypatch):
    from zerograd._remote import RemoteZeroGrad

    zg = RemoteZeroGrad("zg_test_x", progress=False)
    with pytest.raises(ValidationError) as e:
        zg.create("prices", examples=houses_like(50), target="price", task="regression")
    assert e.value.code == "unsupported_task"


def test_tabpfn_guard_knows_the_regressor():
    from zerograd.models import LicenseGuardError, assert_tabpfn_v2_checkpoint

    assert_tabpfn_v2_checkpoint("/cache/tabpfn-v2-regressor.ckpt", "regressor")
    for bad, kind in (("tabpfn-v2-classifier.ckpt", "regressor"), ("tabpfn-v2.5-regressor-v2.5_default.ckpt", "regressor"),
                      ("tabpfn-v2-regressor.ckpt", "classifier")):
        with pytest.raises(LicenseGuardError):
            assert_tabpfn_v2_checkpoint(bad, kind)
