"""The real foundation models on a small task. Slow; downloads weights on first run.

    ZEROGRAD_REAL_MODELS=1 pytest tests/test_real_models.py -s
"""

import os

import pytest
from conftest import titanic_like

import zerograd

pytestmark = pytest.mark.skipif(os.environ.get("ZEROGRAD_REAL_MODELS") != "1", reason="set ZEROGRAD_REAL_MODELS=1")


@pytest.mark.parametrize("name", [m["name"] for m in zerograd.list_models(commercial=True)
                                  if "classification" in m["runs_in_sdk"]])
def test_commercial_model_learns(name, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    if not zerograd.model_info(name)["installed"]:
        pytest.skip(f"{name} is not installed")
    df = titanic_like(300)
    zg = zerograd.ZeroGrad(models=[name], progress=False)
    cap = zg.create(name, examples=df.iloc[:240], target="Survived")
    ev = cap.evaluation
    assert "error" not in ev or ev["error"].isna().all(), ev
    assert ev.iloc[0]["roc_auc"] > 0.7
    preds = cap.predict(df.iloc[240:].drop(columns="Survived"))
    assert len(preds) == 60 and preds["score"].between(0, 1).all()


@pytest.mark.parametrize("name", [m["name"] for m in zerograd.list_models(task="regression", commercial=True)
                                  if "regression" in m["runs_in_sdk"]])
def test_commercial_model_regresses(name, monkeypatch):
    from conftest import houses_like

    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    if not zerograd.model_info(name)["installed"]:
        pytest.skip(f"{name} is not installed")
    df = houses_like(300)
    zg = zerograd.ZeroGrad(models=[name], progress=False)
    cap = zg.create(name, examples=df.iloc[:240], target="price")
    ev = cap.evaluation
    assert "error" not in ev or ev["error"].isna().all(), ev
    assert cap.task == "regression" and ev.iloc[0]["r2"] > 0.6, ev
    assert ev.iloc[0]["rmse"] < cap.baselines.loc["Always the average", "rmse"]
    preds = cap.predict(df.iloc[240:].drop(columns="price"))
    assert len(preds) == 60 and preds["prediction"].notna().all()


def _customers():
    from test_unsupervised import customers

    return customers(300)


@pytest.mark.parametrize("name", ["tabpfn-v2", "tabicl-v2", "fomo-0d"])
def test_real_anomaly_detection(name):
    if not zerograd.model_info(name)["installed"]:
        pytest.skip(f"{name} is not installed")
    from zerograd.unsupervised import anomalies

    res = anomalies(_customers(), labels="fraud", models=[name, "isolation-forest"], progress=False)
    assert res.evaluation.loc[name, "roc_auc"] > 0.75, res.evaluation


@pytest.mark.parametrize("name", ["tabpfn-v2", "tabicl-v2"])
def test_real_imputation_and_synthesis(name):
    import numpy as np

    from zerograd.unsupervised import impute, synthesize

    df = _customers().drop(columns=["fraud", "customer_id"])
    holes = df.copy()
    holes.loc[np.random.default_rng(0).choice(len(df), 40, replace=False), "income"] = np.nan
    res = impute(holes, models=[name, "mean-mode"], progress=False)
    assert res.evaluation.loc[name, "error:income"] < res.evaluation.loc["mean-mode", "error:income"], res.evaluation
    fake = synthesize(df, n=200, models=[name, "independent-columns"], progress=False)
    ev = fake.evaluation
    assert ev.loc[name, "detection_auc"] < ev.loc["independent-columns", "detection_auc"], ev


def test_real_zeus_clustering():
    import numpy as np
    import pandas as pd

    from zerograd.unsupervised import cluster

    rng = np.random.default_rng(0)
    group = rng.integers(0, 3, 300)
    pts = np.array([[0, 0, 0], [6, 6, 0], [0, 6, 6]])[group] + rng.normal(0, 1, (300, 3))
    df = pd.DataFrame(pts, columns=["a", "b", "c"]).assign(group=group)
    res = cluster(df, labels="group", models=["zeus", "kmeans"], progress=False)
    assert res.evaluation.loc["zeus", "adjusted_rand"] > 0.8, res.evaluation
