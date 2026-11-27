import numpy as np
import pandas as pd
import pytest
from conftest import houses_like, titanic_like

import zerograd
from zerograd import ZeroGrad
from zerograd.local.checks import duplicate_share, leaky_columns


@pytest.fixture
def zg(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    return ZeroGrad(models=["test-logreg", "test-prior"], progress=False)


def test_a_column_known_after_the_outcome_is_flagged(zg, capsys, tmp_path):
    df = titanic_like(400)
    rng = np.random.default_rng(1)
    df["refund_reason"] = np.where(df["Survived"] == 1, rng.choice(["a", "b"], len(df)), "none")   # a leak
    zg.progress = True
    cap = zg.create("t", examples=df, target="Survived")
    assert [w["column"] for w in cap.warnings if w["code"] == "possible_leakage"] == ["refund_reason"]
    assert "drop=['refund_reason']" in capsys.readouterr().err
    assert zerograd.load(cap.save(tmp_path / "t.zg"), progress=False).warnings == cap.warnings

    clean = zg.create("t", examples=df, target="Survived", drop=["refund_reason"])
    assert not [w for w in clean.warnings if w["code"] == "possible_leakage"]


def test_regression_leak(zg):
    df = houses_like(300)
    df["price_with_tax"] = df["price"] * 1.21
    cap = zg.create("h", examples=df, target="price")
    assert any(w.get("column") == "price_with_tax" for w in cap.warnings)


def test_ordinary_data_has_no_warnings(zg):
    assert zg.create("t", examples=titanic_like(400), target="Survived").warnings == []


def test_duplicated_rows_between_context_and_holdout(zg):
    df = pd.concat([titanic_like(150)] * 3, ignore_index=True).drop(columns="PassengerId")
    cap = zg.create("t", examples=df, target="Survived")
    dup = [w for w in cap.warnings if w["code"] == "holdout_duplicates"]
    assert dup and dup[0]["share"] > 0.5


def test_check_helpers():
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    assert duplicate_share(X, np.array([[1.0, np.nan], [9.0, 9.0]])) == 0.5
    y = np.array([0, 1] * 30)
    assert leaky_columns(np.c_[y.astype(float)], y, ["copy"], "binary")[0]["column"] == "copy"
    assert leaky_columns(np.zeros((10, 1)), y[:10], ["x"], "binary") == []   # too few rows to judge
