
import numpy as np
import pandas as pd
import pytest
from conftest import titanic_like

import zerograd
from zerograd import LocalCapability, ValidationError, ZeroGrad


@pytest.fixture
def zg(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    return ZeroGrad(models=["test-logreg", "test-prior"], progress=False)


def test_local_by_default_without_a_key(monkeypatch, toy_models):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    zg = ZeroGrad(models=["test-logreg"])
    assert zg.remote is False and isinstance(zg, ZeroGrad)


def test_key_in_env_means_remote_unless_told_otherwise(monkeypatch, toy_models):
    monkeypatch.setenv("ZEROGRAD_API_KEY", "zg_test_x")
    assert ZeroGrad().remote is True
    assert ZeroGrad(remote=False, models=["test-logreg"]).remote is False


def test_championship_picks_the_better_model(zg, capsys):
    zg.progress = True
    df = titanic_like(400)
    cap = zg.create("titanic", examples=df.iloc[:320], target="Survived")
    err = capsys.readouterr().err
    assert "Champion: test-logreg" in err and "dropped PassengerId" in err
    ev = cap.evaluation
    assert ev["champion"].sum() == 1 and ev.loc["test-logreg", "champion"]
    assert ev.loc["test-logreg", "roc_auc"] > ev.loc["test-prior", "roc_auc"]
    assert set(cap.baselines.index) == {"Coin flip (random scores)", "Always the most common class"}
    assert cap.columns["dropped"][0]["name"] == "PassengerId"

    holdout = df.iloc[320:].drop(columns="Survived")
    preds = cap.predict(holdout)
    assert list(preds.columns) == ["prediction", "score", "proba_0", "proba_1"]
    assert (preds.index == holdout.index).all()
    np.testing.assert_allclose(preds["proba_0"] + preds["proba_1"], 1, rtol=1e-6)

    one = cap.predict(holdout.iloc[0].to_dict())
    assert set(one) == {"prediction", "score", "proba"} and type(one["prediction"]) is int


def test_threshold(zg):
    cap = zg.create("t", examples=titanic_like(200), target="Survived")
    X = titanic_like(50, seed=3).drop(columns="Survived")
    assert (cap.predict(X, threshold=0.01)["prediction"] == 1).all()
    cap.update(threshold=0.99)
    assert (cap.predict(X)["prediction"] == 0).all()
    with pytest.raises(ValidationError):
        cap.update(threshold=2)


def test_save_and_load_round_trip(zg, tmp_path):
    cap = zg.create("t", examples=titanic_like(200), target="Survived")
    cap.update(threshold=0.3)
    X = titanic_like(40, seed=5).drop(columns="Survived")
    before = cap.predict(X)
    path = cap.save(tmp_path / "t.zg")
    again = zerograd.load(path, progress=False)
    assert isinstance(again, LocalCapability) and again.threshold == 0.3 and again.champion == cap.champion
    pd.testing.assert_frame_equal(before, again.predict(X))
    pd.testing.assert_frame_equal(cap.evaluation, again.evaluation)


def test_multiclass_and_string_labels(zg):
    df = titanic_like(300)
    df["Class"] = df["Pclass"].map({1: "first", 2: "second", 3: "third"})
    cap = zg.create("pclass", examples=df.drop(columns=["Pclass", "Survived"]), target="Class", metric="f1_macro")
    assert cap.task == "multiclass" and cap.classes == ["first", "second", "third"]
    preds = cap.predict(df.drop(columns=["Pclass", "Survived", "Class"]).head(5))
    assert "score" not in preds and {"proba_first", "proba_second", "proba_third"} <= set(preds)


def test_errors_are_actionable(zg):
    df = titanic_like(100)
    with pytest.raises(ValidationError, match="not in the data"):
        zg.create("x", examples=df, target="Nope")
    with pytest.raises(ValidationError, match="does not apply"):
        zg.create("x", examples=df, target="Survived", metric="f1_macro")
    with pytest.raises(ValidationError, match="remote=True"):
        zg.create("x", examples="gs://bucket/file.parquet", target="Survived")
    cap = zg.create("x", examples=df, target="Survived")
    with pytest.raises(ValidationError, match="missing"):
        cap.predict(df.drop(columns=["Sex", "Survived"]))


def test_csv_path(zg, tmp_path):
    p = tmp_path / "h.csv"
    titanic_like(150).to_csv(p, index=False)
    cap = zg.create("csv", examples=str(p), target="Survived")
    assert len(cap.predict(str(p))) == 150


def test_a_failing_model_does_not_sink_the_championship(toy_models, monkeypatch):
    from zerograd.models import ModelSpec, register_model, unregister_model

    def boom(*a):
        raise RuntimeError("no weights")

    register_model(ModelSpec(name="test-boom", display="Boom", family="t", license="MIT", license_url="-",
                             source="-", commercial=True, install="-", fit_fn=boom, check_fn=lambda: None,
                             default=False), replace=True)
    try:
        zg = ZeroGrad(remote=False, models=["test-boom", "test-logreg"], progress=False)
        cap = zg.create("x", examples=titanic_like(150), target="Survived")
        assert cap.champion == "test-logreg"
        assert "no weights" in cap.evaluation.loc["Boom", "error"]
    finally:
        unregister_model("test-boom")


def test_dates_and_numbers_read_as_text_are_kept():
    from zerograd.local.preprocess import Preprocessor

    n = 120
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "signup": pd.date_range("2024-01-01", periods=n).astype(str),   # what read_csv gives for a date column
        "amount": [f"{v:.2f}" for v in rng.random(n) * 100],            # numbers that arrived as text
        "note": [f"called on day {i}, left a message" for i in range(n)],
        "y": rng.integers(0, 2, n),
    })
    df.loc[3, "signup"] = None
    pre, _ = Preprocessor.fit(df, "y")
    kinds = {c.name: c.kind for c in pre.columns}
    assert kinds == {"signup": "datetime", "amount": "numeric"}
    assert [d["name"] for d in pre.dropped] == ["note"]
    X = pre.transform(df)
    assert X[1, 0] - X[0, 0] == 1.0 and np.isnan(X[3, 0])
    again = Preprocessor.from_dict(pre.to_dict()).transform(pd.DataFrame([{"signup": "2024-01-02", "amount": "5"}]))
    assert again[0, 0] == X[1, 0] and again[0, 1] == 5.0


def test_add_examples_grows_the_context_without_a_championship(zg, tmp_path):
    df = titanic_like(400)
    cap = zg.create("t", examples=df.iloc[:200], target="Survived")
    before, ev = len(cap.context_y), cap.evaluation.copy()
    new = df.iloc[200:300]
    assert cap.add_examples(new) is cap
    assert len(cap.context_y) == before + 100 and cap.rows["added"] == 100
    assert cap.rows["production_context"] == len(cap.context_y)
    np.testing.assert_array_equal(cap.context_y[-100:], cap.pre.encode_target(new["Survived"]))
    pd.testing.assert_frame_equal(cap.evaluation, ev)              # same champion, same leaderboard
    assert len(cap.predict(df.iloc[300:].drop(columns="Survived"))) == 100

    cap.add_examples(df.iloc[300:], max_context_rows=150)           # newest rows stay, older ones are sampled
    assert len(cap.context_y) == 150
    np.testing.assert_array_equal(cap.context_y[-100:], cap.pre.encode_target(df.iloc[300:]["Survived"]))
    again = zerograd.load(cap.save(tmp_path / "t.zg"), progress=False)
    assert len(again.context_y) == 150 and again.rows["added"] == 200


def test_add_examples_rejects_what_it_cannot_use(zg):
    df = titanic_like(200)
    cap = zg.create("t", examples=df, target="Survived")
    with pytest.raises(ValidationError, match="never seen"):
        cap.add_examples(df.assign(Survived=2).head(5))
    with pytest.raises(ValidationError, match="target column"):
        cap.add_examples(df.drop(columns="Survived").head(5))
    n = len(cap.context_y)
    cap.add_examples(df.assign(Survived=None).head(5))               # no labels: nothing to add
    assert len(cap.context_y) == n


def test_add_examples_on_regression(zg):
    from conftest import houses_like

    df = houses_like(300)
    cap = zg.create("h", examples=df.iloc[:200], target="price")
    cap.add_examples(df.iloc[200:])
    assert cap.context_y.dtype.kind == "f" and len(cap.predict(df.drop(columns="price").head(3))) == 3
