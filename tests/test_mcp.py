import anyio
import pandas as pd
import pytest
from conftest import titanic_like

from zerograd import ValidationError
from zerograd.mcp_server import Workbench, build_server, read_docs


@pytest.fixture
def bench(toy_models, tmp_path, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    return Workbench(tmp_path / "home", models=["test-logreg", "test-prior"])


@pytest.fixture
def files(tmp_path):
    df = titanic_like(400)
    train, new = tmp_path / "train.csv", tmp_path / "new.parquet"
    df.iloc[:320].to_csv(train, index=False)
    df.iloc[320:].drop(columns="Survived").to_parquet(new, index=False)
    test = tmp_path / "test.csv"
    df.iloc[320:].to_csv(test, index=False)
    return train, new, test


def test_inspect_suggests_the_target(bench, files):
    out = bench.inspect_data(str(files[0]), target="Survived")
    assert out["rows"] == 320 and "Survived" in out["suggested_targets"]
    assert out["target"]["labeled_rows"] == 320 and len(out["preview"]) == 5


def test_create_evaluate_predict_and_reopen(bench, files, tmp_path):
    train, new, test = files
    out = bench.create_capability("titanic", str(train), "Survived")
    assert out["champion"] == "test-logreg" and out["score"] > 0.5 and not out["replaced"]
    assert {r["model"] for r in out["leaderboard"]} == {"test-logreg", "test-prior"}

    fresh = Workbench(bench.home)  # a new session reads the saved file
    assert [c["name"] for c in fresh.list_capabilities()["capabilities"]] == ["titanic"]
    ev = fresh.evaluate_capability("titanic")
    assert any(r["label"] == "accuracy" for r in ev["classification_report"])

    pred = fresh.predict("titanic", path=str(new))
    written = pd.read_csv(pred["output_path"])
    assert pred["output_path"].endswith("new.predictions.csv") and len(written) == 80
    assert {"prediction", "score", "Pclass"} <= set(written.columns) and len(pred["preview"]) == 20

    one = fresh.predict("titanic", rows=[{"Pclass": 1, "Sex": "female", "Age": 30}])
    assert len(one["predictions"]) == 1 and "missing_columns" in one

    imp = fresh.explain("titanic", str(test), repeats=1)
    assert imp["importance"] and "column" in imp["importance"][0]


def test_threshold_applies_and_persists(bench, files):
    bench.create_capability("titanic", str(files[0]), "Survived")
    out = bench.optimize_threshold("titanic", "cost", cost_fp=1, cost_fn=10, apply=True)
    assert 0 < out["threshold"] < 1 and out["applied"]
    assert Workbench(bench.home).evaluate_capability("titanic")["threshold"] == out["threshold"]


def test_clear_errors(bench, files):
    with pytest.raises(ValidationError, match="No capability named"):
        bench.predict("nope", path=str(files[1]))
    with pytest.raises(ValidationError, match="name"):
        bench.create_capability("bad name!", str(files[0]), "Survived")
    with pytest.raises(ValidationError, match="No column"):
        bench.create_capability("x", str(files[0]), "nope")
    bench.create_capability("titanic", str(files[0]), "Survived")
    with pytest.raises(ValidationError, match="either path"):
        bench.predict("titanic")
    with pytest.raises(ValidationError, match="Up to 100"):
        bench.predict("titanic", rows=[{"Pclass": 1}] * 101)


def test_docs_offline():
    assert "Install" in read_docs()["topics"]
    hits = [r["topic"] for r in read_docs("threshold cost")["results"]]
    assert "API: LocalCapability.optimize_threshold" in hits
    assert read_docs("zzqqxx")["results"] == []


def test_server_lists_the_tools(bench):
    server = build_server(bench)

    async def names():
        return [t.name for t in await server.list_tools()]

    assert set(anyio.run(names)) == {"inspect_data", "create_capability", "list_capabilities", "evaluate_capability",
                                     "predict", "explain", "optimize_threshold", "list_models", "read_docs"}
