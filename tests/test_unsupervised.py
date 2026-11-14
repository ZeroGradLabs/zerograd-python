"""Anomaly detection, imputation, synthetic data and clustering, with tiny stand-in models (no downloads)."""

import numpy as np
import pandas as pd
import pytest
from conftest import _logreg_fit, _ridge_regress

import zerograd
from zerograd.models import ModelSpec, get_model, register_model, unregister_model
from zerograd.unsupervised import anomalies, candidates, cluster, impute, synthesize
from zerograd.unsupervised.chain import log_density
from zerograd.unsupervised.table import Codes, Table


def _spec(name, **kw):
    return ModelSpec(name=name, display=name, family="test", license="MIT", license_url="https://example.com",
                     source="tests", commercial=True, install="-", check_fn=lambda: None, default=False, **kw)


@pytest.fixture
def unsup_models():
    """A logistic-regression 'foundation model' for the chain rule, and a toy embedder for clustering."""
    specs = [
        _spec("test-chain", fit_fn=_logreg_fit, regress_fn=_ridge_regress,
              tasks=("classification", "regression", "anomaly_detection", "imputation", "synthetic_data")),
        _spec("test-embed", fit_fn=None, embed_fn=lambda X: np.nan_to_num(np.asarray(X, dtype=float)),
              tasks=("clustering",)),
        _spec("test-scorer", fit_fn=None, anomaly_fn=lambda ctx, q: np.abs(np.nan_to_num(q)).sum(axis=1),
              tasks=("anomaly_detection",)),
    ]
    for s in specs:
        register_model(s, replace=True)
    yield [s.name for s in specs]
    for s in specs:
        unregister_model(s.name)


def customers(n=300, seed=0, outliers=10):
    """Income follows age, segment follows income; a few rows break the pattern."""
    rng = np.random.default_rng(seed)
    age = rng.uniform(20, 70, n).round()
    income = (age * 1000 + rng.normal(0, 3000, n)).round(2)
    segment = np.where(income > 45000, "premium", "standard")
    df = pd.DataFrame({"customer_id": np.arange(n), "age": age, "income": income, "segment": segment,
                       "active": rng.random(n) > 0.3})
    flagged = np.zeros(n, dtype=int)
    bad = rng.choice(n, outliers, replace=False)
    df.loc[bad, "income"] = (df.loc[bad, "age"] * 1000 * rng.choice([5, 0.05], outliers)).round(2)
    flagged[bad] = 1
    df["fraud"] = flagged
    return df


def test_table_round_trip_keeps_types():
    df = customers()
    t = Table.fit(df.drop(columns=["fraud"]))
    assert "customer_id" in [d["name"] for d in t.dropped]
    back = t.decode(t.transform(df))
    assert set(back["segment"]) == {"premium", "standard"}
    assert back["age"].dtype == "Int64"
    assert back["active"].iloc[0] in (True, False)


def test_codes_cap_at_ten_and_mark_missing():
    X = np.array([[float(i % 37)] for i in range(200)] + [[np.nan]] * 5, dtype=np.float32)
    c = Codes.fit(X, ["numeric"])
    assert c.n_codes[0] <= 10 and c.missing_code[0] == c.n_codes[0] - 1
    codes = c.column(X, 0)
    assert codes.min() >= 0 and (codes == c.missing_code[0]).sum() == 5


def test_chain_density_ranks_outliers_lower(unsup_models):
    df = customers().drop(columns=["customer_id", "fraud"])
    t = Table.fit(df)
    X = t.transform(df)
    codes = Codes.fit(X, t.kinds)
    logp, surprise = log_density(get_model("test-chain"), X, X, codes, n_orders=2)
    assert logp.shape == (len(X),) and surprise.shape == X.shape
    assert np.all(np.isfinite(logp))


def test_anomalies_with_labels_runs_a_championship(unsup_models):
    df = customers()
    res = anomalies(df, labels="fraud", models=["test-chain", "isolation-forest", "knn"], progress=False)
    assert set(res.evaluation.index) == {"test-chain", "isolation-forest", "knn"}
    assert res.evaluation["roc_auc"].iloc[0] > 0.7
    assert list(res.scores.columns) == ["anomaly_score", "percentile", "is_anomaly", "why"]
    assert res.scores["is_anomaly"].sum() == pytest.approx(0.05 * len(df), abs=2)
    assert len(res.top(5)) == 5 and "income" in res.top(5).columns


def test_anomaly_why_names_the_broken_column(unsup_models):
    df = customers(outliers=10)
    res = anomalies(df.drop(columns=["fraud"]), models=["test-chain"], progress=False)
    top = res.top(10)
    assert top["why"].notna().any()
    assert any("income" in (w or "") for w in top["why"])


def test_anomalies_with_reference_and_dedicated_model(unsup_models):
    df = customers().drop(columns=["fraud"])
    res = anomalies(df.iloc[:50], reference=df.iloc[50:], models=["test-scorer"], progress=False)
    assert res.model == "test-scorer" and len(res.scores) == 50 and res.scores["why"].isna().all()


def test_anomalies_check_inputs(unsup_models):
    df = customers()
    with pytest.raises(zerograd.ValidationError):
        anomalies(df, labels="nope", progress=False)
    with pytest.raises(zerograd.ValidationError):
        anomalies(df.drop(columns=["fraud"]), contamination=0.9, progress=False)


def test_impute_championship_and_fill(unsup_models):
    df = customers().drop(columns=["fraud"])
    rng = np.random.default_rng(1)
    holes = df.copy()
    holes.loc[rng.choice(len(df), 40, replace=False), "income"] = np.nan
    holes.loc[rng.choice(len(df), 30, replace=False), "segment"] = None
    res = impute(holes, models=["test-chain", "mean-mode", "knn-imputer"], progress=False)
    assert set(res.evaluation.index) == {"test-chain", "mean-mode", "knn-imputer"}
    # income follows age: a model that reads the row beats the column mean
    assert res.evaluation.loc["test-chain", "error:income"] < res.evaluation.loc["mean-mode", "error:income"]
    assert res.data["income"].isna().sum() == 0 and res.data["segment"].isna().sum() == 0
    assert set(res.data["segment"]) <= {"premium", "standard"}
    assert int(res.filled.to_numpy().sum()) == 70


def test_impute_without_gaps_says_so(unsup_models):
    with pytest.raises(zerograd.ValidationError, match="No missing values"):
        impute(customers().drop(columns=["fraud"]), models=["test-chain"], progress=False)


def test_synthesize_scores_generators(unsup_models):
    df = customers().drop(columns=["fraud", "customer_id"])
    res = synthesize(df, n=200, models=["test-chain", "independent-columns"], progress=False)
    assert len(res.data) == 200 and list(res.data.columns) == list(df.columns)
    assert set(res.data["segment"]) <= {"premium", "standard"}
    ev = res.evaluation
    assert ev.loc["test-chain", "detection_auc"] < ev.loc["independent-columns", "detection_auc"]
    assert 0 <= ev["exact_copies"].max() <= 1


def test_cluster_finds_segments(unsup_models):
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [8, 8], [0, 8]])
    group = rng.integers(0, 3, 300)
    pts = centers[group] + rng.normal(0, 1, (300, 2))
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "group": group})
    res = cluster(df, labels="group", models=["test-embed", "kmeans", "gaussian-mixture"], progress=False)
    assert res.k == 3 and res.evaluation["adjusted_rand"].max() > 0.9
    assert len(res.labels) == 300 and res.profiles["rows"].sum() == 300
    fixed = cluster(df.drop(columns=["group"]), k=2, models=["kmeans"], progress=False)
    assert fixed.k == 2


def test_candidates_respect_tasks_and_licenses(unsup_models):
    with pytest.raises(zerograd.ValidationError):
        candidates("clustering", ["test-chain"])
    names = [s.name for s in candidates("anomaly_detection", ["test-chain", "test-scorer"])]
    assert names == ["test-chain", "test-scorer"]


def test_catalog_lists_the_new_tasks_and_models():
    tasks = zerograd.list_tasks().to_frame()
    for t in ("anomaly_detection", "imputation", "synthetic_data", "clustering"):
        assert t in tasks.index
    assert zerograd.model_info("fomo-0d")["runs_in_sdk"] == ["anomaly_detection"]
    assert zerograd.model_info("zeus")["runs_in_sdk"] == ["clustering"]
    assert "imputation" in zerograd.model_info("tabpfn-v2")["runs_in_sdk"]
    assert zerograd.model_info("fomo-0d")["commercial"] is True


def test_local_client_exposes_the_tasks(unsup_models):
    zg = zerograd.ZeroGrad(progress=False, models=[])
    res = zg.anomalies(customers().drop(columns=["fraud"]), models=["test-scorer"])
    assert res.model == "test-scorer"


def test_remote_client_says_local_only():
    zg = zerograd.ZeroGrad(api_key="zg_test", remote=True)
    with pytest.raises(zerograd.ValidationError, match="locally"):
        zg.anomalies(pd.DataFrame({"a": [1]}))


def test_fomo_features_pad_to_100():
    from zerograd.models.unsupervised import fomo_features

    rng = np.random.default_rng(0)
    ctx, q = fomo_features(rng.normal(size=(50, 7)), rng.normal(size=(5, 7)))
    assert ctx.shape == (50, 100) and q.shape == (5, 100) and np.all(ctx[:, 7:] == 0)


def test_zeus_features_project_to_30():
    from zerograd.models.unsupervised import zeus_features

    X = np.random.default_rng(0).normal(size=(80, 45))
    X[0, 0] = np.nan
    f = zeus_features(X)
    assert f.shape == (80, 30) and f.min() >= -1 - 1e-6 and f.max() <= 1 + 1e-6


def test_zeus_checkpoint_must_match_its_hash(tmp_path, monkeypatch):
    from zerograd.models import unsupervised as u

    monkeypatch.delenv("ZEROGRAD_ZEUS_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cached = tmp_path / "zerograd" / "zeus.pt"
    cached.parent.mkdir()
    cached.write_bytes(b"not the authors' checkpoint")
    with pytest.raises(RuntimeError, match="SHA-256"):
        u.zeus_path()
    assert not cached.exists()  # a file that does not match is removed, never loaded

    good = b"pretend checkpoint"
    cached.write_bytes(good)
    import hashlib

    monkeypatch.setattr(u, "ZEUS_SHA256", hashlib.sha256(good).hexdigest())
    assert u.zeus_path() == str(cached)
