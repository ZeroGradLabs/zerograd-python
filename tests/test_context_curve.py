import matplotlib
import numpy as np
import pandas as pd
import pytest
from conftest import titanic_like

from zerograd import ContextCurve, ValidationError, ZeroGrad
from zerograd.curve import check_sizes, default_sizes, recommend

matplotlib.use("Agg")


@pytest.fixture
def zg(toy_models, monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    return ZeroGrad(models=["test-logreg", "test-prior"], progress=False)


def test_recommend_picks_the_smallest_size_within_tolerance():
    sizes = [250, 500, 1000, 2000]
    assert recommend([0.80, 0.86, 0.869, 0.870], sizes, tolerance=0.005) == 1000
    assert recommend([0.80, 0.86, 0.869, 0.870], sizes, tolerance=0.0) == 2000
    assert recommend([0.50, 0.40, 0.31, 0.30], sizes, higher_is_better=False, tolerance=0.02) == 1000
    assert recommend([None, float("nan"), 0.7, 0.6], sizes) == 1000
    assert recommend([None, None], [1, 2]) is None


def test_default_sizes_end_at_the_full_context():
    assert default_sizes(3_000) == [250, 500, 1_000, 2_000, 3_000]
    assert default_sizes(200) == [200]
    assert check_sizes([500, 250, 250], 1_000) == [250, 500]
    with pytest.raises(ValueError, match="2,000"):
        check_sizes([500, 2_000], 1_000)


def test_local_context_curve(zg):
    df = titanic_like(1_200)
    curve = zg.context_curve(df, "Survived", sizes=[50, 200, 800])
    assert isinstance(curve, ContextCurve) and isinstance(curve, pd.DataFrame)
    assert list(curve.index) == [50, 200, 800] and curve.index.name == "rows"
    assert {"roc_auc", "fit_seconds", "predict_seconds", "ms_per_row", "recommended"} <= set(curve.columns)
    assert curve.model == "test-logreg" and curve.metric == "roc_auc"
    assert curve.recommended in curve.index and curve["recommended"].sum() == 1
    assert curve.loc[curve.recommended, "recommended"]
    assert "rows of context" in curve.summary()
    assert "recommended" in curve._repr_html_()
    # slices are plain frames, not curves without their metadata
    assert type(curve[["roc_auc"]]) is pd.DataFrame


def test_local_context_curve_with_an_explicit_test_and_metric(zg):
    df = titanic_like(900, seed=1)
    curve = zg.context_curve(df.iloc[:700], "Survived", test=df.iloc[700:], model="test-prior", metric="log_loss")
    assert curve.model == "test-prior" and list(curve.index)[-1] == 700
    # the prior model ignores the features: every size scores about the same, so the smallest wins
    assert curve.recommended == curve.index.min()
    with pytest.raises(ValidationError, match="target column"):
        zg.context_curve(df.iloc[:700], "Survived", test=df.iloc[700:].drop(columns="Survived"))


def test_capability_context_curve_and_shrinking_the_context(zg):
    df = titanic_like(1_000, seed=2)
    cap = zg.create("titanic", examples=df.iloc[:800], target="Survived")
    full = len(cap.context_y)
    curve = cap.context_curve(df.iloc[800:], sizes=[100, 300, full])
    assert curve.model == "test-logreg" and list(curve.index) == [100, 300, full]
    cap.update(threshold=0.4)
    before = cap.predict(df.iloc[800:].drop(columns="Survived"))
    cap.update(context_rows=300)
    assert len(cap.context_y) == 300 and cap.threshold == 0.4  # shrinking keeps the threshold
    after = cap.predict(df.iloc[800:].drop(columns="Survived"))
    assert len(after) == len(before)
    assert set(np.unique(cap.context_y)) == {0, 1}
    cap.update()
    assert cap.threshold is None


def test_plot_marks_the_recommended_size(zg):
    curve = zg.context_curve(titanic_like(600), "Survived", sizes=[60, 240, 480])
    ax = curve.plot()
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("recommended:") for t in texts)
    assert ax.get_xscale() == "log"
    assert len(ax.figure.axes) == 2  # score and latency on two axes


def test_plot_with_pandas_keywords_falls_back_to_dataframe_plot(zg):
    curve = zg.context_curve(titanic_like(400), "Survived", sizes=[40, 160])
    ax = curve.plot(y="roc_auc", marker="o", ylim=(0.5, 1), title="ROC AUC")
    assert ax.get_title() == "ROC AUC" and ax.get_ylim() == (0.5, 1)
