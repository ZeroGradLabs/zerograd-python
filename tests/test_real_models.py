"""The real foundation models on a small task. Slow; downloads weights on first run.

    ZEROGRAD_REAL_MODELS=1 pytest tests/test_real_models.py -s
"""

import os

import pytest
from conftest import titanic_like

import zerograd

pytestmark = pytest.mark.skipif(os.environ.get("ZEROGRAD_REAL_MODELS") != "1", reason="set ZEROGRAD_REAL_MODELS=1")


@pytest.mark.parametrize("name", [m["name"] for m in zerograd.list_models(commercial=True)])
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
