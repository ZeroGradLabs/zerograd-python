"""remote=True against a fake platform (no server, no network)."""

import json

import httpx
import pytest
from conftest import titanic_like

from zerograd import AuthError, LimitError, NotFoundError, RemoteZeroGrad, UsageWarning, ValidationError, ZeroGrad

CAP = {"id": "cap_1", "name": "titanic", "current_version": 1, "champion": "tabpfn-v2", "status": "ready",
       "classes": [0, 1], "versions": [{"version": 1, "champion_display": "TabPFN v2"}]}
EVAL = {"champion": "tabpfn-v2", "metric": "roc_auc", "rows": {"holdout": 60},
        "metrics": [{"name": "roc_auc"}, {"name": "accuracy"}],
        "candidates": [{"model": "tabpfn-v2", "display": "TabPFN v2", "roc_auc": 0.88, "accuracy": 0.8, "seconds": 3, "error": None},
                       {"model": "tabicl-v2", "display": "TabICLv2", "roc_auc": 0.86, "accuracy": 0.79, "seconds": 2, "error": None}]}


class Platform:
    def __init__(self):
        self.calls = []
        self.limit = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        assert request.headers["authorization"] == "Bearer zg_test_key"
        assert request.headers["user-agent"].startswith("zerograd-python/")
        path, m = request.url.path, request.method
        if path == "/v1/capabilities" and m == "POST":
            assert b"examples" in request.content and b"Survived" in request.content
            return httpx.Response(202, json={"run_id": "run_1", "capability_id": "cap_1"})
        if path == "/v1/runs/run_1":
            return httpx.Response(200, json={"id": "run_1", "kind": "create", "status": "succeeded", "progress": 1,
                                             "log": [{"message": "Compared 2 models"}]})
        if path == "/v1/capabilities/cap_1" and m == "GET":
            return httpx.Response(200, json=CAP)
        if path == "/v1/capabilities/cap_1/evaluation":
            return httpx.Response(200, json=EVAL)
        if path in ("/v1/capabilities/cap_1/predict", "/v1/capabilities/titanic/predict"):
            if self.limit:
                return httpx.Response(429, json={"error": "usage_limit", "message": "Monthly row limit reached."})
            headers = {"x-zerograd-usage-warning": "Workspace at 85% of its monthly rows."}
            if request.headers["content-type"].startswith("application/json"):
                n = len(json.loads(request.content)["rows"])
            else:
                n = 3
            preds = [{"prediction": 1, "score": 0.9, "proba": {"0": 0.1, "1": 0.9}}] * n
            return httpx.Response(200, json={"predictions": preds, "classes": [0, 1], "version": 1}, headers=headers)
        if path == "/v1/capabilities/nope":
            return httpx.Response(404, json={"error": "not_found", "message": "No capability 'nope'."})
        if path == "/v1/usage":
            return httpx.Response(401, json={"error": "unauthorized", "message": "Invalid API key."})
        if path == "/v1/bad":
            return httpx.Response(422, json={"error": "missing_target", "message": "Target column 'x' is not in the data."})
        return httpx.Response(500, json={"error": "server", "message": f"unexpected {m} {path}"})


@pytest.fixture
def platform():
    return Platform()


@pytest.fixture
def client(platform):
    http = httpx.Client(transport=httpx.MockTransport(platform))
    return ZeroGrad("zg_test_key", base_url="http://platform.test", http_client=http, progress=False, max_retries=0)


def test_remote_client_is_built_from_a_key(client):
    assert isinstance(client, RemoteZeroGrad) and isinstance(client, ZeroGrad) and client.remote is True


def test_remote_only_options_are_refused_locally(monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)
    with pytest.raises(TypeError, match="remote=True"):
        ZeroGrad(base_url="http://x")
    with pytest.raises(ValueError, match="local competitors"):
        ZeroGrad("zg_test_key", models=["tabpfn-v2"])


def test_create_evaluate_predict(client, platform):
    df = titanic_like(80)
    cap = client.create("titanic", examples=df, target="Survived", sensitive="keep")
    assert cap.id == "cap_1" and cap.champion_display == "TabPFN v2"
    ev = cap.evaluation
    assert ev.loc["TabPFN v2", "champion"] and not ev.loc["TabICLv2", "champion"]
    with pytest.warns(UsageWarning):
        preds = cap.predict(df.drop(columns="Survived").head(3), sensitive="keep")
    assert list(preds.columns) == ["prediction", "score", "proba_0", "proba_1"]
    one = client.predict("titanic", {"Pclass": 3, "Sex": "male", "Age": 22.0, "Fare": 7.25}, sensitive="keep")
    assert one["prediction"] == 1 and one["version"] == 1


def test_errors_map_to_types(client, platform):
    with pytest.raises(NotFoundError):
        client.capability("nope")
    with pytest.raises(AuthError):
        client.usage()
    with pytest.raises(ValidationError, match="not in the data"):
        client._t.request("GET", "/v1/bad")
    platform.limit = True
    with pytest.raises(LimitError):
        client.predict("titanic", {"Pclass": 3}, sensitive="keep")
