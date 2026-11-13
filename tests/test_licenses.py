
import pytest
from conftest import titanic_like

import zerograd
from zerograd import NonCommercialModelWarning, ZeroGrad
from zerograd.models import LicenseGuardError, assert_tabpfn_v2_checkpoint, get_model, resolve_models


@pytest.fixture(autouse=True)
def no_key(monkeypatch):
    monkeypatch.delenv("ZEROGRAD_API_KEY", raising=False)


def test_noncommercial_models_are_off_by_default(toy_models):
    defaults = resolve_models(None)
    assert all(m.commercial for m in defaults)
    with pytest.raises(PermissionError, match="noncommercial="):
        ZeroGrad(models=["test-logreg", "test-nc"])
    with pytest.raises(PermissionError):
        ZeroGrad(models=["tabpfn-2.5"])


def test_no_blanket_switch(toy_models):
    with pytest.raises(ValueError, match="Name the non-commercial models"):
        ZeroGrad(noncommercial=True)


def test_enabling_one_by_name_warns_with_its_license(toy_models):
    with pytest.warns(NonCommercialModelWarning, match="non-commercial model.*example.com/license.*own account"):
        zg = ZeroGrad(models=["test-logreg"], noncommercial=["test-nc"], progress=False)
    assert [s.name for s in zg.specs] == ["test-logreg", "test-nc"]
    with pytest.warns(NonCommercialModelWarning):
        ZeroGrad(noncommercial=["tabpfn-2.5"])


def test_commercial_names_do_not_go_in_noncommercial(toy_models):
    with pytest.raises(ValueError, match="commercial-use model"):
        ZeroGrad(noncommercial=["tabpfn-v2"])


def test_noncommercial_never_with_remote(toy_models):
    with pytest.raises(PermissionError, match="never available with remote=True"):
        ZeroGrad("zg_test_x", noncommercial=["tabpfn-2.5"])
    with pytest.raises(PermissionError):
        ZeroGrad(remote=True, api_key="zg_test_x", noncommercial=["test-nc"])


def test_saved_noncommercial_champion_needs_opt_in_to_load(toy_models, tmp_path):
    with pytest.warns(NonCommercialModelWarning):
        zg = ZeroGrad(models=["test-prior"], noncommercial=["test-nc"], progress=False)
    cap = zg.create("x", examples=titanic_like(200), target="Survived")
    assert cap.champion == "test-nc" and cap.noncommercial == ["test-nc"]
    path = cap.save(tmp_path / "nc.zg")
    with pytest.raises(PermissionError):
        zerograd.load(path)
    with pytest.warns(NonCommercialModelWarning):
        zerograd.load(path, noncommercial=["test-nc"])


def test_tabpfn_v2_guard_refuses_newer_weights():
    assert_tabpfn_v2_checkpoint("/x/tabpfn-v2-classifier.ckpt")
    for bad in ("tabpfn-v2.5-classifier-v2.5_default.ckpt", "tabpfn-v2_6-classifier.ckpt",
                "tabpfn-v3-classifier-v3_default.ckpt", "tabpfn-v2-regressor.ckpt"):
        with pytest.raises(LicenseGuardError):
            assert_tabpfn_v2_checkpoint(bad)


def test_guard_runs_before_loading_an_overridden_path(monkeypatch):
    from zerograd.models.commercial import tabpfn_v2_path

    monkeypatch.setenv("ZEROGRAD_TABPFN_V2_PATH", "/w/tabpfn-v2.5-classifier-v2.5_default.ckpt")
    with pytest.raises(LicenseGuardError):
        tabpfn_v2_path()


def test_prior_labs_attribution_is_the_licensed_wording():
    assert get_model("tabpfn-v2").attribution == "Built with PriorLabs-TabPFN"
    assert get_model("tabpfn-2.5").attribution == "Built with PriorLabs-TabPFN"


def test_every_model_states_license_source_and_use():
    for m in zerograd.list_models():
        assert m["license"] and m["license_url"].startswith("http") and m["weights"]
        if not m["commercial"]:
            assert "Not for production" in m["notes"] and not m["remote"] and m["license_acceptance"]
