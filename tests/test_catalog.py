import pytest

import zerograd


def test_lists_every_model_with_filters():
    allm = zerograd.list_models()
    names = allm.names()
    assert {"tabpfn-v2", "tabicl-v2", "mitra-v2", "mitra", "tabdpt", "tabpfn-2.5", "tabfm", "exaone-tabular",
            "limix-2"} <= set(names)
    com = zerograd.list_models(commercial=True)
    nc = zerograd.list_models(commercial=False)
    assert all(r["commercial"] for r in com) and not any(r["commercial"] for r in nc)
    assert len(com) + len(nc) == len(allm)
    assert names.index("tabpfn-v2") < names.index("tabpfn-2.5")  # commercial first
    assert zerograd.list_models(task="imputation").names() == ["limix-2"]
    assert set(zerograd.list_models(remote=True).names()) == {"tabpfn-v2", "tabicl-v2"}
    with pytest.raises(ValueError, match="Unknown task"):
        zerograd.list_models(task="clustering")


def test_rows_frames_and_lookups():
    cat = zerograd.list_models()
    row = cat["tabpfn-v2"]
    assert row["commercial"] and "classification" in row["tasks"] and row["attribution"]
    df = cat.to_frame()
    assert df.loc["tabfm", "commercial"] == False  # noqa: E712
    assert zerograd.model_info("tabdpt")["license"] == "Apache 2.0"


def test_renders_in_terminal_and_notebook():
    cat = zerograd.list_models()
    text = repr(cat)
    assert "NON-COMMERCIAL" in text and "tabpfn-v2" in text and "remote=True" in text
    page = cat._repr_html_()
    assert "Commercial use" in page and "Non-commercial" in page and "<table" in page
    assert "no models match" in repr(zerograd.list_models(task="imputation", commercial=True))


def test_tasks():
    tasks = zerograd.list_tasks()
    assert tasks.names() == ["classification", "regression", "imputation"]
    assert tasks["classification"]["in_sdk"] and not tasks["regression"]["in_sdk"]
    assert "tabpfn-v2" in tasks["classification"]["commercial_models"]
    assert "limix-2" in tasks["imputation"]["noncommercial_models"]
    assert "<table" in tasks._repr_html_() and "classification" in repr(tasks)
