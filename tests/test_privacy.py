
import numpy as np
import pandas as pd
import pytest

from zerograd import SensitiveDataWarning, Vault, drop_sensitive, scan


def frame():
    return pd.DataFrame({
        "Name": ["Ana Pérez", "Juan Gómez", "Li Wei"],
        "contact": ["ana@x.com", "juan@y.org", "li@z.net"],
        "customerPhone": ["+54 11 5555 0000", "+54 11 5555 0001", "+54 11 5555 0002"],
        "notes": ["4111 1111 1111 1111", "5500 0000 0000 0004", "4012 8888 8888 1881"],
        "product_name": ["a", "b", "c"],
        "card_type": ["visa", "master", "visa"],
        "age": [30, 41, np.nan],
        "plan": ["gold", "basic", "gold"],
        "churned": [1, 0, 1],
    })


def test_scan_by_name_and_content():
    found = {f.column: f.kind for f in scan(frame())}
    assert found == {"Name": "person name", "contact": "email", "customerPhone": "phone", "notes": "bank or card"}


def test_drop_sensitive_warns_and_keeps_what_you_ask():
    with pytest.warns(SensitiveDataWarning):
        out = drop_sensitive(frame(), keep=["contact"])
    assert "Name" not in out and "notes" not in out and "contact" in out and "age" in out


def test_vault_round_trip():
    df = frame().drop(columns=["Name", "contact", "customerPhone", "notes"])
    v = Vault(Vault.new_key())
    enc = v.encode(df, target="churned")
    # nothing readable leaves
    assert not set(df.columns) & set(enc.columns)
    assert not {"gold", "basic", "visa"} & set(enc.astype(str).values.ravel())
    # same value -> same token; numbers keep their order; missing stays missing
    plan = enc[v.column("plan")]
    assert plan.iloc[0] == plan.iloc[2] != plan.iloc[1]
    age = enc[v.column("age")]
    assert age.iloc[1] > age.iloc[0] and np.isnan(age.iloc[2])
    # labels keep their order and decode back
    y = enc[v.column("churned")]
    assert sorted(set(y))[0].startswith("y00_")
    preds = pd.DataFrame({"prediction": y, "score": [0.9, 0.1, 0.8], f"proba_{sorted(set(y))[0]}": [0.1, 0.9, 0.2],
                          f"proba_{sorted(set(y))[1]}": [0.9, 0.1, 0.8]})
    out = v.decode(preds)
    assert out["prediction"].tolist() == [1, 0, 1] and {"proba_0", "proba_1"} <= set(out.columns)


def test_vault_decodes_elsewhere_and_is_keyed():
    df = frame()[["plan", "churned"]]
    a = Vault("k" * 32).encode(df, target="churned")
    b = Vault("k" * 32).encode(df, target="churned")
    c = Vault("z" * 32).encode(df, target="churned")
    assert a.equals(b) and not a.equals(c)
    fresh = Vault("k" * 32)
    tok = a.iloc[0, 1]
    assert fresh.decode({"prediction": tok}, labels=[0, 1], target="churned")["prediction"] == 1


def test_short_key_is_refused():
    with pytest.raises(ValueError):
        Vault("short")
