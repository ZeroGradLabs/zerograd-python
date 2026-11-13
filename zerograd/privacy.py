"""Keep sensitive data on your side.

Two tools, both run on your machine before anything is sent:

* ``scan`` / ``drop_sensitive`` find columns that hold personal data (emails, phones, card numbers,
  national ids, names, addresses…) by their name and by what is in them, warn, and drop them.
  ``ZeroGrad.create`` and ``Capability.predict`` do this by default.

* ``Vault`` pseudonymises a table with a key only you hold: column names and categories become keyed
  tokens, numbers go through a keyed linear map, and the target's labels become tokens too. The same
  value always gives the same token, so the model still learns. ``vault.decode`` turns predictions back
  into your labels. ZeroGrad never sees the key or the original values.

    from zerograd import ZeroGrad, Vault
    vault = Vault(os.environ["ZEROGRAD_VAULT_KEY"])          # Vault.new_key() makes one
    cap = zg.create("churn", examples=df, target="churned", vault=vault)
    preds = cap.predict(new_df, vault=vault)                  # decoded: your labels, your index

What it is and is not: pseudonymisation, not encryption. Relations between columns, how often each
category appears and the order of numbers are kept, because that is what the model learns from.
Someone holding the tokens cannot read the values, but could still see, say, that a column has 12
categories. If a column must not leave in any form, drop it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class SensitiveDataWarning(UserWarning):
    """Columns that look like personal data were found (and, by default, dropped before upload)."""


# ---------------------------------------------------------------------------------------------- scan

# Column names that usually hold personal data. Matched on the lower-cased name split into words.
_NAME_RULES: list[tuple[str, str]] = [
    ("email", r"e[-_ ]?mail|correo"),
    ("phone", r"phone|telefono|teléfono|tel|celular|cel|mobile|movil|móvil|whatsapp"),
    ("national id", r"ssn|social[-_ ]?security|dni|cuit|cuil|curp|rut|nif|nie|cpf|passport|pasaporte|tax[-_ ]?id|national[-_ ]?id"),
    ("bank or card", r"iban|cbu|cvu|swift|card[-_ ]?(number|num|no)|credit[-_ ]?card|tarjeta|cc[-_ ]?num(ber)?|pan|cvv|account[-_ ]?(number|no)"),
    ("person name", r"name|nombre|apellido|first[-_ ]?name|last[-_ ]?name|full[-_ ]?name|surname"),
    ("address", r"address|direccion|dirección|domicilio|street|calle|zip|postal|codigo[-_ ]?postal"),
    ("birth date", r"birth|dob|nacimiento|fecha[-_ ]?nac"),
    ("network id", r"ip|ip[-_ ]?address|mac[-_ ]?address|device[-_ ]?id|imei"),
]
# Names that contain a rule word but are not personal data.
_NAME_EXCEPTIONS = re.compile(r"^(file|host|product|company|item|brand|model|city|country|state|plan)[-_ ]?name$|^zip[-_ ]?code[-_ ]?region$")

_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
_PHONE = re.compile(r"^\+?[\d\s().-]{8,20}$")
_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_CUIT = re.compile(r"^\d{2}-?\d{8}-?\d$")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _cuit_ok(s: str) -> bool:
    d = re.sub(r"\D", "", s)
    if len(d) != 11:
        return False
    w = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    r = 11 - sum(int(a) * b for a, b in zip(d[:10], w)) % 11
    return int(d[10]) == (0 if r == 11 else 9 if r == 10 else r)


def _content_kind(values: pd.Series) -> tuple[str, float] | None:
    s = values.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return None
    s = s.sample(min(len(s), 500), random_state=0) if len(s) > 500 else s
    checks = [
        ("email", s.str.match(_EMAIL)),
        ("bank or card", s.map(lambda v: (lambda d: 13 <= len(d) <= 19 and re.fullmatch(r"[\d\s-]+", v) is not None and _luhn(d))(re.sub(r"\D", "", v)))),
        ("national id", s.map(lambda v: bool(_CUIT.match(v)) and _cuit_ok(v))),
        ("network id", s.str.match(_IP)),
        ("phone", s.map(lambda v: bool(_PHONE.match(v)) and sum(c.isdigit() for c in v) >= 8 and not re.fullmatch(r"\d+(\.\d+)?", v))),
    ]
    for kind, hit in checks:
        share = float(hit.mean())
        if share >= 0.6:
            return kind, share
    return None


@dataclass
class Finding:
    column: str
    kind: str
    why: str  # "name" or "content"

    def __str__(self) -> str:
        how = "its name" if self.why == "name" else "its values"
        return f"{self.column!r} looks like {self.kind} (by {how})"


def scan(df: pd.DataFrame, *, ignore: Iterable[str] = ()) -> list[Finding]:
    """Columns that look like personal data, by name and by content. Nothing is changed."""
    skip = set(ignore)
    found: list[Finding] = []
    for col in df.columns:
        if col in skip:
            continue
        # camelCase -> camel_case, then match rule words between separators
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(col).strip()).lower()
        hit = None
        if not _NAME_EXCEPTIONS.match(name):
            for kind, rule in _NAME_RULES:
                if re.search(rf"(^|[^a-z0-9áéíóúñ])({rule})([^a-z0-9áéíóúñ]|$)", name):
                    hit = Finding(str(col), kind, "name")
                    break
        if hit is None and (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            c = _content_kind(df[col])
            if c:
                hit = Finding(str(col), c[0], "content")
        if hit:
            found.append(hit)
    return found


def drop_sensitive(df: pd.DataFrame, *, keep: Iterable[str] = (), warn: bool = True, _stacklevel: int = 2) -> pd.DataFrame:
    """`df` without the columns `scan` flags (except those in `keep`), with a SensitiveDataWarning listing them."""
    found = scan(df, ignore=keep)
    if not found:
        return df
    if warn:
        warnings.warn(
            "Dropped before upload: " + "; ".join(str(f) for f in found)
            + ". Pass sensitive='keep' or keep=[...] if a column is safe to send.",
            SensitiveDataWarning,
            stacklevel=_stacklevel,
        )
    return df.drop(columns=[f.column for f in found])


# --------------------------------------------------------------------------------------------- vault

class Vault:
    """Pseudonymise tables with a key only you hold, and turn predictions back into your labels.

    The same key always gives the same tokens, so use one key per capability for its whole life
    (training and every prediction). Keep it like a password: it is what makes the tokens unreadable.
    """

    def __init__(self, key: str | bytes):
        if isinstance(key, str):
            key = key.encode()
        if len(key) < 16:
            raise ValueError("Use a key of at least 16 bytes. Vault.new_key() makes a good one.")
        self._key = key
        self._labels: dict[str, object] = {}  # token -> original target label, filled by encode()
        self._columns: dict[str, str] = {}    # token -> original column name

    @staticmethod
    def new_key() -> str:
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode().rstrip("=")

    def __repr__(self) -> str:
        return f"<Vault {len(self._columns)} columns, {len(self._labels)} labels known>"

    # tokens ------------------------------------------------------------------------------------
    def _mac(self, *parts: object) -> bytes:
        msg = "\x1f".join(str(p) for p in parts).encode()
        return hmac.new(self._key, msg, hashlib.sha256).digest()

    def column(self, name: str) -> str:
        """The token a column name becomes."""
        tok = "c_" + self._mac("col", name).hex()[:10]
        self._columns[tok] = name
        return tok

    def _value(self, col: str, v: object) -> str:
        return "v_" + self._mac("val", col, _canon(v)).hex()[:12]

    def _label(self, col: str, v: object, rank: int) -> str:
        # the rank keeps the classes in your order (so "positive" stays the second class of a yes/no target)
        tok = f"y{rank:02d}_" + self._mac("label", col, _canon(v)).hex()[:8]
        self._labels[tok] = v
        return tok

    def _labels_for(self, col: str, values: Iterable[object]) -> dict[str, str]:
        uniq = sorted({_canon(v): v for v in values if not _missing(v)}.items(), key=lambda kv: _sort_key(kv[1]))
        return {k: self._label(col, v, i) for i, (k, v) in enumerate(uniq)}

    def _affine(self, col: str) -> tuple[float, float]:
        h = self._mac("num", col)
        a = 0.5 + int.from_bytes(h[:4], "big") / 2**32 * 2.5   # scale in [0.5, 3)
        b = (int.from_bytes(h[4:8], "big") / 2**32 - 0.5) * 2e3  # shift in [-1000, 1000)
        return a, b

    # tables ------------------------------------------------------------------------------------
    def encode(self, df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
        """A copy of `df` with every column name, category and number replaced by its keyed version.

        Pass `target` when encoding training examples, so its labels become tokens that `decode` can read
        back. Missing values stay missing.
        """
        out = {}
        for col in df.columns:
            s = df[col]
            name = self.column(str(col))
            if target is not None and col == target:
                toks = self._labels_for(str(col), s.tolist())
                out[name] = s.map(lambda v: None if _missing(v) else toks[_canon(v)])  # noqa: B023 (applied now)
            elif pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
                if pd.api.types.is_datetime64_any_dtype(s):
                    a, b = self._affine(str(col))
                    num = s.astype("int64").where(s.notna()) / 86_400e9  # days
                    out[name] = num * a + b
                else:
                    out[name] = s.map(lambda v, c=str(col): None if _missing(v) else self._value(c, v))
            else:
                a, b = self._affine(str(col))
                out[name] = s.astype(float) * a + b
        return pd.DataFrame(out, index=df.index)

    def encode_row(self, row: dict) -> dict:
        return self.encode(pd.DataFrame([row])).iloc[0].where(lambda x: x.notna(), None).to_dict()

    def label(self, token: str) -> object:
        """The original target label behind a token (known once `encode` saw it, or after `load`)."""
        return self._labels.get(token, token)

    def decode(self, preds: pd.DataFrame | dict, labels: Iterable[object] | None = None, target: str | None = None) -> pd.DataFrame | dict:
        """Predictions back in your labels: `prediction` and the `proba_<label>` columns.

        Works out of the box in the session that encoded the examples. Elsewhere, `vault.load(path)` first,
        or pass the original `labels` and the `target` column name so the tokens can be rebuilt.
        """
        if labels is not None:
            if target is None:
                raise ValueError("Pass target= with labels= so the label tokens can be rebuilt.")
            self._labels_for(target, labels)
        if isinstance(preds, dict):
            out = dict(preds)
            if "prediction" in out:
                out["prediction"] = self.label(out["prediction"])
            if isinstance(out.get("proba"), dict):
                out["proba"] = {self.label(k): p for k, p in out["proba"].items()}
            return out
        df = preds.copy()
        if "prediction" in df:
            df["prediction"] = df["prediction"].map(self.label)
        df.columns = [f"proba_{self.label(c[6:])}" if re.match(r"proba_y\d\d_", str(c)) else c for c in df.columns]
        return df

    # persistence -------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Write the label and column maps (never the key) so another process can decode."""
        Path(path).write_text(json.dumps({"labels": {k: _canon(v) for k, v in self._labels.items()},
                                          "label_types": {k: type(v).__name__ for k, v in self._labels.items()},
                                          "columns": self._columns}, indent=2))

    def load(self, path: str | Path) -> "Vault":
        d = json.loads(Path(path).read_text())
        cast = {"int": int, "float": float, "bool": lambda s: s == "True", "int64": int, "float64": float}
        for k, v in d.get("labels", {}).items():
            t = d.get("label_types", {}).get(k)
            try:
                self._labels[k] = cast[t](v) if t in cast else v
            except (TypeError, ValueError):
                self._labels[k] = v
        self._columns.update(d.get("columns", {}))
        return self


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA or v is pd.NaT


def _sort_key(v: object) -> tuple:
    return (0, float(v), "") if isinstance(v, (int, float, np.integer, np.floating, bool, np.bool_)) else (1, 0.0, str(v))


def _canon(v: object) -> str:
    """One spelling per value, so 1, 1.0 and np.int64(1) give the same token."""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    return str(v)
