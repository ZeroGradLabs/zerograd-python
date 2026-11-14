"""A table with no target, as the numeric matrix the models read, and back.

Same rules as the supervised preprocessor (drop ids, free text, constants and empty columns; categories become
ordinal codes; missing stays NaN), plus the way back to your dtypes for imputed and synthetic values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..errors import ValidationError
from ..local.preprocess import Column, _drop_reason, _encode, _profile

MAX_CODES = 10  # the foundation models read at most 10 classes, so every column becomes at most 10 codes


@dataclass
class Table:
    columns: list[Column]
    dropped: list[dict[str, str]]
    integer: dict[str, bool] = field(default_factory=dict)  # numeric columns that held whole numbers

    @classmethod
    def fit(cls, df: pd.DataFrame, *, drop: list[str] | None = None, max_columns: int = 100,
            min_rows: int = 20) -> "Table":
        if not isinstance(df, pd.DataFrame):
            raise ValidationError("Pass a DataFrame, or a CSV or Parquet path.")
        if len(df) < min_rows:
            raise ValidationError(f"Need at least {min_rows} rows, got {len(df)}.", code="too_few_rows")
        drop = set(drop or [])
        dropped, columns, integer = [], [], {}
        for name in df.columns:
            s = df[name]
            reason = "dropped by request" if name in drop else _drop_reason(str(name), s, len(df))
            if reason:
                dropped.append({"name": str(name), "reason": reason})
                continue
            col = _profile(str(name), s)
            columns.append(col)
            if col.kind == "numeric":
                v = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=np.float64)
                integer[col.name] = bool(len(v)) and bool(np.all(v == np.round(v)))
        if not columns:
            raise ValidationError("No usable columns left after profiling.", code="no_features", details={"dropped": dropped})
        if len(columns) > max_columns:
            raise ValidationError(f"{len(columns)} usable columns; the limit is {max_columns}. Drop some with drop=[...].",
                                  code="too_many_columns")
        return cls(columns=columns, dropped=dropped, integer=integer)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def kinds(self) -> list[str]:
        return [c.kind for c in self.columns]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c.name for c in self.columns if c.name not in df.columns]
        if missing:
            raise ValidationError(f"Missing columns: {', '.join(missing)}.", code="missing_columns", details={"missing": missing})
        out = np.empty((len(df), len(self.columns)), dtype=np.float32)
        for j, col in enumerate(self.columns):
            out[:, j] = _encode(col, df[col.name])
        return out

    def decode_column(self, j: int, values: np.ndarray) -> pd.Series:
        col = self.columns[j]
        v = np.asarray(values, dtype=np.float64)
        if col.kind == "categorical":
            cats = col.categories
            return pd.Series([None if np.isnan(x) else cats[int(np.clip(round(x), 0, len(cats) - 1))] for x in v],
                             dtype=object)
        if col.kind == "boolean":
            return pd.Series([None if np.isnan(x) else bool(round(x)) for x in v], dtype=object)
        if col.kind == "datetime":
            return pd.Series(pd.to_datetime(v * 86400.0, unit="s", utc=True, errors="coerce"))
        if self.integer.get(col.name):
            return pd.Series(np.round(v)).astype("Int64")
        return pd.Series(v)

    def decode(self, X: np.ndarray, index: Any = None) -> pd.DataFrame:
        out = pd.DataFrame({c.name: self.decode_column(j, X[:, j]).reset_index(drop=True)
                            for j, c in enumerate(self.columns)})
        if index is not None:
            out.index = index
        return out


@dataclass
class Codes:
    """Every column as at most 10 codes: quantile bins for numbers, categories for the rest, and a code for missing.

    The chain rule asks a classifier for the probability of each code, so a table of any types becomes a set of
    classification questions the foundation models already answer.
    """

    edges: list[np.ndarray | None]  # numeric: inner bin edges; None for discrete columns
    maps: list[dict[int, int] | None]  # discrete: category index -> code (rare categories share the "other" code)
    n_codes: list[int]
    missing_code: list[int | None]
    values: list[list[np.ndarray]]  # per column, per code: the reference values in it (to sample real values)

    @classmethod
    def fit(cls, X: np.ndarray, kinds: list[str], n_bins: int = MAX_CODES) -> "Codes":
        edges, maps, n_codes, missing_code, values = [], [], [], [], []
        for j, kind in enumerate(kinds):
            x = X[:, j].astype(np.float64)
            obs = x[~np.isnan(x)]
            has_missing = bool(np.isnan(x).any())
            room = n_bins - (1 if has_missing else 0)
            if kind in ("numeric", "datetime") and np.unique(obs).size > room:
                qs = np.unique(np.quantile(obs, np.linspace(0, 1, room + 1)[1:-1]))
                edges.append(qs)
                maps.append(None)
                k = len(qs) + 1
            else:
                cats, counts = np.unique(obs, return_counts=True)
                order = cats[np.argsort(-counts, kind="stable")]
                keep = order[: room - 1] if len(order) > room else order
                m = {int(c): i for i, c in enumerate(keep)}
                k = len(keep) + (1 if len(order) > room else 0)
                for c in order[len(keep):]:
                    m[int(c)] = len(keep)  # "other"
                edges.append(None)
                maps.append(m)
            miss = k if has_missing else None
            n_codes.append(k + (1 if has_missing else 0))
            missing_code.append(miss)
            codes = cls._column(x, edges[-1], maps[-1], miss)
            values.append([x[codes == c] for c in range(n_codes[-1])])
        return cls(edges, maps, n_codes, missing_code, values)

    @staticmethod
    def _column(x: np.ndarray, edges, m, miss) -> np.ndarray:
        out = np.full(len(x), -1 if miss is None else miss, dtype=np.int64)
        ok = ~np.isnan(x)
        if edges is not None:
            out[ok] = np.searchsorted(edges, x[ok], side="right")
        else:
            other = max(m.values()) if m else 0
            out[ok] = [m.get(int(v), other) for v in x[ok]]
        return out

    def column(self, X: np.ndarray, j: int) -> np.ndarray:
        """Codes of column j; -1 where a value is missing and missing never happened in the reference."""
        return self._column(X[:, j].astype(np.float64), self.edges[j], self.maps[j], self.missing_code[j])

    def sample_value(self, j: int, code: int, rng: np.random.Generator) -> float:
        pool = self.values[j][code] if code < len(self.values[j]) else np.array([])
        if code == self.missing_code[j] or len(pool) == 0:
            return np.nan
        return float(pool[rng.integers(len(pool))])
