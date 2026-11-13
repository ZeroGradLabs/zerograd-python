"""Turn a customer's DataFrame into the numeric matrix the foundation models read.

The models handle missing values and mixed scales themselves, so this stays small:
drop columns that cannot help (ids, free text, constants, empty), encode categories as ordinal codes,
and remember every decision so predict applies exactly the same transform.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ._errors import Invalid

_ID_NAME = re.compile(r"(^id$|^.*_id$|^.*Id$|^.*ID$|^uuid$|^index$)")


@dataclass
class Column:
    name: str
    kind: str  # "numeric" | "categorical" | "datetime" | "boolean"
    categories: list[str] = field(default_factory=list)


@dataclass
class Preprocessor:
    target: str
    task: str  # "binary" | "multiclass" | "regression"
    classes: list[Any]  # empty for regression
    columns: list[Column]
    dropped: list[dict[str, str]]
    target_stats: dict[str, float] = field(default_factory=dict)  # regression: mean, std, min, max of the target

    # ---------- fitting ----------
    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        target: str,
        *,
        drop: list[str] | None = None,
        max_columns: int = 100,
        max_classes: int = 10,
        min_rows: int = 20,
        task: str | None = None,
    ) -> tuple["Preprocessor", pd.DataFrame]:
        """Profile `df`, decide what to keep, and return the preprocessor plus the rows it kept.

        `task`: "classification", "regression", or None to decide from the target. A numeric target is
        regression when it has more distinct values than `max_classes` or any value with decimals.
        """
        if task not in (None, "classification", "regression"):
            raise Invalid(f"Unknown task '{task}'. Use 'classification', 'regression' or leave it out.",
                          code="invalid_task")
        if target not in df.columns:
            raise Invalid(
                f"Target column '{target}' is not in the data. Columns: {', '.join(map(str, df.columns[:30]))}.",
                code="missing_target",
            )
        df = df[df[target].notna()].reset_index(drop=True)
        if len(df) < min_rows:
            raise Invalid(f"Need at least {min_rows} labeled rows, got {len(df)}.", code="too_few_rows")

        y = df[target]
        if task is None:
            task = "regression" if _looks_continuous(y, max_classes) else "classification"
        stats: dict[str, float] = {}
        if task == "regression":
            num = _numeric_target(y, target)
            if num.nunique() < 2:
                raise Invalid(f"Target '{target}' has a single value ({num.iloc[0]!r}); nothing to predict.",
                              code="single_class")
            classes: list[Any] = []
            stats = {"mean": float(num.mean()), "std": float(num.std()), "min": float(num.min()),
                     "max": float(num.max())}
        else:
            classes = sorted(pd.unique(y).tolist(), key=_sort_key)
            if len(classes) < 2:
                raise Invalid(f"Target '{target}' has a single value ({classes[0]!r}); nothing to predict.",
                              code="single_class")
            if len(classes) > max_classes:
                hint = (" To predict it as a number, pass task='regression'."
                        if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y) else "")
                raise Invalid(
                    f"Target '{target}' has {len(classes)} classes; the limit is {max_classes}.{hint}",
                    code="too_many_classes",
                )

        drop = set(drop or [])
        dropped: list[dict[str, str]] = []
        columns: list[Column] = []
        n = len(df)
        for name in df.columns:
            if name == target:
                continue
            s = df[name]
            reason = _drop_reason(str(name), s, n) if name not in drop else "dropped by request"
            if reason:
                dropped.append({"name": str(name), "reason": reason})
                continue
            columns.append(_profile(str(name), s))

        if not columns:
            raise Invalid("No usable feature columns left after profiling.", code="no_features", dropped=dropped)
        if len(columns) > max_columns:
            raise Invalid(
                f"{len(columns)} usable columns; the limit is {max_columns}. Drop some with `drop=[...]`.",
                code="too_many_columns",
            )

        if task == "classification":
            task = "binary" if len(classes) == 2 else "multiclass"
        return cls(target=target, task=task, classes=classes, columns=columns, dropped=dropped,
                   target_stats=stats), df

    @property
    def regression(self) -> bool:
        return self.task == "regression"

    # ---------- applying ----------
    @property
    def feature_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c.name for c in self.columns if c.name not in df.columns]
        if missing:
            raise Invalid(
                f"Column{'s' if len(missing) > 1 else ''} {', '.join(repr(m) for m in missing)} "
                f"{'are' if len(missing) > 1 else 'is'} missing. This capability expects {_listing(self.feature_names)}.",
                code="missing_columns",
                missing=missing,
            )
        out = np.empty((len(df), len(self.columns)), dtype=np.float32)
        for j, col in enumerate(self.columns):
            out[:, j] = _encode(col, df[col.name])
        return out

    def encode_target(self, y: pd.Series) -> np.ndarray:
        if self.regression:
            return _numeric_target(y, self.target).to_numpy(dtype=np.float64)
        lookup = {_norm(c): i for i, c in enumerate(self.classes)}
        return np.array([lookup[_norm(v)] for v in y], dtype=np.int64)

    # ---------- persistence ----------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["classes"] = [_jsonable(c) for c in self.classes]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Preprocessor":
        return cls(
            target=d["target"],
            task=d["task"],
            classes=d["classes"],
            columns=[Column(**c) for c in d["columns"]],
            dropped=d["dropped"],
            target_stats=d.get("target_stats") or {},
        )

    def describe(self) -> dict[str, Any]:
        return {
            "used": [{"name": c.name, "kind": c.kind} for c in self.columns],
            "dropped": self.dropped,
        }


def _looks_continuous(y: pd.Series, max_classes: int) -> bool:
    if pd.api.types.is_bool_dtype(y) or not pd.api.types.is_numeric_dtype(y):
        return False
    v = y.to_numpy(dtype=np.float64)
    return y.nunique() > max_classes or bool(np.any(v != np.round(v)))


def _numeric_target(y: pd.Series, target: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(y):
        raise Invalid(f"Target '{target}' is true/false: that is classification, not regression.", code="invalid_task")
    num = pd.to_numeric(y, errors="coerce")
    bad = num.isna() & y.notna()
    if bad.any():
        raise Invalid(f"Regression needs a numeric target; '{target}' has values like {y[bad].iloc[0]!r}.",
                      code="non_numeric_target")
    return num.astype(np.float64)


def _listing(names: list[str], limit: int = 12) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


def _sort_key(v: Any) -> tuple[int, Any]:
    if isinstance(v, (bool, np.bool_)):
        return (0, int(v))
    if isinstance(v, (int, float, np.integer, np.floating)):
        return (0, float(v))
    return (1, str(v))


def _norm(v: Any) -> Any:
    """Make labels comparable after a JSON round trip (1 vs 1.0 vs np.int64(1))."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if float(v).is_integer() else float(v)
    return str(v)


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


def _drop_reason(name: str, s: pd.Series, n: int) -> str | None:
    non_null = s.notna().sum()
    if non_null == 0:
        return "empty"
    nunique = s.nunique(dropna=True)
    if nunique <= 1 and s.isna().sum() == 0:
        return "constant"
    is_text = s.dtype == object or isinstance(s.dtype, pd.StringDtype)
    if _ID_NAME.match(name) and nunique >= 0.95 * non_null:
        return "identifier"
    if pd.api.types.is_integer_dtype(s) and nunique == n and n > 50 and _is_sequence(s):
        return "identifier"
    if is_text and nunique > 20 and nunique > 0.5 * non_null:
        return "free text or high-cardinality id"
    return None


def _is_sequence(s: pd.Series) -> bool:
    v = np.sort(s.dropna().to_numpy())
    return len(v) > 1 and bool(np.all(np.diff(v) == 1))


def _profile(name: str, s: pd.Series) -> Column:
    if pd.api.types.is_bool_dtype(s):
        return Column(name, "boolean")
    if pd.api.types.is_numeric_dtype(s):
        return Column(name, "numeric")
    if pd.api.types.is_datetime64_any_dtype(s):
        return Column(name, "datetime")
    # Strings that are really numbers ("3.5", "12") stay numeric.
    as_num = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() > 0 and as_num.notna().sum() >= 0.98 * s.notna().sum():
        return Column(name, "numeric")
    cats = sorted({str(v) for v in s.dropna().unique()})
    return Column(name, "categorical", categories=cats)


def _encode(col: Column, s: pd.Series) -> np.ndarray:
    if col.kind == "numeric":
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)
    if col.kind == "boolean":
        return s.map(lambda v: np.nan if pd.isna(v) else float(bool(v))).to_numpy(dtype=np.float32)
    if col.kind == "datetime":
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        days = (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds() / 86400.0
        return days.to_numpy(dtype=np.float32, na_value=np.nan)
    lookup = {c: float(i) for i, c in enumerate(col.categories)}
    # Unseen categories become missing: the models already know what to do with NaN.
    return s.map(lambda v: np.nan if pd.isna(v) else lookup.get(str(v), np.nan)).to_numpy(dtype=np.float32)


def decode_label(p: Preprocessor, index: int) -> Any:
    return p.classes[index]
