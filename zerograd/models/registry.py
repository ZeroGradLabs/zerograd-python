"""The model registry: every foundation model the local engine can run, with its license.

Each entry knows how to check that it can run (package installed), how to fit on a context, and what
its license allows. Weights are never shipped with this package: every adapter downloads them from the
author's official repository into your own Hugging Face cache, the first time you use the model.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class FittedModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class FittedRegressor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# Every task a registered model can do. `in_sdk`: this version of the SDK runs it today.
TASKS: dict[str, dict] = {
    "classification": {"display": "Classification", "in_sdk": True,
                       "description": "Predict a category: binary (churn, fraud, default) or multiclass (up to 10 classes)."},
    "regression": {"display": "Regression", "in_sdk": True,
                   "description": "Predict a number (amount, price, demand, time). Runs locally in this version."},
    "anomaly_detection": {"display": "Anomaly detection", "in_sdk": True,
                          "description": "Rank rows by how unusual they are, with no labels: fraud, errors, rare cases. "
                                         "Says which columns make each row unusual."},
    "imputation": {"display": "Imputation", "in_sdk": True,
                   "description": "Fill in missing values from the rest of the row, after a championship on cells we hide."},
    "synthetic_data": {"display": "Synthetic data", "in_sdk": True,
                       "description": "New rows with the same columns and relations as yours, scored by how hard they "
                                      "are to tell apart from the real ones."},
    "clustering": {"display": "Clustering", "in_sdk": True,
                   "description": "Group rows into segments with no labels, and describe what sets each segment apart."},
}


class NonCommercialModelWarning(UserWarning):
    """Shown every time a non-commercial model is enabled."""


NONCOMMERCIAL_NOTICE = (
    "{display} is a non-commercial model. Review its license before using it: {license_url}. "
    "Its weights come from {source}, downloaded with your own account; ZeroGrad does not host, "
    "redistribute or license them, and you are responsible for complying with the author's terms. "
    "Not for production or any commercial use."
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display: str
    family: str
    license: str
    license_url: str
    source: str  # where the weights come from (the author's official repository)
    commercial: bool
    install: str  # the pip command that makes it available
    fit_fn: Callable[[np.ndarray, np.ndarray, int, bool], FittedModel]
    check_fn: Callable[[], str | None]
    attribution: str | None = None
    max_classes: int | None = None
    notes: str = ""
    default: bool = True  # commercial models: part of the default championship when installed
    tasks: tuple[str, ...] = ("classification",)  # what the model itself can do (see TASKS)
    on_platform: bool = False  # also runs on the ZeroGrad platform (remote=True)
    acceptance: str | None = None  # a license you accept before the download, and where
    weights_fn: Callable[[], list[tuple[str, str]]] | None = None  # (repo, file) on Hugging Face, for the downloader
    extra: dict = field(default_factory=dict)
    # Regression: (X, y float, serving) -> a model with predict(X) -> one number per row. None: not in the SDK yet.
    regress_fn: Callable[[np.ndarray, np.ndarray, bool], FittedRegressor] | None = None
    regression_weights_fn: Callable[[], list[tuple[str, str]]] | None = None
    # Anomaly detection with a dedicated model: (X context, X query) -> one score per query row, higher = more unusual.
    # Classifiers need none: they detect anomalies, impute and synthesize through zerograd.unsupervised's chain rule.
    anomaly_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    # Clustering: X -> one embedding row per input row, clustered with k-means.
    embed_fn: Callable[[np.ndarray], np.ndarray] | None = None

    @property
    def sdk_tasks(self) -> tuple[str, ...]:
        """The tasks this version of the SDK runs with this model (a subset of `tasks`)."""
        ok = {"classification": self.fit_fn is not None, "regression": self.regress_fn is not None,
              "anomaly_detection": self.anomaly_fn is not None or self.fit_fn is not None,
              "imputation": self.fit_fn is not None, "synthetic_data": self.fit_fn is not None,
              "clustering": self.embed_fn is not None}
        return tuple(t for t in TASKS if t in self.tasks and ok.get(t))

    def runs(self, task: str) -> bool:
        """task: "classification", "regression", or a capability's task ("binary", "multiclass", "regression")."""
        return ("classification" if task in ("binary", "multiclass") else task) in self.sdk_tasks

    def weights_for(self, task: str) -> list[tuple[str, str]]:
        fn = self.regression_weights_fn if task == "regression" else self.weights_fn
        return fn() if fn is not None else []

    def unavailable_reason(self) -> str | None:
        return self.check_fn()

    @property
    def available(self) -> bool:
        return self.unavailable_reason() is None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool = False) -> FittedModel:
        """serving=True keeps the encoded context (KV cache) where the model supports it: faster repeated predicts."""
        _ensure_weights(self, "classification")
        return self.fit_fn(X, y, n_classes, serving)

    def fit_regressor(self, X: np.ndarray, y: np.ndarray, serving: bool = False) -> FittedRegressor:
        """Fit on a numeric target. predict(X) returns one float per row."""
        if self.regress_fn is None:
            raise NotImplementedError(f"{self.display} does not run regression in this version of the SDK")
        _ensure_weights(self, "regression")
        return Point(self.regress_fn(X, np.asarray(y, dtype=np.float64), serving))

    def describe(self) -> dict:
        reason = self.unavailable_reason()
        return {
            "name": self.name,
            "display": self.display,
            "family": self.family,
            "commercial_use": "yes" if self.commercial else "no",
            "tasks": list(self.tasks),
            "runs_in_sdk": list(self.sdk_tasks),
            "remote": self.on_platform,
            "license_acceptance": self.acceptance,
            "license": self.license,
            "license_url": self.license_url,
            "weights": self.source,
            "attribution": self.attribution,
            "available": reason is None,
            "unavailable_reason": reason,
            "install": self.install,
        }


def _ensure_weights(spec: ModelSpec, task: str) -> None:
    """Weights a model is about to need, downloaded through zerograd.weights so they are on its list (and removable)."""
    if spec.weights_fn is None and spec.regression_weights_fn is None:
        return
    from ..weights import ensure

    ensure(spec, task)


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec, *, replace: bool = False) -> ModelSpec:
    """Add a model to the registry (your own, or an adapter we do not ship). Returns the spec."""
    if spec.name in _REGISTRY and not replace:
        raise ValueError(f"A model named '{spec.name}' is already registered; pass replace=True to override it.")
    _REGISTRY[spec.name] = spec
    return spec


def unregister_model(name: str) -> None:
    _REGISTRY.pop(name, None)


def get_model(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Known models: {', '.join(_REGISTRY)}.") from None


def all_models() -> list[ModelSpec]:
    return list(_REGISTRY.values())


def resolve_models(models: list[str] | None, noncommercial: list[str] | tuple[str, ...] | bool = ()) -> list[ModelSpec]:
    """Which models a championship runs.

    `models=None` means every default commercial model that is installed. Non-commercial models only run when
    named in `noncommercial` (a list of names, not a blanket switch) and each one shows its license notice.
    """
    if noncommercial is True:
        raise ValueError(
            "Name the non-commercial models you want, e.g. noncommercial=['tabpfn-2.5']. "
            "Each has its own license; there is no switch that turns them all on."
        )
    allowed_nc = set(noncommercial or ())
    for name in allowed_nc:
        spec = get_model(name)
        if spec.commercial:
            raise ValueError(f"'{name}' is a commercial-use model; list it in models=, not noncommercial=.")

    if models is None:
        chosen = [m for m in _REGISTRY.values() if m.commercial and m.default and m.available]
        chosen += [get_model(n) for n in allowed_nc]
    else:
        chosen = [get_model(n) for n in models]
        for spec in chosen:
            if not spec.commercial and spec.name not in allowed_nc:
                raise PermissionError(
                    f"'{spec.name}' is a non-commercial model and is off by default. To use it for research or "
                    f"evaluation, enable it explicitly with noncommercial=['{spec.name}'] after reviewing its "
                    f"license: {spec.license_url}"
                )
        chosen += [get_model(n) for n in allowed_nc if n not in {s.name for s in chosen}]

    for spec in chosen:
        if not spec.commercial:
            warn_noncommercial(spec)
    return chosen


def warn_noncommercial(spec: ModelSpec) -> None:
    warnings.warn(
        NONCOMMERCIAL_NOTICE.format(display=spec.display, license_url=spec.license_url, source=spec.source),
        NonCommercialModelWarning,
        stacklevel=4,
    )


# ---------------------------------------------------------------- shared helpers for adapters
class Aligned:
    """Wraps a classifier so predict_proba always returns every class 0..n_classes-1, in order."""

    def __init__(self, model, n_classes: int, seen: list[int] | None = None, predict: Callable | None = None):
        self.model = model
        self.n_classes = n_classes
        self._predict = predict or model.predict_proba
        if seen is None:
            seen = [int(c) for c in getattr(model, "classes_", range(n_classes))]
        self.seen = seen

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = np.asarray(self._predict(X), dtype=np.float64)
        if self.seen == list(range(self.n_classes)) and proba.shape[1] == self.n_classes:
            return proba
        out = np.zeros((len(X), self.n_classes))
        out[:, self.seen] = proba[:, : len(self.seen)]
        return out


class Point:
    """Wraps a regressor so predict always returns a flat float64 array, one value per row."""

    def __init__(self, model, predict: Callable | None = None):
        self.model = model
        self._predict = predict or model.predict

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict(X), dtype=np.float64).reshape(-1)


def device() -> str:
    forced = os.environ.get("ZEROGRAD_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"  # Apple GPUs: ZEROGRAD_DEVICE=mps
    except ImportError:
        return "cpu"


def missing(*modules: str) -> Callable[[], str | None]:
    """A check that passes when every module imports."""

    def check() -> str | None:
        import importlib.util

        for mod in modules:
            try:
                found = importlib.util.find_spec(mod) is not None
            except ModuleNotFoundError:  # a dotted name whose parent package is missing
                found = False
            if not found:
                return f"the {mod} package is not installed"
        return None

    return check
