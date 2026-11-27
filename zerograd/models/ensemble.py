"""The ensemble the championship adds: every model that finished, averaged with equal weights.

Its name lists the members (`ensemble:tabpfn-v2+tabicl-v2`), so a saved capability can rebuild it with
`get_model` and the license gates check each member. Equal weights on purpose: weights tuned on the holdout
would make the ensemble look better there than it is.
"""

from __future__ import annotations

import numpy as np

from .registry import ModelSpec

PREFIX = "ensemble:"


def is_ensemble(name: str) -> bool:
    return name.startswith(PREFIX)


def ensemble_name(members: list[str]) -> str:
    return PREFIX + "+".join(members)


def member_names(name: str) -> list[str]:
    return [m for m in name[len(PREFIX):].split("+") if m]


class _MeanProba:
    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.mean([np.asarray(m.predict_proba(X), dtype=np.float64) for m in self.models], axis=0)


class _MeanPoint:
    def __init__(self, models: list):
        self.models = models

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([np.asarray(m.predict(X), dtype=np.float64).reshape(-1) for m in self.models], axis=0)


def ensemble_spec(name: str, lookup) -> ModelSpec:
    """The spec for `name` ("ensemble:a+b"), with `lookup` resolving each member."""
    names = member_names(name)
    if len(names) < 2 or any(is_ensemble(n) for n in names):
        raise ValueError(f"'{name}' is not a valid ensemble: it needs two or more models, e.g. ensemble:tabpfn-v2+tabicl-v2.")
    members = [lookup(n) for n in names]
    tasks = tuple(t for t in ("classification", "regression") if all(m.runs(t) for m in members))
    limits = [m.max_classes for m in members if m.max_classes is not None]
    attributions = [m.attribution for m in members if m.attribution]

    def fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool):
        return _MeanProba([m.fit(X, y, n_classes, serving=serving) for m in members])

    def regress(X: np.ndarray, y: np.ndarray, serving: bool):
        return _MeanPoint([m.fit_regressor(X, y, serving=serving) for m in members])

    def check() -> str | None:
        return next((f"{m.display}: {r}" for m in members if (r := m.unavailable_reason())), None)

    return ModelSpec(
        name=name,
        display="Ensemble (" + " + ".join(m.display for m in members) + ")",
        family="Ensemble",
        license="Each member's own license",
        license_url="",
        source=", ".join(m.source for m in members),
        commercial=all(m.commercial for m in members),
        install="",
        fit_fn=fit if "classification" in tasks else None,  # type: ignore[arg-type]
        check_fn=check,
        attribution="; ".join(dict.fromkeys(attributions)) or None,
        max_classes=min(limits) if limits else None,
        default=False,
        tasks=tasks,
        on_platform=False,
        notes="The average of its members' predictions, with equal weights.",
        regress_fn=regress if "regression" in tasks else None,
        extra={"members": names},
    )
