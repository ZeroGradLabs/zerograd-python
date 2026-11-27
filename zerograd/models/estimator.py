"""Bring any estimator into the championship: scikit-learn, XGBoost, LightGBM, CatBoost, or your own class.

    import zerograd
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    zerograd.register_estimator("hgb", HistGradientBoostingClassifier(), regressor=HistGradientBoostingRegressor())
    zg = ZeroGrad()
    cap = zg.create("churn", examples=history, target="churned")    # hgb competes with the foundation models

Anything with `fit(X, y)` plus `predict_proba(X)` (classification) or `predict(X)` (regression) works. Unlike the
foundation models, these estimators *are* trained: `fit` runs on the context rows each time a championship, a
cross-validation fold or a loaded capability needs the model.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from .registry import Aligned, ModelSpec, Point, get_model, register_model

_MISSING_MODES = ("auto", "impute", "pass")


def register_estimator(
    name: str,
    estimator: Any = None,
    *,
    regressor: Any = None,
    display: str | None = None,
    missing: str = "auto",
    default: bool = True,
    license: str = "Your own (registered with register_estimator)",
    license_url: str = "",
    commercial: bool = True,
    replace: bool = False,
) -> ModelSpec:
    """Register an estimator so championships, cross-validation and the scikit-learn wrappers can use it by `name`.

    `estimator` is used for classification when it has `predict_proba`, and for regression when it only has
    `predict`. Pass `regressor=` as well to cover both tasks under one name. Either one can be an instance (it is
    copied before every fit, so it is never trained in place) or a function with no arguments that returns a new one.

    The model sees what the foundation models see: a float matrix with categories as ordinal codes, dates as days
    and missing values as NaN. `missing="auto"` fills NaN with each column's context median for scikit-learn
    estimators that don't accept NaN, and passes NaN through otherwise; `"impute"` always fills, `"pass"` never does.

    `default=True` puts it in every championship that doesn't list `models=`; with False it runs only when named.
    Register it again (in the same process) before `zerograd.load()` opens a capability whose champion it is.
    """
    if missing not in _MISSING_MODES:
        raise ValueError(f"missing is one of {', '.join(_MISSING_MODES)}; got {missing!r}.")
    if estimator is None and regressor is None:
        raise ValueError("Pass an estimator, a regressor=, or both.")
    try:
        existing = get_model(name)
    except ValueError:
        existing = None
    if existing is not None and not replace:
        raise ValueError(f"A model named '{name}' is already registered; pass replace=True to override it.")

    classifier = None
    if estimator is not None:
        probe = _instance(estimator)
        if hasattr(probe, "predict_proba"):
            classifier = estimator
        elif regressor is None and hasattr(probe, "predict"):
            regressor = estimator
        else:
            raise TypeError(f"{type(probe).__name__} has no predict_proba: it can't be a classifier. "
                            "Pass a regressor as regressor= instead.")
    if regressor is not None:
        probe = _instance(regressor)
        if not (hasattr(probe, "fit") and hasattr(probe, "predict")):
            raise TypeError(f"{type(probe).__name__} needs fit(X, y) and predict(X) to be a regressor.")
    if classifier is not None and not hasattr(_instance(classifier), "fit"):
        raise TypeError(f"{type(_instance(classifier)).__name__} needs fit(X, y) and predict_proba(X).")

    tasks = tuple(t for t, est in (("classification", classifier), ("regression", regressor)) if est is not None)
    label = display or _class_name(classifier if classifier is not None else regressor)
    return register_model(ModelSpec(
        name=name,
        display=label,
        family="Your estimators",
        license=license,
        license_url=license_url,
        source="your code",
        commercial=commercial,
        install="",
        fit_fn=_classify(classifier, missing) if classifier is not None else None,  # type: ignore[arg-type]
        check_fn=lambda: None,
        default=default,
        tasks=tasks,
        notes="Trained on the context rows (not a foundation model).",
        regress_fn=_regress(regressor, missing) if regressor is not None else None,
        extra={"estimator": True},
    ), replace=replace)


def _instance(est: Any) -> Any:
    """A fresh, unfitted copy: a factory is called, an instance is cloned (scikit-learn) or deep-copied."""
    if isinstance(est, type) or (callable(est) and not hasattr(est, "fit")):
        return est()
    try:
        from sklearn.base import clone

        return clone(est)
    except Exception:  # not a scikit-learn estimator
        return copy.deepcopy(est)


def _class_name(est: Any) -> str:
    if isinstance(est, type) or (callable(est) and not hasattr(est, "fit")):
        return getattr(est, "__name__", "Estimator")
    return type(est).__name__


def _allows_nan(model: Any) -> bool:
    """scikit-learn says whether an estimator takes NaN; anything else is assumed to handle its own input."""
    try:
        from sklearn.base import BaseEstimator
    except ImportError:  # pragma: no cover - scikit-learn is a dependency
        return True
    if not isinstance(model, BaseEstimator):
        return True
    try:
        from sklearn.utils import get_tags  # scikit-learn 1.6+

        return bool(get_tags(model).input_tags.allow_nan)
    except ImportError:
        return bool(model._get_tags().get("allow_nan", False))
    except Exception:
        return False


class _Imputed:
    """Fills NaN with the context's column medians before the wrapped model sees the rows."""

    def __init__(self, X: np.ndarray):
        with np.errstate(all="ignore"):
            med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0) if len(X) else np.zeros(X.shape[1])
        self.fill = np.where(np.isnan(med), 0.0, med)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.take(self.fill, np.nonzero(bad)[1])
        return X


def _prep(model: Any, X: np.ndarray, missing: str) -> Callable[[np.ndarray], np.ndarray]:
    if missing == "impute" or (missing == "auto" and not _allows_nan(model)):
        return _Imputed(X)
    return lambda Xq: Xq


def _classify(est: Any, missing: str):
    def fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool):
        model = _instance(est)
        prep = _prep(model, X, missing)
        model.fit(prep(X), y)
        seen = getattr(model, "classes_", None)
        seen = [int(c) for c in (np.unique(y) if seen is None else seen)]
        return Aligned(model, n_classes, seen=seen, predict=lambda Xq: model.predict_proba(prep(Xq)))

    return fit


def _regress(est: Any, missing: str):
    def fit(X: np.ndarray, y: np.ndarray, serving: bool):
        model = _instance(est)
        prep = _prep(model, X, missing)
        model.fit(prep(X), y)
        return Point(model, predict=lambda Xq: model.predict(prep(Xq)))

    return fit
