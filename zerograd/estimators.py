"""scikit-learn estimators around the foundation models: drop them into a Pipeline, GridSearchCV or cross_val_score.

    from zerograd import ZeroGradClassifier
    from sklearn.model_selection import cross_val_score

    clf = ZeroGradClassifier(model="tabicl-v2")
    cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

`fit` keeps a stratified sample of the rows as the model's context (nothing is trained); `predict_proba` and
`predict` run the model with that context. Columns go through the same preprocessing as `zg.create`: categories,
booleans and dates are encoded, missing values stay missing, and ids, free text and constant columns are left out.
No championship here: you pick the model, and scikit-learn does the comparing.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .errors import ValidationError
from .local import championship as champ
from .local.preprocess import Preprocessor
from .models import ModelSpec, resolve_models

_TARGET = "__zerograd_target__"


class _ZeroGradEstimator(BaseEstimator):
    _task = "classification"

    def __init__(self, model: str | None = None, *, max_context_rows: int = 10_000,
                 noncommercial: Iterable[str] = (), random_state: int = 0):
        self.model = model
        self.max_context_rows = max_context_rows
        self.noncommercial = noncommercial
        self.random_state = random_state

    # ------------------------------------------------------------------ helpers
    def _spec(self) -> ModelSpec:
        nc = list(self.noncommercial or ())
        specs = resolve_models([self.model] if self.model else None, nc)
        fits = [s for s in specs if s.runs(self._task)]
        if not fits:
            which = f"'{self.model}' does not run" if self.model else "No installed model runs"
            raise ValidationError(f"{which} {self._task} in this version of the SDK.", code="unsupported_task")
        return fits[0]

    @staticmethod
    def _frame(X: Any, columns: list[str] | None = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValidationError("X must be 2-dimensional (rows by columns).", code="invalid_input")
        names = columns if columns is not None and len(columns) == arr.shape[1] else [f"x{j}" for j in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=names)

    def _fit(self, X: Any, y: Any, task: str) -> None:
        df = self._frame(X)
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray([str(c) for c in X.columns], dtype=object)
        self.n_features_in_ = df.shape[1]
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(df):
            raise ValidationError("y must be one label per row of X.", code="invalid_input")
        if _TARGET in df.columns:
            raise ValidationError(f"X has a column named {_TARGET}; rename it.", code="invalid_input")
        spec = self._spec()
        df = df.assign(**{_TARGET: y})
        pre, kept = Preprocessor.fit(df, _TARGET, task=task, min_rows=2, max_columns=10_000,
                                     max_classes=spec.max_classes or 10)
        Xn, yn = pre.transform(kept), pre.encode_target(kept[_TARGET])
        keep = champ.stratified_sample(champ.strata(yn, pre.task), int(self.max_context_rows), seed=self.random_state)
        self.spec_ = spec
        self.preprocessor_ = pre
        self.context_rows_ = int(len(keep))
        if pre.regression:
            self.model_ = spec.fit_regressor(Xn[keep], yn[keep], serving=True)
        else:
            self.model_ = spec.fit(Xn[keep], yn[keep], len(pre.classes), serving=True)

    def _transform(self, X: Any) -> np.ndarray:
        check_is_fitted(self, "model_")
        names = list(getattr(self, "feature_names_in_", [])) or None
        df = self._frame(X, names)
        if df.shape[1] != self.n_features_in_:
            raise ValidationError(f"X has {df.shape[1]} columns; this estimator was fitted with {self.n_features_in_}.",
                                  code="invalid_input")
        return self.preprocessor_.transform(df)

    @property
    def attribution(self) -> str | None:
        """The attribution the model's license asks you to show, if any (e.g. Built with PriorLabs-TabPFN)."""
        spec = getattr(self, "spec_", None) or self._spec()
        return spec.attribution

    def __sklearn_tags__(self):  # the models read missing values as they are
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class ZeroGradClassifier(ClassifierMixin, _ZeroGradEstimator):
    """A tabular foundation model as a scikit-learn classifier (binary or up to 10 classes).

    `model` is a model name such as "tabpfn-v2" or "tabicl-v2" (default: the first installed commercial model).
    `max_context_rows` caps the rows kept as context (a stratified sample). `noncommercial` names
    non-commercial models to allow, for research and evaluation only.
    """

    _task = "classification"

    def fit(self, X: Any, y: Any) -> "ZeroGradClassifier":
        self._fit(X, y, "classification")
        self.classes_ = np.asarray(self.preprocessor_.classes)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        Xn = self._transform(X)
        if not len(Xn):
            return np.zeros((0, len(self.classes_)))
        proba = champ.predict_in_chunks(self.model_, Xn, 10_000)
        proba = np.clip(proba, 0, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class ZeroGradRegressor(RegressorMixin, _ZeroGradEstimator):
    """A tabular foundation model as a scikit-learn regressor. Same parameters as ZeroGradClassifier."""

    _task = "regression"

    def fit(self, X: Any, y: Any) -> "ZeroGradRegressor":
        self._fit(X, y, "regression")
        return self

    def predict(self, X: Any) -> np.ndarray:
        Xn = self._transform(X)
        if not len(Xn):
            return np.zeros(0)
        return champ.predict_in_chunks(self.model_, Xn, 10_000, champ.REGRESSION)


__all__ = ["ZeroGradClassifier", "ZeroGradRegressor"]
