import numpy as np
import pandas as pd
import pytest

from zerograd.models import Aligned, ModelSpec, register_model, unregister_model


def _logreg_fit(X, y, n_classes, serving):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=500)).fit(X, y)
    return Aligned(model, n_classes)


def _prior_fit(X, y, n_classes, serving):
    class Prior:
        def predict_proba(self, Xq):
            p = np.bincount(y, minlength=n_classes) / len(y)
            return np.tile(p, (len(Xq), 1))

    return Prior()


def _spec(name, fit, commercial=True, **kw):
    return ModelSpec(name=name, display=name, family="test", license="MIT", license_url="https://example.com/license",
                     source="tests", commercial=commercial, install="-", fit_fn=fit, check_fn=lambda: None,
                     default=False, **kw)


@pytest.fixture
def toy_models():
    """Two tiny commercial models and one non-commercial one, registered only for the test."""
    specs = [_spec("test-logreg", _logreg_fit), _spec("test-prior", _prior_fit),
             _spec("test-nc", _logreg_fit, commercial=False)]
    for s in specs:
        register_model(s, replace=True)
    yield [s.name for s in specs]
    for s in specs:
        unregister_model(s.name)


def titanic_like(n=300, seed=0):
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    pclass = rng.choice([1, 2, 3], n)
    age = rng.normal(30, 12, n).round()
    age[rng.random(n) < 0.1] = np.nan
    fare = rng.gamma(2, 15, n).round(2)
    logit = -0.5 + 2.2 * (sex == "female") - 0.8 * (pclass - 2) - 0.02 * (np.nan_to_num(age, nan=30) - 30)
    survived = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"PassengerId": np.arange(1, n + 1), "Pclass": pclass, "Sex": sex, "Age": age,
                         "Fare": fare, "Survived": survived})
