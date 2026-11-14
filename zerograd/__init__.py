"""ZeroGrad: predictions over your tabular data with foundation models. Nothing to train.

    from zerograd import ZeroGrad
    zg = ZeroGrad()                                   # local, on your machine
    cap = zg.create("titanic-survival", examples=train_df, target="Survived")
    cap.evaluation                                    # every model on the same holdout
    cap.predict(new_df)

    zg = ZeroGrad(remote=True)                        # the ZeroGrad platform (ZEROGRAD_API_KEY)

scikit-learn: ZeroGradClassifier / ZeroGradRegressor. Honest evaluation: zg.cross_validate(df, target, groups=, time=).
Then cap.optimize_threshold("cost", cost_fn=20) and cap.importance(test_df).

No target? zg.anomalies(df), zg.impute(df), zg.synthesize(df) and zg.cluster(df) run locally (zerograd.unsupervised).
"""

from ._remote import Capability, Pipeline, RemoteZeroGrad, Run
from ._version import __version__
from .client import ZeroGrad
from .curve import ContextCurve
from .errors import (
    APIError,
    AuthError,
    LimitError,
    NotFoundError,
    RunFailedError,
    UsageWarning,
    ValidationError,
    ZeroGradError,
)
from .estimators import ZeroGradClassifier, ZeroGradRegressor
from .local import LocalCapability, LocalZeroGrad, load
from .local.threshold import ThresholdResult
from .local.validation import CVResult
from .models import LicenseGuardError, NonCommercialModelWarning, list_models, list_tasks, model_info, register_model
from .privacy import SensitiveDataWarning, Vault, drop_sensitive, scan
from .unsupervised import AnomalyResult, ClusterResult, ImputeResult, SyntheticResult

__all__ = [
    "ZeroGrad", "LocalZeroGrad", "RemoteZeroGrad", "LocalCapability", "Capability", "Run", "Pipeline", "load", "ContextCurve",
    "list_models", "list_tasks", "model_info", "register_model", "NonCommercialModelWarning", "LicenseGuardError",
    "ZeroGradError", "AuthError", "NotFoundError", "ValidationError", "LimitError",
    "RunFailedError", "APIError", "UsageWarning",
    "Vault", "scan", "drop_sensitive", "SensitiveDataWarning",
    "AnomalyResult", "ImputeResult", "SyntheticResult", "ClusterResult",
    "ZeroGradClassifier", "ZeroGradRegressor", "CVResult", "ThresholdResult",
    "__version__",
]
