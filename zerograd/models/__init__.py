"""Every foundation model the local engine knows, with its license.

    from zerograd.models import list_models
    list_models()                                   # name, tasks, license, commercial use, weights, installed
    list_models(task="classification", commercial=True)
    list_tasks()
"""

from . import commercial as _commercial  # noqa: F401  (registers the commercial models)
from . import noncommercial as _noncommercial  # noqa: F401  (registers the non-commercial models, off by default)
from .catalog import ModelCatalog, TaskCatalog, list_models, list_tasks, model_info
from .commercial import LicenseGuardError, assert_tabpfn_v2_checkpoint
from .registry import (
    TASKS,
    Aligned,
    FittedModel,
    FittedRegressor,
    ModelSpec,
    NonCommercialModelWarning,
    Point,
    all_models,
    get_model,
    register_model,
    resolve_models,
    unregister_model,
)

__all__ = [
    "ModelSpec", "FittedModel", "FittedRegressor", "Aligned", "Point", "NonCommercialModelWarning", "LicenseGuardError",
    "register_model", "unregister_model", "get_model", "all_models", "resolve_models", "list_models",
    "list_tasks", "model_info", "ModelCatalog", "TaskCatalog", "TASKS",
    "assert_tabpfn_v2_checkpoint",
]
