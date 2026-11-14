"""Non-commercial models: off by default, for research and evaluation only.

Rules this module follows, whatever you configure:

* None of them runs unless you name it: ``ZeroGrad(noncommercial=["tabpfn-2.5"])``. There is no switch that
  turns them all on, and a ``NonCommercialModelWarning`` with the license link is shown each time.
* Their weights are never hosted or redistributed by ZeroGrad. Each adapter downloads them from the author's
  official repository with your own account: TabPFN 2.5+ asks you to sign in and accept Prior Labs' license
  in the browser; the others use your Hugging Face token (``HF_TOKEN``) and cache.
* They never run on the ZeroGrad platform: ``ZeroGrad(remote=True)`` refuses them.
* You are responsible for complying with each author's license. See TERMS.md.

The TabFM, EXAONE Tabular and LimiX adapters follow each project's published usage and are marked
experimental: they are not part of our tested matrix.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .commercial import TABPFN_LIMITS
from .registry import Aligned, FittedModel, FittedRegressor, ModelSpec, Point, device, missing, register_model

_NC = "Not for production or commercial use."


# ---------------------------------------------------------------- TabPFN 2.5 / 2.6 / 3 / 3.5 (Prior Labs, non-commercial)
def _tabpfn_version_fit(version_name: str):
    def fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        # The tabpfn package downloads these gated weights itself, after you sign in and accept the license.
        model = TabPFNClassifier.create_default_for_version(
            getattr(ModelVersion, version_name), device=device(), random_state=0,
            fit_mode="fit_with_cache" if serving else "fit_preprocessors", **TABPFN_LIMITS,
        )
        model.fit(X, y)
        return Aligned(model, n_classes)

    return fit


def _tabpfn_version_regress(version_name: str):
    def regress(X: np.ndarray, y: np.ndarray, serving: bool) -> FittedRegressor:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        model = TabPFNRegressor.create_default_for_version(
            getattr(ModelVersion, version_name), device=device(), random_state=0,
            fit_mode="fit_with_cache" if serving else "fit_preprocessors", **TABPFN_LIMITS,
        )
        model.fit(X, y)
        return Point(model)

    return regress


def _tabpfn_version_check(version_name: str):
    base = missing("tabpfn")

    def check() -> str | None:
        reason = base()
        if reason:
            return reason
        from tabpfn.constants import ModelVersion

        if not hasattr(ModelVersion, version_name):
            return "your tabpfn package does not include this version; upgrade it with pip install -U tabpfn"
        return None

    return check


for _name, _display, _version, _repo in (
    ("tabpfn-2.5", "TabPFN-2.5", "V2_5", "tabpfn_2_5"),
    ("tabpfn-2.6", "TabPFN-2.6", "V2_6", "tabpfn_2_6"),
    ("tabpfn-3", "TabPFN-3", "V3", "tabpfn_3"),
    ("tabpfn-3.5", "TabPFN-3.5", "V3_5", "tabpfn_3_5"),
):
    register_model(ModelSpec(
        name=_name,
        display=_display,
        family="TabPFN",
        license="Prior Labs non-commercial license",
        license_url=f"https://huggingface.co/Prior-Labs/{_repo}",
        source=f"huggingface.co/Prior-Labs/{_repo} (gated: sign in and accept the license when asked)",
        commercial=False,
        install="pip install zerograd",
        attribution="Built with PriorLabs-TabPFN",
        max_classes=10,
        notes=_NC,
        tasks=("classification", "regression"),
        acceptance="Prior Labs license, accepted in the browser on first download (Prior Labs account)",
        fit_fn=_tabpfn_version_fit(_version),
        regress_fn=_tabpfn_version_regress(_version),
        check_fn=_tabpfn_version_check(_version),
    ))


# ---------------------------------------------------------------- TabFM (Google)
def _tabfm_fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

    clf = TabFMClassifier(model=tabfm_v1_0_0_pytorch.load(model_type="classification"))
    clf.fit(X, y)
    return Aligned(clf, n_classes)


register_model(ModelSpec(
    name="tabfm",
    display="TabFM 1.0 (Google)",
    family="TabFM",
    license="TabFM Non-Commercial License v1.0 (code Apache 2.0)",
    license_url="https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE",
    source="huggingface.co/google/tabfm-1.0.0-pytorch",
    commercial=False,
    install='pip install "tabfm[pytorch]"',
    notes="Experimental adapter. " + _NC,
    tasks=("classification", "regression"),
    acceptance="Read and accept the license on the model page before use",
    fit_fn=_tabfm_fit,
    check_fn=missing("tabfm"),
))


# ---------------------------------------------------------------- EXAONE Tabular (LG AI Research)
def _exaone_fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
    from exaonetabular import EXAONETabularClassifier

    clf = EXAONETabularClassifier.from_pretrained(device=device())
    clf.fit(X, y)
    return Aligned(clf, n_classes)


register_model(ModelSpec(
    name="exaone-tabular",
    display="EXAONE Tabular (LG)",
    family="EXAONE",
    license="EXAONE AI Model License 1.2 - NC (code BSD-3)",
    license_url="https://huggingface.co/LG-AI-Research/EXAONE-Tabular/blob/main/LICENSE",
    source="huggingface.co/LG-AI-Research/EXAONE-Tabular",
    commercial=False,
    install='pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"',
    notes="Experimental adapter. " + _NC,
    tasks=("classification", "regression"),
    acceptance="Read and accept the license on the model page before use",
    fit_fn=_exaone_fit,
    check_fn=missing("exaonetabular"),
))


# ---------------------------------------------------------------- LimiX-2 (Stable AI)
LIMIX_REPO, LIMIX_FILE = "stable-ai/LimiX-2", "LimiX-2.ckpt"


def _limix_config() -> str:
    explicit = os.environ.get("ZEROGRAD_LIMIX_CONFIG")
    if explicit:
        return explicit
    import inference  # the LimiX checkout installs this package

    for root in (Path(inference.__file__).resolve().parents[1], Path(inference.__file__).resolve().parent):
        cfg = root / "config" / "cls_default_noretrieval_v2.json"
        if cfg.exists():
            return str(cfg)
    raise FileNotFoundError("Could not find LimiX's cls_default_noretrieval_v2.json; set ZEROGRAD_LIMIX_CONFIG to its path.")


class _LimiX:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch
        from huggingface_hub import hf_hub_download
        from inference.predictor import LimiXPredictor

        for k, v in (("RANK", "0"), ("WORLD_SIZE", "1"), ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", "29500")):
            os.environ.setdefault(k, v)
        self.X, self.y = X, y
        self.model = LimiXPredictor(device=torch.device(device()), model_path=hf_hub_download(LIMIX_REPO, LIMIX_FILE),
                                    inference_config=_limix_config())

    def predict_proba(self, Xq: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(self.X, self.y, Xq, task_type="Classification"))


def _limix_check() -> str | None:
    import importlib.util

    try:
        found = importlib.util.find_spec("inference.predictor") is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        return "LimiX is not installed (it is installed from its GitHub checkout, not PyPI)"
    return None


def _limix_fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
    m = _LimiX(X, y)
    return Aligned(m, n_classes, seen=[int(c) for c in np.unique(y)])


register_model(ModelSpec(
    name="limix-2",
    display="LimiX-2 (Stable AI)",
    family="LimiX",
    license="StableAI LimiX Non-Commercial License",
    license_url="https://huggingface.co/stable-ai/LimiX-2/blob/main/LICENSE",
    source=f"huggingface.co/{LIMIX_REPO}",
    commercial=False,
    install="git clone https://github.com/limix-ldm-ai/LimiX && pip install -e LimiX  (needs Python 3.12+)",
    notes="Experimental adapter. " + _NC,
    tasks=("classification", "regression", "imputation"),
    acceptance="Read and accept the license on the model page before use",
    fit_fn=_limix_fit,
    check_fn=_limix_check,
))
