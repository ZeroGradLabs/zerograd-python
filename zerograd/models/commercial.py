"""Models whose weights allow commercial use. On by default when their package is installed.

| model      | license                                   | weights (official repository)        |
|------------|-------------------------------------------|--------------------------------------|
| TabPFN v2  | Prior Labs License (Apache 2.0 + attribution) | huggingface.co/Prior-Labs/TabPFN-v2-clf |
| TabICLv2   | BSD-3-Clause                              | huggingface.co/jingang/TabICL        |
| Mitra-v2   | Apache 2.0                                | huggingface.co/autogluon/mitra-classifier-2 |
| Mitra      | Apache 2.0                                | huggingface.co/autogluon/mitra-classifier |
| TabDPT     | Apache 2.0                                | huggingface.co/Layer6/TabDPT         |
"""

from __future__ import annotations

import os

import numpy as np

from .registry import Aligned, FittedModel, ModelSpec, device, missing, register_model

# ---------------------------------------------------------------- TabPFN v2
# Only the v2 weights allow commercial use. The `tabpfn` package defaults to newer, non-commercial weights,
# so this adapter pins the v2 checkpoint and refuses anything else. TabPFN 2.5+ live in noncommercial.py.
TABPFN_V2_REPO = "Prior-Labs/TabPFN-v2-clf"
TABPFN_V2_FILE = "tabpfn-v2-classifier.ckpt"
_NON_COMMERCIAL_MARKERS = ("v2.5", "v2_5", "v2-5", "v2.6", "v2_6", "v2-6", "v3")


class LicenseGuardError(RuntimeError):
    """A checkpoint was about to load under a model whose license it does not have."""


def assert_tabpfn_v2_checkpoint(path: str) -> None:
    name = os.path.basename(path).lower()
    if not name.startswith("tabpfn-v2-classifier") or any(m in name for m in _NON_COMMERCIAL_MARKERS):
        raise LicenseGuardError(
            f"Refusing to load '{name}' as tabpfn-v2: only TabPFN v2 classifier weights allow commercial use. "
            "Newer TabPFN versions are available as non-commercial models (see zerograd.models)."
        )


def tabpfn_v2_path() -> str:
    """ZEROGRAD_TABPFN_V2_PATH, else the official checkpoint in your Hugging Face cache (downloaded once)."""
    path = os.environ.get("ZEROGRAD_TABPFN_V2_PATH")
    if not path:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=TABPFN_V2_REPO, filename=TABPFN_V2_FILE)
    assert_tabpfn_v2_checkpoint(path)
    return path


def _tabpfn_fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
    # TabPFN refuses more than 1,000 rows on CPU unless told otherwise; slow beats refusing here.
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2, model_path=tabpfn_v2_path(), device=device(), random_state=0,
        fit_mode="fit_with_cache" if serving else "fit_preprocessors",
    )
    model.fit(X, y)
    return Aligned(model, n_classes)


def _tabpfn_weights() -> list[tuple[str, str]]:
    return [] if os.environ.get("ZEROGRAD_TABPFN_V2_PATH") else [(TABPFN_V2_REPO, TABPFN_V2_FILE)]


def _tabicl_weights() -> list[tuple[str, str]]:
    return [] if os.environ.get("ZEROGRAD_TABICL_PATH") else [("jingang/TabICL", TABICL_CHECKPOINT)]


def _mitra_weights(repo: str):
    return lambda: [(repo, "config.json"), (repo, "model.safetensors")]


def _tabdpt_weights() -> list[tuple[str, str]]:
    from tabdpt.estimator import _HF_REPO_ID, _MODEL_NAME

    return [(_HF_REPO_ID, _MODEL_NAME)]


# ---------------------------------------------------------------- TabICL v2
TABICL_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def _tabicl_fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
    from tabicl import TabICLClassifier

    model = TabICLClassifier(
        checkpoint_version=TABICL_CHECKPOINT,
        model_path=os.environ.get("ZEROGRAD_TABICL_PATH"),  # None: the Hugging Face cache
        allow_auto_download=True,
        device=device(),
        random_state=0,
        kv_cache=serving,
    )
    model.fit(X, y)
    return Aligned(model, n_classes)


# ---------------------------------------------------------------- Mitra (v1 and v2, same architecture)
def _mitra_fit(repo: str):
    def fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

        model = MitraClassifier(hf_model=repo, fine_tune=False, device=device(), n_estimators=1, verbose=False)
        model.fit(X, y)
        # Mitra returns one column per class it saw in the context, in sorted order.
        return Aligned(model, n_classes, seen=[int(c) for c in np.unique(y)])

    return fit


# ---------------------------------------------------------------- TabDPT
def _tabdpt_fit(X: np.ndarray, y: np.ndarray, n_classes: int, serving: bool) -> FittedModel:
    from tabdpt import TabDPTClassifier

    model = TabDPTClassifier(device=device(), compile=False, verbose=False)
    model.fit(X, y)

    def predict(Xq: np.ndarray) -> np.ndarray:
        return model.ensemble_predict_proba(Xq, n_ensembles=8, context_size=2048, seed=0)

    return Aligned(model, n_classes, seen=[int(c) for c in np.unique(y)], predict=predict)


TABPFN_V2 = register_model(ModelSpec(
    name="tabpfn-v2",
    display="TabPFN v2",
    family="TabPFN",
    license="Prior Labs License v1.0 (Apache 2.0 with attribution)",
    license_url="https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt",
    source=f"huggingface.co/{TABPFN_V2_REPO}",
    commercial=True,
    install='pip install "zerograd[local]"',
    attribution="Built with PriorLabs-TabPFN",
    max_classes=10,
    tasks=("classification", "regression"),
    on_platform=True,
    fit_fn=_tabpfn_fit,
    weights_fn=_tabpfn_weights,
    check_fn=missing("tabpfn", "huggingface_hub"),
))

TABICL_V2 = register_model(ModelSpec(
    name="tabicl-v2",
    display="TabICLv2",
    family="TabICL",
    license="BSD-3-Clause",
    license_url="https://github.com/soda-inria/tabicl/blob/main/LICENSE",
    source="huggingface.co/jingang/TabICL",
    commercial=True,
    install='pip install "zerograd[local]"',
    tasks=("classification", "regression"),
    on_platform=True,
    fit_fn=_tabicl_fit,
    weights_fn=_tabicl_weights,
    check_fn=missing("tabicl"),
))

MITRA_V2 = register_model(ModelSpec(
    name="mitra-v2",
    display="Mitra-v2",
    family="Mitra",
    license="Apache 2.0",
    license_url="https://huggingface.co/autogluon/mitra-classifier-2",
    source="huggingface.co/autogluon/mitra-classifier-2",
    commercial=True,
    install='pip install "zerograd[mitra]"',
    max_classes=10,
    notes="Zero-shot (no fine-tuning). The published benchmark numbers use fine-tuning with bagging on a GPU.",
    tasks=("classification", "regression"),
    fit_fn=_mitra_fit("autogluon/mitra-classifier-2"),
    weights_fn=_mitra_weights("autogluon/mitra-classifier-2"),
    check_fn=missing("autogluon.tabular"),
))

MITRA_V1 = register_model(ModelSpec(
    name="mitra",
    display="Mitra",
    family="Mitra",
    license="Apache 2.0",
    license_url="https://huggingface.co/autogluon/mitra-classifier",
    source="huggingface.co/autogluon/mitra-classifier",
    commercial=True,
    install='pip install "zerograd[mitra]"',
    max_classes=10,
    default=False,  # superseded by Mitra-v2; run it by name
    tasks=("classification", "regression"),
    fit_fn=_mitra_fit("autogluon/mitra-classifier"),
    weights_fn=_mitra_weights("autogluon/mitra-classifier"),
    check_fn=missing("autogluon.tabular"),
))

TABDPT = register_model(ModelSpec(
    name="tabdpt",
    display="TabDPT",
    family="TabDPT",
    license="Apache 2.0",
    license_url="https://huggingface.co/Layer6/TabDPT",
    source="huggingface.co/Layer6/TabDPT",
    commercial=True,
    install='pip install "zerograd[tabdpt]"',
    tasks=("classification", "regression"),
    fit_fn=_tabdpt_fit,
    weights_fn=_tabdpt_weights,
    check_fn=missing("tabdpt"),
))
