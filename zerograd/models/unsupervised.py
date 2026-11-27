"""Foundation models for tasks with no target: anomaly detection and clustering. Both allow commercial use.

| model    | task              | license    | weights (official repository)                      |
|----------|-------------------|------------|----------------------------------------------------|
| FoMo-0D  | anomaly detection | MIT        | huggingface.co/YuchenShen/FoMo-0D                  |
| ZEUS     | clustering        | Apache 2.0 | the authors' checkpoint (github.com/gmum/zeus)     |
| MatrixFormer | imputation    | Apache 2.0 | huggingface.co/Tabimpute/MatrixFormer              |

FoMo-0D and ZEUS have their model code vendored in `zerograd/models/_vendor` with each license; the weights download from the
authors on first use. TabPFN v2 and TabICLv2 also run these tasks, through the chain rule in zerograd.unsupervised.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

from .registry import ModelSpec, device, missing, register_model

# ---------------------------------------------------------------- FoMo-0D
FOMO_REPO = "YuchenShen/FoMo-0D"
FOMO_FEATURES = 100
FOMO_CONTEXT = 5000  # the context length it was pretrained with (inliers)
FOMO_ARCH = dict(emsize=256, nhid=512, nlayers=4, nhead=4, num_R=500)


def _fomo_model():
    import torch
    from torch import nn

    from ._vendor.fomo0d import encoders
    from ._vendor.fomo0d.transformer import TransformerModel

    class FoMo0D(nn.Module):  # mirrors fomo_hub.FoMo0DHub, so the Hub checkpoint loads as is
        def __init__(self):
            super().__init__()
            self.model = TransformerModel(
                encoder=encoders.get_normalized_uniform_encoder(encoders.Linear)(FOMO_FEATURES, FOMO_ARCH["emsize"]),
                nhead=FOMO_ARCH["nhead"], ninp=FOMO_ARCH["emsize"], nhid=FOMO_ARCH["nhid"],
                nlayers=FOMO_ARCH["nlayers"], dropout=0.0, style_encoder=None,
                y_encoder=encoders.Linear(1, emsize=FOMO_ARCH["emsize"]), input_normalization=False, pos_encoder=None,
                decoder_dict={"standard": (None, 2)}, init_method=None, efficient_eval_masking=True,
                decoder_once_dict={}, num_global_att_tokens=0,
                model_para_dict={"num_R": FOMO_ARCH["num_R"], "last_layer_no_R": True},
            )
            # part of the checkpoint (its training loss), unused at inference
            self.model.criterion = nn.CrossEntropyLoss(weight=torch.ones(2) / 2, reduction="none", ignore_index=-33)

    net = FoMo0D()
    state = _fomo_state_dict()
    if any(k.startswith("model.") for k in state):
        net.load_state_dict(state)
    else:
        net.model.load_state_dict(state)
    return net.eval().to(device()), torch


def _fomo_state_dict() -> dict:
    """ZEROGRAD_FOMO0D_PATH (a .safetensors, or the authors' Lightning best.ckpt), else the Hub checkpoint."""
    import torch

    path = os.environ.get("ZEROGRAD_FOMO0D_PATH")
    if not path:
        from ..weights import hub_file

        path = hub_file(FOMO_REPO, "model.safetensors", model="fomo-0d")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # Lightning saved the inner TransformerModel under "model."; the Hub wrapper nests it under "model." too.
    return {k[len("model."):] if k.startswith("model.") else k: v for k, v in state.items()}


def fomo_features(X_ctx: np.ndarray, X_query: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The authors' inference transform: impute, quantile-normalize, scale, pad (or subsample) to 100 features."""
    from sklearn.preprocessing import QuantileTransformer

    rng = np.random.default_rng(seed)
    both = np.vstack([X_ctx, X_query]).astype(np.float64)
    med = np.nanmedian(np.where(np.isnan(X_ctx), np.nan, X_ctx), axis=0) if len(X_ctx) else np.zeros(both.shape[1])
    med = np.where(np.isnan(med), 0.0, med)
    both = np.where(np.isnan(both), med, both)
    d = both.shape[1]
    if d > FOMO_FEATURES:
        both = both[:, np.sort(rng.choice(d, FOMO_FEATURES, replace=False))]
        d = FOMO_FEATURES
    n_ctx = len(X_ctx)
    mean, std = both.mean(axis=0), both.std(axis=0) + 1e-6
    both = (both - mean) / std
    for j in range(d):
        col = both[:, j]
        if np.unique(col[:n_ctx]).size > 1:
            qt = QuantileTransformer(output_distribution="normal", n_quantiles=min(1000, n_ctx), random_state=seed)
            both[:, j] = qt.fit(col[:n_ctx, None]).transform(col[:, None])[:, 0]
    both = both / (d / FOMO_FEATURES)
    if d < FOMO_FEATURES:
        both = np.hstack([both, np.zeros((len(both), FOMO_FEATURES - d))])
    return both[:n_ctx].astype(np.float32), both[n_ctx:].astype(np.float32)


def _fomo_scores(X_ctx: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """Probability that each query row is an outlier, given the context rows as (mostly) normal data."""
    net, torch = _fomo_model()
    rng = np.random.default_rng(0)
    if len(X_ctx) > FOMO_CONTEXT - 1:
        X_ctx = X_ctx[rng.choice(len(X_ctx), FOMO_CONTEXT - 1, replace=False)]
    ctx, query = fomo_features(X_ctx, X_query)
    dev = next(net.parameters()).device
    train = torch.from_numpy(ctx).to(dev).unsqueeze(1)
    out = []
    with torch.no_grad():
        # Each query row attends to the context only, so a block of query rows scores like one row at a time.
        for start in range(0, len(query), 512):
            q = torch.from_numpy(query[start:start + 512]).to(dev).unsqueeze(1)
            logits = net.model(train, None, q).squeeze(1)
            out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


FOMO_0D = register_model(ModelSpec(
    name="fomo-0d",
    display="FoMo-0D",
    family="FoMo",
    license="MIT",
    license_url="https://github.com/A-Chicharito-S/FoMo-0D/blob/main/LICENSE",
    source=f"huggingface.co/{FOMO_REPO}",
    commercial=True,
    install="pip install zerograd",
    default=False,  # not a classifier: never part of a supervised championship
    notes="Zero-shot outlier detection (Shen, Wen, Akoglu; CMU, ICML 2025). Up to 100 features; more are subsampled.",
    tasks=("anomaly_detection",),
    fit_fn=None,  # type: ignore[arg-type]
    anomaly_fn=_fomo_scores,
    weights_fn=lambda: [(FOMO_REPO, "model.safetensors")],
    check_fn=missing("torch", "huggingface_hub", "safetensors"),
))


# ---------------------------------------------------------------- ZEUS
ZEUS_DRIVE_ID = "1D7uikacymUnmmMxjUjBuCNIomqhBWS67"  # the checkpoint the authors link from github.com/gmum/zeus
ZEUS_SHA256 = "eb60086459de338b1795ce2849e02e0d7e59cd95af43d01534dd84083f2b749b"  # the downloaded file must match this exactly, or it is not loaded
ZEUS_DIM = 30
ZEUS_ARCH = dict(embed_dim=512, n_head=4, hid_dim=1024, n_layers=12, num_gaussians=10)
ZEUS_ROWS = 2000  # rows embedded together; more rows are assigned to the nearest cluster


def zeus_path() -> str:
    """ZEROGRAD_ZEUS_PATH, else the authors' checkpoint, downloaded once to ~/.cache/zerograd."""
    path = os.environ.get("ZEROGRAD_ZEUS_PATH")
    if path:
        return path
    dest = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "zerograd" / "zeus.pt"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.usercontent.google.com/download?id={ZEUS_DRIVE_ID}&export=download&confirm=t"
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed https URL)
        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("Could not download the ZEUS checkpoint from Google Drive. Download it from "
                               "github.com/gmum/zeus and set ZEROGRAD_ZEUS_PATH.")
        tmp.rename(dest)
    if _sha256(dest) != ZEUS_SHA256:
        dest.unlink(missing_ok=True)
        raise RuntimeError("The downloaded ZEUS checkpoint does not match the expected SHA-256, so it was not loaded. "
                           "Download it from github.com/gmum/zeus and set ZEROGRAD_ZEUS_PATH if you trust it.")
    from ..weights import adopt

    adopt("zeus", dest)  # ZeroGrad's own cache folder, so the file is on its list and remove_weights can delete it
    return str(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _zeus_model():
    import torch

    from ._vendor.zeus import encoders
    from ._vendor.zeus.zeus import ZeusTransformerModel

    a = ZEUS_ARCH
    net = ZeusTransformerModel(
        encoders.Linear(ZEUS_DIM, a["embed_dim"], replace_nan_by_zero=True), a["num_gaussians"], a["embed_dim"],
        a["n_head"], a["hid_dim"], a["n_layers"], 0.0, efficient_eval_masking=True, n_clusters=a["num_gaussians"],
    )
    ckpt = torch.load(zeus_path(), map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    result = net.load_state_dict(state, strict=False)
    # The authors load with strict=False. The checkpoint has no last decoder layer, which embedding never uses
    # (it reads the encoder output), so only decoder keys may be missing.
    missing_keys = [k for k in result.missing_keys if not k.startswith("decoder.")]
    if missing_keys:
        raise RuntimeError(f"The ZEUS checkpoint does not match the model: missing {missing_keys[:5]}")
    return net.eval().to(device()), torch


def zeus_features(X: np.ndarray) -> np.ndarray:
    """The authors' preprocessing: impute, standardize, PCA to 30 dimensions when wider, scale to [-1, 1], pad."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    X = np.asarray(X, dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), np.where(np.isnan(med), 0.0, med), X)
    X = StandardScaler().fit_transform(X)
    if X.shape[1] > ZEUS_DIM:
        X = PCA(n_components=ZEUS_DIM, random_state=0).fit_transform(X)
    X = MinMaxScaler((-1, 1)).fit_transform(X)
    if X.shape[1] < ZEUS_DIM:
        X = np.hstack([X, np.zeros((len(X), ZEUS_DIM - X.shape[1]))])
    return X.astype(np.float32)


def _zeus_embed(X: np.ndarray) -> np.ndarray:
    net, torch = _zeus_model()
    feats = zeus_features(X)
    with torch.no_grad():
        emb = net.predict_embedding(torch.from_numpy(feats).to(next(net.parameters()).device))
    return emb.cpu().numpy().astype(np.float64)


ZEUS = register_model(ModelSpec(
    name="zeus",
    display="ZEUS",
    family="ZEUS",
    license="Apache 2.0",
    license_url="https://github.com/gmum/zeus/blob/main/legal/LICENSE.txt",
    source="github.com/gmum/zeus (checkpoint on the authors' Google Drive)",
    commercial=True,
    install="pip install zerograd",
    default=False,
    notes=("Zero-shot embeddings for clustering (Marszałek et al., Jagiellonian University, 2025), "
           f"clustered with k-means. Embeds up to {ZEUS_ROWS:,} rows at once."),
    tasks=("clustering",),
    fit_fn=None,  # type: ignore[arg-type]
    embed_fn=_zeus_embed,
    check_fn=missing("torch"),
    extra={"max_rows": ZEUS_ROWS, "fetch": zeus_path},  # fetch: weights outside Hugging Face, for zerograd.download
))


# ---------------------------------------------------------------- MatrixFormer
# Matrix completion (Saha, Feitelberg, Choi, Dwivedi, Agarwal; arXiv 2610.06751). Its own runtime (`matrixformer`,
# installed from the authors' GitHub) downloads the checkpoint from Tabimpute/MatrixFormer at a pinned release and
# checks its SHA-256. Numbers only: categories go in as codes and come back as the nearest code in the column.
MATRIXFORMER_REPO = "Tabimpute/MatrixFormer"
MATRIXFORMER_GIT = "git+https://github.com/jacobf18/MatrixFormer.git"
_matrixformer_cache: dict = {}


def _matrixformer_impute(X: np.ndarray) -> np.ndarray:
    dev = device()
    model = _matrixformer_cache.get(dev)
    if model is None:
        from matrixformer import MatrixFormer

        # bfloat16 needs a recent NVIDIA GPU; float32 works everywhere it runs (CPU and CUDA, not MPS).
        model = MatrixFormer.from_pretrained(device=dev, precision="float32")
        _matrixformer_cache.clear()
        _matrixformer_cache[dev] = model
    return np.asarray(model.impute(np.asarray(X, dtype=np.float64)), dtype=np.float64)


MATRIXFORMER = register_model(ModelSpec(
    name="matrixformer",
    display="MatrixFormer",
    family="TabImpute",
    license="Apache 2.0",
    license_url="https://huggingface.co/Tabimpute/MatrixFormer",
    source=f"huggingface.co/{MATRIXFORMER_REPO}",
    commercial=True,
    install=f'pip install "matrixformer @ {MATRIXFORMER_GIT}"',
    default=False,  # not a classifier: never part of a supervised championship
    notes=("Zero-shot matrix completion (TabImpute authors, 2026). Up to 2,000 rows by 1,000 columns per window; "
           "larger tables are filled in overlapping windows."),
    tasks=("imputation",),
    fit_fn=None,  # type: ignore[arg-type]
    impute_fn=_matrixformer_impute,
    check_fn=missing("matrixformer"),
))
