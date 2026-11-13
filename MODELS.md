# Models

Generated from the registry (`zerograd.list_models()`), which is also what the license guards enforce.
ZeroGrad hosts no weights: each model downloads them from its author's repository the first time you use it.

## Commercial use (on by default when installed)

| Model | Name | Tasks | License | Weights | Install | Also with `remote=True` |
|---|---|---|---|---|---|---|
| TabICLv2 | `tabicl-v2` | classification, regression | [BSD-3-Clause](https://github.com/soda-inria/tabicl/blob/main/LICENSE) | huggingface.co/jingang/TabICL | `pip install zerograd` | yes |
| TabPFN v2 | `tabpfn-v2` | classification, regression | [Prior Labs License v1.0 (Apache 2.0 with attribution)](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt) | huggingface.co/Prior-Labs/TabPFN-v2-clf, huggingface.co/Prior-Labs/TabPFN-v2-reg | `pip install zerograd` | yes |
| Mitra-v2 | `mitra-v2` | classification, (regression) | [Apache 2.0](https://huggingface.co/autogluon/mitra-classifier-2) | huggingface.co/autogluon/mitra-classifier-2 | `pip install "zerograd[mitra]"` | no |
| TabDPT | `tabdpt` | classification, regression | [Apache 2.0](https://huggingface.co/Layer6/TabDPT) | huggingface.co/Layer6/TabDPT | `pip install "zerograd[tabdpt]"` | no |
| Mitra | `mitra` | classification, (regression) | [Apache 2.0](https://huggingface.co/autogluon/mitra-classifier) | huggingface.co/autogluon/mitra-classifier | `pip install "zerograd[mitra]"` | no |

`mitra` (v1) runs only when named; Mitra-v2 replaces it in the default championship.
(regression) = the model can do it, this version of the SDK does not run it with that model yet. Mitra-v2's
regressor needs a fine-tune on a GPU, so it is left for a later version.
TabPFN v2 asks you to show **Built with PriorLabs-TabPFN** in products built with it (see NOTICE).

## Non-commercial (off by default, research and evaluation only)

Each one runs only when you name it (`ZeroGrad(noncommercial=["tabpfn-2.5"])`), warns with its license, downloads with your own account, and never runs with `remote=True`. See [TERMS.md](TERMS.md).

| Model | Name | Tasks | License | Weights | Before the first download | Install |
|---|---|---|---|---|---|---|
| EXAONE Tabular (LG) | `exaone-tabular` | classification, (regression) | [EXAONE AI Model License 1.2 - NC (code BSD-3)](https://huggingface.co/LG-AI-Research/EXAONE-Tabular/blob/main/LICENSE) | huggingface.co/LG-AI-Research/EXAONE-Tabular | Read and accept the license on the model page before use | `pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"` |
| LimiX-2 (Stable AI) | `limix-2` | classification, (regression), (imputation) | [StableAI LimiX Non-Commercial License](https://huggingface.co/stable-ai/LimiX-2/blob/main/LICENSE) | huggingface.co/stable-ai/LimiX-2 | Read and accept the license on the model page before use | `git clone https://github.com/limix-ldm-ai/LimiX && pip install -e LimiX  (needs Python 3.12+)` |
| TabFM 1.0 (Google) | `tabfm` | classification, (regression) | [TabFM Non-Commercial License v1.0 (code Apache 2.0)](https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE) | huggingface.co/google/tabfm-1.0.0-pytorch | Read and accept the license on the model page before use | `pip install "tabfm[pytorch]"` |
| TabPFN-2.5 | `tabpfn-2.5` | classification, regression | [Prior Labs non-commercial license](https://huggingface.co/Prior-Labs/tabpfn_2_5) | huggingface.co/Prior-Labs/tabpfn_2_5 (gated: sign in and accept the license when asked) | Prior Labs license, accepted in the browser on first download (Prior Labs account) | `pip install zerograd` |
| TabPFN-2.6 | `tabpfn-2.6` | classification, regression | [Prior Labs non-commercial license](https://huggingface.co/Prior-Labs/tabpfn_2_6) | huggingface.co/Prior-Labs/tabpfn_2_6 (gated: sign in and accept the license when asked) | Prior Labs license, accepted in the browser on first download (Prior Labs account) | `pip install zerograd` |
| TabPFN-3 | `tabpfn-3` | classification, regression | [Prior Labs non-commercial license](https://huggingface.co/Prior-Labs/tabpfn_3) | huggingface.co/Prior-Labs/tabpfn_3 (gated: sign in and accept the license when asked) | Prior Labs license, accepted in the browser on first download (Prior Labs account) | `pip install zerograd` |
| TabPFN-3.5 | `tabpfn-3.5` | classification, regression | [Prior Labs non-commercial license](https://huggingface.co/Prior-Labs/tabpfn_3_5) | huggingface.co/Prior-Labs/tabpfn_3_5 (gated: sign in and accept the license when asked) | Prior Labs license, accepted in the browser on first download (Prior Labs account) | `pip install zerograd` |

The TabFM, EXAONE Tabular and LimiX-2 adapters follow each project's published usage and are experimental: they are not in the tested matrix.

## Tasks

| Task | In this version | Description |
|---|---|---|
| Classification | yes | Predict a category: binary (churn, fraud, default) or multiclass (up to 10 classes). |
| Regression | yes, locally | Predict a number (amount, price, demand, time) with TabPFN v2, TabICLv2 and TabDPT (and TabPFN 2.5+, non-commercial). Not with `remote=True` yet. |
| Imputation | not yet | Fill in missing values from the rest of the row. Supported by some models; not in the SDK yet. |
