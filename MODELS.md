# Models

Generated from the registry (`zerograd.list_models()`), which is also what the license guards enforce.
ZeroGrad hosts no weights: each model downloads them from its author's repository the first time you use it.

## Commercial use (on by default when installed)

| Model | Name | Tasks | License | Weights | Install | Also with `remote=True` |
|---|---|---|---|---|---|---|
| TabICLv2 | `tabicl-v2` | classification, regression, anomaly detection, imputation, synthetic data | [BSD-3-Clause](https://github.com/soda-inria/tabicl/blob/main/LICENSE) | huggingface.co/jingang/TabICL | `pip install zerograd` | yes |
| TabPFN v2 | `tabpfn-v2` | classification, regression, anomaly detection, imputation, synthetic data | [Prior Labs License v1.0 (Apache 2.0 with attribution)](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/main/LICENSE.txt) | huggingface.co/Prior-Labs/TabPFN-v2-clf, huggingface.co/Prior-Labs/TabPFN-v2-reg | `pip install zerograd` | yes |
| Mitra-v2 | `mitra-v2` | classification, (regression) | [Apache 2.0](https://huggingface.co/autogluon/mitra-classifier-2) | huggingface.co/autogluon/mitra-classifier-2 | `pip install "zerograd[mitra]"` | no |
| TabDPT | `tabdpt` | classification, regression | [Apache 2.0](https://huggingface.co/Layer6/TabDPT) | huggingface.co/Layer6/TabDPT | `pip install "zerograd[tabdpt]"` | no |
| Inpriora | `inpriora` | classification, regression | [Apache 2.0](https://huggingface.co/ai-for-data/inpriora/blob/main/LICENSE) | huggingface.co/ai-for-data/inpriora | `pip install "inpriora @ git+https://github.com/ai-for-data/inpriora.git"` or `zerograd.install_model("inpriora")` | no |
| Mitra | `mitra` | classification, (regression) | [Apache 2.0](https://huggingface.co/autogluon/mitra-classifier) | huggingface.co/autogluon/mitra-classifier | `pip install "zerograd[mitra]"` | no |
| FoMo-0D | `fomo-0d` | anomaly detection | [MIT](https://github.com/A-Chicharito-S/FoMo-0D/blob/main/LICENSE) | huggingface.co/YuchenShen/FoMo-0D | `pip install zerograd` | no |
| ZEUS | `zeus` | clustering | [Apache 2.0](https://github.com/gmum/zeus/blob/main/legal/LICENSE.txt) | the authors' checkpoint, linked from github.com/gmum/zeus | `pip install zerograd` | no |

`mitra` (v1) runs only when named; Mitra-v2 replaces it in the default championship.
(regression) = the model can do it, this version of the SDK does not run it with that model yet. Mitra-v2's
regressor needs a fine-tune on a GPU, so it is left for a later version.
TabPFN v2 asks you to show **Built with PriorLabs-TabPFN** in products built with it (see NOTICE).
TabPFN v2 and TabICLv2 run anomaly detection, imputation and synthetic data through the chain rule in
`zerograd.unsupervised` (any classifier in the registry can). FoMo-0D and ZEUS are zero-shot models for one task
each; their inference code is vendored in `zerograd/models/_vendor` with the authors' licenses.

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
| Regression | yes, locally | Predict a number (amount, price, demand, time) with TabPFN v2, TabICLv2, TabDPT and Inpriora (and TabPFN 2.5+, non-commercial). Not with `remote=True` yet. |
| Anomaly detection | yes, locally | `zg.anomalies(df)`: every row scored, the unusual ones flagged, with the columns why. TabPFN v2, TabICLv2, FoMo-0D; Isolation Forest, LOF and kNN to compare. |
| Imputation | yes, locally | `zg.impute(df)`: a championship on hidden cells (TabPFN v2, TabICLv2 against mean/mode, KNN and iterative imputation), then the champion fills the gaps. |
| Synthetic data | yes, locally | `zg.synthesize(df, n)`: new rows scored by how hard they are to tell apart from yours (detection AUC) and exact copies. TabPFN v2, TabICLv2. |
| Clustering | yes, locally | `zg.cluster(df)`: segments with what sets each apart. ZEUS embeddings with k-means, against k-means and a Gaussian mixture. |
