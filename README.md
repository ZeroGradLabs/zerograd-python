<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/banner-dark.gif">
    <source media="(prefers-color-scheme: light)" srcset="docs/banner-light.gif">
    <img alt="ZeroGrad: one Python interface to tabular foundation models" src="https://raw.githubusercontent.com/zerogradlabs/zerograd-python/main/docs/banner-light.gif" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://zerograd.dev"><img alt="Website: zerograd.dev" src="https://img.shields.io/badge/website-zerograd.dev-black"></a>
  <a href="https://pypi.org/project/zerograd/"><img alt="PyPI" src="https://img.shields.io/pypi/v/zerograd"></a>
  <a href="https://github.com/zerogradlabs/zerograd-python/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zerogradlabs/zerograd-python/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://colab.research.google.com/github/zerogradlabs/zerograd-python/blob/main/examples/quickstart.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center"><a href="https://zerograd.dev"><b>zerograd.dev</b></a> · <a href="https://zerograd.dev/docs">Docs</a> · <a href="https://zerograd.dev/pricing">Platform</a></p>

## Tabular foundation models

The usual way to predict something from a table (churn, fraud, default, demand) is to train a model for that
problem: pick an algorithm, tune it, validate it, retrain it when the data changes.

Tabular foundation models work differently. They are pretrained once, on millions of synthetic datasets, to learn
*in context*: you give them your labeled rows as context and they predict new rows in a single forward pass, with no
training and no hyperparameters. On small and medium tables they are competitive with tuned gradient boosting.

Several research groups have released them: TabPFN (Prior Labs), TabICL (Inria), Mitra (Amazon), TabDPT (Layer 6)
and others. Each comes with its own package, checkpoints, preprocessing and license.

## What ZeroGrad does

ZeroGrad puts tabular foundation models (TFMs) behind one interface, and keeps the model zoo up to date as new models
come out. The same two calls, `create` and `predict`, work with every model, on your machine or on the ZeroGrad
platform.

It's gradientless tabular ML: no training, no gradient descent, no hyperparameter search. Your rows go in as context
and a pretrained model predicts from them in a single forward pass. If you've used scikit-learn the interface will
feel familiar, and `ZeroGradClassifier` and `ZeroGradRegressor` drop straight into pipelines (their `fit` only stores
your rows as context). From the same `ZeroGrad()` object you get:

* Classification, binary or multiclass (up to 10 classes)
* Regression on numeric targets
* Anomaly detection, with the columns behind each flag
* Imputation of missing values
* Synthetic data shaped like your table
* Clustering

Forecasting, causal modeling, marketing mix modeling and recommendations are coming soon.

On top of that:

* **A championship instead of a guess.** Which model is best depends on the data. ZeroGrad scores every model on the
  same holdout of your rows, adds their average as one more candidate (the ensemble), and keeps the best one as the
  champion, with the comparison in a table.
* **Licenses you can see.** Some models allow commercial use and some do not. `zerograd.list_models()` shows each
  license and where the weights come from; non-commercial models stay off unless you enable them by name.
* **Local or cloud, same code.** Locally, nothing leaves your machine. With `remote=True` the same capability runs on
  the platform, with versions, scheduled championships, monitoring and batch pipelines. The platform is in private
  preview, so keys aren't self-serve yet.

## Quickstart

```bash
pip install zerograd
```

```python
from zerograd import ZeroGrad

zg = ZeroGrad()                                        # local: runs on your machine
cap = zg.create("churn", examples=history, target="churned")
cap.evaluation                                         # every model on the same holdout; the champion is marked
cap.predict(new_customers)                             # prediction, score and probability per class
```

<p align="center"><img src="https://raw.githubusercontent.com/zerogradlabs/zerograd-python/main/docs/demo.gif" alt="A ZeroGrad championship in the terminal" width="760"></p>

Open source under Apache 2.0. Try it without installing anything, in Colab:

| Notebook | What it shows | |
|---|---|---|
| [Six use cases](examples/use-cases.ipynb) | The whole SDK in one run: churn, house prices, cross-validation and thresholds, anomalies, clustering, and outcomes on the platform with a key | [Open in Colab](https://colab.research.google.com/github/zerogradlabs/zerograd-python/blob/main/examples/use-cases.ipynb) |
| [Quickstart](examples/quickstart.ipynb) | Model catalog, a championship on breast cancer data, predictions, saving, and a context curve on Adult | [Open in Colab](https://colab.research.google.com/github/zerogradlabs/zerograd-python/blob/main/examples/quickstart.ipynb) |
| [Titanic](examples/titanic.ipynb) | A championship you check against your own holdout, one-row predictions, how much history you need, multiclass | [Open in Colab](https://colab.research.google.com/github/zerogradlabs/zerograd-python/blob/main/examples/titanic.ipynb) |
| [Evaluate and decide](examples/evaluation.ipynb) | Cross-validation at random, by customer and by date, a threshold by the cost of each mistake, column importance, scikit-learn | [Open in Colab](https://colab.research.google.com/github/zerogradlabs/zerograd-python/blob/main/examples/evaluation.ipynb) |
| [Without a target](examples/no-target.ipynb) | Anomalies with the columns why, imputation, synthetic rows scored by detection AUC, clustering with profiles | [Open in Colab](https://colab.research.google.com/github/zerogradlabs/zerograd-python/blob/main/examples/no-target.ipynb) |

## Install

```bash
pip install zerograd                  # local engine with TabPFN v2 and TabICLv2
pip install "zerograd[all]"           # adds Mitra-v2, TabDPT and Kumo-Tabular
```

Or install one model's own package from Python, from a fixed recipe, into the Python you're running (a notebook
kernel or Colab included):

```python
zerograd.install_model("tabdpt")                          # commercial, optional
zerograd.install_model("limix-2", accept_license=True)    # non-commercial: read its license first (Python 3.12+)
zerograd.install_model("exaone-tabular", dry_run=True)    # only print the commands
```

Installing a non-commercial model doesn't turn it on: it still runs only when you name it in `noncommercial=[...]`.

Python 3.10+. A GPU helps but is not required: on CPU, histories up to a few thousand rows take seconds to
minutes. Weights download from each model's official Hugging Face repository the first time you use it and
are cached there (`HF_HOME`); ZeroGrad hosts none. Importing zerograd or building `ZeroGrad()` downloads nothing.

```python
zerograd.download(["tabpfn-v2"])          # fetch now instead of on first use, e.g. before a demo without wifi
zerograd.cached_weights()                 # what is on disk, how big, and whether ZeroGrad downloaded it
zerograd.remove_weights()                 # delete what ZeroGrad downloaded; the next run fetches it again
zerograd.config(auto_remove_weights=True) # or ZEROGRAD_AUTO_REMOVE_WEIGHTS=1: delete this session's downloads at exit
```

ZeroGrad keeps a list of the files it downloaded, so `remove_weights` never touches weights another project put in
your Hugging Face cache. With auto-remove on, the files go when the Python process ends normally (a script
finishing, a notebook kernel shutting down or restarting), not if it is killed.

## What models are there?

```python
import zerograd

zerograd.list_models()                                  # every model: tasks, license, commercial use, installed
zerograd.list_models(task="classification", commercial=True)
zerograd.list_tasks()                                   # every task and which models do it
zerograd.list_models().to_frame()                       # as a DataFrame
```

In a notebook these render as tables with a badge for commercial and non-commercial use. In this version:

| Model | Name | License | Commercial use | Default |
|---|---|---|---|---|
| TabPFN v2 | `tabpfn-v2` | Prior Labs License (Apache 2.0 + attribution) | yes | yes |
| TabICLv2 | `tabicl-v2` | BSD-3-Clause | yes | yes |
| Mitra-v2 | `mitra-v2` | Apache 2.0 | yes | yes, with `pip install "zerograd[mitra]"` |
| TabDPT | `tabdpt` | Apache 2.0 | yes | yes, with `pip install "zerograd[tabdpt]"` |
| Kumo-Tabular (NVIDIA) | `kumo-tabular` | OpenMDW-1.1 (code Apache 2.0) | yes | yes, with `pip install "zerograd[kumo]"` (Python 3.11+) |
| Kumo-Tabular Medium, Small | `kumo-tabular-medium`, `kumo-tabular-small` | OpenMDW-1.1 (code Apache 2.0) | yes | by name |
| Inpriora (Ant Group) | `inpriora` | Apache 2.0 | yes | yes, with `zerograd.install_model("inpriora")` |
| Mitra | `mitra` | Apache 2.0 | yes | by name |
| FoMo-0D | `fomo-0d` | MIT | yes | anomaly detection |
| ZEUS | `zeus` | Apache 2.0 | yes | clustering |
| MatrixFormer | `matrixformer` | Apache 2.0 | yes | imputation, with `zerograd.install_model("matrixformer")` |

The full list, with sources and the models that are off by default, is in [MODELS.md](MODELS.md).
This version runs **classification** (binary and multiclass, up to 10 classes) and **regression** (locally, with
TabPFN v2, TabICLv2, TabDPT, Kumo-Tabular and Inpriora), plus four tasks with no target: anomaly detection, imputation, synthetic data and
clustering. `zerograd.list_tasks()` shows which models run each task.

## Benchmark

We ran 12 OpenML classification datasets and 6 regression datasets, 3 seeds each. The champion
is picked on the training rows only, and every number below comes from a 20% test split that nothing saw during
selection. The baseline is scikit-learn's HistGradientBoosting with default settings, and regression also gets Ridge
as a linear baseline.

| | champion against HistGradientBoosting (wins / ties / losses) |
|---|---|
| ROC AUC | 11 / 0 / 1 (mean 0.894 against 0.866) |
| Log loss | 12 / 0 / 0 (0.323 against 0.454) |
| Brier score | 12 / 0 / 0 |
| Accuracy | 10 / 1 / 1 |
| Regression RMSE | 6 / 0 / 0 (and 6 / 0 / 0 against Ridge) |

The one loss is churn, by 0.004 of ROC AUC: TabPFN v2 alone would have won it, but the championship picked another
model on the training rows. Per-dataset tables for every model are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md), and
`python benchmarks/run.py` reproduces them.

## What the SDK can do

Everything below is explained, with code, in the [docs](https://zerograd.dev/docs).

* **[Championship](https://zerograd.dev/docs#how-it-works):** every model scored on the same holdout of your rows, with the champion and baselines in one table.
* **[Predictions](https://zerograd.dev/docs#predict):** a DataFrame or a single row in, the prediction, score and probabilities out.
* **[Classification](https://zerograd.dev/docs#classification) and [regression](https://zerograd.dev/docs#regression):** binary and multiclass (up to 10 classes), or numeric targets, picked automatically.
* **[Context size](https://zerograd.dev/docs#context-size):** find how many rows of history you need, trading a little quality for much faster predictions.
* **[Evaluation](https://zerograd.dev/docs#evaluation):** [cross-validation](https://zerograd.dev/docs#cross-validation) at random, by group or by time, [thresholds](https://zerograd.dev/docs#optimize-threshold) chosen by the cost of each mistake, and [column importance](https://zerograd.dev/docs#importance).
* **Classification report:** `cap.classification_report()` gives precision, recall, F1 and support per class, as in scikit-learn, on the holdout or on rows you pass, at the threshold the capability uses.
* **[scikit-learn](https://zerograd.dev/docs#sklearn):** `ZeroGradClassifier` and `ZeroGradRegressor` work in pipelines, grid search and `cross_val_score`.
* **[No target](https://zerograd.dev/docs#no-target):** [anomaly detection](https://zerograd.dev/docs#anomaly-detection) with the columns why, [imputation](https://zerograd.dev/docs#imputation), [synthetic data](https://zerograd.dev/docs#synthetic-data) and [clustering](https://zerograd.dev/docs#clustering).
* **[Privacy](https://zerograd.dev/docs#privacy):** find and drop sensitive columns, or pseudonymize them with a local vault, before anything leaves your machine.
* **New rows, no retraining:** `cap.add_examples(new_rows)` adds labeled rows to the context, and the next prediction already uses them.
* **Save and open:** `cap.save()` writes one file and `zerograd.load()` opens it again.
* **Bring your own model:** `zerograd.register_estimator("hgb", HistGradientBoostingClassifier())` puts any estimator with `fit` and `predict_proba` (or `predict`, for regression) in the championship: scikit-learn, XGBoost, LightGBM, CatBoost or your own class. For a full adapter with license and weights, use `zerograd.register_model`.
* **[The ZeroGrad platform](https://zerograd.dev/docs#platform):** with `remote=True` the same code runs on the platform (private preview), with versions, scheduled championships, monitoring and batch pipelines.

## Use it from an agent (MCP)

ZeroGrad ships an [MCP](https://modelcontextprotocol.io) server, so Claude Code, Cursor, Codex or any MCP client can
create capabilities and predict with plain language. It runs on your machine: nothing is uploaded.

```bash
pip install "zerograd[mcp]"
claude mcp add zerograd -- zerograd-mcp      # or, with no install: claude mcp add zerograd -- uvx --from "zerograd[mcp]" zerograd-mcp
```

Other clients take the same command (`zerograd-mcp`, stdio). Then ask, for example: *"Predict churn in
customers.csv and tell me which columns matter."*

| Tool | What it does |
| --- | --- |
| `inspect_data` | Columns, a likely target and columns that look like personal data, from a CSV or Parquet file. |
| `create_capability` | Runs the championship on the labeled rows and saves the capability by name. |
| `list_capabilities` | The capabilities saved on this machine, with champion and score. |
| `evaluate_capability` | Leaderboard, baselines and the classification report. |
| `predict` | Writes predictions next to the input file and returns a summary; or up to 100 rows inline. |
| `explain` | Column importance on rows with the target. |
| `optimize_threshold` | The score cut of a yes/no capability from the cost of a false alarm and of a miss. |
| `list_models` | Models, licenses and what is installed. |
| `read_docs` | This README and the API reference of the installed version, offline, for anything the tools do not cover. |

The agent passes file paths, not rows, so big tables never go through its context. Capabilities are saved in
`~/.zerograd/capabilities` (`ZEROGRAD_HOME` moves it) and are the same files as `cap.save()`.

If your agent writes code with the SDK instead, [AGENTS.md](AGENTS.md) has what it should know, and the docs are at
[zerograd.dev/llms.txt](https://zerograd.dev/llms.txt) in Markdown.

## Licenses and attribution

The SDK is Apache 2.0. Each model has its own license, shown by `zerograd.list_models()`, and **you are
responsible for complying with it** ([TERMS.md](TERMS.md)). If you build a product with TabPFN v2, its license
asks you to show **"Built with PriorLabs-TabPFN"**; `cap.attribution` tells you when the champion needs it.

Some newer models are released under non-commercial licenses. They are off by default, can be enabled one by
one by name for research and evaluation, and never run on the ZeroGrad platform. See
[MODELS.md](MODELS.md#non-commercial-off-by-default-research-and-evaluation-only) and [TERMS.md](TERMS.md).

## Development

```bash
pip install -e ".[dev]"
pytest                                        # fast tests, no models, no network
ZEROGRAD_REAL_MODELS=1 pytest tests/test_real_models.py   # the real models (downloads weights)
```

## Security

See [SECURITY.md](SECURITY.md). Releases are published from GitHub Actions with PyPI trusted publishing.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The models have their own licenses: [MODELS.md](MODELS.md).
