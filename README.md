<h1 align="center">ZeroGrad</h1>

<p align="center"><b>Predictions over your tabular data with open foundation models. Nothing to train.</b></p>

<p align="center">
  <a href="https://pypi.org/project/zerograd/"><img alt="PyPI" src="https://img.shields.io/pypi/v/zerograd"></a>
  <a href="https://github.com/zerogradlab/zerograd-python/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/zerogradlab/zerograd-python/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://colab.research.google.com/github/zerogradlab/zerograd-python/blob/main/examples/quickstart.ipynb"><img alt="Open in Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

<p align="center"><img src="https://raw.githubusercontent.com/zerogradlab/zerograd-python/main/docs/demo.gif" alt="A ZeroGrad championship in the terminal" width="760"></p>

ZeroGrad runs a **championship**: tabular foundation models (TabPFN, TabICL, Mitra, TabDPT) read your labeled
history as context, every one is scored on the same holdout, and the best becomes your champion. Then you predict
with it. No training loop, no hyperparameters, no feature engineering.

```bash
pip install zerograd
```

```python
from zerograd import ZeroGrad

zg = ZeroGrad()                                        # runs on your machine
cap = zg.create("churn", examples=history, target="churned")
cap.evaluation                                         # every model on the same holdout; the champion is marked
cap.predict(new_customers)                             # prediction, score, proba_<class>
```

Open source under Apache 2.0. In local mode no data leaves your machine.

## Install

```bash
pip install zerograd                  # local engine with TabPFN v2 and TabICLv2
pip install "zerograd[all]"           # adds Mitra-v2 and TabDPT
```

Python 3.10+. A GPU helps but is not required: on CPU, histories up to a few thousand rows take seconds to
minutes. Weights download from each model's official Hugging Face repository the first time you use it and
are cached there (`HF_HOME`); ZeroGrad hosts none. Try it without installing anything in
[Colab](https://colab.research.google.com/github/zerogradlab/zerograd-python/blob/main/examples/quickstart.ipynb).

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
| Mitra | `mitra` | Apache 2.0 | yes | by name |

The full list, with sources and the models that are off by default, is in [MODELS.md](MODELS.md).
The task this version runs is **classification** (binary and multiclass, up to 10 classes).

## The championship

```python
cap = zg.create(
    "titanic-survival",
    examples="train.csv",          # a DataFrame or a CSV/Parquet path
    target="Survived",
    drop=["boat", "body"],         # columns only known after the outcome
    metric="roc_auc",              # or pr_auc, log_loss, accuracy, balanced_accuracy, f1, precision, recall, f1_macro
)
cap.evaluation       # one row per model: metrics on the holdout, seconds, champion
cap.baselines        # the same holdout scored without a model (random, majority class)
cap.columns          # which columns were used and why the others were dropped
```

What happens:

* Columns that cannot help are dropped and reported: identifiers, free text, constant and empty columns.
* 20% of the labeled rows are held out (stratified) and every holdout row is scored by every model.
* The context the models read is the rest, or a stratified sample of up to 10,000 rows.
* The champion keeps a context drawn from all labeled rows for predictions.

Pick the competitors with `ZeroGrad(models=["tabpfn-v2", "mitra-v2"])`.

## Predict

```python
preds = cap.predict(new_df)            # DataFrame aligned to new_df's index
cap.predict({"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25})   # one row -> dict
cap.update(threshold=0.3)              # binary: score >= 0.3 predicts the positive class
```

## Save and open

```python
cap.save("titanic.zg")                 # preprocessing, evaluation and context rows; no weights
cap = zerograd.load("titanic.zg")
```

The file holds a sample of your labeled rows as the model's context: treat it like the data itself.

## Licenses and attribution

The SDK is Apache 2.0. Each model has its own license, shown by `zerograd.list_models()`, and **you are
responsible for complying with it** ([TERMS.md](TERMS.md)). If you build a product with TabPFN v2, its license
asks you to show **"Built with PriorLabs-TabPFN"**; `cap.attribution` tells you when the champion needs it.

Some newer models are released under non-commercial licenses. They are off by default, can be enabled one by
one by name for research and evaluation, and never run on the ZeroGrad platform. See
[MODELS.md](MODELS.md#non-commercial-off-by-default-research-and-evaluation-only) and [TERMS.md](TERMS.md).

## Bring your own model

Anything with `fit` and `predict_proba` can compete:

```python
from zerograd.models import Aligned, ModelSpec, register_model

def fit(X, y, n_classes, serving):
    model = MyClassifier().fit(X, y)
    return Aligned(model, n_classes)

register_model(ModelSpec(
    name="my-model", display="My model", family="mine", license="MIT", license_url="https://...",
    source="local", commercial=True, install="-", fit_fn=fit, check_fn=lambda: None,
))
ZeroGrad(models=["tabpfn-v2", "my-model"])
```

## The ZeroGrad platform (`remote=True`)

The same calls run on the ZeroGrad platform with an API key:

```python
zg = ZeroGrad(remote=True)             # reads ZEROGRAD_API_KEY; an API key alone also selects it
cap = zg.create("churn", examples=history, target="churned")
```

The platform adds what a team needs to run predictions in production: versions and promotion, scheduled
championships on fresh history, batch pipelines from your bucket, monitoring, thresholds chosen by cost,
Explain and Ask, and workspaces for teams. `zerograd.privacy` (`scan`, `drop_sensitive`, `Vault`) keeps
personal data on your side before anything is sent. See [zerograd.dev](https://zerograd.dev).

`remote=None` (the default) picks the platform when an API key is given or `ZEROGRAD_API_KEY` is set;
pass `remote=False` to stay local.

## Output

The SDK talks on stderr: a greeting, a spinner per model, a podium at the end. Off a terminal (CI, logs) it prints
plain lines; `progress=False` or `ZEROGRAD_QUIET=1` silences it, and `NO_COLOR` turns colors off. License warnings
are never silenced.

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
