# AGENTS.md

Notes for coding agents (Claude Code, Codex, Cursor and friends), in two parts: using the `zerograd` package in
someone's project, and working on this repo. Humans are welcome too. The full docs are at https://zerograd.dev/docs,
and as Markdown for agents at https://zerograd.dev/llms.txt (index) and https://zerograd.dev/llms-full.txt (all of it).

## Using zerograd in a project

ZeroGrad predicts from tables with pretrained tabular foundation models (TabPFN v2, TabICLv2, Mitra-v2, TabDPT).
Nothing gets trained: your labeled rows are the model's context, every installed model is scored on the same
holdout, and the best one (the champion) answers. It runs on the user's machine with no account and no API key.

```bash
pip install zerograd              # TabPFN v2 and TabICLv2, Python 3.10+
pip install "zerograd[all]"       # adds Mitra-v2, TabDPT and the MCP server
```

```python
import zerograd
from zerograd import ZeroGrad

zg = ZeroGrad()                                       # local
cap = zg.create("churn", examples=df, target="churned")   # a DataFrame or a CSV/Parquet path
cap.evaluation                                        # every model on the same holdout, champion marked
preds = cap.predict(new_rows)                         # DataFrame aligned to new_rows' index
cap.save("churn.zg"); cap = zerograd.load("churn.zg")
```

Things that are easy to get wrong:

- The leaderboard has an `Ensemble (...)` row: the equal-weight average of every model that finished. It can be the
  champion, and then every prediction runs all its members. `create(..., ensemble=False)` leaves it out.
- When new labeled rows arrive, `cap.add_examples(rows)` puts them in the context with no retraining. It doesn't
  re-run the championship; call `zg.create` again for that.
- The task comes from the target column. Categories, booleans and whole numbers with up to 10 distinct values are a
  classification; other numbers are a regression. Pass `task="classification"` or `task="regression"` to force it.
- Drop columns that are only known after the outcome (a cancellation reason, a chargeback date) with `drop=[...]`.
  They make the holdout look great and production fail. Identifiers, free text and constant columns are dropped
  for you, and `cap.columns` lists what was used and what was dropped, with the reason.
- Classification predictions have `prediction`, `score` (binary only) and `proba_<class>`. Regression only has
  `prediction`, and passing a threshold to it is an error.
- If rows from the same customer or from later dates exist, check with
  `zg.cross_validate(df, target, groups="customer_id")` or `time="date"` before trusting a random holdout.
- For a yes/no decision, pick the cut from costs: `cap.optimize_threshold("cost", cost_fp=1, cost_fn=20)`.
- `cap.importance(rows_with_target)` measures which columns matter. Don't read it off the model.
- Without a target: `zg.anomalies(df)`, `zg.impute(df)`, `zg.synthesize(df)`, `zg.cluster(df)`.
- To compare against a model the user already has (gradient boosting, a logistic regression, their own class),
  register it with `zerograd.register_estimator("hgb", HistGradientBoostingClassifier())` and it joins the
  championship. It needs `fit` and `predict_proba` (or `predict` for regression). Unlike the foundation models it
  is trained on the context, and it must be registered again before `zerograd.load()` opens a capability it won.
- In scikit-learn code, `ZeroGradClassifier` and `ZeroGradRegressor` work like any estimator. Their `fit` only
  stores the rows as context.
- The first run of each model downloads its weights from the authors' Hugging Face repository and caches them. A GPU
  helps but isn't required. Bigger contexts cost more compute per prediction; `zg.context_curve(...)` finds a
  smaller one that scores about the same.

Licenses: only models that allow commercial use run by default. Non-commercial ones (like TabPFN 2.5) stay off
unless the user names them with `ZeroGrad(noncommercial=[...])`, so don't turn them on yourself for a commercial
project. If the champion is TabPFN v2, the product has to show "Built with PriorLabs-TabPFN" (`cap.attribution`
tells you). `zerograd.list_models()` has every model with its license; [MODELS.md](MODELS.md) has the details.

The platform: `ZeroGrad(remote=True)` with `ZEROGRAD_API_KEY` set runs the same code on https://api.zerograd.dev,
with versions, pipelines and outcomes. It's in private preview, so keys aren't self-serve yet. Unless the user
already has a key, stay local. Note that `ZeroGrad()` switches to the platform on its own when `ZEROGRAD_API_KEY` is
in the environment; pass `remote=False` to stay local.

Errors are typed (`ValidationError`, `NotFoundError`, `LimitError`, `AuthError`, `RunFailedError`, all
`ZeroGradError`), and their message says what to change, for example which columns are missing.

### MCP

If the user wants ZeroGrad as tools rather than code:

```bash
pip install "zerograd[mcp]"
claude mcp add zerograd -- zerograd-mcp
```

Any MCP client works with the same command over stdio. The tools pass file paths, not rows, and save capabilities
in `~/.zerograd/capabilities` (`ZEROGRAD_HOME` moves it). Those are regular `.zg` files that `zerograd.load()` opens.

## Working on this repo

```bash
uv venv && uv pip install -e ".[dev]"     # or pip install -e ".[dev]"
ruff check .
pytest -q                                  # fast: tiny test models, no weights, no network
ZEROGRAD_REAL_MODELS=1 pytest tests/test_real_models.py -s   # the real models; slow, downloads weights
```

CI runs `ruff check .` and `pytest -q` on Python 3.10 to 3.13, plus `uv build` and `twine check`. Both have to pass.

### Map

| Path | What |
| --- | --- |
| `zerograd/client.py` | `ZeroGrad()`: picks local or remote |
| `zerograd/local/` | Local engine: preprocessing, championship, validation, thresholds, reports |
| `zerograd/_remote.py`, `zerograd/_http.py` | Platform client: capabilities, runs, uploads, pipelines, retries |
| `zerograd/models/` | Model registry and adapters. `commercial.py` and `noncommercial.py` hold the specs, `_vendor/` is third-party code kept as its authors wrote it (ruff skips it) |
| `zerograd/unsupervised/` | Anomalies, imputation, synthetic data, clustering |
| `zerograd/estimators.py` | scikit-learn estimators |
| `zerograd/privacy.py` | Personal-data scan and `Vault` |
| `zerograd/mcp_server.py` | The `zerograd-mcp` server. `read_docs` serves README.md and MODELS.md from the installed wheel |
| `zerograd/weights.py`, `zerograd/install.py` | Weight downloads and cache, fixed install recipes |
| `tests/` | `conftest.py` registers toy models (`toy_models` fixture) so tests never need real weights |
| `examples/` | Notebooks; commit them without outputs (`scripts/clean_notebook_outputs.py`) |

### Conventions

- Line length 120, ruff rules E, F, I, B. Type hints with `from __future__ import annotations`.
- Tests use the `toy_models` fixture and unset `ZEROGRAD_API_KEY`. New behavior gets a fast test; anything that
  needs real weights goes in `tests/test_real_models.py`, behind `ZEROGRAD_REAL_MODELS=1`.
- A new model needs a `ModelSpec` with its license, license URL, weights source and whether it allows commercial
  use. The license guards and `list_models()` read the same registry, so keep it accurate, and add it to MODELS.md.
  Never make a non-commercial model run by default or on the platform.
- Error messages say what to change, in a full sentence, and name the columns, values or limits involved.
- User-facing text (README, docstrings, errors) is plain English, second person, no speed claims and no
  comparisons that knock other tools.
- The platform API isn't public yet. Remote features keep working with `remote=True`, but docs lead with local use.

### Workflow

GitHub flow ([CONTRIBUTING.md](CONTRIBUTING.md)): branch off `main` with a `feature/`, `fix/`, `docs/`, `chore/` or
`hotfix/` prefix, one change per PR, squash merge. Releases are `v*` tags on `main` ([RELEASING.md](RELEASING.md));
don't bump the version or push tags unless you're asked to. Report security issues as described in
[SECURITY.md](SECURITY.md), never in a public issue.
