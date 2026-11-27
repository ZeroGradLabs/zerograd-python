# Changelog

## Unreleased

* **Technical ties in the podium.** A paired bootstrap of the holdout gives every model a 90% interval of the
  selection metric (`ci_low`, `ci_high` in `cap.evaluation`) and marks with `tied` the ones the holdout can't tell
  apart from the champion. The podium shows them with "≈ tied" and a note under the champion line, so a gap of
  0.002 AUC on 300 rows doesn't read as a win.

## 0.2.5 (2026-11-27)

* **TabDPT 1.3.** The `tabdpt` extra and `install_model("tabdpt")` now require tabdpt 1.3, which loads the newer
  `tabdpt1_3` weights.
* **Kumo-Tabular (NVIDIA).** `kumo-tabular` joins the default championship for classification and regression when
  installed with `pip install "zerograd[kumo]"` (Python 3.11+); `kumo-tabular-medium` and `kumo-tabular-small` are
  lighter checkpoints you run by name. Weights under OpenMDW-1.1, code Apache 2.0.
* **MatrixFormer for imputation.** `zerograd.install_model("matrixformer")` adds the TabImpute authors' zero-shot
  matrix-completion model (Apache 2.0) to the `zg.impute` championship. Models can now fill a table themselves
  (`ModelSpec.impute_fn`) instead of going through the chain rule.
* **Inpriora (Ant Group).** `zerograd.install_model("inpriora")` adds Inpriora (code and weights Apache 2.0) to the
  default championship for classification and regression.
* **Your own estimators in the championship.** `zerograd.register_estimator(name, estimator, regressor=None)` takes
  anything with `fit(X, y)` and `predict_proba(X)` (classification) or `predict(X)` (regression), from
  scikit-learn, XGBoost, LightGBM, CatBoost or your own code, and runs it next to the foundation models on the same
  holdout. It's copied before every fit, NaN is filled with context medians for scikit-learn estimators that don't
  accept it, and it works with `cross_validate`, `save`/`load` and `ZeroGradClassifier(model=name)`.
* **Dates and numbers that arrive as text are kept.** A date column read from a CSV ("2024-01-31") used to be dropped
  as free text; it is now read as a date. Numbers in text form with many distinct values ("12.50") are now read as
  numbers instead of being dropped.

## 0.2.4 (2026-11-15)

* **MCP server for agents.** `pip install "zerograd[mcp]"` adds `zerograd-mcp`, a local stdio server for Claude Code,
  Cursor, Codex and other MCP clients, with tools to inspect a table, create, list and evaluate capabilities,
  predict (to a file, or a few rows inline), explain, pick a threshold, list models and read the docs offline.
  Capabilities are saved in `~/.zerograd/capabilities`.

## 0.2.3 (2026-11-14)

* **Install a model from Python.** `zerograd.install_model(name)` runs the model's fixed recipe (pip, pip from git,
  or git clone and pip install -e) with this Python's pip, so it works in notebooks and Colab. Non-commercial models
  need `accept_license=True` and stay off until named in `noncommercial=[...]`; `dry_run=True` shows the commands;
  LimiX-2 checks for Python 3.12+.
* **Classification report.** `cap.classification_report()` (precision, recall, F1, support per class, accuracy, macro
  and weighted averages) on the championship holdout or on rows you pass, at the capability's threshold.
* **Context curves.** `curve.plot()` draws the score and the time per row as two charts side by side instead of one
  with two y axes. A smaller context is only recommended if it also predicted at least 10% faster than the
  best-scoring one in that run; before, a noisy timing could recommend a smaller context that was slower, with a
  summary saying it was faster.
* **Quieter downloads.** Hugging Face's "unauthenticated requests, set a HF_TOKEN" notice is hidden while ZeroGrad
  downloads public weights (`ZEROGRAD_SHOW_HF_WARNINGS=1` shows it).
* **Executed notebooks.** A workflow runs the notebooks in `examples/` on CPU (on demand and on each release) and
  opens a pull request with their outputs, so the results show on GitHub.

## 0.2.2 (2026-11-14)

* **TabPFN v2 on CPU with more than 1,000 rows.** It failed with "Running on CPU with more than 1000 samples is not
  allowed": the adapter set `TABPFN_ALLOW_CPU_LARGE_DATASET`, but tabpfn reads it only when first imported, and
  listing the models already imports it. It now passes `ignore_pretraining_limits=True` (the SDK keeps its own
  context and column limits).
* **Weights on your disk.** `zerograd.download(...)` fetches weights on purpose, `zerograd.cached_weights()` lists
  them, `zerograd.remove_weights(...)` deletes the ones ZeroGrad downloaded (never files other projects put in the
  Hugging Face cache), and `zerograd.config(auto_remove_weights=True)` or `ZEROGRAD_AUTO_REMOVE_WEIGHTS=1` deletes
  each session's downloads when the process ends.
* **Greeting.** It says "installed · weights download on first use" until the weights are on disk, and "ready" after.
* **Errors you can read.** A model that did not finish shows its whole first line in the podium and in
  `cap.evaluation`; `cap.errors` has the full message.
* `examples/use-cases.ipynb` walks the SDK in six use cases on public data, in Colab: churn, house prices,
  evaluation and thresholds, anomalies, clustering, and outcomes on the platform with an API key, with plots
  (ROC, precision-recall, calibration, the podium, the context curve, cost by threshold, importance, segments).

## 0.2.1 (2026-11-14)

* `zg.cluster` checks the SHA-256 of the ZEUS checkpoint it downloads and loads it with `weights_only=True`. A file that does not match is deleted and never loaded.

## 0.2.0 (2026-11-14)

First open source release, Apache 2.0. `pip install zerograd` includes the local engine with TabPFN v2 and TabICLv2.

* **Local engine.** `ZeroGrad()` without an API key runs the championship and predictions on your machine:
  every model scored on the same holdout, champion by ROC AUC (or the metric you pick), predictions with the
  champion. Save a capability to one file and open it again with `zerograd.load`.
* **Models.** Commercial use, on by default when installed: TabPFN v2, TabICLv2, Mitra-v2, TabDPT (and Mitra v1
  by name). Non-commercial, off by default and only by name: TabPFN 2.5 / 2.6 / 3 / 3.5, TabFM, EXAONE Tabular,
  LimiX-2. Weights download from each author's repository; nothing is hosted by ZeroGrad.
* **Regression.** A numeric target runs a regression championship (TabPFN v2, TabICLv2, TabDPT; TabPFN 2.5+ by
  name, non-commercial): RMSE (default), MAE and R² on a holdout stratified on deciles of the target, average and
  median baselines, `predict` with one `prediction` column, save and load, context curves with a relative tolerance.
  `task="regression"` or `task="classification"` overrides the detection. Local only for now.
* **No target.** `zg.anomalies`, `zg.impute`, `zg.synthesize` and `zg.cluster` (also in `zerograd.unsupervised`):
  anomaly detection with the columns why, imputation after a championship on hidden cells, synthetic rows scored by
  detection AUC and exact copies, and clustering with segment profiles. TabPFN v2 and TabICLv2 through the chain
  rule; FoMo-0D (MIT) for anomalies and ZEUS (Apache 2.0) for clustering, both zero-shot, with their inference code
  vendored. Isolation Forest, LOF, kNN, mean/mode, KNN and iterative imputation, k-means and a Gaussian mixture
  compete as baselines. Local only for now.
* **Evaluation and decisions.** `zg.cross_validate` runs every model on k folds, split at random, by group
  (`groups=`) or by time (`time=`, walking forward), with KS and lift for binary tasks and a baseline;
  `cap.optimize_threshold` picks the cut by cost, F1, Youden, or recall/precision under a constraint, from the
  championship holdout; `cap.importance` measures each column by permutation. `ZeroGradClassifier` and
  `ZeroGradRegressor` put any model inside scikit-learn (pipelines, grid search, `cross_val_score`).
* **Catalog.** `zerograd.list_models(task=..., commercial=...)` and `zerograd.list_tasks()`, as tables in
  notebooks and terminals, from the same registry the license guards use.
* **Bring your own model** with `zerograd.register_model`.
* **Context curves.** `zg.context_curve(...)` and `cap.context_curve(test)` score one model on growing context
  sizes, with timings, and recommend the smallest size within a tolerance of the best score; `.plot()` draws
  score and latency. `cap.update(context_rows=n)` predicts with the smaller context.
* `remote=True` (or an API key) keeps the platform client from 0.1: capabilities with versions, batch runs,
  pipelines, scheduled championships.
* **Outcomes (platform).** Remote predictions carry a `prediction_id`, and `id_column=` keeps your own id as the
  reference. `cap.send_outcomes(...)` sends what actually happened (outcome, decision, observed_at) by either one,
  with a vault if the capability was created with one. `cap.performance(cohort=...)` returns the production
  metrics per cohort of prediction date, only for cohorts past the capability's `maturity_days`.

## 0.1.0

Platform client only (private).
