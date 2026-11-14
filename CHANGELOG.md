# Changelog

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
  evaluation and thresholds, anomalies, clustering, and outcomes on the platform with an API key.

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
