# Changelog

## 0.2.0 (unreleased)

First open source release, Apache 2.0. `pip install zerograd` includes the local engine with TabPFN v2 and TabICLv2.

* **Local engine.** `ZeroGrad()` without an API key runs the championship and predictions on your machine:
  every model scored on the same holdout, champion by ROC AUC (or the metric you pick), predictions with the
  champion. Save a capability to one file and open it again with `zerograd.load`.
* **Models.** Commercial use, on by default when installed: TabPFN v2, TabICLv2, Mitra-v2, TabDPT (and Mitra v1
  by name). Non-commercial, off by default and only by name: TabPFN 2.5 / 2.6 / 3 / 3.5, TabFM, EXAONE Tabular,
  LimiX-2. Weights download from each author's repository; nothing is hosted by ZeroGrad.
* **Catalog.** `zerograd.list_models(task=..., commercial=...)` and `zerograd.list_tasks()`, as tables in
  notebooks and terminals, from the same registry the license guards use.
* **Bring your own model** with `zerograd.register_model`.
* `remote=True` (or an API key) keeps the platform client from 0.1: capabilities with versions, batch runs,
  pipelines, scheduled championships.

## 0.1.0

Platform client only (private).
