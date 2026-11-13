"""A short, reproducible benchmark: ZeroGrad's models and champion against gradient boosting.

    pip install "zerograd[all]"
    python benchmarks/run.py                        # the default OpenML datasets, 3 seeds
    python benchmarks/run.py --datasets 31 37 --seeds 1
    python benchmarks/run.py --csv my.csv --target churned

For each dataset and seed: a stratified 80/20 split. The test 20% is never seen by anything.
* Every installed commercial model is fit on the training rows (a stratified sample of up to 10,000) and
  scored on test.
* The ZeroGrad champion is chosen by the championship on the training rows only, then scored on test.
* HistGradientBoosting (scikit-learn, defaults) is the classic baseline, fit on the same training rows.
Writes benchmarks/results.csv and prints a Markdown table of mean test ROC AUC, with wins and losses.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

os.environ.setdefault("ZEROGRAD_QUIET", "1")
warnings.filterwarnings("ignore")

from summarize import report  # noqa: E402  (benchmarks/summarize.py)

from zerograd import ZeroGrad  # noqa: E402
from zerograd.local.championship import stratified_sample  # noqa: E402
from zerograd.local.preprocess import Preprocessor  # noqa: E402
from zerograd.models import resolve_models  # noqa: E402

# OpenML classification datasets that appear in tabular benchmarks (TabArena among them): small to medium,
# binary and multiclass, numeric and categorical. Ids are OpenML dataset ids.
DEFAULT_DATASETS = {
    31: "credit-g",
    37: "diabetes",
    1464: "blood-transfusion",
    1494: "qsar-biodeg",
    1067: "kc1",
    40981: "Australian",
    40701: "churn",
    1489: "phoneme",
    40983: "wilt",
    1487: "ozone-level-8hr",
    23: "cmc",
    54: "vehicle",
}
MAX_ROWS = 20_000  # bigger datasets are subsampled (stratified) so a CPU run ends today


def load_openml(data_id: int) -> tuple[pd.DataFrame, str, str]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = bunch.frame
    target = bunch.target.name if hasattr(bunch.target, "name") else bunch.target_names[0]
    return df, target, bunch.details.get("name", str(data_id))


def auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=list(range(proba.shape[1]))))


def scores(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Every test metric for one model: ranking, probabilities and hard predictions at argmax (0.5 for binary)."""
    k = proba.shape[1]
    proba = np.clip(proba, 1e-15, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    out = {"test_roc_auc": auc(y_true, proba),
           "test_log_loss": float(log_loss(y_true, proba, labels=list(range(k)))),
           "test_accuracy": float(accuracy_score(y_true, pred)),
           "test_balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
           "test_f1": float(f1_score(y_true, pred, average="binary" if k == 2 else "macro"))}
    onehot = np.eye(k)[y_true]
    out["test_brier"] = float(((proba - onehot) ** 2).sum(axis=1).mean() / (1 if k > 2 else 2))
    return out


def run_one(df: pd.DataFrame, target: str, name: str, seed: int, specs) -> list[dict]:
    df = df[df[target].notna()].reset_index(drop=True)
    if len(df) > MAX_ROWS:
        df = df.iloc[stratified_sample(df[target].astype(str).to_numpy(), MAX_ROWS, seed=seed)].reset_index(drop=True)
    train, test = train_test_split(df, test_size=0.2, stratify=df[target], random_state=seed)
    pre, kept = Preprocessor.fit(train, target)
    Xtr, ytr = pre.transform(kept), pre.encode_target(kept[target])
    Xte, yte = pre.transform(test), pre.encode_target(test[target])
    ctx = stratified_sample(ytr, 10_000, seed=seed)
    n = len(pre.classes)
    rows = []

    def record(model: str, fn) -> None:
        t0 = time.monotonic()
        try:
            values, err = scores(yte, fn()), None
        except Exception as exc:  # a model failing is a result too
            values, err = {"test_roc_auc": None}, f"{type(exc).__name__}: {exc}"[:200]
        rows.append({"dataset": name, "seed": seed, "rows": len(df), "classes": n, "model": model,
                     **values, "seconds": round(time.monotonic() - t0, 1), "error": err})

    for spec in specs:
        record(spec.display, lambda spec=spec: spec.fit(Xtr[ctx], ytr[ctx], n, serving=False).predict_proba(Xte))
    record("HistGradientBoosting", lambda: HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr).predict_proba(Xte))

    def champion():
        cap = ZeroGrad(remote=False, models=[s.name for s in specs], progress=False).create(name, train, target)
        out = cap.predict(test.drop(columns=target))
        rows_champ.append(cap.champion_display)
        return out[[f"proba_{c}" for c in cap.classes]].to_numpy()

    rows_champ: list[str] = []
    record("ZeroGrad champion", champion)
    if rows_champ:
        rows[-1]["champion"] = rows_champ[0]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", type=int, help="OpenML dataset ids (default: the built-in list)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", nargs="*", help="model names (default: every installed commercial model)")
    ap.add_argument("--csv", help="run on your own CSV instead of OpenML")
    ap.add_argument("--target", help="target column for --csv")
    ap.add_argument("--out", default=str(Path(__file__).with_name("results.csv")))
    args = ap.parse_args()

    specs = resolve_models(args.models)
    print(f"Models: {', '.join(s.display for s in specs)} + HistGradientBoosting baseline")
    if args.csv:
        sources = [(pd.read_csv(args.csv), args.target, Path(args.csv).stem)]
    else:
        sources = []
        for data_id in args.datasets or DEFAULT_DATASETS:
            df, target, name = load_openml(data_id)
            sources.append((df, target, name))

    results = []
    for df, target, name in sources:
        for seed in range(args.seeds):
            print(f"… {name} (seed {seed}, {len(df):,} rows)", flush=True)
            results += run_one(df, target, name, seed, specs)
            pd.DataFrame(results).to_csv(args.out, index=False)

    print("\n" + report(pd.DataFrame(results), args.out))
    print(f"\nAll rows: {args.out}")


if __name__ == "__main__":
    main()
