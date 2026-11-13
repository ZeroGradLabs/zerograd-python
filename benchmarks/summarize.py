"""Turn benchmark rows into the README table. Needs only pandas.

    python benchmarks/summarize.py benchmarks/results.csv
    python benchmarks/summarize.py part-*.csv --out benchmarks/results.csv   # merge the CI matrix parts
"""

from __future__ import annotations

import argparse

import pandas as pd


def markdown(table: pd.DataFrame, higher: bool = True, fmt: str = "%.3f") -> str:
    cols = list(table.columns)
    lines = ["| dataset | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in table.iterrows():
        best = row.max() if higher else row.min()
        cells = [(f"**{fmt % v}**" if v == best else fmt % v) if pd.notna(v) else "—" for v in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(res: pd.DataFrame, out: str | None = None) -> str:
    """Classification and regression rows, each in its own section."""
    if "task" not in res:
        res = res.assign(task="classification")
    res = res.assign(task=res["task"].fillna("classification"))
    parts = []
    if (res["task"] == "classification").any():
        parts.append(classification_report(res[res["task"] == "classification"]))
    if (res["task"] == "regression").any():
        parts.append(regression_report(res[res["task"] == "regression"]))
    errors = res[res["error"].notna()] if "error" in res else res.iloc[0:0]
    if len(errors):
        parts.append(f"{len(errors)} model runs failed; see the error column" + (f" in {out}" if out else ""))
    return "\n\n".join(parts)


def regression_report(res: pd.DataFrame) -> str:
    """RMSE per dataset, R² per dataset (comparable across datasets), and the champion's wins on each metric.
    A tie is within 0.5% of HistGradientBoosting (RMSE, MAE, seconds) or 0.001 (R²)."""
    rmse = res.pivot_table(index="dataset", columns="model", values="test_rmse", aggfunc="mean")
    r2 = res.pivot_table(index="dataset", columns="model", values="test_r2", aggfunc="mean")
    parts = ["Regression: mean test RMSE (lower is better)", "", markdown(rmse, higher=False, fmt="%.4g"),
             "", "Regression: mean test R² (higher is better)", "", markdown(r2.round(3))]
    lines = []
    for col, label, higher, rel in [("test_rmse", "RMSE", False, True), ("test_mae", "MAE", False, True),
                                    ("test_r2", "R²", True, False), ("seconds", "Seconds per fit + predict", False, True)]:
        t = res.pivot_table(index="dataset", columns="model", values=col, aggfunc="mean")
        if "HistGradientBoosting" not in t or "ZeroGrad champion" not in t:
            continue
        base, champ = t["HistGradientBoosting"], t["ZeroGrad champion"]
        d = (champ - base) * (1 if higher else -1)
        tol = (0.005 * base.abs()) if rel else 1e-3
        lines.append(f"| {label} | {int((d > tol).sum())} / {int((d.abs() <= tol).sum())} / {int((d < -tol).sum())} |")
    if lines:
        parts += ["", "Regression: ZeroGrad champion vs HistGradientBoosting (wins / ties / losses)", "",
                  "| metric | W / T / L |", "|---|---|", *lines]
    return "\n".join(parts)


def classification_report(res: pd.DataFrame) -> str:
    raw = res.pivot_table(index="dataset", columns="model", values="test_roc_auc", aggfunc="mean")
    parts = ["Mean test ROC AUC (higher is better)", "", markdown(raw.round(3))]
    if "HistGradientBoosting" in raw and "ZeroGrad champion" in raw:
        diff = raw["ZeroGrad champion"] - raw["HistGradientBoosting"]
        parts += ["", f"ZeroGrad champion vs HistGradientBoosting: {int((diff > 0.001).sum())} wins, "
                  f"{int((diff.abs() <= 0.001).sum())} ties, {int((diff < -0.001).sum())} losses "
                  f"(mean difference {diff.mean():+.3f})"]
    # the other metrics, champion against HistGradientBoosting, averaged over datasets
    other = [("test_log_loss", "Log loss", False), ("test_brier", "Brier score", False), ("test_accuracy", "Accuracy", True),
             ("test_balanced_accuracy", "Balanced accuracy", True), ("test_f1", "F1", True), ("seconds", "Seconds per fit + predict", False)]
    lines = []
    for col, label, higher in other:
        if col not in res:
            continue
        t = res.pivot_table(index="dataset", columns="model", values=col, aggfunc="mean")
        if "HistGradientBoosting" not in t or "ZeroGrad champion" not in t:
            continue
        d = (t["ZeroGrad champion"] - t["HistGradientBoosting"]) * (1 if higher else -1)
        lines.append(f"| {label} | {t['HistGradientBoosting'].mean():.3f} | {t['ZeroGrad champion'].mean():.3f} | "
                     f"{int((d > 1e-3).sum())} / {int((d.abs() <= 1e-3).sum())} / {int((d < -1e-3).sum())} |")
    if lines:
        parts += ["", "Other metrics, mean over datasets (wins / ties / losses for the champion)", "",
                  "| metric | HistGradientBoosting | ZeroGrad champion | W / T / L |", "|---|---|---|---|", *lines]
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="one or more results CSVs written by run.py")
    ap.add_argument("--out", help="write the merged rows here")
    args = ap.parse_args()
    res = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    if args.out:
        res.to_csv(args.out, index=False)
    print(report(res, args.out))


if __name__ == "__main__":
    main()
