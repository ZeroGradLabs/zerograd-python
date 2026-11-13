"""Turn benchmark rows into the README table. Needs only pandas.

    python benchmarks/summarize.py benchmarks/results.csv
    python benchmarks/summarize.py part-*.csv --out benchmarks/results.csv   # merge the CI matrix parts
"""

from __future__ import annotations

import argparse

import pandas as pd


def markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| dataset | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in table.iterrows():
        best = row.max()
        cells = [("**%.3f**" % v if v == best else "%.3f" % v) if pd.notna(v) else "—" for v in row]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report(res: pd.DataFrame, out: str | None = None) -> str:
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
    errors = res[res["error"].notna()] if "error" in res else res.iloc[0:0]
    if len(errors):
        parts += ["", f"{len(errors)} model runs failed; see the error column" + (f" in {out}" if out else "")]
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
