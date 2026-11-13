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
