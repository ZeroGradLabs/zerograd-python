"""The context-curve figure for the README and the docs: TabPFN v2 on 12,000 rows of OpenML Adult.

    pip install -e ".[dev]" && python scripts/context_curve_figure.py     # writes docs/context-curve.png and .csv
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402

from zerograd import ZeroGrad  # noqa: E402

model = sys.argv[1] if len(sys.argv) > 1 else "tabpfn-v2"
adult = fetch_openml("adult", version=2, as_frame=True).frame.sample(12_000, random_state=0)
zg = ZeroGrad(remote=False, models=[model], progress=not os.environ.get("CI"))
curve = zg.context_curve(adult.iloc[:10_000], "class", test=adult.iloc[10_000:],
                         sizes=[250, 500, 1_000, 2_000, 5_000, 10_000])
print(curve.round(4).to_string())
print(curve.summary())
fig, ax = plt.subplots(figsize=(8, 4.2))
curve.plot(ax=ax, title=f"Adult census income · {curve.model} · 2,000 test rows")
fig.tight_layout()
fig.savefig("docs/context-curve.png", dpi=200)
curve.to_csv("docs/context-curve.csv")
