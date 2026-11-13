"""The demo behind the README GIF: a typed-out session that really runs.

    python examples/demo.py                 # every commercial model you have installed
    vhs scripts/demo.tape                   # records docs/demo.gif (https://github.com/charmbracelet/vhs)
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")


def typed(line: str, pause: float = 0.5) -> None:
    sys.stderr.write("\033[38;5;141m>>>\033[0m ")
    for ch in line:
        sys.stderr.write(ch)
        sys.stderr.flush()
        time.sleep(0.035)
    sys.stderr.write("\n")
    sys.stderr.flush()
    time.sleep(pause)


TITANIC = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

typed("from zerograd import ZeroGrad", 0.2)
import pandas as pd  # noqa: E402

from zerograd import ZeroGrad  # noqa: E402

df = pd.read_csv(os.environ.get("DEMO_CSV", TITANIC))
train = df.sample(frac=0.8, random_state=0)
new = df.drop(train.index).drop(columns="Survived")

models = os.environ.get("DEMO_MODELS")
typed("zg = ZeroGrad()")
zg = ZeroGrad(models=models.split(",") if models else None)
time.sleep(0.8)
typed('cap = zg.create("titanic", examples=train, target="Survived")')
cap = zg.create("titanic", examples=train, target="Survived")
time.sleep(1.2)
typed("cap.predict(new).head(3)")
print(cap.predict(new).head(3).round(3).to_string(), file=sys.stderr)  # noqa: T201
time.sleep(2.5)
