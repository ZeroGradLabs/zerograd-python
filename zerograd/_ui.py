"""How the SDK talks: one greeting, live spinners and bars while models work, a podium at the end.

Rules it keeps:
* Everything goes to stderr, so stdout stays yours.
* Animation only on an interactive terminal or a notebook. In CI, pipes and log files you get plain,
  one-per-event lines (or nothing with ZEROGRAD_QUIET=1).
* Colors only where they render, and never with NO_COLOR set.
* `progress=False` (or ZEROGRAD_QUIET=1) silences all of it. Warnings about licenses are not silenced.
"""

from __future__ import annotations

import os
import random
import sys
import textwrap
import threading
import time
from contextlib import contextmanager
from typing import Iterator

# A line of personality per moment, picked at random. Shown only when output is on.
PHRASES: dict[str, tuple[str, ...]] = {
    "init": (
        "Hold on tight, going gradless.",
        "Look mama, no gradients!!",
        "Zero gradients were harmed.",
        "Gradients? Where we're going, we don't need gradients.",
        "Backprop called in sick. We don't need it.",
        "Pretrained on a million worlds. Ready for yours.",
    ),
    "download": (
        "Fetching a brain, first time only.",
        "Downloading a few million synthetic datasets' worth of intuition.",
        "Unpacking weights. You'll only wait for this once.",
    ),
    "championship": (
        "May the best prior win.",
        "Same holdout for everyone. No favorites.",
        "Let them read the room.",
        "Ranking models, not hyperparameters.",
    ),
    "fit": (
        "Reading your history. Not training on it.",
        "Context in. Epochs: zero.",
        "In-context learning, in your context.",
    ),
    "predict": (
        "Answers, no epochs.",
        "Looks like attention is all you need for tables as well.",
        "One forward pass, as promised.",
    ),
    "champion": (
        "Gradient-free and proud of it.",
        "Trained: never. Right: often.",
        "No GPUs were overworked in the making of this model.",
    ),
}


def fmt_value(v: float) -> str:
    """A metric for people: 0.931 for scores, 12.4 or 34,567 for errors in the target's unit."""
    v = float(v)
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    return f"{v:.3f}"


def error_summary(err: str, limit: int = 300) -> str:
    """The first line of an error ("RuntimeError: what went wrong"), whole up to `limit` characters."""
    first = next((ln.strip() for ln in str(err).splitlines() if ln.strip()), "")
    return first if len(first) <= limit else first[: limit - 1] + "…"


def phrase(moment: str) -> str:
    return random.choice(PHRASES[moment])


SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
MEDALS = ("🥇", "🥈", "🥉")
_greeted = False


def quiet_env() -> bool:
    return os.environ.get("ZEROGRAD_QUIET", "").lower() in ("1", "true", "yes")


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


def _ci() -> bool:
    return any(os.environ.get(k) for k in ("CI", "GITHUB_ACTIONS", "BUILDKITE", "JENKINS_URL", "GITLAB_CI"))


class Console:
    def __init__(self, enabled: bool = True, stream=None):
        self.stream = stream or sys.stderr
        self.enabled = enabled and not quiet_env()
        notebook = _in_notebook()
        tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.live = self.enabled and (tty or notebook) and not _ci()
        self.color = self.live and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
        self._line = ""
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ styles
    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def dim(self, t: str) -> str:
        return self._c("2", t)

    def bold(self, t: str) -> str:
        return self._c("1", t)

    def accent(self, t: str) -> str:  # ZeroGrad violet
        return self._c("38;5;141", t)

    def good(self, t: str) -> str:
        return self._c("32", t)

    def warn(self, t: str) -> str:
        return self._c("33", t)

    # ------------------------------------------------------------------ output
    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clear(self) -> None:
        if self._line:
            self._write("\r" + " " * len(self._line) + "\r")
            self._line = ""

    def say(self, text: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._clear()
            self._write(text + "\n")

    def status(self, text: str) -> None:
        """A line that rewrites itself (live), or nothing (plain: the caller says the important bits)."""
        if not self.live:
            return
        with self._lock:
            pad = max(0, len(self._line) - len(text))
            self._write("\r" + text + " " * pad)
            self._line = text

    def done(self, text: str) -> None:
        self.say(f"{self.good('✓')} {text}")

    # ------------------------------------------------------------------ pieces
    def greet(self, mode: str, detail: str) -> None:
        """The hello, once per process."""
        global _greeted
        if _greeted or not self.enabled:
            return
        _greeted = True
        from ._version import __version__

        tag = phrase("init")
        self.say(f"{self.accent('◆ ZeroGrad')} {self.dim(__version__)} · {mode} · {detail}")
        self.say(f"  {self.dim(tag)}")

    @contextmanager
    def spinner(self, text: str, done: str | None = None) -> Iterator["Spinner"]:
        sp = Spinner(self, text)
        sp.start()
        ok = False
        try:
            yield sp
            ok = True
        finally:
            sp.stop()
            if ok and done:
                self.done(f"{done} {self.dim(_secs(sp.elapsed))}")

    def bar(self, label: str, frac: float, right: str = "", width: int = 24) -> str:
        frac = min(1.0, max(0.0, frac))
        filled = int(round(frac * width))
        bar = self.accent("━" * filled) + self.dim("━" * (width - filled))
        return f"  {label} {bar} {right}"

    def podium(self, rows: list[tuple[str, float | None, str | None]], metric_label: str, attribution: str | None,
               baseline: float | None = None, *, higher_is_better: bool = True,
               baseline_label: str = "majority class", ties: list[str] | tuple[str, ...] = (),
               holdout_rows: int | None = None) -> None:
        """rows: (display, value, error), best first. Lower-is-better metrics (RMSE) get the longest bar at the lowest.

        `ties` are the models within noise of the champion: they get a ≈ and a note under the champion line."""
        if not self.enabled:
            return
        ok = [r for r in rows if r[2] is None and r[1] is not None]
        if not ok:
            return
        sign = 1.0 if higher_is_better else -1.0
        top = max(sign * v for _, v, _ in ok)
        low = min([sign * v for _, v, _ in ok] + ([sign * baseline] if baseline is not None else []))
        span = max(top - low, 1e-9)
        name_w = max(len(r[0]) for r in rows)
        self.say("")
        for i, (name, value, err) in enumerate(rows):
            medal = MEDALS[i] if i < len(MEDALS) and err is None else "  "
            if err is not None:
                self.say(f"  {medal} {name:<{name_w}}  {self.warn('did not finish')}")
                for line in textwrap.wrap(error_summary(err), 76):
                    self.say(f"     {' ' * name_w}  {self.dim(line)}")
                continue
            if value is None:
                self.say(f"  {medal} {name:<{name_w}}  {self.dim('n/a')}")
                continue
            width = 4 + int(round(20 * (sign * value - low) / span)) if len(ok) > 1 else 24
            bar = (self.accent if i == 0 else self.dim)("█" * width)
            label = self.bold(f"{name:<{name_w}}") if i == 0 else f"{name:<{name_w}}"
            tie = f" {self.dim('≈ tied')}" if name in ties else ""
            self.say(f"  {medal} {label}  {bar} {fmt_value(value)}{tie}")
        if baseline is not None:
            label = "without a model".ljust(name_w)
            self.say("     " + self.dim(f"{label}  {fmt_value(baseline)}  ({metric_label}, {baseline_label})"))
        champ = ok[0][0] if rows and rows[0][2] is None else None
        if champ:
            self.say("")
            self.say(f"{self.accent('◆')} {self.bold('Champion:')} {champ} · {metric_label} {fmt_value(rows[0][1])}"
                     + (f" {self.dim('· ' + attribution)}" if attribution else ""))
            if ties:
                rows_txt = f" on {holdout_rows:,} holdout rows" if holdout_rows else " on this holdout"
                who = ", ".join(ties)
                note = (f"≈ Technical tie with {who}: the gap is within noise{rows_txt} (90% bootstrap interval). "
                        "Either is a sound pick. If one is really better, more labeled rows will show it.")
                for line in textwrap.wrap(note, 78, subsequent_indent="  "):
                    self.say(self.dim("  " + line))


class Spinner:
    def __init__(self, console: Console, text: str):
        self.c = console
        self.text = text
        self.suffix = ""
        self.t0 = time.monotonic()
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def update(self, text: str | None = None, suffix: str | None = None) -> None:
        if text is not None:
            self.text = text
        if suffix is not None:
            self.suffix = suffix

    def start(self) -> None:
        if not self.c.live:
            self.c.say(f"… {self.text}")
            return
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _run(self) -> None:
        i = 0
        while not self._stop.wait(0.08):
            frame = self.c.accent(SPIN[i % len(SPIN)])
            self.c.status(f"{frame} {self.text}{self.suffix} {self.c.dim(_secs(self.elapsed))}")
            i += 1

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join()
        with self.c._lock:
            self.c._clear()


def _secs(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"
