from __future__ import annotations

import sys


class Bar:
    """A one-line progress bar on stderr that also prints finished stages as ✓ lines. Works in notebooks."""

    def __init__(self, enabled: bool = True, width: int = 28):
        self.enabled = enabled
        self.width = width
        self.seen = 0
        self.last = ""

    def update(self, run: dict) -> None:
        if not self.enabled:
            return
        log = run.get("log") or []
        active = run.get("status") in ("queued", "running")
        # Finished stages become ✓ lines; the latest one stays on the bar while the run is active.
        upto = len(log) - 1 if active else len(log)
        for entry in log[self.seen : upto]:
            self._line(f"✓ {entry['message']}")
        self.seen = max(self.seen, upto)
        if active:
            p = float(run.get("progress") or 0)
            filled = int(round(p * self.width))
            text = f"{run.get('message', '')[:60]:<60} {'█' * filled}{'░' * (self.width - filled)} {p:>4.0%}"
            if text != self.last:
                sys.stderr.write("\r" + text)
                sys.stderr.flush()
                self.last = text

    def _line(self, text: str) -> None:
        if self.last:
            sys.stderr.write("\r" + " " * (len(self.last) + 2) + "\r")
            self.last = ""
        sys.stderr.write(text + "\n")
        sys.stderr.flush()

    def done(self, text: str | None = None) -> None:
        if not self.enabled:
            return
        if self.last:
            sys.stderr.write("\r" + " " * (len(self.last) + 2) + "\r")
            self.last = ""
        if text:
            sys.stderr.write(text + "\n")
        sys.stderr.flush()
