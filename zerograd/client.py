"""``ZeroGrad``: one entry point, two engines.

* Local (default without an API key): the championship and predictions run on your machine with open
  tabular foundation models. Free, Apache 2.0, nothing leaves your computer.
* Remote (``remote=True``): the ZeroGrad platform runs them for you, with versions, monitoring, scheduled
  championships, cost-based thresholds, Explain, Ask and teams.
"""

from __future__ import annotations

import abc
import os
from typing import Any, Iterable

from ._remote import RemoteZeroGrad
from .local import LocalZeroGrad

_REMOTE_ONLY = ("base_url", "timeout", "max_retries", "http_client")


class ZeroGrad(abc.ABC):
    """Build a client.

    >>> zg = ZeroGrad()                    # local: runs here (no API key needed)
    >>> zg = ZeroGrad(remote=True)         # the platform: reads ZEROGRAD_API_KEY

    `remote=None` (default) picks the platform when an API key is given or ZEROGRAD_API_KEY is set,
    and local otherwise. Pass `remote=False` to stay local with a key in the environment.

    Local options: `models=[...]` picks the competitors (default: every commercial model installed);
    `noncommercial=[...]` names non-commercial models to add for research only (off by default; never
    available with remote=True). See `zerograd.models.list_models()`.
    """

    remote: bool

    def __new__(cls, api_key: str | None = None, *, remote: bool | None = None, models: Iterable[str] | None = None,
                noncommercial: Iterable[str] = (), **options: Any):
        if remote is None:
            remote = bool(api_key or os.environ.get("ZEROGRAD_API_KEY"))
        if remote:
            if noncommercial:
                raise PermissionError(
                    "Non-commercial models only run locally, on your own machine and under your own license "
                    "acceptance. They are never available with remote=True."
                )
            if models is not None:
                raise ValueError("models= picks local competitors; the platform runs its own championship.")
            return RemoteZeroGrad(api_key, **options)
        wrong = [k for k in options if k in _REMOTE_ONLY]
        if wrong:
            raise TypeError(f"{', '.join(wrong)} only apply with remote=True.")
        if api_key:
            raise TypeError("An API key only applies with remote=True.")
        return LocalZeroGrad(models=models, noncommercial=noncommercial, **options)


ZeroGrad.register(RemoteZeroGrad)
ZeroGrad.register(LocalZeroGrad)
