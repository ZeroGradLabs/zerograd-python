from __future__ import annotations

import random
import time
import warnings
from typing import Any

import httpx

from ._version import __version__
from .errors import APIError, UsageWarning, ZeroGradError, from_response

RETRY_STATUS = {429, 502, 503, 504}


class Transport:
    """httpx with auth, retries (cold starts, 429, 5xx gateways) and error mapping."""

    def __init__(self, base_url: str, api_key: str, timeout: float, http_client: httpx.Client | None, max_retries: int):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        headers = {"Authorization": f"Bearer {api_key}", "User-Agent": f"zerograd-python/{__version__}"}
        if http_client is not None:
            http_client.headers.update(headers)
            self.http = http_client
        else:
            self.http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout)
        self._warned = False

    def request(self, method: str, path: str, *, raw: bool = False, **kwargs: Any) -> Any:
        url = path if path.startswith("http") else self.base_url + path
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.http.request(method, url, **kwargs)
            except httpx.TransportError as exc:  # connection reset, cold start timeouts
                last = exc
                self._sleep(attempt)
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                if not (resp.status_code == 429 and _code(resp) == "usage_limit"):
                    self._sleep(attempt, resp.headers.get("retry-after"))
                    continue
            self._usage_warning(resp)
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except ValueError:
                    body = resp.text
                raise from_response(resp.status_code, body)
            if raw:
                return resp
            return resp.json() if resp.content else None
        raise APIError(f"Could not reach {self.base_url}: {last}") from last

    def _sleep(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after and retry_after.isdigit():
            time.sleep(min(float(retry_after), 30))
        else:
            time.sleep(min(0.5 * 2**attempt, 8) + random.random() * 0.25)

    def _usage_warning(self, resp: httpx.Response) -> None:
        msg = resp.headers.get("x-zerograd-usage-warning")
        if msg and not self._warned:
            self._warned = True
            warnings.warn(f"{msg} At the limit, predict raises LimitError.", UsageWarning, stacklevel=4)

    def put_chunk(self, url: str, data: bytes, headers: dict[str, str]) -> httpx.Response:
        """Upload chunks go to a session URL (Cloud Storage in production). Never send our API key there."""
        if url.startswith(self.base_url):
            return self.http.put(url, content=data, headers=headers)
        for attempt in range(self.max_retries + 1):
            try:
                return httpx.put(url, content=data, headers=headers, timeout=300)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                self._sleep(attempt)
        raise ZeroGradError("unreachable")


def _code(resp: httpx.Response) -> str | None:
    try:
        return resp.json().get("error")
    except ValueError:
        return None
