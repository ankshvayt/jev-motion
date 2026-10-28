"""The HTTP seam the two real backends share.

A backend never calls urllib itself: it goes through a `Transport`, so tests swap in a fake and `JEVM_OFFLINE` is
enforced in one place per backend, before the transport is touched.
"""
import http.client
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable

from jevlib.backends.base import BackendError
from jevlib.env import redact

# (method, url, body, headers, timeout) -> (status, response headers, response body). Raises OSError or TimeoutError
# when the server cannot be reached; an HTTP error status is returned, not raised.
Transport = Callable[[str, str, bytes | None, dict, float], tuple[int, dict, bytes]]

SNIPPET_CHARS = 1500
_NUMERIC = re.compile(r"\d+(\.\d+)?")


def urllib_transport(method, url, body, headers, timeout):
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, dict(e.headers.items()) if e.headers else {}, e.read()
        finally:
            e.close()
    except http.client.HTTPException as e:
        raise OSError(f"{type(e).__name__}: {e}") from None


def header(headers, name):
    """A response header by name, whatever its case; None when absent."""
    wanted = name.lower()
    for k, v in headers.items():
        if k.lower() == wanted:
            return v
    return None


def retry_after(headers):
    """Seconds from a numeric `Retry-After`; None when absent or not a plain number (an HTTP date is ignored)."""
    value = header(headers, "Retry-After")
    if value is not None and _NUMERIC.fullmatch(value.strip()):
        return float(value)
    return None


def snippet(raw, limit=SNIPPET_CHARS):
    """The start of a response body as text, for an error message."""
    return raw.decode("utf-8", "replace")[:limit]


def parse_result(raw, latency_s, secrets=()):
    """Turn a 200 response body into what `Backend.send` returns; BackendError when it is not a System One answer."""
    try:
        resp = json.loads(raw.decode("utf-8"))
    except ValueError:
        raise BackendError(redact(f"non-JSON response: {snippet(raw)}", secrets)) from None
    if not isinstance(resp, dict) or not isinstance(resp.get("answers"), dict):
        raise BackendError(redact(f"response has no answers: {snippet(raw)}", secrets))
    usage = resp.get("usage") if isinstance(resp.get("usage"), dict) else {}
    return {"answers": resp["answers"], "model": resp.get("model"),
            "input_tokens": int(usage.get("input_tokens") or 0), "latency_s": latency_s}
