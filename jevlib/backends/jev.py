"""The Jev backend: TypeSafe System One in the cloud.

The key (`JEV_API`) is read when a request is sent and never stored on the backend; every error message passes through
`redact` with it. The images header goes on a request that carries an image and on no other. Images must already fit Jev's limits (`images.fit_for_jev`): out-of-range ones are refused
here before anything is sent, because Jev rejects rather than resizes them.
"""
import base64
import json
import time
from pathlib import Path

from jevlib.backends.base import IMAGES_HEADER, BackendError, BackendUnavailable, LimitError
from jevlib.backends.http import parse_result, retry_after, snippet, urllib_transport
from jevlib.env import load_key, offline, redact
from jevlib.images import ImageLimitError, check_limits, image_size

URL = "https://api.typesafe.ai/v1/systemone"
USER_AGENT = "jev-motion/0.1"
RETRY_STATUS = frozenset({408, 429, 529})
BACKOFF_CAP_S = 20


def retryable(status):
    return status in RETRY_STATUS or status >= 500


class JevBackend:
    name = "jev"

    def __init__(self, *, key_loader=load_key, url=URL, timeout=90.0, max_tries=5, transport=urllib_transport,
                 sleep=time.sleep):
        self._key_loader = key_loader
        self.url = url
        self.timeout = timeout
        self.max_tries = max_tries
        self._transport = transport
        self._sleep = sleep

    def _key(self):
        key = self._key_loader()
        if not key:
            raise BackendUnavailable("JEV_API is not set in the environment or a .env file")
        return key

    def _ready_key(self):
        """The key, once we know a request may go out at all."""
        if offline():
            raise BackendUnavailable("offline mode: JEVM_OFFLINE is set")
        return self._key()

    def header_flags(self, req):
        return [IMAGES_HEADER] if req.images else []

    def version(self):
        return ""

    def health(self):
        """No request is spent on this: Jev can take one when we are online and hold a key."""
        self._ready_key()
        return {"status": "ready"}

    def body(self, req, model):
        body = {"model": model, "state": req.state, "questions": req.questions}
        if req.images:
            assets = {}
            for ref, path in req.images.items():
                try:
                    check_limits(path)
                    fmt = image_size(path)[2]
                    data = Path(path).read_bytes()
                except (ImageLimitError, OSError) as e:
                    raise LimitError(str(e)) from None
                assets[ref] = {"type": "image", "format": fmt, "data_base64": base64.b64encode(data).decode("ascii")}
            body["assets"] = assets
        return body

    def _headers(self, req, key):
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json", "User-Agent": USER_AGENT}
        if req.images:
            headers[IMAGES_HEADER] = "true"
        return headers

    def headers(self, req):
        """What is sent with `req`, `Authorization` included: never log, print or cache this dict."""
        return self._headers(req, self._key())

    def send(self, req, model):
        key = self._ready_key()
        secrets = (key,)
        data = json.dumps(self.body(req, model)).encode("utf-8")
        headers = self._headers(req, key)
        last = None
        for attempt in range(self.max_tries):
            t0 = time.monotonic()
            wait = min(BACKOFF_CAP_S, 2 ** attempt)
            try:
                code, resp_headers, raw = self._transport("POST", self.url, data, headers, self.timeout)
            except OSError as e:
                last = BackendUnavailable(redact(f"Jev is not reachable: {type(e).__name__}: {e}", secrets))
            else:
                if code == 200:
                    return parse_result(raw, time.monotonic() - t0, secrets)
                last = BackendError(redact(f"HTTP {code}: {snippet(raw)}", secrets), status=code)
                if not retryable(code):
                    raise last
                asked = retry_after(resp_headers)
                wait = wait if asked is None else asked
            if attempt < self.max_tries - 1:
                self._sleep(wait)
        raise last
