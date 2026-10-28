"""The local backend: Perplexity's Decider app, or any server with its API, on this machine.

Decider takes Jev's request shape with three differences handled here: no `assets` field (every image ref that is a
whole value in `state` or a question becomes an inline `image_url` data-URL part at the same place), no `box` question,
and at most 8 images per question. It serves one request at a time and
runs its questions in series, so every POST in this process goes through one module-level slot. No auth, no cost.
"""
import base64
import json
import os
import plistlib
import threading
import time
from pathlib import Path
from xml.parsers.expat import ExpatError

from jevlib.backends.base import BackendError, BackendUnavailable, LimitError
from jevlib.backends.http import parse_result, snippet, urllib_transport
from jevlib.env import offline
from jevlib.images import ImageLimitError, image_size

DEFAULT_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL = "pplx-decider-flash-v1.2"
INFO_PLIST = Path("/Applications/Decider.app/Contents/Info.plist")
MAX_IMAGES = 8
USER_AGENT = "jev-motion/0.1"
HEALTH_TIMEOUT_S = 5.0
BACKOFF_CAP_S = 10

# The server takes one request at a time; this holds for every LocalBackend in the process and whatever thread pool
# the caller uses.
_IN_FLIGHT = threading.BoundedSemaphore(1)


def _whole_refs(obj, images, found):
    """Add to `found` every image ref that is a whole string value anywhere inside `obj`."""
    if isinstance(obj, str):
        if obj in images:
            found.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _whole_refs(v, images, found)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _whole_refs(v, images, found)
    return found


class LocalBackend:
    name = "local"

    def __init__(self, *, url=None, timeout=600.0, max_tries=6, transport=urllib_transport, sleep=time.sleep,
                 plist=INFO_PLIST):
        self.url = (url or os.environ.get("JEVM_DECIDER_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.max_tries = max_tries
        self._transport = transport
        self._sleep = sleep
        self._plist = Path(plist)

    def header_flags(self, req):
        return []

    def version(self):
        """The Decider app version from its Info.plist; `""` when the app or the key is not there."""
        try:
            with open(self._plist, "rb") as f:
                info = plistlib.load(f)
        except (OSError, ValueError, ExpatError):
            return ""
        version = info.get("CFBundleShortVersionString") if isinstance(info, dict) else None
        return version if isinstance(version, str) else ""

    def health(self):
        """`GET /health`; BackendUnavailable unless Decider is `ready` with images on. Adds `app_version`."""
        if offline():
            raise BackendUnavailable("offline mode: JEVM_OFFLINE is set")
        try:
            code, _, raw = self._transport("GET", self.url + "/health", None, {"User-Agent": USER_AGENT},
                                           HEALTH_TIMEOUT_S)
        except OSError as e:
            raise BackendUnavailable(f"Decider is not reachable at {self.url}: {type(e).__name__}: {e}") from None
        if code != 200:
            raise BackendUnavailable(f"Decider /health answered HTTP {code}: {snippet(raw, 200)}")
        try:
            info = json.loads(raw.decode("utf-8"))
        except ValueError:
            info = None
        if not isinstance(info, dict):
            raise BackendUnavailable(f"Decider /health is not JSON: {snippet(raw, 200)}")
        if info.get("status") != "ready":
            raise BackendUnavailable(f"Decider is not ready (status {info.get('status')!r})")
        if info.get("images") is not True:
            raise BackendUnavailable("Decider has images turned off")
        return {**info, "app_version": self.version()}

    def body(self, req, model):
        """The Decider request body. LimitError for a Box, more than 8 images on a question, an unreadable image,
        or an image ref that is embedded in a longer string or appears nowhere."""
        for qid, q in req.questions.items():
            if isinstance(q, dict) and q.get("type") == "box":
                raise LimitError(f"question {qid!r}: Decider has no `box` question type")
        in_state = _whole_refs(req.state, req.images, set())
        worst = max([len(in_state)] + [len(in_state | _whole_refs(q, req.images, set()))
                                       for q in req.questions.values()])
        if worst > MAX_IMAGES:
            raise LimitError(f"a question carries {worst} images; Decider takes at most {MAX_IMAGES} per question")
        parts, seen = {}, set()

        def part(ref):
            if ref not in parts:
                path = req.images[ref]
                try:
                    kind = image_size(path)[2]
                    data = Path(path).read_bytes()
                except (ImageLimitError, OSError) as e:
                    raise LimitError(str(e)) from None
                fmt = "jpeg" if kind == "jpg" else kind
                url = f"data:image/{fmt};base64," + base64.b64encode(data).decode("ascii")
                parts[ref] = {"type": "image_url", "image_url": {"url": url}}
            return parts[ref]

        def sub(obj):
            if isinstance(obj, str):
                if obj in req.images:
                    seen.add(obj)
                    return part(obj)
                for ref in req.images:
                    if ref in obj:
                        raise LimitError(f"image ref {ref} is embedded in a longer string; "
                                         "an inline image must be a whole value")
                return obj
            if isinstance(obj, dict):
                return {k: sub(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [sub(v) for v in obj]
            return obj

        state, questions = sub(req.state), sub(req.questions)
        missing = sorted(set(req.images) - seen)
        if missing:
            raise LimitError(f"image refs not found as whole values in state or questions: {missing[:3]}")
        return {"model": model, "state": state, "questions": questions}

    def send(self, req, model):
        if offline():
            raise BackendUnavailable("offline mode: JEVM_OFFLINE is set")
        data = json.dumps(self.body(req, model), ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        last = None
        for attempt in range(self.max_tries):
            try:
                with _IN_FLIGHT:
                    t0 = time.monotonic()
                    code, _, raw = self._transport("POST", self.url + "/v1/decisions", data, headers, self.timeout)
                    elapsed = time.monotonic() - t0
            except OSError as e:
                last = BackendUnavailable(f"Decider is not reachable at {self.url}: {type(e).__name__}: {e}")
            else:
                if code == 200:
                    return parse_result(raw, elapsed)
                last = BackendError(f"HTTP {code}: {snippet(raw)}", status=code)
                if not (code == 429 or code >= 500):
                    raise last
            if attempt < self.max_tries - 1:
                self._sleep(min(BACKOFF_CAP_S, 0.5 * 2 ** attempt))
        raise last
