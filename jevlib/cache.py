"""The client-side answer cache. It stores raw answers keyed on the full request.

The key keeps dict insertion order on purpose: a Choice's options are a JSON object, so a request with the options
reversed must be a different entry (a sorted-key hash would silently replay forward answers for reversed ones).
Never `sort_keys=True` on anything that becomes a request. A record never holds headers or credentials.
"""
import hashlib
import json
import os
import threading
from pathlib import Path

from jevlib.paths import cache_dir

KEY_PREFIX = b"jevm-cache-v2\n"
FORBIDDEN_KEYS = ("headers", "authorization")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_key(backend, model, backend_version, header_flags, state, questions, image_shas):
    """`image_shas` are the sha256 of the images as sent (after resizing), in reference order."""
    body = json.dumps({"state": state, "questions": questions},
                      sort_keys=False, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(
        KEY_PREFIX
        + backend.encode() + b"\n"
        + model.encode() + b"\n"
        + backend_version.encode() + b"\n"
        + ",".join(sorted(header_flags)).encode() + b"\n"
        + body.encode()
        + b"\n" + ",".join(image_shas).encode()
    ).hexdigest()


def _forbidden(obj):
    if isinstance(obj, dict):
        return any(str(k).lower() in FORBIDDEN_KEYS or _forbidden(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_forbidden(v) for v in obj)
    return False


class Cache:
    def __init__(self, root=None):
        """`root` defaults to `jevlib.paths.cache_dir()` (`JEVM_CACHE_DIR`, else `.jevm-cache/` here)."""
        self.root = Path(root) if root is not None else cache_dir()

    def path(self, key):
        return self.root / key[:2] / f"{key}.json"

    def get(self, key):
        """The stored record, or None on a miss or an unreadable file."""
        try:
            return json.loads(self.path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key, record):
        """Write `record` atomically (temp file, then os.replace). Refuses a record that carries headers."""
        if _forbidden(record):
            raise ValueError("cache record must not hold headers or credentials")
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
