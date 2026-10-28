"""The API key, offline mode and redaction. JEV_API is read by code only: never printed, logged, cached or written.

Any error text that could carry the key goes through `redact` before it reaches stdout, stderr or a file.
"""
import os
from pathlib import Path


def load_key(name="JEV_API", start=None, environ=None):
    """The key from the environment, else from a `.env` file in `start` (default: the current working directory). Only
    that one folder is read; parent folders are not searched. Accepts `NAME=value` and `export NAME=value`, with or
    without quotes. Returns "" when absent."""
    environ = os.environ if environ is None else environ
    key = (environ.get(name) or "").strip()
    if key:
        return key
    env = (Path(start) if start else Path.cwd()) / ".env"
    if not env.is_file():
        return ""
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        k, sep, v = line.partition("=")
        if sep and k.strip() == name:
            return v.strip().strip("\"'")
    return ""


def redact(text, secrets):
    """`text` with every non-empty secret replaced by REDACTED."""
    for s in secrets:
        if s:
            text = text.replace(s, "REDACTED")
    return text


def offline():
    """True when JEVM_OFFLINE is set: the real backends then raise BackendUnavailable before any network call."""
    return os.environ.get("JEVM_OFFLINE", "") not in ("", "0")
