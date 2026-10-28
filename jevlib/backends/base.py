"""The contract every decision backend meets: the errors it raises and the methods it offers."""
from typing import Protocol, runtime_checkable

from jevlib.request import Request

# Sent with value "true" on every Jev request that carries an image, and on no other. `header_flags` reports it
# by this name so it is part of the cache key.
IMAGES_HEADER = "TypeSafe-Danger-Zone-Images"


class BackendUnavailable(RuntimeError):
    """The backend cannot be reached or used at all (no key, offline mode, connection refused, health failed)."""


class BackendError(RuntimeError):
    """The backend answered, but with an error. `status` is the HTTP status when there was one."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class LimitError(BackendError):
    """A request rejected by our own checks before anything was sent (image size, token cap, no Box on Decider)."""


@runtime_checkable
class Backend(Protocol):
    name: str

    def send(self, req: Request, model: str) -> dict:
        """One request. Returns `{"answers": dict, "model": str | None, "input_tokens": int, "latency_s": float}`."""

    def header_flags(self, req: Request) -> list[str]:
        """The request-affecting headers this request would carry; they go into the cache key."""

    def version(self) -> str:
        """The app version for a local backend; `""` for Jev."""

    def health(self) -> dict:
        """Raises BackendUnavailable when the backend cannot take a request."""
