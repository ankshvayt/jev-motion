"""The module contract. A module is pure: it builds requests from the spec and kit manifest, and turns the
answers into decision rows. It does no I/O, so a policy change re-runs `decide()` with zero new requests.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jevlib.request import Answer, Request


class Module:
    id: str = ""                       # module id used in decisions.json, e.g. "universal.captions"
    decisions: tuple[str, ...] = ()    # decision ids this module decides, e.g. ("CAP-02", "CAP-03")

    def applies(self, spec: dict, kit: dict) -> bool:
        raise NotImplementedError

    def requests(self, spec: dict, kit: dict, policy: dict) -> list[Request]:
        raise NotImplementedError

    def decide(self, spec: dict, kit: dict, answers: list[Answer], policy: dict) -> list[dict]:
        """`answers[i]` answers `requests(spec, kit, policy)[i]`. Returns decision rows; never touches the network."""
        raise NotImplementedError
