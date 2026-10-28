"""Decision policy: lanes, route reasons, the never-auto list (docs/never-auto.md), the review band and the highlight budget.

The never-auto list is code, not policy data: a policy file can lower or raise thresholds, but it can never make a
`NEVER_AUTO` decision eligible for the `auto` lane. Nothing here touches the network or imports another task's code.
"""
import copy
import json
import math
from pathlib import Path

# Claims (STY-06, QA-05) and the decisions someone has to look at an image for (AST-01, QA-01); docs/never-auto.md.
NEVER_AUTO = frozenset("STY-06 QA-05 AST-01 QA-01".split())

LANES = ("auto", "agent", "human")

# Why a row went to its lane (docs/decisions.md). `presence_low` marks an AST-01 drop on a low presence Noul.
ROUTE_REASONS = (
    "above_auto", "in_review_band", "order_flip", "never_auto", "budget_edge",
    "jev_unavailable", "jev_error", "local_unavailable", "backend_fallback", "quarantined",
    "module_not_helping", "order_agree", "presence_low",
)

# A decision with no entry in the policy file is reviewed by the agent, never applied on its own.
_NO_ENTRY = {"lane": "agent", "can_auto": False, "evidence": {"label": "Default", "ref": "no policy entry"}}


def load_policy(path: Path | None = None) -> dict:
    """Read a policy file (default: `policies/default.json`). The default path comes from `jevlib.paths` when needed."""
    if path is None:
        from jevlib.paths import POLICIES_DIR
        path = POLICIES_DIR / "default.json"
    return json.loads(Path(path).read_text(encoding="utf-8"))


def decision_policy(policy: dict, decision: str) -> dict:
    """The policy entry for one decision id, as a copy. A never-auto id always comes back with `can_auto` false."""
    entry = copy.deepcopy(policy.get("decisions", {}).get(decision, _NO_ENTRY))
    if decision in NEVER_AUTO:
        entry["can_auto"] = False
        if entry.get("lane") == "auto":
            entry["lane"] = "agent"
    return entry


def in_band(p: float, threshold: float, band: float) -> bool:
    """True when `p` is within `band` of `threshold` (edges included). Such a value never goes to `auto`."""
    return abs(p - threshold) <= band + 1e-9


def round_half_up(x: float) -> int:
    """Round halves up (Python's `round` rounds halves to even). The epsilon absorbs float error such as 6.4999999999."""
    return math.floor(x + 0.5 + 1e-9)


def budget_k(density: float, n: int) -> int:
    """How many of `n` units a density budget keeps: at least one unless there are none."""
    if n <= 0:
        return 0
    return max(1, round_half_up(density * n))
