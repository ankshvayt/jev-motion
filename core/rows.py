"""Decision rows (docs/decisions.md): the row builder and the evidence blocks that go into it.

`evidence_from` reads an `Answer` by attribute only (request.qs.id and .version, backend, model, request_sha256, cached), so
this module imports nothing from `jevlib`.
"""
from .policy import LANES, NEVER_AUTO, ROUTE_REASONS


def make_row(*, decision: str, module: str, target: dict, value: dict, p: float | None, confidence: dict | None, lane: str,
             route_reasons: list[str], evidence: dict, alternatives: list[dict] | None = None) -> dict:
    """One decision row. Raises ValueError on an unknown lane or route reason. A never-auto decision asked for `auto` leaves in the
    `agent` lane with `never_auto` among its reasons, whatever the caller or the policy file said."""
    if lane not in LANES:
        raise ValueError(f"unknown lane {lane!r}; expected one of {LANES}")
    reasons = list(route_reasons)
    for reason in reasons:
        if reason not in ROUTE_REASONS:
            raise ValueError(f"unknown route reason {reason!r}")
    if decision in NEVER_AUTO and lane == "auto":
        lane = "agent"
        if "never_auto" not in reasons:
            reasons.append("never_auto")
    return {
        "id": f"{decision}:{target['id']}",
        "decision": decision,
        "module": module,
        "target": target,
        "value": value,
        "alternatives": list(alternatives or []),
        "p": p,
        "confidence": confidence,
        "lane": lane,
        "route_reasons": reasons,
        "status": "proposed",
        "resolved": None,
        "evidence": evidence,
    }


def evidence_from(answer, policy_entry: dict, *, order_check: str | None = None) -> dict:
    """The row's `evidence` block for a decision an answer produced. `order_check` is "agree", "flip" or None (not asked)."""
    return {
        "label": policy_entry["evidence"]["label"],
        "ref": policy_entry["evidence"]["ref"],
        "qset": answer.request.qs.id,
        "qset_version": answer.request.qs.version,
        "backend": answer.backend,
        "model": answer.model,
        "request_sha256": answer.request_sha256,
        "order_check": order_check,
        "cached": answer.cached,
    }


def code_evidence(policy_entry: dict) -> dict:
    """The `evidence` block for a row code decided without a model: the same keys, backend "none", no request."""
    return {
        "label": policy_entry["evidence"]["label"],
        "ref": policy_entry["evidence"]["ref"],
        "qset": None,
        "qset_version": None,
        "backend": "none",
        "model": None,
        "request_sha256": None,
        "order_check": None,
        "cached": False,
    }


def error_route(answer) -> tuple[str, list[str]]:
    """Lane and reasons for a row whose answer failed: the agent decides it, and the reasons say why no model did."""
    return "agent", [answer.error] + list(answer.route_reasons)
