"""Focus crop (AST-01): where the window's focus should sit on a source frame.

Two requests per frame: `presence` (a Noul, question set `presence`, local-first) and `box` (a Box plus the same Noul
bundled, question set `crop_box`, Jev only). `crop_decide` gates on the presence p, expands Jev's box to the window
aspect and turns it into the focus helper's `(scale, x, y)` through `geometry`. Rows are never `auto`: a human or
the agent looks at the overlay. `crop_decide` reads answers only and never touches the network.
"""
from pathlib import Path

from core.policy import decision_policy
from core.rows import error_route, evidence_from, make_row
from jevlib.images import draw_rects, unfit_box
from jevlib.request import Request, box_frac, image_ref, load_qset, noul_p, thresholds_for
from modules.universal import geometry

MODULE_ID = "universal.assets"
DECISION = "AST-01"


def crop_requests(frame, want, sensitivity, qsets=None):
    """`[presence, box]` requests for one frame (the image Jev is sent). `want` names the whole region the shot needs.
    `qsets` maps a question set id (`presence`, `crop_box`) to a `QuestionSet` to use instead of the stored one."""
    qsets = qsets or {}
    presence_qs = qsets.get("presence") or load_qset("presence")
    box_qs = qsets.get("crop_box") or load_qset("crop_box")
    frame = Path(frame)
    ref = image_ref(frame, "img")
    presence = Request(presence_qs, {"image": ref}, {"vis": presence_qs.question("vis", want=want)},
                       {ref: frame}, sensitivity, "presence")
    box = Request(box_qs, {"image": ref},
                  {"box": box_qs.question("box", want=want), "vis": box_qs.question("vis", want=want)},
                  {ref: frame}, sensitivity, "box")
    return [presence, box]


def _gate(presence, box):
    """`(answer, p, thresholds)` for the presence gate, or None when neither answer holds a usable `vis`.

    A presence answer that came from Decider on its own gates with Decider's thresholds; one that came from Jev, or from
    Decider standing in (`backend_fallback`), does not. Then the Noul bundled with the Box gates, on Jev's thresholds.
    """
    if presence.ok and presence.backend == "local" and "backend_fallback" not in presence.route_reasons:
        p = noul_p(presence, "vis")
        if p is not None:
            return presence, p, thresholds_for(presence.request.qs, "local")
    if box.ok:
        p = noul_p(box, "vis")
        if p is not None:
            return box, p, thresholds_for(box.request.qs, "jev")
    return None


def crop_decide(answers, *, fit, body, scale_range, target, asset_id, t, want, policy):
    """The AST-01 row for `answers` = `[presence, box]` (the order `crop_requests` returns).

    `fit` is how the source frame became the sent one, `body` the window body `[W, H]`, `scale_range` the manifest's
    `focus_scale`. Lane is always `agent`. Value is `{"found": True, ...}` with the expanded rectangle in source px
    (`box_px`), Jev's tight box as fractions of the source (`box`) and the focus arguments (`clamped` lists what had to
    move: `scale`, `x`, `y`); `{"found": False, ...}` when the gate p is under `drop_below`; `found` None when no answer
    was usable. The row's evidence is the Box answer when a box was used, otherwise the answer the gate p came from.
    """
    presence, box_ans = answers
    entry = decision_policy(policy, DECISION)
    base = {"asset": asset_id, "t": t, "want": want}

    def carry(own):
        """The row's `route_reasons`: its own, then those of the presence and box answers (`backend_fallback` when
        Decider was down for the presence request, even though the gate then used the box answer's `vis`), de-duplicated."""
        return list(dict.fromkeys([*own, *presence.route_reasons, *box_ans.route_reasons]))

    def unusable(answer):
        lane, reasons = error_route(answer) if not answer.ok else ("agent", ["jev_error"])
        return make_row(decision=DECISION, module=MODULE_ID, target=target, value={"found": None, **base}, p=None,
                        confidence=None, lane=lane, route_reasons=carry(reasons), evidence=evidence_from(answer, entry))

    def row(value, p, reason, answer):
        return make_row(decision=DECISION, module=MODULE_ID, target=target, value=value, p=p,
                        confidence={"metric": "box_presence", "value": p}, lane="agent", route_reasons=carry([reason]),
                        evidence=evidence_from(answer, entry))

    gate = _gate(presence, box_ans)
    if gate is None:
        return unusable(next((a for a in (box_ans, presence) if not a.ok), box_ans))
    gate_answer, p, th = gate
    if p < th["drop_below"]:
        return row({"found": False, **base}, p, "presence_low", gate_answer)

    frac = box_frac(box_ans, "box") if box_ans.ok else None
    if frac is None:
        return unusable(box_ans)
    tight = unfit_box(frac, fit)
    if tight[2] <= tight[0] or tight[3] <= tight[1]:
        return unusable(box_ans)
    rect = geometry.expand(tight, body[0] / body[1], fit.src_w, fit.src_h)
    focus = geometry.to_focus(fit.src_w, fit.src_h, body, rect, scale_range)
    value = {
        "found": True, **base,
        "box_px": [round(v, 1) for v in rect],
        "box": {"left": round(tight[0] / fit.src_w, 4), "top": round(tight[1] / fit.src_h, 4),
                "right": round(tight[2] / fit.src_w, 4), "bottom": round(tight[3] / fit.src_h, 4)},
        "focus": {"scale": round(focus["scale"], 3), "x": round(focus["x"], 1), "y": round(focus["y"], 1)},
        "body": list(body),
        "clamped": focus["clamped"],
    }
    return row(value, p, "above_auto" if p >= th["propose_at"] else "in_review_band", box_ans)


def overlay(frame, row, fit, out):
    """Write `out`: the sent `frame` with the row's expanded rectangle in red, labelled with the scale and p. A row with
    no box (not found, or no usable answer) gives the plain frame. `fit` maps the row's source px onto the sent frame."""
    rects = []
    value = row["value"]
    if value.get("found"):
        pl, pt, pr, pb = fit.pad
        kx, ky = (fit.w - pl - pr) / fit.src_w, (fit.h - pt - pb) / fit.src_h
        x0, y0, x1, y1 = value["box_px"]
        label = f"{DECISION} S={value['focus']['scale']:.2f} P={row['p']:.2f}"
        rects.append(((pl + x0 * kx, pt + y0 * ky, pl + x1 * kx, pt + y1 * ky), label))
    return draw_rects(frame, rects, out)
