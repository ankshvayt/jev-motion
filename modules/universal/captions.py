"""Captions: CAP-02 which caption groups get the highlight, CAP-03 which words in them.

One request pair per batch of caption groups: the forward request, then the same request with the CAP-03 options reversed
(the order check). Each group is asked three questions: `need` (unused by the policy, kept
because answers move when other questions in the request change), `need2` (the CAP-02 score) and `pick` (the CAP-03 span).

CAP-02 is a budget, not a threshold: groups are ranked by forward `need2` and the top K are kept, K = highlight
density x caption groups. A Noul is not a degree, so the score only orders the groups. CAP-03 is the Choice over every
1-3 word span of the group, `none` ignored. A span row exists only for a group that is highlighted or that the agent is deciding,
and is `auto` only when its CAP-02 row is `auto` with the highlight on and both orders agree. `decide()` is pure and never
touches the network.
"""
import dataclasses

from core.module import Module
from core.policy import budget_k, decision_policy, in_band
from core.rows import error_route, evidence_from, make_row
from jevlib.request import Request, choice_conf, choice_probs, choice_top, load_qset, noul_p

QSET = "highlight"
MAX_SPAN = 3
WINDOW_FROM = 60   # more caption groups than this and a request carries a window of the transcript, not all of it
WINDOW = 6         # caption groups of context either side of a batch


def spans(words: list[str], max_span: int = MAX_SPAN) -> list[tuple[str, tuple[int, int], str]]:
    """`[(id, (a, b), text)]` for every contiguous span of 1..`max_span` words, in reading order, original text kept."""
    out = []
    for a in range(len(words)):
        for n in range(1, max_span + 1):
            b = a + n - 1
            if b < len(words):
                out.append((f"h{len(out)}", (a, b), " ".join(words[a:b + 1])))
    return out


def build_request(qs, transcript_text: str, batch: list[dict], reverse: bool) -> Request:
    """One request over `batch` (items with `text` and `words`). Group `i` of the batch gets `need_{i}`, `need2_{i}` and
    `pick_{i}`; the pick criteria are the group's spans then `none`. `reverse` reverses the spans (ids stay with their span)
    and keeps `none` last. The Choice order is part of the request, so a reversed request is a different cache entry."""
    state = {"story_transcript": transcript_text, "groups": [{"text": g["text"]} for g in batch]}
    questions = {}
    for i, g in enumerate(batch):
        options = [(cid, text) for cid, _span, text in spans(g["words"])]
        if reverse:
            options.reverse()
        questions[f"need_{i}"] = qs.question("need_{i}", i=i)
        questions[f"need2_{i}"] = qs.question("need2_{i}", i=i)
        pick = qs.question("pick_{i}", i=i)
        criteria = dict(options)
        criteria["none"] = pick["criteria"]["none"]
        pick["criteria"] = criteria
        questions[f"pick_{i}"] = pick
    return Request(qs, state, questions)


def rank(scores: list[float]) -> list[int]:
    """Indices of `scores`, best first; equal scores keep their input order."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def _groups(spec):
    """The spec's caption groups with their words: id, times, first and last word id, every word id, word texts, text."""
    words = spec["transcript"]["words"]
    index = {w["id"]: k for k, w in enumerate(words)}
    out = []
    for g in spec["units"]["caption_groups"]:
        first, last = g["words"]
        if first not in index or last not in index:
            raise ValueError(f"caption group {g['id']}: word {first!r} or {last!r} is not in the transcript")
        run = words[index[first]:index[last] + 1]
        texts = [w["text"] for w in run]
        out.append({"id": g["id"], "t": [g["start"], g["end"]], "bounds": [first, last], "wids": [w["id"] for w in run],
                    "words": texts, "text": " ".join(texts)})
    return out


def _batches(groups, size):
    """`[(start, groups)]`: consecutive batches of at most `size` caption groups and the index each one starts at."""
    return [(s, groups[s:s + size]) for s in range(0, len(groups), size)]


def _story(spec, groups, start, size):
    """The `story_transcript` for one batch: the whole transcript, or with many groups the groups around the batch."""
    if len(groups) <= WINDOW_FROM:
        t = spec["transcript"]
        return t.get("text") or " ".join(w["text"] for w in t["words"])
    return " ".join(g["text"] for g in groups[max(0, start - WINDOW):start + size + WINDOW])


def _carry(own, *answers):
    """A row's `route_reasons`: the module's own reasons, then those of each answer the row came from, de-duplicated."""
    return list(dict.fromkeys([*own, *(r for a in answers for r in a.route_reasons)]))


class CaptionsModule(Module):
    id = "universal.captions"
    decisions = ("CAP-02", "CAP-03")
    QSET = QSET

    def applies(self, spec, kit):
        return bool(spec.get("units", {}).get("caption_groups"))

    def requests(self, spec, kit, policy):
        """For each batch: the forward request (tag `b<k>:fwd`), then the reverse (tag `b<k>:rev`)."""
        qs = load_qset(self.QSET)
        groups = _groups(spec)
        out = []
        for k, (start, batch) in enumerate(_batches(groups, qs.batch)):
            story = _story(spec, groups, start, len(batch))
            for reverse, name in ((False, "fwd"), (True, "rev")):
                req = build_request(qs, story, batch, reverse)
                out.append(dataclasses.replace(req, tag=f"b{k}:{name}"))
        return out

    def decide(self, spec, kit, answers, policy):
        groups = _groups(spec)
        if not groups:
            return []
        qs = load_qset(self.QSET)
        batches = _batches(groups, qs.batch)
        if len(answers) != 2 * len(batches):
            raise ValueError(f"expected {2 * len(batches)} answers (forward and reverse for {len(batches)} batches), "
                             f"got {len(answers)}")
        band = policy["review_band"]
        cap02, cap03 = decision_policy(policy, "CAP-02"), decision_policy(policy, "CAP-03")

        # For each group: the forward and reverse answers of its batch and its index in that batch.
        asked = []
        for k, (_start, batch) in enumerate(batches):
            asked += [(answers[2 * k], answers[2 * k + 1], i) for i in range(len(batch))]
        need2 = [noul_p(fwd, f"need2_{i}") if fwd.ok else None for fwd, _rev, i in asked]

        # Rank the scored groups; K counts every caption group, scored or not.
        scored = [g for g, p in enumerate(need2) if p is not None]
        ranked = [scored[j] for j in rank([need2[g] for g in scored])]
        k_keep = budget_k(spec["constraints"]["highlight"]["density"], len(groups))
        top = set(ranked[:k_keep])
        place = {g: r + 1 for r, g in enumerate(ranked)}
        kth = need2[ranked[min(k_keep, len(ranked)) - 1]] if ranked else None

        rows = []
        for g, group in enumerate(groups):
            fwd, rev, i = asked[g]
            target = {"kind": "caption_group", "id": group["id"], "t": group["t"], "words": group["bounds"]}
            if need2[g] is None:
                lane, reasons = error_route(fwd) if not fwd.ok else ("agent", ["jev_error"])
                parent = make_row(decision="CAP-02", module=self.id, target=target, value={"highlight": None}, p=None,
                                  confidence=None, lane=lane, route_reasons=_carry(reasons, fwd),
                                  evidence=evidence_from(fwd, cap02))
            else:
                in_top = g in top
                near = in_band(need2[g], kth, band)
                if not (in_top or near):
                    continue
                auto = in_top and not near
                parent = make_row(
                    decision="CAP-02", module=self.id, target=target,
                    value={"highlight": in_top, "rank": place[g], "k": k_keep}, p=need2[g],
                    confidence={"metric": "rank_margin", "value": need2[g] - kth},
                    lane="auto" if auto else "agent",
                    route_reasons=_carry(["above_auto"] if auto else ["budget_edge"], fwd), evidence=evidence_from(fwd, cap02))
            rows.append(parent)
            # A span row only where the highlight is on or the agent is deciding it, and only when there is a forward answer
            # to take the suggested span from.
            if fwd.ok and (parent["value"]["highlight"] is True or parent["lane"] == "agent"):
                rows.append(self._span_row(group, target, fwd, rev, i, cap03, parent))
        return rows

    def _span_row(self, group, target, fwd, rev, i, entry, parent):
        """The CAP-03 row for a group whose CAP-02 row is `parent`. Text comes from the spec words only. It is `auto` only
        when the parent is `auto` with the highlight on and the two orders agree; every other span goes to the agent, with the
        parent's reason (`in_review_band`, or `jev_error` when the parent errored) ahead of the span's own."""
        by_id = {cid: (span, text) for cid, span, text in spans(group["words"])}

        def value_of(cid):
            (a, b), text = by_id[cid]
            return {"span": [group["wids"][a], group["wids"][b]], "text": text}

        qid = f"pick_{i}"
        pick, p = choice_top(fwd, qid)
        if pick not in by_id:
            value, confidence, alternatives, order = {"span": None, "text": None}, None, [], None
            p, lane, reasons = None, "agent", ["jev_error"]
        else:
            probs = choice_probs(fwd, qid)
            likely = sorted((cid for cid in probs if cid in by_id), key=lambda cid: -probs[cid])[:3]
            value, confidence = value_of(pick), {"metric": "choice_conf", "value": choice_conf(probs)}
            alternatives = [{"value": value_of(cid), "p": probs[cid]} for cid in likely]
            if not rev.ok:
                lane, reasons = "agent", [rev.error]
                order = None
            elif choice_top(rev, qid)[0] == pick:
                lane, reasons, order = "auto", ["above_auto", "order_agree"], "agree"
            else:
                lane, reasons, order = "agent", ["order_flip"], "flip"
        if not (parent["lane"] == "auto" and parent["value"]["highlight"] is True):
            lane = "agent"
            why = "jev_error" if parent["value"]["highlight"] is None else "in_review_band"
            reasons = [why] + [r for r in reasons if r not in ("above_auto", "order_agree")]
        return make_row(decision="CAP-03", module=self.id, target=target, value=value, p=p, confidence=confidence,
                        lane=lane, route_reasons=_carry(reasons, fwd, rev),
                        evidence=evidence_from(fwd, entry, order_check=order), alternatives=alternatives)
