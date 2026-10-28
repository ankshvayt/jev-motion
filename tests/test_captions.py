"""Tests for modules.universal.captions: the highlight budget (CAP-02) and the span (CAP-03). From the tool folder:

    python3 -m unittest tests.test_captions -q

Offline: answers are built by hand from tests/fixtures/captions/canned.json, or come from a Client over a FakeBackend.
"""
import copy
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from core.policy import load_policy
from jevlib.cache import Cache
from jevlib.client import Client
from jevlib.fake import FakeBackend, choice, noul
from jevlib.request import Answer, load_qset
from modules.universal.captions import CaptionsModule, build_request, rank, spans

FIXTURES = Path(__file__).resolve().parent / "fixtures"

SPEC = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
KIT = json.loads((FIXTURES / "manifest_min.json").read_text(encoding="utf-8"))
CANNED = json.loads((FIXTURES / "captions" / "canned.json").read_text(encoding="utf-8"))
POLICY = load_policy()
GROUP_IDS = [g["id"] for g in SPEC["units"]["caption_groups"]]
MODULE = CaptionsModule()


def make_spec(density=0.25):
    spec = copy.deepcopy(SPEC)
    spec["constraints"]["highlight"]["density"] = density
    return spec


def synthetic_spec(n_groups, density=0.25):
    """`n_groups` caption groups of two words each ("alpha<g>", "beta<g>"); only what the captions module reads."""
    words, groups = [], []
    for g in range(n_groups):
        words += [{"id": f"w{2 * g}", "text": f"alpha{g}"}, {"id": f"w{2 * g + 1}", "text": f"beta{g}"}]
        groups.append({"id": f"cg{g}", "words": [f"w{2 * g}", f"w{2 * g + 1}"], "start": float(g), "end": g + 1.0})
    text = " ".join(w["text"] for w in words)
    return {"transcript": {"text": text, "words": words}, "units": {"caption_groups": groups},
            "constraints": {"highlight": {"density": density}}}


def canned_table(req, need2=None):
    """The answers a request over the fixture's four groups gets: `need2` scores (default from the fixture) and the
    forward or reverse pick probabilities, padded with 0.0 for every option the fixture does not list."""
    reverse = req.tag.endswith(":rev")
    table = {}
    for i, cg in enumerate(GROUP_IDS):
        canned = CANNED[cg]
        options = list(req.questions[f"pick_{i}"]["criteria"])
        probs = {o: 0.0 for o in options}
        listed = canned["reverse" if reverse else "forward"]
        assert set(listed) <= set(options), (cg, sorted(set(listed) - set(options)))
        probs.update(listed)
        table[f"need_{i}"] = noul(canned["need"])
        table[f"need2_{i}"] = noul(canned["need2"] if need2 is None else need2[i])
        table[f"pick_{i}"] = choice(probs)
    return table


def answer(req, table, **fields):
    base = dict(request=req, answers={q: table[q] for q in req.questions}, backend="jev", model="jev-1.13.0",
                model_resolved=None, input_tokens=100, latency_s=0.0, cached=False, request_sha256="ab" * 32)
    base.update(fields)
    return Answer(**base)


def failed(req, error="jev_unavailable"):
    return Answer(request=req, answers=None, backend="none", model="", model_resolved=None, input_tokens=0, latency_s=0.0,
                  cached=False, request_sha256="", error=error, detail="down", route_reasons=[])


def decide_with(spec, need2=None, tweak=None):
    """requests() -> canned answers (optionally tweaked: `tweak(req, table)`) -> decide(). Returns (rows by id, reqs)."""
    reqs = MODULE.requests(spec, KIT, POLICY)
    answers = []
    for req in reqs:
        table = canned_table(req, need2)
        if tweak:
            tweak(req, table)
        answers.append(answer(req, table))
    rows = MODULE.decide(spec, KIT, answers, POLICY)
    return {r["id"]: r for r in rows}, reqs


class SpanTests(unittest.TestCase):
    def test_spans_ids_and_order(self):
        got = spans(["one", "invoice", "now"])
        self.assertEqual(got, [
            ("h0", (0, 0), "one"), ("h1", (0, 1), "one invoice"), ("h2", (0, 2), "one invoice now"),
            ("h3", (1, 1), "invoice"), ("h4", (1, 2), "invoice now"), ("h5", (2, 2), "now")])

    def test_spans_max_span_and_empty(self):
        self.assertEqual([t for _i, _s, t in spans(["a", "b", "c"], max_span=1)], ["a", "b", "c"])
        self.assertEqual(spans([]), [])

    def test_rank_ties_keep_input_order(self):
        self.assertEqual(rank([0.5, 0.9, 0.5, 0.9]), [1, 3, 0, 2])


class RequestTests(unittest.TestCase):
    def test_reverse_keeps_none_last(self):
        qs = load_qset("highlight")
        batch = [{"text": "one invoice now", "words": ["one", "invoice", "now"]},
                 {"text": "in prompt", "words": ["in", "prompt"]}]
        fwd = build_request(qs, "story", batch, reverse=False)
        rev = build_request(qs, "story", batch, reverse=True)
        for i in (0, 1):
            f, r = fwd.questions[f"pick_{i}"]["criteria"], rev.questions[f"pick_{i}"]["criteria"]
            self.assertEqual(list(f)[-1], "none")
            self.assertEqual(list(r)[-1], "none")
            self.assertEqual(list(r)[:-1], list(reversed(list(f)[:-1])))      # ids stay with their span
            self.assertEqual(dict(f), dict(r))
            self.assertEqual(fwd.questions[f"need2_{i}"], rev.questions[f"need2_{i}"])
        self.assertEqual(list(fwd.questions["pick_0"]["criteria"])[:6], ["h0", "h1", "h2", "h3", "h4", "h5"])
        self.assertEqual(fwd.state, rev.state)
        self.assertEqual(fwd.state["groups"], [{"text": "one invoice now"}, {"text": "in prompt"}])

    def test_questions_per_group_and_substitution(self):
        qs = load_qset("highlight")
        req = build_request(qs, "story", [{"text": "a b", "words": ["a", "b"]}, {"text": "c", "words": ["c"]}], False)
        self.assertEqual(list(req.questions), ["need_0", "need2_0", "pick_0", "need_1", "need2_1", "pick_1"])
        self.assertIn("`groups[1].text`", req.questions["need_1"]["instructions"])
        self.assertIn("`groups[1].text`", req.questions["need2_1"]["instructions"]["question"])
        self.assertIn("`groups[1].text`", req.questions["pick_1"]["instructions"])
        self.assertNotIn("{i}", json.dumps(req.questions))
        self.assertEqual(req.sensitivity, "public")
        self.assertEqual(req.images, {})

    def test_requests_are_fwd_rev_pairs_per_batch(self):
        reqs = MODULE.requests(make_spec(), KIT, POLICY)
        self.assertEqual([r.tag for r in reqs], ["b0:fwd", "b0:rev"])
        self.assertEqual([g["text"] for g in reqs[0].state["groups"]],
                         ["Sort every invoice into folders", "in one click.", "In my test of 50 invoices,",
                          "it got 47 right."])
        self.assertEqual(reqs[0].state["story_transcript"], SPEC["transcript"]["text"])
        self.assertNotEqual(list(reqs[0].questions["pick_0"]["criteria"]), list(reqs[1].questions["pick_0"]["criteria"]))

    def test_requests_split_into_batches_of_the_qset(self):
        reqs = MODULE.requests(synthetic_spec(9), KIT, POLICY)
        self.assertEqual([r.tag for r in reqs], ["b0:fwd", "b0:rev", "b1:fwd", "b1:rev", "b2:fwd", "b2:rev"])
        self.assertEqual([len(r.state["groups"]) for r in reqs], [4, 4, 4, 4, 1, 1])
        self.assertEqual(reqs[4].state["groups"], [{"text": "alpha8 beta8"}])
        self.assertEqual(list(reqs[4].questions), ["need_0", "need2_0", "pick_0"])

    def test_story_is_the_window_around_the_batch_over_sixty_groups(self):
        spec = synthetic_spec(70)
        reqs = MODULE.requests(spec, KIT, POLICY)

        def text(lo, hi):
            return " ".join(f"alpha{g} beta{g}" for g in range(lo, hi))
        self.assertEqual(reqs[0].state["story_transcript"], text(0, 10))        # groups 0-3, plus 6 after
        self.assertEqual(reqs[10].state["story_transcript"], text(14, 30))      # batch 5: groups 20-23, 6 either side
        self.assertEqual(reqs[-1].state["story_transcript"], text(62, 70))      # last batch: groups 68-69
        self.assertEqual(reqs[0].state["story_transcript"], reqs[1].state["story_transcript"])
        at_sixty = MODULE.requests(synthetic_spec(60), KIT, POLICY)
        self.assertEqual(at_sixty[0].state["story_transcript"], synthetic_spec(60)["transcript"]["text"])

    def test_story_falls_back_to_the_joined_words(self):
        spec = make_spec()
        del spec["transcript"]["text"]
        reqs = MODULE.requests(spec, KIT, POLICY)
        self.assertEqual(reqs[0].state["story_transcript"], " ".join(w["text"] for w in SPEC["transcript"]["words"]))

    def test_applies(self):
        self.assertTrue(MODULE.applies(SPEC, KIT))
        self.assertFalse(MODULE.applies({"units": {"caption_groups": []}}, KIT))
        self.assertFalse(MODULE.applies({}, KIT))

    def test_module_identity(self):
        self.assertEqual((MODULE.id, MODULE.decisions, MODULE.QSET), ("universal.captions", ("CAP-02", "CAP-03"), "highlight"))


class DecideTests(unittest.TestCase):
    def test_budget_and_band_lanes(self):
        rows, _ = decide_with(make_spec(0.25), need2=[0.95, 0.90, 0.40, 0.10])      # K = 1, K-th score .95
        cap02 = {t: rows[f"CAP-02:{t}"] for t in GROUP_IDS if f"CAP-02:{t}" in rows}
        self.assertEqual(sorted(cap02), ["cg0", "cg1"])
        self.assertEqual(cap02["cg0"]["lane"], "agent")                              # top K, but it is the K-th itself
        self.assertEqual(cap02["cg0"]["route_reasons"], ["budget_edge"])
        self.assertEqual(cap02["cg0"]["value"], {"highlight": True, "rank": 1, "k": 1})
        self.assertEqual(cap02["cg1"]["lane"], "agent")                              # within .10 of .95
        self.assertEqual(cap02["cg1"]["route_reasons"], ["budget_edge"])
        self.assertEqual(cap02["cg1"]["value"], {"highlight": False, "rank": 2, "k": 1})
        self.assertEqual(sorted(i for i in rows if i.startswith("CAP-03")), ["CAP-03:cg0", "CAP-03:cg1"])
        for cg in ("cg0", "cg1"):                                                     # both parents are in the agent's lane
            self.assertEqual((rows[f"CAP-03:{cg}"]["lane"], rows[f"CAP-03:{cg}"]["route_reasons"]), ("agent", ["in_review_band"]))

    def test_auto_when_clear_of_band(self):
        rows, _ = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10])       # K = 2, K-th score .80
        cg0, cg1 = rows["CAP-02:cg0"], rows["CAP-02:cg1"]
        self.assertEqual((cg0["lane"], cg0["route_reasons"]), ("auto", ["above_auto"]))
        self.assertEqual(cg0["value"], {"highlight": True, "rank": 1, "k": 2})
        self.assertEqual(cg0["p"], 0.99)
        self.assertEqual(cg0["confidence"]["metric"], "rank_margin")
        self.assertAlmostEqual(cg0["confidence"]["value"], 0.19)
        self.assertEqual((cg1["lane"], cg1["route_reasons"]), ("agent", ["budget_edge"]))   # the K-th is in its own band
        self.assertAlmostEqual(cg1["confidence"]["value"], 0.0)
        self.assertNotIn("CAP-02:cg2", rows)
        self.assertNotIn("CAP-02:cg3", rows)

    def test_target_and_evidence(self):
        rows, reqs = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10])
        cap02, cap03 = rows["CAP-02:cg0"], rows["CAP-03:cg0"]
        self.assertEqual(cap02["target"], {"kind": "caption_group", "id": "cg0", "t": [0.0, 1.5], "words": ["w0", "w4"]})
        self.assertEqual(cap03["target"], cap02["target"])
        self.assertEqual((cap02["module"], cap03["module"]), ("universal.captions",) * 2)
        self.assertEqual((cap02["status"], cap02["resolved"]), ("proposed", None))
        for row, order in ((cap02, None), (cap03, "agree")):
            ev = row["evidence"]
            self.assertEqual((ev["label"], ev["qset"], ev["qset_version"], ev["backend"], ev["model"], ev["order_check"]),
                             ("Default", "highlight", 1, "jev", "jev-1.13.0", order))
            self.assertEqual(ev["request_sha256"], "ab" * 32)
        self.assertEqual(cap02["evidence"]["ref"], POLICY["decisions"]["CAP-02"]["evidence"]["ref"])
        self.assertEqual(cap03["evidence"]["ref"], POLICY["decisions"]["CAP-03"]["evidence"]["ref"])
        self.assertEqual(len(reqs), 2)

    def test_cap03_auto_when_both_orders_agree(self):
        rows, _ = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10])
        row = rows["CAP-03:cg0"]
        self.assertEqual((row["lane"], row["route_reasons"]), ("auto", ["above_auto", "order_agree"]))
        self.assertEqual(row["value"], {"span": ["w4", "w4"], "text": "folders"})
        self.assertEqual(row["p"], 0.55)
        self.assertEqual(row["confidence"]["metric"], "choice_conf")
        self.assertAlmostEqual(row["confidence"]["value"], (0.55 - 1 / 13) / (1 - 1 / 13))   # 12 spans + none
        self.assertEqual([a["value"]["text"] for a in row["alternatives"]], ["folders", "invoice", "invoice into folders"])
        self.assertEqual([a["p"] for a in row["alternatives"]], [0.55, 0.25, 0.10])
        self.assertEqual(row["alternatives"][1]["value"]["span"], ["w2", "w2"])
        # cg1 is the K-th (its CAP-02 row is `agent`), so its span is shown to the agent and is not applied
        self.assertEqual(rows["CAP-03:cg1"]["value"], {"span": ["w6", "w7"], "text": "one click."})
        self.assertEqual((rows["CAP-03:cg1"]["lane"], rows["CAP-03:cg1"]["route_reasons"]), ("agent", ["in_review_band"]))
        self.assertNotIn("CAP-03:cg2", rows)

    def test_cap03_order_flip_goes_to_agent(self):
        rows, _ = decide_with(make_spec(0.75), need2=[0.99, 0.80, 0.50, 0.10])      # K = 3: cg2 is the K-th
        flip = rows["CAP-03:cg2"]                                                     # forward h13, reverse h12
        self.assertEqual((flip["lane"], flip["route_reasons"]), ("agent", ["in_review_band", "order_flip"]))   # its parent is the K-th
        self.assertEqual(flip["value"], {"span": ["w12", "w13"], "text": "50 invoices,"})   # the forward pick
        self.assertEqual(flip["p"], 0.45)
        self.assertEqual(flip["evidence"]["order_check"], "flip")
        self.assertEqual([a["value"]["text"] for a in flip["alternatives"]], ["50 invoices,", "50", "of 50 invoices,"])
        self.assertEqual(rows["CAP-03:cg0"]["lane"], "auto")
        self.assertEqual(rows["CAP-03:cg1"]["lane"], "auto")
        self.assertNotIn("CAP-03:cg3", rows)

    def test_cap03_order_flip_with_auto_parent_carries_only_order_flip(self):
        rows, _ = decide_with(make_spec(0.5), need2=[0.60, 0.30, 0.99, 0.10])       # K = 2, K-th .60: cg2 is clear of it
        self.assertEqual(rows["CAP-02:cg2"]["lane"], "auto")
        flip = rows["CAP-03:cg2"]
        self.assertEqual((flip["lane"], flip["route_reasons"]), ("agent", ["order_flip"]))

    def test_no_span_row_when_not_highlighted(self):
        rows, _ = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10])       # cg2, cg3: not in the top K, not in the band
        for cg in ("cg2", "cg3"):
            self.assertNotIn(f"CAP-02:{cg}", rows)
            self.assertNotIn(f"CAP-03:{cg}", rows)
        # in every case below a span row has a parent that is highlighted or in the agent's lane, and never stands alone
        for density, need2 in ((0.25, [0.95, 0.90, 0.40, 0.10]), (0.5, [0.99, 0.80, 0.30, 0.10]), (0.75, [0.99, 0.80, 0.50, 0.10]),
                               (0.5, [0.60, 0.30, 0.99, 0.10]), (1.0, [0.9, 0.5, 0.3, 0.1]), (0.25, [0.5, 0.5, 0.5, 0.5])):
            with self.subTest(density=density, need2=need2):
                rows, _ = decide_with(make_spec(density), need2=need2)
                for rid, row in rows.items():
                    if row["decision"] == "CAP-03":
                        parent = rows[f"CAP-02:{row['target']['id']}"]
                        self.assertTrue(parent["value"]["highlight"] is True or parent["lane"] == "agent", rid)

    def test_span_row_present_for_band_group_in_agent_lane(self):
        rows, _ = decide_with(make_spec(0.25), need2=[0.95, 0.90, 0.40, 0.10])      # K = 1; cg1 is in the band but not kept
        parent, span = rows["CAP-02:cg1"], rows["CAP-03:cg1"]
        self.assertEqual((parent["value"]["highlight"], parent["lane"]), (False, "agent"))
        self.assertEqual((span["lane"], span["route_reasons"]), ("agent", ["in_review_band"]))   # orders agree, still not auto
        self.assertEqual(span["value"], {"span": ["w6", "w7"], "text": "one click."})            # the agent sees the suggestion
        self.assertEqual(span["evidence"]["order_check"], "agree")
        self.assertEqual(rows["CAP-03:cg0"]["lane"], "agent")                        # the K-th: highlight on, parent in the band

    def test_captions_row_carries_backend_fallback(self):
        # Forward and reverse answers that carry route_reasons: CAP-02 takes the forward answer's, CAP-03 the forward's then
        # the reverse's, each after the row's own reasons and with repeats removed. No lane changes.
        spec = make_spec(0.5)
        reqs = MODULE.requests(spec, KIT, POLICY)
        answers = []
        for req in reqs:
            reasons = ["backend_fallback"] if req.tag.endswith(":fwd") else ["backend_fallback", "quarantined"]
            answers.append(answer(req, canned_table(req, [0.99, 0.80, 0.30, 0.10]), route_reasons=reasons))
        rows = {r["id"]: r for r in MODULE.decide(spec, KIT, answers, POLICY)}
        got = {rid: (r["lane"], r["route_reasons"]) for rid, r in rows.items()}
        self.assertEqual(got["CAP-02:cg0"], ("auto", ["above_auto", "backend_fallback"]))
        self.assertEqual(got["CAP-03:cg0"], ("auto", ["above_auto", "order_agree", "backend_fallback", "quarantined"]))
        self.assertEqual(got["CAP-02:cg1"], ("agent", ["budget_edge", "backend_fallback"]))
        self.assertEqual(got["CAP-03:cg1"], ("agent", ["in_review_band", "backend_fallback", "quarantined"]))
        # A reverse answer that failed: its error leads, then the notices of both answers.
        bad = failed(reqs[1])
        bad.route_reasons = ["backend_fallback"]
        rows = {r["id"]: r for r in MODULE.decide(spec, KIT, [answers[0], bad], POLICY)}
        self.assertEqual((rows["CAP-03:cg0"]["lane"], rows["CAP-03:cg0"]["route_reasons"]),
                         ("agent", ["jev_unavailable", "backend_fallback"]))

    def test_span_row_when_parent_errored_carries_jev_error(self):
        def no_score(req, table):
            table["need2_0"] = {}                     # the answer has no `need2` for group 0
        rows, _ = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10], tweak=no_score)
        parent, span = rows["CAP-02:cg0"], rows["CAP-03:cg0"]
        self.assertEqual((parent["lane"], parent["value"], parent["route_reasons"]), ("agent", {"highlight": None}, ["jev_error"]))
        self.assertEqual((span["lane"], span["route_reasons"]), ("agent", ["jev_error"]))
        self.assertEqual(span["value"]["text"], "folders")

    def test_span_auto_only_when_parent_auto_true(self):
        # parent `auto` with the highlight on, both orders agree: auto
        rows, _ = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10])
        self.assertEqual((rows["CAP-02:cg0"]["lane"], rows["CAP-02:cg0"]["value"]["highlight"]), ("auto", True))
        self.assertEqual((rows["CAP-03:cg0"]["lane"], rows["CAP-03:cg0"]["route_reasons"]),
                         ("auto", ["above_auto", "order_agree"]))
        # parent `agent` (the K-th, highlight on): the span agrees in both orders and still is not auto
        self.assertEqual((rows["CAP-02:cg1"]["lane"], rows["CAP-02:cg1"]["value"]["highlight"]), ("agent", True))
        self.assertEqual(rows["CAP-03:cg1"]["lane"], "agent")
        # the rule over every combination: an `auto` span has an `auto` parent with the highlight on, and agreeing orders
        for density, need2 in ((0.25, [0.95, 0.90, 0.40, 0.10]), (0.5, [0.99, 0.80, 0.30, 0.10]), (0.75, [0.99, 0.80, 0.50, 0.10]),
                               (0.5, [0.60, 0.30, 0.99, 0.10]), (1.0, [0.9, 0.5, 0.3, 0.1]), (0.5, [0.99, 0.98, 0.10, 0.05])):
            with self.subTest(density=density, need2=need2):
                rows, _ = decide_with(make_spec(density), need2=need2)
                for rid, row in rows.items():
                    if row["decision"] == "CAP-03" and row["lane"] == "auto":
                        parent = rows[f"CAP-02:{row['target']['id']}"]
                        self.assertEqual((parent["lane"], parent["value"]["highlight"]), ("auto", True), rid)
                        self.assertEqual(row["evidence"]["order_check"], "agree", rid)
                    if row["decision"] == "CAP-03" and rows[f"CAP-02:{row['target']['id']}"]["lane"] != "auto":
                        self.assertEqual(row["lane"], "agent", rid)
                        self.assertIn(row["route_reasons"][0], ("in_review_band", "jev_error"), rid)

    def test_cap03_none_ignored(self):
        def none_leads(req, table):
            probs = table["pick_0"]["probabilities"]
            probs["none"] = 0.60                      # none is now the most probable option, in both orders
            table["pick_0"] = choice(probs)
        rows, _ = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10], tweak=none_leads)
        row = rows["CAP-03:cg0"]
        self.assertEqual(row["value"]["text"], "folders")
        self.assertEqual(row["p"], 0.55)              # the best option that is not none
        self.assertEqual(row["lane"], "auto")
        self.assertNotIn("none", json.dumps(row["alternatives"]))

    def test_cap03_without_pick_probabilities_goes_to_agent_without_a_span(self):
        def no_pick(req, table):
            table["pick_0"] = {"probabilities": {}}
        rows, _ = decide_with(make_spec(0.5), need2=[0.99, 0.80, 0.30, 0.10], tweak=no_pick)
        row = rows["CAP-03:cg0"]
        self.assertEqual(row["value"], {"span": None, "text": None})
        self.assertEqual((row["lane"], row["route_reasons"], row["p"]), ("agent", ["jev_error"], None))
        self.assertEqual(rows["CAP-02:cg0"]["lane"], "auto")

    def test_text_copied_from_words_not_from_jev(self):
        spec = make_spec(0.5)
        spec["transcript"]["words"][4]["text"] = "folders!"

        def jev_invents_text(req, table):
            for i in range(4):
                table[f"pick_{i}"]["text"] = "INVENTED BY JEV"
        rows, reqs = decide_with(spec, need2=[0.99, 0.80, 0.30, 0.10], tweak=jev_invents_text)
        self.assertEqual(rows["CAP-03:cg0"]["value"], {"span": ["w4", "w4"], "text": "folders!"})
        self.assertIn("folders!", reqs[0].questions["pick_0"]["criteria"].values())
        self.assertNotIn("INVENTED", json.dumps(list(rows.values())))

    def test_failed_forward_answer_routes_to_agent(self):
        spec = make_spec()

        def responder(req, model):
            raise AssertionError("an unhealthy backend must not be sent anything")
        with tempfile.TemporaryDirectory() as tmp:
            client = Client({"jev": FakeBackend("jev", responder, healthy=False)}, Cache(tmp), log=lambda *_: None)
            answers = client.run_many(MODULE.requests(spec, KIT, POLICY))
        rows = MODULE.decide(spec, KIT, answers, POLICY)
        self.assertEqual([r["id"] for r in rows], [f"CAP-02:{t}" for t in GROUP_IDS])   # no CAP-03 without a forward answer
        for row in rows:
            self.assertEqual(row["lane"], "agent")
            self.assertIn("jev_unavailable", row["route_reasons"])
            self.assertEqual(row["value"], {"highlight": None})
            self.assertIsNone(row["p"])
            self.assertIsNone(row["confidence"])
            self.assertEqual(row["evidence"]["backend"], "none")

    def test_failed_reverse_answer_keeps_the_span_for_agent(self):
        spec = make_spec(0.5)
        reqs = MODULE.requests(spec, KIT, POLICY)
        need2 = [0.99, 0.80, 0.30, 0.10]
        answers = [answer(reqs[0], canned_table(reqs[0], need2)), failed(reqs[1])]
        rows = {r["id"]: r for r in MODULE.decide(spec, KIT, answers, POLICY)}
        self.assertEqual(rows["CAP-02:cg0"]["lane"], "auto")                          # CAP-02 needs the forward answer only
        span = rows["CAP-03:cg0"]
        self.assertEqual((span["lane"], span["route_reasons"]), ("agent", ["jev_unavailable"]))
        self.assertEqual(span["value"]["text"], "folders")
        self.assertIsNone(span["evidence"]["order_check"])

    def test_unscored_groups_do_not_rank_but_count_toward_k(self):
        spec = synthetic_spec(8, density=0.25)                                        # K = 2 of 8, two batches
        reqs = MODULE.requests(spec, KIT, POLICY)

        def table(req, scores):
            n = len(req.state["groups"])
            out = {}
            for i in range(n):
                options = list(req.questions[f"pick_{i}"]["criteria"])
                out[f"need_{i}"], out[f"need2_{i}"] = noul(0.5), noul(scores[i])
                out[f"pick_{i}"] = choice({o: (1.0 if o == "h0" else 0.0) for o in options})
            return out
        answers = [answer(reqs[0], table(reqs[0], [0.9, 0.7, 0.65, 0.3])), answer(reqs[1], table(reqs[1], [0.9] * 4)),
                   failed(reqs[2]), failed(reqs[3])]
        rows = {r["id"]: r for r in MODULE.decide(spec, KIT, answers, POLICY)}
        for g in (4, 5, 6, 7):
            self.assertEqual(rows[f"CAP-02:cg{g}"]["value"], {"highlight": None})
        self.assertEqual(rows["CAP-02:cg0"]["value"], {"highlight": True, "rank": 1, "k": 2})
        self.assertEqual(rows["CAP-02:cg1"]["value"], {"highlight": True, "rank": 2, "k": 2})
        self.assertEqual(rows["CAP-02:cg1"]["route_reasons"], ["budget_edge"])        # the K-th score .7
        self.assertEqual(rows["CAP-02:cg2"]["value"], {"highlight": False, "rank": 3, "k": 2})   # .65 is within .10 of .7
        self.assertNotIn("CAP-02:cg3", rows)

    def test_decide_rejects_a_wrong_number_of_answers(self):
        reqs = MODULE.requests(make_spec(), KIT, POLICY)
        with self.assertRaises(ValueError):
            MODULE.decide(make_spec(), KIT, [answer(reqs[0], canned_table(reqs[0]))], POLICY)

    def test_decide_makes_no_network_call(self):
        spec = make_spec(0.5)
        reqs = MODULE.requests(spec, KIT, POLICY)
        answers = [answer(r, canned_table(r, [0.99, 0.80, 0.30, 0.10])) for r in reqs]
        with mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network call")):
            rows = MODULE.decide(spec, KIT, answers, POLICY)
        self.assertEqual(len(rows), 4)

    def test_through_a_client_over_a_fake_backend(self):
        spec = make_spec(0.5)

        def responder(req, model):
            table = canned_table(req, [0.99, 0.80, 0.30, 0.10])
            return {"answers": {q: table[q] for q in req.questions}, "model": model, "input_tokens": 100, "latency_s": 0.0}
        backend = FakeBackend("jev", responder)
        with tempfile.TemporaryDirectory() as tmp:
            client = Client({"jev": backend}, Cache(tmp), log=lambda *_: None)
            reqs = MODULE.requests(spec, KIT, POLICY)
            answers = client.run_many(reqs)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual({m for _r, m in backend.calls}, {"jev-1.13.0"})
        rows, _ = decide_with(spec, need2=[0.99, 0.80, 0.30, 0.10])
        got = {r["id"]: r for r in MODULE.decide(spec, KIT, answers, POLICY)}
        self.assertEqual(sorted(got), sorted(rows))
        for rid, row in got.items():
            self.assertEqual((row["lane"], row["value"]), (rows[rid]["lane"], rows[rid]["value"]), rid)
            self.assertEqual(row["evidence"]["backend"], "jev")
            self.assertNotEqual(row["evidence"]["request_sha256"], "")


if __name__ == "__main__":
    unittest.main()
