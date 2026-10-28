"""Tests for modules.universal.claims: STY-06 (is a spoken number measured?) and the exact-quote half of QA-05.

Offline: answers are built by hand or come from FakeBackends through a Client with a temporary Cache. The spec is
tests/fixtures/spec_min.json; source texts and the 10 synthetic controls are in tests/fixtures/claims/.
"""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from core.policy import NEVER_AUTO, load_policy
from jevlib.cache import Cache
from jevlib.client import Client
from jevlib.fake import FakeBackend, by_qid, noul
from jevlib.request import Answer, load_qset
from modules.universal.claims import (ClaimsModule, has_number_hint, load_sources, normalise, numbers_in, quote_check,
                                      stat_requests)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CLAIMS = FIXTURES / "claims"
POLICY = load_policy()
SPEC = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
SENTENCE_2 = "In my test of 50 invoices, it got 47 right."


def _one_number_spec(root=CLAIMS):
    """The fixture spec with the number word "one" of sentence 1 replaced, so only sentence 2 has a number hint."""
    spec = copy.deepcopy(SPEC)
    spec["transcript"]["words"][6]["text"] = "a"
    spec["project"]["root"] = str(root)
    return spec


SPEC1 = _one_number_spec()


def _answer(req, has_number, measured, *, backend="local", reasons=()):
    answers = {"has_number": noul(has_number), "stat_is_measured": noul(measured)}
    return Answer(request=req, answers=answers, backend=backend, model="m", model_resolved=None, input_tokens=10,
                  latency_s=0.0, cached=False, request_sha256="ab" * 32, route_reasons=list(reasons))


def _failed(req, error="jev_unavailable", reasons=()):
    return Answer(request=req, answers=None, backend="none", model="", model_resolved=None, input_tokens=0,
                  latency_s=0.0, cached=False, request_sha256="", error=error, detail="down", route_reasons=list(reasons))


def _decide(measured, has_number=0.90, spec=SPEC1):
    module = ClaimsModule()
    reqs = module.requests(spec, {}, POLICY)
    return module.decide(spec, {}, [_answer(r, has_number, measured) for r in reqs], POLICY)


class HintTests(unittest.TestCase):
    def test_has_number_hint(self):
        self.assertTrue(has_number_hint("Now that's 100× productive"))
        self.assertTrue(has_number_hint("Not one, multiple."))
        self.assertFalse(has_number_hint("The folder is here"))

    def test_hint_triggers(self):
        for text in ("It got 47 right.", "Ten times better.", "Twenty-five tools", "a million reasons", "cut errors by half",
                     "it doubled, twice", "12 PERCENT"):
            self.assertTrue(has_number_hint(text), text)
        for text in ("Someone is here.", "Nothing to see", "Halfway there", "The folder is here"):
            self.assertFalse(has_number_hint(text), text)


class RequestTests(unittest.TestCase):
    def test_one_request_per_number_sentence(self):
        reqs = ClaimsModule().requests(SPEC, {}, POLICY)
        self.assertEqual(len(reqs), 2)   # sentence 1 has the number word "one", sentence 2 has digits
        self.assertEqual(reqs[1].state, {"scene": {"words": SENTENCE_2}})
        self.assertEqual([r.tag for r in reqs], ["sn0", "sn1"])
        for r in reqs:
            self.assertEqual(r.qs.id, "stat_check")
            self.assertEqual(list(r.questions), ["has_number", "stat_is_measured"])
            self.assertEqual(r.images, {})

    def test_sentence_without_a_number_hint_sends_no_request(self):
        reqs = ClaimsModule().requests(SPEC1, {}, POLICY)
        self.assertEqual([r.tag for r in reqs], ["sn1"])

    def test_applies_when_the_spec_has_sentences(self):
        self.assertTrue(ClaimsModule().applies(SPEC, {}))
        self.assertFalse(ClaimsModule().applies({"units": {"sentences": []}}, {}))
        self.assertFalse(ClaimsModule().applies({}, {}))

    def test_module_identity(self):
        m = ClaimsModule()
        self.assertEqual((m.id, m.decisions, m.QSET), ("universal.claims", ("STY-06", "QA-05"), "stat_check"))

    def test_controls_go_as_one_bundle_each(self):
        controls = json.loads((CLAIMS / "controls.json").read_text(encoding="utf-8"))
        self.assertEqual(len(controls), 10)
        self.assertEqual(sum(c["measured"] for c in controls), 5)
        qs = load_qset("stat_check")
        reqs = stat_requests([(f"c{i}", c["text"]) for i, c in enumerate(controls)], qs)
        self.assertEqual(len(reqs), 10)
        for c, r in zip(controls, reqs):
            self.assertTrue(has_number_hint(c["text"]), c["text"])
            self.assertEqual(r.state, {"scene": {"words": c["text"]}})
            self.assertEqual(r.questions, qs.questions)

    def test_stat_requests_loads_the_question_set_by_default(self):
        reqs = stat_requests([("sn1", SENTENCE_2)])
        self.assertEqual(reqs[0].qs, load_qset("stat_check"))


class QuestionSetTests(unittest.TestCase):
    def test_stat_check_set(self):
        qs = load_qset("stat_check")
        self.assertEqual((qs.version, qs.route, qs.backend, qs.order_check, qs.batch), (1, "local", "local", False, None))
        self.assertEqual(qs.models, {"jev": "jev-1.13.0", "local": "pplx-decider-flash-v1.2"})
        want = {"has_number": 0.50, "opinion_below": 0.30, "source_at": 0.70}
        self.assertEqual(qs.thresholds, {"jev": want, "local": want})
        self.assertEqual(qs.calibrated, {"local": {"app_version": "0.1.0", "model": "pplx-decider-flash-v1.2"}})
        self.assertTrue(qs.has_local_calibration())
        self.assertEqual(list(qs.questions), ["has_number", "stat_is_measured"])
        self.assertTrue(all(q["type"] == "noul" for q in qs.questions.values()))


class DecideTests(unittest.TestCase):
    def test_lanes(self):
        (opinion,) = _decide(0.05)
        self.assertEqual((opinion["decision"], opinion["lane"], opinion["value"]["class"]), ("STY-06", "agent", "opinion"))
        self.assertEqual(opinion["route_reasons"], ["never_auto"])

        sty, qa = _decide(0.85)
        self.assertEqual((sty["lane"], sty["value"]["class"]), ("human", "needs_source"))
        self.assertEqual(sty["route_reasons"], ["never_auto"])
        self.assertEqual((qa["decision"], qa["lane"]), ("QA-05", "human"))

        (unsure,) = _decide(0.50)
        self.assertEqual((unsure["lane"], unsure["value"]["class"]), ("agent", "unsure"))
        self.assertEqual(unsure["route_reasons"], ["in_review_band"])

        self.assertEqual(_decide(0.85, has_number=0.20), [])

    def test_threshold_edges(self):
        self.assertEqual(_decide(0.30)[0]["value"]["class"], "unsure")        # opinion_below is a strict "below"
        self.assertEqual(_decide(0.2999)[0]["value"]["class"], "opinion")
        self.assertEqual(_decide(0.70)[0]["value"]["class"], "needs_source")  # source_at is "at or above"
        self.assertEqual(_decide(0.6999)[0]["value"]["class"], "unsure")
        self.assertEqual(len(_decide(0.05, has_number=0.50)), 1)              # has_number is a strict "below"
        self.assertEqual(_decide(0.05, has_number=0.4999), [])

    def test_sty_row_shape(self):
        sty, _ = _decide(0.85)
        self.assertEqual(sty["id"], "STY-06:sn1")
        self.assertEqual(sty["module"], "universal.claims")
        self.assertEqual(sty["target"], {"kind": "sentence", "id": "sn1", "t": [2.4, 5.4], "words": ["w8", "w17"]})
        self.assertEqual(sty["value"], {"class": "needs_source", "has_number": 0.90, "stat_is_measured": 0.85})
        self.assertEqual((sty["p"], sty["confidence"]), (0.85, {"metric": "noul_p", "value": 0.85}))
        self.assertEqual((sty["status"], sty["resolved"], sty["alternatives"]), ("proposed", None, []))
        ev = sty["evidence"]
        self.assertEqual((ev["label"], ev["qset"], ev["qset_version"], ev["backend"]), ("Default", "stat_check", 1, "local"))
        self.assertEqual(ev["request_sha256"], "ab" * 32)

    def test_qa05_row_is_code_only(self):
        _, qa = _decide(0.85)
        self.assertEqual(qa["id"], "QA-05:sn1")
        self.assertEqual(qa["target"]["kind"], "claim")
        self.assertEqual((qa["p"], qa["confidence"], qa["lane"], qa["route_reasons"]), (None, None, "human", ["never_auto"]))
        self.assertEqual((qa["evidence"]["backend"], qa["evidence"]["request_sha256"], qa["evidence"]["qset"]),
                         ("none", None, None))
        # The fixture project holds the sentence word for word.
        self.assertEqual(qa["value"], {"match": "quote_found", "source": "a3", "numbers": ["50", "47"]})

    def test_qa05_without_a_readable_source(self):
        spec = _one_number_spec(root=CLAIMS / "no-such-project")
        _, qa = _decide(0.85, spec=spec)
        self.assertEqual(qa["value"], {"match": "no_sources", "source": None, "numbers": ["50", "47"]})

    def test_no_qa05_row_unless_needs_source(self):
        for measured in (0.0, 0.05, 0.2999, 0.30, 0.50, 0.6999):
            self.assertEqual([r["decision"] for r in _decide(measured)], ["STY-06"], measured)

    def test_failed_answer_routes_to_agent(self):
        module = ClaimsModule()
        reqs = module.requests(SPEC1, {}, POLICY)
        rows = module.decide(SPEC1, {}, [_failed(r, reasons=["backend_fallback"]) for r in reqs], POLICY)
        self.assertEqual(len(rows), 1)
        (row,) = rows
        self.assertEqual((row["decision"], row["lane"], row["value"]), ("STY-06", "agent", {"class": None}))
        self.assertEqual(row["route_reasons"], ["jev_unavailable", "backend_fallback"])
        self.assertEqual((row["p"], row["confidence"]), (None, None))
        self.assertEqual(row["evidence"]["backend"], "none")

    def test_never_auto_rows(self):
        self.assertTrue({"STY-06", "QA-05"} <= NEVER_AUTO)
        rows = []
        for measured in (0.0, 0.05, 0.30, 0.50, 0.70, 0.85, 1.0):
            rows += _decide(measured)
        module = ClaimsModule()
        rows += module.decide(SPEC1, {}, [_failed(r) for r in module.requests(SPEC1, {}, POLICY)], POLICY)
        self.assertTrue(rows)
        self.assertNotIn("auto", {r["lane"] for r in rows})

    def test_local_thresholds_used_when_local_answered(self):
        qs = replace(load_qset("stat_check"), thresholds={
            "jev": {"has_number": 0.50, "opinion_below": 0.30, "source_at": 0.70},
            "local": {"has_number": 0.50, "opinion_below": 0.60, "source_at": 0.90}})
        (req,) = stat_requests([("sn1", SENTENCE_2)], qs)
        module = ClaimsModule()
        (on_local,) = module.decide(SPEC1, {}, [_answer(req, 0.9, 0.50, backend="local")], POLICY)
        (on_jev,) = module.decide(SPEC1, {}, [_answer(req, 0.9, 0.50, backend="jev")], POLICY)
        self.assertEqual(on_local["value"]["class"], "opinion")    # 0.50 < the local opinion_below of 0.60
        self.assertEqual(on_jev["value"]["class"], "unsure")       # between Jev's 0.30 and 0.70

    def test_answers_must_match_the_requests(self):
        with self.assertRaises(ValueError):
            ClaimsModule().decide(SPEC, {}, [], POLICY)

    def test_decide_makes_no_network_call(self):
        reqs = ClaimsModule().requests(SPEC1, {}, POLICY)
        with mock.patch("socket.socket.connect", side_effect=AssertionError("network")) as connect, \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("network")) as urlopen:
            rows = ClaimsModule().decide(SPEC1, {}, [_answer(r, 0.9, 0.85) for r in reqs], POLICY)
        self.assertEqual(len(rows), 2)
        connect.assert_not_called()
        urlopen.assert_not_called()


class ThroughTheClientTests(unittest.TestCase):
    """Requests and answers over the real Client and two FakeBackends (no network)."""

    def _run(self, local, jev):
        with tempfile.TemporaryDirectory() as tmp:
            client = Client({"jev": jev, "local": local}, Cache(root=Path(tmp)), log=lambda message: None)
            module = ClaimsModule()
            answers = client.run_many(module.requests(SPEC1, {}, POLICY))
            return module.decide(SPEC1, {}, answers, POLICY)

    def test_decider_answers_the_set(self):
        table = {"has_number": noul(0.94), "stat_is_measured": noul(0.85)}
        local = FakeBackend("local", by_qid(table), version="0.1.0")
        jev = FakeBackend("jev", by_qid(table))
        rows = self._run(local, jev)
        self.assertEqual([r["id"] for r in rows], ["STY-06:sn1", "QA-05:sn1"])
        self.assertEqual(rows[0]["evidence"]["backend"], "local")
        self.assertEqual((len(local.calls), len(jev.calls)), (1, 0))

    def test_claims_row_carries_backend_fallback(self):
        # Decider down, Jev answers: the answers carry backend_fallback, so every row built from them must.
        for measured, want in ((0.85, [("STY-06", "human", ["never_auto", "backend_fallback"]),
                                       ("QA-05", "human", ["never_auto", "backend_fallback"])]),
                               (0.50, [("STY-06", "agent", ["in_review_band", "backend_fallback"])]),
                               (0.05, [("STY-06", "agent", ["never_auto", "backend_fallback"])])):
            table = {"has_number": noul(0.94), "stat_is_measured": noul(measured)}
            local = FakeBackend("local", by_qid(table), version="0.1.0", healthy=False)
            rows = self._run(local, FakeBackend("jev", by_qid(table)))
            self.assertEqual([(r["decision"], r["lane"], r["route_reasons"]) for r in rows], want, measured)
            self.assertEqual(rows[0]["evidence"]["backend"], "jev")

    def test_route_reasons_are_the_rows_own_then_the_answers_without_repeats(self):
        (req,) = ClaimsModule().requests(SPEC1, {}, POLICY)
        reasons = ["backend_fallback", "never_auto", "backend_fallback"]
        sty, qa = ClaimsModule().decide(SPEC1, {}, [_answer(req, 0.9, 0.85, backend="jev", reasons=reasons)], POLICY)
        self.assertEqual(sty["route_reasons"], ["never_auto", "backend_fallback"])
        self.assertEqual(qa["route_reasons"], ["never_auto", "backend_fallback"])
        failed = ClaimsModule().decide(SPEC1, {}, [_failed(req, reasons=["backend_fallback"])], POLICY)
        self.assertEqual(failed[0]["route_reasons"], ["jev_unavailable", "backend_fallback"])

    def test_everything_down_still_finishes_in_the_agent_lane(self):
        table = {"has_number": noul(0.94), "stat_is_measured": noul(0.85)}
        rows = self._run(FakeBackend("local", by_qid(table), version="0.1.0", healthy=False),
                         FakeBackend("jev", by_qid(table), healthy=False))
        self.assertEqual([r["decision"] for r in rows], ["STY-06"])
        self.assertEqual((rows[0]["lane"], rows[0]["value"]), ("agent", {"class": None}))
        self.assertIn("jev_unavailable", rows[0]["route_reasons"])


class NormaliseTests(unittest.TestCase):
    def test_normalise(self):
        cases = {
            "Errors fell 12 percent.": "errors fell 12%",
            "Errors fell 12 PERCENT": "errors fell 12%",
            "It rose 12.5 percent to 3.5": "it rose 12.5% to 3.5",
            "48,000 invoices, 1,000,000 tokens": "48000 invoices 1000000 tokens",
            "Now that’s 100× productive": "now that s 100x productive",
            "It’s  “great” — really–truly": "it s great really truly",
            "A, B; C: D!": "a b c d",
            "  spaced \n out\t": "spaced out",
            "version 3.5 is out.": "version 3.5 is out",
            "": "",
        }
        for text, want in cases.items():
            self.assertEqual(normalise(text), want, text)

    def test_numbers_in(self):
        self.assertEqual(numbers_in(SENTENCE_2), ["50", "47"])
        self.assertEqual(numbers_in("From 12 percent to 3.5 percent, 48,000 times 100×."), ["12", "3.5", "48000", "100"])
        self.assertEqual(numbers_in("nine more partners"), [])


class QuoteCheckTests(unittest.TestCase):
    SOURCES = {"a3": "Errors fell from 12% to 7% in our benchmark."}

    def test_quote_check(self):
        found = quote_check("In the vendor's report, errors fell from 12 percent to 7 percent.", self.SOURCES)
        self.assertEqual(found, {"match": "numbers_found", "source": "a3", "numbers": ["12", "7"]})
        missing = quote_check("In the vendor's report, errors fell from 12 percent to 9 percent.", self.SOURCES)
        self.assertEqual(missing, {"match": "not_found", "source": None, "numbers": ["12", "9"]})
        self.assertEqual(quote_check("Errors fell from 12 percent to 7 percent.", self.SOURCES),
                         {"match": "quote_found", "source": "a3", "numbers": ["12", "7"]})
        self.assertEqual(quote_check("Errors fell from 12 percent to 7 percent.", {}),
                         {"match": "no_sources", "source": None, "numbers": ["12", "7"]})

    def test_numbers_match_whole_numbers_only(self):
        sources = {"a1": "It got 47 right out of 150."}
        self.assertEqual(quote_check("It got 7 right today.", sources)["match"], "not_found")
        self.assertEqual(quote_check("It got 15 right today.", sources)["match"], "not_found")
        self.assertEqual(quote_check("It got 47 right today.", sources)["match"], "numbers_found")

    def test_all_numbers_must_sit_in_one_source(self):
        sources = {"a1": "Errors fell to 7%.", "a2": "Errors began at 12%."}
        self.assertEqual(quote_check("Errors fell from 12 percent to 7 percent today.", sources)["match"], "not_found")

    def test_a_claim_with_no_digits_can_only_be_a_quote(self):
        sources = {"n": "Partners: nine more partners are building on it, per the launch announcement."}
        self.assertEqual(quote_check("Nine more partners are building on it.", sources),
                         {"match": "quote_found", "source": "n", "numbers": []})
        self.assertEqual(quote_check("Ten more partners are building on it.", sources)["match"], "not_found")

    def test_quote_survives_formatting(self):
        sources = {"p": "He said:  “Errors fell—fast”  in 2025 (see 48,000 runs)."}
        self.assertEqual(quote_check("errors fell - fast in 2025, see 48000 runs", sources)["match"], "quote_found")

    def test_an_empty_claim_is_not_found(self):
        self.assertEqual(quote_check("...", {"a": "anything"})["match"], "not_found")


class LoadSourcesTests(unittest.TestCase):
    def _spec(self, assets):
        spec = copy.deepcopy(SPEC)
        spec["project"]["root"] = str(CLAIMS)
        spec["assets"] = assets
        return spec

    def test_readable_local_text_assets_only(self):
        assets = [
            {"id": "t", "kind": "source_text", "uri": "sources/notes.txt", "sensitivity": "public"},
            {"id": "m", "kind": "source_text", "uri": "sources/report.md", "sensitivity": "public"},
            {"id": "h", "kind": "source_text", "uri": "sources/page.html", "sensitivity": "public"},
            {"id": "url", "kind": "source_text", "uri": "https://example.com/report.html", "sensitivity": "public"},
            {"id": "gone", "kind": "source_text", "uri": "sources/missing.md", "sensitivity": "public"},
            {"id": "pdf", "kind": "source_text", "uri": "sources/report.pdf", "sensitivity": "public"},
            {"id": "img", "kind": "image", "uri": "sources/report.md", "sensitivity": "public"},
        ]
        sources = load_sources(self._spec(assets))
        self.assertEqual(list(sources), ["t", "m", "h"])
        self.assertIn("nine more partners", sources["t"])

    def test_html_tags_are_stripped(self):
        assets = [{"id": "h", "kind": "source_text", "uri": "sources/page.html", "sensitivity": "public"}]
        text = load_sources(self._spec(assets))["h"]
        self.assertNotIn("<", text)
        self.assertEqual(quote_check("Errors fell from 12 percent to 9 percent & stayed there.", {"h": text})["match"],
                         "quote_found")

    def test_no_source_assets(self):
        self.assertEqual(load_sources(self._spec([])), {})
        self.assertEqual(load_sources({"project": {"root": str(CLAIMS)}}), {})

    def test_the_fixture_spec_asset_resolves_under_the_project_root(self):
        self.assertEqual(list(load_sources(SPEC1)), ["a3"])


if __name__ == "__main__":
    unittest.main()
