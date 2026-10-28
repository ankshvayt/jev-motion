"""Tests for core.policy, core.rows and core.module, and the shared fixtures (stdlib unittest). From the tool folder:

    python3 -m unittest tests.test_policy_rows -q

Offline and pure: answers are SimpleNamespace stand-ins, the policy is read from policies/default.json or a temp file.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.module import Module
from core.policy import (LANES, NEVER_AUTO, ROUTE_REASONS, budget_k, decision_policy, in_band, load_policy,
                         round_half_up)
from core.rows import code_evidence, error_route, evidence_from, make_row

TOOL = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
POLICY_FILE = TOOL / "policies" / "default.json"

NEVER_AUTO_IDS = "STY-06 QA-05 AST-01 QA-01"


def _row(decision="CAP-03", lane="auto", reasons=("above_auto",), target=None, **kw):
    entry = {"label": "Default", "ref": "docs/question-sets.md"}
    return make_row(decision=decision, module="universal.captions", target=target or {"kind": "caption_group", "id": "cg14"},
                    value={"span": ["w115", "w116"]}, p=0.86, confidence={"metric": "choice_conf", "value": 0.83},
                    lane=lane, route_reasons=list(reasons), evidence=code_evidence({"evidence": entry}), **kw)


def _answer(**kw):
    fields = dict(request=SimpleNamespace(qs=SimpleNamespace(id="highlight", version=2)), backend="jev", model="jev-1.13.0",
                  request_sha256="ab" * 32, cached=False, error=None, route_reasons=[])
    fields.update(kw)
    return SimpleNamespace(**fields)


class NeverAutoTests(unittest.TestCase):
    def test_never_auto_list_is_the_documented_ids(self):
        self.assertEqual(NEVER_AUTO, frozenset(NEVER_AUTO_IDS.split()))
        documented = (TOOL / "docs" / "never-auto.md").read_text(encoding="utf-8")
        for d in NEVER_AUTO:
            self.assertIn(f"`{d}`", documented, d)
        self.assertEqual(len(NEVER_AUTO), 4)
        self.assertFalse(NEVER_AUTO & {"CAP-02", "CAP-03"})                  # the caption decisions may settle on their own

    def test_never_auto_cannot_be_auto(self):
        for d in sorted(NEVER_AUTO):
            row = _row(decision=d, lane="auto")
            self.assertEqual(row["lane"], "agent", d)
            self.assertIn("never_auto", row["route_reasons"], d)

    def test_never_auto_row_in_other_lanes_is_left_alone(self):
        row = _row(decision="QA-05", lane="human", reasons=("in_review_band",))
        self.assertEqual(row["lane"], "human")
        self.assertEqual(row["route_reasons"], ["in_review_band"])

    def test_ordinary_decision_stays_auto(self):
        row = _row(decision="CAP-03", lane="auto")
        self.assertEqual(row["lane"], "auto")
        self.assertNotIn("never_auto", row["route_reasons"])

    def test_never_auto_ignores_policy_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loud.json"
            path.write_text(json.dumps({"policy": "loud@1", "review_band": 0.1, "module_not_helping": 0.4, "decisions": {
                "STY-06": {"lane": "auto", "can_auto": True, "evidence": {"label": "Default", "ref": "x"}}}}), encoding="utf-8")
            policy = load_policy(path)
        entry = decision_policy(policy, "STY-06")
        self.assertIs(entry["can_auto"], False)
        self.assertNotEqual(entry["lane"], "auto")
        self.assertEqual(_row(decision="STY-06", lane="auto")["lane"], "agent")

    def test_decision_policy_does_not_alias_the_policy(self):
        policy = load_policy(POLICY_FILE)
        entry = decision_policy(policy, "CAP-03")
        entry["evidence"]["ref"] = "changed"
        self.assertEqual(policy["decisions"]["CAP-03"]["evidence"]["ref"], "docs/question-sets.md")

    def test_decision_without_policy_entry_is_conservative(self):
        entry = decision_policy({"decisions": {}}, "VIS-05")
        self.assertIs(entry["can_auto"], False)
        self.assertIn("label", entry["evidence"])
        self.assertIn("ref", entry["evidence"])


class NumbersTests(unittest.TestCase):
    def test_in_band_edges(self):
        self.assertTrue(in_band(0.60, 0.70, 0.10))
        self.assertFalse(in_band(0.5999, 0.70, 0.10))
        self.assertTrue(in_band(0.80, 0.70, 0.10))
        self.assertFalse(in_band(0.8001, 0.70, 0.10))
        self.assertTrue(in_band(0.70, 0.70, 0.10))

    def test_round_half_up(self):
        self.assertEqual([round_half_up(x) for x in (5.5, 8.75, 0.5, 1.5, 2.5, 30.000000000000004, 0.4)], [6, 9, 1, 2, 3, 30, 0])

    def test_budget_k(self):
        self.assertEqual(budget_k(0.25, 22), 6)
        self.assertEqual(budget_k(0.25, 35), 9)
        self.assertEqual(budget_k(0.10, 300), 30)
        self.assertEqual(budget_k(0.01, 10), 1)
        self.assertEqual(budget_k(0.25, 0), 0)


class RowTests(unittest.TestCase):
    def test_row_id_and_defaults(self):
        row = _row()
        self.assertEqual(row["id"], "CAP-03:cg14")
        self.assertEqual(row["status"], "proposed")
        self.assertIsNone(row["resolved"])
        self.assertEqual(row["alternatives"], [])

    def test_row_has_the_7_4_fields(self):
        row = _row(alternatives=[{"value": {"span": ["w114", "w116"]}, "p": 0.11}])
        self.assertEqual(list(row), ["id", "decision", "module", "target", "value", "alternatives", "p", "confidence", "lane",
                                     "route_reasons", "status", "resolved", "evidence"])
        self.assertEqual(row["decision"], "CAP-03")
        self.assertEqual(row["module"], "universal.captions")
        self.assertEqual(row["alternatives"][0]["p"], 0.11)

    def test_row_does_not_alias_route_reasons(self):
        reasons = ["above_auto"]
        row = _row(decision="STY-06", lane="auto", reasons=reasons)
        self.assertEqual(reasons, ["above_auto"])
        self.assertEqual(row["route_reasons"], ["above_auto", "never_auto"])

    def test_unknown_lane_or_reason_raises(self):
        with self.assertRaises(ValueError):
            _row(lane="robot")
        with self.assertRaises(ValueError):
            _row(reasons=("bogus",))
        with self.assertRaises(ValueError):
            _row(reasons=("above_auto", "bogus"))

    def test_enums(self):
        self.assertEqual(LANES, ("auto", "agent", "human"))
        for reason in ("above_auto", "in_review_band", "order_flip", "never_auto", "budget_edge", "jev_unavailable",
                       "jev_error", "local_unavailable", "backend_fallback", "quarantined", "module_not_helping",
                       "order_agree", "presence_low"):
            self.assertIn(reason, ROUTE_REASONS)
        self.assertEqual(len(ROUTE_REASONS), 13)

    def test_evidence_from_answer(self):
        entry = {"lane": "order_agree", "can_auto": True, "evidence": {"label": "Default", "ref": "docs/question-sets.md"}}
        ev = evidence_from(_answer(), entry, order_check="agree")
        self.assertEqual(ev, {"label": "Default", "ref": "docs/question-sets.md", "qset": "highlight", "qset_version": 2, "backend": "jev",
                              "model": "jev-1.13.0", "request_sha256": "ab" * 32, "order_check": "agree", "cached": False})
        self.assertIsNone(evidence_from(_answer(cached=True), entry)["order_check"])
        self.assertIs(evidence_from(_answer(cached=True), entry)["cached"], True)

    def test_code_evidence(self):
        ev = code_evidence({"evidence": {"label": "Default", "ref": "docs/never-auto.md"}})
        self.assertEqual(ev["backend"], "none")
        self.assertEqual((ev["label"], ev["ref"]), ("Default", "docs/never-auto.md"))
        self.assertEqual(set(ev), set(evidence_from(_answer(), {"evidence": {"label": "", "ref": ""}})))
        self.assertIs(ev["cached"], False)

    def test_error_route(self):
        ans = _answer(error="jev_unavailable", backend="none", route_reasons=["backend_fallback"])
        self.assertEqual(error_route(ans), ("agent", ["jev_unavailable", "backend_fallback"]))
        self.assertEqual(ans.route_reasons, ["backend_fallback"])
        self.assertEqual(error_route(_answer(error="jev_error")), ("agent", ["jev_error"]))


class ModuleTests(unittest.TestCase):
    def test_base_module_declares_the_contract(self):
        m = Module()
        for call in (lambda: m.applies({}, {}), lambda: m.requests({}, {}, {}), lambda: m.decide({}, {}, [], {})):
            with self.assertRaises(NotImplementedError):
                call()

    def test_subclass_can_fill_the_contract(self):
        class Caps(Module):
            id = "universal.captions"
            decisions = ("CAP-02", "CAP-03")

            def applies(self, spec, kit):
                return True

            def requests(self, spec, kit, policy):
                return []

            def decide(self, spec, kit, answers, policy):
                return []

        m = Caps()
        self.assertEqual((m.id, m.decisions), ("universal.captions", ("CAP-02", "CAP-03")))
        self.assertTrue(m.applies({}, {}))


class PolicyFileTests(unittest.TestCase):
    def test_policy_file_has_p0_decisions(self):
        policy = load_policy(POLICY_FILE)
        self.assertEqual(sorted(policy["decisions"]), ["AST-01", "CAP-02", "CAP-03", "QA-01", "QA-05", "STY-06"])
        self.assertEqual((policy["policy"], policy["review_band"], policy["module_not_helping"]), ("default@1", 0.10, 0.40))
        self.assertIs(policy["decisions"]["CAP-02"]["can_auto"], True)
        self.assertIs(policy["decisions"]["CAP-03"]["can_auto"], True)
        for d in ("STY-06", "QA-05", "AST-01", "QA-01"):
            self.assertIs(policy["decisions"][d]["can_auto"], False, d)

    def test_load_policy_default_path(self):
        self.assertEqual(load_policy(), load_policy(POLICY_FILE))

    def test_policy_entries_carry_evidence(self):
        policy = load_policy(POLICY_FILE)
        for d, entry in policy["decisions"].items():
            self.assertTrue(entry["evidence"]["label"] and entry["evidence"]["ref"], d)
            self.assertEqual(decision_policy(policy, d)["evidence"], entry["evidence"])


class FixtureTests(unittest.TestCase):
    def test_fixtures_load(self):
        manifest = json.loads((FIXTURES / "manifest_min.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest), {"manifest_version", "kit", "sources", "canvases", "themes", "emphasis", "captions", "rules",
                                         "qa_context"})
        self.assertEqual(manifest["manifest_version"], "1")
        self.assertEqual(set(manifest["canvases"]), {"short", "long"})
        self.assertEqual(manifest["kit"]["fingerprint"], "")
        self.assertTrue(manifest["qa_context"]["short"].startswith("A frame from a vertical 1080x1920 Short"))
        self.assertIsNone(manifest["qa_context"]["long"])
        spec = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
        self.assertEqual(len(spec["transcript"]["words"]), 18)
        self.assertEqual(len(spec["units"]["caption_groups"]), 4)
        self.assertEqual(len(spec["units"]["sentences"]), 2)

    def test_spec_fixture_is_consistent(self):
        spec = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
        words = spec["transcript"]["words"]
        self.assertEqual([w["id"] for w in words], [f"w{i}" for i in range(18)])
        for i, w in enumerate(words):
            self.assertAlmostEqual(w["start"], 0.3 * i, places=6)
            self.assertAlmostEqual(w["end"] - w["start"], 0.3, places=6)
        self.assertEqual(spec["transcript"]["text"], " ".join(w["text"] for w in words))
        groups = {g["id"]: g["words"] for g in spec["units"]["caption_groups"]}
        self.assertEqual(groups, {"cg0": ["w0", "w4"], "cg1": ["w5", "w7"], "cg2": ["w8", "w13"], "cg3": ["w14", "w17"]})
        sentences = [" ".join(w["text"] for w in words[int(s["words"][0][1:]):int(s["words"][1][1:]) + 1]) for s in spec["units"]["sentences"]]
        self.assertEqual(sentences, ["Sort every invoice into folders in one click.", "In my test of 50 invoices, it got 47 right."])
        assets = {a["id"]: a for a in spec["assets"]}
        self.assertEqual((assets["a1"]["kind"], assets["a1"]["uri"], assets["a1"]["sensitivity"], assets["a1"]["dims"]),
                         ("screen_recording", "raw/screen.mov", "private", [1920, 1080]))
        self.assertEqual((assets["a2"]["kind"], assets["a2"]["uri"], assets["a2"]["sensitivity"]), ("image", "assets/shot.png", "public"))
        self.assertEqual((assets["a3"]["kind"], assets["a3"]["uri"], assets["a3"]["sensitivity"]), ("source_text", "sources/report.md", "public"))
        self.assertEqual(spec["constraints"]["highlight"]["density"], 0.25)
        self.assertEqual(spec["decide"]["budget_usd"], 0.10)
        self.assertEqual(spec["project"]["root"], "jevm-fixture")

    def test_spec_fixture_manifest_sha_matches_the_manifest_fixture(self):
        spec = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
        manifest = TOOL / spec["kit"]["manifest"]
        self.assertEqual(manifest.resolve(), (FIXTURES / "manifest_min.json").resolve())
        self.assertEqual(spec["kit"]["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
