"""Tests for jevlib.request: question sets, requests, answers and the accessors over them."""
import json
import tempfile
import unittest
from pathlib import Path

from jevlib.request import (Answer, Request, SENSITIVITIES, box_frac, choice_conf, choice_probs, choice_top,
                            image_ref, load_qset, noul_p, strictest, thresholds_for)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "qsets"

GOOD = {"id": "t", "version": 1, "route": "jev", "backend": "jev",
        "models": {"jev": "jev-1.13.0", "local": "pplx-decider-flash-v1.2"},
        "thresholds": {"jev": {}, "local": None}, "calibrated": {"local": None}, "order_check": False,
        "batch": 1, "evidence": {"label": "Default", "ref": "x"}, "notes": "",
        "questions": {"q": {"type": "noul", "instructions": "i"}}}


def write_set(root, version=1, **over):
    data = dict(GOOD, version=version, **over)
    data = {k: v for k, v in data.items() if v is not ...}
    (Path(root) / f"t@{version}.json").write_text(json.dumps(data), encoding="utf-8")


def answer(answers, **kw):
    qs = load_qset("sample", root=FIXTURES)
    base = dict(request=Request(qs, {}, {}), answers=answers, backend="jev", model="jev-1.13.0",
                model_resolved=None, input_tokens=0, latency_s=0.0, cached=False, request_sha256="")
    base.update(kw)
    return Answer(**base)


class QuestionSetTests(unittest.TestCase):
    def test_load_qset_highest_version(self):
        self.assertEqual(load_qset("sample", root=FIXTURES).version, 2)
        self.assertEqual(load_qset("sample", version=1, root=FIXTURES).version, 1)

    def test_load_qset_fields(self):
        qs = load_qset("sample", root=FIXTURES)
        self.assertEqual((qs.id, qs.route, qs.backend, qs.batch, qs.order_check), ("sample", "jev", "jev", 4, True))
        self.assertEqual(qs.models["jev"], "jev-1.13.0")
        self.assertFalse(qs.has_local_calibration())

    def test_load_qset_missing_raises_file_not_found_naming_the_id(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError) as cm:
                load_qset("nope", root=Path(d))
            self.assertIn("nope", str(cm.exception))
            with self.assertRaises(FileNotFoundError) as cm:
                load_qset("sample", version=9, root=FIXTURES)
            self.assertIn("sample", str(cm.exception))

    def test_load_qset_missing_field_and_bad_route_raise_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            write_set(d, route=...)
            with self.assertRaises(ValueError) as cm:
                load_qset("t", root=Path(d))
            self.assertIn("route", str(cm.exception))
        with tempfile.TemporaryDirectory() as d:
            write_set(d, route="cloud")
            with self.assertRaises(ValueError) as cm:
                load_qset("t", root=Path(d))
            self.assertIn("cloud", str(cm.exception))

    def test_load_qset_accepts_all_four_routes(self):
        for route in ("jev", "local", "either", "jev-for-speed"):
            with tempfile.TemporaryDirectory() as d:
                write_set(d, route=route)
                self.assertEqual(load_qset("t", root=Path(d)).route, route)

    def test_question_substitutes_without_format(self):
        q = load_qset("sample", root=FIXTURES).question("q_{i}", i="3")
        self.assertIn("`groups[3].text`", q["instructions"])
        self.assertIn("{not_a_sub}", q["instructions"])
        self.assertIn('{"k": 1}', q["instructions"])

    def test_question_returns_a_deep_copy(self):
        qs = load_qset("sample", root=FIXTURES)
        qs.question("q_{i}", i="1")["instructions"] = "changed"
        self.assertIn("{i}", qs.questions["q_{i}"]["instructions"])

    def test_has_local_calibration_follows_local_thresholds(self):
        with tempfile.TemporaryDirectory() as d:
            write_set(d, thresholds={"jev": {"need": 0.5}, "local": {"need": 0.7}})
            qs = load_qset("t", root=Path(d))
            self.assertTrue(qs.has_local_calibration())
            self.assertEqual(thresholds_for(qs, "local"), {"need": 0.7})
            self.assertEqual(thresholds_for(qs, "jev"), {"need": 0.5})

    def test_thresholds_for_local_none_uses_jev(self):
        qs = load_qset("sample", root=FIXTURES)
        self.assertEqual(thresholds_for(qs, "local"), {"need": 0.5})
        self.assertEqual(thresholds_for(qs, "jev"), {"need": 0.5})


class RequestTests(unittest.TestCase):
    def test_image_ref_is_stable_and_tagged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a b.png"
            p.write_bytes(b"pixels")
            ref = image_ref(p, "frame")
            self.assertEqual(ref, image_ref(p, "frame"))
            self.assertTrue(ref.startswith("<<frame-"))
            self.assertTrue(ref.endswith(">>"))
            self.assertEqual(len(ref), 22)
            self.assertTrue(image_ref(p).startswith("<<img-"))

    def test_image_refs_in_order_whole_values_only(self):
        qs = load_qset("sample", root=FIXTURES)
        with tempfile.TemporaryDirectory() as d:
            refs = []
            for i in range(3):
                p = Path(d) / f"{i}.png"
                p.write_bytes(bytes([i]) * 8)
                refs.append((image_ref(p), p))
            (ref1, path1), (ref2, path2), (ref3, path3) = refs
            state = {"b": ref2, "a": [ref1], "c": "see " + ref3}
            req = Request(qs, state, {}, images={ref1: path1, ref2: path2, ref3: path3})
            self.assertEqual(req.image_refs_in_order(), [ref2, ref1])

    def test_image_refs_walk_state_then_questions_and_dedupe(self):
        qs = load_qset("sample", root=FIXTURES)
        with tempfile.TemporaryDirectory() as d:
            ps = []
            for i in range(2):
                p = Path(d) / f"{i}.png"
                p.write_bytes(bytes([i + 5]) * 8)
                ps.append((image_ref(p), p))
            (ra, pa), (rb, pb) = ps
            req = Request(qs, {"x": {"deep": [rb]}},
                          {"q": {"type": "choice", "options": [ra, rb]}}, images={ra: pa, rb: pb})
            self.assertEqual(req.image_refs_in_order(), [rb, ra])

    def test_has_box(self):
        qs = load_qset("sample", root=FIXTURES)
        noul_only = Request(qs, {}, {"a": {"type": "noul", "instructions": "i"}})
        with_box = Request(qs, {}, {"a": {"type": "noul", "instructions": "i"},
                                    "b": {"type": "box", "instructions": "i"}})
        self.assertFalse(noul_only.has_box())
        self.assertTrue(with_box.has_box())

    def test_request_defaults_are_independent(self):
        qs = load_qset("sample", root=FIXTURES)
        a, b = Request(qs, {}, {}), Request(qs, {}, {})
        self.assertEqual((a.sensitivity, a.tag, a.images), ("public", "", {}))
        self.assertIsNot(a.images, b.images)


class AccessorTests(unittest.TestCase):
    def test_choice_conf(self):
        self.assertAlmostEqual(choice_conf({"a": 1, "b": 0, "c": 0}), 1.0, places=6)
        self.assertAlmostEqual(choice_conf({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}), 0.0, places=6)
        self.assertAlmostEqual(choice_conf({"a": 0.6, "b": 0.4}), 0.2, places=6)
        self.assertEqual(choice_conf({"a": 1.0}), 1.0)
        self.assertEqual(choice_conf({}), 1.0)

    def test_choice_top_excludes_none(self):
        ans = answer({"p": {"choice": "none", "confidence": 0.5,
                            "probabilities": {"none": 0.5, "h1": 0.3, "h0": 0.2}}})
        self.assertEqual(choice_top(ans, "p"), ("h1", 0.3))
        self.assertEqual(choice_top(ans, "p", exclude=()), ("none", 0.5))
        self.assertEqual(choice_top(answer({"p": {"probabilities": {"none": 1.0}}}), "p"), (None, 0.0))

    def test_noul_and_choice_probs_and_box_frac(self):
        ans = answer({"n": {"noul": 0.25},
                      "c": {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}},
                      "b": {"box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8}, "confidence": 0.6}})
        self.assertEqual(noul_p(ans, "n"), 0.25)
        self.assertEqual(choice_probs(ans, "c"), {"a": 0.7, "b": 0.3})
        self.assertEqual(box_frac(ans, "b"), {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8})
        for missing in (noul_p(ans, "zz"), box_frac(ans, "zz")):
            self.assertIsNone(missing)
        self.assertEqual(choice_probs(ans, "zz"), {})

    def test_accessors_on_a_failed_answer(self):
        ans = answer(None, backend="none", error="jev_unavailable")
        self.assertFalse(ans.ok)
        self.assertIsNone(noul_p(ans, "n"))
        self.assertEqual(choice_probs(ans, "c"), {})
        self.assertEqual(choice_top(ans, "c"), (None, 0.0))
        self.assertIsNone(box_frac(ans, "b"))
        self.assertEqual(ans.route_reasons, [])
        self.assertTrue(answer({}).ok)

    def test_strictest(self):
        self.assertEqual(SENSITIVITIES, ("public", "own_render", "private"))
        self.assertEqual(strictest(["public", "own_render"]), "own_render")
        self.assertEqual(strictest(["public", None]), "private")
        self.assertEqual(strictest(["private", "public"]), "private")
        self.assertEqual(strictest([]), "public")


if __name__ == "__main__":
    unittest.main()
