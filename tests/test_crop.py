"""Focus crop AST-01: requests, decision and overlay (stdlib unittest). From the tool folder:

    python3 -m unittest tests.test_crop -q

Offline: answers are built by hand around the requests `crop_requests` makes. The window body and focus scale range come
from the manifest fixture. The overlay test needs ffmpeg.
"""
import dataclasses
import json
import socket
import subprocess
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from core.policy import load_policy
from jevlib import fake
from jevlib.cache import Cache
from jevlib.client import Client
from jevlib.images import FFMPEG, Fit, image_size, mean_luma, unfit_box
from jevlib.request import Answer, load_qset
from modules.universal import assets, geometry

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WANT = "the export dialog with the format menu and the save button"
TARGET = {"kind": "shot", "id": "s03"}
BOX = (0.30, 0.40, 0.70, 0.95)  # fractions of a 1920x1080 frame sent as 1568x882, no padding


def manifest():
    return json.loads((FIXTURES / "manifest_min.json").read_text(encoding="utf-8"))


def build_answer(req, answers, *, backend="jev", model="jev-1.14.0", reasons=(), error=None):
    return Answer(request=req, answers=answers, backend=backend, model=model, model_resolved=None, input_tokens=100,
                  latency_s=0.0, cached=False, request_sha256="ab" * 32 if error is None else "", error=error,
                  route_reasons=list(reasons))


class CropCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.frame = self.tmp / "frame.jpg"
        self.frame.write_bytes(b"\xff\xd8 not decoded by the request builder \xff\xd9")
        self.reqs = assets.crop_requests(self.frame, WANT, "public")
        self.policy = load_policy()
        m = manifest()
        self.body = m["canvases"]["short"]["window_body"]
        self.scale_range = m["rules"]["focus_scale"]
        self.fit = Fit(1920, 1080, 1568 / 1920, (0, 0, 0, 0), 1568, 882)

    def presence_answer(self, p=None, *, backend="local", reasons=(), error=None):
        answers = None if p is None else {"vis": fake.noul(p)}
        model = "pplx-decider-flash-v1.2" if backend == "local" else "jev-1.14.0"
        return build_answer(self.reqs[0], answers, backend=backend, model=model, reasons=reasons, error=error)

    def box_answer(self, vis=0.95, frac=BOX, *, reasons=(), error=None):
        if error is not None:
            return build_answer(self.reqs[1], None, reasons=reasons, error=error, model="")
        left, top, right, bottom = frac
        return build_answer(self.reqs[1], {"box": fake.box(left, top, right, bottom), "vis": fake.noul(vis)},
                            reasons=reasons)

    def decide(self, presence, box, *, fit=None, **over):
        kw = dict(fit=fit or self.fit, body=self.body, scale_range=self.scale_range, target=TARGET, asset_id="a_demo",
                  t=20.8, want=WANT, policy=self.policy)
        kw.update(over)
        return assets.crop_decide([presence, box], **kw)

    def fallback_presence(self, p=0.0):
        """A presence answer that cannot be the gate: Decider fell back to Jev."""
        return self.presence_answer(p, backend="jev", reasons=["backend_fallback"])


class RequestTests(CropCase):
    def test_module_id(self):
        self.assertEqual(assets.MODULE_ID, "universal.assets")

    def test_crop_requests_presence_then_box(self):
        presence, box = self.reqs
        self.assertEqual([r.tag for r in self.reqs], ["presence", "box"])
        self.assertFalse(presence.has_box())
        self.assertTrue(box.has_box())
        self.assertEqual(list(presence.questions), ["vis"])
        self.assertEqual(list(box.questions), ["box", "vis"])
        self.assertEqual(box.questions["box"]["type"], "box")
        self.assertEqual(box.questions["vis"]["type"], "noul")
        for req in self.reqs:
            for q in req.questions.values():
                self.assertIn(WANT, q["instructions"])
                self.assertNotIn("{want}", q["instructions"])
        self.assertEqual((presence.qs.id, box.qs.id), ("presence", "crop_box"))

    def test_crop_requests_share_one_image_ref(self):
        for req in self.reqs:
            [ref] = req.image_refs_in_order()
            self.assertTrue(ref.startswith("<<img-") and ref.endswith(">>"))
            self.assertEqual(req.state, {"image": ref})
            self.assertEqual(req.images, {ref: self.frame})
        self.assertEqual(self.reqs[0].state, self.reqs[1].state)

    def test_crop_requests_carry_the_sensitivity(self):
        private = assets.crop_requests(self.frame, WANT, "private")
        self.assertEqual([r.sensitivity for r in private], ["private", "private"])
        self.assertEqual([r.sensitivity for r in self.reqs], ["public", "public"])

    def test_qsets_override_for_eval(self):
        over = dataclasses.replace(load_qset("presence"), version=99)
        reqs = assets.crop_requests(self.frame, WANT, "public", qsets={"presence": over})
        self.assertEqual(reqs[0].qs.version, 99)
        self.assertEqual(reqs[1].qs.id, "crop_box")

    def test_qset_files(self):
        box, presence = load_qset("crop_box"), load_qset("presence")
        self.assertEqual((box.route, box.backend, box.models), ("jev", "jev", {"jev": "jev-1.14.0", "local": None}))
        self.assertEqual(box.thresholds, {"jev": {"drop_below": 0.30, "propose_at": 0.70}, "local": None})
        self.assertFalse(box.has_local_calibration())
        self.assertEqual((presence.route, presence.backend), ("local", "local"))
        self.assertEqual(presence.models, {"jev": "jev-1.14.0", "local": "pplx-decider-flash-v1.2"})
        self.assertEqual(presence.thresholds["local"], {"drop_below": 0.30, "propose_at": 0.50})
        self.assertEqual(presence.calibrated["local"], {"app_version": "0.1.0", "model": "pplx-decider-flash-v1.2"})
        self.assertEqual(list(presence.questions), ["vis"])


class GateTests(CropCase):
    def test_gate_uses_local_presence_when_local_answered(self):
        # 0.55 is above the local propose_at (0.50) but under Jev's (0.70); the bundled vis would say 0.95.
        row = self.decide(self.presence_answer(0.55), self.box_answer(vis=0.95))
        self.assertEqual(row["p"], 0.55)
        self.assertEqual(row["route_reasons"], ["above_auto"])
        self.assertEqual(row["confidence"], {"metric": "box_presence", "value": 0.55})
        low = self.decide(self.presence_answer(0.20), self.box_answer(vis=0.95))
        self.assertEqual((low["p"], low["value"]["found"], low["route_reasons"]), (0.20, False, ["presence_low"]))
        band = self.decide(self.presence_answer(0.40), self.box_answer(vis=0.95))
        self.assertEqual(band["route_reasons"], ["in_review_band"])

    def test_gate_uses_bundled_vis_on_fallback(self):
        row = self.decide(self.fallback_presence(0.99), self.box_answer(vis=0.55))
        self.assertEqual(row["p"], 0.55)
        self.assertEqual(row["route_reasons"], ["in_review_band", "backend_fallback"])
        self.assertTrue(row["value"]["found"])
        # A local answer that is itself a stand-in carries backend_fallback and does not gate either.
        stand_in = self.presence_answer(0.99, backend="local", reasons=["backend_fallback"])
        self.assertEqual(self.decide(stand_in, self.box_answer(vis=0.55))["p"], 0.55)

    def test_crop_row_carries_backend_fallback(self):
        # Decider down, Jev answers both requests: the presence answer carries backend_fallback, and the row must too.
        table = {"vis": fake.noul(0.55), "box": fake.box(*BOX)}
        local = fake.FakeBackend("local", fake.by_qid(table), version="0.1.0", healthy=False)
        jev = fake.FakeBackend("jev", fake.by_qid(table))
        with tempfile.TemporaryDirectory() as td:
            answers = Client({"jev": jev, "local": local}, Cache(root=Path(td)), log=lambda message: None).run_many(self.reqs)
        self.assertEqual([a.route_reasons for a in answers], [["backend_fallback"], []])
        row = self.decide(*answers)
        self.assertEqual((row["lane"], row["value"]["found"], row["p"]), ("agent", True, 0.55))   # no lane change
        self.assertEqual(row["route_reasons"], ["in_review_band", "backend_fallback"])
        # The not-found and unusable rows carry it too, once, with the row's own reasons first.
        low = self.decide(self.fallback_presence(), self.box_answer(vis=0.1))
        self.assertEqual((low["lane"], low["route_reasons"]), ("agent", ["presence_low", "backend_fallback"]))
        stand_in = self.presence_answer(0.99, backend="local", reasons=["backend_fallback"])
        both = self.decide(stand_in, self.box_answer(vis=0.9, reasons=["backend_fallback"]))
        self.assertEqual(both["route_reasons"], ["above_auto", "backend_fallback"])
        failed = self.decide(self.fallback_presence(), self.box_answer(error="jev_unavailable"))
        self.assertEqual((failed["lane"], failed["route_reasons"]), ("agent", ["jev_unavailable", "backend_fallback"]))

    def test_gate_uses_bundled_vis_when_presence_failed(self):
        presence = self.presence_answer(error="local_unavailable", reasons=[])
        row = self.decide(presence, self.box_answer(vis=0.85))
        self.assertEqual(row["p"], 0.85)
        self.assertEqual(row["route_reasons"], ["above_auto"])

    def test_local_presence_drops_without_needing_the_box(self):
        row = self.decide(self.presence_answer(0.10), self.box_answer(error="jev_unavailable"))
        self.assertEqual((row["p"], row["value"]["found"], row["route_reasons"]), (0.10, False, ["presence_low"]))

    def test_lanes(self):
        cases = ((0.9, True, "above_auto"), (0.5, True, "in_review_band"), (0.1, False, "presence_low"))
        for p, found, reason in cases:
            with self.subTest(p=p):
                row = self.decide(self.fallback_presence(), self.box_answer(vis=p))
                self.assertEqual(row["lane"], "agent")
                self.assertEqual(row["value"]["found"], found)
                self.assertEqual(row["route_reasons"][0], reason)
                self.assertEqual(row["p"], p)
                self.assertEqual(row["confidence"], {"metric": "box_presence", "value": p})
        # The thresholds belong to the band above them.
        self.assertEqual(self.decide(self.fallback_presence(), self.box_answer(vis=0.30))["route_reasons"][0], "in_review_band")
        self.assertEqual(self.decide(self.fallback_presence(), self.box_answer(vis=0.70))["route_reasons"][0], "above_auto")

    def test_not_found_value_has_no_geometry(self):
        row = self.decide(self.fallback_presence(), self.box_answer(vis=0.1))
        self.assertEqual(row["value"], {"found": False, "asset": "a_demo", "t": 20.8, "want": WANT})
        self.assertEqual(row["decision"], "AST-01")
        self.assertEqual(row["id"], "AST-01:s03")
        self.assertEqual(row["module"], assets.MODULE_ID)


class ValueTests(CropCase):
    def test_found_value_shape(self):
        row = self.decide(self.fallback_presence(), self.box_answer(vis=0.9))
        v = row["value"]
        self.assertEqual(list(v), ["found", "asset", "t", "want", "box_px", "box", "focus", "body", "clamped"])
        self.assertTrue(v["found"])
        self.assertEqual((v["asset"], v["t"], v["want"]), ("a_demo", 20.8, WANT))
        self.assertEqual(v["body"], list(self.body))
        self.assertEqual(list(v["focus"]), ["scale", "x", "y"])
        self.assertEqual(v["clamped"], [])
        self.assertTrue(self.scale_range[0] <= v["focus"]["scale"] <= self.scale_range[1])
        json.dumps(row)

    def test_box_fractions_map_through_fit(self):
        # A 1000x500 source sent at half size and padded top and bottom: the box has to come back in source px.
        fit = Fit(src_w=1000, src_h=500, scale=0.5, pad=(0, 20, 0, 20), w=500, h=290)
        frac = (0.2, 70 / 290, 0.6, 170 / 290)
        want_tight = (200.0, 100.0, 600.0, 300.0)
        self.assertEqual(tuple(round(v, 6) for v in unfit_box(dict(zip(("left", "top", "right", "bottom"), frac)), fit)),
                         want_tight)
        row = self.decide(self.fallback_presence(), self.box_answer(vis=0.9, frac=frac), fit=fit)
        v = row["value"]
        expanded = geometry.expand(want_tight, self.body[0] / self.body[1], 1000, 500)
        for got, want in zip(v["box_px"], expanded):
            self.assertAlmostEqual(got, want, delta=0.06)
        self.assertEqual(v["box"], {"left": 0.2, "top": 0.2, "right": 0.6, "bottom": 0.6})
        want_focus = geometry.to_focus(1000, 500, self.body, expanded, self.scale_range)
        self.assertAlmostEqual(v["focus"]["scale"], want_focus["scale"], delta=0.001)
        self.assertAlmostEqual(v["focus"]["x"], want_focus["x"], delta=0.06)
        self.assertAlmostEqual(v["focus"]["y"], want_focus["y"], delta=0.06)

    def test_box_px_has_the_window_aspect_and_stays_inside_the_frame(self):
        v = self.decide(self.fallback_presence(), self.box_answer(vis=0.9))["value"]
        x0, y0, x1, y1 = v["box_px"]
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), self.body[0] / self.body[1], places=2)
        self.assertTrue(0 <= x0 < x1 <= 1920 and 0 <= y0 < y1 <= 1080)

    def test_small_box_clamps_the_scale(self):
        v = self.decide(self.fallback_presence(), self.box_answer(vis=0.9, frac=(0.45, 0.45, 0.55, 0.55)))["value"]
        self.assertEqual(v["clamped"], ["scale"])
        self.assertEqual(v["focus"]["scale"], self.scale_range[1])

    def test_near_whole_frame_box_clamps_the_scale_up(self):
        v = self.decide(self.fallback_presence(), self.box_answer(vis=0.9, frac=(0.0, 0.0, 1.0, 1.0)))["value"]
        self.assertEqual(v["clamped"], ["scale"])
        self.assertEqual(v["focus"]["scale"], self.scale_range[0])

    def edge_row(self, box_px):
        sw, sh = 1920, 1080
        frac = (box_px[0] / sw, box_px[1] / sh, box_px[2] / sw, box_px[3] / sh)
        return self.decide(self.fallback_presence(), self.box_answer(vis=0.9, frac=frac))["value"]

    def assert_focus_in_region(self, v):
        """The focus the row carries shows a rectangle inside the cover region, and its origin is inside the body."""
        region = geometry.cover_region(1920, 1080, self.body)
        shown = geometry.focus_rect(1920, 1080, self.body, v["focus"]["scale"], v["focus"]["x"], v["focus"]["y"])
        for got, lo, hi in ((shown[0], region[0], region[2]), (shown[2], region[0], region[2]),
                            (shown[1], region[1], region[3]), (shown[3], region[1], region[3])):
            self.assertTrue(lo - 0.5 <= got <= hi + 0.5, (shown, region))
        self.assertTrue(0 <= v["focus"]["x"] <= self.body[0] and 0 <= v["focus"]["y"] <= self.body[1], v["focus"])

    def test_row_focus_moves_into_the_cover_region_at_the_left_edge(self):
        v = self.edge_row((0, 300, 600, 800))
        self.assertEqual(v["clamped"], ["x"])
        self.assertEqual(v["box_px"][0], 0.0)  # box_px stays the expanded box; only the focus window moved
        self.assert_focus_in_region(v)

    def test_row_focus_moves_into_the_cover_region_at_the_right_edge(self):
        v = self.edge_row((1320, 300, 1920, 800))
        self.assertEqual(v["clamped"], ["x"])
        self.assertEqual(v["box_px"][2], 1920.0)
        self.assert_focus_in_region(v)

    def test_row_clamped_is_empty_for_a_centred_box(self):
        v = self.edge_row((660, 390, 1260, 690))
        self.assertEqual(v["clamped"], [])
        self.assert_focus_in_region(v)


class FailureTests(CropCase):
    def test_failed_box_routes_to_agent(self):
        row = self.decide(self.fallback_presence(), self.box_answer(error="jev_unavailable"))
        self.assertEqual(row["lane"], "agent")
        self.assertIsNone(row["value"]["found"])
        self.assertEqual(set(row["value"]), {"found", "asset", "t", "want"})
        self.assertIsNone(row["p"])
        self.assertIsNone(row["confidence"])
        self.assertEqual(row["route_reasons"][0], "jev_unavailable")

    def test_both_failed(self):
        presence = self.presence_answer(error="local_unavailable")
        row = self.decide(presence, self.box_answer(error="jev_error"))
        self.assertEqual(row["lane"], "agent")
        self.assertEqual(row["route_reasons"][0], "jev_error")

    def test_degenerate_box_routes_to_agent(self):
        for frac in ((0.4, 0.4, 0.4, 0.8), (0.4, 0.6, 0.8, 0.6), (0.8, 0.4, 0.4, 0.8)):
            with self.subTest(frac=frac):
                row = self.decide(self.fallback_presence(), self.box_answer(vis=0.9, frac=frac))
                self.assertEqual(row["lane"], "agent")
                self.assertIsNone(row["value"]["found"])
                self.assertEqual(row["route_reasons"][:1], ["jev_error"])

    def test_missing_answers_route_to_agent(self):
        no_vis = build_answer(self.reqs[1], {"box": fake.box(0.1, 0.1, 0.5, 0.5)})
        row = self.decide(self.fallback_presence(), no_vis)
        self.assertIsNone(row["value"]["found"])
        self.assertEqual(row["route_reasons"][:1], ["jev_error"])
        no_box = build_answer(self.reqs[1], {"vis": fake.noul(0.9)})
        row = self.decide(self.fallback_presence(), no_box)
        self.assertIsNone(row["value"]["found"])
        self.assertEqual(row["route_reasons"][:1], ["jev_error"])

    def test_decide_makes_no_network_call(self):
        with mock.patch.object(socket, "socket", side_effect=AssertionError("network")), \
                mock.patch.object(urllib.request, "urlopen", side_effect=AssertionError("network")):
            row = self.decide(self.presence_answer(0.8), self.box_answer(vis=0.9))
        self.assertTrue(row["value"]["found"])


class OverlayTests(CropCase):
    def run_ffmpeg(self, *args):
        subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True)

    def test_overlay_draws_the_rect_on_the_sent_frame(self):
        folder = self.tmp / "dir with space"
        folder.mkdir()
        sent = folder / "sent.jpg"
        self.run_ffmpeg("-f", "lavfi", "-i", "color=c=black:s=1568x882", "-frames:v", "1", "-update", "1", str(sent))
        row = self.decide(self.fallback_presence(), self.box_answer(vis=0.9))
        out = assets.overlay(sent, row, self.fit, folder / "overlay.jpg")
        self.assertEqual(out, folder / "overlay.jpg")
        self.assertEqual(image_size(out)[:2], (1568, 882))
        self.assertGreater(mean_luma(out), mean_luma(sent) + 0.5)

    def test_overlay_maps_source_px_through_a_padded_fit(self):
        sent = self.tmp / "sent.jpg"
        self.run_ffmpeg("-f", "lavfi", "-i", "color=c=black:s=500x290", "-frames:v", "1", "-update", "1", str(sent))
        fit = Fit(src_w=1000, src_h=500, scale=0.5, pad=(0, 20, 0, 20), w=500, h=290)
        row = self.decide(self.fallback_presence(), self.box_answer(vis=0.9, frac=(0.2, 70 / 290, 0.6, 170 / 290)), fit=fit)
        out = assets.overlay(sent, row, fit, self.tmp / "o.jpg")
        self.assertEqual(image_size(out)[:2], (500, 290))
        self.assertGreater(mean_luma(out), mean_luma(sent) + 0.5)

    def test_overlay_of_a_row_without_a_box_is_the_plain_frame(self):
        sent = self.tmp / "sent.jpg"
        self.run_ffmpeg("-f", "lavfi", "-i", "color=c=black:s=1568x882", "-frames:v", "1", "-update", "1", str(sent))
        row = self.decide(self.fallback_presence(), self.box_answer(vis=0.1))
        out = assets.overlay(sent, row, self.fit, self.tmp / "none.jpg")
        self.assertEqual(image_size(out)[:2], (1568, 882))
        self.assertAlmostEqual(mean_luma(out), mean_luma(sent), delta=0.5)


if __name__ == "__main__":
    unittest.main()
