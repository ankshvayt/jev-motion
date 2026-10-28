"""Focus-crop geometry (stdlib unittest). From the tool folder:

    python3 -m unittest tests.test_geometry -q

Offline and pure. The window body and the three synthetic cases are read from tests/fixtures/crop/focus_cases.json.
"""
import json
import unittest
from pathlib import Path

from modules.universal import geometry

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "crop" / "focus_cases.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def within(test, got, want, tol):
    for g, w in zip(got, want):
        test.assertAlmostEqual(g, w, delta=tol)


class FocusGeometryTests(unittest.TestCase):
    def setUp(self):
        self.fx = load_fixture()
        self.body = self.fx["body"]

    def test_focus_rect_reproduces_the_hand_rects(self):
        for case in self.fx["cases"]:
            with self.subTest(case["id"]):
                sw, sh = case["src"]
                rect = geometry.focus_rect(sw, sh, self.body, *case["focus"])
                within(self, rect, case["hand"], 1.0)

    def test_rect_to_focus_inverts_focus_rect(self):
        for case in self.fx["cases"]:
            with self.subTest(case["id"]):
                sw, sh = case["src"]
                scale, x, y = case["focus"]
                rect = geometry.focus_rect(sw, sh, self.body, scale, x, y)
                got = geometry.rect_to_focus(sw, sh, self.body, rect)
                self.assertAlmostEqual(got[0], scale, delta=0.01)
                within(self, got[1:], (x, y), 1.0)

    def test_rect_to_focus_at_scale_one_does_not_divide_by_zero(self):
        sw, sh = self.fx["cases"][0]["src"]
        rect = geometry.focus_rect(sw, sh, self.body, 1.0, 100, 100)
        scale, x, y = geometry.rect_to_focus(sw, sh, self.body, rect)
        self.assertAlmostEqual(scale, 1.0)
        self.assertEqual((x, y), (self.body[0] / 2, self.body[1] / 2))

    def test_expand_grows_to_the_window_aspect_inside_the_frame(self):
        iw, ih = self.fx["expand_case"]["src"]
        aspect = self.body[0] / self.body[1]
        box = (0, 0, 100, 400)
        x0, y0, x1, y1 = geometry.expand(box, aspect, iw, ih)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), aspect)
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, iw)
        self.assertLessEqual(y1, ih)
        whole = geometry.expand((0, 0, iw, ih), aspect, iw, ih)
        self.assertAlmostEqual((whole[2] - whole[0]) / (whole[3] - whole[1]), aspect)
        self.assertLessEqual(whole[3] - whole[1], ih + 1e-6)


class ToFocusTests(unittest.TestCase):
    def setUp(self):
        self.fx = load_fixture()
        self.body = self.fx["body"]
        self.range = (1.4, 2.0)
        self.sw, self.sh = self.fx["cases"][0]["src"]

    def test_inside_the_range_nothing_moves(self):
        case = self.fx["cases"][0]
        scale, x, y = case["focus"]
        rect = geometry.focus_rect(self.sw, self.sh, self.body, scale, x, y)
        out = geometry.to_focus(self.sw, self.sh, self.body, rect, self.range)
        self.assertEqual(out["clamped"], [])
        self.assertAlmostEqual(out["scale"], scale, delta=0.01)
        within(self, (out["x"], out["y"]), (x, y), 1.0)
        within(self, out["rect"], rect, 1e-6)

    def test_to_focus_clamps_and_recentres(self):
        W, H = self.body
        # Centred on the body centre, shown at scale 2.6: above the range.
        rect = geometry.focus_rect(self.sw, self.sh, self.body, 2.6, W / 2, H / 2)
        out = geometry.to_focus(self.sw, self.sh, self.body, rect, self.range)
        self.assertEqual(out["scale"], 2.0)
        self.assertEqual(out["clamped"], ["scale"])
        cx, cy = geometry.centre(rect)
        ox, oy = geometry.centre(out["rect"])
        self.assertAlmostEqual(ox, cx, delta=1.0)
        self.assertAlmostEqual(oy, cy, delta=1.0)
        x0, y0, x1, y1 = out["rect"]
        self.assertTrue(0 <= x0 < x1 <= self.sw and 0 <= y0 < y1 <= self.sh)
        # The returned arguments show the returned rectangle.
        back = geometry.focus_rect(self.sw, self.sh, self.body, out["scale"], out["x"], out["y"])
        within(self, back, out["rect"], 1e-6)
        # Clamped up: a rectangle bigger than the range allows shrinks to scale 1.4.
        wide = geometry.focus_rect(self.sw, self.sh, self.body, 1.1, W / 2, H / 2)
        low = geometry.to_focus(self.sw, self.sh, self.body, wide, self.range)
        self.assertEqual(low["scale"], 1.4)
        self.assertEqual(low["clamped"], ["scale"])

    def focus_for(self, box, src=None):
        """`to_focus` for a tight box in source px of `src` (default: the wide source): expand, then convert."""
        sw, sh = src or (self.sw, self.sh)
        rect = geometry.expand(box, self.body[0] / self.body[1], sw, sh)
        return rect, geometry.to_focus(sw, sh, self.body, rect, self.range)

    def assert_in_region(self, out, src=None):
        sw, sh = src or (self.sw, self.sh)
        rx0, ry0, rx1, ry1 = geometry.cover_region(sw, sh, self.body)
        x0, y0, x1, y1 = out["rect"]
        for got, lo, hi in ((x0, rx0, rx1), (x1, rx0, rx1), (y0, ry0, ry1), (y1, ry0, ry1)):
            self.assertTrue(lo - 1e-6 <= got <= hi + 1e-6, (out["rect"], (rx0, ry0, rx1, ry1)))
        # The arguments show exactly that rectangle, and the origin stays inside the body.
        back = geometry.focus_rect(sw, sh, self.body, out["scale"], out["x"], out["y"])
        within(self, back, out["rect"], 1e-6)
        self.assertTrue(-1e-6 <= out["x"] <= self.body[0] + 1e-6 and -1e-6 <= out["y"] <= self.body[1] + 1e-6, out)

    def test_cover_region(self):
        W, H = self.body
        rx0, ry0, rx1, ry1 = geometry.cover_region(self.sw, self.sh, self.body)
        c = H / self.sh  # the wide source is limited by height
        self.assertAlmostEqual(rx1 - rx0, W / c)
        self.assertAlmostEqual((rx0 + rx1) / 2, self.sw / 2)
        self.assertEqual((ry0, ry1), (0.0, float(self.sh)))
        tw, th = self.fx["tall_src"]
        tx0, ty0, tx1, ty1 = geometry.cover_region(tw, th, self.body)
        c = W / tw  # the tall source is limited by width
        self.assertEqual((tx0, tx1), (0.0, float(tw)))
        self.assertEqual(ty0, 0.0)
        self.assertAlmostEqual(ty1, H / c)
        self.assertLess(ty1, th)

    def test_to_focus_keeps_the_clamped_rect_inside_the_cover_region(self):
        W, H = self.body
        rx0, ry0, rx1, ry1 = geometry.cover_region(self.sw, self.sh, self.body)
        c = max(W / self.sw, H / self.sh)
        w, h = W / (c * 3.0), H / (c * 3.0)
        corner = geometry.to_focus(self.sw, self.sh, self.body, (0.0, 0.0, w, h), self.range)  # a box implying scale 3
        self.assertEqual(corner["clamped"], ["scale", "x", "y"])
        self.assertAlmostEqual(corner["rect"][0], rx0)
        self.assertAlmostEqual(corner["rect"][1], ry0)
        self.assertAlmostEqual((corner["rect"][2] - corner["rect"][0]) / (corner["rect"][3] - corner["rect"][1]), W / H)
        self.assert_in_region(corner)
        far = geometry.to_focus(self.sw, self.sh, self.body, (self.sw - w, self.sh - h, self.sw, self.sh), self.range)
        self.assertEqual(far["clamped"], ["scale", "x", "y"])
        self.assertAlmostEqual(far["rect"][2], rx1)
        self.assertAlmostEqual(far["rect"][3], ry1)
        self.assert_in_region(far)

    def test_focus_clamped_into_cover_region_left_edge(self):
        rect, out = self.focus_for((0, 300, 600, 800))  # hugs x=0; the cover cannot show the first 300 px or so
        rx0 = geometry.cover_region(self.sw, self.sh, self.body)[0]
        self.assertLess(rect[0], rx0)
        self.assertEqual(out["clamped"], ["x"])
        self.assertAlmostEqual(out["rect"][0], rx0)
        self.assertAlmostEqual(out["scale"], geometry.rect_to_focus(self.sw, self.sh, self.body, rect)[0])
        self.assertAlmostEqual(out["rect"][2] - out["rect"][0], rect[2] - rect[0])  # shifted, not shrunk
        self.assertAlmostEqual(out["rect"][1], rect[1])
        self.assert_in_region(out)

    def test_focus_clamped_into_cover_region_right_edge(self):
        rect, out = self.focus_for((self.sw - 600, 300, self.sw, 800))
        rx1 = geometry.cover_region(self.sw, self.sh, self.body)[2]
        self.assertGreater(rect[2], rx1)
        self.assertEqual(out["clamped"], ["x"])
        self.assertAlmostEqual(out["rect"][2], rx1)
        self.assertAlmostEqual(out["rect"][2] - out["rect"][0], rect[2] - rect[0])
        self.assertAlmostEqual(out["rect"][1], rect[1])
        self.assert_in_region(out)

    def test_focus_unclamped_for_centred_box_reports_empty(self):
        rect, out = self.focus_for((660, 390, 1260, 690))
        self.assertEqual(out["clamped"], [])
        within(self, out["rect"], rect, 1e-6)
        self.assertTrue(self.range[0] <= out["scale"] <= self.range[1])
        self.assert_in_region(out)

    def test_focus_clamped_into_cover_region_bottom_edge_tall_source(self):
        # A tall source loses its bottom: a box low in the frame moves up and its scale rises into the range.
        tall = tuple(self.fx["tall_src"])
        rect, out = self.focus_for((100, 1400, 1000, 1900), tall)
        self.assertEqual(out["clamped"], ["scale", "y"])
        self.assertAlmostEqual(out["rect"][3], geometry.cover_region(*tall, self.body)[3])
        self.assert_in_region(out, tall)
        # A tall source shows its full width: a box at the left edge does not move on x.
        _, left = self.focus_for((0, 300, 700, 900), tall)
        self.assertNotIn("x", left["clamped"])
        self.assert_in_region(left, tall)

    def test_every_case_stays_in_region_unclamped(self):
        for case in self.fx["cases"]:
            with self.subTest(case["id"]):
                sw, sh = case["src"]
                rect = geometry.focus_rect(sw, sh, self.body, *case["focus"])
                out = geometry.to_focus(sw, sh, self.body, rect, self.range)
                self.assertEqual(out["clamped"], [])
                self.assert_in_region(out, (sw, sh))


class OverlapTests(unittest.TestCase):
    def test_iou_and_centre_inside(self):
        a = (0, 0, 10, 10)
        self.assertEqual(geometry.iou(a, a), 1.0)
        self.assertEqual(geometry.iou(a, (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(geometry.iou(a, (5, 0, 15, 10)), 50 / 150)
        self.assertEqual(geometry.iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)
        self.assertTrue(geometry.centre_inside((4, 4, 6, 6), a))
        self.assertTrue(geometry.centre_inside((8, 8, 12, 12), a))
        self.assertFalse(geometry.centre_inside((9, 9, 21, 21), a))
        self.assertFalse(geometry.centre_inside((-30, 4, -20, 6), a))


if __name__ == "__main__":
    unittest.main()
