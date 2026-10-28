"""Image fitting, frame extraction and bitmap labels. Synthetic inputs are made with ffmpeg in temp dirs."""
import hashlib
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from jevlib import images
from jevlib.images import (FFMPEG, Fit, ImageLimitError, check_limits, draw_rects, fit_for_jev, frame_at, has_exif,
                           image_size, label_png, mean_luma, probe_video, sample_times, tile_sheet, unfit_box)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True)


def make_image(path, w, h, color="0x336699"):
    """A solid-colour PNG or JPEG (by suffix), made by ffmpeg."""
    run(["-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}", "-frames:v", "1", "-update", "1", str(path)])
    return Path(path)


def make_luma_video(path, w=320, h=320, fps=10, dur=3):
    """Frame N has luma 8*N, so the luma of an extracted frame says which frame was cut."""
    run(["-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps}:d={dur}", "-vf", "geq=lum='8*N':cb=128:cr=128",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "12", "-g", "1", str(path)])
    return Path(path)


def splice_exif(src, dst):
    """Copy a JPEG with an APP1 'Exif\\0\\0' segment spliced in right after SOI (a valid TIFF block, Orientation = 6)."""
    data = Path(src).read_bytes()
    assert data[:2] == b"\xff\xd8"
    tiff = b"II*\x00\x08\x00\x00\x00" + struct.pack("<H", 1) + struct.pack("<HHII", 0x0112, 3, 1, 6) + struct.pack("<I", 0)
    payload = b"Exif\x00\x00" + tiff
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    Path(dst).write_bytes(data[:2] + seg + data[2:])
    return Path(dst)


class TmpCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)


class ImageSizeTests(TmpCase):
    def test_image_size_png_and_jpg(self):
        self.assertEqual(image_size(make_image(self.tmp / "a.png", 320, 240)), (320, 240, "png"))
        self.assertEqual(image_size(make_image(self.tmp / "a.jpg", 321, 241)), (321, 241, "jpg"))

    def test_image_size_jpeg_with_exif_segment(self):
        src = make_image(self.tmp / "a.jpg", 400, 300)
        self.assertEqual(image_size(splice_exif(src, self.tmp / "e.jpg")), (400, 300, "jpg"))

    def test_image_size_rejects_other_files(self):
        bad = self.tmp / "x.bin"
        bad.write_bytes(b"not an image at all, just bytes")
        with self.assertRaises(ImageLimitError):
            image_size(bad)

    def test_has_exif(self):
        src = make_image(self.tmp / "a.jpg", 400, 300)
        self.assertFalse(has_exif(src))
        self.assertTrue(has_exif(splice_exif(src, self.tmp / "e.jpg")))
        self.assertFalse(has_exif(make_image(self.tmp / "a.png", 400, 300)))

    def test_check_limits(self):
        check_limits(make_image(self.tmp / "ok.png", 882, 1568))
        for name, w, h in (("narrow", 200, 300), ("wide", 1600, 300), ("area", 1568, 1569)):
            with self.subTest(name):
                with self.assertRaises(ImageLimitError):
                    check_limits(make_image(self.tmp / f"{name}.png", w, h))
        # each side within 280-1568 but area under 78,400 is impossible (280*280 = 78,400); the area floor and
        # ceiling are checked anyway
        self.assertEqual((images.MIN_AREA, images.MAX_AREA), (78_400, 2_458_624))


class FitTests(TmpCase):
    def test_fit_landscape_1920x1080(self):
        out, fit = fit_for_jev(make_image(self.tmp / "l.png", 1920, 1080), self.tmp / "l_out.jpg")
        self.assertEqual(image_size(out), (1568, 882, "jpg"))
        self.assertEqual((fit.src_w, fit.src_h, fit.w, fit.h, fit.pad), (1920, 1080, 1568, 882, (0, 0, 0, 0)))
        self.assertAlmostEqual(fit.scale, 1568 / 1920, places=4)
        check_limits(out)

    def test_fit_portrait_1080x1920(self):
        out, fit = fit_for_jev(make_image(self.tmp / "p.png", 1080, 1920), self.tmp / "p_out.jpg")
        self.assertEqual(image_size(out), (882, 1568, "jpg"))
        self.assertEqual((fit.w, fit.h), (882, 1568))

    def test_fit_small_image_padded(self):
        out, fit = fit_for_jev(make_image(self.tmp / "s.png", 200, 100), self.tmp / "s_out.jpg")
        w, h, _ = image_size(out)
        self.assertGreaterEqual(min(w, h), images.MIN_SIDE)
        self.assertTrue(images.MIN_AREA <= w * h <= images.MAX_AREA)
        box = unfit_box({"left": 0, "top": 0, "right": 1, "bottom": 1}, fit)
        for got, want in zip(box, (0, 0, 200, 100)):
            self.assertAlmostEqual(got, want, delta=1)

    def test_fit_extreme_aspect_is_padded_and_box_maps_back(self):
        out, fit = fit_for_jev(make_image(self.tmp / "w.png", 3000, 100), self.tmp / "w_out.jpg")
        w, h, _ = image_size(out)
        self.assertEqual((w, h), (fit.w, fit.h))
        self.assertGreaterEqual(min(w, h), images.MIN_SIDE)
        self.assertLessEqual(max(w, h), images.MAX_SIDE)
        self.assertGreater(fit.pad[1] + fit.pad[3], 0)
        self.assertEqual(fit.pad[0] + fit.pad[2], 0)
        full = unfit_box({"left": 0, "top": 0, "right": 1, "bottom": 1}, fit)
        self.assertEqual(full, (0.0, 0.0, 3000.0, 100.0))  # padding is clamped away
        # the content band, in sent fractions, maps to the whole source
        band = unfit_box({"left": 0, "top": fit.pad[1] / fit.h, "right": 1, "bottom": 1 - fit.pad[3] / fit.h}, fit)
        for got, want in zip(band, (0, 0, 3000, 100)):
            self.assertAlmostEqual(got, want, delta=1)

    def test_unfit_box_scaled_frame(self):
        _, fit = fit_for_jev(make_image(self.tmp / "l.png", 1920, 1080), self.tmp / "l_out.jpg")
        x0, y0, x1, y1 = unfit_box({"left": 0.25, "top": 0.5, "right": 0.75, "bottom": 1.0}, fit)
        for got, want in zip((x0, y0, x1, y1), (480, 540, 1440, 1080)):
            self.assertAlmostEqual(got, want, delta=1)

    def test_fit_strips_exif(self):
        src = splice_exif(make_image(self.tmp / "a.jpg", 400, 300), self.tmp / "e.jpg")
        self.assertTrue(has_exif(src))
        out, fit = fit_for_jev(src, self.tmp / "e_out.jpg")
        self.assertFalse(has_exif(out))
        self.assertEqual(image_size(out)[:2], (fit.w, fit.h))
        self.assertEqual((fit.src_w, fit.src_h), (400, 300))  # Orientation = 6 is ignored: the source grid is kept
        self.assertGreaterEqual(min(fit.w, fit.h), images.MIN_SIDE)

    def test_fit_exif_jpeg_within_limits_keeps_size(self):
        src = splice_exif(make_image(self.tmp / "a.jpg", 600, 400), self.tmp / "e.jpg")
        out, fit = fit_for_jev(src, self.tmp / "e_out.jpg")
        self.assertEqual(image_size(out), (600, 400, "jpg"))
        self.assertEqual((fit.scale, fit.pad), (1.0, (0, 0, 0, 0)))
        self.assertFalse(has_exif(out))

    def test_fit_passthrough_keeps_bytes(self):
        src = make_image(self.tmp / "ok.jpg", 882, 1568)
        self.assertFalse(has_exif(src))
        out, fit = fit_for_jev(src, self.tmp / "ok_out.jpg")
        self.assertEqual(file_sha256(out), file_sha256(src))
        self.assertEqual(fit, Fit(882, 1568, 1.0, (0, 0, 0, 0), 882, 1568))

    def test_fit_passthrough_png_keeps_bytes(self):
        src = make_image(self.tmp / "ok.png", 640, 480)
        out, _ = fit_for_jev(src, self.tmp / "ok_out.png")
        self.assertEqual(file_sha256(out), file_sha256(src))


class VideoTests(TmpCase):
    def test_probe_video(self):
        info = probe_video(make_luma_video(self.tmp / "v.mp4", 320, 240, fps=10, dur=3))
        self.assertEqual((info["width"], info["height"]), (320, 240))
        self.assertAlmostEqual(info["duration_s"], 3.0, delta=0.15)
        self.assertAlmostEqual(info["fps"], 10.0, places=2)

    def test_frame_at_is_exact_seek(self):
        video = make_luma_video(self.tmp / "v.mp4")
        out, fit = frame_at(video, 1.5, self.tmp / "f.jpg")
        self.assertTrue(out.is_file())
        self.assertTrue(114 <= mean_luma(out) <= 126, mean_luma(out))
        self.assertEqual(fit.pad, (0, 0, 0, 0))
        self.assertEqual((fit.src_w, fit.src_h), (320, 320))
        self.assertEqual(image_size(out)[:2], (fit.w, fit.h))
        # a neighbouring frame is clearly different, so the window above does prove the frame
        other, _ = frame_at(video, 1.7, self.tmp / "g.jpg")
        self.assertGreater(mean_luma(other) - mean_luma(out), 10)

    def test_frame_at_long_edge_and_even_dims(self):
        video = make_luma_video(self.tmp / "v.mp4", 1080, 1920, fps=5, dur=1)
        out, fit = frame_at(video, 0.4, self.tmp / "f.jpg")
        self.assertEqual(image_size(out), (882, 1568, "jpg"))
        out2, fit2 = frame_at(video, 0.4, self.tmp / "f2.jpg", long_edge=800)
        self.assertEqual(image_size(out2)[:2], (450, 800))
        self.assertEqual((fit2.src_w, fit2.src_h, fit2.w, fit2.h), (1080, 1920, 450, 800))

    def test_frame_at_path_with_space(self):
        d = self.tmp / "a b"
        d.mkdir()
        video = make_luma_video(d / "my video.mp4")
        out, _ = frame_at(video, 0.5, d / "out frame.jpg")
        self.assertTrue(out.is_file())
        self.assertGreater(out.stat().st_size, 0)

    def test_frame_at_past_the_end_raises(self):
        video = make_luma_video(self.tmp / "v.mp4", dur=1)
        with self.assertRaises(RuntimeError):
            frame_at(video, 30.0, self.tmp / "f.jpg")

    def test_sample_times(self):
        times = sample_times(45.2, 1.5)
        self.assertEqual(len(times), 31)
        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 45.0)

    def test_sample_times_short_render(self):
        self.assertEqual(sample_times(1.0, 1.5), [0.0])
        self.assertEqual(sample_times(0.0, 1.5), [0.0])

    def test_sample_times_rejects_bad_interval(self):
        with self.assertRaises(ValueError):
            sample_times(10.0, 0)


class LabelTests(TmpCase):
    def test_label_png(self):
        out = label_png("T=1.5S crop", self.tmp / "l.png", scale=2)
        self.assertEqual(out, self.tmp / "l.png")
        w, h, kind = image_size(out)
        self.assertEqual(kind, "png")
        self.assertEqual(h, (7 + 4) * 2)
        self.assertEqual(w, (len("T=1.5S crop") * 6 - 1 + 4) * 2)

    def test_tile_sheet_labels(self):
        tiles = [(make_image(self.tmp / f"t{i}.jpg", 882, 1568, color=c), f"T={i}.0S flag")
                 for i, c in enumerate(("red", "green", "blue"))]
        out = tile_sheet(tiles, self.tmp / "sheet.jpg")
        self.assertTrue(out.is_file())
        w, h, kind = image_size(out)
        self.assertEqual(kind, "jpg")
        self.assertGreaterEqual(w, 300)
        self.assertEqual(w, 3 * 300 + 2 * 4)  # three columns, 4 px spacing between them (tile filter `padding`)
        self.assertEqual(h, round(300 * 1568 / 882 / 2) * 2)  # one row

    def test_tile_sheet_wraps_rows_and_mixed_sizes(self):
        tiles = [(make_image(self.tmp / "a.png", 400, 300), "A"), (make_image(self.tmp / "b.png", 300, 600), "B"),
                 (make_image(self.tmp / "c.png", 500, 500), "C")]
        out = tile_sheet(tiles, self.tmp / "sheet.png", cols=2, tile_w=200)
        w, h, kind = image_size(out)
        self.assertEqual(kind, "png")
        self.assertEqual(w, 2 * 200 + 4)
        self.assertEqual(h, 2 * 400 + 4)  # tallest tile (200x400) sets the cell height

    def test_tile_sheet_empty(self):
        self.assertIsNone(tile_sheet([], self.tmp / "none.jpg"))
        self.assertFalse((self.tmp / "none.jpg").exists())

    def test_draw_rects_writes_image(self):
        src = make_image(self.tmp / "s.png", 882, 1568, color="black")
        out = draw_rects(src, [((100, 200, 500, 700), "crop"), ((600, 900, 880, 1500), "")],
                         self.tmp / "d.jpg")
        self.assertTrue(out.is_file())
        self.assertEqual(image_size(out), (882, 1568, "jpg"))
        self.assertGreater(mean_luma(out), mean_luma(src) + 0.1)  # boxes were drawn on black

    def test_draw_rects_clamps_and_handles_none(self):
        src = make_image(self.tmp / "s.png", 400, 300, color="black")
        out = draw_rects(src, [((-50, -20, 900, 700), "big")], self.tmp / "d1.jpg")
        self.assertEqual(image_size(out)[:2], (400, 300))
        out = draw_rects(src, [], self.tmp / "d2.jpg")
        self.assertEqual(image_size(out)[:2], (400, 300))

    def test_draw_rects_path_with_space(self):
        d = self.tmp / "a b"
        d.mkdir()
        src = make_image(d / "my src.png", 400, 300, color="black")
        out = draw_rects(src, [((10, 10, 200, 150), "a b")], d / "my out.jpg")
        self.assertTrue(out.is_file())

    def test_mean_luma_black_and_white(self):
        black = make_image(self.tmp / "k.png", 320, 320, color="black")
        white = make_image(self.tmp / "w.png", 320, 320, color="white")
        self.assertLess(mean_luma(black), 4)
        self.assertGreater(mean_luma(white), 250)


if __name__ == "__main__":
    unittest.main()
