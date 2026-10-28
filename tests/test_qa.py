"""Tests for modules.universal.qa: frame QA QA-01. From the tool folder:

    python3 -m unittest tests.test_qa -q

Offline: a FakeBackend answers from three canned tables (tests/fixtures/qa/frame_answers.json), the client cache is a
temp dir, and the renders are made by ffmpeg in temp dirs.
"""
import copy
import json
import socket
import subprocess
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from core.policy import load_policy
from jevlib import images
from jevlib.cache import Cache
from jevlib.client import Client
from jevlib.fake import FakeBackend, noul
from jevlib.request import Answer
from modules.universal import qa

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest_min.json").read_text(encoding="utf-8"))
SPEC = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
TABLES = json.loads((FIXTURES / "qa" / "frame_answers.json").read_text(encoding="utf-8"))["frames"]
CHAIN = TABLES[0]["p"]      # frame 4.5: caption_on_face 0.86, both gates up
OCCLUDED = TABLES[1]["p"]   # frame 19.5: text_occluded 0.95
CHECK_ONLY = TABLES[2]["p"]  # frame 30.0: nothing at 0.70, empty_region 0.55
TH = {"flag": 0.70, "check": 0.40, "gate": 0.50}


def run_ffmpeg(args):
    subprocess.run([images.FFMPEG, "-v", "error", "-y", *args], check=True)


def make_render(path, w=1080, h=1920, dur=3, fps=10):
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c=0x202020:s={w}x{h}:r={fps}:d={dur}", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-crf", "35", "-preset", "ultrafast", str(path)])
    return Path(path)


def make_jpg(path, w=300, h=520, color="0x336699"):
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}", "-frames:v", "1", "-update", "1", str(path)])
    return Path(path)


def make_spec(root, *, w=1080, h=1920, canvas="short", family="short"):
    spec = copy.deepcopy(SPEC)
    spec["project"]["root"] = str(root)
    spec["kit"]["manifest"] = str(FIXTURES / "manifest_min.json")
    spec["kit"]["canvas"] = canvas
    spec["format"]["family"] = family
    spec["canvas"].update(width=w, height=h, aspect=f"{w // 120}:{h // 120}")
    path = Path(root) / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return spec, path


def frame_responder(tables):
    """Answers every question of a request from `tables[i % len(tables)]`, `i` being the frame's number in its file name."""
    def responder(req, model):
        (path,) = req.images.values()
        p = tables[(int(Path(path).stem.split("_")[1]) - 1) % len(tables)]
        return {"answers": {q: noul(p[q]) for q in req.questions}, "model": model, "input_tokens": 4770, "latency_s": 0.01}
    return responder


def make_client(tmp, tables, *, jev_healthy=True):
    jev = FakeBackend("jev", frame_responder(tables), healthy=jev_healthy)
    local = FakeBackend("local", frame_responder(tables), healthy=False)
    return Client({"jev": jev, "local": local}, Cache(Path(tmp) / "cache"), log=lambda *a, **k: None), jev


def run_cli(*args, **kw):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = qa.cli_qa(*args, **kw)
    return code, out.getvalue(), err.getvalue()


def fake_row(t, file, flags, p, *, checks=()):
    """A frame row as `frame_rows` makes it, for write_outputs tests."""
    return {"t": t, "file": str(file), "qset": qa.QSET, "qset_version": 1, "backend": "jev", "model": "jev-1.14.0",
            "request_sha256": "ab" * 32, "route_reasons": [], "p": {n: p for n in flags}, "flags": list(flags),
            "checks": list(checks), "tokens": 4770, "latency_s": 0.5, "cached": False, "error": None, "detail": ""}


class ClassifyTests(unittest.TestCase):
    def test_classify_gate_blocks_caption_on_face(self):
        p = {"caption_on_face": 0.9, "face_visible": 0.3, "caption_visible": 0.98}
        self.assertEqual(qa.classify(p, TH), ([], []))
        self.assertEqual(qa.classify({**p, "face_visible": 0.5}, TH), (["caption_on_face"], []))
        self.assertEqual(qa.classify({**p, "face_visible": 0.9, "caption_visible": 0.49}, TH), ([], []))
        self.assertEqual(qa.classify({"caption_on_face": 0.9}, TH), ([], []))

    def test_classify_flag_and_check_bands(self):
        for value, want in ((0.70, (["overlapping_panels"], [])), (0.69, ([], ["overlapping_panels"])), (0.40, ([], ["overlapping_panels"])),
                            (0.39, ([], [])), (0.99, (["overlapping_panels"], []))):
            self.assertEqual(qa.classify({"overlapping_panels": value}, TH), want, value)

    def test_classify_ignores_gates_and_unknown_questions(self):
        p = {"face_visible": 0.99, "caption_visible": 0.99, "made_up": 0.99, "overlapping_panels": None}
        self.assertEqual(qa.classify(p, TH), ([], []))

    def test_classify_follows_the_thresholds_it_is_given(self):
        self.assertEqual(qa.classify({"overlapping_panels": 0.55}, {**TH, "flag": 0.5}), (["overlapping_panels"], []))

    def test_classify_matches_the_canned_tables(self):
        for table in TABLES:
            with self.subTest(t=table["t"]):
                self.assertEqual(qa.classify(table["p"], TH), (table["flags"], table["checks"]))


class QsetTests(unittest.TestCase):
    def test_qset_pins(self):
        from jevlib.request import load_qset
        qs = load_qset(qa.QSET)
        self.assertEqual((qs.id, qs.version, qs.route, qs.backend), ("qa_frame.short", 1, "jev", "jev"))
        self.assertEqual(qs.models, {"jev": "jev-1.14.0", "local": "pplx-decider-flash-v1.2"})
        self.assertEqual(qs.thresholds, {"jev": TH, "local": None})
        self.assertFalse(qs.has_local_calibration())
        self.assertFalse(qs.order_check)
        self.assertIsNone(qs.batch)

    def test_question_order_and_kit_neutral_wording(self):
        from jevlib.request import load_qset
        got = load_qset(qa.QSET).questions
        self.assertEqual(list(got)[3], "empty_region")
        self.assertEqual(list(got)[-2:], list(qa.GATES))
        self.assertEqual(len(qa.DEFECTS), 7)
        for word in ("yellow", "#"):
            self.assertNotIn(word, json.dumps(got).lower())

    def test_defects_are_the_seven_ids_in_order(self):
        self.assertEqual(len(qa.DEFECTS), 7)
        self.assertEqual(qa.GATES, ("face_visible", "caption_visible"))
        from jevlib.request import load_qset
        self.assertEqual(list(load_qset(qa.QSET).questions), [*qa.DEFECTS, *qa.GATES])


class RequestTests(unittest.TestCase):
    def test_request_shape(self):
        with tempfile.TemporaryDirectory() as td:
            f1, f2 = Path(td) / "f_0001.jpg", Path(td) / "f_0002.jpg"
            f1.write_bytes(b"one")
            f2.write_bytes(b"two")
            reqs = qa.qa_requests([(0.0, f1), (1.5, f2)], MANIFEST["qa_context"]["short"], "private")
        self.assertEqual(len(reqs), 2)
        for req in reqs:
            self.assertEqual(list(req.state), ["frame", "context"])
            self.assertEqual(len(req.questions), 9)
            self.assertEqual(len(req.images), 1)
            (ref,) = req.images
            self.assertTrue(ref.startswith("<<frame-"), ref)
            self.assertEqual(req.state["frame"], ref)
            self.assertEqual(req.image_refs_in_order(), [ref])
            self.assertEqual(req.state["context"], MANIFEST["qa_context"]["short"])
            self.assertEqual((req.sensitivity, req.qs.id), ("private", qa.QSET))
            self.assertFalse(req.has_box())
        self.assertNotEqual(list(reqs[0].images), list(reqs[1].images))
        self.assertEqual(reqs[0].tag, "t0.0")


class CliTests(unittest.TestCase):
    def test_cli_qa_writes_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "proj"
            root.mkdir()
            spec, spec_path = make_spec(root)
            render = make_render(root / "short.mp4")
            client, jev = make_client(td, [CHAIN, CHECK_ONLY])
            code, out, err = run_cli(render, spec_path, client=client)
            self.assertEqual((code, err), (0, ""), out)
            qa_dir = root / "work" / "jevm" / "qa" / "short"
            data = json.loads((qa_dir / "qa.json").read_text(encoding="utf-8"))
            self.assertEqual(list(data), ["meta", "frames", "decisions"])
            self.assertEqual([f["t"] for f in data["frames"]], [0.0, 1.5])
            self.assertEqual(data["frames"][0]["flags"], ["caption_on_face"])
            self.assertEqual(data["frames"][1]["checks"], ["empty_region"])
            self.assertTrue((qa_dir / "frames" / "f_0001.jpg").is_file())
            self.assertEqual(len(data["decisions"]), 1)
            d = data["decisions"][0]
            self.assertEqual(d["id"], "QA-01:t0.0")
            self.assertEqual(d["target"], {"kind": "frame", "id": "t0.0", "t": [0.0, 0.0]})
            self.assertEqual((d["lane"], d["status"], d["route_reasons"]), ("agent", "proposed", ["above_auto"]))
            self.assertEqual(d["value"], {"flags": ["caption_on_face"], "checks": [], "p": {"caption_on_face": 0.86}})
            self.assertEqual((d["p"], d["confidence"]), (0.86, {"metric": "noul_p", "value": 0.86}))
            self.assertEqual(d["evidence"]["qset"], qa.QSET)
            self.assertEqual(data["meta"]["frames"], 2)
            self.assertEqual(data["meta"]["flagged_frames"], 1)
            self.assertEqual(data["meta"]["backends_used"]["jev"]["requests"], 2)
            md = (qa_dir / "qa.md").read_text(encoding="utf-8")
            self.assertIn("jev-1.14.0", md)
            self.assertIn("| 0.0 | caption_on_face | 0.86 |", md)
            self.assertNotIn("empty_region", md)   # the check band is logged, not shown
            self.assertTrue((qa_dir / "flagged.jpg").is_file())
            self.assertIn("flagged", out)
            # the QA context comes from the manifest the spec points at; frames carry the strictest asset sensitivity
            req = jev.calls[0][0]
            self.assertEqual(req.state["context"], MANIFEST["qa_context"]["short"])
            self.assertEqual(req.sensitivity, "private")
            self.assertEqual(jev.calls[0][1], "jev-1.14.0")

            # a clean re-run on a fresh cache removes the stale sheet
            client2, _ = make_client(Path(td) / "second", [CHECK_ONLY])
            code, _, _ = run_cli(render, spec_path, client=client2)
            self.assertEqual(code, 0)
            self.assertFalse((qa_dir / "flagged.jpg").exists())
            data = json.loads((qa_dir / "qa.json").read_text(encoding="utf-8"))
            self.assertEqual((data["decisions"], data["meta"]["flagged_frames"]), ([], 0))
            self.assertIn("No frame flagged", (qa_dir / "qa.md").read_text(encoding="utf-8"))

    def test_cli_qa_aspect_mismatch_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, spec_path = make_spec(root)
            render = make_render(root / "wide.mp4", w=1920, h=1080, dur=1)
            client, jev = make_client(td, [CHECK_ONLY])
            code, _, err = run_cli(render, spec_path, client=client)
            self.assertEqual(code, 2)
            self.assertIn("1920x1080", err)
            self.assertIn("1080x1920", err)
            self.assertEqual(jev.calls, [])
            self.assertFalse((root / "work").exists())

    def test_cli_qa_long_form_exits_2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, spec_path = make_spec(root, w=1920, h=1080, canvas="long", family="long")
            render = make_render(root / "long.mp4", w=1920, h=1080, dur=1)
            client, jev = make_client(td, [CHECK_ONLY])
            code, _, err = run_cli(render, spec_path, client=client)
            self.assertEqual(code, 2)
            self.assertIn("no frame QA set for canvas", err)
            self.assertEqual(jev.calls, [])

    def test_cli_qa_missing_inputs_exit_2(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, spec_path = make_spec(root)
            render = make_render(root / "r.mp4", dur=1)
            self.assertEqual(run_cli(root / "nope.mp4", spec_path)[0], 2)
            self.assertEqual(run_cli(render, root / "nope.json")[0], 2)
            broken = root / "broken.mp4"
            broken.write_text("not a video", encoding="utf-8")
            self.assertEqual(run_cli(broken, spec_path)[0], 2)
            bad_spec = root / "bad.json"
            bad_spec.write_text("{}", encoding="utf-8")
            self.assertEqual(run_cli(render, bad_spec)[0], 2)
            self.assertEqual(run_cli(render, spec_path, every=0.0)[0], 2)

    def test_cli_qa_project_path_with_space(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "my project"
            root.mkdir()
            _, spec_path = make_spec(root)
            render = make_render(root / "cut v1.mp4", dur=1)
            client, _ = make_client(td, [CHAIN])
            code, _, err = run_cli(render, spec_path, client=client)
            self.assertEqual((code, err), (0, ""))
            qa_dir = root / "work" / "jevm" / "qa" / "cut v1"
            for name in ("qa.json", "qa.md", "flagged.jpg", "frames/f_0001.jpg"):
                self.assertTrue((qa_dir / name).is_file(), name)

    def test_cli_qa_render_shorter_than_every_gives_one_frame(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, spec_path = make_spec(root)
            render = make_render(root / "tiny.mp4", dur=1)
            client, jev = make_client(td, [CHECK_ONLY])
            code, _, _ = run_cli(render, spec_path, every=5.0, client=client)
            self.assertEqual(code, 0)
            data = json.loads((root / "work/jevm/qa/tiny/qa.json").read_text(encoding="utf-8"))
            self.assertEqual([f["t"] for f in data["frames"]], [0.0])
            self.assertEqual(len(jev.calls), 1)

    def test_cli_qa_dry_run_and_budget_send_nothing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, spec_path = make_spec(root)
            render = make_render(root / "r.mp4", dur=1)
            client, jev = make_client(td, [CHAIN])
            self.assertEqual(run_cli(render, spec_path, client=client, dry_run=True)[0], 0)
            self.assertEqual(run_cli(render, spec_path, client=client, budget=0.0)[0], 3)
            self.assertEqual(jev.calls, [])
            self.assertFalse((root / "work/jevm/qa/r/qa.json").exists())

    def test_cli_qa_jev_down_counts_errored_frames_and_exits_0(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, spec_path = make_spec(root)
            render = make_render(root / "r.mp4", dur=1)
            client, _ = make_client(td, [CHAIN], jev_healthy=False)
            code, _, _ = run_cli(render, spec_path, client=client)
            self.assertEqual(code, 0)
            qa_dir = root / "work/jevm/qa/r"
            data = json.loads((qa_dir / "qa.json").read_text(encoding="utf-8"))
            self.assertEqual(data["frames"][0]["error"], "jev_unavailable")
            self.assertEqual((data["decisions"], data["meta"]["errored_frames"]), ([], 1))
            self.assertIn("1 errored", (qa_dir / "qa.md").read_text(encoding="utf-8"))
            self.assertFalse((qa_dir / "flagged.jpg").exists())


class OutputTests(unittest.TestCase):
    def test_flagged_sheet_caps_at_8(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            rows = []
            for i in range(10):
                f = make_jpg(td / f"f_{i + 1:04d}.jpg", color=f"0x{i * 20:02x}6699")
                rows.append(fake_row(i * 1.5, f, ["overlapping_panels", "hard_to_read"], 0.71 + i * 0.01))
            rows[3]["p"] = {"overlapping_panels": 0.99, "hard_to_read": 0.2}   # one early frame outranks the later ones
            rows.append(fake_row(99.0, make_jpg(td / "f_0011.jpg"), [], 0.0, checks=["overlapping_panels"]))
            with mock.patch.object(qa.images, "tile_sheet", wraps=images.tile_sheet) as sheet:
                meta = qa.write_outputs(td / "out", rows, [], {"render": "x.mp4"})
            tiles, out = sheet.call_args.args[:2]
            self.assertEqual(len(tiles), 8)
            ts = [float(label.split("=")[1].split("S")[0]) for _, label in tiles]
            self.assertEqual(ts, sorted(ts))
            self.assertIn(4.5, ts)                               # the 0.99 frame is kept
            self.assertNotIn(0.0, ts)                            # the two weakest are dropped
            self.assertNotIn(1.5, ts)
            self.assertEqual(tiles[0][1], "T=3.0S overlapping_panels")      # label: time and first flag
            self.assertEqual(Path(out), td / "out" / "flagged.jpg")
            self.assertTrue((td / "out" / "flagged.jpg").is_file())
            self.assertEqual((meta["frames"], meta["flagged_frames"], meta["check_frames"]), (11, 10, 1))
            self.assertEqual(meta["render"], "x.mp4")

    def test_write_outputs_tokens_dollars_and_errors(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a = fake_row(0.0, td / "a.jpg", [], 0.0)
            b = dict(fake_row(1.5, td / "b.jpg", [], 0.0), cached=True)
            c = dict(fake_row(3.0, td / "c.jpg", [], 0.0), error="jev_error", backend="none", model="", tokens=None)
            meta = qa.write_outputs(td / "out", [a, b, c], [], {})
            self.assertEqual((meta["input_tokens"], meta["cached_frames"], meta["errored_frames"]), (4770, 1, 1))
            self.assertEqual(meta["errors"], {"jev_error": 1})
            self.assertAlmostEqual(meta["usd"], 4770 * 0.042 / 1e6, places=6)
            head = (td / "out" / "qa.md").read_text(encoding="utf-8").splitlines()
            line = next(x for x in head if "frames" in x and "tokens" in x)
            for part in ("jev", "jev-1.14.0", "3 frames", "0 flagged", "1 errored", "4770", "<$0.001"):
                self.assertIn(part, line)


class DecideTests(unittest.TestCase):
    def test_decide_makes_no_network_call(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "f_0001.jpg"
            f.write_bytes(b"x")
            reqs = qa.qa_requests([(4.5, f), (19.5, f)], MANIFEST["qa_context"]["short"], "public")
            answers = [Answer(r, {q: noul(t["p"][q]) for q in r.questions}, "jev", "jev-1.14.0", "jev-1.14.0", 4770, 0.6,
                              False, "cd" * 32) for r, t in zip(reqs, TABLES)]
            boom = AssertionError("decide must not touch the network")
            with mock.patch.object(socket.socket, "connect", side_effect=boom), \
                    mock.patch.object(urllib.request, "urlopen", side_effect=boom):
                rows = qa.frame_rows([(4.5, f), (19.5, f)], answers)
                decisions = qa.qa_decisions(rows, load_policy())
        self.assertEqual([d["id"] for d in decisions], ["QA-01:t4.5", "QA-01:t19.5"])
        self.assertEqual(decisions[1]["value"], {"flags": ["text_occluded"], "checks": ["empty_region"],
                                                 "p": {"text_occluded": 0.95}})
        self.assertEqual(decisions[1]["evidence"]["request_sha256"], "cd" * 32)
        self.assertEqual(decisions[0]["evidence"]["label"], "Default")

    def test_frame_rows_keep_errors_and_skip_them_in_decisions(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "f_0001.jpg"
            f.write_bytes(b"x")
            (req,) = qa.qa_requests([(0.0, f)], "ctx", "public")
            err = Answer(req, None, "none", "", None, 0, 0.0, False, "", error="jev_unavailable", detail="no key",
                         route_reasons=[])
            (row,) = qa.frame_rows([(0.0, f)], [err])
        self.assertEqual((row["error"], row["flags"], row["checks"], row["p"]), ("jev_unavailable", [], [], {}))
        self.assertEqual(qa.qa_decisions([row], load_policy()), [])

    def test_decisions_keep_the_answer_route_reasons(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "f_0001.jpg"
            f.write_bytes(b"x")
            (req,) = qa.qa_requests([(0.0, f)], "ctx", "public")
            ans = Answer(req, {q: noul(CHAIN[q]) for q in req.questions}, "jev", "jev-1.14.0", None, 10, 0.1, False,
                         "ef" * 32, route_reasons=["backend_fallback"])
            rows = qa.frame_rows([(0.0, f)], [ans])
            (d,) = qa.qa_decisions(rows, load_policy())
        self.assertEqual(d["route_reasons"], ["above_auto", "backend_fallback"])


if __name__ == "__main__":
    unittest.main()
