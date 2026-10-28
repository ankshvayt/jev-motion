"""The plan and crop runner and the `jevm.py` command line (stdlib unittest). From the tool folder:

    python3 -m unittest tests.test_runner -q

Offline: every test runs a real `Client` over `FakeBackend`s and a temporary cache, never `make_client`. Each test builds
a project in a temporary folder whose name has a space, from tests/fixtures/spec_min.json and the manifest fixture.
Image and video sources are made with ffmpeg. The command-line tests patch the command a flag reaches and check the
arguments it gets. The eval tests run the real harness (`evals/harness.py`) in `--offline` mode on its scripted answers.
"""
import contextlib
import datetime
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jevm
from core import decisions_io, runner
from jevlib.cache import Cache
from jevlib.client import Client
from jevlib.fake import FakeBackend, box, choice, noul
from jevlib.images import FFMPEG, image_size

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOOL = Path(__file__).resolve().parent.parent
NEED2 = [0.95, 0.60, 0.10, 0.10]  # per caption group in the one batch: with K=2 the first is clearly in, the second is the edge


def respond(req, model):
    """Answers every qid by its type. Highlight need2 varies by group so the budget has an edge; the rest are fixed."""
    answers = {}
    for qid, question in req.questions.items():
        kind = question["type"]
        if qid.startswith("need2_"):
            answers[qid] = noul(NEED2[int(qid.split("_")[1])])
        elif kind == "noul":
            answers[qid] = noul(0.9)
        elif kind == "choice":
            names = list(question["criteria"])
            rest = 0.1 / max(1, len(names) - 1)
            answers[qid] = choice({n: (0.9 if k == 0 else rest) for k, n in enumerate(names)})
        elif kind == "box":
            answers[qid] = box(0.3, 0.3, 0.7, 0.7)
        else:
            raise KeyError(kind)
    return {"answers": answers, "model": model, "input_tokens": 100, "latency_s": 0.0}


def run_ffmpeg(*args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True)


def call(fn, *args, **kwargs):
    """`fn` with stdout and stderr captured: (result, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = fn(*args, **kwargs)
    return result, out.getvalue(), err.getvalue()


class RunnerCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.root = self.tmp / "project dir"
        (self.root / "assets").mkdir(parents=True)
        (self.root / "sources").mkdir()
        (self.root / "sources" / "report.md").write_text("In my test of 50 invoices, it got 47 right.\n", encoding="utf-8")
        run_ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=800x450", "-frames:v", "1", str(self.root / "assets" / "shot.png"))
        self.spec_path = self.write_spec()
        self.work = self.root / "work" / "jevm"

    def write_spec(self, mutate=None):
        spec = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
        spec["project"]["root"] = str(self.root)
        spec["kit"]["manifest"] = str(FIXTURES / "manifest_min.json")
        spec["constraints"]["highlight"]["density"] = 0.5
        if mutate:
            mutate(spec)
        path = self.work_dir() / "spec.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        return path

    def work_dir(self):
        return self.root / "work" / "jevm"

    def client(self, *, jev_healthy=True, cache="cache"):
        self.jev = FakeBackend("jev", respond, healthy=jev_healthy)
        self.local = FakeBackend("local", respond, healthy=False)
        self.logs = []
        return Client({"jev": self.jev, "local": self.local}, Cache(self.tmp / cache), log=self.logs.append)

    def decisions(self):
        return json.loads((self.work / "decisions.json").read_text(encoding="utf-8"))


class PlanTests(RunnerCase):
    def test_plan_writes_decisions_json_and_md(self):
        rc, out, _ = call(runner.plan, self.spec_path, client=self.client())
        self.assertEqual(rc, 0)
        doc = self.decisions()
        self.assertEqual({r["module"] for r in doc["decisions"]}, {"universal.captions", "universal.claims"})
        self.assertEqual({r["decision"] for r in doc["decisions"]}, {"CAP-02", "CAP-03", "STY-06", "QA-05"})
        jev = doc["backends_used"]["jev"]
        self.assertEqual(jev["requests"], len(self.jev.calls))
        self.assertGreater(jev["requests"], 0)
        self.assertEqual(jev["models"]["text"], "jev-1.13.0")
        self.assertEqual(doc["backends_used"]["local"]["requests"], 0)
        self.assertEqual(doc["project_id"], "jevm-fixture")
        self.assertEqual(doc["kit"]["id"], "example")
        self.assertEqual(len(doc["spec_sha256"]), 64)
        md = (self.work / "decisions.md").read_text(encoding="utf-8")
        self.assertIn("universal.captions", md)
        self.assertIn("jev-1.13.0", md)
        for lane in ("auto", "agent", "human"):
            self.assertIn(f"lane {lane}: ", out)
        self.assertIn(str(self.work / "decisions.json"), out)

    def test_plan_dry_run_sends_nothing_exit_0(self):
        client = self.client()
        rc, _, _ = call(runner.plan, self.spec_path, dry_run=True, client=client)
        self.assertEqual(rc, 0)
        self.assertEqual(self.jev.calls + self.local.calls, [])
        self.assertFalse((self.work / "decisions.json").exists())
        self.assertTrue(any(line.startswith("jev: ") for line in self.logs))
        self.assertIn("dry run: nothing sent", self.logs)

    def test_plan_over_budget_exit_3_sends_nothing(self):
        rc, _, _ = call(runner.plan, self.spec_path, budget=0.0, client=self.client())
        self.assertEqual(rc, 3)
        self.assertEqual(self.jev.calls + self.local.calls, [])
        self.assertFalse((self.work / "decisions.json").exists())

    def test_plan_spec_budget_is_the_default_and_budget_overrides_it(self):
        path = self.write_spec(lambda s: s["decide"].update(budget_usd=0.0))
        rc, _, _ = call(runner.plan, path, client=self.client())
        self.assertEqual(rc, 3)
        rc, _, _ = call(runner.plan, path, budget=1.0, client=self.client())
        self.assertEqual(rc, 0)

    def test_plan_stage_filter_runs_one_module(self):
        rc, _, _ = call(runner.plan, self.spec_path, stage="claims", client=self.client())
        self.assertEqual(rc, 0)
        self.assertEqual({r["module"] for r in self.decisions()["decisions"]}, {"universal.claims"})
        self.assertEqual({req.qs.id for req, _ in self.jev.calls}, {"stat_check"})
        rc, _, _ = call(runner.plan, self.spec_path, stage="captions", client=self.client())
        self.assertEqual(rc, 0)
        self.assertEqual({req.qs.id for req, _ in self.jev.calls}, {"highlight"})
        # the captions run replaced only its own rows: the claims rows from the first run are still there
        self.assertEqual({r["module"] for r in self.decisions()["decisions"]}, {"universal.captions", "universal.claims"})

    def test_plan_only_picks_the_module_that_decides_the_id(self):
        call(runner.plan, self.spec_path, only=["STY-06"], client=self.client())
        self.assertEqual({r["module"] for r in self.decisions()["decisions"]}, {"universal.claims"})
        call(runner.plan, self.spec_path, only=["CAP-03"], stage="captions", client=self.client())
        self.assertEqual({r["module"] for r in self.decisions()["decisions"]}, {"universal.captions", "universal.claims"})

    def test_plan_unknown_stage_or_id_exits_2(self):
        for kwargs, word in (({"stage": "nope"}, "stage"), ({"only": ["ZZZ-99"]}, "ZZZ-99")):
            rc, _, err = call(runner.plan, self.spec_path, client=self.client(), **kwargs)
            self.assertEqual(rc, 2)
            self.assertIn(word, err)
            self.assertEqual(self.jev.calls, [])

    def test_plan_stage_and_only_must_both_hold(self):
        rc, out, _ = call(runner.plan, self.spec_path, stage="claims", only=["CAP-02"], client=self.client())
        self.assertEqual(rc, 0)
        self.assertIn("nothing to do", out)
        self.assertEqual(self.jev.calls, [])

    def test_plan_rerun_keeps_resolved_rows(self):
        self.assertEqual(call(runner.plan, self.spec_path, client=self.client())[0], 0)
        auto = [r for r in self.decisions()["decisions"] if r["lane"] == "auto"]
        self.assertTrue(auto, "the fixture answers should leave at least one auto row")
        target = auto[0]["id"]
        rc, _, _ = call(jevm.main, ["resolve", str(self.work / "decisions.json"), target, "--accept", "--by", "agent"])
        self.assertEqual(rc, 0)
        client = self.client()  # same cache folder: the second run is answered from the cache
        self.assertEqual(call(runner.plan, self.spec_path, client=client)[0], 0)
        self.assertEqual(self.jev.calls, [])
        row = next(r for r in self.decisions()["decisions"] if r["id"] == target)
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["resolved"]["by"], "agent")
        ids = [r["id"] for r in self.decisions()["decisions"]]
        self.assertEqual(len(ids), len(set(ids)))
        # the cache hits are added to the totals; the live request count did not grow
        self.assertGreater(self.decisions()["backends_used"]["jev"]["cache_hits"], 0)

    def test_plan_manifest_sha_mismatch_exit_2(self):
        path = self.write_spec(lambda s: s["kit"].update(manifest_sha256="0" * 64))
        client = self.client()
        rc, _, err = call(runner.plan, path, client=client)
        self.assertEqual(rc, 2)
        self.assertIn("jevm kit build", err)
        self.assertIn("jevm spec build", err)
        self.assertEqual(self.jev.calls, [])
        self.assertFalse((self.work / "decisions.json").exists())

    def test_plan_changed_manifest_file_exit_2(self):
        copy_path = self.root / "manifest.json"
        manifest = json.loads((FIXTURES / "manifest_min.json").read_text(encoding="utf-8"))
        copy_path.write_text(json.dumps(manifest), encoding="utf-8")
        built_against = runner.file_sha256(copy_path)
        path = self.write_spec(lambda s: s["kit"].update(manifest=str(copy_path), manifest_sha256=built_against))
        self.assertEqual(call(runner.plan, path, dry_run=True, client=self.client())[0], 0)
        manifest["kit"]["name"] = "Changed"
        copy_path.write_text(json.dumps(manifest), encoding="utf-8")
        rc, _, err = call(runner.plan, path, dry_run=True, client=self.client())
        self.assertEqual(rc, 2)
        self.assertIn("changed since this spec was built", err)

    def test_plan_bad_spec_or_missing_file_exit_2(self):
        rc, _, err = call(runner.plan, self.tmp / "missing.json", client=self.client())
        self.assertEqual(rc, 2)
        self.assertIn("cannot read", err)
        bad = self.write_spec(lambda s: s["units"]["caption_groups"][0].update(words=["w0", "w999"]))
        rc, _, err = call(runner.plan, bad, client=self.client())
        self.assertEqual(rc, 2)
        self.assertIn("jevm spec validate", err)

    def test_plan_unreadable_decisions_file_is_never_overwritten(self):
        self.work.mkdir(parents=True, exist_ok=True)
        (self.work / "decisions.json").write_text("not json", encoding="utf-8")
        client = self.client()
        rc, _, err = call(runner.plan, self.spec_path, client=client)
        self.assertEqual(rc, 2)
        self.assertIn("not valid JSON", err)
        self.assertEqual(self.jev.calls, [])
        self.assertEqual((self.work / "decisions.json").read_text(encoding="utf-8"), "not json")

    def test_plan_jev_down_every_row_goes_to_agent_and_says_so(self):
        rc, out, _ = call(runner.plan, self.spec_path, client=self.client(jev_healthy=False))
        self.assertEqual(rc, 0)
        rows = self.decisions()["decisions"]
        self.assertTrue(rows)
        self.assertEqual({r["lane"] for r in rows}, {"agent"})
        self.assertTrue(all("jev_unavailable" in r["route_reasons"] for r in rows))
        self.assertIn("no model answer (jev_unavailable)", out)

    def test_plan_makes_its_client_from_the_spec_and_flags(self):
        with mock.patch.object(runner, "make_client", return_value=self.client()) as made:
            self.assertEqual(call(runner.plan, self.spec_path, dry_run=True, no_jev=True, no_images=True)[0], 0)
        args, kwargs = made.call_args
        self.assertEqual(args[0]["policy"], "default@1")
        self.assertEqual((kwargs["no_jev"], kwargs["no_images"]), (True, True))


class CropTests(RunnerCase):
    def make_video(self):
        (self.root / "raw").mkdir(exist_ok=True)
        run_ffmpeg("-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "1", "-pix_fmt", "yuv420p",
                   str(self.root / "raw" / "screen.mov"))

    def test_crop_writes_row_and_overlay(self):
        rc, out, _ = call(runner.crop, self.spec_path, beat="b1", asset="a2", want="the answer card", client=self.client())
        self.assertEqual(rc, 0)
        row = next(r for r in self.decisions()["decisions"] if r["id"] == "AST-01:b1")
        self.assertEqual(row["lane"], "agent")
        self.assertEqual(row["target"], {"kind": "beat", "id": "b1"})
        self.assertEqual((row["value"]["found"], row["value"]["asset"], row["value"]["t"]), (True, "a2", None))
        self.assertEqual(row["value"]["body"], json.loads((FIXTURES / "manifest_min.json").read_text())
                         ["canvases"]["short"]["window_body"])
        overlay = self.work / "crops" / "b1.jpg"
        self.assertTrue(overlay.is_file())
        self.assertEqual(image_size(overlay)[:2], (800, 450))
        self.assertTrue((self.work / "crops" / "src" / "b1.jpg").is_file())
        self.assertIn("AST-01:b1", out)
        self.assertIn(str(overlay), out)
        self.assertIn("crops/b1.jpg", (self.work / "decisions.md").read_text(encoding="utf-8"))
        self.assertEqual(self.decisions()["backends_used"]["jev"]["requests"], len(self.jev.calls))

    def test_crop_rerun_replaces_the_row_and_keeps_others(self):
        call(runner.plan, self.spec_path, stage="claims", client=self.client())
        before = len(self.decisions()["decisions"])
        for _ in range(2):
            self.assertEqual(call(runner.crop, self.spec_path, beat="b1", asset="a2", want="x", client=self.client())[0], 0)
        ids = [r["id"] for r in self.decisions()["decisions"]]
        self.assertEqual(ids.count("AST-01:b1"), 1)
        self.assertEqual(len(ids), before + 1)

    def test_crop_video_without_t_exit_2(self):
        client = self.client()
        rc, _, err = call(runner.crop, self.spec_path, beat="b1", asset="a1", want="x", client=client)
        self.assertEqual(rc, 2)
        self.assertIn("--t", err)
        self.assertEqual(self.jev.calls, [])
        self.assertFalse((self.work / "decisions.json").exists())

    def test_crop_video_with_t_cuts_the_frame_there(self):
        self.make_video()
        rc, _, _ = call(runner.crop, self.spec_path, beat="v1", asset="a1", t=0.5, want="the panel", client=self.client())
        self.assertEqual(rc, 0)
        row = next(r for r in self.decisions()["decisions"] if r["id"] == "AST-01:v1")
        self.assertEqual((row["value"]["asset"], row["value"]["t"]), ("a1", 0.5))
        self.assertEqual(image_size(self.work / "crops" / "src" / "v1.jpg")[:2], (1568, 882))

    def test_crop_video_past_the_end_exit_2(self):
        self.make_video()
        rc, _, err = call(runner.crop, self.spec_path, beat="v1", asset="a1", t=60.0, want="x", client=self.client())
        self.assertEqual(rc, 2)
        self.assertIn("cannot prepare a frame", err)

    def test_crop_long_form_without_body_exit_2(self):
        path = self.write_spec(lambda s: s["kit"].update(canvas="long"))
        client = self.client()
        rc, _, err = call(runner.crop, path, beat="b1", asset="a2", want="x", client=client)
        self.assertEqual(rc, 2)
        self.assertIn("--body WxH", err)
        self.assertEqual(self.jev.calls, [])
        rc, _, _ = call(runner.crop, path, beat="b1", asset="a2", want="x", body="640x400", client=client)
        self.assertEqual(rc, 0)
        self.assertEqual(self.decisions()["decisions"][0]["value"]["body"], [640, 400])

    def test_crop_bad_input_exit_2(self):
        cases = [({"asset": "zz"}, "unknown asset"), ({"asset": "a3"}, "source_text"), ({"beat": "../x"}, "--beat"),
                 ({"beat": ""}, "--beat"), ({"want": "  "}, "--want"), ({"body": "wide"}, "--body")]
        for over, word in cases:
            kwargs = {"beat": "b1", "asset": "a2", "want": "x", **over}
            client = self.client()
            rc, _, err = call(runner.crop, self.spec_path, client=client, **kwargs)
            self.assertEqual(rc, 2, over)
            self.assertIn(word, err)
            self.assertEqual(self.jev.calls, [])
        (self.root / "assets" / "shot.png").unlink()
        rc, _, err = call(runner.crop, self.spec_path, beat="b1", asset="a2", want="x", client=self.client())
        self.assertEqual(rc, 2)
        self.assertIn("no such file", err)

    def test_crop_dry_run_and_budget_send_nothing(self):
        for kwargs, expected in (({"dry_run": True}, 0), ({"budget": 0.0}, 3)):
            client = self.client()
            rc, _, _ = call(runner.crop, self.spec_path, beat="b1", asset="a2", want="x", client=client, **kwargs)
            self.assertEqual(rc, expected)
            self.assertEqual(self.jev.calls + self.local.calls, [])
            self.assertFalse((self.work / "decisions.json").exists())

    def test_crop_no_images_gives_a_agent_row_without_a_box(self):
        client = self.client()
        client.no_images = True
        rc, _, _ = call(runner.crop, self.spec_path, beat="b1", asset="a2", want="x", client=client)
        self.assertEqual(rc, 0)
        self.assertEqual(self.jev.calls, [])
        row = self.decisions()["decisions"][0]
        self.assertEqual((row["lane"], row["value"]["found"]), ("agent", None))


CROP_LINE = re.compile(r"^AST-01:(\S+): .*box_px \[[^\]]+\], focus \{scale [\d.]+, x [\d.]+, y [\d.]+\}, clamped (\S+), "
                       r"lane agent.*; overlay (.+)$")


class CropBoxTests(RunnerCase):
    """`crop --box`: a pixel box on the source image (800x450 here), no model request."""
    BOX = "240,135,560,315"  # what `respond` answers for the model path: (0.3, 0.3, 0.7, 0.7) of 800x450

    def box_crop(self, beat="b1", box=BOX, **kwargs):
        with mock.patch.object(runner, "make_client", side_effect=AssertionError("--box must not build a client")):
            return call(runner.crop, self.spec_path, beat=beat, asset="a2", box=box, **kwargs)

    def row(self, beat="b1"):
        return next(r for r in self.decisions()["decisions"] if r["id"] == f"AST-01:{beat}")

    def overlay_bytes(self, beat="b1"):
        return (self.work / "crops" / f"{beat}.jpg").read_bytes()

    def test_box_gives_the_same_focus_as_the_model_path_for_the_same_box(self):
        self.assertEqual(call(runner.crop, self.spec_path, beat="m", asset="a2", want="x", client=self.client())[0], 0)
        self.assertEqual(self.box_crop("b")[0], 0)
        model, boxed = self.row("m")["value"], self.row("b")["value"]
        for key in ("found", "box_px", "box", "focus", "body", "clamped"):
            self.assertEqual(boxed[key], model[key], key)

    def test_box_sends_no_request_and_writes_an_overridden_agent_row(self):
        rc, out, _ = self.box_crop(note="tight on the card")
        self.assertEqual(rc, 0)
        row = self.row()
        self.assertEqual((row["lane"], row["status"], row["p"], row["route_reasons"]), ("agent", "overridden", None, []))
        self.assertEqual(row["evidence"]["backend"], "none")
        self.assertEqual((row["value"]["found"], row["value"]["asset"], row["value"]["t"]), (True, "a2", None))
        resolved = row["resolved"]
        self.assertEqual((resolved["value"], resolved["by"], resolved["note"]), (row["value"], "agent", "tight on the card"))
        self.assertRegex(resolved["at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(self.decisions()["backends_used"]["jev"]["requests"], 0)
        self.assertIn("no model request sent", out)

    def test_box_by_human_and_a_rerun_replaces_the_row_and_its_note(self):
        self.assertEqual(self.box_crop(note="first")[0], 0)
        self.assertEqual(self.box_crop(note="second", by="human")[0], 0)
        ids = [r["id"] for r in self.decisions()["decisions"]]
        self.assertEqual(ids.count("AST-01:b1"), 1)
        self.assertEqual((self.row()["resolved"]["by"], self.row()["resolved"]["note"]), ("human", "second"))

    def test_box_redraws_the_overlay_after_a_model_crop(self):
        call(runner.crop, self.spec_path, beat="b1", asset="a2", want="x", client=self.client())
        before = self.overlay_bytes()
        self.assertEqual(self.box_crop(box="100,50,200,100")[0], 0)
        self.assertNotEqual(self.overlay_bytes(), before)
        self.assertEqual(image_size(self.work / "crops" / "b1.jpg")[:2], (800, 450))

    def test_box_prints_one_crop_line(self):
        rc, out, _ = self.box_crop()
        lines = [line for line in out.splitlines() if line.startswith("AST-01")]
        self.assertEqual(len(lines), 1)
        m = CROP_LINE.match(lines[0])
        self.assertIsNotNone(m, lines[0])
        self.assertEqual(m[3], str(self.work / "crops" / "b1.jpg"))
        self.assertNotIn(" p ", lines[0])
        focus = self.row()["value"]["focus"]
        self.assertIn(f"focus {{scale {focus['scale']:.2f}, x {focus['x']}, y {focus['y']}}}", lines[0])

    def test_model_crop_prints_p_box_focus_and_clamped(self):
        rc, out, _ = call(runner.crop, self.spec_path, beat="b1", asset="a2", want="x", client=self.client())
        [line] = [line for line in out.splitlines() if line.startswith("AST-01")]
        self.assertRegex(line, r"^AST-01:b1: p 0\.\d\d, box_px \[")
        self.assertIsNotNone(CROP_LINE.match(line), line)
        self.assertNotIn("overridden", line)

    def test_box_clamps_like_the_model_path(self):
        self.assertEqual(self.box_crop("small", "395,220,405,230")[0], 0)   # far smaller than the scale cap allows
        self.assertIn("scale", self.row("small")["value"]["clamped"])
        self.assertEqual(self.row("small")["value"]["focus"]["scale"], 2.0)
        self.assertEqual(self.box_crop("edge", "0,0,120,70")[0], 0)        # the window cannot centre on the corner
        value = self.row("edge")["value"]
        self.assertTrue({"x", "y"} & set(value["clamped"]), value["clamped"])
        rc, out, _ = self.box_crop("edge", "0,0,120,70")
        self.assertRegex(out, r"clamped [xy,scale]*[xy]")

    def test_box_partly_outside_is_clipped_wholly_outside_is_rejected(self):
        self.assertEqual(self.box_crop("clip", "600,300,2000,2000")[0], 0)
        box = self.row("clip")["value"]["box"]
        self.assertEqual((box["right"], box["bottom"]), (1.0, 1.0))
        rc, _, err = self.box_crop("out", "900,10,1000,50")
        self.assertEqual(rc, 2)
        self.assertIn("outside the 800x450 source image", err)
        self.assertNotIn("AST-01:out", [r["id"] for r in self.decisions()["decisions"]])

    def test_bad_box_exits_2_with_one_line(self):
        for bad in ("1,2,3", "a,b,c,d", "10,10,5,20", "5,5,5,9", "nan,0,10,10", "0,0,inf,10", "", "1,2,3,4,5"):
            rc, _, err = self.box_crop(box=bad)
            self.assertEqual(rc, 2, bad)
            self.assertTrue(err.startswith("crop: --box"), (bad, err))
            self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertFalse((self.work / "decisions.json").exists())

    def test_want_is_optional_with_box_and_required_without(self):
        self.assertEqual(self.box_crop()[0], 0)
        self.assertEqual(self.row()["value"]["want"], "")
        self.assertEqual(self.box_crop(want="the card")[0], 0)
        self.assertEqual(self.row()["value"]["want"], "the card")
        rc, _, err = call(runner.crop, self.spec_path, beat="b2", asset="a2", client=self.client())
        self.assertEqual(rc, 2)
        self.assertIn("--want", err)

    def test_box_dry_run_has_nothing_to_estimate(self):
        rc, out, _ = self.box_crop(dry_run=True)
        self.assertEqual(rc, 0)
        self.assertIn("nothing to estimate", out)
        self.assertFalse((self.work / "decisions.json").exists())

    def test_box_on_a_video_uses_the_native_frame(self):
        (self.root / "raw").mkdir(exist_ok=True)
        run_ffmpeg("-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "1", "-pix_fmt", "yuv420p",
                   str(self.root / "raw" / "screen.mov"))
        rc, _, err = call(runner.crop, self.spec_path, beat="v1", asset="a1", t=0.5, box="100,50,300,150")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.row("v1")["value"]["t"], 0.5)
        rc, _, err = call(runner.crop, self.spec_path, beat="v2", asset="a1", t=0.5, box="700,10,800,50")
        self.assertEqual((rc, "640x360" in err), (2, True))


class ResolveRedrawTests(RunnerCase):
    FOCUS = '{"found":true,"focus":{"scale":2.0,"x":300,"y":100}}'

    def setUp(self):
        super().setUp()
        self.assertEqual(call(runner.crop, self.spec_path, beat="b1", asset="a2", want="x", client=self.client())[0], 0)
        self.decisions_path = str(self.work / "decisions.json")
        self.overlay = self.work / "crops" / "b1.jpg"

    def resolve(self, *extra, ids="AST-01:b1"):
        return call(jevm.main, ["resolve", self.decisions_path, ids, *extra, "--by", "agent"])

    def test_resolve_value_with_a_focus_redraws_the_overlay(self):
        before = self.overlay.read_bytes()
        rc, out, _ = self.resolve("--value", self.FOCUS)
        self.assertEqual(rc, 0)
        self.assertNotEqual(self.overlay.read_bytes(), before)
        self.assertIn("redrew", out)
        self.assertEqual(image_size(self.overlay)[:2], (800, 450))

    def test_resolve_value_with_a_box_px_redraws_the_overlay(self):
        before = self.overlay.read_bytes()
        rc, out, _ = self.resolve("--value", '{"found":true,"box_px":[10,10,200,150]}')
        self.assertEqual(rc, 0)
        self.assertNotEqual(self.overlay.read_bytes(), before)

    def test_resolve_accept_other_rows_and_text_values_leave_the_overlay_alone(self):
        before = self.overlay.read_bytes()
        self.assertEqual(self.resolve("--accept")[0], 0)
        rc, out, _ = self.resolve("--value", "looks fine")
        self.assertEqual(rc, 0)
        self.assertNotIn("redrew", out)
        self.assertEqual(self.overlay.read_bytes(), before)

    def test_resolve_without_the_source_frame_says_so_and_still_succeeds(self):
        (self.work / "crops" / "src" / "b1.fit.json").unlink()
        before = self.overlay.read_bytes()
        rc, out, _ = self.resolve("--value", self.FOCUS)
        self.assertEqual(rc, 0)
        self.assertIn("no source frame", out)
        self.assertEqual(len(out.strip().splitlines()), 2)  # the note and "resolved 1 row"
        self.assertEqual(self.overlay.read_bytes(), before)
        self.assertEqual(next(r for r in self.decisions()["decisions"] if r["id"] == "AST-01:b1")["status"], "overridden")

    def test_resolve_with_a_malformed_focus_never_fails(self):
        rc, out, _ = self.resolve("--value", '{"focus":{"scale":"big"}}')
        self.assertEqual(rc, 0)
        self.assertIn("not redrawn", out)


class CliTests(unittest.TestCase):
    def run_main(self, argv):
        return call(jevm.main, argv)

    def test_cli_help_lists_the_commands(self):
        rc, out, _ = self.run_main(["--help"])
        self.assertEqual(rc, 0)
        for name in ("kit", "spec", "plan", "crop", "qa", "resolve", "eval"):
            self.assertRegex(out, rf"(?m)^\s+{name}\s")

    def test_cli_no_command_or_a_bad_command_line_exits_2(self):
        self.assertEqual(self.run_main([])[0], 2)
        self.assertEqual(self.run_main(["nope"])[0], 2)
        self.assertEqual(self.run_main(["plan"])[0], 2)
        self.assertEqual(self.run_main(["resolve", "d.json", "a", "--accept", "--by", "auto"])[0], 2)
        self.assertEqual(self.run_main(["plan", "s.json", "--budget", "-1"])[0], 2)

    def test_cli_unknown_commands_and_flags_exit_2(self):
        for argv in (["nope", "x"], ["kit", "nope"], ["eval", "--nope", "m"], ["spec", "build", "--project", "p",
                     "--words", "w.json", "--format", "short", "--nope"]):
            self.assertEqual(self.run_main(argv)[0], 2, argv)

    def test_cli_python_floor(self):
        self.assertIn("3.15", jevm.check_python((3, 14, 9)))
        self.assertIn("3.14.9", jevm.check_python((3, 14, 9)))
        self.assertIsNone(jevm.check_python((3, 15, 0)))
        self.assertIsNone(jevm.check_python())
        with mock.patch.object(jevm, "check_python", return_value="jevm needs Python 3.15 or newer; this is Python 3.9.0"):
            rc, _, err = self.run_main(["plan", "s.json"])
        self.assertEqual(rc, 2)
        self.assertIn("needs Python 3.15", err)

    def test_cli_plan_passes_its_flags(self):
        with mock.patch.object(runner, "plan", return_value=3) as plan:
            rc, _, _ = self.run_main(["plan", "s.json", "--stage", "captions", "--only", "CAP-02, STY-06", "--dry-run",
                                      "--no-jev", "--no-images", "--budget", "0.5"])
        self.assertEqual(rc, 3)
        plan.assert_called_once_with("s.json", stage="captions", only=["CAP-02", "STY-06"], dry_run=True, no_jev=True,
                                     no_images=True, budget=0.5, kit_dir=None)
        with mock.patch.object(runner, "plan", return_value=0) as plan:
            self.run_main(["plan", "s.json", "--kit-dir", "my kit"])
        plan.assert_called_once_with("s.json", stage=None, only=None, dry_run=False, no_jev=False, no_images=False,
                                     budget=None, kit_dir="my kit")

    def test_cli_crop_passes_its_flags(self):
        with mock.patch.object(runner, "crop", return_value=0) as crop:
            self.run_main(["crop", "s.json", "--beat", "b1", "--asset", "a2", "--t", "12.5", "--want", "the card",
                           "--body", "400x300", "--budget", "0.2", "--dry-run"])
        crop.assert_called_once_with("s.json", beat="b1", asset="a2", t=12.5, want="the card", body="400x300",
                                     dry_run=True, no_jev=False, no_images=False, budget=0.2, kit_dir=None,
                                     box=None, by="agent", note="")
        self.assertEqual(self.run_main(["crop", "s.json", "--beat", "b1"])[0], 2)  # --asset is required

    def test_cli_crop_passes_the_box_flags(self):
        with mock.patch.object(runner, "crop", return_value=0) as crop:
            self.run_main(["crop", "s.json", "--beat", "b1", "--asset", "a2", "--box", "1,2,30,40", "--by", "human",
                           "--note", "tight"])
        crop.assert_called_once_with("s.json", beat="b1", asset="a2", t=None, want=None, body=None, dry_run=False,
                                     no_jev=False, no_images=False, budget=None, kit_dir=None, box="1,2,30,40",
                                     by="human", note="tight")

    def test_cli_qa_and_resolve_pass_their_flags(self):
        with mock.patch.object(jevm.qa, "cli_qa", return_value=0) as cli:
            self.run_main(["qa", "r.mp4", "--spec", "s.json", "--every", "2", "--no-images", "--budget", "0.3"])
        cli.assert_called_once_with("r.mp4", "s.json", every=2.0, dry_run=False, no_jev=False, no_images=True, budget=0.3,
                                    kit_dir=None)
        with mock.patch.object(decisions_io, "cli_resolve", return_value=0) as cli:
            self.run_main(["resolve", "d.json", "a,b", "--value", "[1]", "--by", "human", "--note", "why"])
        cli.assert_called_once_with("d.json", "a,b", False, "[1]", "human", "why")
        self.assertEqual(self.run_main(["resolve", "d.json", "a", "--accept", "--by", "someone"])[0], 2)
        with mock.patch.object(decisions_io, "cli_resolve", return_value=0) as cli:
            self.run_main(["resolve", "d.json", "a", "--accept", "--by", "agent"])
        cli.assert_called_once_with("d.json", "a", True, None, "agent", "")
        self.assertEqual(self.run_main(["resolve", "d.json", "a", "--by", "agent"])[0], 2)  # --accept or --value

    def test_cli_kit_and_spec_pass_their_flags(self):
        with mock.patch.object(jevm.kit, "cli_build", return_value=0) as build:
            self.run_main(["kit", "build"])
        build.assert_called_once_with(None, None)
        with mock.patch.object(jevm.kit, "cli_build", return_value=0) as build:
            self.run_main(["kit", "build", "--kit-dir", "k", "--adapter", "k/adapter.py"])
        build.assert_called_once_with("k", "k/adapter.py")
        with mock.patch.object(jevm.kit, "cli_check", return_value=1) as check:
            rc, _, _ = self.run_main(["kit", "check", "--kit-dir", "other"])
        self.assertEqual(rc, 1)
        check.assert_called_once_with("other")
        self.assertEqual(self.run_main(["kit", "check", "--adapter", "a.py"])[0], 2)    # --adapter is a build flag
        with mock.patch.object(jevm.spec_validate, "cli_validate", return_value=1) as validate:
            rc, _, _ = self.run_main(["spec", "validate", "spec.json"])
        self.assertEqual(rc, 1)
        validate.assert_called_once_with("spec.json", None)
        with mock.patch.object(jevm.spec_build, "cli_build", return_value=0) as build:
            self.run_main(["spec", "build", "--project", "p", "--format", "short", "--words", "w.json", "--assets", "a.json",
                           "--kit-dir", "k", "-o", "out.json"])
        build.assert_called_once_with("p", "short", "w.json", "a.json", "k", "out.json")
        with mock.patch.object(jevm.spec_build, "cli_build", return_value=0) as build:
            rc, _, err = self.run_main(["spec", "build", "--project", "p", "--words", "w.json"])
        self.assertEqual(rc, 2)
        self.assertIn("--format", err)
        build.assert_not_called()

    def test_cli_spec_validate_runs_for_real(self):
        rc, out, _ = self.run_main(["spec", "validate", str(FIXTURES / "spec_min.json"), "--kit-dir", str(TOOL / "examples" / "kit")])
        self.assertEqual(rc, 1)                                     # the example kit is not the fixture manifest
        rc, out, _ = self.run_main(["spec", "validate", str(FIXTURES / "spec_min.json")])
        self.assertEqual(rc, 0)
        self.assertIn("valid:", out)

    # The eval tests run the real harness (evals/harness.py) on its own frozen answers: `--offline` takes about a tenth
    # of a second, so no stand-in module is needed. A test that has to see what jevm passes patches one harness function
    # (a wrapper or a mock), never the module.
    def harness(self):
        import evals.harness
        return evals.harness

    def temp_dir(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return Path(td.name)

    def test_eval_offline_dry_run_estimates_and_writes_nothing(self):
        out_dir = self.temp_dir() / "out dir"
        rc, out, _ = self.run_main(["eval", "--offline", "--dry-run", "--out", str(out_dir)])
        self.assertEqual(rc, 0)
        self.assertIn("jev: 58 requests", out)
        self.assertIn("dry run: nothing sent", out)
        self.assertFalse(out_dir.exists())

    def test_eval_offline_and_out_passthrough(self):
        h = self.harness()
        out_dir = self.temp_dir() / "out dir"
        with mock.patch.object(h, "offline_client", wraps=h.offline_client) as offline:
            rc, out, _ = self.run_main(["eval", "--offline", "--out", str(out_dir)])
        self.assertEqual(rc, 0)
        self.assertIn("eval jev (offline replay)", out)
        self.assertIn("Result: all gates pass.", out)
        offline.assert_called_once()
        self.assertEqual(offline.call_args.kwargs["force_backend"], "jev")
        self.assertTrue(Path(offline.call_args.kwargs["cache_dir"]).name.startswith("jevm-eval-"))
        written = sorted(p.name for p in out_dir.iterdir())
        self.assertEqual([n.rsplit(".", 1)[1] for n in written], ["json", "md"])
        self.assertTrue(all(re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{6}-jev-all-offline\.(json|md)", n) for n in written), written)
        self.assertTrue(json.loads(next(out_dir.glob("*.json")).read_text())["meta"]["offline"])

    def test_eval_offline_passes_the_same_flags_as_the_harness_cli(self):
        h = self.harness()
        folder = self.temp_dir()
        # report names carry the start time to the second: pin the clock so the two runs name their files alike
        with mock.patch.object(h, "_now", return_value=datetime.datetime(2026, 10, 24, 13, 5, 9)):
            mine = self.run_main(["eval", "--offline", "--qset", "stat_check", "--model", "jev-1.13.0", "--budget", "1",
                                  "--out", str(folder / "a")])
            theirs = call(h.main, ["--offline", "--qset", "stat_check", "--model", "jev-1.13.0", "--budget", "1",
                                   "--out", str(folder / "b")])
        self.assertEqual((mine[0], theirs[0]), (0, 0))
        self.assertEqual(sorted(p.name for p in (folder / "a").iterdir()), sorted(p.name for p in (folder / "b").iterdir()))
        self.assertEqual(mine[1].replace(str(folder / "a"), "X"), theirs[1].replace(str(folder / "b"), "X"))

    def test_eval_offline_rejects_the_flags_it_cannot_honour(self):
        for flag in ("--no-jev", "--no-images"):
            rc, _, err = self.run_main(["eval", "--offline", flag, "--out", str(self.temp_dir())])
            self.assertEqual(rc, 2)
            self.assertIn("do not apply", err)

    def test_eval_defaults_to_jev_backend(self):
        h = self.harness()
        folder = self.temp_dir()
        # No flag: the harness builds the client, and jevm hands it no backend, so the harness forces jev.
        replay = h.offline_client(cache_dir=folder / "c1")
        with mock.patch.object(h, "make_client", return_value=replay) as made:
            rc, _, _ = self.run_main(["eval", "--dry-run"])
        self.assertEqual(rc, 0)
        made.assert_called_once_with(force_backend="jev")
        # --no-jev and --no-images: jevm builds the client itself, and it is forced to jev too.
        for flags in (["--no-jev"], ["--no-images"], ["--no-jev", "--no-images"]):
            replay = h.offline_client(cache_dir=folder / "c2")
            with mock.patch.object(jevm, "make_client", return_value=replay) as made:
                rc, _, _ = self.run_main(["eval", "--dry-run", *flags])
            self.assertEqual(rc, 0, flags)
            made.assert_called_once_with(None, no_jev="--no-jev" in flags, no_images="--no-images" in flags,
                                         force_backend="jev")
        # An explicit --backend wins, on every path.
        with mock.patch.object(jevm, "make_client", return_value=h.offline_client(cache_dir=folder / "c3")) as made:
            self.run_main(["eval", "--dry-run", "--no-images", "--backend", "local"])
        self.assertEqual(made.call_args.kwargs["force_backend"], "local")
        with mock.patch.object(h, "offline_client", wraps=h.offline_client) as offline:
            self.run_main(["eval", "--offline", "--dry-run", "--backend", "local"])
        self.assertEqual(offline.call_args.kwargs["force_backend"], "local")
        # The jev default reaches the report: the offline replay was forced to jev, so its files say jev.
        out_dir = folder / "out"
        self.run_main(["eval", "--offline", "--out", str(out_dir)])
        self.assertEqual(json.loads(next(out_dir.glob("*.json")).read_text())["meta"]["backend"], "jev")

    def test_eval_returns_harness_exit_code(self):
        h = self.harness()
        for code in (0, 1, 2, 3):
            with mock.patch.object(h, "run_eval", return_value=code) as run:
                rc, _, _ = self.run_main(["eval", "--qset", "highlight", "--module", "captions", "--model", "jev-9",
                                          "--dry-run", "--budget", "0.2"])
            self.assertEqual(rc, code)
        run.assert_called_once_with(qset="highlight", module="captions", backend=None, model="jev-9", dry_run=True,
                                    budget=0.2)
        folder = self.temp_dir()
        kept = self.run_main(["eval", "--offline", "--out", str(folder)])[0]
        self.assertEqual(kept, 0)                                   # all gates pass
        for argv in (["eval", "--qset", "nope"], ["eval", "--module", "nope"]):
            self.assertEqual(self.run_main(argv)[0], 2, argv)       # the harness's usage errors
        self.assertEqual(self.run_main(["eval", "--offline", "--budget", "0", "--out", str(folder / "x")])[0], 3)
        self.assertFalse((folder / "x").exists())

    def test_eval_a_missed_number_exits_1(self):
        # The offline replay on `local` serves the scripted Jev answers, so the QA parity number (6/8 in the set's
        # local_reference) is missed by the harness's own arithmetic: a real, deterministic exit 1 through the real harness.
        rc, out, _ = self.run_main(["eval", "--offline", "--backend", "local", "--out", str(self.temp_dir())])
        self.assertEqual(rc, 1)
        self.assertIn("eval local (offline replay)", out)
        self.assertIn("FAIL", out)

    def test_cli_eval_without_the_harness_exits_2(self):
        gone = ModuleNotFoundError("No module named 'evals'", name="evals")
        with mock.patch.object(jevm.importlib, "import_module", side_effect=gone):
            rc, _, err = self.run_main(["eval"])
        self.assertEqual(rc, 2)
        self.assertIn("not installed", err)
        broken = ModuleNotFoundError("No module named 'other'", name="other")
        with mock.patch.object(jevm.importlib, "import_module", side_effect=broken):
            with self.assertRaises(ModuleNotFoundError):
                self.run_main(["eval"])

    def test_cli_runs_as_a_script_from_any_folder(self):
        with tempfile.TemporaryDirectory(prefix="jevm elsewhere ") as td:
            done = subprocess.run([sys.executable, str(TOOL / "jevm.py"), "nope"], cwd=td, capture_output=True, text=True)
            self.assertEqual(done.returncode, 2)
            self.assertIn("invalid choice", done.stderr)
            done = subprocess.run([sys.executable, str(TOOL / "jevm.py"), "--help"], cwd=td, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0)
            self.assertIn("resolve", done.stdout)
            self.assertEqual(os.listdir(td), [])


class ReadmeTests(unittest.TestCase):
    def test_readme_gives_relative_commands_and_states_the_scope(self):
        text = (TOOL / "README.md").read_text(encoding="utf-8")
        self.assertNotRegex(text, r"/(Users|home)/")                 # relative paths only
        self.assertIn("Remotion", text)
        self.assertRegex(text, r"never edits a composition")
        self.assertIn("not affiliated with TypeSafe", text)
        for command in ("kit build", "kit check", "spec build", "plan", "crop", "qa", "resolve", "eval"):
            self.assertIn(f"jevm.py {command}", text)
        for word in ("JEVM_OFFLINE", "JEVM_DECIDER_URL", "JEVM_KIT_DIR", "JEVM_CACHE_DIR", "JEV_API", "decisions.json",
                     "agent", "human", "use_when.json", "formats.json", "--kit-dir"):
            self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
