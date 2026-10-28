"""The whole tool through its command line, and the routing rules on the shipped question sets (stdlib unittest). From
the tool folder:

    python3 -m unittest tests.test_integration -q

The command-line tests run `python3 jevm.py ...` as a subprocess with JEVM_OFFLINE=1 and a Decider URL nothing listens
on, in a project folder whose name has a space ("jevm project") and a separate, empty working directory, so a path split
at the space shows up as a stray file. The kit is examples/kit, reached through JEVM_KIT_DIR. The routing tests run a real
`Client` over `FakeBackend`s with the real qsets/*.json, and `runner.plan` over them; nothing here reaches the network.

`Project` is not a test case: tests/test_secrets.py reuses it to look for the key in what a run writes.
"""
import contextlib
import dataclasses
import io
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

from core import runner
from jevlib.backends.jev import JevBackend
from jevlib.backends.local import LocalBackend
from jevlib.cache import Cache
from jevlib.client import LOCAL_ROUTES, Client
from jevlib.fake import FakeBackend, box, choice, noul
from jevlib.images import FFMPEG
from jevlib.request import Request, image_ref, load_qset

TESTS = Path(__file__).resolve().parent
TOOL = TESTS.parent
JEVM = TOOL / "jevm.py"
FIXTURES = TESTS / "fixtures"
EXAMPLE_KIT = TOOL / "examples" / "kit"
DECIDER_NOWHERE = "http://127.0.0.1:9"          # the discard port: nothing listens, so even a leak would be refused
DECIDER_VERSION = "0.1.0"                       # the Decider app version the shipped local sets are calibrated for
SHIPPED_QSETS = ("crop_box", "highlight", "presence", "qa_frame.short", "stat_check")
IMAGE_QSETS = ("crop_box", "presence", "qa_frame.short")
KIT_NUMBERS = re.compile(r"\b(920|760|1560)\b")       # the example kit's window body and word-band bottom


def child_env():
    """This process's environment, offline, Decider unreachable, without `JEV_API`, with the example kit in
    `JEVM_KIT_DIR` and no `JEVM_CACHE_DIR`. The children run in an empty folder, so no `.env` is there to read."""
    env = dict(os.environ)
    env.pop("JEV_API", None)
    env.pop("JEVM_CACHE_DIR", None)
    env.update(JEVM_OFFLINE="1", JEVM_DECIDER_URL=DECIDER_NOWHERE, JEVM_KIT_DIR=str(EXAMPLE_KIT))
    return env


def run_ffmpeg(*args):
    subprocess.run([FFMPEG, "-v", "error", "-y", *args], check=True)


def cache_listing():
    """`.jevm-cache/` in the tool folder as a sorted list of paths, or None when it does not exist."""
    cache = TOOL / ".jevm-cache"
    if not cache.is_dir():
        return None
    return sorted(str(p.relative_to(cache)) for p in cache.rglob("*"))


class Project:
    """A throwaway project folder named "jevm project" and an empty `cwd` beside it, driven through `jevm.py`.

    `output` collects every stdout and stderr the commands printed, for the tests that look for a leak.
    """

    def __init__(self, tmp):
        self.tmp = Path(tmp).resolve()
        self.cwd = self.tmp / "cwd"
        self.root = self.tmp / "jevm project"
        self.cwd.mkdir()
        self.root.mkdir()
        spec = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
        self.words = self.root / "words.json"
        self.words.write_text(json.dumps(spec["transcript"]["words"]), encoding="utf-8")
        self.work = self.root / "work" / "jevm"
        self.spec = self.work / "spec.json"
        self.output = []

    def run(self, *args):
        proc = subprocess.run([sys.executable, str(JEVM), *[str(a) for a in args]], capture_output=True, text=True,
                              env=child_env(), cwd=self.cwd, timeout=120)
        self.output += [proc.stdout, proc.stderr]
        return proc

    def build_spec(self, *extra):
        proc = self.run("spec", "build", "--project", self.root, "--format", "short", "--words", self.words, *extra)
        if proc.returncode != 0:
            raise AssertionError(f"spec build exited {proc.returncode}: {proc.stderr}")
        return self.spec

    def decisions(self):
        return json.loads((self.work / "decisions.json").read_text(encoding="utf-8"))


class CliCase(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.project = Project(td.name)

    def planned(self):
        """The project after `spec build` and `plan`, both exit 0."""
        self.project.build_spec()
        plan = self.project.run("plan", self.project.spec)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        return plan

    def assertNoTraceback(self, proc):
        self.assertNotIn("Traceback", proc.stderr)


class OfflineCliTests(CliCase):
    def test_end_to_end_offline(self):
        p = self.project
        cache_before = cache_listing()
        check = p.run("kit", "check")
        self.assertEqual(check.returncode, 0, check.stderr)
        self.assertEqual(p.build_spec(), p.spec)
        spec = json.loads(p.spec.read_text(encoding="utf-8"))
        self.assertEqual(Path(spec["project"]["root"]).resolve(), p.root)
        valid = p.run("spec", "validate", p.spec)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        plan = p.run("plan", p.spec)
        self.assertEqual(plan.returncode, 0, plan.stderr)
        self.assertNoTraceback(plan)
        self.assertTrue((p.work / "decisions.json").is_file())
        self.assertTrue((p.work / "decisions.md").is_file())
        # A path split at the space would have written into `cwd` or beside the project, not inside it.
        self.assertEqual(os.listdir(p.cwd), [])
        self.assertEqual(sorted(os.listdir(p.tmp)), ["cwd", "jevm project"])
        self.assertEqual(sorted(os.listdir(p.work)), ["decisions.json", "decisions.md", "spec.json"])
        self.assertEqual(cache_listing(), cache_before, "a project run must not touch the tool folder's cache")
        self.assertEqual(spec["kit"]["manifest"], str(EXAMPLE_KIT.resolve() / "manifest.json"))

    def test_plan_offline_all_agent_lane(self):
        plan = self.planned()
        doc = self.project.decisions()
        rows = doc["decisions"]
        self.assertGreater(len(rows), 0)
        for row in rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(row["lane"], "agent")
                self.assertIn("jev_unavailable", row["route_reasons"])
                self.assertEqual(row["evidence"]["backend"], "none")
        self.assertEqual({r["decision"] for r in rows} >= {"CAP-02", "STY-06"}, True)
        self.assertEqual(doc["backends_used"]["jev"]["requests"], 0)
        self.assertEqual(doc["backends_used"]["local"]["requests"], 0)
        self.assertIn("jev_unavailable", plan.stdout)
        self.assertIn("lane agent: ", plan.stdout)
        self.assertIn("lane auto: 0 rows", plan.stdout)
        self.assertIn("lane human: 0 rows", plan.stdout)

    def test_resolve_through_cli(self):
        self.planned()
        decisions = self.project.work / "decisions.json"
        first = self.project.decisions()["decisions"][0]["id"]
        done = self.project.run("resolve", decisions, first, "--accept", "--by", "agent")
        self.assertEqual(done.returncode, 0, done.stderr)
        row = next(r for r in self.project.decisions()["decisions"] if r["id"] == first)
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["resolved"]["by"], "agent")
        self.assertEqual(row["resolved"]["value"], row["value"])
        unknown = self.project.run("resolve", decisions, "NOPE:nothing", "--accept", "--by", "agent")
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("unknown id", unknown.stderr)

    def test_replan_keeps_a_resolution_whose_value_did_not_change(self):
        self.planned()
        decisions = self.project.work / "decisions.json"
        first = self.project.decisions()["decisions"][0]["id"]
        self.assertEqual(self.project.run("resolve", decisions, first, "--accept", "--by", "human").returncode, 0)
        again = self.project.run("plan", self.project.spec)
        self.assertEqual(again.returncode, 0, again.stderr)
        row = next(r for r in self.project.decisions()["decisions"] if r["id"] == first)
        self.assertEqual((row["status"], row["resolved"]["by"]), ("accepted", "human"))

    def test_qa_offline_exit_0(self):
        p = self.project
        p.build_spec()
        render = p.root / "renders" / "my render.mp4"
        render.parent.mkdir()
        run_ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=1080x1920:d=2:r=30", "-pix_fmt", "yuv420p", str(render))
        qa = p.run("qa", render, "--spec", p.spec)
        self.assertEqual(qa.returncode, 0, qa.stderr)
        self.assertNoTraceback(qa)
        out_dir = p.work / "qa" / "my render"
        data = json.loads((out_dir / "qa.json").read_text(encoding="utf-8"))
        self.assertTrue((out_dir / "qa.md").is_file())
        self.assertEqual(len(data["frames"]), 2)
        for frame in data["frames"]:
            self.assertEqual(frame["error"], "jev_unavailable")
            self.assertEqual((frame["backend"], frame["flags"]), ("none", []))
            self.assertTrue(Path(frame["file"]).is_file())
            self.assertEqual(Path(frame["file"]).parent, out_dir / "frames")
        self.assertEqual(data["meta"]["errors"], {"jev_unavailable": 2})
        self.assertEqual(os.listdir(p.cwd), [])

    def test_crop_offline_agent_lane(self):
        p = self.project
        (p.root / "assets").mkdir()
        run_ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=800x450", "-frames:v", "1", str(p.root / "assets" / "shot.png"))
        assets = p.root / "assets.json"
        assets.write_text(json.dumps([{"id": "a2", "kind": "image", "uri": "assets/shot.png", "sensitivity": "private",
                                       "description": "Screenshot of the answer card"}]), encoding="utf-8")
        p.build_spec("--assets", assets)
        crop = p.run("crop", p.spec, "--beat", "b1", "--asset", "a2", "--want", "the answer card")
        self.assertEqual(crop.returncode, 0, crop.stderr)
        self.assertNoTraceback(crop)
        row = next(r for r in p.decisions()["decisions"] if r["id"] == "AST-01:b1")
        self.assertEqual(row["lane"], "agent")
        self.assertIn("jev_unavailable", row["route_reasons"])
        self.assertTrue((p.work / "crops" / "b1.jpg").is_file())
        self.assertEqual(os.listdir(p.cwd), [])

    def test_no_kit_numbers_in_code(self):
        hits, seen = [], 0
        for path in sorted(TOOL.rglob("*.py")):
            rel = path.relative_to(TOOL)
            if rel.parts[0] in ("kits", "tests", "evalsets", "examples") or "__pycache__" in rel.parts:
                continue
            seen += 1
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if KIT_NUMBERS.search(line):
                    hits.append(f"{rel}:{n}")
        self.assertGreater(seen, 10, "the walk found almost no code: the check would pass for nothing")
        self.assertEqual(hits, [], "kit geometry numbers belong in kits/, not in code")


def write_png(path, w=300, h=300, shade=0x80):
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes([shade]) * w for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def respond(req, model):
    """Answers every qid by its type, so one responder serves every shipped question set."""
    answers = {}
    for qid, question in req.questions.items():
        kind = question["type"]
        if kind == "noul":
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


def call(fn, *args, **kwargs):
    """`fn` with stdout and stderr captured: (result, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = fn(*args, **kwargs)
    return result, out.getvalue(), err.getvalue()


class RoutingCase(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.image = write_png(self.tmp / "frame.png")
        self.logs = []
        self.n = 0

    def backends(self, *, jev_up=True, local_up=True):
        self.jev = FakeBackend("jev", respond, healthy=jev_up)
        self.local = FakeBackend("local", respond, version=DECIDER_VERSION, healthy=local_up)

    def client(self, **kw):
        self.n += 1
        return Client({"jev": self.jev, "local": self.local}, Cache(self.tmp / f"cache{self.n}"),
                      log=self.logs.append, **kw)

    def request(self, qs, sensitivity="public"):
        """One request on `qs`: its first question template, with a frame when the set judges images."""
        state, images = {"i": 0, "text": "line 0"}, {}
        if qs.id in IMAGE_QSETS:
            ref = image_ref(self.image)
            state["frame"], images[ref] = ref, self.image
        template = next(iter(qs.questions))
        return Request(qs, state, {"q_0": qs.question(template, i="0")}, images, sensitivity, tag=qs.id)


class PrivateRoutingTests(RoutingCase):
    def test_private_goes_local_only_where_routing_allows(self):
        cases = [
            # (set, overrides to the shipped set, sensitivity, backend that answers)
            ("stat_check", {}, "private", "local"),
            ("stat_check", {"backend": "jev"}, "public", "jev"),
            ("stat_check", {"backend": "jev"}, "private", "local"),          # the route allows it: privacy decides
            ("stat_check", {"route": "jev", "backend": "jev"}, "private", "jev"),  # the route does not
            ("presence", {}, "private", "local"),
            ("presence", {"backend": "jev"}, "own_render", "jev"),
            ("presence", {"route": "jev", "backend": "jev"}, "private", "jev"),
            ("highlight", {}, "private", "jev"),                              # route jev
            ("highlight", {"route": "either"}, "private", "jev"),             # route allows it, no local calibration
            ("qa_frame.short", {}, "private", "jev"),
            ("crop_box", {}, "private", "jev"),
        ]
        for qset_id, overrides, sensitivity, want in cases:
            with self.subTest(qset=qset_id, overrides=overrides, sensitivity=sensitivity):
                self.backends()
                qs = dataclasses.replace(load_qset(qset_id), **overrides)
                ans = self.client().ask(self.request(qs, sensitivity))
                self.assertTrue(ans.ok, ans.detail)
                self.assertEqual(ans.backend, want)
                self.assertEqual((len(self.jev.calls), len(self.local.calls)), (0, 1) if want == "local" else (1, 0))

    def test_the_shipped_sets_split_into_local_capable_and_not(self):
        local_capable = {i for i in SHIPPED_QSETS
                         if load_qset(i).route in LOCAL_ROUTES and load_qset(i).has_local_calibration()}
        self.assertEqual(local_capable, {"presence", "stat_check"})
        for qset_id in set(SHIPPED_QSETS) - local_capable:
            with self.subTest(qset=qset_id):
                self.backends()
                ans = self.client().ask(self.request(dataclasses.replace(load_qset(qset_id), backend="jev"), "private"))
                self.assertEqual((ans.backend, len(self.local.calls)), ("jev", 0))

    def test_box_never_goes_local(self):
        crop = load_qset("crop_box")
        hostile = dataclasses.replace(crop, route="local", backend="local",
                                      thresholds={"jev": {}, "local": {"vis": 0.5}},
                                      calibrated={"local": {"app_version": DECIDER_VERSION, "model": crop.models["local"]}})
        for qs in (crop, hostile):
            for sensitivity in ("public", "private"):
                with self.subTest(route=qs.route, sensitivity=sensitivity):
                    self.backends()
                    req = self.request(qs, sensitivity)
                    self.assertTrue(req.has_box())
                    ans = self.client().ask(req)
                    self.assertEqual(ans.backend, "jev")
                    self.assertEqual(len(self.local.calls), 0)
        self.backends(jev_up=False)                         # Jev down: Decider cannot stand in for a Box either
        ans = self.client().ask(self.request(hostile, "private"))
        self.assertEqual((ans.backend, ans.error, len(self.local.calls)), ("none", "jev_unavailable", 0))
        self.backends()                                     # a parity run forced onto Decider is refused, not sent
        ans = self.client(force_backend="local").ask(self.request(crop, "public"))
        self.assertEqual((ans.error, ans.detail, len(self.local.calls)), ("local_unavailable", "box_on_local", 0))


class FallbackTests(RoutingCase):
    def test_fallback_both_ways_per_request(self):
        stat = load_qset("stat_check")
        self.backends(local_up=False)                       # Decider down: its set runs on Jev
        ans = self.client().ask(self.request(stat))
        self.assertEqual((ans.backend, ans.route_reasons), ("jev", ["backend_fallback"]))
        either = dataclasses.replace(stat, backend="jev")   # a calibrated set that starts on Jev
        self.backends(jev_up=False)                         # Jev down: the calibrated set runs on Decider ...
        ans = self.client().ask(self.request(either))
        self.assertEqual((ans.backend, ans.route_reasons), ("local", ["backend_fallback"]))
        ans = self.client().ask(self.request(load_qset("highlight")))   # ... and an uncalibrated one goes to the agent
        self.assertEqual((ans.backend, ans.error, ans.route_reasons), ("none", "jev_unavailable", []))
        self.assertEqual(len(self.jev.calls), 0)


class PlanFallbackTests(RoutingCase):
    """`runner.plan` on the fixture spec with the shipped qsets, one backend down at a time."""

    def setUp(self):
        super().setUp()
        self.root = self.tmp / "project dir"
        (self.root / "assets").mkdir(parents=True)
        (self.root / "sources").mkdir()
        (self.root / "sources" / "report.md").write_text("In my test of 50 invoices, it got 47 right.\n", encoding="utf-8")
        run_ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=800x450", "-frames:v", "1", str(self.root / "assets" / "shot.png"))
        spec = json.loads((FIXTURES / "spec_min.json").read_text(encoding="utf-8"))
        spec["project"]["root"] = str(self.root)
        spec["kit"]["manifest"] = str(FIXTURES / "manifest_min.json")
        spec["constraints"]["highlight"]["density"] = 0.5
        self.work = self.root / "work" / "jevm"
        self.work.mkdir(parents=True)
        self.spec_path = self.work / "spec.json"
        self.spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    def rows(self, decision):
        doc = json.loads((self.work / "decisions.json").read_text(encoding="utf-8"))
        return [r for r in doc["decisions"] if r["decision"] == decision]

    def test_jev_down_decider_up_stat_checks_still_answered(self):
        self.backends(jev_up=False)
        rc, out, _ = call(runner.plan, self.spec_path, client=self.client())
        self.assertEqual(rc, 0)
        self.assertEqual(self.jev.calls, [])
        captions = self.rows("CAP-02")
        self.assertGreater(len(captions), 0)
        for row in captions:                                # the highlight set has no local calibration
            self.assertEqual((row["lane"], row["evidence"]["backend"]), ("agent", "none"))
            self.assertIn("jev_unavailable", row["route_reasons"])
        stats = self.rows("STY-06")
        self.assertGreater(len(stats), 0)
        for row in stats:                                   # the stat set runs on Decider
            self.assertEqual(row["evidence"]["backend"], "local")
            self.assertNotIn("jev_unavailable", row["route_reasons"])
        self.assertGreater(len(self.local.calls), 0)
        self.assertIn("Jev unavailable", "\n".join(self.logs))

    def plan_decider_down(self):
        self.backends(local_up=False)
        rc, _, _ = call(runner.plan, self.spec_path, client=self.client())
        self.assertEqual(rc, 0)
        self.assertEqual(self.local.calls, [])
        return self.rows("STY-06")

    def test_decider_down_jev_up_stat_checks_run_on_jev(self):
        stats = self.plan_decider_down()
        self.assertGreater(len(stats), 0)
        for row in stats:
            self.assertEqual(row["evidence"]["backend"], "jev")
        for row in self.rows("CAP-02"):
            self.assertEqual(row["evidence"]["backend"], "jev")
            self.assertNotIn("backend_fallback", row["route_reasons"])
        self.assertIn("Decider unavailable", "\n".join(self.logs))

    # Design 7.3 and the README: "its rows carry backend_fallback".
    def test_decider_down_stat_check_rows_carry_backend_fallback(self):
        for row in self.plan_decider_down():
            self.assertIn("backend_fallback", row["route_reasons"])

    def test_no_key_jev_down_decider_down_exit_0_all_agent(self):
        """The real backends, not fakes: no key (so Jev is down for that reason, not for offline mode) and Decider
        refusing connections. The transports are stand-ins, so nothing can reach the network."""
        def forbidden(*args, **kwargs):
            raise AssertionError("a request reached the network")

        def refused(*args, **kwargs):
            raise ConnectionRefusedError("connection refused")

        jev = JevBackend(key_loader=lambda: "", transport=forbidden)
        local = LocalBackend(url=DECIDER_NOWHERE, transport=refused, plist=self.tmp / "no-such.plist",
                             sleep=lambda seconds: None)
        client = Client({"jev": jev, "local": local}, Cache(self.tmp / "cache"), log=self.logs.append)
        with mock.patch.dict(os.environ):
            os.environ.pop("JEVM_OFFLINE", None)
            rc, out, _ = call(runner.plan, self.spec_path, client=client)
        self.assertEqual(rc, 0)
        doc = json.loads((self.work / "decisions.json").read_text(encoding="utf-8"))
        self.assertGreater(len(doc["decisions"]), 0)
        for row in doc["decisions"]:
            with self.subTest(row=row["id"]):
                self.assertEqual(row["lane"], "agent")
                self.assertIn("jev_unavailable", row["route_reasons"])
        text = "\n".join(self.logs)
        self.assertIn("JEV_API is not set", text)
        self.assertNotIn("offline mode", text)
        self.assertIn("Decider is not reachable", text)
        self.assertIn("jev_unavailable", out)


if __name__ == "__main__":
    unittest.main()
