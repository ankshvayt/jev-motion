"""Tests for jevlib.client: routing, fallback, cache use, ledger and preflight (docs/fallback.md).

Everything runs on two FakeBackends and a temporary Cache; no network, no key, no real backend class.
"""
import importlib.util
import json
import os
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
import zlib
from dataclasses import replace
from pathlib import Path
from unittest import mock

from jevlib import tokens
from jevlib.backends.base import BackendError, BackendUnavailable, LimitError
from jevlib.cache import Cache
from jevlib.client import Client, Ledger, make_client
from jevlib.fake import FakeBackend, box, by_qid, choice, noul
from jevlib.request import Request, image_ref, load_qset, noul_p

FIXTURES = Path(__file__).parent / "fixtures" / "qsets"
CALIBRATED = {"local": {"app_version": "0.1.0", "model": "pplx-decider-flash-v1.2"}}
FAKE_KEY = "test-key-not-a-real-key-123"


def _qset(**over):
    return replace(load_qset("sample", 2, root=FIXTURES), **over)


JEV_SET = _qset(id="jevset")
# Routes that let a private request go local; every one has local thresholds and a calibration record.
LOCAL_SET = _qset(id="localset", route="local", backend="local",
                  thresholds={"jev": {"need": 0.5}, "local": {"need": 0.4}}, calibrated=CALIBRATED)
EITHER_SET = _qset(id="eitherset", route="either", backend="jev",
                   thresholds={"jev": {"need": 0.5}, "local": {"need": 0.4}}, calibrated=CALIBRATED)
BOX_SET = replace(EITHER_SET, id="boxset")

NOUL_Q = {"q_0": {"type": "noul", "instructions": "Is the line fine?"}}
BOX_Q = {"b_0": {"type": "box", "instructions": "Where is the subject?"}}


def write_png(path, w=300, h=300, shade=0x80):
    """A valid grey PNG of `w` x `h`; `shade` makes files with different bytes."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + bytes([shade]) * w for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def make_req(qs=JEV_SET, i=0, *, sensitivity="public", image=None, questions=None):
    state = {"i": i, "text": f"line {i}"}
    images = {}
    if image is not None:
        ref = image_ref(image)
        state["frame"] = ref
        images[ref] = image
    return Request(qs, state, questions or NOUL_Q, images, sensitivity, tag=f"t{i}")


def noul_by_index(name, model_override=None, tokens_=100, hook=None):
    """A responder answering p = i / 20 for every qid, where i is `state["i"]`."""
    def responder(req, model):
        if hook:
            hook(req)
        return {"answers": {qid: noul(req.state["i"] / 20) for qid in req.questions},
                "model": model_override or model, "input_tokens": tokens_, "latency_s": 0.5}
    return responder


class ClientCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.logs = []
        self.jev = FakeBackend("jev", noul_by_index("jev"))
        self.local = FakeBackend("local", noul_by_index("local"), version="0.1.0")

    def client(self, *, jev=None, local=None, cache=None, **kw):
        kw.setdefault("log", self.logs.append)
        return Client({"jev": jev or self.jev, "local": local or self.local},
                      cache or Cache(self.tmp / "cache"), **kw)


class RoutingTests(ClientCase):
    def test_private_image_goes_local_when_route_allows_and_calibrated(self):
        img = write_png(self.tmp / "a.png")
        for route in ("local", "either", "jev-for-speed"):
            with self.subTest(route=route):
                jev = FakeBackend("jev", noul_by_index("jev"))
                local = FakeBackend("local", noul_by_index("local"), version="0.1.0")
                qs = replace(EITHER_SET, route=route)
                c = self.client(jev=jev, local=local, cache=Cache(self.tmp / f"cache-{route}"))
                ans = c.ask(make_req(qs, image=img, sensitivity="private"))
                self.assertEqual(ans.backend, "local")
                self.assertEqual(ans.route_reasons, [])
                self.assertEqual((len(local.calls), len(jev.calls)), (1, 0))
                self.assertEqual(c.route(make_req(qs, image=img, sensitivity="private")), ("local", []))

    def test_private_image_stays_jev_when_route_is_jev(self):
        img = write_png(self.tmp / "a.png")
        qs = replace(EITHER_SET, route="jev")
        ans = self.client().ask(make_req(qs, image=img, sensitivity="private"))
        self.assertEqual(ans.backend, "jev")
        self.assertEqual(len(self.local.calls), 0)

    def test_private_image_stays_jev_without_local_calibration(self):
        img = write_png(self.tmp / "a.png")
        qs = replace(EITHER_SET, thresholds={"jev": {"need": 0.5}, "local": None})
        ans = self.client().ask(make_req(qs, image=img, sensitivity="private"))
        self.assertEqual(ans.backend, "jev")
        self.assertEqual(len(self.local.calls), 0)

    def test_public_and_own_render_images_follow_the_set_default(self):
        img = write_png(self.tmp / "a.png")
        for sens in ("public", "own_render"):
            with self.subTest(sensitivity=sens):
                c = self.client(cache=Cache(self.tmp / f"cache-{sens}"))
                self.assertEqual(c.route(make_req(EITHER_SET, image=img, sensitivity=sens)), ("jev", []))

    def test_box_never_goes_local(self):
        box_resp = FakeBackend("jev", by_qid({"b_0": box(0.1, 0.1, 0.5, 0.5)}))
        c = self.client(jev=box_resp)
        ans = c.ask(make_req(BOX_SET, questions=BOX_Q, sensitivity="private"))
        self.assertEqual(ans.backend, "jev")
        self.assertEqual(len(self.local.calls), 0)
        forced = self.client(jev=box_resp, force_backend="local", cache=Cache(self.tmp / "cache-forced"))
        bad = forced.ask(make_req(BOX_SET, questions=BOX_Q, sensitivity="private"))
        self.assertEqual(bad.error, "local_unavailable")
        self.assertEqual(bad.detail, "box_on_local")
        self.assertEqual(bad.backend, "none")
        self.assertEqual(len(self.local.calls), 0)

    def test_local_down_falls_back_to_jev(self):
        local = FakeBackend("local", noul_by_index("local"), version="0.1.0", healthy=False)
        c = self.client(local=local)
        ans = c.ask(make_req(LOCAL_SET))
        self.assertEqual(ans.backend, "jev")
        self.assertIn("backend_fallback", ans.route_reasons)
        self.assertTrue(ans.ok)
        self.assertEqual(local.calls, [])

    def test_jev_down_falls_back_to_local_only_with_calibration(self):
        jev = FakeBackend("jev", noul_by_index("jev"), healthy=False)
        c = self.client(jev=jev)
        calibrated = c.ask(make_req(EITHER_SET, 1))
        self.assertEqual(calibrated.backend, "local")
        self.assertEqual(calibrated.route_reasons, ["backend_fallback"])
        uncalibrated = c.ask(make_req(JEV_SET, 2))
        self.assertEqual(uncalibrated.error, "jev_unavailable")
        self.assertEqual(uncalibrated.backend, "none")
        self.assertIsNone(uncalibrated.answers)
        self.assertEqual(jev.calls, [], "no key or offline: Jev's health says so before anything is sent")

    def test_jev_refusing_mid_run_is_down_for_the_rest_of_the_run(self):
        jev = FakeBackend("jev", noul_by_index("jev"), raises=BackendUnavailable("connection refused"))
        c = self.client(jev=jev)
        first = c.ask(make_req(EITHER_SET, 1))
        self.assertEqual((first.backend, first.route_reasons), ("local", ["backend_fallback"]))
        self.assertEqual(c.ask(make_req(JEV_SET, 2)).error, "jev_unavailable")
        self.assertEqual(len(jev.calls), 1, "the second request never reaches the dead Jev")
        self.assertTrue(any(m.startswith("Jev unavailable: connection refused") for m in self.logs))

    def test_jev_error_after_retries(self):
        jev = FakeBackend("jev", noul_by_index("jev"), raises=BackendError("boom", status=500))
        c = self.client(jev=jev)
        ans = c.ask(make_req(JEV_SET))
        self.assertEqual(ans.error, "jev_error")
        self.assertFalse(ans.ok)
        self.assertIn("boom", ans.detail)
        self.assertEqual(len(self.local.calls), 0)
        again = c.ask(make_req(JEV_SET, 1))
        self.assertEqual(again.error, "jev_error")
        self.assertEqual(len(jev.calls), 2, "an error is not an outage: the next request still tries Jev")

    def test_jev_error_on_a_calibrated_set_is_answered_by_local(self):
        jev = FakeBackend("jev", noul_by_index("jev"), raises=BackendError("boom", status=500))
        ans = self.client(jev=jev).ask(make_req(EITHER_SET))
        self.assertEqual((ans.backend, ans.route_reasons), ("local", ["backend_fallback"]))

    def test_both_backends_failing_reports_the_jev_error(self):
        jev = FakeBackend("jev", noul_by_index("jev"), raises=BackendError("jev boom", status=500))
        local = FakeBackend("local", noul_by_index("local"), version="0.1.0", raises=BackendError("local boom", 400))
        ans = self.client(jev=jev, local=local).ask(make_req(EITHER_SET))
        self.assertEqual(ans.error, "jev_error")
        self.assertEqual(ans.route_reasons, ["backend_fallback"])
        self.assertIn("jev boom", ans.detail)
        self.assertIn("local boom", ans.detail)

    def test_no_jev_sends_nothing_to_jev(self):
        c = self.client(no_jev=True)
        calibrated = c.ask(make_req(EITHER_SET, 1))
        self.assertEqual((calibrated.backend, calibrated.route_reasons), ("local", ["backend_fallback"]))
        uncalibrated = c.ask(make_req(JEV_SET, 2))
        self.assertEqual(uncalibrated.error, "jev_unavailable")
        self.assertEqual(uncalibrated.detail, "no_jev")
        self.assertEqual(self.jev.calls, [])

    def test_no_jev_with_a_box_goes_to_agent_not_local(self):
        c = self.client(no_jev=True)
        ans = c.ask(make_req(BOX_SET, questions=BOX_Q))
        self.assertEqual(ans.error, "jev_unavailable")
        self.assertEqual((self.jev.calls, self.local.calls), ([], []))

    def test_no_images_sends_no_image(self):
        img = write_png(self.tmp / "a.png")
        c = self.client(no_images=True)
        with_image = c.ask(make_req(EITHER_SET, image=img, sensitivity="private"))
        self.assertEqual(with_image.error, "jev_unavailable")
        self.assertEqual(with_image.detail, "no_images")
        self.assertEqual((self.jev.calls, self.local.calls), ([], []))
        text_only = c.ask(make_req(JEV_SET, 1))
        self.assertTrue(text_only.ok)
        self.assertEqual(len(self.jev.calls), 1)

    def test_decider_version_change_keeps_set_on_jev_and_prints_notice(self):
        local = FakeBackend("local", noul_by_index("local"), version="0.2.0")
        c = self.client(local=local)
        first, second = c.ask(make_req(LOCAL_SET, 1)), c.ask(make_req(LOCAL_SET, 2))
        for ans in (first, second):
            self.assertEqual(ans.backend, "jev")
            self.assertIn("backend_fallback", ans.route_reasons)
        self.assertEqual(local.calls, [])
        notices = [m for m in self.logs if "jevm eval --backend local --qset localset" in m]
        self.assertEqual(len(notices), 1, "once per set")
        self.assertIn("Decider changed", notices[0])

    def test_decider_model_change_counts_as_a_change(self):
        stale = replace(LOCAL_SET, calibrated={"local": {"app_version": "0.1.0", "model": "pplx-decider-flash-v1.1"}})
        ans = self.client().ask(make_req(stale))
        self.assertEqual(ans.backend, "jev")
        self.assertTrue(any("jevm eval --backend local --qset localset" in m for m in self.logs))

    def test_a_local_set_without_a_calibration_record_runs_on_jev(self):
        bare = replace(LOCAL_SET, calibrated={"local": None})
        ans = self.client().ask(make_req(bare))
        self.assertEqual((ans.backend, ans.route_reasons), ("jev", ["backend_fallback"]))
        self.assertFalse(any("Decider changed" in m for m in self.logs))

    def test_forced_backend_skips_calibration_and_never_falls_back(self):
        local = FakeBackend("local", noul_by_index("local"), version="9.9")
        c = self.client(local=local, force_backend="local")
        self.assertEqual(c.ask(make_req(LOCAL_SET)).backend, "local")
        down = FakeBackend("local", noul_by_index("local"), version="0.1.0", healthy=False)
        ans = self.client(local=down, force_backend="local", cache=Cache(self.tmp / "c2")).ask(make_req(JEV_SET))
        self.assertEqual(ans.error, "local_unavailable")
        self.assertEqual(self.jev.calls, [])
        jev_down = FakeBackend("jev", noul_by_index("jev"), healthy=False)
        forced_jev = self.client(jev=jev_down, force_backend="jev", cache=Cache(self.tmp / "c3"))
        self.assertEqual(forced_jev.ask(make_req(EITHER_SET)).error, "jev_unavailable")
        self.assertEqual(self.local.calls, [])

    def test_error_text_is_redacted_in_detail_and_log(self):
        jev = FakeBackend("jev", noul_by_index("jev"), raises=BackendUnavailable(f"auth failed for {FAKE_KEY}"))
        c = self.client(jev=jev, secrets=[FAKE_KEY])
        ans = c.ask(make_req(JEV_SET))
        self.assertIn("REDACTED", ans.detail)
        self.assertNotIn(FAKE_KEY, ans.detail)
        self.assertNotIn(FAKE_KEY, "\n".join(self.logs))


class LocalFailureTests(ClientCase):
    def test_local_400_falls_back_to_jev_for_that_request(self):
        for exc in (BackendError("bad request", status=400), LimitError("too many images")):
            with self.subTest(exc=type(exc).__name__):
                def hook(req, exc=exc):
                    if req.state["i"] == 1:
                        raise exc
                jev = FakeBackend("jev", noul_by_index("jev"))
                local = FakeBackend("local", noul_by_index("local", hook=hook), version="0.1.0")
                c = self.client(jev=jev, local=local, cache=Cache(self.tmp / type(exc).__name__))
                a0, a1, a2 = (c.ask(make_req(LOCAL_SET, i)) for i in range(3))
                self.assertEqual((a0.backend, a1.backend, a2.backend), ("local", "jev", "local"))
                self.assertEqual(a1.route_reasons, ["backend_fallback"])
                self.assertEqual((a0.route_reasons, a2.route_reasons), ([], []))
                self.assertEqual((len(local.calls), len(jev.calls)), (3, 1))

    def test_local_unreachable_mid_run_marks_it_unhealthy_for_the_run(self):
        def hook(req):
            raise BackendUnavailable("connection refused")
        local = FakeBackend("local", noul_by_index("local", hook=hook), version="0.1.0")
        c = self.client(local=local)
        first, second = c.ask(make_req(LOCAL_SET, 1)), c.ask(make_req(LOCAL_SET, 2))
        for ans in (first, second):
            self.assertEqual((ans.backend, ans.route_reasons), ("jev", ["backend_fallback"]))
        self.assertEqual(len(local.calls), 1, "the second request never reaches the dead Decider")

    def test_local_and_jev_both_unavailable_reports_local_unavailable(self):
        local = FakeBackend("local", noul_by_index("local"), version="0.1.0", healthy=False)
        jev = FakeBackend("jev", noul_by_index("jev"), healthy=False)
        ans = self.client(jev=jev, local=local).ask(make_req(LOCAL_SET))
        self.assertFalse(ans.ok)
        self.assertEqual(ans.error, "jev_unavailable")


class CacheAndDriftTests(ClientCase):
    def test_cache_hit_second_call_no_send(self):
        c = self.client()
        first, second = c.ask(make_req(JEV_SET, 3)), c.ask(make_req(JEV_SET, 3))
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(len(self.jev.calls), 1)
        self.assertEqual(second.answers, first.answers)
        self.assertEqual(second.request_sha256, first.request_sha256)
        self.assertEqual(second.model, "jev-1.13.0")
        again = self.client(jev=FakeBackend("jev", noul_by_index("jev"))).ask(make_req(JEV_SET, 3))
        self.assertTrue(again.cached, "a new client over the same cache folder reuses the answer")

    def test_reversed_options_not_served_from_cache(self):
        def q(criteria):
            return {"c_0": {"type": "choice", "instructions": "Which?", "criteria": criteria}}
        jev = FakeBackend("jev", by_qid({"c_0": choice({"a": 0.7, "b": 0.3})}))
        c = self.client(jev=jev)
        c.ask(make_req(JEV_SET, questions=q({"a": "A", "b": "B"})))
        reversed_ans = c.ask(make_req(JEV_SET, questions=q({"b": "B", "a": "A"})))
        self.assertFalse(reversed_ans.cached)
        self.assertEqual(len(jev.calls), 2)

    def test_cache_is_keyed_by_the_backend_actually_used(self):
        cache = Cache(self.tmp / "shared")
        # Same model id and no Decider version on both sides, so only the backend name can tell the keys apart.
        qs = replace(LOCAL_SET, models={"jev": "same-model", "local": "same-model"},
                     calibrated={"local": {"app_version": "", "model": "same-model"}})
        down = FakeBackend("local", noul_by_index("local"), healthy=False)
        via_jev = self.client(local=down, cache=cache).ask(make_req(qs, 4))
        self.assertEqual(via_jev.backend, "jev")
        local = FakeBackend("local", noul_by_index("local"))
        via_local = self.client(local=local, cache=cache).ask(make_req(qs, 4))
        self.assertEqual((via_local.backend, via_local.cached), ("local", False))
        self.assertEqual(len(local.calls), 1)

    def test_cache_key_covers_image_file_bytes(self):
        a, b = write_png(self.tmp / "a.png", shade=10), write_png(self.tmp / "b.png", shade=200)
        ref = "<<img-fixed>>"

        def with_file(path):
            return Request(JEV_SET, {"i": 0, "frame": ref}, NOUL_Q, {ref: path})

        c = self.client()
        c.ask(with_file(a))
        self.assertTrue(c.ask(with_file(a)).cached)
        self.assertFalse(c.ask(with_file(b)).cached, "same state and ref, different file bytes")

    def test_model_drift_logged_once(self):
        jev = FakeBackend("jev", noul_by_index("jev", model_override="jev-1.13.1"))
        c = self.client(jev=jev)
        c.ask(make_req(JEV_SET, 1))
        c.ask(make_req(JEV_SET, 2))
        drift = [m for m in self.logs if "model drift" in m]
        self.assertEqual(drift, ["model drift: jevset pinned jev-1.13.0, got jev-1.13.1"])
        self.assertEqual(c.ask(make_req(JEV_SET, 1)).model_resolved, "jev-1.13.1")

    def test_no_drift_message_when_model_matches(self):
        self.client().ask(make_req(JEV_SET))
        self.assertFalse([m for m in self.logs if "drift" in m])


class RunManyTests(ClientCase):
    def test_run_many_keeps_order_and_local_serial(self):
        lock = threading.Lock()
        inflight, peak = {"jev": 0, "local": 0}, {"jev": 0, "local": 0}
        local_started = threading.Event()
        overlap = []

        def watch(name):
            def hook(req):
                if name == "local":
                    local_started.set()
                elif req.state["i"] == 0:
                    overlap.append(local_started.wait(5))
                with lock:
                    inflight[name] += 1
                    peak[name] = max(peak[name], inflight[name])
                time.sleep(0.03)
                with lock:
                    inflight[name] -= 1
            return hook

        jev = FakeBackend("jev", noul_by_index("jev", hook=watch("jev")))
        local = FakeBackend("local", noul_by_index("local", hook=watch("local")), version="0.1.0")
        c = self.client(jev=jev, local=local, jev_concurrency=4)
        reqs = [make_req(LOCAL_SET if i % 3 == 2 else JEV_SET, i) for i in range(12)]
        answers = c.run_many(reqs)
        self.assertEqual([noul_p(a, "q_0") for a in answers], [i / 20 for i in range(12)])
        self.assertEqual([a.backend for a in answers], ["local" if i % 3 == 2 else "jev" for i in range(12)])
        self.assertEqual(peak["local"], 1)
        self.assertGreater(peak["jev"], 1)
        self.assertEqual(overlap, [True], "the local queue runs while the Jev pool is busy")

    def test_run_many_serialises_a_jev_request_that_falls_back_to_local(self):
        lock = threading.Lock()
        state = {"now": 0, "peak": 0}

        def hook(req):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.02)
            with lock:
                state["now"] -= 1

        jev = FakeBackend("jev", noul_by_index("jev"), raises=BackendError("boom", 500))
        local = FakeBackend("local", noul_by_index("local", hook=hook), version="0.1.0")
        c = self.client(jev=jev, local=local, jev_concurrency=6)
        answers = c.run_many([make_req(EITHER_SET, i) for i in range(8)])
        self.assertTrue(all(a.backend == "local" for a in answers))
        self.assertEqual(state["peak"], 1)

    def test_run_many_returns_errors_in_place(self):
        c = self.client(no_jev=True)
        answers = c.run_many([make_req(JEV_SET, 0), make_req(EITHER_SET, 1)])
        self.assertEqual([a.ok for a in answers], [False, True])

    def test_run_many_empty(self):
        self.assertEqual(self.client().run_many([]), [])


class PreflightTests(ClientCase):
    LINE_JEV = r"^jev: \d+ requests \(\d+ cached\), ~\d+ input tokens, ~\$\d+\.\d+$"
    LINE_LOCAL = r"^local: \d+ requests \(\d+ cached\), ~\d+\.\d+ s serial$"

    def test_preflight_dry_run_returns_0_and_budget_exceeded_returns_3(self):
        c = self.client()
        c.ask(make_req(JEV_SET, 0))
        reqs = [make_req(JEV_SET, 0), make_req(JEV_SET, 1), make_req(LOCAL_SET, 2)]
        sent_before = len(self.jev.calls) + len(self.local.calls)

        self.logs.clear()
        self.assertEqual(c.preflight(reqs, budget_usd=None, dry_run=True), 0)
        lines = [m for m in self.logs if m.startswith(("jev:", "local:"))]
        self.assertEqual(len(lines), 2, "one line per backend")
        self.assertRegex(lines[0], self.LINE_JEV)
        self.assertRegex(lines[1], self.LINE_LOCAL)
        self.assertIn("jev: 2 requests (1 cached)", lines[0])
        self.assertIn("local: 1 requests (0 cached)", lines[1])

        self.assertEqual(c.preflight(reqs, budget_usd=1e-12, dry_run=False), 3)
        self.assertIsNone(c.preflight(reqs, budget_usd=100.0, dry_run=False))
        self.assertIsNone(c.preflight(reqs, budget_usd=None, dry_run=False))
        self.assertEqual(len(self.jev.calls) + len(self.local.calls), sent_before, "preflight never sends")

    def test_dry_run_wins_over_an_exceeded_budget_but_says_so(self):
        c = self.client()
        self.assertEqual(c.preflight([make_req(JEV_SET)], budget_usd=1e-12, dry_run=True), 0)
        self.assertTrue(any("budget" in m for m in self.logs))

    def test_estimate_counts_tokens_for_uncached_requests_only(self):
        c = self.client()
        c.ask(make_req(JEV_SET, 0))
        est = c.estimate([make_req(JEV_SET, 0), make_req(JEV_SET, 1)])
        text = json.dumps({"state": make_req(JEV_SET, 1).state, "questions": NOUL_Q},
                          separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(est["jev"]["requests"], 2)
        self.assertEqual(est["jev"]["cached"], 1)
        self.assertEqual(est["jev"]["input_tokens"], tokens.request_tokens(len(text), []))
        self.assertAlmostEqual(est["jev"]["usd"], tokens.usd(est["jev"]["input_tokens"]))

    def test_estimate_uses_the_size_an_image_is_sent_at(self):
        img = write_png(self.tmp / "wide.png", 1920, 1080)
        req = make_req(JEV_SET, image=img)
        est = self.client().estimate([req])
        text = json.dumps({"state": req.state, "questions": req.questions}, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(est["jev"]["input_tokens"], tokens.request_tokens(len(text), [(1568, 882)]))

    def test_estimate_local_seconds_and_routing(self):
        img = write_png(self.tmp / "a.png")
        questions = {f"q_{k}": {"type": "noul", "instructions": "x"} for k in range(3)}
        req = make_req(LOCAL_SET, image=img, questions=questions)
        c = self.client()
        est = c.estimate([req, make_req(EITHER_SET, 1)])
        self.assertEqual(est["local"]["requests"], 1)
        self.assertAlmostEqual(est["local"]["serial_s"], tokens.local_seconds(3, 1))
        self.assertEqual(est["jev"]["requests"], 1)

    def test_preflight_says_how_many_rows_go_to_agent(self):
        c = self.client(no_jev=True)
        c.preflight([make_req(JEV_SET, 0), make_req(JEV_SET, 1), make_req(EITHER_SET, 2)], budget_usd=None, dry_run=True)
        self.assertTrue(any(m.startswith("none: 2 requests") for m in self.logs))
        self.assertTrue(any(m.startswith("jev: 0 requests") for m in self.logs))
        self.assertEqual(self.jev.calls, [])


class LedgerTests(ClientCase):
    def test_ledger_to_dict_shape(self):
        img = write_png(self.tmp / "a.png")
        jev = FakeBackend("jev", noul_by_index("jev", tokens_=1000))
        local = FakeBackend("local", noul_by_index("local"), version="0.1.0")
        image_set = replace(JEV_SET, id="imgset", models={"jev": "jev-1.14.0", "local": "pplx-decider-flash-v1.2"})
        c = self.client(jev=jev, local=local)
        c.ask(make_req(JEV_SET, 1))
        c.ask(make_req(image_set, 2, image=img))
        c.ask(make_req(LOCAL_SET, 3))
        c.ask(make_req(JEV_SET, 1))
        d = c.ledger.to_dict()
        self.assertEqual(set(d), {"jev", "local"})
        self.assertEqual(set(d["jev"]), {"models", "requests", "input_tokens", "usd", "cache_hits"})
        self.assertEqual(set(d["local"]), {"model", "app_version", "requests", "serial_s", "cache_hits"})
        self.assertEqual(d["jev"]["models"], {"text": "jev-1.13.0", "image": "jev-1.14.0"})
        self.assertEqual((d["jev"]["requests"], d["jev"]["cache_hits"], d["jev"]["input_tokens"]), (2, 1, 2000))
        self.assertAlmostEqual(d["jev"]["usd"], tokens.usd(2000))
        self.assertEqual(d["local"]["model"], "pplx-decider-flash-v1.2")
        self.assertEqual(d["local"]["app_version"], "0.1.0")
        self.assertEqual((d["local"]["requests"], d["local"]["cache_hits"]), (1, 0))
        self.assertAlmostEqual(d["local"]["serial_s"], 0.5)
        json.dumps(d)

    def test_empty_ledger_has_the_same_keys(self):
        d = Ledger().to_dict()
        self.assertEqual(d["jev"], {"models": {"text": "", "image": ""}, "requests": 0, "input_tokens": 0,
                                    "usd": 0.0, "cache_hits": 0})
        self.assertEqual(d["local"], {"model": "", "app_version": "", "requests": 0, "serial_s": 0.0, "cache_hits": 0})

    def test_failed_requests_are_not_counted(self):
        jev = FakeBackend("jev", noul_by_index("jev"), raises=BackendError("boom", 500))
        c = self.client(jev=jev)
        c.ask(make_req(JEV_SET))
        self.assertEqual(c.ledger.to_dict()["jev"]["requests"], 0)


class MakeClientTests(unittest.TestCase):
    """make_client imports the real backends lazily; these tests stand in stub modules so they run without them."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.built = {}

        outer = self

        class StubJev:
            name = "jev"

            def __init__(self, **kw):
                outer.built["jev"] = kw

            def send(self, req, model):
                raise BackendError(f"rejected {FAKE_KEY}", status=401)

            def header_flags(self, req):
                return []

            def version(self):
                return ""

            def health(self):
                return {}

        class StubLocal(StubJev):
            name = "local"

            def __init__(self, **kw):
                outer.built["local"] = kw

        jev_mod, local_mod = types.ModuleType("jevlib.backends.jev"), types.ModuleType("jevlib.backends.local")
        jev_mod.JevBackend, local_mod.LocalBackend = StubJev, StubLocal
        patchers = [mock.patch.dict(sys.modules, {"jevlib.backends.jev": jev_mod, "jevlib.backends.local": local_mod}),
                    mock.patch("jevlib.client.load_key", return_value=FAKE_KEY),
                    mock.patch.dict(os.environ)]
        for p in patchers:
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("JEVM_DECIDER_URL", None)

    def test_defaults(self):
        c = make_client(cache_dir=self.tmp)
        self.assertEqual(sorted(c.backends), ["jev", "local"])
        self.assertEqual(c.jev_concurrency, 6)
        self.assertIsNone(self.built["local"]["url"], "no url: LocalBackend picks its own default")
        self.assertEqual(c.cache.root, self.tmp)

    def test_decide_block_sets_url_and_concurrency(self):
        decide = {"policy": "default", "budget_usd": 1.0,
                  "backends": {"jev": {"concurrency": 3}, "local": {"url": "http://127.0.0.1:9999"}}}
        c = make_client(decide, cache_dir=self.tmp)
        self.assertEqual(c.jev_concurrency, 3)
        self.assertEqual(self.built["local"]["url"], "http://127.0.0.1:9999")

    def test_env_url_beats_the_spec_url(self):
        os.environ["JEVM_DECIDER_URL"] = "http://127.0.0.1:7777"
        decide = {"backends": {"local": {"url": "http://127.0.0.1:9999"}}}
        make_client(decide, cache_dir=self.tmp)
        self.assertEqual(self.built["local"]["url"], "http://127.0.0.1:7777")

    def test_flags_pass_through(self):
        c = make_client(cache_dir=self.tmp, no_jev=True, no_images=True, force_backend="jev")
        self.assertEqual((c.no_jev, c.no_images, c.force_backend), (True, True, "jev"))

    def test_the_key_is_redacted_from_error_text(self):
        logs = []
        c = make_client(cache_dir=self.tmp, log=logs.append)
        ans = c.ask(make_req(JEV_SET))
        self.assertEqual(ans.error, "jev_error")
        self.assertIn("REDACTED", ans.detail)
        self.assertNotIn(FAKE_KEY, ans.detail)
        self.assertNotIn(FAKE_KEY, "\n".join(logs))

    @unittest.skipUnless(importlib.util.find_spec("jevlib.backends.jev") and importlib.util.find_spec("jevlib.backends.local"),
                         "the real backends are not in the tree")
    def test_builds_the_real_backends(self):
        sys.modules.pop("jevlib.backends.jev", None)
        sys.modules.pop("jevlib.backends.local", None)
        c = make_client(cache_dir=self.tmp)
        self.assertEqual(sorted(c.backends), ["jev", "local"])
        self.assertEqual([c.backends[n].name for n in ("jev", "local")], ["jev", "local"])


if __name__ == "__main__":
    unittest.main()
