"""Tests for the Jev and Decider backends.

Every test drives a backend through a recording fake transport and a recording `sleep`: nothing here opens a socket.
`tests/__init__.py` turns JEVM_OFFLINE on for the whole run, so each case turns it off again for itself; only
`test_offline_env_blocks_both` leaves it on. Keys in these tests are made-up strings, never the real JEV_API.
"""
import base64
import http.client
import io
import json
import os
import plistlib
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import zlib
from pathlib import Path
from unittest import mock

from jevlib.backends import http as bhttp
from jevlib.backends import jev, local
from jevlib.backends.base import (IMAGES_HEADER, Backend, BackendError, BackendUnavailable, LimitError)
from jevlib.backends.jev import JevBackend
from jevlib.backends.local import LocalBackend
from jevlib.request import Request, image_ref, load_qset

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "qsets"
KEY = "sk-jev-test-0123456789abcdef"
JEV_MODEL = "jev-1.14.0"
LOCAL_MODEL = "pplx-decider-flash-v1.2"
NOUL_Q = {"q": {"type": "noul", "instructions": "Is it?"}}


# ---------------------------------------------------------------- helpers
def write_png(path, w, h):
    """A valid grey PNG of w x h pixels."""
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x80" * w for _ in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


def write_jpeg(path, w, h):
    """A JPEG that is only a header (SOI, SOF0, EOI): enough for the size and format checks, which never decode."""
    sof = b"\xff\xc0" + struct.pack(">H", 11) + b"\x08" + struct.pack(">HH", h, w) + b"\x01\x01\x11\x00"
    path.write_bytes(b"\xff\xd8" + sof + b"\xff\xd9")
    return path


def reply(answers=None, model="m-resolved", tokens=7):
    """A 200 step for FakeTransport."""
    body = {"model": model, "answers": {"q": {"noul": 0.5}} if answers is None else answers,
            "usage": {"input_tokens": tokens}}
    return 200, {}, json.dumps(body).encode("utf-8")


def status(code, headers=None, body=b""):
    return code, headers or {}, body


class FakeTransport:
    """A scripted `http.Transport`. A step is a `(status, headers, body)` tuple, or an exception to raise.

    `repeat` replays the last step forever; `delay` holds each call open that long (for the single-flight test).
    `.calls` records every call.
    """

    def __init__(self, *steps, repeat=False, delay=0.0):
        self.steps = list(steps)
        self.repeat = repeat
        self.delay = delay
        self.calls = []
        self.in_flight = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(self, method, url, body, headers, timeout):
        with self._lock:
            self.calls.append({"method": method, "url": url, "body": body, "headers": dict(headers),
                               "timeout": timeout})
            step = self.steps[0] if self.repeat and len(self.steps) == 1 else self.steps.pop(0)
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay:
                time.sleep(self.delay)
            if isinstance(step, BaseException):
                raise step
            return step
        finally:
            with self._lock:
                self.in_flight -= 1


class BackendCase(unittest.TestCase):
    """Online by default (JEVM_OFFLINE off), a temp dir, a recording sleep and the `sample@1` question set."""

    def setUp(self):
        env = mock.patch.dict(os.environ, {"JEVM_OFFLINE": "0"})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("JEVM_DECIDER_URL", None)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.sleeps = []
        self.qs = load_qset("sample", 1, root=FIXTURES)

    def request(self, state=None, questions=None, images=None):
        return Request(self.qs, {"text": "hello"} if state is None else state,
                       NOUL_Q if questions is None else questions, images or {})

    def jev(self, transport, **kw):
        return JevBackend(key_loader=lambda: KEY, transport=transport, sleep=self.sleeps.append, **kw)

    def local(self, transport, **kw):
        kw.setdefault("plist", self.dir / "no-such-Info.plist")
        return LocalBackend(transport=transport, sleep=self.sleeps.append, **kw)

    def refs(self, n, path):
        """`n` distinct refs that all point at one file."""
        return {f"<<img-{i:012d}>>": path for i in range(n)}


# ---------------------------------------------------------------- the transport seam
class HttpTests(unittest.TestCase):
    def response(self, status_code=200, headers=None, body=b"{}"):
        resp = mock.MagicMock()
        resp.status = status_code
        resp.headers = headers or {}
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    def test_urllib_transport_success(self):
        resp = self.response(200, {"X-Test": "1"}, b'{"ok":1}')
        with mock.patch("urllib.request.urlopen", return_value=resp) as urlopen:
            out = bhttp.urllib_transport("POST", "http://x.test/a b", b"{}", {"A": "b"}, 3.5)
        self.assertEqual(out, (200, {"X-Test": "1"}, b'{"ok":1}'))
        request = urlopen.call_args.args[0]
        self.assertEqual((request.get_method(), request.data), ("POST", b"{}"))
        self.assertEqual(request.get_header("A"), "b")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 3.5)

    def test_urllib_transport_http_error_comes_back_as_a_status(self):
        err = urllib.error.HTTPError("http://x.test", 429, "Too Many", {"Retry-After": "3"}, io.BytesIO(b"slow"))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            out = bhttp.urllib_transport("GET", "http://x.test", None, {}, 1.0)
        self.assertEqual(out, (429, {"Retry-After": "3"}, b"slow"))

    def test_urllib_transport_unreachable_raises_oserror(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            with self.assertRaises(OSError):
                bhttp.urllib_transport("GET", "http://x.test", None, {}, 1.0)
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("slow")):
            with self.assertRaises(OSError):
                bhttp.urllib_transport("GET", "http://x.test", None, {}, 1.0)

    def test_urllib_transport_wraps_http_client_errors(self):
        with mock.patch("urllib.request.urlopen", side_effect=http.client.IncompleteRead(b"ab")):
            with self.assertRaises(OSError):
                bhttp.urllib_transport("GET", "http://x.test", None, {}, 1.0)

    def test_retry_after_numeric_only(self):
        self.assertEqual(bhttp.retry_after({"Retry-After": "3"}), 3.0)
        self.assertEqual(bhttp.retry_after({"retry-after": "2.5"}), 2.5)
        self.assertIsNone(bhttp.retry_after({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}))
        self.assertIsNone(bhttp.retry_after({"Retry-After": "-1"}))
        self.assertIsNone(bhttp.retry_after({}))

    def test_snippet_truncates_and_tolerates_bad_bytes(self):
        self.assertEqual(len(bhttp.snippet(b"x" * 5000)), 1500)
        self.assertIn("�", bhttp.snippet(b"ab\xff"))


# ---------------------------------------------------------------- Jev
class JevBackendTests(BackendCase):
    def test_implements_the_backend_protocol(self):
        b = self.jev(FakeTransport())
        self.assertIsInstance(b, Backend)
        self.assertEqual((b.name, b.version()), ("jev", ""))
        self.assertEqual(jev.URL, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(jev.IMAGES_HEADER, "TypeSafe-Danger-Zone-Images")

    def test_jev_images_header_only_with_images(self):
        png = write_png(self.dir / "a.png", 300, 300)
        ref = image_ref(png)
        b = self.jev(FakeTransport())
        with_image = self.request({"frame": ref}, images={ref: png})
        text_only = self.request()
        self.assertEqual(b.headers(with_image)[IMAGES_HEADER], "true")
        self.assertNotIn(IMAGES_HEADER, b.headers(text_only))
        self.assertEqual(b.header_flags(with_image), [IMAGES_HEADER])
        self.assertEqual(b.header_flags(text_only), [])

    def test_jev_send_shape(self):
        t = FakeTransport(reply({"q": {"noul": 0.9}}, model="jev-1.14.0", tokens=123))
        out = self.jev(t).send(self.request(), JEV_MODEL)
        self.assertEqual(out["answers"], {"q": {"noul": 0.9}})
        self.assertEqual((out["model"], out["input_tokens"]), ("jev-1.14.0", 123))
        self.assertGreaterEqual(out["latency_s"], 0.0)
        (call,) = t.calls
        self.assertEqual((call["method"], call["url"], call["timeout"]), ("POST", jev.URL, 90.0))
        self.assertEqual(call["headers"]["Authorization"], "Bearer " + KEY)
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertEqual(call["headers"]["User-Agent"], "jev-motion/0.1")
        self.assertEqual(json.loads(call["body"]),
                         {"model": JEV_MODEL, "state": {"text": "hello"}, "questions": NOUL_Q})

    def test_jev_body_keeps_choice_option_order(self):
        questions = {"c": {"type": "choice", "options": {"zeta": "z", "alpha": "a"}}}
        t = FakeTransport(reply({"c": {"choice": "zeta"}}))
        self.jev(t).send(self.request(questions=questions), JEV_MODEL)
        sent = json.loads(t.calls[0]["body"])
        self.assertEqual(list(sent["questions"]["c"]["options"]), ["zeta", "alpha"])

    def test_jev_body_assets(self):
        jpg = write_jpeg(self.dir / "a.jpg", 400, 300)
        png = write_png(self.dir / "b.png", 300, 300)
        r1, r2 = image_ref(jpg), image_ref(png, tag="frame")
        b = self.jev(FakeTransport())
        body = b.body(self.request({"a": r1, "b": r2}, images={r1: jpg, r2: png}), JEV_MODEL)
        self.assertEqual(body["assets"][r1], {"type": "image", "format": "jpg",
                                              "data_base64": base64.b64encode(jpg.read_bytes()).decode("ascii")})
        self.assertEqual(body["assets"][r2], {"type": "image", "format": "png",
                                              "data_base64": base64.b64encode(png.read_bytes()).decode("ascii")})
        self.assertEqual(body["model"], JEV_MODEL)
        self.assertNotIn("assets", b.body(self.request(), JEV_MODEL))

    def test_jev_rejects_out_of_range_image_before_sending(self):
        small = write_png(self.dir / "small.png", 100, 100)
        huge = write_png(self.dir / "wide.png", 1600, 300)
        for path in (small, huge):
            t = FakeTransport()
            ref = image_ref(path)
            with self.assertRaises(LimitError):
                self.jev(t).send(self.request({"f": ref}, images={ref: path}), JEV_MODEL)
            self.assertEqual(t.calls, [])

    def test_jev_rejects_unreadable_image_before_sending(self):
        t = FakeTransport()
        junk = self.dir / "junk.png"
        junk.write_bytes(b"not an image at all")
        for path in (junk, self.dir / "missing.png"):
            with self.assertRaises(LimitError):
                self.jev(t).send(self.request({"f": "<<img-x>>"}, images={"<<img-x>>": path}), JEV_MODEL)
        self.assertEqual(t.calls, [])

    def test_jev_retries_429_honours_retry_after(self):
        t = FakeTransport(status(429, {"Retry-After": "3"}, b"slow down"), reply())
        out = self.jev(t).send(self.request(), JEV_MODEL)
        self.assertEqual(self.sleeps, [3])
        self.assertEqual(len(t.calls), 2)
        self.assertEqual(out["input_tokens"], 7)

    def test_jev_retry_after_lookup_ignores_header_case(self):
        t = FakeTransport(status(429, {"retry-after": "2"}), reply())
        self.jev(t).send(self.request(), JEV_MODEL)
        self.assertEqual(self.sleeps, [2])

    def test_jev_retries_408_529_and_5xx(self):
        for code in (408, 529, 500, 502, 503, 504):
            self.sleeps.clear()
            t = FakeTransport(status(code), reply())
            self.jev(t).send(self.request(), JEV_MODEL)
            self.assertEqual((len(t.calls), self.sleeps), (2, [1]), code)

    def test_jev_backoff_doubles_capped(self):
        t = FakeTransport(status(503, body=b"down"), repeat=True)
        with self.assertRaises(BackendError) as cm:
            self.jev(t).send(self.request(), JEV_MODEL)
        self.assertEqual(len(t.calls), 5)
        self.assertEqual(self.sleeps, [1, 2, 4, 8])
        self.assertEqual(cm.exception.status, 503)
        self.assertNotIsInstance(cm.exception, (LimitError, BackendUnavailable))
        self.sleeps.clear()
        t = FakeTransport(status(503), repeat=True)
        with self.assertRaises(BackendError):
            self.jev(t, max_tries=7).send(self.request(), JEV_MODEL)
        self.assertEqual(len(t.calls), 7)
        self.assertEqual(self.sleeps[-2:], [16, 20])
        self.assertEqual(self.sleeps, [1, 2, 4, 8, 16, 20])

    def test_jev_400_raises_at_once_redacted(self):
        t = FakeTransport(status(400, body=json.dumps({"error": f"bad key {KEY}"}).encode()))
        with self.assertRaises(BackendError) as cm:
            self.jev(t).send(self.request(), JEV_MODEL)
        err = cm.exception
        self.assertEqual((len(t.calls), self.sleeps, err.status), (1, [], 400))
        self.assertIn("REDACTED", str(err))
        self.assertNotIn(KEY, str(err))
        self.assertIsNone(err.__cause__)

    def test_jev_other_4xx_raise_at_once(self):
        for code in (401, 403, 404, 422):
            t = FakeTransport(status(code, body=b"no"))
            with self.assertRaises(BackendError) as cm:
                self.jev(t).send(self.request(), JEV_MODEL)
            self.assertEqual((len(t.calls), cm.exception.status), (1, code))

    def test_jev_exhausted_error_is_redacted_too(self):
        t = FakeTransport(status(503, body=f"echo {KEY}".encode()), repeat=True)
        with self.assertRaises(BackendError) as cm:
            self.jev(t, max_tries=2).send(self.request(), JEV_MODEL)
        self.assertNotIn(KEY, str(cm.exception))
        self.assertIn("REDACTED", str(cm.exception))

    def test_jev_connection_error_retried_then_unavailable(self):
        t = FakeTransport(OSError(f"refused for {KEY}"), repeat=True)
        with self.assertRaises(BackendUnavailable) as cm:
            self.jev(t, max_tries=3).send(self.request(), JEV_MODEL)
        self.assertEqual((len(t.calls), self.sleeps), (3, [1, 2]))
        self.assertNotIn(KEY, str(cm.exception))
        t = FakeTransport(TimeoutError("slow"), reply())
        self.sleeps.clear()
        self.assertEqual(self.jev(t).send(self.request(), JEV_MODEL)["input_tokens"], 7)

    def test_jev_no_key_unavailable(self):
        t = FakeTransport()
        b = JevBackend(key_loader=lambda: "", transport=t, sleep=self.sleeps.append)
        with self.assertRaises(BackendUnavailable) as cm:
            b.send(self.request(), JEV_MODEL)
        self.assertIn("JEV_API", str(cm.exception))
        self.assertEqual(t.calls, [])
        with self.assertRaises(BackendUnavailable):
            b.health()
        with self.assertRaises(BackendUnavailable):
            b.headers(self.request())

    def test_jev_non_json_and_malformed_responses_are_errors(self):
        for raw in (b"<html>oops</html>", b"[]", b'{"model": "m"}', b'{"answers": []}'):
            t = FakeTransport(status(200, body=raw))
            with self.assertRaises(BackendError) as cm:
                self.jev(t).send(self.request(), JEV_MODEL)
            self.assertEqual(len(t.calls), 1)
            self.assertIsNone(cm.exception.status)

    def test_jev_missing_usage_counts_zero_tokens(self):
        t = FakeTransport(status(200, body=b'{"answers": {"q": {"noul": 0.1}}}'))
        out = self.jev(t).send(self.request(), JEV_MODEL)
        self.assertEqual((out["input_tokens"], out["model"]), (0, None))

    def test_jev_health_is_local_and_never_carries_the_key(self):
        t = FakeTransport()
        info = self.jev(t).health()
        self.assertEqual(info["status"], "ready")
        self.assertNotIn(KEY, json.dumps(info))
        self.assertEqual(t.calls, [])

    def test_jev_never_stores_the_key(self):
        t = FakeTransport(reply())
        b = self.jev(t)
        b.send(self.request(), JEV_MODEL)
        self.assertNotIn(KEY, repr(vars(b)))


# ---------------------------------------------------------------- Decider
class LocalBackendTests(BackendCase):
    def test_implements_the_backend_protocol_and_constants(self):
        b = self.local(FakeTransport())
        self.assertIsInstance(b, Backend)
        self.assertEqual(b.name, "local")
        self.assertEqual(b.header_flags(self.request()), [])
        self.assertEqual((local.DEFAULT_URL, local.DEFAULT_MODEL, local.MAX_IMAGES),
                         ("http://127.0.0.1:8080", "pplx-decider-flash-v1.2", 8))
        self.assertEqual(local.INFO_PLIST, Path("/Applications/Decider.app/Contents/Info.plist"))

    def test_local_send_shape(self):
        t = FakeTransport(reply({"q": {"noul": 0.99}}, model=LOCAL_MODEL, tokens=55))
        out = self.local(t).send(self.request(), LOCAL_MODEL)
        self.assertEqual(out["answers"], {"q": {"noul": 0.99}})
        self.assertEqual((out["model"], out["input_tokens"]), (LOCAL_MODEL, 55))
        (call,) = t.calls
        self.assertEqual((call["method"], call["url"], call["timeout"]),
                         ("POST", "http://127.0.0.1:8080/v1/decisions", 600.0))
        self.assertNotIn("Authorization", call["headers"])
        self.assertNotIn(IMAGES_HEADER, call["headers"])
        self.assertEqual(json.loads(call["body"]),
                         {"model": LOCAL_MODEL, "state": {"text": "hello"}, "questions": NOUL_Q})

    def test_local_inlines_images(self):
        jpg = write_jpeg(self.dir / "a.jpg", 400, 300)
        png = write_png(self.dir / "b.png", 300, 300)
        r1, r2, r3 = image_ref(jpg), image_ref(png), "<<img-instr000001>>"
        questions = {"q": {"type": "noul", "instructions": {"text": "Same as the sample?", "sample": r3},
                           "criteria": [r2]}}
        b = self.local(FakeTransport())
        body = b.body(self.request({"frame": r1, "note": "plain"}, questions, {r1: jpg, r2: png, r3: jpg}),
                      LOCAL_MODEL)
        b64 = base64.b64encode(jpg.read_bytes()).decode("ascii")
        self.assertEqual(body["state"]["frame"],
                         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}})
        self.assertEqual(body["state"]["note"], "plain")
        self.assertEqual(body["questions"]["q"]["instructions"]["sample"]["image_url"]["url"],
                         "data:image/jpeg;base64," + b64)
        self.assertEqual(body["questions"]["q"]["criteria"][0]["image_url"]["url"],
                         "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii"))
        self.assertEqual(body["questions"]["q"]["instructions"]["text"], "Same as the sample?")
        self.assertNotIn("assets", body)
        self.assertEqual(body["model"], LOCAL_MODEL)

    def test_local_body_keeps_insertion_order_and_leaves_the_request_untouched(self):
        png = write_png(self.dir / "a.png", 300, 300)
        ref = image_ref(png)
        state = {"z": "last-key-first", "frame": ref, "a": ["x", ref]}
        questions = {"c": {"type": "choice", "options": {"zeta": "z", "alpha": "a"}}}
        req = self.request(state, questions, {ref: png})
        body = self.local(FakeTransport()).body(req, LOCAL_MODEL)
        self.assertEqual(list(body["state"]), ["z", "frame", "a"])
        self.assertEqual(list(body["questions"]["c"]["options"]), ["zeta", "alpha"])
        self.assertEqual(req.state["frame"], ref)
        self.assertEqual(req.state["a"], ["x", ref])

    def test_local_rejects_box_before_sending(self):
        t = FakeTransport()
        questions = {"ok": {"type": "noul", "instructions": "x"}, "where": {"type": "box", "instructions": "y"}}
        with self.assertRaises(LimitError) as cm:
            self.local(t).send(self.request(questions=questions), LOCAL_MODEL)
        self.assertIn("box", str(cm.exception).lower())
        self.assertEqual(t.calls, [])

    def test_local_rejects_more_than_8_images(self):
        png = write_png(self.dir / "a.png", 300, 300)
        t = FakeTransport(reply(), reply())
        b = self.local(t)
        nine = self.refs(9, png)
        with self.assertRaises(LimitError) as cm:
            b.send(self.request({"frames": list(nine)}, images=nine), LOCAL_MODEL)
        self.assertIn("8", str(cm.exception))
        self.assertEqual(t.calls, [])
        eight = self.refs(8, png)
        b.send(self.request({"frames": list(eight)}, images=eight), LOCAL_MODEL)
        self.assertEqual(len(t.calls), 1)

    def test_local_image_limit_counts_state_plus_each_question(self):
        png = write_png(self.dir / "a.png", 300, 300)
        refs = self.refs(9, png)
        names = list(refs)
        b = self.local(FakeTransport())
        # 5 in the state + 4 in one question = 9 for that question
        questions = {"q": {"type": "choice", "options": {f"o{i}": r for i, r in enumerate(names[5:])}}}
        with self.assertRaises(LimitError):
            b.body(self.request({"frames": names[:5]}, questions, refs), LOCAL_MODEL)
        # 4 in the state + 4 in each of two questions = 8 per question, 12 in all: fine
        questions = {"a": {"type": "choice", "options": {f"o{i}": r for i, r in enumerate(names[4:8])}},
                     "b": {"type": "choice", "options": {f"o{i}": r for i, r in enumerate(names[:4])}}}
        refs8 = {r: png for r in names[:8]}
        body = b.body(self.request({"frames": names[:4]}, questions, refs8), LOCAL_MODEL)
        self.assertEqual(len(body["questions"]), 2)

    def test_local_rejects_embedded_ref(self):
        png = write_png(self.dir / "a.png", 300, 300)
        ref = image_ref(png)
        t = FakeTransport()
        with self.assertRaises(LimitError) as cm:
            self.local(t).send(self.request({"note": f"look at {ref} now"}, images={ref: png}), LOCAL_MODEL)
        self.assertIn("whole value", str(cm.exception))
        self.assertEqual(t.calls, [])

    def test_local_rejects_missing_ref(self):
        png = write_png(self.dir / "a.png", 300, 300)
        ref = image_ref(png)
        t = FakeTransport()
        with self.assertRaises(LimitError):
            self.local(t).send(self.request({"note": "no image mentioned"}, images={ref: png}), LOCAL_MODEL)
        self.assertEqual(t.calls, [])

    def test_local_rejects_unreadable_image(self):
        t = FakeTransport()
        junk = self.dir / "junk.bin"
        junk.write_bytes(b"GIF89a....")
        with self.assertRaises(LimitError):
            self.local(t).send(self.request({"f": "<<img-x>>"}, images={"<<img-x>>": junk}), LOCAL_MODEL)
        with self.assertRaises(LimitError):
            self.local(t).send(self.request({"f": "<<img-x>>"}, images={"<<img-x>>": self.dir / "gone.png"}),
                               LOCAL_MODEL)
        self.assertEqual(t.calls, [])

    def test_local_health_requires_ready_and_images(self):
        ready = {"status": "ready", "model": "Qwen/Qwen3.5-4B", "images": True, "mode": "per-question"}
        t = FakeTransport(status(200, body=json.dumps(ready).encode()))
        info = self.local(t).health()
        self.assertEqual((t.calls[0]["method"], t.calls[0]["url"]), ("GET", "http://127.0.0.1:8080/health"))
        self.assertEqual(info["status"], "ready")
        self.assertEqual(info["app_version"], "")
        for bad in ({"status": "loading", "images": True}, {"status": "ready", "images": False},
                    {"status": "ready"}, {"images": True}):
            t = FakeTransport(status(200, body=json.dumps(bad).encode()))
            with self.assertRaises(BackendUnavailable, msg=bad):
                self.local(t).health()

    def test_local_health_failures_are_unavailable(self):
        for step in (status(500), status(200, body=b"not json"), status(200, body=b"[]"),
                     ConnectionRefusedError("refused"), TimeoutError("slow")):
            with self.assertRaises(BackendUnavailable, msg=repr(step)):
                self.local(FakeTransport(step)).health()
        self.assertEqual(self.sleeps, [])

    def test_local_health_carries_app_version(self):
        plist = self.dir / "Info.plist"
        plist.write_bytes(self.plist_bytes({"CFBundleShortVersionString": "0.1.0"}))
        t = FakeTransport(status(200, body=b'{"status": "ready", "images": true}'))
        self.assertEqual(self.local(t, plist=plist).health()["app_version"], "0.1.0")

    @staticmethod
    def plist_bytes(data):
        return plistlib.dumps(data)

    def test_local_version_from_plist(self):
        plist = self.dir / "Info.plist"
        plist.write_bytes(self.plist_bytes({"CFBundleShortVersionString": "0.1.0", "CFBundleVersion": "9"}))
        self.assertEqual(self.local(FakeTransport(), plist=plist).version(), "0.1.0")
        self.assertEqual(self.local(FakeTransport(), plist=self.dir / "missing.plist").version(), "")
        plist.write_bytes(self.plist_bytes({"CFBundleVersion": "9"}))
        self.assertEqual(self.local(FakeTransport(), plist=plist).version(), "")
        for junk in (b"not a plist", b"<?xml version='1.0'?><plist><dict>", b"", self.plist_bytes(["a"])):
            plist.write_bytes(junk)
            self.assertEqual(self.local(FakeTransport(), plist=plist).version(), "", junk)

    def test_local_400_raises_at_once(self):
        t = FakeTransport(status(400, body=b"bad request"))
        with self.assertRaises(BackendError) as cm:
            self.local(t).send(self.request(), LOCAL_MODEL)
        self.assertEqual((len(t.calls), self.sleeps, cm.exception.status), (1, [], 400))
        self.assertNotIsInstance(cm.exception, (LimitError, BackendUnavailable))
        self.assertIn("bad request", str(cm.exception))

    def test_local_retries_connection_error(self):
        t = FakeTransport(OSError("connection refused"), OSError("connection refused"), reply())
        out = self.local(t).send(self.request(), LOCAL_MODEL)
        self.assertEqual(len(t.calls), 3)
        self.assertEqual(self.sleeps, [0.5, 1.0])
        self.assertEqual(out["input_tokens"], 7)

    def test_local_retries_timeouts_429_and_5xx(self):
        for step in (TimeoutError("slow"), status(429), status(500), status(503)):
            self.sleeps.clear()
            t = FakeTransport(step, reply())
            self.local(t).send(self.request(), LOCAL_MODEL)
            self.assertEqual((len(t.calls), self.sleeps), (2, [0.5]), repr(step))

    def test_local_backoff_capped_and_unreachable_is_unavailable(self):
        t = FakeTransport(ConnectionRefusedError("refused"), repeat=True)
        with self.assertRaises(BackendUnavailable):
            self.local(t).send(self.request(), LOCAL_MODEL)
        self.assertEqual(len(t.calls), 6)
        self.assertEqual(self.sleeps, [0.5, 1.0, 2.0, 4.0, 8.0])
        self.sleeps.clear()
        t = FakeTransport(ConnectionRefusedError("refused"), repeat=True)
        with self.assertRaises(BackendUnavailable):
            self.local(t, max_tries=9).send(self.request(), LOCAL_MODEL)
        self.assertEqual(self.sleeps, [0.5, 1.0, 2.0, 4.0, 8.0, 10, 10, 10])

    def test_local_server_error_after_retries_is_a_backend_error(self):
        t = FakeTransport(status(503, body=b"busy"), repeat=True)
        with self.assertRaises(BackendError) as cm:
            self.local(t).send(self.request(), LOCAL_MODEL)
        self.assertEqual((len(t.calls), cm.exception.status), (6, 503))
        self.assertNotIsInstance(cm.exception, BackendUnavailable)

    def test_local_malformed_response_is_an_error(self):
        t = FakeTransport(status(200, body=b"nope"))
        with self.assertRaises(BackendError):
            self.local(t).send(self.request(), LOCAL_MODEL)
        self.assertEqual(len(t.calls), 1)

    def test_local_single_flight(self):
        t = FakeTransport(reply(), reply(), reply(), reply(), delay=0.02)
        backends = [self.local(t), self.local(t)]
        results, errors = [], []

        def work(backend):
            try:
                results.append(backend.send(self.request(), LOCAL_MODEL))
            except Exception as e:  # collected, asserted below
                errors.append(e)

        threads = [threading.Thread(target=work, args=(backends[i % 2],)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(10)
        self.assertEqual((errors, len(results), len(t.calls)), ([], 4, 4))
        self.assertEqual(t.peak, 1)

    def test_local_slot_is_free_after_a_failed_send(self):
        # a failed send must release the in-flight slot
        t = FakeTransport(status(400), reply())
        b = self.local(t)
        with self.assertRaises(BackendError):
            b.send(self.request(), LOCAL_MODEL)
        self.assertEqual(b.send(self.request(), LOCAL_MODEL)["input_tokens"], 7)

    def test_local_url_from_env(self):
        with mock.patch.dict(os.environ, {"JEVM_DECIDER_URL": "http://10.0.0.5:9000/"}):
            t = FakeTransport(reply(), status(200, body=b'{"status": "ready", "images": true}'))
            b = self.local(t)
            b.send(self.request(), LOCAL_MODEL)
            b.health()
            explicit = self.local(FakeTransport(), url="http://example.test:1")
        self.assertEqual([c["url"] for c in t.calls],
                         ["http://10.0.0.5:9000/v1/decisions", "http://10.0.0.5:9000/health"])
        self.assertEqual(explicit.url, "http://example.test:1")

    def test_local_default_url(self):
        t = FakeTransport(reply())
        self.local(t).send(self.request(), LOCAL_MODEL)
        self.assertEqual(t.calls[0]["url"], "http://127.0.0.1:8080/v1/decisions")


# ---------------------------------------------------------------- both
class OfflineTests(BackendCase):
    def test_offline_env_blocks_both(self):
        with mock.patch.dict(os.environ, {"JEVM_OFFLINE": "1"}):
            t = FakeTransport(reply(), reply())
            for backend in (self.jev(t), self.local(t)):
                with self.assertRaises(BackendUnavailable):
                    backend.send(self.request(), JEV_MODEL)
                with self.assertRaises(BackendUnavailable):
                    backend.health()
            self.assertEqual(t.calls, [])

    def test_offline_zero_is_not_offline(self):
        with mock.patch.dict(os.environ, {"JEVM_OFFLINE": "0"}):
            t = FakeTransport(reply())
            self.assertEqual(self.jev(t).send(self.request(), JEV_MODEL)["input_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
