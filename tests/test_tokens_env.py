"""Tests for jevlib.tokens, jevlib.env, jevlib.paths and the FakeBackend."""
import os
import tempfile
import unittest
from pathlib import Path

from jevlib import fake, paths, tokens
from jevlib.backends.base import (IMAGES_HEADER, Backend, BackendError, BackendUnavailable, LimitError)
from jevlib.env import load_key, offline, redact
from jevlib.request import Request, image_ref, load_qset

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "qsets"


class TokenTests(unittest.TestCase):
    def test_image_tokens_882x1568(self):
        self.assertEqual(tokens.image_tokens(882, 1568), 3982)

    def test_text_tokens(self):
        self.assertEqual(tokens.text_tokens(10), 3)
        self.assertEqual(tokens.text_tokens(8), 2)
        self.assertEqual(tokens.text_tokens(0), 0)

    def test_request_tokens_adds_text_and_images(self):
        self.assertEqual(tokens.request_tokens(400, [(882, 1568), (882, 1568)]), 100 + 2 * 3982)
        self.assertEqual(tokens.request_tokens(400, []), 100)

    def test_usd(self):
        self.assertAlmostEqual(tokens.usd(1_000_000), 0.042, places=9)
        self.assertEqual(tokens.usd(0), 0.0)

    def test_fmt_usd(self):
        self.assertEqual(tokens.fmt_usd(0), "$0.000")
        self.assertEqual(tokens.fmt_usd(0.0), "$0.000")
        self.assertEqual(tokens.fmt_usd(1e-9), "<$0.001")
        self.assertEqual(tokens.fmt_usd(0.000999), "<$0.001")
        self.assertEqual(tokens.fmt_usd(0.001), "$0.001")
        self.assertEqual(tokens.fmt_usd(0.1), "$0.100")
        self.assertEqual(tokens.fmt_usd(0.0424), "$0.042")
        self.assertEqual(tokens.fmt_usd(12), "$12.000")
        self.assertEqual(tokens.JEV_USD_PER_M_INPUT, 0.042)

    def test_constants(self):
        self.assertEqual((tokens.QUESTION_BUDGET, tokens.QUESTION_CAP, tokens.REQUEST_CAP),
                         (26_000, 32_000, 64_000))

    def test_local_seconds(self):
        self.assertAlmostEqual(tokens.local_seconds(5, 1), 0.82 + 4 * 0.446, places=3)
        self.assertAlmostEqual(tokens.local_seconds(8, 0), 0.512, places=3)
        self.assertAlmostEqual(tokens.local_seconds(1, 1), 0.82, places=3)
        self.assertEqual(tokens.local_seconds(0, 0), 0.0)

    def test_chunk_respects_budget(self):
        chunks = tokens.chunk([10_000] * 5, lambda c: c, budget=26_000)
        self.assertEqual([len(c) for c in chunks], [2, 2, 1])

    def test_chunk_counts_overhead_per_chunk(self):
        chunks = tokens.chunk([10_000] * 4, lambda c: c, budget=26_000, overhead=7_000)
        self.assertEqual([len(c) for c in chunks], [1, 1, 1, 1])

    def test_chunk_keeps_order_and_never_drops_an_oversize_item(self):
        chunks = tokens.chunk(["a", "b", "c"], {"a": 5_000, "b": 30_000, "c": 5_000}.get)
        self.assertEqual(chunks, [["a"], ["b"], ["c"]])
        self.assertEqual(tokens.chunk([], lambda c: c), [])


class EnvTests(unittest.TestCase):
    def test_load_key_environ_wins(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".env").write_text("JEV_API=from-file\n", encoding="utf-8")
            self.assertEqual(load_key(start=Path(d), environ={"JEV_API": " from-env "}), "from-env")

    def test_load_key_dotenv_export_and_quotes(self):
        cases = {"JEV_API=plain": "plain", "export JEV_API=exported": "exported",
                 'JEV_API="double"': "double", "JEV_API='single'": "single",
                 "export JEV_API = \"spaced\"": "spaced"}
        for line, want in cases.items():
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / ".env").write_text("# note\nOTHER=1\n" + line + "\n", encoding="utf-8")
                self.assertEqual(load_key(start=Path(d), environ={}), want, line)

    def test_load_key_reads_only_the_one_folder(self):
        """`.env` in `start` only: a parent folder's `.env` is never read."""
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".env").write_text("JEV_API=up\n", encoding="utf-8")
            deep = Path(d) / "a b" / "c"
            deep.mkdir(parents=True)
            self.assertEqual(load_key(start=deep, environ={}), "")
            self.assertEqual(load_key(start=Path(d), environ={}), "up")

    def test_load_key_defaults_to_the_current_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".env").write_text("JEV_API=here\n", encoding="utf-8")
            old = Path.cwd()
            os.chdir(d)
            try:
                self.assertEqual(load_key(environ={}), "here")
            finally:
                os.chdir(old)

    def test_load_key_absent_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / ".env").write_text("OTHER=1\n", encoding="utf-8")
            self.assertEqual(load_key("JEVM_TEST_NO_SUCH_KEY", start=Path(d), environ={}), "")

    def test_redact(self):
        self.assertEqual(redact("x KEY y", ["KEY"]), "x REDACTED y")
        self.assertEqual(redact("x KEY y KEY", ["KEY", ""]), "x REDACTED y REDACTED")
        self.assertEqual(redact("untouched", ["", None]), "untouched")

    def test_tests_run_offline(self):
        self.assertEqual(os.environ["JEVM_OFFLINE"], "1")
        self.assertTrue(offline())


class PathsTests(unittest.TestCase):
    def test_tool_layout(self):
        self.assertTrue((paths.TOOL_ROOT / "jevm.py").is_file())
        self.assertFalse(hasattr(paths, "WORKSPACE"))
        self.assertEqual(paths.QSETS_DIR, paths.TOOL_ROOT / "qsets")
        self.assertEqual(paths.EVAL_OUT_DIR, paths.TOOL_ROOT / "eval")

    def test_project_root_and_work_dir(self):
        rel = {"project": {"root": "videos/some video"}}
        self.assertEqual(paths.project_root(rel), Path.cwd() / "videos" / "some video")
        self.assertEqual(paths.work_dir(rel), Path.cwd() / "videos" / "some video" / "work" / "jevm")
        absolute = {"project": {"root": "/tmp/p q"}}
        self.assertEqual(paths.project_root(absolute), Path("/tmp/p q"))
        self.assertEqual(paths.work_dir(absolute), Path("/tmp/p q/work/jevm"))

    def test_cache_dir(self):
        spec = {"project": {"root": "/tmp/p q"}}
        self.assertEqual(paths.cache_dir(spec, environ={}), Path("/tmp/p q/work/jevm/.jevm-cache"))
        self.assertEqual(paths.cache_dir(None, environ={}), Path.cwd() / ".jevm-cache")
        for given in (spec, None):
            self.assertEqual(paths.cache_dir(given, environ={"JEVM_CACHE_DIR": "/tmp/c d"}), Path("/tmp/c d"))
        self.assertEqual(paths.cache_dir(spec, environ={"JEVM_CACHE_DIR": "  "}), Path("/tmp/p q/work/jevm/.jevm-cache"))


def request(questions, images=None):
    return Request(load_qset("sample", root=FIXTURES), {}, questions, images=images or {})


class FakeBackendTests(unittest.TestCase):
    def test_fake_by_qid_answers_only_requested_qids(self):
        table = {"a": fake.noul(0.9), "b": fake.noul(0.1), "unused": fake.noul(0.5)}
        send = fake.by_qid(table, tokens=77)
        out = send(request({"b": {"type": "noul"}, "a": {"type": "noul"}}), "jev-1.13.0")
        self.assertEqual(list(out["answers"]), ["b", "a"])
        self.assertEqual(out["answers"]["a"], {"noul": 0.9})
        self.assertEqual((out["input_tokens"], out["model"]), (77, "jev-1.13.0"))
        self.assertIn("latency_s", out)
        with self.assertRaises(KeyError):
            send(request({"missing": {"type": "noul"}}), "jev-1.13.0")

    def test_builders_return_system_one_shapes(self):
        self.assertEqual(fake.noul(0.3), {"noul": 0.3})
        c = fake.choice({"a": 0.2, "b": 0.8})
        self.assertEqual((c["choice"], c["probabilities"]), ("b", {"a": 0.2, "b": 0.8}))
        self.assertAlmostEqual(c["confidence"], 0.8)
        self.assertEqual(fake.box(0.1, 0.2, 0.9, 0.8),
                         {"box": {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8}, "confidence": 0.6})
        self.assertEqual(fake.box(0, 0, 1, 1, confidence=0.9)["confidence"], 0.9)

    def test_send_records_calls_and_returns_the_responder_dict(self):
        be = fake.FakeBackend("jev", fake.by_qid({"a": fake.noul(0.5)}), version="")
        req = request({"a": {"type": "noul"}})
        out = be.send(req, "jev-1.13.0")
        self.assertEqual(be.calls, [(req, "jev-1.13.0")])
        self.assertEqual(out["answers"], {"a": {"noul": 0.5}})
        self.assertEqual(be.name, "jev")
        self.assertIsInstance(be, Backend)

    def test_jev_header_rule_images_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            p.write_bytes(b"png")
            ref = image_ref(p)
            with_img = Request(load_qset("sample", root=FIXTURES), {"image": ref}, {}, images={ref: p})
            no_img = request({"a": {"type": "noul"}})
            jev = fake.FakeBackend("jev", fake.by_qid({}))
            local = fake.FakeBackend("local", fake.by_qid({}), version="0.1.0")
            self.assertEqual(jev.header_flags(with_img), [IMAGES_HEADER])
            self.assertEqual(jev.header_flags(no_img), [])
            self.assertEqual(local.header_flags(with_img), [])
            self.assertEqual(local.version(), "0.1.0")
            self.assertEqual(jev.version(), "")

    def test_health_and_raises(self):
        up = fake.FakeBackend("local", fake.by_qid({}))
        self.assertEqual(up.health(), {"status": "ready", "images": True})
        down = fake.FakeBackend("local", fake.by_qid({}), healthy=False)
        with self.assertRaises(BackendUnavailable):
            down.health()
        with self.assertRaises(BackendUnavailable):
            down.send(request({}), "m")
        boom = fake.FakeBackend("jev", fake.by_qid({}), raises=BackendError("HTTP 400", status=400))
        with self.assertRaises(BackendError) as cm:
            boom.send(request({}), "m")
        self.assertEqual(cm.exception.status, 400)
        self.assertEqual(len(boom.calls), 1)

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(BackendUnavailable, RuntimeError))
        self.assertTrue(issubclass(BackendError, RuntimeError))
        self.assertTrue(issubclass(LimitError, BackendError))
        self.assertIsNone(BackendError("x").status)


if __name__ == "__main__":
    unittest.main()
