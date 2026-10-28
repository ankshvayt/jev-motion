"""Tests for jevlib.cache: the order-preserving cache key and the atomic on-disk record store."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from jevlib.cache import Cache, cache_key, file_sha256


def key(**over):
    args = dict(backend="jev", model="jev-1.13.0", backend_version="", header_flags=[],
                state={"s": "x"}, questions={"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
                image_shas=[])
    args.update(over)
    return cache_key(**args)


class CacheKeyTests(unittest.TestCase):
    def test_key_same_request_same_key(self):
        self.assertEqual(key(), key())

    def test_key_differs_on_reversed_options(self):
        fwd = {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}
        rev = {"q": {"type": "choice", "criteria": {"b": "B", "a": "A"}}}
        self.assertNotEqual(key(questions=fwd), key(questions=rev))

    def test_key_differs_on_backend_model_and_version(self):
        base = key()
        varied = {key(backend="local"), key(model="jev-1.14.0"), key(backend_version="0.1.0")}
        self.assertEqual(len(varied), 3)
        self.assertNotIn(base, varied)

    def test_key_differs_on_image_bytes(self):
        self.assertNotEqual(key(image_shas=["aa"]), key(image_shas=["ab"]))

    def test_key_ignores_header_flag_order_but_not_content(self):
        self.assertEqual(key(header_flags=["b", "a"]), key(header_flags=["a", "b"]))
        self.assertNotEqual(key(header_flags=["a"]), key())

    def test_key_matches_the_spec_formula_byte_for_byte(self):
        state, questions = {"s": "café"}, {"q": {"type": "noul", "instructions": "i"}}
        body = json.dumps({"state": state, "questions": questions}, sort_keys=False,
                          separators=(",", ":"), ensure_ascii=False)
        want = hashlib.sha256(b"jevm-cache-v2\n" + b"local\n" + b"m\n" + b"0.1.0\n" + b"h1,h2\n"
                              + body.encode() + b"\n" + b"aa,bb").hexdigest()
        got = cache_key("local", "m", "0.1.0", ["h2", "h1"], state, questions, ["aa", "bb"])
        self.assertEqual(got, want)

    def test_file_sha256(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a b.bin"
            p.write_bytes(b"hello")
            self.assertEqual(file_sha256(p), hashlib.sha256(b"hello").hexdigest())


class CacheStoreTests(unittest.TestCase):
    def test_put_get_roundtrip_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            c = Cache(Path(d))
            k = key()
            rec = {"answers": {"q": {"noul": 0.42}}, "input_tokens": 12, "note": "café"}
            self.assertIsNone(c.get(k))
            c.put(k, rec)
            self.assertEqual(c.get(k), rec)
            self.assertEqual(list(Path(d).rglob("*.tmp")), [])
            self.assertTrue((Path(d) / k[:2] / f"{k}.json").is_file())

    def test_put_refuses_headers(self):
        with tempfile.TemporaryDirectory() as d:
            c = Cache(Path(d))
            for bad in ({"headers": {"x": "y"}}, {"Authorization": "Bearer x"},
                        {"answers": {}, "meta": [{"Headers": 1}]}):
                with self.assertRaises(ValueError):
                    c.put(key(), bad)
            self.assertEqual(list(Path(d).rglob("*.json")), [])

    def test_get_treats_a_corrupt_record_as_a_miss(self):
        with tempfile.TemporaryDirectory() as d:
            c = Cache(Path(d))
            k = key()
            (Path(d) / k[:2]).mkdir()
            (Path(d) / k[:2] / f"{k}.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(c.get(k))


if __name__ == "__main__":
    unittest.main()
