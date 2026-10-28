"""Tests for core.decisions_io: the decisions.json / decisions.md writer and `jevm resolve`. From the tool folder:

    python3 -m unittest tests.test_decisions_io -q

Offline and pure: rows come from `make_row`, files live in temporary folders (one with a space in its name).
"""
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

from core import decisions_io as dio
from core.policy import load_policy
from core.rows import code_evidence, make_row

NOW = datetime(2026, 10, 24, 9, 30, tzinfo=timezone.utc)
LATER = datetime(2026, 10, 24, 10, 45, 5, tzinfo=timezone.utc)
POLICY = load_policy()


def _row(decision="CAP-02", target_id="cg1", module="universal.captions", value=None, lane="auto", reasons=("above_auto",),
         p=0.8, alternatives=None, evidence=None, t=(1.0, 2.0), words=("w1", "w3")):
    entry = {"evidence": {"label": "Default", "ref": "docs/question-sets.md"}}
    return make_row(decision=decision, module=module,
                    target={"kind": "caption_group", "id": target_id, "t": list(t), "words": list(words)},
                    value=value if value is not None else {"span": ["w1", "w2"], "text": "one two"}, p=p,
                    confidence={"metric": "noul_p", "value": p}, lane=lane, route_reasons=list(reasons),
                    evidence=evidence or code_evidence(entry), alternatives=alternatives)


def _doc(*rows):
    doc = dio.new_doc(project_id="2026-10-24-test", spec_sha256="ab" * 32, kit_id="example", manifest_sha256="cd" * 32, now=NOW)
    doc["decisions"] = [copy.deepcopy(r) for r in rows]
    return doc


def _ids(doc):
    return [r["id"] for r in doc["decisions"]]


def _by_id(doc):
    return {r["id"]: r for r in doc["decisions"]}


def _quiet(fn, *args, **kwargs):
    """Call `fn` with stdout and stderr captured; returns (result, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        result = fn(*args, **kwargs)
    return result, out.getvalue(), err.getvalue()


def _section(md, heading):
    """The text of the `## heading` section of a decisions.md (up to the next `## `)."""
    for part in md.split("\n## "):
        if part.startswith(heading):
            return part
    raise AssertionError(f"no section {heading!r} in:\n{md}")


class _TempDirCase(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory(prefix="jevm decisions ")
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)


class NewDocAndLoadTests(_TempDirCase):
    def test_new_doc_shape(self):
        doc = dio.new_doc(project_id="proj-a", spec_sha256="s" * 64, kit_id="example", manifest_sha256="m" * 64, now=NOW)
        self.assertEqual(list(doc), ["contract_version", "project_id", "spec_sha256", "kit", "created_at", "backends_used",
                                     "decisions"])
        self.assertEqual(doc["contract_version"], dio.CONTRACT_VERSION)
        self.assertEqual(dio.CONTRACT_VERSION, "1")
        self.assertEqual(doc["project_id"], "proj-a")
        self.assertEqual(doc["spec_sha256"], "s" * 64)
        self.assertEqual(doc["kit"], {"id": "example", "manifest_sha256": "m" * 64})
        self.assertEqual(doc["created_at"], "2026-10-24T09:30:00Z")
        self.assertEqual(doc["decisions"], [])
        self.assertEqual(set(doc["backends_used"]), {"jev", "local"})
        self.assertEqual(doc["backends_used"]["jev"]["requests"], 0)
        self.assertEqual(doc["backends_used"]["local"]["serial_s"], 0)

    def test_new_doc_default_now_is_a_utc_stamp(self):
        doc = dio.new_doc(project_id="p", spec_sha256="", kit_id="k", manifest_sha256="")
        self.assertRegex(doc["created_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    def test_load_missing_file_is_none(self):
        self.assertIsNone(dio.load(self.dir / "nope.json"))

    def test_load_round_trips_a_written_doc(self):
        doc = _doc(_row())
        json_path, _ = dio.write(doc, POLICY, self.dir)
        self.assertEqual(dio.load(json_path), doc)

    def test_load_corrupt_file_raises_value_error(self):
        bad = self.dir / "decisions.json"
        bad.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            dio.load(bad)
        bad.write_text(json.dumps({"no": "decisions"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            dio.load(bad)


class MergeTests(unittest.TestCase):
    def test_merge_replaces_module_rows_only(self):
        doc = _doc(_row("CAP-02", "cg1", "universal.captions"), _row("CAP-02", "cg2", "universal.captions"),
                   _row("AST-01", "b1", "universal.assets", value={"box": [0, 0, 1, 1]}))
        fresh = [_row("CAP-02", "cg2", "universal.captions", value={"span": ["w9", "w9"]}),
                 _row("CAP-02", "cg3", "universal.captions")]
        out = dio.merge(doc, fresh, modules={"universal.captions"})
        self.assertEqual(sorted(_ids(out)), ["AST-01:b1", "CAP-02:cg2", "CAP-02:cg3"])
        self.assertNotIn("CAP-02:cg1", _ids(out))                    # a module row no longer produced is dropped
        self.assertEqual(_by_id(out)["CAP-02:cg2"]["value"], {"span": ["w9", "w9"]})
        self.assertEqual(_by_id(out)["AST-01:b1"]["value"], {"box": [0, 0, 1, 1]})   # other module untouched

    def test_merge_keeps_row_position(self):
        doc = _doc(_row("CAP-02", "cg1"), _row("AST-01", "b1", "universal.assets"), _row("CAP-02", "cg2"))
        out = dio.merge(doc, [_row("CAP-02", "cg1"), _row("CAP-02", "cg2"), _row("CAP-02", "cg3")],
                        modules={"universal.captions"})
        self.assertEqual(_ids(out), ["CAP-02:cg1", "AST-01:b1", "CAP-02:cg2", "CAP-02:cg3"])

    def test_merge_keeps_resolution_when_value_unchanged(self):
        doc = _doc(_row("CAP-02", "cg1", lane="agent", reasons=("in_review_band",)))
        resolved = {"value": {"span": ["w1", "w2"], "text": "one two"}, "by": "agent", "at": "2026-10-24T10:00:00Z",
                    "changed": False, "reviewed": True, "note": "ok"}
        doc["decisions"][0]["status"] = "accepted"
        doc["decisions"][0]["resolved"] = resolved
        fresh = _row("CAP-02", "cg1", lane="agent", reasons=("in_review_band",), p=0.79)   # same value, p moved a little
        out = dio.merge(doc, [fresh], modules={"universal.captions"})
        row = _by_id(out)["CAP-02:cg1"]
        self.assertEqual(row["status"], "accepted")
        self.assertEqual(row["resolved"], resolved)
        self.assertEqual(row["p"], 0.79)                              # the proposal fields are the fresh ones

    def test_merge_resets_when_value_changed(self):
        doc = _doc(_row("CAP-02", "cg1", lane="agent", reasons=("in_review_band",)))
        doc["decisions"][0]["status"] = "overridden"
        doc["decisions"][0]["resolved"] = {"value": {"span": ["w2", "w2"]}, "by": "agent", "at": "x", "changed": True,
                                           "reviewed": True, "note": ""}
        fresh = _row("CAP-02", "cg1", lane="agent", reasons=("in_review_band",), value={"span": ["w5", "w6"]})
        out = dio.merge(doc, [fresh], modules={"universal.captions"})
        row = _by_id(out)["CAP-02:cg1"]
        self.assertEqual(row["status"], "proposed")
        self.assertIsNone(row["resolved"])

    def test_merge_compares_values_as_json(self):
        """A fresh row may hold tuples where the file holds lists; that is not a changed value."""
        doc = _doc(_row("CAP-02", "cg1", value={"span": ["w1", "w2"]}))
        doc["decisions"][0]["status"] = "accepted"
        doc["decisions"][0]["resolved"] = {"value": {"span": ["w1", "w2"]}, "by": "agent", "at": "x", "changed": False,
                                           "reviewed": True, "note": ""}
        fresh = _row("CAP-02", "cg1", value={"span": ("w1", "w2")})
        out = dio.merge(doc, [fresh], modules={"universal.captions"})
        self.assertEqual(_by_id(out)["CAP-02:cg1"]["status"], "accepted")
        json.dumps(out)                                               # and everything stored is plain JSON

    def test_merge_by_ids_keeps_other_rows_of_same_module(self):
        doc = _doc(_row("AST-01", "b1", "universal.assets", value={"box": [0, 0, 0.5, 0.5]}),
                   _row("AST-01", "b2", "universal.assets", value={"box": [0, 0, 0.6, 0.6]}))
        doc["decisions"][1]["status"] = "accepted"
        doc["decisions"][1]["resolved"] = {"value": {"box": [0, 0, 0.6, 0.6]}, "by": "agent", "at": "x", "changed": False,
                                           "reviewed": True, "note": ""}
        fresh = _row("AST-01", "b1", "universal.assets", value={"box": [0.1, 0.1, 0.5, 0.5]})
        out = dio.merge(doc, [fresh], ids={"AST-01:b1"})
        self.assertEqual(_ids(out), ["AST-01:b1", "AST-01:b2"])
        self.assertEqual(_by_id(out)["AST-01:b1"]["value"], {"box": [0.1, 0.1, 0.5, 0.5]})
        self.assertEqual(_by_id(out)["AST-01:b2"]["status"], "accepted")      # untouched, resolution and all

    def test_merge_by_ids_drops_an_id_that_is_no_longer_produced(self):
        doc = _doc(_row("AST-01", "b1", "universal.assets"), _row("AST-01", "b2", "universal.assets"))
        out = dio.merge(doc, [], ids={"AST-01:b1"})
        self.assertEqual(_ids(out), ["AST-01:b2"])

    def test_merge_an_incoming_id_always_replaces_the_same_id(self):
        doc = _doc(_row("CAP-02", "cg1"))
        out = dio.merge(doc, [_row("CAP-02", "cg1", value={"span": ["w7", "w7"]})])    # neither modules nor ids named
        self.assertEqual(_ids(out), ["CAP-02:cg1"])                                      # no duplicate id
        self.assertEqual(out["decisions"][0]["value"], {"span": ["w7", "w7"]})

    def test_merge_duplicate_incoming_ids_raise(self):
        with self.assertRaises(ValueError):
            dio.merge(_doc(), [_row("CAP-02", "cg1"), _row("CAP-02", "cg1")], modules={"universal.captions"})

    def test_merge_does_not_alter_the_rows_it_was_given(self):
        doc = _doc(_row("CAP-02", "cg1"))
        doc["decisions"][0]["status"] = "accepted"
        doc["decisions"][0]["resolved"] = {"value": {"span": ["w1", "w2"], "text": "one two"}, "by": "agent", "at": "x",
                                           "changed": False, "reviewed": True, "note": ""}
        fresh = _row("CAP-02", "cg1")
        before = copy.deepcopy(fresh)
        dio.merge(doc, [fresh], modules={"universal.captions"})
        self.assertEqual(fresh, before)


class SetBackendsTests(unittest.TestCase):
    LEDGER = {"jev": {"models": {"text": "jev-1.13.0", "image": "jev-1.14.0"}, "requests": 10, "input_tokens": 1000,
                      "usd": 0.5, "cache_hits": 2},
              "local": {"model": "pplx-decider-flash-v1.2", "app_version": "0.1.0", "requests": 4, "serial_s": 2.5,
                        "cache_hits": 1}}

    def test_set_backends_adds_counters(self):
        doc = _doc()
        dio.set_backends(doc, self.LEDGER)
        out = dio.set_backends(doc, self.LEDGER)
        self.assertIs(out, doc)
        jev, local = doc["backends_used"]["jev"], doc["backends_used"]["local"]
        self.assertEqual((jev["requests"], jev["input_tokens"], jev["usd"], jev["cache_hits"]), (20, 2000, 1.0, 4))
        self.assertEqual(jev["models"], {"text": "jev-1.13.0", "image": "jev-1.14.0"})
        self.assertEqual((local["requests"], local["serial_s"], local["cache_hits"]), (8, 5.0, 2))
        self.assertEqual((local["model"], local["app_version"]), ("pplx-decider-flash-v1.2", "0.1.0"))

    def test_set_backends_unions_models_and_keeps_old_ones(self):
        doc = _doc()
        dio.set_backends(doc, self.LEDGER)
        dio.set_backends(doc, {"jev": {"models": {"text": "jev-1.14.0", "image": ""}, "requests": 1, "input_tokens": 5,
                                       "usd": 0.0, "cache_hits": 0}, "local": {}})
        self.assertEqual(doc["backends_used"]["jev"]["models"]["text"], "jev-1.13.0, jev-1.14.0")
        self.assertEqual(doc["backends_used"]["jev"]["models"]["image"], "jev-1.14.0")
        self.assertEqual(doc["backends_used"]["local"]["requests"], 4)

    def test_set_backends_on_a_doc_without_the_block(self):
        doc = {"decisions": []}
        dio.set_backends(doc, self.LEDGER)
        self.assertEqual(doc["backends_used"]["jev"]["requests"], 10)


class RenderTests(_TempDirCase):
    def test_md_lists_agent_rows_and_human_block(self):
        agent = _row("CAP-02", "cg5", lane="agent", reasons=("in_review_band",), p=0.55, t=(12.44, 13.4),
                      words=("w112", "w118"), alternatives=[{"value": {"span": ["w114", "w116"]}, "p": 0.11}])
        human = _row("QA-05", "claim3", "universal.claims", value={"quote": "100x faster"}, lane="human",
                     reasons=("never_auto",), p=0.2, t=(20.0, 22.5), words=("w200", "w204"))
        auto = [_row("CAP-03", f"cg{i}", lane="auto") for i in range(7)]
        md = dio.render_md(_doc(agent, human, *auto), POLICY)
        rows = [ln for ln in md.splitlines() if ln.startswith("|") and "CAP-02:cg5" in ln]
        self.assertEqual(len(rows), 1)                                # the agent-lane row sits in a table
        self.assertIn("in_review_band", rows[0])
        self.assertIn("12.44", rows[0])
        self.assertIn("w112", rows[0])
        self.assertIn("0.11", rows[0])                                # a top-3 entry with its p
        block = _section(md, "Questions for the human reviewer")
        q = [ln for ln in block.splitlines() if "QA-05:claim3" in ln]
        self.assertEqual(len(q), 1)
        self.assertIn("?", q[0])                                      # phrased as a question
        self.assertNotIn("|", q[0])                                   # a block, not a table row
        self.assertNotIn("QA-05:claim3", "\n".join(ln for ln in md.splitlines() if ln.startswith("|")))
        summary = [ln for ln in md.splitlines() if ln.lstrip("- ").startswith("CAP-03")]
        self.assertEqual(len(summary), 1)                             # auto rows: one line per decision type
        self.assertIn("7", summary[0])
        self.assertNotIn("CAP-03:cg0", md)

    def test_md_no_human_block_without_human_rows(self):
        md = dio.render_md(_doc(_row("CAP-03", "cg1")), POLICY)
        self.assertNotIn("Questions for the human reviewer", md)

    def test_md_header_lines_per_backend(self):
        doc = _doc(_row())
        dio.set_backends(doc, SetBackendsTests.LEDGER)
        md = dio.render_md(doc, POLICY)
        self.assertIn("jev-1.13.0", md)
        self.assertIn("jev-1.14.0", md)
        self.assertIn("pplx-decider-flash-v1.2", md)
        self.assertIn("0.1.0", md)                                    # the Decider app version
        self.assertIn("serial", md)
        self.assertIn("cache hit", md)

    def test_md_module_not_helping_counts_only_can_auto(self):
        caps = [_row("CAP-02", f"cg{i}", lane="agent" if i < 3 else "auto", reasons=("in_review_band",) if i < 3 else
                     ("above_auto",)) for i in range(4)]
        assets = [_row("AST-01", f"b{i}", "universal.assets", value={"box": [0, 0, 1, 1]}, lane="agent") for i in range(4)]
        md = dio.render_md(_doc(*caps, *assets), POLICY)
        self.assertIn("module_not_helping", _section(md, "universal.captions"))
        self.assertNotIn("module_not_helping", _section(md, "universal.assets"))

    def test_md_module_not_helping_is_more_than_the_policy_share(self):
        """2 of 5 is exactly 40%: not more than the limit. 3 of 5 is."""
        def md_for(n_agent):
            rows = [_row("CAP-02", f"cg{i}", lane="agent" if i < n_agent else "auto", reasons=("in_review_band",)
                         if i < n_agent else ("above_auto",)) for i in range(5)]
            return dio.render_md(_doc(*rows), POLICY)
        self.assertNotIn("module_not_helping", md_for(2))
        self.assertIn("module_not_helping", md_for(3))

    def test_md_module_not_helping_lists_the_modules_auto_rows_for_agent(self):
        rows = [_row("CAP-02", f"cg{i}", lane="agent" if i < 3 else "auto") for i in range(4)]
        md = dio.render_md(_doc(*rows), POLICY)
        table = "\n".join(ln for ln in md.splitlines() if ln.startswith("|"))
        self.assertIn("CAP-02:cg3", table)                            # an auto row, listed because the module is not helping

    def test_md_lists_backend_fallback_notice(self):
        fb = _row("CAP-02", "cg1", lane="agent", reasons=("backend_fallback", "in_review_band"))
        down = _row("CAP-03", "cg2", lane="agent", reasons=("jev_unavailable",))
        md = dio.render_md(_doc(fb, down), POLICY)
        notices = _section(md, "Notices")
        self.assertIn("backend_fallback", notices)
        self.assertIn("CAP-02:cg1", notices)
        self.assertIn("jev_unavailable", notices)
        self.assertIn("CAP-03:cg2", notices)

    def test_md_says_which_backend_answered_each_stage(self):
        class _Ans:
            request = type("R", (), {"qs": type("Q", (), {"id": "highlight", "version": 2})()})()
            backend, model, request_sha256, cached = "local", "pplx-decider-flash-v1.2", "f" * 64, False
        from core.rows import evidence_from
        local = _row("CAP-02", "cg1", evidence=evidence_from(_Ans, {"evidence": {"label": "Default", "ref": "r"}}))
        code = _row("CAP-02", "cg2")
        section = _section(dio.render_md(_doc(local, code), POLICY), "universal.captions")
        self.assertIn("local", section)
        self.assertIn("pplx-decider-flash-v1.2", section)
        self.assertIn("code", section)

    def test_md_resolved_rows_leave_the_open_table(self):
        open_row = _row("AST-01", "b1", "universal.assets", lane="agent", reasons=("in_review_band",))
        done = _row("AST-01", "b2", "universal.assets", lane="agent", reasons=("in_review_band",))
        done["status"] = "accepted"
        done["resolved"] = {"value": done["value"], "by": "agent", "at": "x", "changed": False, "reviewed": True, "note": ""}
        md = dio.render_md(_doc(open_row, done), POLICY)
        table = "\n".join(ln for ln in md.splitlines() if ln.startswith("|"))
        self.assertIn("AST-01:b1", table)
        self.assertNotIn("AST-01:b2", table)
        self.assertIn("AST-01:b2", md)                                # still named, as resolved
        self.assertIn("accepted", md)

    def test_md_escapes_table_pipes(self):
        row = _row("CAP-02", "cg1", lane="agent", value={"text": "a | b"}, reasons=("in_review_band",))
        line = [ln for ln in dio.render_md(_doc(row), POLICY).splitlines() if ln.startswith("|") and "CAP-02:cg1" in ln][0]
        self.assertNotIn("a | b", line)
        self.assertIn("a \\| b", line)

    def test_md_links_images_in_crops(self):
        for sub, name in (("crops", "shot 1.png"), ("crops", "b.webp"), ("other", "strip.jpg")):
            (self.dir / sub).mkdir(exist_ok=True)
            (self.dir / sub / name).write_bytes(b"x")
        (self.dir / "crops" / "notes.txt").write_text("not an image", encoding="utf-8")
        md = dio.render_md(_doc(_row()), POLICY, media_dir=self.dir)
        self.assertIn("(crops/shot%201.png)", md)
        self.assertIn("(crops/b.webp)", md)
        self.assertNotIn("strip.jpg", md)
        self.assertNotIn("notes.txt", md)
        self.assertNotIn("crops/", dio.render_md(_doc(_row()), POLICY))   # no media_dir: no image section

    def test_md_empty_doc(self):
        md = dio.render_md(dio.new_doc(project_id="p", spec_sha256="", kit_id="k", manifest_sha256="", now=NOW), POLICY)
        self.assertIn("No decisions", md)


class WriteTests(_TempDirCase):
    def test_write_creates_both_files_in_a_folder_with_a_space(self):
        out_dir = self.dir / "work and more" / "jevm"
        json_path, md_path = dio.write(_doc(_row(), _row("CAP-02", "cg2", lane="agent")), POLICY, out_dir)
        self.assertEqual((json_path.name, md_path.name), ("decisions.json", "decisions.md"))
        self.assertEqual((json_path.parent, md_path.parent), (out_dir, out_dir))
        self.assertTrue(json.loads(json_path.read_text(encoding="utf-8"))["decisions"])
        self.assertTrue(md_path.read_text(encoding="utf-8").startswith("# Decisions"))
        self.assertEqual(sorted(p.name for p in out_dir.iterdir()), ["decisions.json", "decisions.md"])   # no temp files left

    def test_write_overwrites_atomically(self):
        dio.write(_doc(_row()), POLICY, self.dir)
        dio.write(_doc(_row("CAP-02", "cg9")), POLICY, self.dir)
        self.assertEqual(_ids(dio.load(self.dir / "decisions.json")), ["CAP-02:cg9"])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["decisions.json", "decisions.md"])


class ResolveTests(_TempDirCase):
    def _write(self, *rows):
        json_path, md_path = dio.write(_doc(*rows), POLICY, self.dir)
        return json_path, md_path

    def test_resolve_accept_and_value(self):
        a = _row("CAP-02", "cg1", lane="agent", reasons=("in_review_band",))
        b = _row("CAP-02", "cg2", lane="agent", reasons=("in_review_band",))
        path, _ = self._write(a, b)
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="looks right", now=LATER)
        self.assertEqual((rc, err), (0, ""))
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg2"], accept=False, value='{"span": ["w1", "w2"]}', by="human",
                            note="", now=LATER)
        self.assertEqual((rc, err), (0, ""))
        rows = _by_id(dio.load(path))
        one, two = rows["CAP-02:cg1"], rows["CAP-02:cg2"]
        self.assertEqual(one["status"], "accepted")
        self.assertEqual(one["resolved"], {"value": a["value"], "by": "agent", "at": "2026-10-24T10:45:05Z",
                                           "changed": False, "reviewed": True, "note": "looks right"})
        self.assertEqual(two["status"], "overridden")
        self.assertEqual(two["resolved"], {"value": {"span": ["w1", "w2"]}, "by": "human", "at": "2026-10-24T10:45:05Z",
                                           "changed": True, "reviewed": True, "note": ""})
        self.assertEqual(two["value"], b["value"])                    # the proposal is kept beside the override

    def test_resolve_value_parses_json_else_string(self):
        path, _ = self._write(_row("CAP-02", "cg1"), _row("CAP-02", "cg2"), _row("CAP-02", "cg3"))
        for rid, raw in (("CAP-02:cg1", "0.25"), ("CAP-02:cg2", "plain words"), ("CAP-02:cg3", "NaN")):
            self.assertEqual(_quiet(dio.resolve, path, [rid], accept=False, value=raw, by="agent", note="")[0], 0)
        rows = _by_id(dio.load(path))
        self.assertEqual(rows["CAP-02:cg1"]["resolved"]["value"], 0.25)
        self.assertEqual(rows["CAP-02:cg2"]["resolved"]["value"], "plain words")
        self.assertEqual(rows["CAP-02:cg3"]["resolved"]["value"], "NaN")      # kept as text: the file stays valid JSON

    def test_resolve_several_ids_at_once(self):
        path, _ = self._write(_row("CAP-02", "cg1"), _row("CAP-02", "cg2"), _row("CAP-02", "cg3"))
        rc, _, _ = _quiet(dio.resolve, path, ["CAP-02:cg1", "CAP-02:cg3"], accept=True, value=None, by="agent", note="")
        self.assertEqual(rc, 0)
        rows = _by_id(dio.load(path))
        self.assertEqual([rows[i]["status"] for i in ("CAP-02:cg1", "CAP-02:cg2", "CAP-02:cg3")],
                         ["accepted", "proposed", "accepted"])

    def test_resolve_unknown_id_exit_2_file_untouched(self):
        path, md_path = self._write(_row("CAP-02", "cg1"))
        before_json, before_md = path.read_bytes(), md_path.read_bytes()
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1", "CAP-02:nope"], accept=True, value=None, by="agent", note="")
        self.assertEqual(rc, 2)
        self.assertIn("CAP-02:nope", err)
        self.assertEqual(path.read_bytes(), before_json)             # not even the known id was resolved
        self.assertEqual(md_path.read_bytes(), before_md)

    def test_resolve_bad_arguments_exit_2(self):
        path, _ = self._write(_row("CAP-02", "cg1"))
        before = path.read_bytes()
        cases = [dict(accept=False, value=None, by="agent"),          # neither
                 dict(accept=True, value="x", by="agent"),            # both
                 dict(accept=True, value=None, by="somebody")]         # unknown actor
        for kw in cases:
            rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], note="", **kw)
            self.assertEqual(rc, 2, kw)
            self.assertTrue(err.strip(), kw)
        rc, _, _ = _quiet(dio.resolve, path, [], accept=True, value=None, by="agent", note="")
        self.assertEqual(rc, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_resolve_missing_or_corrupt_file_exit_2(self):
        rc, _, err = _quiet(dio.resolve, self.dir / "none.json", ["x"], accept=True, value=None, by="agent", note="")
        self.assertEqual(rc, 2)
        self.assertTrue(err.strip())
        bad = self.dir / "decisions.json"
        bad.write_text("[1, 2", encoding="utf-8")
        rc, _, _ = _quiet(dio.resolve, bad, ["x"], accept=True, value=None, by="agent", note="")
        self.assertEqual(rc, 2)
        self.assertEqual(bad.read_text(encoding="utf-8"), "[1, 2")

    def test_resolve_rerenders_decisions_md(self):
        path, md_path = self._write(_row("AST-01", "b1", "universal.assets", lane="agent", reasons=("in_review_band",)))
        self.assertIn("| AST-01:b1", md_path.read_text(encoding="utf-8"))
        _quiet(dio.resolve, path, ["AST-01:b1"], accept=True, value=None, by="agent", note="")
        md = md_path.read_text(encoding="utf-8")
        self.assertNotIn("| AST-01:b1", md)                          # no longer open
        self.assertIn("accepted", md)

    def test_resolve_keeps_unknown_top_level_keys_and_leaves_no_temp_files(self):
        path = self.dir / "decisions.json"
        doc = _doc(_row("CAP-02", "cg1"))
        doc["extra"] = {"keep": True}
        path.write_text(json.dumps(doc), encoding="utf-8")
        _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
        self.assertEqual(dio.load(path)["extra"], {"keep": True})
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["decisions.json", "decisions.md"])

    def test_resolve_on_qa_json(self):
        qa = self.dir / "qa.json"
        flag = _row("QA-01", "f3", "universal.qa", value={"verdict": "flag"}, lane="agent", reasons=("in_review_band",))
        flag["target"] = {"kind": "frame", "id": "f3", "t": [4.0, 4.0]}
        qa.write_text(json.dumps({"frames": 12, "decisions": [flag]}), encoding="utf-8")
        rc, _, err = _quiet(dio.resolve, qa, ["QA-01:f3"], accept=False, value="real", by="agent", note="clipped caption")
        self.assertEqual((rc, err), (0, ""))
        doc = json.loads(qa.read_text(encoding="utf-8"))
        row = doc["decisions"][0]
        self.assertEqual(row["status"], "overridden")
        self.assertEqual(row["resolved"]["value"], "real")
        self.assertEqual(row["resolved"]["note"], "clipped caption")
        self.assertEqual(doc["frames"], 12)                           # the rest of qa.json is kept
        self.assertEqual([p.name for p in self.dir.iterdir()], ["qa.json"])   # no decisions.md beside qa.json

    def test_cli_resolve_takes_comma_separated_ids(self):
        path, _ = self._write(_row("CAP-02", "cg1"), _row("CAP-02", "cg2"), _row("CAP-02", "cg3"))
        rc, out, err = _quiet(dio.cli_resolve, str(path), "CAP-02:cg1, CAP-02:cg2,", True, None, "agent", "")
        self.assertEqual((rc, err), (0, ""))
        self.assertIn("2", out)
        rows = _by_id(dio.load(path))
        self.assertEqual([rows[i]["status"] for i in ("CAP-02:cg1", "CAP-02:cg2", "CAP-02:cg3")],
                         ["accepted", "accepted", "proposed"])
        rc, _, err = _quiet(dio.cli_resolve, str(path), "CAP-02:ghost", True, None, "agent", "")
        self.assertEqual(rc, 2)
        self.assertIn("CAP-02:ghost", err)
        rc, _, _ = _quiet(dio.cli_resolve, str(path), " , ", True, None, "agent", "")
        self.assertEqual(rc, 2)


class FilesystemErrorTests(_TempDirCase):
    """Any OSError becomes a one-line message and exit 2 at the CLI entry points, a ValueError from the library, never a
    traceback and never a line of file content."""
    LEAK = "do-not-print-me-12345"

    def _clean_exit_2(self, rc, err, *, mentions=()):
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1, err)
        self.assertNotIn("Traceback", err)
        self.assertNotIn(self.LEAK, err)
        for text in mentions:
            self.assertIn(text, err)

    def _lock(self, path):
        """chmod 000 `path` (restored on cleanup); skip the test where that does not stop this process (root)."""
        mode = path.stat().st_mode & 0o777
        self.addCleanup(os.chmod, path, mode)
        os.chmod(path, 0)
        if os.access(path, os.R_OK):
            self.skipTest("chmod 000 does not block this process (running as root?)")

    def _file(self, name="decisions.json", **extra):
        path = self.dir / name
        doc = _doc(_row("CAP-02", "cg1", lane="agent", reasons=("in_review_band",)))
        doc.update(extra)
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def test_resolve_parent_is_a_file_exits_2(self):
        blocker = self.dir / "blocker"
        blocker.write_text(self.LEAK, encoding="utf-8")
        path = blocker / "decisions.json"
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
        self._clean_exit_2(rc, err, mentions=(str(path),))
        self.assertEqual(blocker.read_text(encoding="utf-8"), self.LEAK)       # the file in the way is untouched
        rc, out, err = _quiet(dio.cli_resolve, str(path), "CAP-02:cg1", True, None, "agent", "")
        self._clean_exit_2(rc, err)
        self.assertEqual(out, "")                                              # no success line

    def test_resolve_unreadable_file_exits_2(self):
        path = self._file(marker=self.LEAK)
        before = path.read_bytes()
        self._lock(path)
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
        self._clean_exit_2(rc, err, mentions=("Permission denied",))
        os.chmod(path, 0o644)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse((self.dir / "decisions.md").exists())

    def test_resolve_path_is_a_folder_exits_2(self):
        folder = self.dir / "decisions.json"
        folder.mkdir()
        rc, _, err = _quiet(dio.resolve, folder, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
        self._clean_exit_2(rc, err)

    def test_resolve_path_with_a_null_byte_exits_2(self):
        rc, _, err = _quiet(dio.resolve, str(self.dir / "a\0b.json"), ["x"], accept=True, value=None, by="agent", note="")
        self._clean_exit_2(rc, err)

    def test_resolve_unwritable_folder_exits_2_and_leaves_the_file(self):
        path = self._file()
        before = path.read_bytes()
        os.chmod(self.dir, 0o555)
        self.addCleanup(os.chmod, self.dir, 0o755)
        if os.access(self.dir, os.W_OK):
            self.skipTest("a read-only folder does not block this process (running as root?)")
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
        self._clean_exit_2(rc, err, mentions=("cannot write",))
        self.assertEqual(path.read_bytes(), before)

    def test_resolve_md_not_writable_says_the_json_was_updated(self):
        path = self._file()
        (self.dir / "decisions.md").mkdir()                                    # a folder where the md should go
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
        self._clean_exit_2(rc, err, mentions=("decisions.json was updated but decisions.md was not",))
        self.assertEqual(dio.load(path)["decisions"][0]["status"], "accepted")

    def test_resolve_unreadable_image_folder_exits_2_before_writing(self):
        path = self._file()
        crops = self.dir / "crops"
        crops.mkdir()
        before = path.read_bytes()
        self._lock(crops)
        rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
        self._clean_exit_2(rc, err, mentions=("crops",))
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            dio.render_md(_doc(_row()), POLICY, media_dir=self.dir)

    def test_resolve_unreadable_policy_exits_2(self):
        path = self._file()
        for exc in (FileNotFoundError(2, "No such file or directory"), PermissionError(13, "Permission denied"),
                    json.JSONDecodeError(self.LEAK, self.LEAK, 0)):
            with mock.patch.object(dio, "load_policy", side_effect=exc):
                rc, _, err = _quiet(dio.resolve, path, ["CAP-02:cg1"], accept=True, value=None, by="agent", note="")
            self._clean_exit_2(rc, err, mentions=("policy",))

    def test_errors_never_echo_file_contents(self):
        path = self.dir / "decisions.json"
        for raw in (('{"decisions": [' + json.dumps(self.LEAK)).encode(),          # truncated JSON
                    json.dumps({"decisions": "x", "k": self.LEAK}).encode(),       # JSON of the wrong shape
                    b"\xff\xfe" + self.LEAK.encode()):                             # not UTF-8
            path.write_bytes(raw)
            rc, _, err = _quiet(dio.resolve, path, ["x"], accept=True, value=None, by="agent", note="")
            self._clean_exit_2(rc, err)
            with self.assertRaises(ValueError) as caught:
                dio.load(path)
            self.assertNotIn(self.LEAK, str(caught.exception))

    def test_load_filesystem_errors_raise_value_error(self):
        blocker = self.dir / "blocker"
        blocker.write_text("x", encoding="utf-8")
        folder = self.dir / "folder.json"
        folder.mkdir()
        for path in (blocker / "decisions.json", folder, str(self.dir / "a\0b")):
            with self.assertRaises(ValueError, msg=str(path)):
                dio.load(path)
        locked = self._file("locked.json")
        self._lock(locked)
        with self.assertRaises(ValueError) as caught:
            dio.load(locked)
        self.assertIn("Permission denied", str(caught.exception))

    def test_load_missing_file_in_an_existing_folder_is_still_none(self):
        self.assertIsNone(dio.load(self.dir / "nope.json"))
        self.assertIsNone(dio.load(self.dir / "no folder" / "nope.json"))

    def test_write_into_missing_parent_exits_2_or_creates(self):
        """Chosen behaviour: `write` creates the folder and its parents (the runner writes into `<project>/work/jevm`, which
        does not exist on the first plan). It never exits: a parent that cannot be made is a ValueError, below."""
        out_dir = self.dir / "work" / "jevm"
        json_path, md_path = dio.write(_doc(_row()), POLICY, out_dir)
        self.assertTrue(json_path.is_file() and md_path.is_file())
        self.assertEqual(json_path.parent, out_dir)

    def test_write_parent_is_a_file_raises_value_error_and_writes_nothing(self):
        blocker = self.dir / "blocker"
        blocker.write_text(self.LEAK, encoding="utf-8")
        for out_dir in (blocker, blocker / "jevm"):
            with self.assertRaises(ValueError) as caught:
                dio.write(_doc(_row()), POLICY, out_dir)
            self.assertNotIn("\n", str(caught.exception))
            self.assertNotIn(self.LEAK, str(caught.exception))
        self.assertEqual(blocker.read_text(encoding="utf-8"), self.LEAK)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["blocker"])     # no temp files left behind

    def test_write_unwritable_folder_raises_value_error(self):
        os.chmod(self.dir, 0o555)
        self.addCleanup(os.chmod, self.dir, 0o755)
        if os.access(self.dir, os.W_OK):
            self.skipTest("a read-only folder does not block this process (running as root?)")
        with self.assertRaises(ValueError) as caught:
            dio.write(_doc(_row()), POLICY, self.dir)
        self.assertIn("cannot write", str(caught.exception))

    def test_write_md_not_writable_says_the_json_was_written(self):
        (self.dir / "decisions.md").mkdir()
        with self.assertRaises(ValueError) as caught:
            dio.write(_doc(_row()), POLICY, self.dir)
        self.assertIn("decisions.json was updated but decisions.md was not", str(caught.exception))
        self.assertTrue(dio.load(self.dir / "decisions.json")["decisions"])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["decisions.json", "decisions.md"])

    def test_merge_does_no_io(self):
        """`merge` and `set_backends` are pure: nothing to catch, and they work with a path that does not exist."""
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("io")), \
                mock.patch.object(Path, "write_text", side_effect=AssertionError("io")), \
                mock.patch("os.replace", side_effect=AssertionError("io")):
            dio.merge(_doc(_row()), [_row()], modules={"universal.captions"})
            dio.set_backends(_doc(), SetBackendsTests.LEDGER)


class PlanResolvePlanTests(_TempDirCase):
    """Review focus 2: `plan`, then `resolve`, then `plan` again keeps every resolution whose value did not change."""

    def _plan(self, rows_by_module, *, existing):
        doc = existing or dio.new_doc(project_id="p", spec_sha256="s", kit_id="example", manifest_sha256="m", now=NOW)
        rows = [r for rs in rows_by_module.values() for r in rs]
        dio.merge(doc, rows, modules=set(rows_by_module))
        dio.write(doc, POLICY, self.dir)
        return doc

    def test_plan_resolve_plan_keeps_unchanged_resolutions(self):
        path = self.dir / "decisions.json"
        caps = [_row("CAP-02", f"cg{i}", lane="agent", reasons=("in_review_band",), value={"span": [f"w{i}", f"w{i + 1}"]})
                for i in range(3)]
        assets = [_row("AST-01", "b1", "universal.assets", value={"box": [0.1, 0.2, 0.3, 0.4]}, lane="agent")]
        self._plan({"universal.captions": caps, "universal.assets": assets}, existing=None)

        self.assertEqual(_quiet(dio.resolve, path, ["CAP-02:cg0", "AST-01:b1"], accept=True, value=None, by="agent",
                                note="fine")[0], 0)
        self.assertEqual(_quiet(dio.resolve, path, ["CAP-02:cg1"], accept=False, value='{"span": ["w9", "w9"]}',
                                by="agent", note="")[0], 0)

        # second plan: cg0 same, cg1 same, cg2 changed; the box now arrives as a tuple of the same floats
        caps2 = [_row("CAP-02", "cg0", lane="agent", reasons=("in_review_band",), value={"span": ["w0", "w1"]}),
                 _row("CAP-02", "cg1", lane="agent", reasons=("in_review_band",), value={"span": ["w1", "w2"]}),
                 _row("CAP-02", "cg2", lane="agent", reasons=("in_review_band",), value={"span": ["w5", "w6"]})]
        assets2 = [_row("AST-01", "b1", "universal.assets", value={"box": (0.1, 0.2, 0.3, 0.4)}, lane="agent")]
        self._plan({"universal.captions": caps2, "universal.assets": assets2}, existing=dio.load(path))

        rows = _by_id(dio.load(path))
        self.assertEqual(rows["CAP-02:cg0"]["status"], "accepted")
        self.assertEqual(rows["CAP-02:cg1"]["status"], "overridden")
        self.assertEqual(rows["CAP-02:cg1"]["resolved"]["value"], {"span": ["w9", "w9"]})
        self.assertEqual(rows["CAP-02:cg2"]["status"], "proposed")     # was never resolved
        self.assertEqual(rows["AST-01:b1"]["status"], "accepted")
        self.assertEqual(rows["AST-01:b1"]["resolved"]["note"], "fine")

        # third plan: cg0's value changes, so its resolution goes; the others stay
        caps3 = [_row("CAP-02", "cg0", lane="agent", reasons=("in_review_band",), value={"span": ["w0", "w2"]})] + caps2[1:]
        self._plan({"universal.captions": caps3}, existing=dio.load(path))
        rows = _by_id(dio.load(path))
        self.assertEqual(rows["CAP-02:cg0"]["status"], "proposed")
        self.assertIsNone(rows["CAP-02:cg0"]["resolved"])
        self.assertEqual(rows["CAP-02:cg1"]["status"], "overridden")
        self.assertEqual(rows["AST-01:b1"]["status"], "accepted")      # that module was not re-planned


if __name__ == "__main__":
    unittest.main()
