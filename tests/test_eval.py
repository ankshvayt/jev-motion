"""The eval harness, `jevm eval` (stdlib unittest). From the tool folder:

    python3 -m unittest tests.test_eval -q

Offline: scripted answers (tests/fixtures/eval/*.json, written by evalsets/make_synthetic.py with the synthetic eval sets)
are replayed through `FakeBackend`s by `harness.offline_client`, keyed by what identifies each request (frame file name,
group texts, crop case and want, sentence), because qids repeat across batches. Every run uses a temporary `Cache` and a
temporary output folder; nothing touches the network or the tool's `eval/`.
"""
import ast
import copy
import datetime
import json
import re
import socket
import tempfile
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from evals import harness
from jevlib.paths import EVALSETS_DIR, TOOL_ROOT
from modules.universal import qa

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eval"
TODAY = "2026-10-24"
START = datetime.datetime(2026, 10, 24, 13, 5, 9)     # a run's start time: names carry 130509
QA = "qa_frame.short"
TABLE = "## jev against local: parity"     # the heading of the parity table
CANNED = harness.load_canned(FIXTURES)
N_QA, N_HIGHLIGHT, N_CROP, N_STAT = 20, 14, 12, 12   # requests per set (crop: presence, box, absent presence per case)


def quiet(fn, *args, **kwargs):
    """`fn(...)` with stdout and stderr captured; returns `(result, stdout, stderr)`."""
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        result = fn(*args, **kwargs)
    return result, out.getvalue(), err.getvalue()


class EvalCase(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.tmp = Path(self._td.name)
        self.out = self.tmp / "eval"
        self.ticks = 0
        clock = mock.patch.object(harness, "_now", self.tick)     # one second per run, so no two runs share a name
        clock.start()
        self.addCleanup(clock.stop)

    def tick(self):
        self.ticks += 1
        return START + datetime.timedelta(seconds=self.ticks - 1)

    def client(self, canned=None, *, force_backend="jev"):
        self.cache_n = getattr(self, "cache_n", 0) + 1
        return harness.offline_client(copy.deepcopy(canned or CANNED), self.tmp / f"cache{self.cache_n}",
                                      force_backend=force_backend, log=lambda *_: None)

    def run_eval(self, client=None, **kw):
        kw.setdefault("out_dir", self.out)
        kw.setdefault("today", TODAY)
        client = client or self.client()
        result = quiet(harness.run_eval, client=client, **kw)
        return result[0], result[1], client

    def paths(self, name="jev", offline=False):
        """The report JSONs of `name` written today, oldest first (the time in the name sorts them)."""
        return sorted(p for p in self.out.glob(f"{TODAY}-*-{name}-*.json") if ("-offline" in p.stem) == offline)

    def report(self, name="jev", offline=False):
        """The newest report of `name`."""
        return json.loads(self.paths(name, offline)[-1].read_text(encoding="utf-8"))

    def md(self, name="jev", offline=False):
        return self.paths(name, offline)[-1].with_suffix(".md").read_text(encoding="utf-8")

    def gates(self, report, key):
        return {g["name"]: g for g in report["results"][key]["gates"]}


class ReplayTests(EvalCase):
    def test_eval_qa_on_the_synthetic_frames(self):
        res = harness.eval_qa(self.client())
        m = res["metrics"]
        self.assertEqual((m["own_flagged"], m["n_pos"]), (10, 12))
        self.assertEqual((m["clean_any_flag"], m["n_neg"]), (2, 8))
        by_q = {q["question"]: q for q in m["per_question"]}
        self.assertEqual((by_q["overlay_on_controls"]["recall_flag"], by_q["overlay_on_controls"]["positives"]), (2, 2))
        self.assertEqual(by_q["empty_region"]["recall_flag"], 2)
        self.assertEqual((by_q["overlapping_panels"]["recall_flag"], by_q["overlapping_panels"]["recall_half"]), (1, 2))
        self.assertEqual(by_q["cut_off_edge"]["recall_half"], 1)
        self.assertEqual(by_q["hard_to_read"]["positives"], 0)
        self.assertEqual(set(by_q), set(qa.DEFECTS))
        self.assertEqual(res["errors"], {})
        self.assertTrue(all(g["ok"] is True for g in res["gates"]), res["gates"])

    def test_eval_qa_gates_the_caption_on_face_question(self):
        """Recall for `caption_on_face` counts only where both gates are up, exactly as the QA module flags it."""
        canned = copy.deepcopy(CANNED)
        face = next(f for f in canned["qa"]["frames"] if f.startswith("caption_on_face_01"))
        canned["qa"]["frames"][face]["face_visible"] = 0.1
        by_q = {q["question"]: q for q in harness.eval_qa(self.client(canned))["metrics"]["per_question"]}
        self.assertEqual((by_q["caption_on_face"]["recall_flag"], by_q["caption_on_face"]["recall_half"]), (1, 1))

    def test_eval_highlight_scores_every_set(self):
        res = harness.eval_highlight(self.client())
        a, b = res["metrics"]["sets"]
        self.assertEqual((a["name"], a["groups"], a["k"], a["tp"], a["hand"]), ("set-a", 12, 3, 2, 4))
        self.assertAlmostEqual(a["precision"], 2 / 3)
        self.assertEqual((b["name"], b["groups"], b["hand"], b["hand_ranks"]), ("set-b", 16, 2, [1, 3]))
        self.assertEqual((a["span"]["exact"], a["span"]["n"]), (4, 4))
        self.assertEqual((b["span"]["exact"], b["span"]["n"]), (2, 2))
        self.assertEqual(a["order_agree"], {"agree": 12, "of": 12})   # the reverse request is served the forward answers
        self.assertEqual({g["name"] for g in res["gates"]}, {"set-a precision at K", "set-b hand ranks"})
        self.assertTrue(all(g["ok"] is True for g in res["gates"]), res["gates"])

    def test_eval_highlight_uses_the_frozen_hand_caption_groups(self):
        """Eval on the hand groups as frozen: the requests carry exactly the frozen group texts, in order, unchunked."""
        client = self.client()
        harness.eval_highlight(client)
        sent = [t for req, _model in client.backends["jev"].calls for t in [g["text"] for g in req.state["groups"]]]
        frozen = [g["text"] for name in harness.highlight_sets()
                  for g in json.loads((EVALSETS_DIR / "highlight" / f"{name}.json").read_text(encoding="utf-8"))["groups"]]
        self.assertEqual(harness.highlight_sets(), ["set-a", "set-b"])
        self.assertEqual(sorted(sent), sorted(frozen * 2))   # forward and reverse of every group

    def test_eval_crop(self):
        res = harness.eval_crop(self.client())
        m = res["metrics"]
        self.assertGreaterEqual(m["mean_iou"], 0.68)
        self.assertEqual((m["centre_inside"], m["n"]), (4, 4))
        self.assertAlmostEqual(m["presence_auc"], 1.0)
        for case in m["cases"]:
            self.assertIs(case["found"], True)
            self.assertIsInstance(case["clamped"], list)
        self.assertTrue(all(g["ok"] is True for g in res["gates"]), res["gates"])

    def test_eval_stat(self):
        res = harness.eval_stat(self.client())
        m = res["metrics"]
        self.assertEqual([(r["id"], r["in_band"]) for r in m["labelled"]], [("l1", True), ("l2", True)])
        self.assertEqual((m["opinion_below"], m["source_at"]), (0.3, 0.7))
        self.assertEqual((m["controls_correct"], m["controls"]), (10, 10))
        self.assertAlmostEqual(m["auc"], 1.0)
        self.assertTrue(all(g["ok"] is True for g in res["gates"]), res["gates"])

    def test_full_offline_run_exits_0_and_prints_the_summary(self):
        code, out, _client = self.run_eval()
        self.assertEqual(code, 0)
        self.assertIn("| set | metric | value | gate | result |", out)
        self.assertIn("own-question recall | 10/12 | >= 8/12 | pass", out)
        self.assertIn("all gates pass", out)

    def test_fixtures_hold_answers_only(self):
        for path in FIXTURES.glob("*_answers.json"):
            text = path.read_text(encoding="utf-8").lower()
            for word in ("authorization", "bearer", "headers", "jev_api"):
                self.assertNotIn(word, text, path.name)
            self.assertIn(json.loads(text).get("missing"), ([], 0), path.name)

    def test_every_eval_frame_is_distinct(self):
        """The cache keys on image bytes, so two identical frames would share one answer and blur the metrics."""
        frames = [p.read_bytes() for p in EVALSETS_DIR.rglob("*.png")]
        self.assertGreater(len(frames), 20)
        self.assertEqual(len(set(frames)), len(frames))


class GateTests(EvalCase):
    def clean_frames(self, canned, n):
        """The first `n` clean frames that nothing flags yet."""
        th = {"flag": 0.7, "check": 0.4, "gate": 0.5}
        names = [Path(i["file"]).name for i in json.loads(
            (EVALSETS_DIR / QA / "manifest.json").read_text(encoding="utf-8"))["negatives"]]
        free = [f for f in names if not qa.classify(canned["qa"]["frames"][f], th)[0]]
        return free[:n]

    def with_extra_clean_flags(self, n):
        canned = copy.deepcopy(CANNED)
        for f in self.clean_frames(canned, n):
            canned["qa"]["frames"][f]["text_occluded"] = 0.95
        return canned

    def test_gate_failure_exit_1(self):
        """The gate is clean <= 4/8 and the synthetic set sits at 2, so 2 extra flagged frames still pass and 3 fail."""
        code2, _o, _c = self.run_eval(self.client(self.with_extra_clean_flags(2)), qset=QA)
        self.assertEqual(code2, 0)
        code3, out, _c = self.run_eval(self.client(self.with_extra_clean_flags(3)), qset=QA)
        self.assertEqual(code3, 1)
        self.assertIn("clean frames flagged | 5/8 | <= 4/8 | FAIL", out)
        self.assertEqual(self.report()["meta"]["exit"], 1)

    def test_a_missing_answer_is_unverified_not_a_pass(self):
        canned = copy.deepcopy(CANNED)
        del canned["crop"]["cases"]["c1-settings"]
        code, out, _c = self.run_eval(self.client(canned))
        self.assertEqual(code, 1)
        report = self.report()
        crop = self.gates(report, "crop_box")
        self.assertIsNone(crop["mean IoU"]["ok"])
        self.assertIsNone(crop["centre inside hand rect"]["ok"])
        self.assertEqual(report["results"]["crop_box"]["errors"], {"jev_error": 3})
        self.assertIn("UNVERIFIED", out)
        for key in (QA, "highlight", "stat_check"):
            self.assertTrue(all(g["ok"] is True for g in report["results"][key]["gates"]), key)

    def test_a_dead_backend_leaves_every_gate_unverified(self):
        client = self.client()
        client.backends["jev"].healthy = False
        code, _out, _c = self.run_eval(client)
        self.assertEqual(code, 1)
        report = self.report()
        gates = [g for res in report["results"].values() for g in res["gates"]]
        self.assertTrue(gates and all(g["ok"] is None for g in gates))

    def test_highlight_gate_needs_every_set_b_hand_group_in_the_top_4(self):
        canned = copy.deepcopy(CANNED)
        for rec in canned["highlight"]["records"]:       # every need2 equal: ties keep input order
            for qid, ans in rec["answers"].items():
                if qid.startswith("need2_"):
                    ans["noul"] = 0.5
        res = harness.eval_highlight(self.client(canned))
        gates = {g["name"]: g for g in res["gates"]}
        self.assertFalse(gates["set-b hand ranks"]["ok"])

    def test_stat_gate_fails_when_an_opinion_leaves_its_band(self):
        canned = copy.deepcopy(CANNED)
        canned["stat"]["sentences"]["Now it runs 100x faster."]["stat_is_measured"] = 0.45
        res = harness.eval_stat(self.client(canned))
        gates = {g["name"]: g for g in res["gates"]}
        self.assertFalse(gates["labelled sentences in band"]["ok"])
        self.assertTrue(gates["controls correct at 0.5"]["ok"])


class SelectionTests(EvalCase):
    def sent(self, client):
        return [req for req, _model in client.backends["jev"].calls]

    def logged_client(self, lines):
        self.cache_n = getattr(self, "cache_n", 0) + 1
        return harness.offline_client(copy.deepcopy(CANNED), self.tmp / f"cache{self.cache_n}", log=lines.append)

    def test_module_and_qset_select_sets(self):
        """Counts are the preflight's (what was planned)."""
        for kw, requests, ids in (({"module": "claims"}, N_STAT, {"stat_check"}),
                                  ({"qset": "crop_box"}, N_CROP, {"crop_box", "presence"}),
                                  ({"module": "qa"}, N_QA, {QA}),
                                  ({"module": "captions"}, N_HIGHLIGHT, {"highlight"}),
                                  ({"module": "claims", "qset": "crop_box"}, N_STAT + N_CROP,
                                   {"stat_check", "crop_box", "presence"})):
            with self.subTest(kw):
                lines = []
                client = self.logged_client(lines)
                code, _out, _err = quiet(harness.run_eval, client=client, out_dir=self.out, today=TODAY, **kw)
                self.assertEqual(code, 0)
                self.assertTrue(any(line.startswith(f"jev: {requests} requests") for line in lines), lines)
                self.assertEqual({r.qs.id for r in self.sent(client)}, ids)

    def test_unknown_selection_exits_2(self):
        for kw in ({"qset": "presence"}, {"module": "nope"}, {"backend": "grok"}):
            with self.subTest(kw):
                code, _out, client = self.run_eval(self.client(), **kw)
                self.assertEqual(code, 2)
                self.assertEqual(self.sent(client), [])

    def test_model_override_replaces_the_jev_pin_of_the_sets_it_runs(self):
        code, _out, client = self.run_eval(self.client(), module="claims", model="jev-9.9.9")
        self.assertEqual(code, 0)
        self.assertEqual({model for _req, model in client.backends["jev"].calls}, {"jev-9.9.9"})
        self.assertEqual(self.report()["meta"]["model_override"], "jev-9.9.9")
        _code, _out, client = self.run_eval(self.client(), module="claims")
        self.assertEqual({model for _req, model in client.backends["jev"].calls}, {"jev-1.13.0"})

    def test_dry_run_sends_nothing(self):
        code, out, client = self.run_eval(self.client(), dry_run=True)
        self.assertEqual(code, 0)
        self.assertEqual(client.backends["jev"].calls, [])
        self.assertEqual(client.backends["local"].calls, [])
        self.assertFalse(self.out.exists())
        self.assertEqual(out, "")      # the estimate goes to the client's log

    def test_preflight_runs_before_any_spend_and_stops_over_budget(self):
        lines = []
        client = harness.offline_client(copy.deepcopy(CANNED), self.tmp / "cache-b", log=lines.append)
        code, _out, _err = quiet(harness.run_eval, client=client, out_dir=self.out, today=TODAY, budget=0.001)
        self.assertEqual(code, 3)
        self.assertEqual(client.backends["jev"].calls, [])
        self.assertTrue(any(line.startswith(f"jev: {N_QA + N_HIGHLIGHT + N_CROP + N_STAT} requests") for line in lines),
                        lines)
        self.assertTrue(any("over budget" in line for line in lines), lines)

    def test_default_budget_is_ten_cents(self):
        lines = []
        client = harness.offline_client(copy.deepcopy(CANNED), self.tmp / "cache-c", log=lines.append)
        code, _out, _err = quiet(harness.run_eval, client=client, out_dir=self.out, today=TODAY, dry_run=True)
        self.assertEqual(code, 0)
        self.assertFalse(any("over budget" in line for line in lines), lines)
        code, _out, _err = quiet(harness.run_eval, client=client, out_dir=self.out, today=TODAY, dry_run=True, budget=0.001)
        self.assertEqual(code, 0)
        self.assertTrue(any("over budget" in line for line in lines), lines)


class LocalBackendTests(EvalCase):
    def test_local_backend_skips_box_and_uses_jev_cut(self):
        lines = []
        client = harness.offline_client(copy.deepcopy(CANNED), self.tmp / "cache-local", force_backend="local",
                                        log=lines.append)
        code, out, _c = self.run_eval(client, backend="local")
        self.assertEqual(client.backends["jev"].calls, [])
        sent = [req for req, _model in client.backends["local"].calls]
        self.assertTrue(any(line.startswith(f"local: {N_QA + N_HIGHLIGHT + 8 + N_STAT} requests") for line in lines),
                        lines)                                   # crop: 8 presence, no Box
        self.assertFalse(any(req.has_box() for req in sent))
        self.assertNotIn("crop_box", {req.qs.id for req in sent})
        report = self.report("local")
        crop = report["results"]["crop_box"]
        self.assertIn("n/a: no Box on local (presence only)", crop["notes"])
        self.assertIsNone(crop["metrics"]["mean_iou"])
        self.assertEqual(crop["gates"], [])
        self.assertAlmostEqual(crop["metrics"]["presence_auc"], 1.0)
        self.assertEqual(crop["cut"], "at the local cut")        # presence has local thresholds
        self.assertEqual(report["results"][QA]["cut"], "at Jev's cut")
        self.assertEqual(report["results"]["highlight"]["cut"], "at Jev's cut")
        self.assertEqual(report["results"]["stat_check"]["cut"], "at the local cut")
        self.assertIn("at Jev's cut", out)
        self.assertIn("n/a: no Box on local", self.md("local"))

    def test_local_exit_is_one_only_when_parity_misses_its_reference_by_more_than_one_item(self):
        """The replay carries Jev's numbers: stat, span overlap and presence AUC meet their references; QA clean flagged
        is 2/8 against a local_reference of 6/8, which is far off, so only the QA row fails."""
        code, _out, _c = self.run_eval(self.client(force_backend="local"), backend="local")
        self.assertEqual(code, 1)
        failed = [(key, p["name"]) for key, res in self.report("local")["results"].items() for p in res["parity"]
                  if p["ok"] is not True]
        self.assertEqual(failed, [(QA, "clean frames flagged")])
        for kw in ({"qset": "stat_check"}, {"qset": "highlight"}, {"qset": "crop_box"}):
            with self.subTest(kw):
                code, _out, _c = self.run_eval(self.client(force_backend="local"), backend="local", **kw)
                self.assertEqual(code, 0)

    def test_local_parity_tolerates_one_item(self):
        canned = copy.deepcopy(CANNED)
        canned["stat"]["sentences"]["This will 10x your business."]["stat_is_measured"] = 0.8      # one control wrong
        code, _o, _c = self.run_eval(self.client(canned, force_backend="local"), backend="local", qset="stat_check")
        self.assertEqual(code, 0)
        canned["stat"]["sentences"]["I have a million reasons to try this."]["stat_is_measured"] = 0.8   # two wrong
        code, _o, _c = self.run_eval(self.client(canned, force_backend="local"), backend="local", qset="stat_check")
        self.assertEqual(code, 1)


class ReportTests(EvalCase):
    def test_report_files_written_and_parity_table(self):
        code, _out, _c = self.run_eval(backend="jev")
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         [f"{TODAY}-130509-jev-all.json", f"{TODAY}-130509-jev-all.md"])
        first = self.md()
        self.assertNotIn("parity", first.lower())
        self.assertIn("## Summary", first)
        self.assertIn("| question | positives | recall@flag | recall@0.5 |", first)

        self.run_eval(self.client(force_backend="local"), backend="local")
        second = self.md("local")
        self.assertIn("parity", second.lower())
        self.assertIn("| stat controls correct | 10/10 | 10/10 | 10/10 |", second)
        self.assertIn("| QA clean frames flagged at 0.7 | 2/8 | 2/8 | 6/8 |", second)
        self.assertIn("QA flag pairs at 0.7: 140/140 (100%)", second)
        self.assertIn("stat same side of 0.5: 12/12 (100%)", second)
        self.assertIn("highlight same top-1 span: 28/28 (100%)", second)

    def test_parity_ignores_a_report_from_another_day_or_an_offline_run(self):
        self.run_eval(today="2026-10-23")
        self.run_eval(self.client(force_backend="local"), backend="local")
        self.assertNotIn(TABLE, self.md("local"))
        self.run_eval(offline=True)                        # jev, today, offline: its own kind
        self.assertNotIn(TABLE, self.md("jev", offline=True))

    def test_parity_partner_must_cover_the_sets_of_this_run(self):
        self.run_eval(qset="highlight")
        self.run_eval(self.client(force_backend="local"), backend="local", qset="stat_check")
        self.assertNotIn(TABLE, self.md("local"))      # the only jev report today is for another set
        self.run_eval(qset="stat_check")
        self.run_eval(self.client(force_backend="local"), backend="local", qset="stat_check")
        self.assertIn(TABLE, self.md("local"))
        self.run_eval()                                            # a later jev run of all four covers one set too
        self.run_eval(self.client(force_backend="local"), backend="local", qset="stat_check")
        self.assertIn(TABLE, self.md("local"))

    def test_report_name_includes_scope_and_time(self):
        """`<date>-<HHMMSS>-<backend>-<scope>[-offline]`: the set ids that ran, sorted, joined by `+`; one stem per run."""
        cases = (({"qset": "highlight"}, "jev", "highlight"),
                 ({"qset": "crop_box"}, "jev", "crop_box+presence"),
                 ({"qset": "crop_box", "backend": "local"}, "local", "crop_box+presence"),
                 ({"module": "claims"}, "jev", "stat_check"),
                 ({"module": "claims", "qset": "highlight"}, "jev", "highlight+stat_check"),
                 ({"qset": QA}, "jev", QA))
        for kw, backend, scope in cases:
            with self.subTest(kw):
                forced = backend if backend == "local" else "jev"
                with mock.patch.object(harness, "_now", return_value=START):
                    code, out, _c = self.run_eval(self.client(force_backend=forced), **kw)
                stem = f"{TODAY}-130509-{backend}-{scope}"
                self.assertTrue((self.out / f"{stem}.md").is_file(), sorted(p.name for p in self.out.iterdir()))
                self.assertTrue((self.out / f"{stem}.json").is_file())
                self.assertIn(f"eval: wrote {self.out / (stem + '.md')} and {stem}.json", out)
        names = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(len(names), 2 * len(cases))
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9._+-]+", n) for n in names), names)
        self.assertEqual({Path(n).stem for n in names if n.endswith(".md")}, {Path(n).stem for n in names if n.endswith(".json")})

    def test_report_name_takes_the_date_and_time_from_the_clock_when_no_date_is_given(self):
        late = datetime.datetime(2027, 1, 2, 3, 4, 5)
        with mock.patch.object(harness, "_now", return_value=late):
            self.run_eval(today=None, qset="highlight", offline=True)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["2027-01-02-030405-jev-highlight-offline.json", "2027-01-02-030405-jev-highlight-offline.md"])

    def test_two_runs_same_day_keep_both_reports(self):
        for qset in ("highlight", "stat_check"):
            code, _out, _err = quiet(harness.main, ["--offline", "--qset", qset, "--out", str(self.out)])
            self.assertEqual(code, 0)
        names = sorted(p.name for p in self.out.iterdir())
        stems = sorted({Path(n).stem for n in names})
        self.assertEqual(len(names), 4, names)
        self.assertEqual(len(stems), 2, names)
        by_scope = {s.split("-jev-")[1]: s for s in stems}
        self.assertEqual(set(by_scope), {"highlight-offline", "stat_check-offline"})
        for scope, needle in (("highlight-offline", "highlight@1"), ("stat_check-offline", "stat_check@1")):
            stem = by_scope[scope]
            self.assertIn(needle, (self.out / f"{stem}.md").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((self.out / f"{stem}.json").read_text(encoding="utf-8"))["meta"]["sets"],
                             [needle.split("@")[0]])

    def test_same_second_collision_gets_suffix(self):
        base = f"{TODAY}-130509-jev-highlight"
        with mock.patch.object(harness, "_now", return_value=START):
            self.run_eval(qset="highlight")
            (self.out / f"{base}.md").write_text("FIRST RUN", encoding="utf-8")
            first_json = (self.out / f"{base}.json").read_text(encoding="utf-8")
            self.run_eval(qset="highlight")
            self.run_eval(qset="highlight")
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         [f"{base}-2.json", f"{base}-2.md", f"{base}-3.json", f"{base}-3.md", f"{base}.json", f"{base}.md"])
        self.assertEqual((self.out / f"{base}.md").read_text(encoding="utf-8"), "FIRST RUN")      # never overwritten
        self.assertEqual((self.out / f"{base}.json").read_text(encoding="utf-8"), first_json)
        self.assertIn("## Summary", (self.out / f"{base}-3.md").read_text(encoding="utf-8"))

    def test_a_lone_leftover_file_still_forces_the_next_suffix(self):
        """A pair shares one stem, so a stray `.md` alone takes the name: the run goes to `-2` for both files."""
        base = f"{TODAY}-130509-jev-stat_check"
        self.out.mkdir()
        (self.out / f"{base}.md").write_text("STRAY", encoding="utf-8")
        with mock.patch.object(harness, "_now", return_value=START):
            self.run_eval(qset="stat_check")
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), [f"{base}-2.json", f"{base}-2.md", f"{base}.md"])
        self.assertEqual((self.out / f"{base}.md").read_text(encoding="utf-8"), "STRAY")

    def test_all_scope_when_no_qset(self):
        self.run_eval()
        self.run_eval(qset=None, module=None, offline=True)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         [f"{TODAY}-130509-jev-all.json", f"{TODAY}-130509-jev-all.md",
                          f"{TODAY}-130510-jev-all-offline.json", f"{TODAY}-130510-jev-all-offline.md"])
        self.assertEqual(self.report()["meta"]["sets"], list(harness.SETS))

    def test_json_report_shape(self):
        self.run_eval()
        report = self.report()
        self.assertEqual(set(report), {"meta", "results", "items"})
        self.assertEqual(report["meta"]["backend"], "jev")
        self.assertEqual(report["meta"]["sets"], list(harness.SETS))
        self.assertEqual(report["meta"]["exit"], 0)
        self.assertEqual(set(report["results"]), set(harness.SETS))
        self.assertEqual(set(report["items"]), {"qa", "highlight", "stat"})
        self.assertIn("backends_used", report["meta"])
        self.assertNotIn("headers", json.dumps(report).lower())

    def test_main_offline_replays_and_marks_the_files(self):
        code, out, _err = quiet(harness.main, ["--offline", "--out", str(self.out)])
        self.assertEqual(code, 0)
        self.assertIn("(offline replay)", out)
        stem = next(self.out.glob("*-jev-all-offline.md")).stem
        text = (self.out / f"{stem}.md").read_text(encoding="utf-8")
        self.assertIn("OFFLINE", text)
        self.assertIn("not a measurement of Jev or Decider", text)
        self.assertTrue(json.loads((self.out / f"{stem}.json").read_text(encoding="utf-8"))["meta"]["offline"])

    def test_main_returns_codes_instead_of_exiting(self):
        for argv, code in ((["--qset", "nope"], 2), (["--nope", "--offline"], 2), (["--help"], 0)):
            with self.subTest(argv):
                self.assertEqual(quiet(harness.main, argv + ["--out", str(self.out)])[0], code)
        self.assertEqual(quiet(harness.main, ["--offline", "--dry-run", "--out", str(self.out)])[0], 0)
        self.assertFalse(self.out.exists())

    def test_main_offline_local_replay_exits_1_on_the_qa_parity_row(self):
        code, _out, _err = quiet(harness.main, ["--offline", "--backend", "local", "--out", str(self.out)])
        self.assertEqual(code, 1)


class SafetyTests(EvalCase):
    def test_a_full_offline_eval_makes_no_network_call(self):
        boom = AssertionError("the eval must not touch the network")
        with mock.patch.object(socket.socket, "connect", side_effect=boom), \
                mock.patch.object(urllib.request, "urlopen", side_effect=boom):
            code, _out, _c = self.run_eval()
        self.assertEqual(code, 0)

    def test_the_real_client_cannot_reach_a_backend_offline(self):
        """JEVM_OFFLINE is set by tests/__init__: a real client answers `jev_unavailable` and every gate is UNVERIFIED."""
        with mock.patch("jevlib.client.load_key", return_value="not-a-real-key"):   # keep the test off any .env
            real = harness.make_client(cache_dir=self.tmp / "real-cache", log=lambda *_: None, force_backend="jev")
        code, _out, _c = self.run_eval(real, module="claims")
        self.assertEqual(code, 1)
        self.assertEqual(self.report()["results"]["stat_check"]["errors"], {"jev_unavailable": N_STAT})

    def test_nothing_is_written_to_the_tool_folders(self):
        def snapshot():
            return {p for d in ("eval", ".jevm-cache") if (TOOL_ROOT / d).exists() for p in (TOOL_ROOT / d).rglob("*")}
        before = snapshot()
        self.run_eval()
        self.assertEqual(snapshot(), before)

    @staticmethod
    def kit_geometry():
        """The example kit's band, safe-zone and window numbers (no .py file of the harness may spell them)."""
        short = json.loads((TOOL_ROOT / "examples" / "kit" / "manifest.json").read_text(encoding="utf-8"))["canvases"]["short"]
        return {*short["bands"]["word"], short["safe"]["top_px"], short["safe"]["bottom_px"], short["window_body"][0]}

    def test_harness_holds_no_kit_geometry(self):
        for path in (TOOL_ROOT / "evals" / "harness.py", Path(__file__)):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            numbers = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, int)}
            self.assertFalse(numbers & self.kit_geometry(), f"{path.name} holds kit geometry")


if __name__ == "__main__":
    unittest.main()
