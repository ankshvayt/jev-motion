"""Tests for spec.build and spec.validate (stdlib unittest). From the tool folder:

    python3 -m unittest tests.test_spec -q

Offline and pure: the kit manifest is tests/fixtures/manifest_min.json, projects live in temp folders, and nothing here
touches the network.
"""
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from jevlib.cache import file_sha256
from jevlib.paths import project_root, work_dir
from spec import build, validate

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SPEC_FIXTURE = FIXTURES / "spec_min.json"
MANIFEST_FIXTURE = FIXTURES / "manifest_min.json"


def fixture_spec():
    return json.loads(SPEC_FIXTURE.read_text(encoding="utf-8"))


def timed_words(texts, gap_after=None):
    """Word dicts of 0.2 s each, back to back, with an extra silence after the word indexes in `gap_after`."""
    gap_after = gap_after or {}
    words, t = [], 0.0
    for i, text in enumerate(texts):
        words.append({"id": f"w{i}", "text": text, "start": round(t, 3), "end": round(t + 0.2, 3)})
        t += 0.2 + gap_after.get(i, 0.0)
    return words


def run(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = fn(*args)
    return code, out.getvalue(), err.getvalue()


class ValidateTests(unittest.TestCase):
    def test_fixture_spec_validates(self):
        self.assertEqual(validate.validate(fixture_spec(), manifest_path=None), [])

    def test_schema_rejects_bad_sensitivity_and_missing_required(self):
        spec = fixture_spec()
        spec["assets"][1]["sensitivity"] = "secret"
        del spec["project"]["title"]
        errors = validate.validate(spec)
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(any(e.startswith("$.assets[1].sensitivity:") and "secret" in e for e in errors), errors)
        self.assertTrue(any(e.startswith("$.project.title:") and "required" in e for e in errors), errors)

    def test_validate_rejects_dangling_word_ref(self):
        spec = fixture_spec()
        spec["units"]["caption_groups"][1]["words"] = ["w0", "w99"]
        errors = validate.validate(spec)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("caption_groups[1]", errors[0])
        self.assertIn("w99", errors[0])

    def test_validate_rejects_reversed_span_and_duplicate_ids(self):
        spec = fixture_spec()
        spec["units"]["sentences"][0]["words"] = ["w7", "w0"]
        spec["assets"][2]["id"] = "a1"
        spec["units"]["caption_groups"][0]["sentence"] = "sn9"
        errors = validate.validate(spec)
        self.assertEqual(len(errors), 3, errors)
        self.assertTrue(any("sentences[0]" in e and "w7" in e for e in errors), errors)
        self.assertTrue(any("duplicate" in e and "a1" in e for e in errors), errors)
        self.assertTrue(any("sn9" in e for e in errors), errors)

    def test_validate_rejects_a_caption_group_over_the_kit_cap(self):
        spec = fixture_spec()
        cap = spec["constraints"]["captions"]["max_words_per_line"] * spec["constraints"]["captions"]["max_lines"]
        spec["units"]["caption_groups"][0]["words"] = ["w0", f"w{cap}"]
        errors = validate.validate(spec)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("caption_groups[0].words", errors[0])
        self.assertIn(f"{cap + 1} words", errors[0])
        spec["units"]["caption_groups"][0]["words"] = ["w0", f"w{cap - 1}"]
        self.assertEqual(validate.validate(spec), [])
        spec["constraints"]["captions"]["max_lines"] = 1
        errors = validate.validate(spec)
        self.assertTrue(any("caption_groups[0].words" in e for e in errors), errors)
        self.assertTrue(any("caption_groups[2].lines" in e for e in errors), errors)

    def test_validate_manifest_sha_mismatch(self):
        spec = fixture_spec()
        with tempfile.TemporaryDirectory() as tmp:
            other = Path(tmp) / "manifest.json"
            other.write_text('{"manifest_version": "other"}', encoding="utf-8")
            errors = validate.validate(spec, manifest_path=other)
            self.assertEqual(len(errors), 1, errors)
            self.assertIn("manifest_sha256", errors[0])
            spec["kit"]["manifest_sha256"] = file_sha256(other)
            self.assertEqual(validate.validate(spec, manifest_path=other), [])

    def test_validate_skips_sha_when_manifest_file_is_absent(self):
        spec = fixture_spec()
        spec["kit"]["manifest"] = "no/such/manifest.json"
        self.assertEqual(validate.validate(spec), [])

    def test_validator_keywords(self):
        schema = {
            "type": "object", "required": ["n"], "additionalProperties": False,
            "properties": {
                "n": {"type": ["integer", "null"], "minimum": 1, "maximum": 3},
                "s": {"type": "string", "pattern": "^a+$", "enum": ["a", "aa"]},
                "pair": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}},
            },
        }
        ok = [{"n": 1}, {"n": None}, {"n": 3.0}, {"n": 2, "s": "aa", "pair": ["x", "y"]}]
        for instance in ok:
            self.assertEqual(validate.validate_schema(instance, schema), [], instance)
        bad = {
            "not an object": [],
            "bool is not an integer": {"n": True},
            "below minimum": {"n": 0},
            "above maximum": {"n": 4},
            "missing required": {},
            "extra property": {"n": 1, "z": 1},
            "pattern": {"n": 1, "s": "b"},
            "enum": {"n": 1, "s": "aaa"},
            "too few items": {"n": 1, "pair": ["x"]},
            "too many items": {"n": 1, "pair": ["x", "y", "z"]},
            "item type": {"n": 1, "pair": ["x", 2]},
        }
        for name, instance in bad.items():
            errors = validate.validate_schema(instance, schema)
            self.assertTrue(errors, name)
            self.assertTrue(all(e.startswith("$") for e in errors), (name, errors))
        self.assertEqual(validate.validate_schema({"n": 1, "pair": ["x", 2]}, schema)[0][:12], "$.pair[1]: e")

    def test_cli_validate_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            code, out, _ = run(validate.cli_validate, SPEC_FIXTURE)
            self.assertEqual(code, 0)
            self.assertIn("valid", out)

            bad = fixture_spec()
            bad["assets"][0]["sensitivity"] = "secret"
            bad_path = tmp / "bad.json"
            bad_path.write_text(json.dumps(bad), encoding="utf-8")
            code, _, err = run(validate.cli_validate, bad_path)
            self.assertEqual(code, 1)
            self.assertIn("$.assets[0].sensitivity", err)

            code, _, err = run(validate.cli_validate, tmp / "missing.json")
            self.assertEqual(code, 2)
            self.assertIn("missing.json", err)

            junk = tmp / "junk.json"
            junk.write_text("not json", encoding="utf-8")
            self.assertEqual(run(validate.cli_validate, junk)[0], 2)

            listy = tmp / "list.json"
            listy.write_text("[]", encoding="utf-8")
            self.assertEqual(run(validate.cli_validate, listy)[0], 1)


class ChunkingTests(unittest.TestCase):
    def test_sentences_split_on_punctuation_and_long_pause(self):
        words = timed_words(["One", "two.", "Three", "four?", "Five!", "Six…", "seven", "eight", "nine"],
                            gap_after={6: 0.60, 7: 0.59})
        got = [(s["id"], s["words"]) for s in build.sentences(words)]
        self.assertEqual(got, [("sn0", ["w0", "w1"]), ("sn1", ["w2", "w3"]), ("sn2", ["w4", "w4"]),
                               ("sn3", ["w5", "w5"]), ("sn4", ["w6", "w6"]), ("sn5", ["w7", "w8"])])
        first = build.sentences(words)[0]
        self.assertEqual((first["start"], first["end"]), (words[0]["start"], words[1]["end"]))

    def test_sentences_ignore_closing_quotes_after_the_mark(self):
        words = timed_words(["He", "said", 'stop."', "Then", "left"])
        self.assertEqual([s["words"] for s in build.sentences(words)], [["w0", "w2"], ["w3", "w4"]])

    @staticmethod
    def groups_for(texts, gap_after=None, per_line=5, max_lines=2):
        words = timed_words(texts, gap_after)
        return build.caption_groups(words, build.sentences(words), per_line, max_lines)

    @staticmethod
    def spans(groups):
        return [g["words"] for g in groups]

    def test_caption_groups_rules(self):
        plain = [f"word{i}" for i in range(12)]
        groups = self.groups_for(plain)
        self.assertEqual([g["id"] for g in groups], ["cg0", "cg1"])
        self.assertEqual(self.spans(groups), [["w0", "w5"], ["w6", "w11"]])
        self.assertEqual([g["lines"] for g in groups], [2, 2])
        self.assertEqual({g["sentence"] for g in groups}, {"sn0"})

        with self.subTest("comma after word 6 closes the group"):
            texts = list(plain)
            texts[5] += ","
            groups = self.groups_for(texts)
            self.assertEqual(self.spans(groups), [["w0", "w5"], ["w6", "w11"]])
            self.assertEqual([g["lines"] for g in groups], [2, 2])

        with self.subTest("comma after word 7 closes the group where a plain split would not"):
            texts = list(plain)
            texts[6] += ","
            groups = self.groups_for(texts)
            self.assertEqual(self.spans(groups), [["w0", "w6"], ["w7", "w11"]])
            self.assertEqual([g["lines"] for g in groups], [2, 1])

        with self.subTest("comma inside a short group does not close it"):
            texts = list(plain)
            texts[1] += ","
            self.assertEqual(self.spans(self.groups_for(texts)), [["w0", "w5"], ["w6", "w11"]])

        with self.subTest("a gap of 0.30 s closes the group"):
            groups = self.groups_for(plain, gap_after={2: 0.30})
            self.assertEqual(self.spans(groups), [["w0", "w2"], ["w3", "w11"]])
            self.assertEqual([g["lines"] for g in groups], [1, 2])

        with self.subTest("a sentence end closes the group and carries the sentence id"):
            texts = list(plain)
            texts[3] += "."
            groups = self.groups_for(texts)
            self.assertEqual(self.spans(groups), [["w0", "w3"], ["w4", "w11"]])
            self.assertEqual([g["sentence"] for g in groups], ["sn0", "sn1"])

        with self.subTest("group timing spans its first and last word"):
            words = timed_words(plain)
            first = build.caption_groups(words, build.sentences(words), 5, 2)[0]
            self.assertEqual((first["start"], first["end"]), (words[0]["start"], words[5]["end"]))

    def test_caption_groups_cap_for_a_long_sentence_without_pauses(self):
        texts = [f"word{i}" for i in range(23)]
        groups = self.groups_for(texts)
        sizes = [int(g["words"][1][1:]) - int(g["words"][0][1:]) + 1 for g in groups]
        self.assertEqual(sum(sizes), 23)
        self.assertTrue(all(n <= 10 for n in sizes), sizes)
        self.assertEqual(sizes, [5, 6, 6, 6])
        self.assertEqual([g["lines"] for g in groups], [1, 2, 2, 2])

    def test_caption_groups_prefer_one_line_where_a_gap_allows(self):
        texts = [f"word{i}" for i in range(23)]
        groups = self.groups_for(texts, gap_after={4: 0.15, 9: 0.15, 13: 0.15, 18: 0.15})
        self.assertEqual(self.spans(groups), [["w0", "w4"], ["w5", "w9"], ["w10", "w13"], ["w14", "w18"], ["w19", "w22"]])
        self.assertEqual({g["lines"] for g in groups}, {1})

    def test_caption_groups_split_at_the_largest_gap(self):
        texts = [f"word{i}" for i in range(8)]
        groups = self.groups_for(texts, gap_after={1: 0.16, 4: 0.25, 5: 0.20})
        self.assertEqual(self.spans(groups), [["w0", "w4"], ["w5", "w7"]])
        self.assertEqual([g["lines"] for g in groups], [1, 1])

    def test_caption_groups_split_ties_nearest_the_middle(self):
        texts = [f"word{i}" for i in range(8)]
        groups = self.groups_for(texts, gap_after={1: 0.20, 3: 0.20, 5: 0.20})
        self.assertEqual(self.spans(groups), [["w0", "w3"], ["w4", "w7"]])

    def test_caption_groups_keep_two_lines_only_without_a_gap_to_split_at(self):
        texts = [f"word{i}" for i in range(8)]
        self.assertEqual(self.spans(self.groups_for(texts)), [["w0", "w7"]])
        self.assertEqual(self.spans(self.groups_for(texts, gap_after={3: 0.14})), [["w0", "w7"]])
        self.assertEqual(self.spans(self.groups_for(texts, gap_after={3: 0.15})), [["w0", "w3"], ["w4", "w7"]])
        self.assertEqual([g["lines"] for g in self.groups_for(texts)], [2])

    def test_caption_groups_size_comes_from_the_arguments(self):
        texts = [f"word{i}" for i in range(23)]
        for per_line, max_lines in ((3, 1), (4, 2), (6, 3)):
            groups = self.groups_for(texts, per_line=per_line, max_lines=max_lines)
            sizes = [int(g["words"][1][1:]) - int(g["words"][0][1:]) + 1 for g in groups]
            self.assertEqual(sum(sizes), 23)
            self.assertTrue(all(n <= per_line * max_lines for n in sizes), (per_line, max_lines, sizes))
            self.assertTrue(all(g["lines"] <= max_lines for g in groups), (per_line, max_lines))


class LoadTests(unittest.TestCase):
    def test_load_words_shapes_and_ids(self):
        rows = [{"word": "Hello", "start": 0.0, "end": 0.3, "line": 1},
                {"text": "  ", "start": 0.3, "end": 0.4},
                {"text": "world.", "start": -0.02, "end": 0.9, "speaker": "s1", "conf": 0.5},
                {"text": "", "start": 1, "end": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in (("list.json", rows), ("obj.json", {"words": rows})):
                path = Path(tmp) / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(build.load_words(path), [
                    {"id": "w0", "text": "Hello", "start": 0.0, "end": 0.3},
                    {"id": "w1", "text": "world.", "start": -0.02, "end": 0.9, "speaker": "s1", "conf": 0.5}])
            bad = Path(tmp) / "bad.json"
            bad.write_text(json.dumps([{"text": "oops", "start": 0}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                build.load_words(bad)
            bad.write_text(json.dumps({"nope": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                build.load_words(bad)

    def test_load_assets(self):
        self.assertEqual(build.load_assets(None)[0], [])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "assets.json"
            path.write_text(json.dumps({"assets": [
                {"id": "a1", "kind": "image", "uri": "x.png", "description": "d", "sensitivity": "public"},
                {"id": "a2", "kind": "video", "uri": "y.mp4"}]}), encoding="utf-8")
            assets, notes = build.load_assets(path)
            self.assertEqual([a["sensitivity"] for a in assets], ["public", "private"])
            self.assertTrue(any("a2" in n and "sensitivity" in n for n in notes), notes)
            self.assertTrue(any("a2" in n and "description" in n for n in notes), notes)
            self.assertFalse(any("a1" in n for n in notes), notes)
            path.write_text(json.dumps([{"kind": "image", "uri": "x.png"}]), encoding="utf-8")
            with self.assertRaises(ValueError):
                build.load_assets(path)


class BuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.project = self.tmp / "proj"
        (self.project / "work").mkdir(parents=True)
        self.words = self.project / "work" / "words.json"
        self.words.write_text(json.dumps(fixture_spec()["transcript"]["words"]), encoding="utf-8")
        self.assets = self.project / "work" / "assets.json"
        self.assets.write_text(json.dumps([
            {"id": "a1", "kind": "screen_recording", "uri": "raw/screen.mov", "description": "A screen recording",
             "sensitivity": "public"},
            {"id": "a2", "kind": "image", "uri": "assets/shot.png", "description": "A screenshot"}]), encoding="utf-8")

    def build(self, **kw):
        return build.build(self.project, "short", self.words, assets_path=self.assets,
                           manifest_path=MANIFEST_FIXTURE, **kw)

    def test_build_from_fixture_words(self):
        spec, notes = self.build()
        self.assertEqual(validate.validate(spec), [])
        fixture = fixture_spec()
        for block in ("spec_version", "format", "canvas", "constraints", "decide", "creator_declared"):
            self.assertEqual(spec[block], fixture[block], block)
        self.assertEqual({**spec["kit"], "manifest": ""}, {**fixture["kit"], "manifest": ""})
        self.assertEqual(spec["kit"]["manifest"], str(MANIFEST_FIXTURE.resolve()))
        self.assertEqual(spec["format"]["declared_by"], "human")
        self.assertEqual(spec["constraints"]["highlight"]["density"], 0.25)
        self.assertEqual(spec["constraints"]["highlight"]["from"], "kit+format")
        self.assertEqual(spec["transcript"], fixture["transcript"])
        self.assertEqual(spec["units"]["sentences"], fixture["units"]["sentences"])
        self.assertEqual([(g["words"], g["lines"], g["sentence"]) for g in spec["units"]["caption_groups"]],
                         [(["w0", "w7"], 2, "sn0"), (["w8", "w13"], 2, "sn1"), (["w14", "w17"], 1, "sn1")])
        self.assertEqual(spec["project"]["id"], "proj")
        self.assertEqual([a["sensitivity"] for a in spec["assets"]], ["public", "private"])
        self.assertTrue(any("a2" in n and "private" in n for n in notes), notes)

    def test_creator_density_override(self):
        spec, _ = self.build(creator_declared={"highlight_density": 0.4})
        self.assertEqual(spec["constraints"]["highlight"]["density"], 0.4)
        self.assertEqual(spec["constraints"]["highlight"]["from"], "kit+creator")
        self.assertEqual(spec["creator_declared"]["highlight_density"], 0.4)
        self.assertEqual(validate.validate(spec), [])

    def test_caption_cap_is_read_from_the_manifest(self):
        manifest = json.loads(MANIFEST_FIXTURE.read_text(encoding="utf-8"))
        manifest["captions"]["max_words_per_line"] = 3
        manifest["captions"]["max_lines"] = 1
        path = self.tmp / "manifest_3x1.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        spec, _ = build.build(self.project, "short", self.words, manifest_path=path)
        index = {w["id"]: i for i, w in enumerate(spec["transcript"]["words"])}
        sizes = [index[g["words"][1]] - index[g["words"][0]] + 1 for g in spec["units"]["caption_groups"]]
        self.assertEqual(sum(sizes), 18)
        self.assertTrue(all(n <= 3 for n in sizes), sizes)
        self.assertEqual({g["lines"] for g in spec["units"]["caption_groups"]}, {1})
        self.assertEqual(spec["constraints"]["captions"], {"max_words_per_line": 3, "max_lines": 1, "from": "kit"})
        self.assertEqual(validate.validate(spec), [])

    def test_build_long_family_uses_its_own_canvas_and_budget(self):
        spec, _ = build.build(self.project, "long", self.words, manifest_path=MANIFEST_FIXTURE)
        self.assertEqual((spec["canvas"]["aspect"], spec["canvas"]["width"], spec["canvas"]["height"]),
                         ("16:9", 1920, 1080))
        self.assertEqual(spec["kit"]["canvas"], "long")
        self.assertEqual(spec["kit"]["theme_policy"], "one_per_video")
        self.assertEqual(spec["decide"]["budget_usd"], 1.0)
        self.assertEqual(spec["constraints"]["highlight"]["density"], 0.10)
        self.assertEqual(spec["constraints"]["safe"], {"all_px": 90, "from": "kit"})
        self.assertEqual(spec["assets"], [])
        self.assertEqual(validate.validate(spec), [])

    def test_build_unknown_format_exit_2(self):
        with self.assertRaises(ValueError) as caught:
            build.build(self.project, "nope", self.words, manifest_path=MANIFEST_FIXTURE)
        for card in ("short", "long"):
            self.assertIn(card, str(caught.exception))
        code, _, err = run(build.cli_build, self.project, "nope", self.words, None, FIXTURES, None)
        self.assertEqual(code, 2)
        self.assertIn("short", err)

    def test_project_root_and_manifest_are_absolute(self):
        """A spec stores absolute paths, so it reads the same from any working directory."""
        inside = self.tmp / "videos" / "my video"
        (inside / "work").mkdir(parents=True)
        words = inside / "work" / "words.json"
        shutil.copy(self.words, words)
        old = Path.cwd()
        os.chdir(self.tmp)
        try:
            spec, _ = build.build(Path("videos") / "my video", "short", words,
                                  manifest_path=Path(os.path.relpath(MANIFEST_FIXTURE, self.tmp)))
        finally:
            os.chdir(old)
        self.assertEqual(spec["project"]["root"], str(inside))
        self.assertEqual(spec["kit"]["manifest"], str(MANIFEST_FIXTURE.resolve()))
        self.assertEqual(spec["transcript"]["source"], "work/words.json")
        self.assertEqual(project_root(spec), inside)
        self.assertEqual(work_dir(spec), inside / "work" / "jevm")

    def test_build_needs_a_kit(self):
        with self.assertRaises(ValueError) as ctx:
            build.build(self.project, "short", self.words)
        self.assertIn("--kit-dir", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            build.build(self.project, "short", self.words, kit_dir=self.tmp / "no kit")
        self.assertIn("kit manifest not found", str(ctx.exception))


class CliBuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.kit_dir = self.tmp / "kits" / "my kit"
        self.kit_dir.mkdir(parents=True)
        shutil.copy(MANIFEST_FIXTURE, self.kit_dir / "manifest.json")
        patcher = mock.patch.dict("os.environ", {"JEVM_KIT_DIR": ""})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = self.tmp / "My Project"
        (self.project / "work").mkdir(parents=True)
        self.words = self.project / "work" / "words.json"
        self.words.write_text(json.dumps({"words": fixture_spec()["transcript"]["words"]}), encoding="utf-8")
        self.out = self.project / "work" / "jevm" / "spec.json"

    def test_cli_build_writes_default_output_and_exits_0(self):
        code, out, err = run(build.cli_build, self.project, "short", self.words, None, self.kit_dir, None)
        self.assertEqual((code, err), (0, ""))
        self.assertTrue(self.out.is_file())
        spec = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(validate.validate(spec), [])
        self.assertEqual(spec["project"]["id"], "my-project")
        self.assertEqual(project_root(spec), self.project)
        self.assertEqual(work_dir(spec) / "spec.json", self.out)
        self.assertEqual(spec["kit"]["manifest_sha256"], file_sha256(MANIFEST_FIXTURE))
        self.assertIn("3 caption groups", out)
        self.assertIn("no asset list", out)

    def test_cli_build_explicit_output_path(self):
        target = self.tmp / "elsewhere" / "spec.json"
        code, _, _ = run(build.cli_build, self.project, "short", self.words, None, self.kit_dir, target)
        self.assertEqual(code, 0)
        self.assertTrue(target.is_file())
        self.assertFalse(self.out.exists())

    def test_cli_build_keeps_creator_declared_on_rebuild(self):
        self.assertEqual(run(build.cli_build, self.project, "short", self.words, None, self.kit_dir, None)[0], 0)
        spec = json.loads(self.out.read_text(encoding="utf-8"))
        spec["creator_declared"] = {"highlight_density": 0.4, "must_show": ["the answer card"]}
        self.out.write_text(json.dumps(spec), encoding="utf-8")
        code, _, _ = run(build.cli_build, self.project, "short", self.words, None, None, None)
        self.assertEqual(code, 0)
        rebuilt = json.loads(self.out.read_text(encoding="utf-8"))
        self.assertEqual(rebuilt["creator_declared"], {"highlight_density": 0.4, "must_show": ["the answer card"]})
        self.assertEqual(rebuilt["constraints"]["highlight"]["density"], 0.4)

    def test_cli_build_bad_input_exits_2(self):
        for words, kit in ((self.project / "nope.json", self.kit_dir), (self.words, self.tmp / "no-such-kit"),
                           (self.words, None)):
            code, _, err = run(build.cli_build, self.project, "short", words, None, kit, None)
            self.assertEqual(code, 2, (words, kit))
            self.assertTrue(err.startswith("jevm spec build:"), err)
        self.assertFalse(self.out.exists())

    def test_cli_build_kit_dir_precedence(self):
        """--kit-dir, else the kit of the spec already at the output path, else JEVM_KIT_DIR."""
        other = self.tmp / "kits" / "other kit"
        other.mkdir(parents=True)
        manifest = json.loads(MANIFEST_FIXTURE.read_text(encoding="utf-8"))
        manifest["kit"]["id"] = "other"
        (other / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with mock.patch.dict("os.environ", {"JEVM_KIT_DIR": str(other)}):
            self.assertEqual(run(build.cli_build, self.project, "short", self.words, None, None, None)[0], 0)
            self.assertEqual(json.loads(self.out.read_text(encoding="utf-8"))["kit"]["id"], "other")
            run(build.cli_build, self.project, "short", self.words, None, self.kit_dir, None)
            self.assertEqual(json.loads(self.out.read_text(encoding="utf-8"))["kit"]["id"], "example")
            run(build.cli_build, self.project, "short", self.words, None, None, None)   # the spec's kit wins
            self.assertEqual(json.loads(self.out.read_text(encoding="utf-8"))["kit"]["id"], "example")

    def test_cli_build_invalid_result_exits_1_and_writes_nothing(self):
        assets = self.project / "work" / "assets.json"
        assets.write_text(json.dumps([{"id": "a1", "kind": "hologram", "uri": "x"}]), encoding="utf-8")
        code, _, err = run(build.cli_build, self.project, "short", self.words, assets, self.kit_dir, None)
        self.assertEqual(code, 1)
        self.assertIn("$.assets[0].kind", err)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
