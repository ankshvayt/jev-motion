"""Tests for `kits/kit.py`: the kit folder lookup, `kit build` (stamping, with and without an adapter), `kit check`
(schema and drift) and `load_manifest`.

Everything is file-based and offline. `examples/kit/` is only read: every build and check runs on a temporary copy of
it, in a folder whose name contains a space, so the shipped example manifest is never written by a test.
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from jevlib.cache import file_sha256
from jevlib.paths import TOOL_ROOT
from kits import kit
from kits.kit import KitError

EXAMPLE = TOOL_ROOT / "examples" / "kit"
FIXTURE = TOOL_ROOT / "tests" / "fixtures" / "manifest_min.json"
VOLATILE = (("kit", "fingerprint"), ("kit", "built_at"), ("sources",))


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _without_volatile(manifest):
    """The manifest minus `kit.fingerprint`, `kit.built_at` and `sources` (the parts `kit build` fills)."""
    out = json.loads(json.dumps(manifest))
    for *parents, leaf in VOLATILE:
        node = out
        for key in parents:
            node = node[key]
        node.pop(leaf, None)
    return out


def run_cli(fn, *args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(*args)
    return code, out.getvalue(), err.getvalue()


class TempKit(unittest.TestCase):
    """A temporary copy of `examples/kit/` at `self.kit` (a path with a space)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="jevm kit ")
        self.addCleanup(tmp.cleanup)
        self.kit = Path(tmp.name) / "my kit"
        shutil.copytree(EXAMPLE, self.kit)
        self.adapter = self.kit / "adapter.py"


class ResolveTests(unittest.TestCase):
    def test_flag_then_spec_then_env(self):
        spec = {"kit": {"manifest": "/kits/from spec/manifest.json"}}
        env = {kit.KIT_DIR_ENV: "/kits/from env"}
        self.assertEqual(kit.resolve_kit_dir("/kits/flag", spec, env), Path("/kits/flag"))
        self.assertEqual(kit.resolve_kit_dir(None, spec, env), Path("/kits/from spec"))
        self.assertEqual(kit.resolve_kit_dir(None, None, env), Path("/kits/from env"))
        self.assertEqual(kit.resolve_kit_dir(None, {"kit": {}}, env), Path("/kits/from env"))

    def test_the_spec_names_the_manifest_file_itself(self):
        spec = {"kit": {"manifest": "/kits/k/custom name.json"}}
        self.assertEqual(kit.resolve_manifest(None, spec, {}), Path("/kits/k/custom name.json"))
        self.assertEqual(kit.resolve_manifest("/kits/flag", spec, {}), Path("/kits/flag/manifest.json"))
        self.assertEqual(kit.resolve_manifest(None, None, {kit.KIT_DIR_ENV: "/kits/env"}), Path("/kits/env/manifest.json"))

    def test_relative_spec_manifest_resolves_from_the_current_directory(self):
        spec = {"kit": {"manifest": "kits/k/manifest.json"}}
        self.assertEqual(kit.resolve_kit_dir(None, spec, {}), Path.cwd() / "kits" / "k")

    def test_no_kit_anywhere_raises(self):
        with self.assertRaises(KitError) as ctx:
            kit.resolve_kit_dir(None, None, {})
        self.assertIn("--kit-dir", str(ctx.exception))
        self.assertIn(kit.KIT_DIR_ENV, str(ctx.exception))


class ExampleKitTests(unittest.TestCase):
    def test_shipped_example_is_valid_and_fresh(self):
        code, out, err = run_cli(kit.cli_check, EXAMPLE)
        self.assertEqual(code, 0, out + err)
        self.assertIn("kit check: example", out)

    def test_example_manifest_matches_its_adapter_and_the_test_fixture(self):
        """The hand-kept test fixture, the shipped example and the adapter's output agree on every number."""
        built = kit.build(EXAMPLE, kit.load_adapter(EXAMPLE / "adapter.py"))
        shipped = _load(EXAMPLE / "manifest.json")
        self.assertEqual(_without_volatile(built), _without_volatile(shipped))
        fixture = _without_volatile(_load(FIXTURE))
        fixture["captions"].pop("source")
        shipped = _without_volatile(shipped)
        shipped["captions"].pop("source")
        self.assertEqual(fixture, shipped)
        self.assertEqual(kit.validate_manifest(_load(FIXTURE)), [])

    def test_load_manifest_returns_manifest_and_file_sha(self):
        manifest, sha = kit.load_manifest(EXAMPLE)
        self.assertEqual(manifest["kit"]["id"], "example")
        self.assertEqual(sha, file_sha256(kit.manifest_path(EXAMPLE)))
        self.assertEqual(kit.manifest_path(EXAMPLE), EXAMPLE / "manifest.json")


class BuildTests(TempKit):
    def test_adapter_build_fills_sources_and_fingerprint(self):
        (self.kit / "manifest.json").unlink()
        code, out, err = run_cli(kit.cli_build, self.kit, self.adapter)
        self.assertEqual(code, 0, out + err)
        manifest, sha = kit.load_manifest(self.kit)
        self.assertEqual(sha, file_sha256(kit.manifest_path(self.kit)))
        self.assertEqual(sorted(manifest["sources"]), ["tokens.css", "use_when.json"])
        for name in manifest["sources"]:
            self.assertEqual(manifest["sources"][name], file_sha256(self.kit / name))
        self.assertRegex(manifest["kit"]["fingerprint"], r"^[0-9a-f]{64}$")
        self.assertRegex(manifest["kit"]["built_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual(_without_volatile(manifest), _without_volatile(_load(EXAMPLE / "manifest.json")))
        run_cli(kit.cli_build, self.kit, self.adapter)
        self.assertEqual(kit.load_manifest(self.kit)[0]["kit"]["fingerprint"], manifest["kit"]["fingerprint"])

    def test_stamp_without_adapter_keeps_the_hand_written_values(self):
        manifest = _load(self.kit / "manifest.json")
        manifest["sources"] = {"tokens.css": "0" * 64}
        manifest["rules"]["punch_max"] = 1.2                        # a hand edit the stamp must keep
        (self.kit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, out, err = run_cli(kit.cli_build, self.kit)
        self.assertEqual(code, 0, out + err)
        stamped = _load(self.kit / "manifest.json")
        self.assertEqual(stamped["rules"]["punch_max"], 1.2)
        self.assertEqual(stamped["sources"], {"tokens.css": file_sha256(self.kit / "tokens.css"),
                                              "use_when.json": file_sha256(self.kit / "use_when.json")})
        self.assertEqual(run_cli(kit.cli_check, self.kit)[0], 0)

    def test_fingerprint_changes_when_a_source_changes(self):
        run_cli(kit.cli_build, self.kit, self.adapter)
        before = kit.load_manifest(self.kit)[0]["kit"]["fingerprint"]
        with open(self.kit / "use_when.json", "ab") as f:
            f.write(b"\n")
        run_cli(kit.cli_build, self.kit, self.adapter)
        self.assertNotEqual(kit.load_manifest(self.kit)[0]["kit"]["fingerprint"], before)

    def test_build_exits_2_when_the_adapter_fails_and_writes_nothing(self):
        css = self.kit / "tokens.css"
        css.write_text(css.read_text(encoding="utf-8").replace("--safe-top: 220px;", ""), encoding="utf-8")
        before = (self.kit / "manifest.json").read_bytes()
        code, _, err = run_cli(kit.cli_build, self.kit, self.adapter)
        self.assertEqual(code, 2)
        self.assertIn("--safe-top", err)
        self.assertEqual((self.kit / "manifest.json").read_bytes(), before)

    def test_build_exits_2_when_the_result_fails_the_schema(self):
        manifest = _load(self.kit / "manifest.json")
        manifest["captions"]["max_lines"] = 0
        (self.kit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, _, err = run_cli(kit.cli_build, self.kit)
        self.assertEqual(code, 2)
        self.assertIn("$.captions.max_lines", err)
        self.assertEqual(_load(self.kit / "manifest.json")["kit"]["fingerprint"], manifest["kit"]["fingerprint"])

    def test_bad_adapters_exit_2(self):
        for name, body in (("broken.py", "raise RuntimeError('boom')\n"), ("empty.py", "X = 1\n")):
            path = self.kit / name
            path.write_text(body, encoding="utf-8")
            code, _, err = run_cli(kit.cli_build, self.kit, path)
            self.assertEqual(code, 2, name)
            self.assertTrue(err.startswith("kit build:"), err)
        self.assertEqual(run_cli(kit.cli_build, self.kit, self.kit / "nope.py")[0], 2)

    def test_a_source_outside_the_kit_folder_is_refused(self):
        for key in ("../outside.txt", "/etc/hosts"):
            manifest = _load(EXAMPLE / "manifest.json")
            manifest["sources"] = {key: "0" * 64}
            (self.kit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            code, _, err = run_cli(kit.cli_build, self.kit)
            self.assertEqual(code, 2, key)
            self.assertIn("inside the kit folder", err)
            code, out, _ = run_cli(kit.cli_check, self.kit)
            self.assertEqual(code, 1, key)
            self.assertIn("inside the kit folder", out)

    def test_no_kit_folder_exits_2(self):
        with unittest.mock.patch.dict("os.environ", {kit.KIT_DIR_ENV: ""}):
            for fn in (kit.cli_build, kit.cli_check):
                code, _, err = run_cli(fn)
                self.assertEqual(code, 2)
                self.assertIn("--kit-dir", err)


class CheckTests(TempKit):
    def test_check_fresh_then_stale(self):
        code, out, _ = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 0, out)
        with open(self.kit / "tokens.css", "ab") as f:
            f.write(b" ")
        code, out, _ = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 1)
        self.assertIn("tokens.css", out)
        self.assertNotIn("use_when.json", out)
        self.assertIn("kit build --kit-dir", out)

    def test_check_names_a_changed_use_when(self):
        with open(self.kit / "use_when.json", "ab") as f:
            f.write(b"\n")
        code, out, _ = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 1)
        self.assertIn("use_when.json", out)

    def test_check_names_an_unrecorded_use_when_and_a_missing_source(self):
        manifest = _load(self.kit / "manifest.json")
        del manifest["sources"]["use_when.json"]
        (self.kit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (self.kit / "tokens.css").unlink()
        code, out, _ = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 1)
        self.assertIn("use_when.json (not in the manifest)", out)
        self.assertIn("tokens.css (missing)", out)

    def test_check_reports_schema_errors(self):
        manifest = _load(self.kit / "manifest.json")
        del manifest["qa_context"]
        manifest["emphasis"]["words"]["value"] = "blue"
        (self.kit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, out, _ = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 1)
        self.assertIn("$.qa_context: required property is missing", out)
        self.assertIn("$.emphasis.words.value", out)

    def test_check_without_a_manifest_exits_2(self):
        (self.kit / "manifest.json").unlink()
        code, _, err = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 2)
        self.assertIn("no manifest", err)
        (self.kit / "manifest.json").write_text("{not json", encoding="utf-8")
        code, _, err = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 2)
        self.assertIn("cannot read it as JSON", err)

    def test_check_flags_a_fingerprint_that_does_not_match_the_sources(self):
        manifest = _load(self.kit / "manifest.json")
        manifest["kit"]["fingerprint"] = "f" * 64
        (self.kit / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, out, _ = run_cli(kit.cli_check, self.kit)
        self.assertEqual(code, 1)
        self.assertIn("kit.fingerprint", out)


if __name__ == "__main__":
    unittest.main()
