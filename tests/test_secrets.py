"""No key in anything this tool holds or writes (stdlib unittest). From the tool folder:

    python3 -m unittest tests.test_secrets -q

Three scans over the whole tool tree (code, fixtures, eval sets, examples, README, and any cache or `eval/` folder that
exists; `.git/` is skipped) plus every folder in the JEVM_SECRET_SCAN environment variable (split on os.pathsep):

* the JEV_API value itself, which `load_key()` reads from the environment or a `.env` in the current directory: the
  test is skipped when there is no key. The key is never printed, and a failure names only `file:line`.
* an `Authorization: Bearer <value>` literal, a vendor-prefixed key, a key-named assignment to a long literal, and a long
  random-looking token. Strings the tests plant on purpose to prove redaction are listed in SYNTHETIC; anything else new
  fails, and the report is `file:line (rule)`, never the text.
* a throwaway project run through the command line, and what it printed.
"""
import math
import os
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from jevlib.env import load_key
from jevlib.paths import TOOL_ROOT
from tests.test_integration import Project

# A made-up string tests/test_backends.py plants to prove redaction. It is not a credential; it is what the scan would
# otherwise flag, and a piece of it counts as the same string.
SYNTHETIC = frozenset({"sk-jev-test-0123456789abcdef"})
PLACEHOLDERS = frozenset({"REDACTED"})

TOKEN_CHARS = r"[A-Za-z0-9._~+/=-]"
SHAPES = (
    ("bearer literal", re.compile(r"(?i)\bbearer[ \t]+(" + TOKEN_CHARS + r"{8,})")),
    ("vendor-prefixed key", re.compile(
        r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}|\b(?:xox[abpr]|gh[pousr])[-_][A-Za-z0-9_-]{16,}"
        r"|\bAKIA[0-9A-Z]{12,}|\bAIza[0-9A-Za-z_-]{20,}")),
    ("key-named assignment", re.compile(
        r"(?i)(?:api[_-]?key|secret|token|passw(?:or)?d|authorization)[\"']?[ \t]*[:=][ \t]*[\"']("
        + TOKEN_CHARS + r"{16,})[\"']")),
    ("long random token", re.compile(r"(?<![A-Za-z0-9_])[A-Za-z0-9_]{40,}(?![A-Za-z0-9_])")),
)


def entropy(text):
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def random_looking(token):
    """Long, not a hex digest (those are everywhere in the fixtures), with digits and letters mixed and some spread."""
    return (not re.fullmatch(r"[0-9a-fA-F]+", token) and sum(c.isdigit() for c in token) >= 6
            and sum(c.isalpha() for c in token) >= 6 and entropy(token) >= 3.5)


def scan_roots():
    extra = [Path(p) for p in os.environ.get("JEVM_SECRET_SCAN", "").split(os.pathsep) if p]
    return [TOOL_ROOT, *extra]


def files_under(roots):
    """`(root, path)` for every regular file under each root, symbolic links not followed and `.git/` skipped."""
    for root in roots:
        for here, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != ".git")
            for name in sorted(names):
                path = Path(here) / name
                if path.is_file() and not path.is_symlink():
                    yield root, path


def label(root, path, line):
    return f"{Path(root).name}/{path.relative_to(root)}:{line}"


def key_hits(roots, needle):
    """`root-name/relative/path:line` for every place the bytes of `needle` appear. Unreadable files are reported too,
    because their content is unproven. The key itself is never part of a result."""
    hits = []
    for root, path in files_under(roots):
        try:
            data = path.read_bytes()
        except OSError:
            hits.append(f"{label(root, path, 0)} (unreadable)")
            continue
        at = data.find(needle)
        while at != -1:
            hits.append(label(root, path, data.count(b"\n", 0, at) + 1))
            at = data.find(needle, at + 1)
    return hits


def shape_hits(roots):
    """`root-name/relative/path:line (rule)` for key-shaped text in every text file. Binary files are skipped."""
    hits = []
    for root, path in files_under(roots):
        try:
            data = path.read_bytes()
        except OSError:
            hits.append(f"{label(root, path, 0)} (unreadable)")
            continue
        if b"\0" in data[:8192]:
            continue
        for number, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for rule, pattern in SHAPES:
                for m in pattern.finditer(line):
                    value = m.group(1) if pattern.groups else m.group(0)
                    if value in PLACEHOLDERS or any(piece in known for piece in (value, m.group(0)) for known in SYNTHETIC):
                        continue
                    if rule == "long random token" and not random_looking(value):
                        continue
                    hits.append(f"{label(root, path, number)} ({rule})")
    return hits


class TreeScanTests(unittest.TestCase):
    def test_no_key_value_in_files(self):
        key = load_key()
        if not key:
            self.skipTest("no JEV_API key in the environment or a .env file")
        hits = key_hits(scan_roots(), key.encode("utf-8"))
        self.assertEqual(hits, [], "the Jev key value is in these files (file:line; the value is not shown)")

    def test_no_key_shaped_text_in_files(self):
        hits = shape_hits(scan_roots())
        self.assertEqual(hits, [], "key-shaped text in these places (file:line and rule; the text is not shown)")

    def test_the_scan_walks_the_whole_tool_tree(self):
        seen = {str(path.relative_to(root)) for root, path in files_under([TOOL_ROOT])}
        for expected in ("jevm.py", "README.md", "qsets/stat_check@1.json", "tests/test_secrets.py",
                         "tests/fixtures/spec_min.json", "examples/kit/manifest.json", "evalsets/crops/manifest.json"):
            self.assertIn(expected, seen)
        self.assertFalse(any(name.startswith(".git/") for name in seen))
        self.assertGreater(len(seen), 80)


class ScannerSelfTests(unittest.TestCase):
    """The scanner must find what it is for and say only where, or the two tests above could pass for nothing."""

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.root = Path(td.name) / "scan root"
        (self.root / "sub dir").mkdir(parents=True)

    def test_key_value_is_found_and_only_its_place_is_reported(self):
        planted = "zz_planted_not_a_real_key_0123456789"
        (self.root / "sub dir" / "notes.txt").write_text(f"first\nsecond {planted} third\n{planted}\n", encoding="utf-8")
        (self.root / "clean.txt").write_text("nothing here\n", encoding="utf-8")
        (self.root / "blob.bin").write_bytes(b"\x00\x01" + planted.encode() + b"\x02")
        hits = key_hits([self.root], planted.encode())
        self.assertEqual(hits, ["scan root/blob.bin:1", "scan root/sub dir/notes.txt:2", "scan root/sub dir/notes.txt:3"])
        self.assertNotIn("planted", "\n".join(hits))
        self.assertEqual(key_hits([self.root], b"absent-needle"), [])

    def test_key_shaped_text_is_found_and_only_its_place_is_reported(self):
        # Each planted value is joined from two pieces so this file never holds one whole, or the tree scan would flag it.
        bearer, vendor = "abcdefgh" + "12345678", "abcdefghijklmnop" + "1234"
        named, long_token = "abcd1234" + "efgh5678ijkl", "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9a" + "B3dE5gH7jK9mN1"
        lines = [
            "Authorization: Bearer " + bearer,                               # bearer literal with a value
            "headers = {'Authorization': 'Bearer ' + key}",                  # built at run time: no literal
            "token = sk-" + "live-" + vendor,                                # vendor-prefixed
            "api_key = '" + named + "'",                                     # key-named assignment
            "x = '" + long_token + "'",                                      # long random token
            "digest = '" + "0123456789abcdef" * 4 + "'",                     # a hex digest is not a key
            "name = 'test_a_long_but_plain_identifier_made_of_words_only'",  # nor is a long identifier
            "Authorization: Bearer REDACTED",                                # a placeholder
            "redaction test uses " + sorted(SYNTHETIC)[0],                   # planted on purpose
        ]
        (self.root / "sub dir" / "code.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
        hits = shape_hits([self.root])
        self.assertEqual(hits, ["scan root/sub dir/code.py:1 (bearer literal)",
                                "scan root/sub dir/code.py:3 (vendor-prefixed key)",
                                "scan root/sub dir/code.py:4 (key-named assignment)",
                                "scan root/sub dir/code.py:5 (long random token)"])
        for secret in (bearer, vendor, named, long_token[:12]):
            self.assertNotIn(secret, "\n".join(hits))

    def test_a_symbolic_link_is_not_followed(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("Authorization: Bearer " + "abcdefgh" + "12345678\n", encoding="utf-8")
        (self.root / "link.txt").symlink_to(outside)
        self.assertEqual(shape_hits([self.root]), [])


class RunLeavesNoKeyTests(unittest.TestCase):
    def test_a_run_prints_and_writes_no_key(self):
        key = load_key()
        if not key:
            self.skipTest("no JEV_API key in the environment or a .env file")
        with tempfile.TemporaryDirectory() as td:
            project = Project(td)
            project.build_spec()
            self.assertEqual(project.run("plan", project.spec).returncode, 0)
            self.assertEqual(project.run("kit", "check").returncode, 0)
            self.assertTrue((project.work / "decisions.json").is_file())
            self.assertEqual(key_hits([project.tmp], key.encode("utf-8")), [])
            self.assertEqual([i for i, text in enumerate(project.output) if key in text], [],
                             "the key is in the output of these commands (index shown; the value is not)")
            self.assertEqual(shape_hits([project.tmp]), [])


if __name__ == "__main__":
    unittest.main()
