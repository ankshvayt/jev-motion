"""Where things live. Tool folders are derived from this file's position, so the tool runs from any directory.

Nothing here assumes a surrounding workspace: a project root in a spec is absolute (what `jevm spec build` writes) or
relative to the current working directory, and the answer cache goes where `cache_dir` says.
"""
import os
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parent.parent
QSETS_DIR = TOOL_ROOT / "qsets"
POLICIES_DIR = TOOL_ROOT / "policies"
EVALSETS_DIR = TOOL_ROOT / "evalsets"
KITS_DIR = TOOL_ROOT / "kits"
EVAL_OUT_DIR = TOOL_ROOT / "eval"
CACHE_DIRNAME = ".jevm-cache"


def project_root(spec):
    """`spec["project"]["root"]`: absolute, or relative to the current working directory. The path may contain spaces."""
    root = Path(spec["project"]["root"]).expanduser()
    return root if root.is_absolute() else Path.cwd() / root


def work_dir(spec):
    """Where jevm writes frames, overlays and qa files for one project. Not created here."""
    return project_root(spec) / "work" / "jevm"


def cache_dir(spec=None, environ=None):
    """The answer cache folder: `JEVM_CACHE_DIR` when set; else `<project>/work/jevm/.jevm-cache/` for a command that has
    a spec (`plan`, `crop`, `qa`); else `.jevm-cache/` in the current working directory (`eval`). Not created here."""
    environ = os.environ if environ is None else environ
    override = (environ.get("JEVM_CACHE_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    if spec is not None:
        return work_dir(spec) / CACHE_DIRNAME
    return Path.cwd() / CACHE_DIRNAME
