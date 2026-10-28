"""Kit manifests: `jevm kit build` stamps a kit's `manifest.json`, `jevm kit check` validates it and reports drift.

A kit is a folder you own, anywhere on disk. jevm ships no brand kit; `examples/kit/` is a neutral example.

    <kit-dir>/manifest.json   required: what jevm reads (canvases, caption limits, highlight rules, QA context).
    <kit-dir>/use_when.json   optional: hand-kept facts the kit's own files do not state. Hashed as a source; passed
                              to an adapter when one builds the manifest.

Which folder: `--kit-dir` on the command line, else the folder of the spec's `kit.manifest`, else `JEVM_KIT_DIR`.

`sources` maps each file the manifest was built from (a path inside the kit folder) to its sha256, and
`kit.fingerprint` is the sha256 of those pairs, sorted. `kit build` writes both, plus `kit.built_at`: on the manifest
you wrote by hand, or on the one an adapter returns (`--adapter FILE.py`: a module with `SOURCE_FILES`, the kit files it
reads, and `build(kit_dir, use_when) -> manifest`). `kit check` validates the manifest against
`kits/manifest.schema.json` and re-hashes every source. Nothing here touches the network.
"""
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from jevlib.cache import file_sha256
from jevlib.paths import KITS_DIR

KIT_DIR_ENV = "JEVM_KIT_DIR"
MANIFEST = "manifest.json"
USE_WHEN = "use_when.json"
SCHEMA_PATH = KITS_DIR / "manifest.schema.json"


class KitError(ValueError):
    """No kit folder, a missing or unreadable manifest, an unusable adapter, or a source path outside the kit folder."""


def resolve_manifest(flag=None, spec=None, environ=None) -> Path:
    """The kit manifest file: `<flag>/manifest.json` (`--kit-dir`), else the spec's `kit.manifest` (absolute, or relative
    to the current directory), else `<JEVM_KIT_DIR>/manifest.json`. Raises `KitError` when none of them names one."""
    if flag:
        return Path(flag).expanduser() / MANIFEST
    manifest = ((spec or {}).get("kit") or {}).get("manifest") if isinstance(spec, dict) else None
    if manifest:
        path = Path(manifest).expanduser()
        return path if path.is_absolute() else Path.cwd() / path
    environ = os.environ if environ is None else environ
    env = (environ.get(KIT_DIR_ENV) or "").strip()
    if env:
        return Path(env).expanduser() / MANIFEST
    raise KitError(f"no kit folder: pass --kit-dir PATH or set {KIT_DIR_ENV} (see examples/kit for the format)")


def resolve_kit_dir(flag=None, spec=None, environ=None) -> Path:
    """The kit folder, by the same precedence as `resolve_manifest`: `flag`, else the folder of the spec's
    `kit.manifest`, else `JEVM_KIT_DIR`. Raises `KitError` when none of them names one."""
    return resolve_manifest(flag, spec, environ).parent


def manifest_path(kit_dir) -> Path:
    return Path(kit_dir) / MANIFEST


def load_manifest(kit_dir) -> tuple[dict, str]:
    """The manifest in `kit_dir` and the sha256 of its file. Raises `KitError` when it is missing or not JSON."""
    path = manifest_path(kit_dir)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise KitError(f"{path}: no manifest; write one (see examples/kit) or run `jevm kit build --adapter FILE.py`") from exc
    except (OSError, ValueError) as exc:
        raise KitError(f"{path}: cannot read it as JSON ({exc})") from exc
    if not isinstance(manifest, dict):
        raise KitError(f"{path}: not a JSON object")
    return manifest, file_sha256(path)


def validate_manifest(manifest) -> list[str]:
    """Schema errors for `manifest` against `kits/manifest.schema.json`; an empty list means valid."""
    from spec.validate import validate_schema
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return validate_schema(manifest, schema)


def _source_file(kit_dir: Path, key: str) -> Path:
    """The file behind a `sources` key. A key is a relative POSIX path inside the kit folder; anything else is refused,
    so a manifest can never make jevm hash a file outside its own kit."""
    rel = PurePosixPath(key)
    if not key or rel.is_absolute() or ".." in rel.parts or "\\" in key:
        raise KitError(f"sources key {key!r}: must be a relative path inside the kit folder")
    return Path(kit_dir) / Path(*rel.parts)


def _fingerprint(sources: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(sorted(sources.items()), separators=(",", ":")).encode("utf-8")).hexdigest()


def load_adapter(path):
    """The adapter module at `path`. It must define `SOURCE_FILES` (paths inside the kit folder) and
    `build(kit_dir, use_when) -> manifest`. Raises `KitError` otherwise."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise KitError(f"adapter not found: {path}")
    spec = importlib.util.spec_from_file_location(f"jevm_kit_adapter_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise KitError(f"{path}: cannot load it as a Python module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                                     # an adapter is user code: report, never crash
        raise KitError(f"{path}: the adapter failed to load ({type(exc).__name__}: {exc})") from exc
    if not callable(getattr(module, "build", None)) or not isinstance(getattr(module, "SOURCE_FILES", None), (list, tuple)):
        raise KitError(f"{path}: an adapter needs SOURCE_FILES (a list of paths) and build(kit_dir, use_when)")
    return module


def _read_use_when(kit_dir: Path):
    """The parsed `use_when.json`, or None when the kit has none."""
    path = kit_dir / USE_WHEN
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KitError(f"{USE_WHEN}: cannot read it as JSON ({exc})") from exc


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build(kit_dir, adapter=None, now=None) -> dict:
    """The stamped manifest for `kit_dir` (not written). With `adapter` (a module from `load_adapter`) the manifest is
    the adapter's and the sources are its `SOURCE_FILES`; without one it is the manifest already in the folder and the
    sources are the keys it records. `use_when.json` joins the sources when it exists. Raises `KitError`."""
    kit_dir = Path(kit_dir)
    use_when = _read_use_when(kit_dir)
    if adapter is not None:
        try:
            manifest = adapter.build(kit_dir, use_when if use_when is not None else {})
        except KitError:
            raise
        except Exception as exc:                                 # user code: name the failure, never crash
            raise KitError(f"the adapter's build() failed ({type(exc).__name__}: {exc})") from exc
        keys = [str(k) for k in adapter.SOURCE_FILES]
    else:
        manifest, _ = load_manifest(kit_dir)
        keys = [k for k in (manifest.get("sources") or {}) if k != USE_WHEN]
    if not isinstance(manifest, dict) or not isinstance(manifest.get("kit"), dict):
        raise KitError("the manifest has no `kit` block")
    if use_when is not None:
        keys.append(USE_WHEN)
    sources = {}
    for key in dict.fromkeys(keys):
        path = _source_file(kit_dir, key)
        try:
            sources[key] = file_sha256(path)
        except OSError as exc:
            raise KitError(f"source {key}: cannot read {path} ({exc.strerror or exc})") from exc
    manifest["sources"] = sources
    manifest["kit"]["fingerprint"] = _fingerprint(sources)
    manifest["kit"]["built_at"] = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return manifest


def cli_build(kit_dir=None, adapter=None) -> int:
    """`jevm kit build`. Exit 0 when written, 2 when there is no kit folder, the adapter or a source cannot be read, or the
    result fails the schema (the old manifest, if any, is left as it was)."""
    try:
        kit_dir = resolve_kit_dir(kit_dir)
        module = load_adapter(adapter) if adapter else None
        manifest = build(kit_dir, module)
    except KitError as exc:
        print(f"kit build: {exc}", file=sys.stderr)
        return 2
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"kit build: {error}", file=sys.stderr)
        print(f"kit build: the manifest fails {SCHEMA_PATH.name} ({len(errors)} error(s)); nothing written", file=sys.stderr)
        return 2
    path = manifest_path(kit_dir)
    try:
        _write_atomic(path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"kit build: cannot write {path} ({exc.strerror or exc})", file=sys.stderr)
        return 2
    print(f"kit build: wrote {path} (kit {manifest['kit'].get('id')}, {len(manifest['sources'])} source(s), "
          f"fingerprint {manifest['kit']['fingerprint'][:12]})")
    return 0


def cli_check(kit_dir=None) -> int:
    """`jevm kit check`. Exit 2 when there is no readable manifest; 1 when it fails the schema or a source changed, went
    missing or is not recorded (`use_when.json`); else 0 with a one-line summary."""
    try:
        kit_dir = resolve_kit_dir(kit_dir)
        manifest, _ = load_manifest(kit_dir)
    except KitError as exc:
        print(f"kit check: {exc}", file=sys.stderr)
        return 2
    errors = validate_manifest(manifest)
    if errors:
        print(f"kit check: {manifest_path(kit_dir)} fails {SCHEMA_PATH.name}:")
        for error in errors:
            print(f"  {error}")
        return 1
    kit_id = manifest["kit"]["id"]
    built = manifest["sources"]
    extra = [USE_WHEN] if (kit_dir / USE_WHEN).is_file() else []
    changed = []
    for key in dict.fromkeys((*built, *extra)):
        try:
            now = file_sha256(_source_file(kit_dir, key))
        except KitError as exc:
            changed.append(f"{key} ({exc})")
            continue
        except OSError:
            changed.append(f"{key} (missing)")
            continue
        if key not in built:
            changed.append(f"{key} (not in the manifest)")
        elif now != built[key]:
            changed.append(key)
    if built and manifest["kit"]["fingerprint"] != _fingerprint(built):
        changed.append("kit.fingerprint (does not match sources)")
    if changed:
        print(f"kit check: the {kit_id} manifest is stale; run `jevm kit build --kit-dir {kit_dir}`. Changed:")
        for name in changed:
            print(f"  {name}")
        return 1
    print(f"kit check: {kit_id} at {kit_dir}: manifest valid, {len(built)} source(s) unchanged")
    return 0
