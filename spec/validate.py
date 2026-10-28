"""`jevm spec validate`: a small JSON Schema validator plus the checks a schema cannot express.

The validator supports `type` (one name or a list), `required`, `properties`, `additionalProperties`, `items`, `enum`,
`minItems`, `maxItems`, `minimum`, `maximum` and `pattern`; any other keyword in `schema.json` is ignored. Errors are
strings that start with the JSON path of the offending value (`$.assets[1].sensitivity: ...`).
"""
import json
import re
import sys
from pathlib import Path

from jevlib.cache import file_sha256

SCHEMA_PATH = Path(__file__).resolve().with_name("schema.json")


def _is_type(value, name):
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, float) and value.is_integer())
    raise ValueError(f"schema uses an unsupported type {name!r}")


def _type_name(value):
    if value is None:
        return "null"
    for name in ("boolean", "integer", "number", "string", "array", "object"):
        if _is_type(value, name):
            return name
    return type(value).__name__


def _check(value, schema, path, errors):
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else types
        if not any(_is_type(value, n) for n in names):
            errors.append(f"{path}: expected {' or '.join(names)}, got {_type_name(value)}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {json.dumps(value, ensure_ascii=False)} is not one of {json.dumps(schema['enum'])}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
    if isinstance(value, str) and "pattern" in schema and not re.search(schema["pattern"], value):
        errors.append(f"{path}: {json.dumps(value, ensure_ascii=False)} does not match {schema['pattern']}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: {len(value)} items, at least {schema['minItems']} required")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: {len(value)} items, at most {schema['maxItems']} allowed")
        if "items" in schema:
            for i, item in enumerate(value):
                _check(item, schema["items"], f"{path}[{i}]", errors)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required property is missing")
        for key, sub in properties.items():
            if key in value:
                _check(value[key], sub, f"{path}.{key}", errors)
        extra = schema.get("additionalProperties", True)
        for key in value:
            if key in properties:
                continue
            if extra is False:
                errors.append(f"{path}.{key}: additional property is not allowed")
            elif isinstance(extra, dict):
                _check(value[key], extra, f"{path}.{key}", errors)


def validate_schema(instance, schema):
    """Errors for `instance` against `schema` (the subset above); an empty list means valid."""
    errors = []
    _check(instance, schema, "$", errors)
    return errors


def _unique_index(items, path, errors):
    """Map id -> position, reporting every repeated id."""
    index = {}
    for i, item in enumerate(items):
        if item["id"] in index:
            errors.append(f"{path}[{i}].id: duplicate id {json.dumps(item['id'])}")
        else:
            index[item["id"]] = i
    return index


def _check_span(path, pair, word_index, errors):
    """`pair` is `[first, last]` word ids: both must exist and `first` must not come after `last`.
    Returns the number of words the span covers, or None when the span is broken."""
    missing = [ref for ref in pair if ref not in word_index]
    for ref in missing:
        errors.append(f"{path}: unknown word id {json.dumps(ref)}")
    if missing:
        return None
    if word_index[pair[0]] > word_index[pair[1]]:
        errors.append(f"{path}: {pair[0]} comes after {pair[1]}")
        return None
    return word_index[pair[1]] - word_index[pair[0]] + 1


def _manifest_file(spec, manifest_path):
    if manifest_path is not None:
        return Path(manifest_path)
    path = Path(spec["kit"]["manifest"]).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def validate(spec, *, manifest_path=None):
    """Schema errors, or, when the schema passes, the semantic errors: unique ids, `[first, last]` refs that exist and are
    ordered, caption groups that name a real sentence and stay within `constraints.captions` (at most
    `max_words_per_line * max_lines` words and `max_lines` lines), and the manifest hash when the manifest file exists.
    `manifest_path` overrides the path stored in `spec["kit"]["manifest"]`."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = validate_schema(spec, schema)
    if errors:
        return errors
    word_index = _unique_index(spec["transcript"]["words"], "$.transcript.words", errors)
    sentence_index = _unique_index(spec["units"]["sentences"], "$.units.sentences", errors)
    _unique_index(spec["units"]["caption_groups"], "$.units.caption_groups", errors)
    _unique_index(spec["assets"], "$.assets", errors)
    for i, sentence in enumerate(spec["units"]["sentences"]):
        _check_span(f"$.units.sentences[{i}].words", sentence["words"], word_index, errors)
    captions = spec["constraints"]["captions"]
    per_line, max_lines = captions["max_words_per_line"], captions["max_lines"]
    for i, group in enumerate(spec["units"]["caption_groups"]):
        size = _check_span(f"$.units.caption_groups[{i}].words", group["words"], word_index, errors)
        if size is not None and size > per_line * max_lines:
            errors.append(f"$.units.caption_groups[{i}].words: {size} words exceed the caption cap of "
                          f"{per_line * max_lines} (max_words_per_line {per_line} x max_lines {max_lines})")
        if group["lines"] > max_lines:
            errors.append(f"$.units.caption_groups[{i}].lines: {group['lines']} lines exceed max_lines {max_lines}")
        if group["sentence"] not in sentence_index:
            errors.append(f"$.units.caption_groups[{i}].sentence: unknown sentence id {json.dumps(group['sentence'])}")
    manifest = _manifest_file(spec, manifest_path)
    if manifest.is_file():
        actual = file_sha256(manifest)
        if actual != spec["kit"]["manifest_sha256"]:
            errors.append(f"$.kit.manifest_sha256: the spec has {spec['kit']['manifest_sha256'][:12]} but {manifest.name} "
                          f"hashes to {actual[:12]}; rebuild the spec")
    return errors


def cli_validate(path, kit_dir=None):
    """Exit 0 valid, 1 invalid (problems on stderr), 2 when the file cannot be read as JSON. With `kit_dir`
    (`--kit-dir`) the manifest hash is checked against `<kit_dir>/manifest.json` instead of the spec's `kit.manifest`."""
    try:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"jevm spec validate: cannot read {path}: {e}", file=sys.stderr)
        return 2
    errors = validate(spec, manifest_path=Path(kit_dir) / "manifest.json" if kit_dir else None)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"invalid: {len(errors)} error(s) in {path}", file=sys.stderr)
        return 1
    print(f"valid: {path}")
    return 0
