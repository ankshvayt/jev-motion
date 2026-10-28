"""`jevm spec build`: words JSON + asset list + a format id + a kit -> `spec.json`.

Pure except for reading its inputs and, in `cli_build`, writing the spec. Nothing here calls a backend. Every number that
describes the kit (canvas size, safe areas, caption limits, highlight rules) is copied from the kit manifest, never typed
here; only the chunking thresholds, the budgets and the backend pins below are this module's own.
"""
import copy
import json
import math
import re
import sys
from pathlib import Path

from core.policy import load_policy
from jevlib.cache import file_sha256
from jevlib import tokens
from jevlib.paths import TOOL_ROOT, work_dir
from kits.kit import manifest_path as kit_manifest_path, resolve_manifest
from spec.validate import validate

FORMATS_PATH = TOOL_ROOT / "formats" / "formats.json"

FPS = 30
BUDGET_USD = {"short": 0.10, "long": 1.0}  # default Jev budget per video, by format family
# `decide.backends`: the pinned Jev models and the local Decider defaults.
BACKENDS = {
    "jev": {"models": {"text": "jev-1.13.0", "image": "jev-1.14.0"}, "concurrency": 6},
    "local": {"name": "Decider (Perplexity, local)", "url": "http://127.0.0.1:8080",
              "model": "pplx-decider-flash-v1.2", "concurrency": 1},
}

# CAP-01 chunking (code only). A mark counts after any closing quote or bracket that follows it.
SENTENCE_END = (".", "?", "!", "…")
SOFT_END = (",", ";", ":", "—")
CLOSERS = "\"')]}”’»"
SENTENCE_GAP_S = 0.60
CAPTION_GAP_S = 0.30
SPLIT_GAP_S = 0.15  # a silence at least this long is a natural place to break a two-line caption
EPS = 1e-9


def _ends_with(text, marks):
    return text.rstrip(CLOSERS).endswith(marks)


def _gap(a, b):
    return b["start"] - a["end"]


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _abs(path):
    """`path` resolved to an absolute path string (a spec stores absolute paths, so it reads the same from any folder)."""
    return str(Path(path).expanduser().resolve())


def _slug(name):
    """A project id from the folder name: lower case, every run of characters outside `a-z0-9._-` becomes one `-`."""
    return re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-._") or "project"


def load_words(path):
    """Read words from a list or `{"words": [...]}`. Entries with empty `text` (or `word`) are dropped, ids become
    `w0, w1, …`, and only `text`, `start`, `end` and, when present, `speaker` and `conf` are kept."""
    data = _load_json(path)
    if isinstance(data, dict):
        data = data.get("words")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of words or an object with a \"words\" list")
    words = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {i} is not an object")
        text = str(entry.get("text") or entry.get("word") or "").strip()
        if not text:
            continue
        try:
            start, end = float(entry["start"]), float(entry["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{path}: word {text!r} (entry {i}) needs a numeric start and end") from None
        word = {"id": f"w{len(words)}", "text": text, "start": start, "end": end}
        for key in ("speaker", "conf"):
            if entry.get(key) is not None:
                word[key] = entry[key]
        words.append(word)
    return words


def sentences(words):
    """A sentence ends after a word ending in `.`, `?`, `!` or `…`, or before a silence of at least 0.60 s."""
    out, first = [], 0
    for i, word in enumerate(words):
        if (i == len(words) - 1 or _ends_with(word["text"], SENTENCE_END)
                or _gap(word, words[i + 1]) >= SENTENCE_GAP_S - EPS):
            out.append({"id": f"sn{len(out)}", "words": [words[first]["id"], word["id"]],
                        "start": words[first]["start"], "end": word["end"]})
            first = i + 1
    return out


def _split(chunk, per_line, limit):
    """Break a run of words into caption groups. A run that fits one line stays whole. A longer run stays whole as a
    two-line group only when it is within `limit` words and has no silence of `SPLIT_GAP_S` inside; otherwise it is cut at
    its largest internal gap (a tie goes to the cut nearest the middle, then the earlier one) and both halves are split
    again. So no group ever holds more than `limit` words."""
    n = len(chunk)
    if n <= per_line:
        return [chunk]
    gaps = [_gap(chunk[i], chunk[i + 1]) for i in range(n - 1)]
    widest = max(gaps)
    if n <= limit and widest < SPLIT_GAP_S - EPS:
        return [chunk]
    ties = [p for p, g in enumerate(gaps) if g >= widest - EPS]
    cut = min(ties, key=lambda p: (abs(p + 1 - n / 2), p))
    return _split(chunk[:cut + 1], per_line, limit) + _split(chunk[cut + 1:], per_line, limit)


def caption_groups(words, sentences, max_words_per_line, max_lines):
    """The one or two lines on screen together; the size comes from the kit (`captions` in the manifest), never from here.
    Runs of words close after a word that ends a sentence, before a silence of at least 0.30 s, or after a word ending in
    `,`, `;`, `:` or `—` once the run holds `max_words_per_line` words. Each run is then cut by `_split`, so a group holds
    at most `max_words_per_line * max_lines` words and `lines = min(max_lines, ceil(n / max_words_per_line))`."""
    limit = max_words_per_line * max_lines
    index = {w["id"]: i for i, w in enumerate(words)}
    sentence_of, sentence_ends = {}, set()
    for s in sentences:
        a, b = index[s["words"][0]], index[s["words"][1]]
        sentence_ends.add(b)
        for i in range(a, b + 1):
            sentence_of[i] = s["id"]
    out, first = [], 0
    for i, word in enumerate(words):
        count = i - first + 1
        if (i in sentence_ends or i == len(words) - 1 or _gap(word, words[i + 1]) >= CAPTION_GAP_S - EPS
                or (_ends_with(word["text"], SOFT_END) and count >= max_words_per_line)):
            for piece in _split(words[first:i + 1], max_words_per_line, limit):
                out.append({"id": f"cg{len(out)}", "words": [piece[0]["id"], piece[-1]["id"]],
                            "start": piece[0]["start"], "end": piece[-1]["end"],
                            "lines": min(max_lines, math.ceil(len(piece) / max_words_per_line)),
                            "sentence": sentence_of[index[piece[0]["id"]]]})
            first = i + 1
    return out


def load_assets(path):
    """Read assets from a list or `{"assets": [...]}`. Returns `(assets, notes)`: a missing `sensitivity` becomes `private`
    and a missing `description` is only noted. `id`, `kind` and `uri` are required."""
    if path is None:
        return [], ["no asset list given: assets is empty"]
    data = _load_json(path)
    if isinstance(data, dict):
        data = data.get("assets")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a list of assets or an object with an \"assets\" list")
    assets, notes = [], []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {i} is not an object")
        asset = copy.deepcopy(entry)
        for key in ("id", "kind", "uri"):
            if not asset.get(key):
                raise ValueError(f"{path}: asset {asset.get('id') or f'entry {i}'} has no {key}")
        if asset.get("sensitivity") is None:
            asset["sensitivity"] = "private"
            notes.append(f"asset {asset['id']}: no sensitivity, set to private")
        if not asset.get("description"):
            notes.append(f"asset {asset['id']}: no description (a Choice needs one per option)")
        assets.append(asset)
    return assets, notes


def build(project, format_id, words_path, *, assets_path=None, kit_dir=None, manifest_path=None,
          formats_path=None, creator_declared=None):
    """Derive the spec. Returns `(spec, notes)`; `notes` lists what code could not fill in. Raises `ValueError` for bad
    input (unknown format, missing manifest, unusable words or assets) and `OSError` for unreadable files.
    The kit is `manifest_path` when given, else `<kit_dir>/manifest.json`; the kit id is the manifest's own `kit.id`.
    `creator_declared` seeds the spec's `creator_declared` block; its `highlight_density` overrides the format card."""
    project = Path(project).resolve()
    formats = _load_json(formats_path or FORMATS_PATH)["formats"]
    card = formats.get(format_id)
    if card is None:
        raise ValueError(f"unknown format {format_id!r}; formats/formats.json has {', '.join(formats)}")
    if manifest_path:
        manifest_file = Path(manifest_path)
    elif kit_dir:
        manifest_file = kit_manifest_path(kit_dir)
    else:
        raise ValueError("no kit: pass --kit-dir PATH or set JEVM_KIT_DIR")
    if not manifest_file.is_file():
        raise ValueError(f"kit manifest not found: {manifest_file} (see examples/kit for the format)")
    manifest = _load_json(manifest_file)
    kit_id = manifest["kit"]["id"]
    family = card["family"]
    if family not in manifest["canvases"]:
        raise ValueError(f"kit {kit_id!r} has no {family!r} canvas, so it cannot build {format_id}")
    canvas = manifest["canvases"][family]

    words = load_words(words_path)
    if not words:
        raise ValueError(f"{words_path}: no words")
    captions = manifest["captions"]
    units_sentences = sentences(words)
    assets, notes = load_assets(assets_path)

    creator = copy.deepcopy(creator_declared) if creator_declared else {}
    creator.setdefault("highlight_density", None)
    override = creator["highlight_density"] is not None
    highlight = copy.deepcopy(manifest["rules"]["highlight"])
    highlight["density"] = creator["highlight_density"] if override else card["highlight_density"]
    highlight["from"] = "kit+creator" if override else "kit+format"

    slug = _slug(project.name)
    theme = manifest["themes"]["ids"][0]
    notes.append(f"project.title is the folder name ({slug}); set the real title")
    notes.append(f"kit.theme defaulted to {theme}; the theme policy is {manifest['themes']['policy'][family]}")

    width, height = canvas["width"], canvas["height"]
    divisor = math.gcd(width, height)
    try:
        source = Path(words_path).resolve().relative_to(project).as_posix()
    except ValueError:
        source = str(Path(words_path).resolve())

    spec = {
        "spec_version": "1.0",
        "project": {"id": slug, "title": slug, "root": str(project)},
        "format": {"id": format_id, "family": family, "origin": "declared", "declared_by": "human",
                   "modules": list(card["modules"])},
        "canvas": {"aspect": f"{width // divisor}:{height // divisor}", "width": width, "height": height, "fps": FPS,
                   "duration_s": round(words[-1]["end"], 3)},
        "kit": {"id": kit_id, "manifest": _abs(manifest_file), "manifest_sha256": file_sha256(manifest_file),
                "canvas": family, "theme_policy": manifest["themes"]["policy"][family], "theme": theme,
                "overrides": {}},
        "transcript": {"origin": "asr", "source": source, "text": " ".join(w["text"] for w in words), "words": words},
        "units": {"sentences": units_sentences,
                  "caption_groups": caption_groups(words, units_sentences, captions["max_words_per_line"],
                                                   captions["max_lines"])},
        "assets": assets,
        "constraints": {
            "safe": {**copy.deepcopy(canvas["safe"]), "from": "kit"},
            "text_min_px": {**canvas["text_min_px"], "from": "kit"},
            "captions": {"max_words_per_line": captions["max_words_per_line"], "max_lines": captions["max_lines"],
                         "from": "kit"},
            "highlight": highlight,
            "emphasis_non_word": {"token": manifest["emphasis"]["non_word"], "from": "kit"},
        },
        "creator_declared": creator,
        "decide": {"policy": load_policy()["policy"], "budget_usd": BUDGET_USD[family],
                   "backends": copy.deepcopy(BACKENDS)},
    }
    return spec, notes


def _existing_spec(path):
    """The earlier spec at `path` as a dict, or None."""
    try:
        data = _load_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _existing_creator(path):
    """The `creator_declared` block of an earlier spec at `path`, so a rebuild keeps what was filled in; else None."""
    block = (_existing_spec(path) or {}).get("creator_declared")
    return block if isinstance(block, dict) else None


def cli_build(project, format_id, words, assets, kit_dir, out):
    """`jevm spec build`. Exit 0 written, 1 the derived spec fails validation (nothing written), 2 bad input.
    The kit folder is `kit_dir` (`--kit-dir`), else the one the existing spec at `out` was built with, else
    `JEVM_KIT_DIR`."""
    default_out = work_dir({"project": {"root": _abs(project)}}) / "spec.json"
    out = Path(out) if out else default_out
    kept = _existing_creator(out)
    try:
        manifest = resolve_manifest(kit_dir, _existing_spec(out))
        spec, notes = build(project, format_id, Path(words), assets_path=Path(assets) if assets else None,
                            manifest_path=manifest, creator_declared=kept)
    except (ValueError, OSError) as e:
        print(f"jevm spec build: {e}", file=sys.stderr)
        return 2
    except KeyError as e:
        print(f"jevm spec build: the kit manifest or format card is missing {e}", file=sys.stderr)
        return 2
    errors = validate(spec)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        print(f"jevm spec build: the derived spec is invalid ({len(errors)} error(s)); nothing written", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    units, canvas, highlight = spec["units"], spec["canvas"], spec["constraints"]["highlight"]
    print(f"wrote {out}")
    print(f"format {spec['format']['id']} ({spec['format']['family']}): {canvas['aspect']} {canvas['width']}x{canvas['height']} "
          f"at {canvas['fps']} fps, {canvas['duration_s']} s")
    print(f"transcript: {len(spec['transcript']['words'])} words, {len(units['sentences'])} sentences, "
          f"{len(units['caption_groups'])} caption groups")
    print(f"highlight density {highlight['density']} ({highlight['from']}); budget {tokens.fmt_usd(spec['decide']['budget_usd'])}; "
          f"{len(spec['assets'])} assets")
    if kept:
        print(f"kept creator_declared from the existing {out.name}")
    if notes:
        print("missing or to check:")
        for note in notes:
            print(f"  - {note}")
    return 0
