"""Example kit adapter: builds `manifest.json` from `tokens.css` and `use_when.json`.

    python3 jevm.py kit build --kit-dir examples/kit --adapter examples/kit/adapter.py

An adapter is optional. Without one, write `manifest.json` by hand and run `jevm kit build --kit-dir DIR` to stamp its
source hashes. With one, jevm calls `build(kit_dir, use_when)` and hashes every file in `SOURCE_FILES` (plus
`use_when.json`), so `jevm kit check` reports when the stylesheet changes after the manifest was built. Read every number
from the kit's own files and raise `KitError` (any exception works) naming the file when one is missing.
"""
import re
from pathlib import Path

SOURCE_FILES = ("tokens.css",)

_ROOT = re.compile(r":root\s*\{([^}]*)\}")
_PROPERTY = re.compile(r"(--[\w-]+)\s*:\s*([^;]+);")
_THEME = re.compile(r'\[data-theme="([\w-]+)"\]')


def _tokens(css):
    block = _ROOT.search(re.sub(r"/\*.*?\*/", "", css, flags=re.S))
    if not block:
        raise ValueError("tokens.css: expected a :root { ... } block")
    return {name: value.strip() for name, value in _PROPERTY.findall(block.group(1))}


def _px(tokens, name):
    value = tokens.get(name, "")
    if not value.endswith("px"):
        raise ValueError(f"tokens.css: expected `{name}: <n>px;` in :root, found {value or 'nothing'!r}")
    return int(float(value[:-2]))


def _int(tokens, name):
    try:
        return int(tokens[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"tokens.css: expected `{name}: <integer>;` in :root") from exc


def _float(tokens, name):
    try:
        return float(tokens[name])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"tokens.css: expected `{name}: <number>;` in :root") from exc


def build(kit_dir, use_when):
    css = (Path(kit_dir) / "tokens.css").read_text(encoding="utf-8")
    t = _tokens(css)
    (sw, sh), (lw, lh) = use_when["canvases"]["short"], use_when["canvases"]["long"]
    top, bottom, word_top = _px(t, "--safe-top"), _px(t, "--safe-bottom"), _px(t, "--band-word-top")
    return {
        "manifest_version": "1",
        "kit": {"id": "example", "name": "Example Kit", "fingerprint": "", "built_at": ""},
        "sources": {},
        "canvases": {
            "short": {
                "width": sw, "height": sh,
                "text_min_px": {"ui": _px(t, "--text-min-ui"), "mono": _px(t, "--text-min-mono")},
                "safe": {"x_px": _px(t, "--safe-x"), "top_px": top, "bottom_px": bottom},
                "bands": {"visual": [top, word_top], "word": [word_top, sh - bottom]},
                "window_body": use_when["window_body"]["short"],
            },
            "long": {
                "width": lw, "height": lh,
                "text_min_px": {"ui": _px(t, "--text-min-ui-long"), "mono": _px(t, "--text-min-mono-long")},
                "safe": {"all_px": _px(t, "--safe-long")},
                "window_body": use_when["window_body"]["long"],
            },
        },
        "themes": {"ids": list(dict.fromkeys(_THEME.findall(css))), "policy": use_when["themes_policy"]},
        "emphasis": {"words": {"token": "accent", "value": t["--accent"].upper()}, "non_word": None},
        "captions": {"max_words_per_line": _int(t, "--caption-words-per-line"), "max_lines": _int(t, "--caption-lines"),
                     "source": "tokens.css: --caption-words-per-line, --caption-lines"},
        "rules": {
            "highlight": {"per_caption_line": _int(t, "--highlight-per-line"),
                          "per_frame": _int(t, "--highlight-per-frame"), "allowed_on": ["words"]},
            "punch_max": _float(t, "--punch-max"),
            "focus_scale": [_float(t, "--focus-scale-min"), _float(t, "--focus-scale-max")],
        },
        "qa_context": dict(use_when["qa_context"]),
    }
