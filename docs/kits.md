# Kits

A kit is your video design system as jevm sees it: canvas sizes, safe zones, caption limits, highlight rules, the focus-crop scale range and a one-paragraph description of the look for frame QA. jevm ships no brand kit. `examples/kit/` is a neutral one to copy.

## Which kit folder

| Source | Used by |
| --- | --- |
| `--kit-dir PATH` | Every command that takes it (`kit build`, `kit check`, `spec build`, `spec validate`, `plan`, `crop`, `qa`). Wins over everything else. |
| The spec's `kit.manifest` | `plan`, `crop`, `qa`, `spec validate`, and `spec build` when it rebuilds a spec that already exists at the output path. |
| `JEVM_KIT_DIR` | Anything above did not name a folder. |

So: flag, then spec, then environment. `spec build` writes the manifest's absolute path and sha256 into the spec; if the manifest changes later, `plan`, `crop` and `qa` refuse the spec until you run `spec build` again.

## Files

```
my-kit/
  manifest.json    required: what jevm reads
  use_when.json    optional: facts your kit files do not state; hashed as a source, passed to an adapter
  ...              your own kit files (tokens, components), listed in manifest.sources to track drift
```

`manifest.json` follows `kits/manifest.schema.json`. The keys jevm reads:

| Key | Used for |
| --- | --- |
| `kit.id`, `kit.name` | The id goes into the spec and `decisions.json`. |
| `canvases.short`, `canvases.long` | `width`, `height`, `text_min_px` (`ui`, `mono`), `safe` (copied into the spec as is), `window_body` (`[W, H]` of a window's body in px, the box a focus crop scales within; `null` when the canvas has none). A format's family (`short` or `long`) picks the canvas. |
| `themes.ids`, `themes.policy` | The spec's default theme (the first id) and theme policy per family (`per_scene` or `one_per_video`). |
| `emphasis.words` | `{"token", "value"}`: the highlight colour token and its `#rrggbb` value. `emphasis.non_word` is the token for non-word emphasis, or `null`. |
| `captions` | `max_words_per_line` and `max_lines`: caption chunking never exceeds their product. |
| `rules.highlight` | `per_caption_line`, `per_frame`, `allowed_on`, copied into the spec's constraints. |
| `rules.focus_scale` | `[min, max]` magnification for focus crops. `rules.punch_max` is optional. |
| `qa_context.<canvas>` | One paragraph describing a frame in this kit (layout, speaker, windows, captions, colours), sent to the frame-QA questions as `context`. `null` means no frame QA on that canvas. |
| `sources`, `kit.fingerprint`, `kit.built_at` | Filled by `jevm kit build`. |

## kit build and kit check

```bash
python3 jevm.py kit build --kit-dir my-kit                         # stamp a hand-written manifest
python3 jevm.py kit build --kit-dir my-kit --adapter my-kit/adapter.py   # build it from the kit files, then stamp
python3 jevm.py kit check --kit-dir my-kit
```

`kit build` re-hashes every file in `sources` (paths inside the kit folder; `use_when.json` joins when it exists), writes `kit.fingerprint` (the sha256 of the sorted hash pairs) and `kit.built_at`, validates the result against the schema and only then writes it. Exit 0, or 2 with nothing written.

`kit check` validates the manifest against the schema and re-hashes the sources. Exit 0 when valid and unchanged; 1 when it fails the schema, a source changed, went missing or is not recorded; 2 when there is no readable manifest.

A `sources` key must be a relative path inside the kit folder. jevm refuses absolute paths and `..`.

## Adapters

An adapter is a Python file you own that builds the manifest from your kit's real files, so the numbers never drift from your stylesheet:

```python
SOURCE_FILES = ("tokens.css",)          # the kit files the manifest is built from

def build(kit_dir, use_when):            # use_when: parsed use_when.json, or {}
    ...
    return manifest                       # sources, fingerprint and built_at are filled by jevm
```

Read every number from the kit's files and raise an error naming the file when one is missing; `kit build` reports it and writes nothing. `examples/kit/adapter.py` reads `examples/kit/tokens.css` and `use_when.json`.
