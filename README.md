# Jev Motion (`jevm`)

A plain-English overview of what it does lives at <https://ankshvayt.github.io/jev-motion/>.

Bounded decisions for HyperFrames video builds, made by a decision model and written down where a build agent can act on them.

`jevm` asks small, typed questions of [Jev](https://typesafe.ai) (TypeSafe's System One model, in the cloud) and/or a local Decider server over HTTP, and writes the answers to `decisions.json` and `decisions.md`:

- **Caption highlights**: which caption groups get a highlight, and which 1-3 words in each (CAP-02, CAP-03).
- **Stat-claim checks**: which spoken numbers are measured or sourced figures that need a source, and whether a claim's quote appears in your source texts (STY-06, QA-05).
- **Focus crops**: where a window's focus should sit on a screenshot or a frame of a screen recording (AST-01).
- **Frame QA triage**: which frames of a render look wrong: occluded text, cut-off edges, captions over a face, empty regions, overlapping panels, overlays on window controls and hard-to-read text (QA-01).

An agent (or a person) reads `decisions.md`, resolves what is left, and writes the composition.

**Scope.** HyperFrames only. Remotion is out of scope: there is no Remotion kit, spec or command. jevm never edits a composition, never runs `npx`, and replaces no HyperFrames check (`npx hyperframes lint`, `npm run check`, the render and your own audits stay as they are). Frame QA is triage, never a ship gate.

This is an unofficial client. It is not affiliated with TypeSafe or with HeyGen/HyperFrames.

## Install

```bash
git clone <this repo> jev-motion
cd jev-motion
python3 jevm.py --help
```

No install step and no dependencies beyond the standard library: Python 3.15 or newer (`jevm.py` exits 2 below it) plus the `ffmpeg` and `ffprobe` command-line tools on your `PATH` (frames, overlays and contact sheets). Put your Jev key in `JEV_API` (see [Environment](#environment)); without one, everything still runs and the rows a model would have answered go to the `agent` lane.

Tests, offline (no network, no key):

```bash
python3 -m unittest discover -s tests -t . -q
```

## Use it from Claude Code or Codex

You already build HyperFrames videos in a project with Claude Code or Codex, and want your agent to call `jevm` as a tool. Set it up once per machine, then once per project.

1. **Get it once**, to a fixed place outside your project. There is no install step: it needs Python 3.15 or newer plus `ffmpeg` and `ffprobe` on your `PATH`. First check the version:

   ```bash
   python3 --version
   ```

   It must show 3.15 or newer (`jevm.py` exits 2 below it). If it is older, install a newer Python; there is no other workaround. Then clone:

   ```bash
   git clone https://github.com/ankshvayt/jev-motion ~/tools/jev-motion
   ```

   Make `jevm` a command:

   ```bash
   mkdir -p ~/.local/bin && ln -s ~/tools/jev-motion/jevm.py ~/.local/bin/jevm
   ```

   `~/.local/bin` must be on your `PATH`. Check:

   ```bash
   jevm --help
   ```

   If you can't change `PATH`, write `python3 ~/tools/jev-motion/jevm.py` wherever this section says `jevm`.

2. **Add your key.** Put `JEV_API` in the environment, or in a `.env` file in the directory you run `jevm` from (the current working directory only; parent folders are not searched), so run `jevm` from your project root. If the project is not a git repo, skip this; otherwise keep the key and jevm's output (`work/` holds decisions, caches, crops and QA frames) out of git:

   ```bash
   printf '.env\nwork/\n' >> .gitignore
   ```

   Without a key everything still runs, and the rows a model would have answered go to the `agent` lane.

3. **Add a kit** (your style as jevm sees it). Copy the example into the project, edit it, re-stamp it, check it:

   ```bash
   cp -r ~/tools/jev-motion/examples/kit jevm-kit
   ```

   Editing the kit is optional for a first run, but set `qa_context` (one paragraph describing a frame in your style, such as a screenshot explainer or a talking head) in `jevm-kit/manifest.json`, because every frame-QA question sees it. The other fields that matter most: `canvases.short` / `canvases.long` (size, `safe`, `text_min_px`, `window_body`), `captions` (`max_words_per_line`, `max_lines`), `emphasis.words.value` (your highlight colour), and `rules.highlight` and `rules.focus_scale`. `kit check` fails with "stale" when a file listed under `sources` (`tokens.css`, `use_when.json`) changed since the last `kit build`, so re-stamp after editing those, then check. After any later manifest change, re-run `spec build`: `plan`, `crop` and `qa` refuse a spec built from an older manifest.

   ```bash
   jevm kit build --kit-dir jevm-kit
   ```

   ```bash
   jevm kit check --kit-dir jevm-kit
   ```

   Instead of passing `--kit-dir` every time, you can `export JEVM_KIT_DIR=jevm-kit`. Details: [docs/kits.md](docs/kits.md).

4. **Give it words.** Run HyperFrames' transcriber from the project root, with `-d .` so the result lands in the root:

   ```bash
   npx hyperframes transcribe assets/voiceover.wav -d .
   ```

   The first run may download a speech model, so it needs network access. Without `-d`, `transcript.json` is written next to the input file (here `assets/transcript.json`), not in the current directory; `-d DIR` sets where it goes. Pass that file to `spec build --words` as it is: it is a word-level list and `id` is ignored. Do not redirect the command's output into the file; `--json` only prints a status line, not the words. For transcripts from other tools, `--words` takes a JSON list of words, or an object with a `words` list. Each word needs `text` (or `word`) plus numeric `start` and `end` in seconds; `speaker` and `conf` are kept if present, anything else is dropped.

   ```json
   [
     {"id": "w0", "text": "Most", "start": 0.0, "end": 0.3},
     {"id": "w1", "text": "teams", "start": 0.35, "end": 0.53}
   ]
   ```

5. **List your assets.** Without an asset list the spec has 0 assets and `crop` has nothing to point at. Write `assets.json` in the project root (`examples/project/assets.json` is a model): a JSON list of assets, or an object with an `assets` list.

   ```json
   [
     {
       "id": "billing",
       "kind": "screenshot",
       "uri": "assets/billing-screenshot.png",
       "description": "Screenshot of the billing page",
       "sensitivity": "public",
       "layer": "third_party_ui"
     },
     {
       "id": "demo",
       "kind": "screen_recording",
       "uri": "assets/demo.mp4",
       "description": "Screen recording of the export flow",
       "sensitivity": "private",
       "layer": "footage"
     }
   ]
   ```

   - `id` (required): lowercase letters, digits, `_` or `-`, starting with a letter. This is what `crop --asset` takes.
   - `kind` (required): `video`, `screen_recording`, `screenshot`, `image`, `logo`, `audio` or `source_text`. `source_text` files (`.md`, `.txt`, `.html`) are where the stat check looks for a claim's quote.
   - `uri` (required): a path relative to the project root (the `--project` folder), or an absolute path. `crop` needs a local image or video file.
   - `description`: optional, but say what the asset shows; `spec build` notes when it is missing.
   - `sensitivity`: `public`, `own_render` or `private`; when missing, `spec build` sets it to `private`. `own_render` is a sensitivity, not a kind: list a draft render of your own as a `video` with `sensitivity: own_render` (`qa` takes the render's path itself, so it needs listing only to crop from it). `qa` records the strictest value of any asset in `qa.json`. A question set with a local route sends `private` images to Decider instead of Jev; no shipped set has one, so today frame QA goes to Jev whatever the value.
   - `layer`: optional, one of `overlay`, `footage` or `third_party_ui`. It is checked against the schema and not used by the commands.

   Pass the file with `--assets assets.json` to `spec build` (step 6 shows the full line). The assets are copied into `spec.json`, so after you change `assets.json`, rebuild the spec.

6. **Tell the agent.** Edit the block below for each project before pasting it. Placeholders: `<KIT_DIR>` your kit folder (e.g. `jevm-kit`), `<WORDS>` the transcript file (e.g. `transcript.json`), `<ASSETS>` the asset list (e.g. `assets.json`), `<short|long>` the format (`short` is vertical 9:16, `long` is horizontal 16:9), `<AUDIO>` the voiceover or video file you transcribe. Paste it into `CLAUDE.md` (Claude Code) or `AGENTS.md` (Codex), after the step 1 setup:

   ```markdown
   ## jevm (bounded video-build decisions)
   `jevm` is on PATH. Run it from the project root so .env is found.
   Kit: ./<KIT_DIR>. Words: ./<WORDS> (written by `npx hyperframes transcribe <AUDIO> -d .` from the project root; without `-d` it lands next to the input file). Assets: ./<ASSETS>. Format: <short|long>.

   Loop, for each video:
   1. `jevm kit check --kit-dir <KIT_DIR>`
   2. `jevm spec build --project . --format <short|long> --words <WORDS> --assets <ASSETS> --kit-dir <KIT_DIR>`, then `jevm spec validate work/jevm/spec.json --kit-dir <KIT_DIR>`. Rebuild the spec after changing the assets, the words or the kit manifest.
   3. `jevm plan work/jevm/spec.json --dry-run` to see the cost, then `jevm plan work/jevm/spec.json`
   4. Read `work/jevm/decisions.md`. Resolve `agent` lane rows yourself, several ids in one call if you like: `jevm resolve work/jevm/decisions.json CAP-02:cg0,CAP-02:cg1 --accept --by agent --note "hook line"` (or `--value V`). A short transcript can send most caption rows to `agent` with `module_not_helping`; that is normal, not a failure.
   5. Send me every `human` lane row in ONE message and wait for my answer. Then `jevm resolve ... --by human`. The stat check (STY-06) also flags sentences with number words like "one click"; if I say a flagged sentence is not a stat, resolve it with `jevm resolve work/jevm/decisions.json STY-06:ID --value '{"class":"opinion"}' --by human`.
   6. For each window focus shot: `jevm crop work/jevm/spec.json --beat B --asset A --want "what to focus on"` (add `--t SECONDS` for a video asset). `B` is any short name you choose (e.g. `b1`, `sort-button`); it becomes the row `AST-01:B` and the overlay `work/jevm/crops/B.jpg`. `A` is an `id` from the file you passed to `--assets`. `--want` must describe ONE region: for two unrelated targets (a button top-right and a column mid-page) run two crops with two beat names. Always look at the overlay: `p` is only confidence that the target is present, not box accuracy. If the box is wrong, re-run with a sharper, more specific `--want` (a new request; the same `--want` is answered from cache), then give the box yourself: `jevm crop work/jevm/spec.json --beat sort-button --asset billing --box 1430,110,1555,170 --note "tight on the button"`, and look at the overlay again. Crop rows are `agent` lane: `--accept` a model row or override it with `--box` before you build.
   7. Build the composition from the resolved rows, then render.
   8. Short only: `jevm qa RENDER.mp4 --spec work/jevm/spec.json`, then look at `work/jevm/qa/<render-stem>/flagged.jpg` (written when frames are flagged). Labels can be imprecise (`text_occluded` where the real issue is a cut-off edge): trust the image, not the label. `qa` exits 2 on a `long` spec, so skip it there.

   Rules:
   - jevm never edits the composition. You do.
   - qa is triage, not a ship gate. Keep `npx hyperframes lint`, `npm run check` and the render as they are.
   - Each run is capped by the budget in `work/jevm/spec.json` (`decide.budget_usd`; `spec build` sets it to $0.10 for `short` and $1.00 for `long`). `--budget USD` overrides it for one run; exit 3 means over budget. Use `--no-jev` to send nothing to TypeSafe, `--no-images` to send no images.
   - Never print or commit `.env` or the key.
   ```

   **When a crop is wrong.** The `crop` command prints one line: `p` (when a model answered), `box_px`, `focus {scale, x, y}`, `clamped` and the overlay path. `p` is the model's confidence that the target is present in the frame (the `vis` answer; the overlay label shows it as `P=`). It says nothing about how tight the box is, so a box can be clearly off at p 0.98. `S=2.00` on the overlay is the focus scale, which the kit's `rules.focus_scale` may cap. `clamped` lists what the scale cap or the window's edges moved (`scale`, `x`, `y`; empty means none). If it contains `x` or `y`, the target sits at the window's edge and can't be centred; that is not an error. The yellow `VIEW` rectangle on the overlay is what the viewer will actually see: the window crops a source whose aspect differs from its body (`object-fit: cover`), so a target outside `VIEW` can't be shown at that `--body`; pass a `--body WxH` with the source's aspect. Then, in order:

   - **Re-ask with a sharper `--want`**: one region, named by visible text or position, e.g. `--want "the blue Sort button, top right"`. A different `--want` is a new request and costs one; the same `--want` (and the same `--t`) on the same file is answered from the cache for free, with the same box. A different `--t` cuts a different frame, so it is a new request too. There is no flag to bypass the cache. Re-using the beat name replaces that beat's row and overlay.
   - **Still wrong: give the box.** Pass `--box x0,y0,x1,y1` instead of `--want`:

     ```bash
     jevm crop work/jevm/spec.json --beat sort-button --asset billing --box 1430,110,1555,170 --note "tight on the button"
     ```

     The numbers are pixels on the source image, the same space as `box_px` (for a video, the frame at `--t` at its native size). The overlay of a large image is scaled down, so measure on the source file, or nudge the printed `box_px`. This makes no model call, computes `focus` with the same geometry as the model path (the box is grown to the window's aspect, so give the target tightly), marks the row `overridden` and redraws the overlay. A box partly outside the image is clipped; one wholly outside, or not four numbers with x0 < x1 and y0 < y1, exits 2. Look at the overlay again. `--by human` and `--note` record who decided and why.
   - **Then resolve it.** Crop rows sit in the `agent` lane and are never applied automatically: `--accept` the model's row, or override it with `--box`, before the build.

   Spend, in short: the budget is per run (each `plan`, `crop` or `qa` compares its own estimate against it, not a running total). To change the default for a project, edit `decide.budget_usd` in `work/jevm/spec.json`, or pass `--budget USD`; `spec build` writes the default again on every rebuild, so an edit in `spec.json` does not survive a rebuild.

7. **Optional: a Claude Code skill.** The same block can live in `.claude/skills/jevm/SKILL.md`, with this frontmatter above it, so Claude Code loads it when the task calls for it:

   ```markdown
   ---
   name: jevm
   description: Use when building a HyperFrames video that needs caption highlights, stat-claim checks, focus crops or frame QA triage from jevm.
   ---
   ```

## Quickstart

Everything below runs from the repo root on the shipped examples. `examples/project/words.json` is a short synthetic transcript; `examples/kit/` is a neutral example kit.

```bash
# 1. Check the kit: the manifest is valid and no kit file changed since it was built.
python3 jevm.py kit check --kit-dir examples/kit

# 2. Words + a format + the kit -> spec.json (written to examples/project/work/jevm/spec.json).
python3 jevm.py spec build --project examples/project --format short \
    --words examples/project/words.json --assets examples/project/assets.json --kit-dir examples/kit

# 3. Plan the caption highlights and stat claims. --dry-run prints the estimate and sends nothing.
python3 jevm.py plan examples/project/work/jevm/spec.json --dry-run
python3 jevm.py plan examples/project/work/jevm/spec.json

# 4. Read examples/project/work/jevm/decisions.md, then resolve rows.
python3 jevm.py resolve examples/project/work/jevm/decisions.json CAP-02:cg1 --accept --by agent

# 5. After you render: triage the frames.
python3 jevm.py qa renders/v1.mp4 --spec examples/project/work/jevm/spec.json

# The eval harness, replaying scripted answers: no network, no key, nothing spent.
python3 jevm.py eval --offline --out /tmp/jevm-eval
```

Without `JEV_API`, step 3 still writes `decisions.json`: the claim rows are answered by Decider when it is running, and every row no model answered goes to the `agent` lane with `jev_unavailable`.

## Commands

The usual loop: `kit check`, `spec build`, `plan`, read `decisions.md`, `resolve`, `crop` for each window shot, build and render the composition, `qa` on the render. `python3 jevm.py <command> --help` lists one command's flags.

| Command | Does |
| --- | --- |
| `python3 jevm.py kit build [--kit-dir DIR] [--adapter FILE.py]` | Stamps `<kit>/manifest.json`: the sha256 of every source file, the fingerprint and the build time. With `--adapter`, builds the manifest from your kit's files first. Validates against `kits/manifest.schema.json` before writing. Exit 0; 2 nothing written. |
| `python3 jevm.py kit check [--kit-dir DIR]` | Validates the manifest against the schema and re-hashes its sources. Exit 0 valid and fresh; 1 invalid or stale; 2 no manifest. |
| `python3 jevm.py spec build --project DIR --format short\|long --words PATH [--assets PATH] [--kit-dir DIR] [-o PATH]` | Words JSON + asset list + a format id + the kit into `spec.json` (default `<project>/work/jevm/spec.json`). Prints what it derived and what is missing. A rebuild keeps `creator_declared`. |
| `python3 jevm.py spec validate spec.json [--kit-dir DIR]` | Schema, ids, word spans, the caption cap and the kit manifest hash. Exit 1 when invalid. |
| `python3 jevm.py plan spec.json [--stage S] [--only ID,ID] [--kit-dir DIR] [spend flags]` | Caption highlights (CAP-02, CAP-03) and stat claims (STY-06, QA-05) into `<project>/work/jevm/decisions.json` and `.md`. `--stage` is `captions` or `claims`. Exit 0; 2 bad input; 3 over budget. |
| `python3 jevm.py crop spec.json --beat B --asset A [--t S] (--want "..." \| --box x0,y0,x1,y1) [--by agent\|human] [--note "..."] [--body WxH] [--kit-dir DIR] [spend flags]` | One AST-01 row (`AST-01:<B>`) and `crops/<B>.jpg`, the sent frame with the box drawn on it, then one line with `p`, `box_px`, `focus` and `clamped`. `--box` takes pixels on the source image, asks no model and marks the row `overridden`. A video asset needs `--t`. The body defaults to the kit's window body for the canvas. |
| `python3 jevm.py qa RENDER.mp4 --spec spec.json [--every S] [--kit-dir DIR] [spend flags]` | A frame every `--every` seconds (default 1.5), each asked the frame-QA set with the kit's `qa_context`. Writes `work/jevm/qa/<render-stem>/{qa.json,qa.md,flagged.jpg}`. Flags are advisory: exit 0. The `short` canvas only. |
| `python3 jevm.py resolve decisions.json ID[,ID...] (--accept \| --value V) --by agent\|human [--note "..."]` | Marks rows accepted, or overridden with `V` (JSON when it parses). Resolving a row again replaces its note. Works on `qa.json` too. Exit 2 on an unknown id. For an `AST-01` row, advanced use: a `--value` with `focus` and/or `box_px` also redraws the overlay (prefer `crop --box`; no source frame beside `decisions.json` gives a note, never a failure). |
| `python3 jevm.py eval [--module M] [--qset Q] [--backend jev\|local] [--model M] [--offline] [--out DIR] [spend flags]` | Scores the question sets against the frozen eval sets in `evalsets/` and writes `<DIR>/<date>-<HHMMSS>-<backend>-<scope>.{md,json}` (default `eval/`). `--offline` replays scripted answers through fake backends. Run once per backend and the second report carries a parity table. Exit 0 all gates pass; 1 a gate failed or could not be measured; 2 bad input; 3 over budget. |

Spend flags: `--dry-run` prints one estimate line per backend (requests, estimated input tokens, estimated dollars, cache hits) and sends nothing; `--budget USD` (default `spec.decide.budget_usd`: $0.10 for `short`, $1.00 for `long`, per run) sends nothing and exits 3 when the estimate is over it; `--no-jev` sends nothing to TypeSafe; `--no-images` sends no image to either backend.

## Environment

| Variable | Effect |
| --- | --- |
| `JEV_API` | The Jev key. Read from the environment, else from a `.env` file in the current working directory (that folder only). Never printed, logged, cached or written. |
| `JEVM_KIT_DIR` | The kit folder when neither `--kit-dir` nor the spec names one. |
| `JEVM_CACHE_DIR` | Where raw answers are cached. Default: `<project>/work/jevm/.jevm-cache/` for `plan`, `crop` and `qa`; `.jevm-cache/` in the current directory for `eval`. |
| `JEVM_DECIDER_URL` | The local Decider's base URL, instead of `spec.decide.backends.local.url` (default `http://127.0.0.1:8080`). |
| `JEVM_OFFLINE` | When set (and not `0`), both real backends refuse before any network call. The tests set it. |

## Kits

A kit is your design system as jevm sees it, in a folder you own: `manifest.json` (required), `use_when.json` (optional) and whatever kit files you list as sources. jevm ships no brand kit; copy `examples/kit/`. The kit folder is `--kit-dir`, else the one the spec was built with, else `JEVM_KIT_DIR`.

The manifest holds canvas sizes and safe zones, text minimums, caption limits (`max_words_per_line`, `max_lines`), highlight rules and colour, the focus-crop scale range, the window body a crop scales within, and `qa_context`: one paragraph describing a frame in your kit, which frame QA sends with every frame. Write it by hand and stamp it with `kit build`, or let an adapter build it from your stylesheet (`examples/kit/adapter.py`). Schema: `kits/manifest.schema.json`. Details: [docs/kits.md](docs/kits.md).

## Formats

`formats/formats.json` holds the format cards `spec build --format` takes: `name`, `family` (`short` or `long`, a canvas the kit must have), `modules`, `highlight_density` (the share of caption groups that get a highlight) and `qa_sets`. Two cards ship: `short` (vertical 9:16) and `long` (horizontal 16:9). Add your own card the same way.

## The decisions contract

`<project>/work/jevm/decisions.json` is what scripts read; `decisions.md` is what the agent reads. Every row has an `id` (`<decision>:<target id>`, stable across re-runs), a proposed `value`, the probability and confidence behind it, a `lane`, its `route_reasons`, a `status` (`proposed`, `accepted`, `overridden`) and an `evidence` block naming the question set, backend, model and cache key. Re-running `plan` keeps a resolution on every row whose proposed value did not change. Full shape and rules: [docs/decisions.md](docs/decisions.md).

## Lanes

| Lane | Meaning | Who acts |
| --- | --- | --- |
| `auto` | Reversible and cheap to redo, above the auto line, both option orders agree. | The build uses it; the agent may override. |
| `agent` | Near a threshold (within the review band), an order flip, a decision the policy says the agent reviews, or any row no model answered. | The build agent accepts, picks an alternative, or overrides. |
| `human` | Claims, and decisions on the never-auto list that need a person. | A human reviewer. The agent sends all of them in one message and waits for the answer. |

The never-auto list ([docs/never-auto.md](docs/never-auto.md)) is code, not policy data: no value in `policies/default.json` can route one of those decisions to `auto`. When more than 40% of a module's rows from decisions that could settle on their own land in `agent` or `human`, `decisions.md` heads that module with `module_not_helping`.

## Backends and fallback

Each question set (`qsets/<id>@<version>.json`) names its route, pins its models and stores thresholds per backend ([docs/question-sets.md](docs/question-sets.md)).

- **Jev** (TypeSafe): pinned models (`jev-1.13.0` for text, `jev-1.14.0` for images and Box). Needs `JEV_API`.
- **Decider** (local, optional): a decision-model server that takes Jev's request shape, at `http://127.0.0.1:8080` by default. Sets with a local calibration (`stat_check`, `presence`) run there first. Box never runs on it.

Fallback works both ways: Decider down or rejecting a request sends it to Jev with `backend_fallback`; no key or Jev down sends calibrated sets to Decider and everything else to the `agent` lane with `jev_unavailable`. A request that fails after retries goes to the `agent` lane with `jev_error`. Every command finishes either way, and `plan` says how many rows got no model answer. Raw answers are cached on an order-preserving key, so a re-run costs nothing. Details: [docs/fallback.md](docs/fallback.md).

## Eval

`jevm eval` runs four frozen eval sets through the same modules the commands use and gates the results: frame QA recall and clean-frame false alarms, highlight precision at K and hand-highlight ranks, crop IoU against a hand rectangle and presence AUC, stat-check bands on labelled sentences and controls. Each set carries its own gates and `local_reference` numbers. The shipped sets in `evalsets/` are synthetic (written by `evalsets/make_synthetic.py`), so a live run on them is a smoke test of your key and backends, not a measurement; freeze your own frames and lines in the same layout to measure.

## Limits

- One canvas pair (`short` 9:16, `long` 16:9) and a minimal manifest. Frame QA covers the `short` canvas; a long-form `qa` exits 2.
- `plan` runs caption highlights and stat claims only.
- Caption chunking (CAP-01) is code: sentences end on `. ? ! …` or a pause of 0.60 s or more; a caption group never exceeds the kit's words per line times lines.
- A caption group gets a highlight row (CAP-02) only when it is in the top K (K = density times the number of groups, at least 1) or within the review band of the K-th score. A span row (CAP-03) exists only for a highlighted group or one in the `agent` lane.
- `crop` needs `--t` for a video: the second to cut the frame at. The crop row is always in the `agent` lane: someone looks at the overlay. A canvas whose manifest has no window body needs `--body WxH`.
- `qa` gives every frame the strictest `sensitivity` of any asset in the spec.
- Only sentences with a digit or a number word go to the stat check.

## Layout

```
jevm.py          the command line
core/            policy, rows, the decisions writer, the plan/crop runner
jevlib/          the backend-neutral client: requests, cache, Jev and Decider backends, images, token estimates
modules/         the decision modules (captions, claims, assets, qa, geometry)
spec/            spec build and validate, spec/schema.json
kits/            kit build and check, kits/manifest.schema.json
qsets/           question sets      policies/  the lane policy      formats/  format cards
evals/           the eval harness   evalsets/  synthetic eval sets
examples/        a neutral kit and a synthetic project
docs/            the decisions contract, the never-auto list, fallback, kits, question sets
```

## License

MIT. See [LICENSE](LICENSE).
