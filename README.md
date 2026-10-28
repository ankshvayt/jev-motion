# Jev Motion (`jevm`)

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
| `python3 jevm.py crop spec.json --beat B --asset A [--t S] --want "..." [--body WxH] [--kit-dir DIR] [spend flags]` | One AST-01 row (`AST-01:<B>`) and `crops/<B>.jpg`, the sent frame with the proposed box drawn on it. A video asset needs `--t`. The body defaults to the kit's window body for the canvas. |
| `python3 jevm.py qa RENDER.mp4 --spec spec.json [--every S] [--kit-dir DIR] [spend flags]` | A frame every `--every` seconds (default 1.5), each asked the frame-QA set with the kit's `qa_context`. Writes `work/jevm/qa/<render-stem>/{qa.json,qa.md,flagged.jpg}`. Flags are advisory: exit 0. The `short` canvas only. |
| `python3 jevm.py resolve decisions.json ID[,ID...] (--accept \| --value V) --by agent\|human [--note "..."]` | Marks rows accepted, or overridden with `V` (JSON when it parses). Works on `qa.json` too. Exit 2 on an unknown id. |
| `python3 jevm.py eval [--module M] [--qset Q] [--backend jev\|local] [--model M] [--offline] [--out DIR] [spend flags]` | Scores the question sets against the frozen eval sets in `evalsets/` and writes `<DIR>/<date>-<HHMMSS>-<backend>-<scope>.{md,json}` (default `eval/`). `--offline` replays scripted answers through fake backends. Run once per backend and the second report carries a parity table. Exit 0 all gates pass; 1 a gate failed or could not be measured; 2 bad input; 3 over budget. |

Spend flags: `--dry-run` prints one estimate line per backend (requests, estimated input tokens, estimated dollars, cache hits) and sends nothing; `--budget USD` (default `spec.decide.budget_usd`) sends nothing and exits 3 when the estimate is over it; `--no-jev` sends nothing to TypeSafe; `--no-images` sends no image to either backend.

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
