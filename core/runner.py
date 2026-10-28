"""The `plan` and `crop` commands: the glue between a spec and `decisions.json`.

`plan` asks every selected module for its requests, sends them together through one client (one preflight, one
budget), hands each module its own answers and merges the rows into `<project>/work/jevm/decisions.json`. A row someone
resolved keeps its resolution while its value is unchanged. `crop` does the same for one focus crop and draws its
overlay; `crop --box` does the same from a pixel box without asking a model, and `redraw_overlays` redraws an overlay
after `resolve --value`. The modules are pure; this file and `decisions_io` do all the writing.
"""
import dataclasses
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from core import decisions_io
from core.policy import LANES, load_policy
from jevlib import images
from jevlib.cache import file_sha256
from jevlib.client import make_client
from jevlib.paths import cache_dir, project_root, work_dir
from kits.kit import KitError, resolve_manifest
from modules.universal import assets
from modules.universal.captions import CaptionsModule
from modules.universal.claims import ClaimsModule
from spec.validate import validate

MODULES = [CaptionsModule(), ClaimsModule()]

_STALE = "$.kit.manifest_sha256"  # how `validate` starts the error for a manifest that changed since the spec was built
_NO_ANSWER = ("jev_unavailable", "jev_error", "local_unavailable")
_VIDEO_KINDS = ("video", "screen_recording")
_IMAGE_KINDS = ("image", "screenshot", "logo")
_BEAT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_BODY = re.compile(r"(\d+)\s*[xX]\s*(\d+)")


def _fail(command, message):
    print(f"{command}: {message}", file=sys.stderr)
    return 2


def _manifest_file(spec, kit_dir=None):
    """The kit manifest: `<kit_dir>/manifest.json` for `--kit-dir`, else the spec's `kit.manifest`, else `JEVM_KIT_DIR`."""
    return resolve_manifest(kit_dir, spec)


def load_context(spec_path, kit_dir=None):
    """`(spec, manifest, manifest sha256, policy)` for the spec at `spec_path`; `kit_dir` (`--kit-dir`) overrides the kit
    folder the spec names. Raises `ValueError` (the caller exits 2) when the spec cannot be read or is invalid, or when the
    kit manifest is not the one the spec was built against."""
    spec_path = Path(spec_path)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {spec_path}: {exc}") from exc
    try:
        manifest_file = _manifest_file(spec, kit_dir)
    except KitError as exc:
        raise ValueError(str(exc)) from exc
    errors = validate(spec, manifest_path=manifest_file)
    stale = [e for e in errors if e.startswith(_STALE)]
    other = [e for e in errors if e not in stale]
    if other:
        more = f" (and {len(other) - 3} more)" if len(other) > 3 else ""
        raise ValueError(f"{spec_path} is not a valid spec: {'; '.join(other[:3])}{more}; run `jevm spec validate`")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the kit manifest {manifest_file}: {exc}; "
                         "check the kit folder (`jevm kit check`), then `jevm spec build`") from exc
    sha = file_sha256(manifest_file)
    if stale or sha != spec["kit"]["manifest_sha256"]:
        raise ValueError(f"the kit manifest {manifest_file.name} changed since this spec was built "
                         f"(spec has {spec['kit']['manifest_sha256'][:12]}, file has {sha[:12]}); "
                         "run `jevm kit build` then `jevm spec build`")
    return spec, manifest, sha, load_policy()


def _short(module):
    """`captions` for `universal.captions`: the name `--stage` takes."""
    return module.id.rsplit(".", 1)[-1]


def _select(stage, only):
    """The modules `--stage` (a module short name) and `--only` (decision ids) pick; both must hold when both are
    given. `--only CAP-03` runs the whole captions module, because its span rows come from the same answers. Raises
    `ValueError` for a name that no module has."""
    stages = [_short(m) for m in MODULES]
    known = [d for m in MODULES for d in m.decisions]
    only = [d for d in (only or []) if d]
    if stage is not None and stage not in stages:
        raise ValueError(f"unknown --stage {stage!r}; the stages are {', '.join(stages)}")
    unknown = [d for d in only if d not in known]
    if unknown:
        raise ValueError(f"unknown decision id {', '.join(unknown)} in --only; the decisions are {', '.join(known)}")
    return [m for m in MODULES
            if (stage is None or _short(m) == stage) and (not only or set(only) & set(m.decisions))]


def _budget(spec, budget):
    return budget if budget is not None else (spec.get("decide") or {}).get("budget_usd")


def _open_doc(spec, spec_path, manifest_sha, out_dir):
    """The decisions document in `out_dir`, or a new one, with the spec and kit hashes brought up to date. Raises
    `ValueError` for a file that is there but unreadable: it is never overwritten, because that would drop resolutions."""
    doc = decisions_io.load(Path(out_dir) / "decisions.json")
    spec_sha = file_sha256(spec_path)
    kit = {"id": spec["kit"]["id"], "manifest_sha256": manifest_sha}
    if doc is None:
        return decisions_io.new_doc(project_id=spec["project"]["id"], spec_sha256=spec_sha, kit_id=kit["id"],
                                    manifest_sha256=manifest_sha)
    doc.update(project_id=spec["project"]["id"], spec_sha256=spec_sha, kit=kit)
    return doc


def _lane_lines(rows):
    lines = []
    for lane in LANES:
        mine = [r for r in rows if r["lane"] == lane]
        by_decision = Counter(r["decision"] for r in mine)
        detail = f" ({', '.join(f'{d} {n}' for d, n in sorted(by_decision.items()))})" if by_decision else ""
        lines.append(f"lane {lane}: {len(mine)} row{'' if len(mine) == 1 else 's'}{detail}")
    return lines


def _report(command, rows, json_path, md_path):
    for line in _lane_lines(rows):
        print(line)
    unanswered = [r for r in rows if any(reason in _NO_ANSWER for reason in r.get("route_reasons", []))]
    if unanswered:
        reasons = sorted({reason for r in unanswered for reason in r["route_reasons"] if reason in _NO_ANSWER})
        print(f"{command}: {len(unanswered)} row{'' if len(unanswered) == 1 else 's'} got no model answer "
              f"({', '.join(reasons)}); the agent decides {'it' if len(unanswered) == 1 else 'them'}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


def plan(spec_path, *, stage=None, only=None, dry_run=False, no_jev=False, no_images=False, budget=None, client=None,
         kit_dir=None):
    """`jevm plan SPEC`: run the selected modules and merge their rows into `<project>/work/jevm/decisions.json`.
    Exit 0, 2 for bad input (an invalid spec, a changed manifest, an unknown `--stage` or `--only`, an unreadable
    `decisions.json`), 3 when the estimate is over budget (nothing is sent; `--dry-run` stops after the estimate with 0).
    `no_jev` and `no_images` configure the client this builds; a `client` passed in keeps its own settings."""
    spec_path = Path(spec_path)
    try:
        spec, manifest, manifest_sha, policy = load_context(spec_path, kit_dir)
        chosen = _select(stage, only)
    except ValueError as exc:
        return _fail("plan", exc)
    chosen = [m for m in chosen if m.applies(spec, manifest)]
    if not chosen:
        print("plan: no selected module applies to this spec; nothing to do")
        return 0
    try:
        per_module = [m.requests(spec, manifest, policy) for m in chosen]
    except ValueError as exc:
        return _fail("plan", exc)
    out_dir = work_dir(spec)
    try:
        doc = _open_doc(spec, spec_path, manifest_sha, out_dir)
    except ValueError as exc:
        return _fail("plan", exc)

    reqs = [r for group in per_module for r in group]
    if client is None:
        client = make_client(spec.get("decide"), cache_dir=cache_dir(spec), no_jev=no_jev, no_images=no_images)
    stop = client.preflight(reqs, budget_usd=_budget(spec, budget), dry_run=dry_run)
    if stop is not None:
        return stop

    answers = client.run_many(reqs)
    rows, at = [], 0
    for module, group in zip(chosen, per_module):
        rows += module.decide(spec, manifest, answers[at:at + len(group)], policy)
        at += len(group)
    decisions_io.merge(doc, rows, modules={m.id for m in chosen})
    decisions_io.set_backends(doc, client.ledger.to_dict())
    json_path, md_path = decisions_io.write(doc, policy, out_dir)
    _report("plan", rows, json_path, md_path)
    return 0


def _asset(spec, asset_id):
    for record in spec.get("assets") or []:
        if record.get("id") == asset_id:
            return record
    known = ", ".join(a.get("id", "?") for a in spec.get("assets") or []) or "none"
    raise ValueError(f"unknown asset {asset_id!r}; the spec has {known}")


def _body(spec, manifest, body):
    """The window body `[W, H]`: `--body WxH`, else the kit manifest's `window_body` for the spec's canvas."""
    if body:
        m = _BODY.fullmatch(str(body).strip())
        if m is None or not (int(m[1]) > 0 and int(m[2]) > 0):
            raise ValueError(f"--body must look like WxH, not {body!r}")
        return [int(m[1]), int(m[2])]
    canvas = spec["kit"]["canvas"]
    try:
        found = manifest["canvases"][canvas]["window_body"]
    except (KeyError, TypeError):
        found = None
    if not found:
        raise ValueError(f"the kit manifest has no window body for canvas {canvas!r}: pass --body WxH")
    return list(found)


def _source_file(spec, record):
    """The local file behind an asset: its `uri` relative to the project root (or absolute)."""
    uri = record["uri"]
    if "://" in uri:
        raise ValueError(f"asset {record['id']} is remote ({uri}); crop needs a local file")
    path = Path(uri)
    if not path.is_absolute():
        path = project_root(spec) / uri
    if not path.is_file():
        raise ValueError(f"asset {record['id']}: no such file {path}")
    return path


def _parse_box(text):
    """`--box x0,y0,x1,y1` as four finite floats with x0 < x1 and y0 < y1. Raises `ValueError` with a one-line message."""
    parts = str(text).split(",")
    try:
        x0, y0, x1, y1 = (float(v) for v in parts)
    except ValueError:
        raise ValueError(f"--box must be four numbers x0,y0,x1,y1 (pixels on the source image), not {text!r}") from None
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise ValueError(f"--box must be four finite numbers, not {text!r}")
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"--box needs x0 < x1 and y0 < y1, not {text!r}")
    return x0, y0, x1, y1


def _clip_box(box, fit):
    """`box` clipped to the `src_w x src_h` source image. A box that only partly leaves the image is clipped; one with
    nothing inside it raises `ValueError`."""
    x0, y0, x1, y1 = box
    clipped = (max(x0, 0.0), max(y0, 0.0), min(x1, float(fit.src_w)), min(y1, float(fit.src_h)))
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError(f"--box {','.join(f'{v:g}' for v in box)} is outside the {fit.src_w}x{fit.src_h} source image")
    return clipped


def _fit_file(frame_dir, beat):
    return Path(frame_dir) / f"{beat}.fit.json"


def _crop_line(row, overlay):
    """The one line `crop` prints: the row, what a model said (`p`), the box and focus it came to, and the overlay."""
    value, parts = row["value"], [f"{row['id']}:"]
    if row["status"] == "overridden":
        parts.append(f"overridden by {row['resolved']['by']} (no model request sent),")
    if row["p"] is not None:
        parts.append(f"p {row['p']:.2f},")
    if value.get("found"):
        focus = value["focus"]
        parts.append(f"box_px {value['box_px']}, focus {{scale {focus['scale']:.2f}, x {focus['x']}, y {focus['y']}}}, "
                     f"clamped {','.join(value['clamped']) or 'none'},")
    elif value.get("found") is False:
        parts.append("not found,")
    parts.append(f"lane {row['lane']}" + "".join(f", {r}" for r in row["route_reasons"]) + f"; overlay {overlay}")
    return " ".join(parts)


def crop(spec_path, *, beat, asset, t=None, want=None, body=None, dry_run=False, no_jev=False, no_images=False, budget=None,
         client=None, kit_dir=None, box=None, by="agent", note=""):
    """`jevm crop SPEC --beat B --asset A [--t S] --want "..."`: one AST-01 row for beat `beat` in `decisions.json` and
    `crops/<beat>.jpg`, the sent frame with the proposed box over it. A video asset needs `t` (the second to cut the
    frame at); `body` is `WxH`, else the kit's window body. Exit 0, 2 for bad input, 3 over budget (`dry_run` gives 0).

    With `box` (`"x0,y0,x1,y1"`, pixels on the source image, a video at its native size) no model is asked and `want` is
    optional: the box goes through the same geometry as a model's, the row is `overridden` by `by` with `note`, and the
    overlay is drawn from it. A box partly outside the image is clipped; one wholly outside, or not four numbers
    with x0 < x1 and y0 < y1, exits 2. Either way the one-line result, with the focus, is printed."""
    spec_path = Path(spec_path)
    try:
        spec, manifest, manifest_sha, policy = load_context(spec_path, kit_dir)
        if _BEAT_ID.fullmatch(beat or "") is None:
            raise ValueError(f"--beat must be letters, digits, '.', '_' or '-', not {beat!r}")
        tight = _parse_box(box) if box is not None else None
        if tight is None and not (want or "").strip():
            raise ValueError("--want must say what the crop has to show (or pass --box x0,y0,x1,y1)")
        record = _asset(spec, asset)
        if record["kind"] not in _VIDEO_KINDS + _IMAGE_KINDS:
            raise ValueError(f"asset {asset} is a {record['kind']}; crop takes a video or an image")
        video = record["kind"] in _VIDEO_KINDS
        if video and t is None:
            raise ValueError(f"asset {asset} is a video: pass --t SECONDS (the second to cut the frame at)")
        body_px = _body(spec, manifest, body)
        source = _source_file(spec, record)
        out_dir = work_dir(spec)
        doc = _open_doc(spec, spec_path, manifest_sha, out_dir)
    except ValueError as exc:
        return _fail("crop", exc)
    if tight is not None and dry_run:
        print("crop: --box sends nothing, so there is nothing to estimate")
        return 0

    t = float(t) if video else None
    try:
        target_frame = out_dir / "crops" / "src" / f"{beat}.jpg"
        sent, fit = images.frame_at(source, t, target_frame) if video else images.fit_for_jev(source, target_frame)
        _fit_file(target_frame.parent, beat).write_text(json.dumps(dataclasses.asdict(fit)), encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as exc:
        return _fail("crop", f"cannot prepare a frame from {source}: {exc}")

    target = {"kind": "beat", "id": beat}
    if tight is not None:
        try:
            row = assets.crop_override(_clip_box(tight, fit), fit=fit, body=body_px, scale_range=manifest["rules"]["focus_scale"],
                                       target=target, asset_id=record["id"], t=t, want=want, policy=policy, by=by, note=note)
        except ValueError as exc:
            return _fail("crop", exc)
    else:
        reqs = assets.crop_requests(sent, want, record.get("sensitivity", "private"))
        if client is None:
            client = make_client(spec.get("decide"), cache_dir=cache_dir(spec), no_jev=no_jev, no_images=no_images)
        stop = client.preflight(reqs, budget_usd=_budget(spec, budget), dry_run=dry_run)
        if stop is not None:
            return stop
        row = assets.crop_decide(client.run_many(reqs), fit=fit, body=body_px, scale_range=manifest["rules"]["focus_scale"],
                                 target=target, asset_id=record["id"], t=t, want=want, policy=policy)
    try:
        overlay = assets.overlay(sent, row, fit, out_dir / "crops" / f"{beat}.jpg")
    except (OSError, RuntimeError, ValueError) as exc:
        return _fail("crop", f"cannot draw the overlay: {exc}")
    decisions_io.merge(doc, [row], ids={row["id"]})
    if tight is not None:
        # merge keeps an old resolution when the value is unchanged; a new --box call carries its own by and note
        next(r for r in doc["decisions"] if r["id"] == row["id"]).update(status=row["status"], resolved=row["resolved"])
    else:
        decisions_io.set_backends(doc, client.ledger.to_dict())
    decisions_io.write(doc, policy, out_dir)
    print(_crop_line(row, overlay))
    return 0


def redraw_overlays(decisions_path, ids):
    """After `resolve --value`: redraw `crops/<beat>.jpg` for each of `ids` that is an overridden AST-01 row whose resolved
    value has a `focus` and/or `box_px`, from that value (the VIEW rectangle needs a `body`, else the row's own is used).
    It needs the frame `crop` saved beside `decisions.json` (`crops/src/<beat>.jpg` and `.fit.json`); without it, or on any
    trouble drawing, it prints a one-line note and goes on. It never raises: the resolve it follows has already succeeded."""
    path = Path(decisions_path)
    try:
        doc = decisions_io.load(path)
    except ValueError:
        return
    for row in (doc or {}).get("decisions", []):
        value = (row.get("resolved") or {}).get("value")
        if (row.get("id") not in ids or row.get("decision") != "AST-01" or row.get("status") != "overridden"
                or not isinstance(value, dict) or not (value.get("focus") or value.get("box_px"))):
            continue
        beat = row["target"]["id"]
        src_dir = path.parent / "crops" / "src"
        try:
            fit_data = json.loads(_fit_file(src_dir, beat).read_text(encoding="utf-8"))
            fit = images.Fit(**{**fit_data, "pad": tuple(fit_data["pad"])})
            frame = src_dir / f"{beat}.jpg"
            if not frame.is_file():
                raise OSError("no frame")
        except (OSError, ValueError, KeyError, TypeError):
            print(f"resolve: {row['id']}: no source frame beside {path.name} (crops/src/{beat}.jpg), overlay not redrawn")
            continue
        body = value.get("body") or (row.get("value") or {}).get("body")
        if not body and not value.get("box_px"):
            print(f"resolve: {row['id']}: a focus with no body (and no box_px) cannot be drawn, overlay not redrawn")
            continue
        shown = {"status": "overridden", "p": None, "value": {**value, "found": True, "body": body}}
        try:
            out = assets.overlay(frame, shown, fit, path.parent / "crops" / f"{beat}.jpg")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            print(f"resolve: {row['id']}: overlay not redrawn ({exc})")
            continue
        print(f"redrew {out} from the resolved value")
