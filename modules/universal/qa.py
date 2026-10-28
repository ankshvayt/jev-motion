"""Frame QA, QA-01: perceptual triage of a Short's render, in whatever kit the spec names.

The protocol: frames cut from the render with exact seeks, ONE request per frame that carries all 7 defect questions
and both gates on the whole frame, no mask. The kit's look reaches the questions only through the manifest's
`qa_context.<canvas>` text, sent as the `context` state field. Code then picks at most 8 flagged frames into one
`flagged.jpg`. Triage only, never a ship gate: `cli_qa` exits 0 even with flags. Wording lives in
`qsets/qa_frame.short@1.json`; the thresholds are its `thresholds.jev`.

Pure parts (`qa_requests`, `classify`, `frame_rows`, `qa_decisions`) never touch the network; `cli_qa` does the I/O.
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from core.policy import decision_policy, load_policy
from core.rows import evidence_from, make_row
from jevlib import images, tokens
from jevlib.client import make_client
from jevlib.paths import cache_dir, work_dir
from kits.kit import KitError, resolve_manifest
from jevlib.request import Request, image_ref, load_qset, noul_p, strictest, thresholds_for

QSET = "qa_frame.short"
DECISION = "QA-01"
MODULE_ID = "universal.qa"
# The 7 defect questions, in the order the qset file keeps them (then the gates).
DEFECTS = ("text_occluded", "cut_off_edge", "caption_on_face", "empty_region", "overlapping_panels", "overlay_on_controls",
           "hard_to_read")
GATES = ("face_visible", "caption_visible")
# The frame QA sets, by kit canvas. Other canvases have no frame QA set.
QA_SETS = {"short": QSET}
MAX_FLAGGED = 8      # frames on flagged.jpg
MIN_EVERY = 0.1      # seconds between frames; keeps the `t<t:.1f>` ids unique
ASPECT_TOLERANCE = 0.01
FRAME_WORKERS = 6
_EPS = 1e-9


def qa_requests(frames, context, sensitivity, qs=None):
    """One request per `(t, path)` frame: state `{"frame": <ref>, "context": context}`, every question, one image."""
    qs = qs or load_qset(QSET)
    reqs = []
    for t, path in frames:
        ref = image_ref(path, "frame")
        reqs.append(Request(qs=qs, state={"frame": ref, "context": context},
                            questions={qid: qs.question(qid) for qid in qs.questions},
                            images={ref: Path(path)}, sensitivity=sensitivity, tag=f"t{t:.1f}"))
    return reqs


def classify(p, th):
    """`(flags, checks)` from `p` = {question: probability} and `th` = {flag, check, gate}. A question flags at
    `flag` and above, and is a check from `check` up to `flag`. `caption_on_face` counts only when both gates reach
    `gate`. The gates themselves are never flags."""
    gate_ok = all((p.get(g) or 0) >= th["gate"] - _EPS for g in GATES)
    flags, checks = [], []
    for name in DEFECTS:
        v = p.get(name)
        if v is None or (name == "caption_on_face" and not gate_ok):
            continue
        if v >= th["flag"] - _EPS:
            flags.append(name)
        elif v >= th["check"] - _EPS:
            checks.append(name)
    return flags, checks


def frame_rows(frames, answers):
    """One row per frame: its time, file, p per question, flags, checks, and what answered it. A frame whose answer
    failed keeps `error` and `detail` and has no p, flags or checks."""
    rows = []
    for (t, path), ans in zip(frames, answers, strict=True):
        qs = ans.request.qs
        row = {"t": t, "file": str(path), "qset": qs.id, "qset_version": qs.version, "backend": ans.backend,
               "model": ans.model, "request_sha256": ans.request_sha256, "route_reasons": list(ans.route_reasons)}
        if ans.ok:
            p = {qid: v for qid in ans.request.questions if (v := noul_p(ans, qid)) is not None}
            flags, checks = classify(p, thresholds_for(qs, ans.backend))
            row.update(p=p, flags=flags, checks=checks, tokens=ans.input_tokens, latency_s=ans.latency_s,
                       cached=ans.cached, error=None, detail="")
        else:
            row.update(p={}, flags=[], checks=[], tokens=None, latency_s=None, cached=False, error=ans.error,
                       detail=ans.detail)
        rows.append(row)
    return rows


def qa_decisions(rows, policy):
    """QA-01 rows, one per flagged frame, for `agent` to look at: the flagged questions and their p, the check-band
    questions beside them. `p` is the highest flag p."""
    entry = decision_policy(policy, DECISION)
    decisions = []
    for r in rows:
        if r["error"] or not r["flags"]:
            continue
        p = {name: r["p"][name] for name in r["flags"]}
        top = max(p.values())
        t = r["t"]
        # `evidence_from` reads an Answer by attribute only; the row holds the same facts, so it stands in for one
        answer = SimpleNamespace(request=SimpleNamespace(qs=SimpleNamespace(id=r["qset"], version=r["qset_version"])),
                                 backend=r["backend"], model=r["model"], request_sha256=r["request_sha256"],
                                 cached=r["cached"])
        decisions.append(make_row(
            decision=DECISION, module=MODULE_ID, target={"kind": "frame", "id": f"t{t:.1f}", "t": [t, t]},
            value={"flags": list(r["flags"]), "checks": list(r["checks"]), "p": p}, p=top,
            confidence={"metric": "noul_p", "value": top}, lane="agent",
            route_reasons=["above_auto"] + [x for x in r["route_reasons"] if x != "above_auto"],
            evidence=evidence_from(answer, entry)))
    return decisions


def write_outputs(out_dir, rows, decisions, meta):
    """Write `qa.json` ({"meta", "frames", "decisions"}), `qa.md` and `flagged.jpg` (removed when nothing is flagged)
    into `out_dir`. Returns `meta` plus the counts worked out from the rows. Only live Jev answers add tokens and
    dollars; cached ones are counted as cached."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    answered = [r for r in rows if not r["error"]]
    flagged = [r for r in rows if r["flags"]]
    check_only = [r for r in rows if r["checks"] and not r["flags"]]
    errors = Counter(r["error"] for r in rows if r["error"])
    live_tokens = sum(r["tokens"] or 0 for r in answered if r["backend"] == "jev" and not r["cached"])
    meta = dict(meta, frames=len(rows), flagged_frames=len(flagged), check_frames=len(check_only),
                errored_frames=sum(errors.values()), errors=dict(errors),
                cached_frames=sum(1 for r in answered if r["cached"]), input_tokens=live_tokens,
                usd=round(tokens.usd(live_tokens), 6))
    (out / "qa.json").write_text(json.dumps({"meta": meta, "frames": rows, "decisions": decisions}, indent=1) + "\n",
                                 encoding="utf-8")

    backend = ", ".join(sorted({r["backend"] for r in answered})) or "none"
    model = ", ".join(sorted({r["model"] for r in answered})) or "none"
    head = [backend, model, f"{len(rows)} frames", f"{len(flagged)} flagged"]
    if errors:
        head.append(f"{meta['errored_frames']} errored ({', '.join(sorted(errors))})")
    head += [f"{live_tokens} input tokens", tokens.fmt_usd(meta["usd"])]
    if meta["cached_frames"]:
        head.append(f"{meta['cached_frames']} cached")
    lines = [f"# Frame QA: {Path(meta.get('render') or out.name).name}", "", " · ".join(head), ""]
    if flagged:
        lines += ["| t (s) | question | p |", "| --- | --- | --- |"]
        lines += [f"| {r['t']:.1f} | {name} | {r['p'][name]:.2f} |" for r in flagged for name in r["flags"]]
        lines += ["", f"Contact sheet: flagged.jpg (at most {MAX_FLAGGED} frames, highest p first, labelled with time "
                      "and first flag)."]
    else:
        lines.append("No frame flagged.")
    if check_only:
        lines += ["", "Check-band questions are logged in qa.json, not shown here."]
    (out / "qa.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sheet = out / "flagged.jpg"
    if flagged:
        top = sorted(flagged, key=lambda r: (-max(r["p"][name] for name in r["flags"]), r["t"]))[:MAX_FLAGGED]
        top.sort(key=lambda r: r["t"])
        images.tile_sheet([(r["file"], f"T={r['t']:.1f}S {r['flags'][0]}") for r in top], sheet, cols=4, tile_w=300)
    else:
        sheet.unlink(missing_ok=True)
    return meta


# ---------------------------------------------------------------- the command
def _bad(message):
    print(f"qa: {message}", file=sys.stderr)
    return 2


def _sample_frames(render, times, frames_dir):
    """Cut one frame per time with exact seeks into `f_0001.jpg`...; returns [(t, path)]. Older frames are removed."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("f_*.jpg"):
        old.unlink()

    def grab(i):
        path, _ = images.frame_at(render, times[i], frames_dir / f"f_{i + 1:04d}.jpg")
        return times[i], path

    with ThreadPoolExecutor(max_workers=FRAME_WORKERS) as pool:
        return list(pool.map(grab, range(len(times))))


def cli_qa(render, spec_path, *, every=1.5, client=None, dry_run=False, no_jev=False, no_images=False, budget=None,
          kit_dir=None):
    """`jevm qa RENDER --spec SPEC`: cut a frame every `every` seconds, ask the QA set about each, write
    `<project>/work/jevm/qa/<render-stem>/{qa.json,qa.md,flagged.jpg}`. Exit 0 (flags are advisory), 2 for bad input
    (a missing render or spec, a render whose aspect is not the spec's, a canvas with no QA set, a kit manifest with
    no `qa_context` for the canvas), 3 over budget. The kit is `kit_dir` (`--kit-dir`), else the spec's, else
    `JEVM_KIT_DIR`. `no_jev` and `no_images` configure the client this builds; a `client` passed in keeps its own
    settings."""
    render, spec_path = Path(render), Path(spec_path)
    for what, path in (("render", render), ("spec", spec_path)):
        if not path.is_file():
            return _bad(f"{what} not found: {path}")
    if every < MIN_EVERY:
        return _bad(f"--every must be at least {MIN_EVERY} s")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        kit_id, canvas = spec["kit"]["id"], spec["kit"]["canvas"]
        cw, ch = spec["canvas"]["width"], spec["canvas"]["height"]
        if not (cw > 0 and ch > 0):
            raise ValueError("canvas size")
        out_dir = work_dir(spec) / "qa" / render.stem
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _bad(f"{spec_path} is not a usable spec ({exc!r})")

    qset_id = QA_SETS.get(canvas)
    if qset_id is None:
        return _bad(f"no frame QA set for canvas {canvas!r} of kit {kit_id!r} (frame QA covers the short canvas)")
    try:
        manifest_file = resolve_manifest(kit_dir, spec)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        context = manifest["qa_context"][canvas]
    except KitError as exc:
        return _bad(str(exc))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _bad(f"cannot read qa_context.{canvas} from the kit manifest ({exc!r})")
    if not context:
        return _bad(f"the kit manifest {manifest_file} has no qa_context for canvas {canvas!r}")

    try:
        info = images.probe_video(render)
    except (OSError, RuntimeError, ValueError) as exc:
        return _bad(f"cannot read the render {render}: {exc}")
    w, h = info["width"], info["height"]
    if abs(w / h - cw / ch) > ASPECT_TOLERANCE * (cw / ch):
        return _bad(f"the render is {w}x{h} but spec.canvas is {cw}x{ch}; frame QA needs a render of the spec's aspect")

    try:
        frames = _sample_frames(render, images.sample_times(info["duration_s"], every), out_dir / "frames")
    except (OSError, RuntimeError, ValueError) as exc:
        return _bad(f"cannot cut frames from {render}: {exc}")

    # The spec has no beats, so every frame carries the strictest sensitivity of any asset in the spec
    sensitivity = strictest(a.get("sensitivity") for a in spec.get("assets") or [])
    qs = load_qset(qset_id)
    if client is None:
        client = make_client(spec.get("decide"), cache_dir=cache_dir(spec), no_jev=no_jev, no_images=no_images)
    reqs = qa_requests(frames, context, sensitivity, qs)
    budget_usd = budget if budget is not None else (spec.get("decide") or {}).get("budget_usd")
    stop = client.preflight(reqs, budget_usd=budget_usd, dry_run=dry_run)
    if stop is not None:
        return stop

    rows = frame_rows(frames, client.run_many(reqs))
    decisions = qa_decisions(rows, load_policy())
    meta = write_outputs(out_dir, rows, decisions, {
        "render": str(render), "spec": str(spec_path), "qset": qs.id, "qset_version": qs.version, "every": every,
        "sensitivity": sensitivity, "thresholds": thresholds_for(qs, "jev"),
        "backends_used": client.ledger.to_dict()})
    print(f"qa: {meta['frames']} frames, {meta['flagged_frames']} flagged (+{meta['check_frames']} check), "
          f"{meta['errored_frames']} errored, ${meta['usd']:.4f} -> {out_dir}")
    return 0
