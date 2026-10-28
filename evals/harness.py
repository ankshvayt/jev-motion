"""`jevm eval`: the frozen eval sets run through the decision modules, scored, gated, written up.

Four sets, each built from the module that owns the decision (never a copy of its wording):

  qa_frame.short    QA-01 over clean and planted frames                      `qa.qa_requests`, `qa.frame_rows`
  highlight         CAP-02 budget and CAP-03 span on every highlight set     `captions.CaptionsModule.requests`
  crop_box (+ presence)  AST-01 box IoU, centre, presence AUC                `assets.crop_requests`, `assets.crop_decide`
  stat_check        STY-06 on labelled sentences and controls                `claims.stat_requests`

The sets live in `evalsets/` and carry their own gates and `local_reference` numbers, so a set you freeze from your own
videos brings its own bar. The shipped sets are synthetic (`evalsets/make_synthetic.py`).

`run_eval` builds every request first, calls `preflight` once (budget, `--dry-run`), then sends them in one pass. Exit 0
when every gate passes on `jev`; 1 when a gate fails or cannot be measured (a request errored: that metric is
UNVERIFIED, never a pass); 2 for bad input; 3 over budget. On `local` the exit is 1 only when a parity number misses its
`local_reference` by more than one item. `eval_*` are callable on their own with a `Client`.

`--offline` (`main`) replays scripted answers (tests/fixtures/eval/*.json, written by `evalsets/make_synthetic.py`)
through fake backends: it checks the harness and is NOT a measurement of Jev or Decider. Its files carry an `-offline`
suffix so they never stand in for a live run. Scoring never touches the network.
"""
import argparse
import dataclasses
import datetime
import json
import re
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

from core.policy import budget_k, load_policy
from jevlib import fake
from jevlib.backends.base import BackendError
from jevlib.cache import Cache
from jevlib.client import Client, make_client
from jevlib.images import Fit, image_size
from jevlib.paths import EVAL_OUT_DIR, EVALSETS_DIR, TOOL_ROOT
from jevlib.request import choice_probs, choice_top, load_qset, noul_p, thresholds_for
from modules.universal import assets, captions, claims, geometry, qa

EPS = 1e-9
FIXTURES_DIR = TOOL_ROOT / "tests" / "fixtures" / "eval"

# `--qset` ids, in run order, and the module each `--module` name stands for.
SETS = ("qa_frame.short", "highlight", "crop_box", "stat_check")
MODULES = {"qa": "qa_frame.short", "captions": "highlight", "assets": "crop_box", "claims": "stat_check"}

QA_HALF_CUT = 0.5        # the second recall column
DENSITY = 0.25           # the highlight rate the highlight sets are scored at: K = budget_k(0.25, groups)
STAT_CONTROL_CUT = 0.5
AGREE_QA_CUT = 0.7

# ---------------------------------------------------------------- small helpers
def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _auc(pos, neg):
    """P(a positive scores above a negative), ties half; None when a side is empty."""
    if not pos or not neg:
        return None
    return sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg) / (len(pos) * len(neg))


def _errors(answers):
    return dict(Counter(a.error for a in answers if not a.ok))


def _latency(answers):
    """Seconds per live request; a cache hit carries no latency."""
    return [a.latency_s for a in answers if a.ok and not a.cached]


def _models(answers):
    return sorted({a.model for a in answers if a.ok})


def _gate(name, value, need, ok, complete=True):
    """One gate row; `ok` None means UNVERIFIED (an answer was missing, so the metric cannot be trusted)."""
    return {"name": name, "value": value, "need": need, "ok": bool(ok) if complete else None}


def _parity(name, value, reference, ok, complete=True):
    """One parity row: a local run's number against the set's `local_reference`, within one item."""
    return {"name": name, "value": value, "reference": reference, "ok": bool(ok) if complete else None}


def highlight_sets(root=None):
    """The highlight set names, sorted: every `evalsets/highlight/<name>.json`."""
    folder = Path(root or EVALSETS_DIR) / "highlight"
    return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []


def _with_model(qs, model):
    """`qs` with `models["jev"]` replaced by `--model`; the wording and thresholds stay."""
    if not model:
        return qs
    return dataclasses.replace(qs, models={**qs.models, "jev": model})


def _result(key, answers, requests, *, metrics, gates, parity, items, notes=()):
    return {"set": key, "models": _models(answers), "requests": len(requests), "errors": _errors(answers),
            "latency_s": _latency(answers), "metrics": metrics, "gates": gates, "parity": parity, "items": items,
            "notes": list(notes)}


# ---------------------------------------------------------------- QA (QA-01)
def _plan_qa(qs=None):
    qs = qs or load_qset(qa.QSET)
    root = EVALSETS_DIR / qa.QSET
    manifest = _read(root / "manifest.json")
    neg = [(n["t"], root / n["file"]) for n in manifest["negatives"]]
    pos = [(p["source_t"] if p.get("source_t") is not None else 0.0, root / p["file"]) for p in manifest["positives"]]
    kinds = [p["type"] for p in manifest["positives"]]
    reqs = qa.qa_requests(neg + pos, manifest["context"], "public", qs)

    def finish(answers):
        return _score_qa(qs, neg, pos, kinds, reqs, answers, manifest["gates"], manifest.get("local_reference") or {})
    return reqs, finish


def _effective(p, name, th):
    """The p a question is scored with: `caption_on_face` counts only when both gates reach the gate threshold."""
    v = p.get(name)
    if v is None:
        return None
    if name == "caption_on_face" and not all((p.get(g) or 0) >= th["gate"] - EPS for g in qa.GATES):
        return 0.0
    return v


def _score_qa(qs, neg, pos, kinds, reqs, answers, gates_cfg, reference):
    rows = qa.frame_rows(neg + pos, answers)
    neg_rows, pos_rows = rows[:len(neg)], rows[len(neg):]
    for row, kind in zip(pos_rows, kinds, strict=True):
        row["type"] = kind
    ok_neg = [r for r in neg_rows if not r["error"]]
    ok_pos = [r for r in pos_rows if not r["error"]]
    th = thresholds_for(qs, next((r["backend"] for r in rows if not r["error"]), "jev"))
    cuts = (th["flag"], QA_HALF_CUT)
    planted = list(dict.fromkeys(kinds))

    per_question = []
    for name in planted + [d for d in qa.DEFECTS if d not in planted]:
        ps = [r for r in ok_pos if r["type"] == name]
        pv = [_effective(r["p"], name, th) for r in ps]
        nv = [_effective(r["p"], name, th) for r in ok_neg]
        per_question.append({
            "question": name, "positives": len(ps),
            "recall_flag": sum(v >= cuts[0] - EPS for v in pv) if ps else None,
            "recall_half": sum(v >= cuts[1] - EPS for v in pv) if ps else None,
            "false_alarms": sum(v >= cuts[0] - EPS for v in nv),
            "mean_p_neg": round(statistics.mean(nv), 3) if nv else None})
    own = sum(1 for r in ok_pos if r["type"] in r["flags"])
    clean = sum(1 for r in ok_neg if r["flags"])
    metrics = {"flag": cuts[0], "check": th["check"], "gate": th["gate"], "n_neg": len(ok_neg), "n_pos": len(ok_pos),
               "per_question": per_question, "clean_any_flag": clean,
               "clean_any_flag_or_check": sum(1 for r in ok_neg if r["flags"] or r["checks"]),
               "pos_any_flag": sum(1 for r in ok_pos if r["flags"]), "own_flagged": own}
    complete = not _errors(answers)
    own_min, clean_max = gates_cfg["own_flagged_min"], gates_cfg["clean_flagged_max"]
    gates = [_gate("own-question recall", f"{own}/{len(ok_pos)}", f">= {own_min}/{len(pos)}", own >= own_min, complete),
             _gate("clean frames flagged", f"{clean}/{len(ok_neg)}", f"<= {clean_max}/{len(neg)}", clean <= clean_max,
                   complete)]
    parity = []
    if "clean_flagged" in reference:
        ref = reference["clean_flagged"]
        parity.append(_parity("clean frames flagged", f"{clean}/{len(ok_neg)}", f"{ref}/{len(neg)}",
                              abs(clean - ref) <= 1, complete))
    items = {"qa": {Path(r["file"]).name: r["p"] for r in ok_neg + ok_pos}}
    return _result(qa.QSET, answers, reqs, metrics=metrics, gates=gates, parity=parity, items=items)


def eval_qa(client, *, qs=None):
    """QA-01 on the frozen frames: per-question recall at the flag cut and at 0.5, false alarms on the clean frames,
    clean frames with any flag, positives flagged by their own question. Gates: the set's `own_flagged_min` and
    `clean_flagged_max`."""
    reqs, finish = _plan_qa(qs)
    return finish(client.run_many(reqs))


# ---------------------------------------------------------------- highlight (CAP-02, CAP-03)
def _highlight_spec(name, ds):
    """A spec holding only what the captions module reads: the transcript and one caption group per frozen group."""
    words, groups = [], []
    for gi, group in enumerate(ds["groups"]):
        first = len(words)
        for text in group["words"]:
            i = len(words)
            words.append({"id": f"w{i + 1}", "text": text, "start": round(i * 0.4, 3), "end": round(i * 0.4 + 0.4, 3)})
        groups.append({"id": f"g{gi + 1}", "words": [words[first]["id"], words[-1]["id"]],
                       "start": words[first]["start"], "end": words[-1]["end"], "lines": 1, "sentence": "s1"})
    return {"transcript": {"origin": "eval", "source": name, "text": ds["transcript"], "words": words},
            "units": {"sentences": [], "caption_groups": groups}, "constraints": {"highlight": {"density": DENSITY}}}


def _plan_highlight(qs=None):
    qs = qs or load_qset(captions.QSET)
    module, policy = captions.CaptionsModule(), load_policy()
    sets, reqs = [], []
    names = highlight_sets()
    if not names:
        raise ValueError(f"no highlight sets in {EVALSETS_DIR / 'highlight'}")
    for name in names:
        ds = _read(EVALSETS_DIR / "highlight" / f"{name}.json")
        # the module builds the requests from the stored question set; an override only swaps the model on them
        built = [dataclasses.replace(r, qs=qs) for r in module.requests(_highlight_spec(name, ds), None, policy)]
        sets.append((name, ds, len(built)))
        reqs += built

    def finish(answers):
        return _score_highlight(qs, sets, reqs, answers)
    return reqs, finish


def _overlap(a, b):
    return a is not None and b is not None and a[0] <= b[1] and b[0] <= a[1]


def _score_set_highlight(qs, name, ds, answers):
    """Scores for one Short from its answers, `[fwd, rev]` per batch. `rank` and `K` count every caption group."""
    groups, hand = ds["groups"], ds["hand"]
    n, size = len(groups), qs.batch
    need2, fwds, revs = [], [], []
    for g in range(n):
        k, i = divmod(g, size)
        fwd, rev = answers[2 * k], answers[2 * k + 1]
        fwds.append(fwd)
        revs.append(rev)
        need2.append(noul_p(fwd, f"need2_{i}") if fwd.ok else None)
    scored = [g for g, p in enumerate(need2) if p is not None]
    order = [scored[j] for j in captions.rank([need2[g] for g in scored])]
    place = {g: r + 1 for r, g in enumerate(order)}
    truth = [g for g, h in enumerate(hand) if h is not None]
    k_keep = budget_k(DENSITY, n)
    tp = len(set(order[:k_keep]) & set(truth))

    span_n = exact = overlap = top3 = pick_none = agree = agree_of = 0
    top1 = []
    for g, group in enumerate(groups):
        i = g % size
        by_id = {cid: span for cid, span, _text in captions.spans(group["words"])}
        pick = None
        if fwds[g].ok:
            cid, _p = choice_top(fwds[g], f"pick_{i}")
            pick = by_id.get(cid)
            if revs[g].ok:
                agree_of += 1
                agree += choice_top(revs[g], f"pick_{i}")[0] == cid
        top1.append(list(pick) if pick else None)
        if hand[g] is None or not fwds[g].ok:
            continue
        want = tuple(hand[g])
        span_n += 1
        exact += pick == want
        overlap += _overlap(pick, want)
        pick_none += pick is None
        probs = choice_probs(fwds[g], f"pick_{i}")
        likely = sorted((cid for cid in probs if cid in by_id), key=lambda cid: -probs[cid])[:3]
        top3 += want in [by_id[cid] for cid in likely]
    stats = {"name": name, "groups": n, "scored": len(scored), "k": k_keep, "hand": len(truth), "tp": tp,
             "precision": tp / k_keep if k_keep else 0.0,
             "hand_ranks": sorted(place[g] for g in truth if g in place),
             "hand_unranked": sum(g not in place for g in truth),
             "span": {"n": span_n, "exact": exact, "overlap": overlap, "top3": top3, "pick_none": pick_none},
             "order_agree": {"agree": agree, "of": agree_of}}
    return stats, top1


def _score_highlight(qs, sets, reqs, answers):
    out, items, start = [], {}, 0
    for name, ds, count in sets:
        stats, top1 = _score_set_highlight(qs, name, ds, answers[start:start + count])
        start += count
        stats["gates"] = dict(ds.get("gates") or {})
        out.append(stats)
        items[name] = top1
    complete = not _errors(answers)
    gates = []
    for s in out:
        if "precision_at_k_min" in s["gates"]:
            need = s["gates"]["precision_at_k_min"]
            gates.append(_gate(f"{s['name']} precision at K", f"{s['tp']}/{s['k']} = {s['precision']:.2f}",
                               f">= {need:.2f}", s["precision"] >= need - EPS, complete))
        if "hand_rank_max" in s["gates"]:
            top, ranks = s["gates"]["hand_rank_max"], s["hand_ranks"]
            gates.append(_gate(f"{s['name']} hand ranks", ", ".join(map(str, ranks)) or "none", f"all <= {top}",
                               bool(ranks) and max(ranks) <= top and not s["hand_unranked"], complete))
    overlap, hand_n = sum(s["span"]["overlap"] for s in out), sum(s["span"]["n"] for s in out)
    total = sum(s["hand"] for s in out)
    parity = [_parity("span overlap, hand groups", f"{overlap}/{total}", f"{total}/{total}",
                      overlap >= total - 1 and hand_n == total, complete)]
    return _result("highlight", answers, reqs, metrics={"density": DENSITY, "sets": out}, gates=gates, parity=parity,
                   items={"highlight": items})


def eval_highlight(client, *, qs=None):
    """CAP-02 and CAP-03 on every highlight set (hand groups, as frozen): precision at K = budget_k(0.25, groups) by
    forward `need2`, where the hand highlights rank, the forward pick against the hand span (exact, overlap, top 3;
    `none` excluded) and forward/reverse order agreement. Gates, per set: `precision_at_k_min` and/or `hand_rank_max`."""
    reqs, finish = _plan_highlight(qs)
    return finish(client.run_many(reqs))


# ---------------------------------------------------------------- crop (AST-01)
def _plan_crop(qsets=None, *, with_box=True):
    qsets = qsets or {}
    manifest = _read(EVALSETS_DIR / "crops" / "manifest.json")
    cases = manifest["cases"]
    wants = {c["id"]: c["want"] for c in cases}
    reqs, groups = [], []
    for case in cases:
        frame = EVALSETS_DIR / "crops" / case["frame"]
        present = assets.crop_requests(frame, case["want"], "public", qsets)
        absent = assets.crop_requests(frame, wants[case["absent_want_of"]], "public", qsets)[0]
        use = [present[0], present[1], absent] if with_box else [present[0], absent]
        groups.append((case, len(use)))
        reqs += use

    def finish(answers):
        return _score_crop(manifest, groups, reqs, answers, with_box)
    return reqs, finish


def _score_crop(manifest, groups, reqs, answers, with_box):
    cases, policy = manifest["cases"], load_policy()
    scale_range = manifest["focus_scale"]
    iou_min = manifest["gates"]["mean_iou_min"]
    reference = manifest.get("local_reference") or {}
    present_p, absent_p, rows, start = [], [], [], 0
    for case, count in groups:
        got = answers[start:start + count]
        start += count
        presence, absent = got[0], got[-1]
        row = {"id": case["id"], "vis_p": noul_p(presence, "vis") if presence.ok else None,
               "absent_p": noul_p(absent, "vis") if absent.ok else None,
               "iou": None, "centre_inside": None, "found": None}
        if row["vis_p"] is not None and row["absent_p"] is not None:
            present_p.append(row["vis_p"])
            absent_p.append(row["absent_p"])
        if with_box:
            fw, fh, _kind = image_size(EVALSETS_DIR / "crops" / case["frame"])
            src_w, src_h = case["src_size"]
            decision = assets.crop_decide(
                got[:2], fit=Fit(src_w, src_h, fw / src_w, (0, 0, 0, 0), fw, fh), body=case["body"],
                scale_range=scale_range, target={"kind": "shot", "id": case["id"]}, asset_id=case["id"], t=case["t"],
                want=case["want"], policy=policy)
            value = decision["value"]
            row["found"] = value["found"]
            if value["found"] is True:
                row.update(iou=geometry.iou(value["box_px"], case["hand"]),
                           centre_inside=geometry.centre_inside(value["box_px"], case["hand"]),
                           scale=value["focus"]["scale"], clamped=value["clamped"])
            elif value["found"] is False:
                row.update(iou=0.0, centre_inside=False)
        rows.append(row)
    scored = [r for r in rows if r["iou"] is not None]
    auc = _auc(present_p, absent_p)
    mean_iou = statistics.mean(r["iou"] for r in scored) if scored and with_box else None
    inside = sum(1 for r in scored if r["centre_inside"])
    complete = not _errors(answers) and len(present_p) == len(cases) and (not with_box or len(scored) == len(cases))
    notes = [] if with_box else ["n/a: no Box on local (presence only)"]
    metrics = {"cases": rows, "mean_iou": mean_iou, "centre_inside": inside, "n": len(cases), "presence_auc": auc}
    if with_box:
        gates = [_gate("mean IoU", f"{mean_iou:.2f}" if mean_iou is not None else "n/a", f">= {iou_min:.2f}",
                       mean_iou is not None and mean_iou >= iou_min, complete),
                 _gate("centre inside hand rect", f"{inside}/{len(cases)}", f"{len(cases)}/{len(cases)}",
                       inside == len(cases), complete)]
    else:
        gates = []
    n_present = len(present_p)
    parity = []
    if "presence_auc" in reference:
        ref = reference["presence_auc"]
        parity.append(_parity("presence AUC", f"{auc:.2f}" if auc is not None else "n/a", f"{ref:.2f}",
                              auc is not None and auc >= ref - 1 / max(n_present, 1), len(present_p) == len(cases)
                              and not _errors(answers)))
    return _result("crop_box", answers, reqs, metrics=metrics, gates=gates, parity=parity, items={}, notes=notes)


def eval_crop(client, *, qsets=None, with_box=True):
    """AST-01 on the frozen crops: per case IoU of the expanded Box against the hand rectangle (before the focus clamp),
    whether its centre is inside the hand rectangle, and presence AUC over the present and absent targets.
    `with_box=False` (a `local` run) skips Box and scores presence only. Gates: mean IoU >= the set's `mean_iou_min` and
    the centre inside on every case."""
    reqs, finish = _plan_crop(qsets, with_box=with_box)
    return finish(client.run_many(reqs))


# ---------------------------------------------------------------- stat (STY-06)
def _plan_stat(qs=None):
    qs = qs or load_qset(claims.ClaimsModule.QSET)
    sentences = _read(EVALSETS_DIR / "stat_check" / "sentences.json")
    labelled = [(s["id"], s["text"], s["measured"]) for s in sentences["labelled"]]
    controls = [(f"c{i + 1}", s["text"], s["measured"]) for i, s in enumerate(sentences["controls"])]
    reqs = claims.stat_requests([(sid, text) for sid, text, _m in labelled + controls], qs)

    def finish(answers):
        return _score_stat(qs, labelled, controls, reqs, answers, sentences.get("local_reference") or {})
    return reqs, finish


def _score_stat(qs, labelled, controls, reqs, answers, reference):
    p = [noul_p(a, "stat_is_measured") if a.ok else None for a in answers]
    th = thresholds_for(qs, next((a.backend for a in answers if a.ok), "jev"))
    in_band, labelled_rows = 0, []
    for (sid, _text, measured), v in zip(labelled, p[:len(labelled)], strict=True):
        ok = v is not None and (v >= th["source_at"] - EPS if measured else v < th["opinion_below"] - EPS)
        in_band += ok
        labelled_rows.append({"id": sid, "measured": measured, "stat_is_measured": v, "in_band": ok})
    correct, pos, neg, rows = 0, [], [], []
    for (sid, text, measured), v in zip(controls, p[len(labelled):], strict=True):
        if v is not None:
            correct += (v >= STAT_CONTROL_CUT) == measured
            (pos if measured else neg).append(v)
        rows.append({"id": sid, "measured": measured, "stat_is_measured": v})
    complete = not _errors(answers) and None not in p
    metrics = {"labelled": labelled_rows, "labelled_in_band": in_band, "opinion_below": th["opinion_below"],
               "source_at": th["source_at"], "controls_correct": correct, "controls": len(controls),
               "auc": _auc(pos, neg), "rows": rows}
    gates = [_gate("labelled sentences in band", f"{in_band}/{len(labelled)}", f"{len(labelled)}/{len(labelled)}",
                   in_band == len(labelled), complete),
             _gate("controls correct at 0.5", f"{correct}/{len(controls)}", f"{len(controls)}/{len(controls)}",
                   correct == len(controls), complete)]
    parity = []
    if "controls_correct" in reference:
        ref = reference["controls_correct"]
        parity.append(_parity("controls correct", f"{correct}/{len(controls)}", f"{ref}/{len(controls)}",
                              correct >= ref - 1, complete))
    items = {"stat": {sid: v for (sid, _t, _m), v in zip(labelled + controls, p, strict=True) if v is not None}}
    return _result("stat_check", answers, reqs, metrics=metrics, gates=gates, parity=parity, items=items)


def eval_stat(client, *, qs=None):
    """STY-06 on the labelled sentences and the controls: each labelled sentence on its side of the stat_check bands
    (an opinion below `opinion_below`, a measured figure at or above `source_at`), controls on the right side of 0.5, AUC
    over the controls. Gates: every labelled sentence in band and every control correct."""
    reqs, finish = _plan_stat(qs)
    return finish(client.run_many(reqs))


# ---------------------------------------------------------------- offline replay of the scripted answers
def load_canned(fixtures=FIXTURES_DIR):
    """The four answer fixtures, loaded: `{"qa", "highlight", "crop", "stat"}`."""
    fixtures = Path(fixtures)
    return {key: _read(fixtures / f"{key_file}_answers.json")
            for key, key_file in (("qa", "qa"), ("highlight", "highlight"), ("crop", "crop"), ("stat", "stat"))}


def canned_responder(canned):
    """A responder for `FakeBackend` that answers each eval request from the scripted answers, found by what identifies
    the request: the frame file name, the texts of the caption groups, the crop case and its `want`, the sentence. The
    reverse highlight request gets the forward answers. A request with no canned answer raises `BackendError`, so that
    metric comes out UNVERIFIED rather than as a pass."""
    wants = {c["id"]: c for c in _read(EVALSETS_DIR / "crops" / "manifest.json")["cases"]}
    by_groups = {tuple(r["groups"]): r["answers"] for r in canned["highlight"]["records"]}

    def answers_for(req):
        qs_id = req.qs.id
        image = next(iter(req.images.values()), None)
        if qs_id == qa.QSET:
            table = canned["qa"]["frames"][image.name]
            return {qid: fake.noul(table[qid]) for qid in req.questions}
        if qs_id == captions.QSET:
            return by_groups[tuple(g["text"] for g in req.state["groups"])]
        if qs_id in ("crop_box", "presence"):
            case = canned["crop"]["cases"][image.stem]
            asked = req.questions["vis"]["instructions"]
            is_present = asked == req.qs.question("vis", want=wants[image.stem]["want"])["instructions"]
            out = {"vis": fake.noul(case["vis"] if is_present else case["absent_vis"])}
            if "box" in req.questions:
                b = case["box"]
                out["box"] = fake.box(b["left"], b["top"], b["right"], b["bottom"], b.get("confidence") or 0.6)
            return out
        if qs_id == claims.ClaimsModule.QSET:
            row = canned["stat"]["sentences"][req.state["scene"]["words"]]
            return {qid: fake.noul(row[qid]) for qid in req.questions}
        raise KeyError(qs_id)

    def responder(req, model):
        try:
            answers = answers_for(req)
        except (KeyError, AttributeError) as exc:
            raise BackendError(f"no canned answer for {req.qs.id} {req.tag or ''} ({exc!r})") from None
        return {"answers": json.loads(json.dumps(answers)), "model": model, "input_tokens": 0, "latency_s": 0.0}
    return responder


def offline_client(canned=None, cache_dir=None, *, force_backend="jev", log=print):
    """A `Client` over two fake backends that replay `canned` (default: the fixtures on disk). `cache_dir` must be a
    folder the caller owns (a temp dir); the tool's `.cache/` is never used."""
    responder = canned_responder(canned or load_canned())
    backends = {"jev": fake.FakeBackend("jev", responder), "local": fake.FakeBackend("local", responder, version="0.1.0")}
    return Client(backends, Cache(cache_dir), force_backend=force_backend, log=log)


# ---------------------------------------------------------------- the run
def _word(ok):
    return "UNVERIFIED" if ok is None else "pass" if ok else "FAIL"


def _selected(qset, module):
    """The set ids to run, in run order: `--qset` and `--module` together select the union; neither selects all four."""
    chosen = set()
    for q in ([qset] if isinstance(qset, str) else list(qset or [])):
        if q not in SETS:
            raise ValueError(f"unknown --qset {q!r}; choose from {', '.join(SETS)}")
        chosen.add(q)
    for m in ([module] if isinstance(module, str) else list(module or [])):
        if m not in MODULES:
            raise ValueError(f"unknown --module {m!r}; choose from {', '.join(MODULES)}")
        chosen.add(MODULES[m])
    return [s for s in SETS if s in chosen] if chosen else list(SETS)


def _bad(message):
    print(f"eval: {message}", file=sys.stderr)
    return 2


def _summary(results, backend):
    """The markdown summary table: gates on `jev`, parity rows against each set's `local_reference` on `local`."""
    if backend == "jev":
        lines = ["| set | metric | value | gate | result |", "| --- | --- | --- | --- | --- |"]
        for res in results.values():
            lines += [f"| {res['title']} | {g['name']} | {g['value']} | {g['need']} | {_word(g['ok'])} |"
                      for g in res["gates"]]
        return lines
    lines = ["| set | metric | value | reference | within one item |", "| --- | --- | --- | --- | --- |"]
    for res in results.values():
        note = f" ({res['cut']})" if res.get("cut") else ""
        lines += [f"| {res['title']}{note} | {g['name']} | {g['value']} | {g['reference']} | {_word(g['ok'])} |"
                  for g in res["parity"]]
        lines += [f"| {res['title']} | {n} | | | |" for n in res["notes"]]
    return lines


def _details(results):
    lines = []
    qa_res = results.get(qa.QSET)
    if qa_res:
        m = qa_res["metrics"]
        lines += ["", f"## {qa_res['title']}", "",
                  f"flag at {m['flag']:.2f}, check at {m['check']:.2f}, gate {m['gate']:.2f}; recall also at 0.50, "
                  f"`caption_on_face` gated on face and caption visible", "",
                  "| question | positives | recall@flag | recall@0.5 | false alarms (clean) | mean p clean |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for q in m["per_question"]:
            rec = "-" if q["positives"] == 0 else f"{q['recall_flag']}/{q['positives']}"
            half = "-" if q["positives"] == 0 else f"{q['recall_half']}/{q['positives']}"
            lines.append(f"| {q['question']} | {q['positives'] or '-'} | {rec} | {half} "
                         f"| {q['false_alarms']}/{m['n_neg']} | {q['mean_p_neg']} |")
        lines += ["", f"Clean frames with any flag {m['clean_any_flag']}/{m['n_neg']} (any flag or check "
                      f"{m['clean_any_flag_or_check']}/{m['n_neg']}); planted frames with any flag "
                      f"{m['pos_any_flag']}/{m['n_pos']}, with their own question flagged "
                      f"{m['own_flagged']}/{m['n_pos']}."]
    hl = results.get("highlight")
    if hl:
        lines += ["", f"## {hl['title']}", "",
                  f"Rank by forward `need2`, keep K = budget_k({hl['metrics']['density']}, groups).", "",
                  "| set | groups | K | hits | P@K | hand ranks | span exact | overlap | top 3 | order agree |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for s in hl["metrics"]["sets"]:
            sp, oa = s["span"], s["order_agree"]
            lines.append(f"| {s['name']} | {s['groups']} | {s['k']} | {s['tp']}/{s['hand']} | {s['precision']:.2f} "
                         f"| {', '.join(map(str, s['hand_ranks']))} of {s['groups']} | {sp['exact']}/{sp['n']} "
                         f"| {sp['overlap']}/{sp['n']} | {sp['top3']}/{sp['n']} | {oa['agree']}/{oa['of']} |")
    cr = results.get("crop_box")
    if cr:
        lines += ["", f"## {cr['title']}", ""]
        if cr["notes"]:
            lines += [f"{n}" for n in cr["notes"]] + [""]
        lines += ["| case | vis p | absent p | IoU | centre inside | scale | clamped |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for c in cr["metrics"]["cases"]:
            iou = "n/a" if c["iou"] is None else f"{c['iou']:.2f}"
            inside = "n/a" if c["centre_inside"] is None else ("yes" if c["centre_inside"] else "no")
            lines.append(f"| {c['id']} | {c['vis_p']} | {c['absent_p']} | {iou} | {inside} | {c.get('scale', '')} "
                         f"| {', '.join(c.get('clamped') or []) or '-'} |")
        auc = cr["metrics"]["presence_auc"]
        lines += ["", f"Presence AUC {'n/a' if auc is None else f'{auc:.2f}'} over {cr['metrics']['n']} present and "
                      f"{cr['metrics']['n']} absent targets."]
    st = results.get("stat_check")
    if st:
        m = st["metrics"]
        auc = m["auc"]
        shown = "; ".join(f"{r['id']} {r['stat_is_measured']} ({'measured' if r['measured'] else 'opinion'}, "
                          f"{'in band' if r['in_band'] else 'OUT of band'})" for r in m["labelled"]) or "none"
        lines += ["", f"## {st['title']}", "", f"Labelled `stat_is_measured` (opinion below {m['opinion_below']:.2f}, "
                  f"measured at {m['source_at']:.2f} or above): {shown}. Controls correct at 0.5 "
                  f"{m['controls_correct']}/{m['controls']}; AUC over the controls {'n/a' if auc is None else f'{auc:.2f}'}."]
    return lines


def _flat(report):
    """The numbers the parity table compares, from one run's results."""
    res = report["results"]
    flat = {}
    if "stat_check" in res:
        m = res["stat_check"]["metrics"]
        flat.update(stat_controls=m["controls_correct"], stat_n=m["controls"], stat_labelled=m["labelled_in_band"],
                    stat_labelled_n=len(m["labelled"]))
    if "highlight" in res:
        sets = res["highlight"]["metrics"]["sets"]
        flat.update(span_overlap=sum(s["span"]["overlap"] for s in sets), span_n=sum(s["hand"] for s in sets),
                    topk_hits=sum(s["tp"] for s in sets), topk_n=sum(s["k"] for s in sets))
    if "crop_box" in res:
        m = res["crop_box"]["metrics"]
        flat.update(presence_auc=m["presence_auc"], crop_iou=m["mean_iou"])
    if qa.QSET in res:
        m = res[qa.QSET]["metrics"]
        flat.update(qa_clean=m["clean_any_flag"], qa_clean_n=m["n_neg"], qa_own=m["own_flagged"], qa_own_n=m["n_pos"])
    return flat


def _references(*reports):
    """`{(set, parity name): reference}` from the parity rows of the given reports (the first one wins)."""
    out = {}
    for report in reports:
        for key, res in report["results"].items():
            for row in res.get("parity") or []:
                out.setdefault((key, row["name"]), row.get("reference", ""))
    return out


def _agreement(mine, other):
    """Answer agreement between two runs over the items both answered: QA flag pairs at 0.7, stat on the same side of
    0.5, highlight on the same top-1 span."""
    a, b = mine["items"], other["items"]
    out = {}
    if a.get("qa") and b.get("qa"):
        same = pairs = 0
        for frame in a["qa"].keys() & b["qa"].keys():
            for name in qa.DEFECTS:
                x, y = a["qa"][frame].get(name), b["qa"][frame].get(name)
                if x is not None and y is not None:
                    pairs += 1
                    same += (x >= AGREE_QA_CUT - EPS) == (y >= AGREE_QA_CUT - EPS)
        out["QA flag pairs at 0.7"] = (same, pairs)
    if a.get("stat") and b.get("stat"):
        ids = a["stat"].keys() & b["stat"].keys()
        out["stat same side of 0.5"] = (sum((a["stat"][i] >= 0.5) == (b["stat"][i] >= 0.5) for i in ids), len(ids))
    if a.get("highlight") and b.get("highlight"):
        same = total = 0
        for name in a["highlight"].keys() & b["highlight"].keys():
            for x, y in zip(a["highlight"][name], b["highlight"][name]):
                total += 1
                same += x == y
        out["highlight same top-1 span"] = (same, total)
    return out


def _parity_table(report, other):
    """The parity table: each backend's gate metric, answer agreement, per-request latency and serial wall, with the
    sets' `local_reference` numbers beside them."""
    mine_b = report["meta"]["backend"]
    first, second = (report, other) if mine_b == "jev" else (other, report)
    a, b = _flat(first), _flat(second)
    ref = _references(first, second)

    def frac(f, n, d):
        return f"{f[n]}/{f[d]}" if n in f else "-"

    def num(f, k):
        return "-" if f.get(k) is None else f"{f[k]:.2f}"

    def timing(r):
        lat = [x for res in r["results"].values() for x in res["latency_s"]]
        return (f"{statistics.median(lat):.2f}", f"{sum(lat):.1f}") if lat else ("-", "-")
    t1, t2 = timing(first), timing(second)
    lines = ["", "## jev against local: parity", "",
             "| metric | jev | local | reference |", "| --- | --- | --- | --- |",
             f"| stat controls correct | {frac(a, 'stat_controls', 'stat_n')} | {frac(b, 'stat_controls', 'stat_n')} "
             f"| {ref.get(('stat_check', 'controls correct'), '')} |",
             f"| stat labelled in band | {frac(a, 'stat_labelled', 'stat_labelled_n')} "
             f"| {frac(b, 'stat_labelled', 'stat_labelled_n')} | |",
             f"| highlight span overlap, hand groups | {frac(a, 'span_overlap', 'span_n')} | {frac(b, 'span_overlap', 'span_n')} "
             f"| {ref.get(('highlight', 'span overlap, hand groups'), '')} |",
             f"| presence AUC | {num(a, 'presence_auc')} | {num(b, 'presence_auc')} "
             f"| {ref.get(('crop_box', 'presence AUC'), '')} |",
             f"| QA clean frames flagged at 0.7 | {frac(a, 'qa_clean', 'qa_clean_n')} | {frac(b, 'qa_clean', 'qa_clean_n')} "
             f"| {ref.get((qa.QSET, 'clean frames flagged'), '')} |",
             f"| QA planted frames flagged by own question | {frac(a, 'qa_own', 'qa_own_n')} | {frac(b, 'qa_own', 'qa_own_n')} | |",
             f"| highlight hits in the top K | {frac(a, 'topk_hits', 'topk_n')} | {frac(b, 'topk_hits', 'topk_n')} | |",
             f"| crop mean IoU (Box is Jev only) | {num(a, 'crop_iou')} | {num(b, 'crop_iou')} | |",
             f"| per-request latency, median (s) | {t1[0]} | {t2[0]} | |",
             f"| serial wall (s) | {t1[1]} | {t2[1]} | |"]
    agree = _agreement({"items": first["items"]}, {"items": second["items"]})
    if agree:
        lines += ["", "Answer agreement between the two backends:", ""]
        lines += [f"- {name}: {same}/{total} ({100 * same / total:.0f}%)" for name, (same, total) in agree.items() if total]
    return lines


def _find_other(out_dir, date, backend, offline, sets):
    """The newest report JSON of the other backend written on `date` by a run of the same kind (live or offline) that
    covered every set in `sets`."""
    other = "local" if backend == "jev" else "jev"
    found = []
    for path in Path(out_dir).glob(f"{date}-*-{other}-*.json"):
        try:
            data = _read(path)
        except (OSError, ValueError):
            continue    # an empty or half-written file: another run has claimed the name and not filled it yet
        meta = data.get("meta") or {}
        if (meta.get("backend") == other and bool(meta.get("offline")) == offline and "results" in data
                and set(sets) <= set(meta.get("sets") or ())):
            found.append((path.stat().st_mtime, path.name, data))
    return max(found, key=lambda t: t[:2])[2] if found else None


def _now():
    """The local wall-clock time a run started at. A seam for tests."""
    return datetime.datetime.now()


def _scope(explicit, qsets_run):
    """`all` when no set was selected, else the question set ids that run (crop_box brings presence), sorted, joined by `+`."""
    if not explicit:
        return "all"
    return "+".join(sorted({q.id for qsets in qsets_run for q in qsets}))


def _stem(date, hms, backend, scope, offline):
    """`<date>-<HHMMSS>-<backend>-<scope>[-offline]`, only `[A-Za-z0-9._+-]`."""
    return re.sub(r"[^A-Za-z0-9._+-]", "_", f"{date}-{hms}-{backend}-{scope}") + ("-offline" if offline else "")


def _claim(out, stem):
    """Create the empty `.json` and `.md` of the first free name (`stem`, then `stem-2`, `stem-3`...) and return that stem.
    A name is free only when neither file exists, and each file is made with `x` so a run never overwrites another's."""
    n = 1
    while True:
        name = stem if n == 1 else f"{stem}-{n}"
        made = []
        try:
            for ext in ("json", "md"):
                path = out / f"{name}.{ext}"
                with open(path, "x", encoding="utf-8"):
                    made.append(path)
            return name
        except FileExistsError:
            for path in made:
                path.unlink(missing_ok=True)
            n += 1


def run_eval(*, qset=None, module=None, backend=None, model=None, dry_run=False, budget=0.10,
             client=None, out_dir=EVAL_OUT_DIR, today=None, offline=False):
    """`jevm eval`: run the selected sets, print the summary, write
    `<out_dir>/<date>-<HHMMSS>-<backend>-<scope>[-offline][-2].{md,json}` (start time, backend, the question set ids that ran
    or `all`; a taken name gets `-2`, `-3`, so no run overwrites another's report).

    `backend` None means `jev` (the client is forced to it, so a calibrated Decider never answers a Jev eval); `local`
    forces Decider, skips Box and labels sets with no local thresholds `at Jev's cut`. `model` replaces `models["jev"]`
    of the sets it runs. A `client` passed in keeps its own settings. `offline` only labels the files. Returns 0, 1 (a gate
    failed or could not be measured; on `local`, a parity number missed its `local_reference` by more than one item),
    2 (bad input) or 3 (over budget); a dry run returns 0 with nothing sent and nothing written.
    """
    started = _now()
    if backend not in (None, "jev", "local"):
        return _bad(f"--backend must be jev or local, not {backend!r}")
    try:
        selected = _selected(qset, module)
    except ValueError as exc:
        return _bad(str(exc))
    name = backend or "jev"

    try:
        plans = []     # (set id, title, qsets used, requests, finish)
        if qa.QSET in selected:
            qs = _with_model(load_qset(qa.QSET), model)
            plans.append((qa.QSET, [qs], *_plan_qa(qs)))
        if "highlight" in selected:
            qs = _with_model(load_qset(captions.QSET), model)
            plans.append(("highlight", [qs], *_plan_highlight(qs)))
        if "crop_box" in selected:
            qsets = {q: _with_model(load_qset(q), model) for q in ("crop_box", "presence")}
            plans.append(("crop_box", list(qsets.values()), *_plan_crop(qsets, with_box=name != "local")))
        if "stat_check" in selected:
            qs = _with_model(load_qset(claims.ClaimsModule.QSET), model)
            plans.append(("stat_check", [qs], *_plan_stat(qs)))
    except (OSError, ValueError, KeyError) as exc:
        return _bad(f"cannot build the eval requests ({exc!r}); are the eval sets in place under {EVALSETS_DIR}?")

    if client is None:
        client = make_client(force_backend=name)
    reqs = [r for _key, _qs, rs, _fin in plans for r in rs]
    stop = client.preflight(reqs, budget_usd=budget, dry_run=dry_run)
    if stop is not None:
        return stop

    answers, results, start = client.run_many(reqs), {}, 0
    for key, qsets, rs, finish in plans:
        res = finish(answers[start:start + len(rs)])
        start += len(rs)
        res["title"] = " + ".join(f"{q.id}@{q.version}" for q in qsets) + (f" ({', '.join(res['models'])})" if res["models"] else "")
        if name == "local":
            ran = [q for q in qsets if not (key == "crop_box" and q.id == "crop_box")]   # no Box on local
            res["cut"] = "at the local cut" if all(q.has_local_calibration() for q in ran) else "at Jev's cut"
        results[key] = res

    if name == "jev":
        failed = [g for res in results.values() for g in res["gates"] if g["ok"] is not True]
    else:
        failed = [g for res in results.values() for g in res["parity"] if g["ok"] is not True]
    exit_code = 1 if failed else 0

    today = today or started.date().isoformat()
    report = {"meta": {"date": today, "backend": name, "model_override": model, "offline": bool(offline),
                       "sets": selected, "budget_usd": budget, "exit": exit_code,
                       "backends_used": client.ledger.to_dict()},
              "results": results, "items": {k: v for res in results.values() for k, v in res["items"].items()}}
    head = [f"# jevm eval: {name}, {model or 'pinned models'}, {today}", ""]
    if offline:
        head += ["OFFLINE: scripted answers replayed from tests/fixtures/eval through fake backends. This checks the "
                 "harness on the synthetic eval sets; it is not a measurement of Jev or Decider.", ""]
    used = report["meta"]["backends_used"]["jev"] if name == "jev" else report["meta"]["backends_used"]["local"]
    if offline:
        head += [f"{len(reqs)} requests replayed; no tokens or dollars are counted", ""]
    else:
        head += [f"{len(reqs)} requests, {used['requests']} live, {used['cache_hits']} cached"
                 + (f", {used['input_tokens']} input tokens, ${used['usd']:.4f}" if name == "jev"
                    else f", {used['serial_s']} s serial"), ""]
    summary = _summary(results, name)
    if not failed:
        verdict = "all gates pass" if name == "jev" else "every parity number within one item of its reference"
    else:
        why = "gate not met" if name == "jev" else "outside one item of its reference"
        verdict = f"{why}: " + "; ".join(f"{g['name']} {_word(g['ok'])}" for g in failed)
    lines = head + ["## Summary", ""] + summary + ["", f"Result: {verdict}."] + _details(results)
    for res in results.values():
        for key, n in res["errors"].items():
            lines.append(f"\n{res['title']}: {n} request(s) failed with `{key}`; the metrics above leave them out.")
    other = _find_other(out_dir, today, name, bool(offline), selected)
    if other:
        lines += _parity_table(report, other)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scope = _scope(qset or module, [qsets for _key, qsets, _rs, _fin in plans])
    stem = _claim(out, _stem(today, started.strftime("%H%M%S"), name, scope, offline))
    (out / f"{stem}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(["", f"eval {name}{' (offline replay)' if offline else ''}: {today}", ""] + summary + ["", f"Result: {verdict}."]))
    print(f"eval: wrote {out / (stem + '.md')} and {stem}.json")
    return exit_code


# ---------------------------------------------------------------- the command
def _parser():
    p = argparse.ArgumentParser(prog="jevm eval", description="Run the frozen eval sets through the decision modules.")
    p.add_argument("--qset", choices=SETS, help="run one question set (crop_box also runs presence)")
    p.add_argument("--module", choices=sorted(MODULES), help="run the set a module owns")
    p.add_argument("--backend", choices=("jev", "local"), help="force one backend (default: jev)")
    p.add_argument("--model", help="replace the Jev model pin of the sets it runs")
    p.add_argument("--dry-run", action="store_true", help="print the estimate and send nothing")
    p.add_argument("--budget", type=float, default=0.10, help="stop when the Jev estimate is over this many dollars")
    p.add_argument("--out", type=Path, default=EVAL_OUT_DIR, help="folder for the .md and .json (default: eval/)")
    p.add_argument("--offline", action="store_true", help="replay the scripted answers; no network, no key")
    return p


def main(argv=None):
    """`python3 -m evals.harness [options]` or `jevm eval [options]`. Returns the exit code, never raises SystemExit."""
    try:
        ns = _parser().parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    kwargs = dict(qset=ns.qset, module=ns.module, backend=ns.backend, model=ns.model, dry_run=ns.dry_run,
                  budget=ns.budget, out_dir=ns.out)
    if not ns.offline:
        return run_eval(**kwargs)
    with tempfile.TemporaryDirectory() as td:
        return run_eval(**kwargs, client=offline_client(cache_dir=td, force_backend=ns.backend or "jev"), offline=True)


if __name__ == "__main__":
    sys.exit(main())
