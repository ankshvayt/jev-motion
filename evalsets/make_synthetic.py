"""Writes the synthetic eval sets in `evalsets/` and their scripted answers in `tests/fixtures/eval/`.

    python3 evalsets/make_synthetic.py        (from the tool root; stdlib only, no network, no key)

Everything here is made up: flat-colour PNG mock-ups drawn by `_png`, invented caption lines and sentences, and answers
scripted to look like a plausible model run (most items right, a few misses and false alarms). The answers exist so
`jevm eval --offline` and the tests can exercise the harness end to end; they are not output of Jev or Decider, and the
numbers they produce say nothing about either. Re-running this rewrites every file it owns with the same content.
"""
import json
import struct
import sys
import zlib
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1]
if str(TOOL) not in sys.path:
    sys.path.insert(0, str(TOOL))

from core.policy import load_policy  # noqa: E402
from modules.universal import captions, geometry  # noqa: E402

EVALSETS = TOOL / "evalsets"
ANSWERS = TOOL / "tests" / "fixtures" / "eval"
NOTE = "Scripted for the synthetic eval sets by evalsets/make_synthetic.py; not model output."

# A neutral palette (RGB).
SLATE, PANEL, INK, MUTED = (30, 36, 48), (238, 241, 245), (60, 66, 80), (150, 158, 172)
SKIN, ACCENT, WHITE, GREY = (200, 162, 122), (56, 189, 248), (250, 250, 250), (110, 114, 122)


def _png(path, w, h, rects, bg=SLATE):
    """A `w` x `h` RGB PNG: `bg` with each `(x0, y0, x1, y1, rgb)` filled in order (clipped to the frame)."""
    rows = [bytearray(bytes(bg) * w) for _ in range(h)]
    for x0, y0, x1, y1, rgb in rects:
        x0, x1 = max(0, int(x0)), min(w, int(x1))
        for y in range(max(0, int(y0)), min(h, int(y1))):
            if x1 > x0:
                rows[y][3 * x0:3 * x1] = bytes(rgb) * (x1 - x0)
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ frame QA (QA-01)
QW, QH = 288, 512
DEFECTS = ("text_occluded", "cut_off_edge", "caption_on_face", "empty_region", "overlapping_panels", "overlay_on_controls",
           "hard_to_read")
PLANTED = ("text_occluded", "cut_off_edge", "caption_on_face", "empty_region", "overlapping_panels", "overlay_on_controls")


def _caption(y=330):
    """Two short caption lines of 'words' (white blocks), one of them in the accent colour."""
    return [(40, y, 88, y + 14, WHITE), (94, y, 150, y + 14, ACCENT), (156, y, 230, y + 14, WHITE),
            (60, y + 22, 120, y + 36, WHITE), (126, y + 22, 210, y + 36, WHITE)]


def _window(x0=24, y0=70, x1=264, y1=280):
    return [(x0, y0, x1, y1, PANEL), (x0, y0, x1, y0 + 16, MUTED)] + [
        (x0 + 14, y, x1 - 30, y + 6, INK) for y in range(y0 + 34, y1 - 20, 22)]


def _speaker():
    return [(84, 380, 204, 500, SKIN)]


def _qa_frame(kind, i):
    if kind == "clean":
        layout = (_window(24, 70 + 6 * i, 264, 270 + 6 * i) if i % 3 else _speaker()) + _caption()
        return layout + (_speaker() if i % 4 == 1 else [])
    if kind == "text_occluded":
        return _window() + _caption() + [(70, 320, 200, 372, MUTED)]
    if kind == "cut_off_edge":
        return _window(120, 70, 330, 280) + _caption()
    if kind == "caption_on_face":
        return [(84, 300, 204, 420, SKIN)] + _caption(350)
    if kind == "empty_region":
        return _caption()
    if kind == "overlapping_panels":
        return _window(24, 60, 220, 240) + _window(56, 100, 252, 280) + _window(88, 140, 280, 320) + _caption(360)
    if kind == "overlay_on_controls":
        buttons = [(30 + 12 * k, 74, 38 + 12 * k, 82, WHITE) for k in range(3)]
        return _window() + buttons + [(34, 72, 96, 86, ACCENT)] + _caption()
    raise KeyError(kind)


def _qa_p(kind, i):
    """Scripted p per question: low on clean frames (two false alarms), high on the planted question (two misses)."""
    p = {name: round(0.02 + 0.01 * ((i + j) % 5), 2) for j, name in enumerate(DEFECTS)}
    p["face_visible"] = 0.96 if kind == "caption_on_face" or (kind == "clean" and i % 4 == 1) else 0.04
    p["caption_visible"] = 0.97
    if kind == "clean":
        if i == 2:
            p["hard_to_read"] = 0.74         # a false alarm
        if i == 5:
            p["empty_region"] = 0.72         # a false alarm
        if i == 6:
            p["text_occluded"] = 0.45        # a check-band value
        return p
    own = {1: 0.88, 2: 0.93}[i]
    if (kind, i) == ("overlapping_panels", 2):
        own = 0.55                           # a miss at the flag cut, a hit at 0.5
    if (kind, i) == ("cut_off_edge", 1):
        own = 0.35                           # a miss at both cuts
    p[kind] = own
    return p


def _tick(uid):
    """A small mark whose place is unique per frame: no two frames are byte-identical, so the answer cache (keyed on the
    image bytes) never serves one frame's answer for another."""
    return [(6 + 9 * uid, 6, 12 + 9 * uid, 12, MUTED)]


def make_qa():
    root = EVALSETS / "qa_frame.short"
    negatives, positives, answers = [], [], {}
    for i in range(8):
        t = round(1.5 * i, 1)
        name = f"neg_t{t:.1f}.png"
        _png(root / "neg" / name, QW, QH, _qa_frame("clean", i) + _tick(i))
        negatives.append({"file": f"neg/{name}", "t": t})
        answers[name] = _qa_p("clean", i)
    for kind in PLANTED:
        for i in (1, 2):
            t = round(1.5 * (2 * PLANTED.index(kind) + i), 1)
            name = f"{kind}_{i:02d}_t{t:.1f}.png"
            _png(root / "pos" / name, QW, QH, _qa_frame(kind, i) + _tick(8 + len(positives)))
            positives.append({"id": name[:-4], "type": kind, "source_t": t, "file": f"pos/{name}"})
            answers[name] = _qa_p(kind, i)
    _write(root / "manifest.json", {
        "note": "Synthetic: flat-colour mock-ups of a vertical Short, 8 clean frames and 2 planted defects per type.",
        "context": "A frame from a vertical Short in a dark slate style: a talking-head speaker, app windows inside light "
                   "frames, word-by-word captions in the lower third, and sky blue used only to highlight one or two "
                   "words of text.",
        "negatives": negatives, "positives": positives,
        "gates": {"own_flagged_min": 8, "clean_flagged_max": 4},
        "local_reference": {"clean_flagged": 6}})
    _write(ANSWERS / "qa_answers.json", {"source": NOTE, "frames": answers, "missing": []})


# ------------------------------------------------------------------------------------------- highlight (CAP-02/03)
HIGHLIGHT = {
    "set-a": {
        "lines": ["Here are three quick tips", "to tidy up your inbox.", "First, turn on smart filters",
                  "so newsletters skip the inbox.", "This tool saves 3 hours a week", "on sorting mail by hand.",
                  "Second, set one reply template", "for the questions you answer daily.",
                  "Third, archive everything older", "than thirty days in one click.", "That is it for today.",
                  "Follow for more quick tips."],
        "hand": {2: [3, 4], 4: [3, 4], 6: [3, 4], 9: [4, 5]},
        "need2": {2: 0.85, 4: 0.92, 6: 0.66, 9: 0.71, 7: 0.80, 0: 0.40},
        "gates": {"precision_at_k_min": 0.5},
    },
    "set-b": {
        "lines": ["Let me show you the export flow.", "Open the report you want to share,", "then click the share button",
                  "in the top right corner.", "Pick PDF from the format menu.", "The whole export takes five seconds",
                  "even for a fifty page report.", "You can also schedule it", "to send every Monday morning.",
                  "Add the people who should get it,", "write a short note,", "and press save.",
                  "Now the report arrives on its own", "and nobody has to ask for it.", "Try it on your next report",
                  "and see how much time it frees up."],
        "hand": {5: [4, 5], 12: [4, 6]},
        "need2": {5: 0.90, 12: 0.74, 8: 0.78, 2: 0.55, 6: 0.60},
        "gates": {"hand_rank_max": 4},
    },
}


def _pick(words, hand):
    """Scripted Choice probabilities over the group's spans: the hand span on top for a hand group, a two-word span
    otherwise, and a little mass on `none`."""
    ids = [(cid, span) for cid, span, _text in captions.spans(words)]
    want = tuple(hand) if hand else next((span for _cid, span in ids if span[1] - span[0] == 1), ids[0][1])
    probs = {cid: 0.0 for cid, _span in ids}
    top = next(cid for cid, span in ids if span == want)
    probs[top] = 0.62 if hand else 0.41
    second = next((cid for cid, _span in ids if cid != top), None)
    if second:
        probs[second] = 0.21
    probs["none"] = round(1.0 - sum(probs.values()), 2)
    return {"type": "choice", "choice": top, "confidence": probs[top], "probabilities": probs}


def make_highlight():
    module, policy = captions.CaptionsModule(), load_policy()
    records = []
    for name, ds in HIGHLIGHT.items():
        groups = [{"text": line, "words": line.split()} for line in ds["lines"]]
        hand = [ds["hand"].get(g) for g in range(len(groups))]
        data = {"name": name, "note": "Synthetic caption groups with one author's hand highlights.",
                "transcript": " ".join(ds["lines"]), "groups": groups, "hand": hand, "gates": ds["gates"]}
        _write(EVALSETS / "highlight" / f"{name}.json", data)
        # Build the requests the harness will send, so the scripted answers line up with the batches exactly.
        words, spec_groups = [], []
        for gi, group in enumerate(groups):
            first = len(words)
            for text in group["words"]:
                words.append({"id": f"w{len(words) + 1}", "text": text, "start": 0.0, "end": 0.0})
            spec_groups.append({"id": f"g{gi + 1}", "words": [words[first]["id"], words[-1]["id"]], "start": 0.0,
                                "end": 0.0, "lines": 1, "sentence": "s1"})
        spec = {"transcript": {"text": data["transcript"], "words": words},
                "units": {"sentences": [], "caption_groups": spec_groups},
                "constraints": {"highlight": {"density": 0.25}}}
        by_text = {g["text"]: gi for gi, g in enumerate(groups)}
        for req in module.requests(spec, None, policy)[::2]:          # forward requests; reverse ones share answers
            texts = [g["text"] for g in req.state["groups"]]
            answers = {}
            for i, text in enumerate(texts):
                gi = by_text[text]
                need2 = ds["need2"].get(gi, round(0.12 + 0.03 * (gi % 4), 2))
                answers[f"need_{i}"] = {"type": "noul", "noul": round(min(0.99, need2 + 0.05), 2)}
                answers[f"need2_{i}"] = {"type": "noul", "noul": need2}
                answers[f"pick_{i}"] = _pick(groups[gi]["words"], hand[gi])
            records.append({"groups": texts, "answers": answers})
    _write(ANSWERS / "highlight_answers.json", {"source": NOTE, "records": records, "missing": 0})


# --------------------------------------------------------------------------------------------------- crops (AST-01)
CW, CH = 640, 360
BODY = [920, 760]
CROPS = [
    ("c1-settings", (380, 60, 620, 330), "the settings panel with the toggle list and the blue save button", "c2-chart"),
    ("c2-chart", (40, 40, 360, 250), "the line chart card with its legend", "c3-table"),
    ("c3-table", (60, 150, 600, 340), "the invoice table with the total row", "c4-sidebar"),
    ("c4-sidebar", (0, 0, 150, 360), "the left sidebar with the folder list", "c1-settings"),
]


def _crop_frame(cid, target):
    x0, y0, x1, y1 = target
    rects = [(0, 0, CW, CH, PANEL), (0, 0, CW, 22, MUTED), (x0, y0, x1, y1, WHITE), (x0, y0, x1, y0 + 12, INK)]
    rects += [(x0 + 10, y, x1 - 10, y + 5, GREY) for y in range(y0 + 24, y1 - 10, 18)]
    if cid == "c1-settings":
        rects.append((x1 - 90, y1 - 34, x1 - 14, y1 - 14, ACCENT))
    return rects


def make_crops():
    cases, answers = [], {}
    for cid, target, want, absent in CROPS:
        name = f"{cid}.png"
        _png(EVALSETS / "crops" / "frames" / name, CW, CH, _crop_frame(cid, target), bg=PANEL)
        rect = geometry.expand(target, BODY[0] / BODY[1], CW, CH)
        dx, dy = 0.04 * (rect[2] - rect[0]), 0.03 * (rect[3] - rect[1])
        hand = [round(max(0.0, rect[0] + dx), 1), round(max(0.0, rect[1] - dy), 1),
                round(min(CW, rect[2] + dx), 1), round(min(CH, rect[3] - dy), 1)]
        cases.append({"id": cid, "frame": f"frames/{name}", "t": 0.0, "want": want, "body": BODY, "src_size": [CW, CH],
                      "hand": hand, "absent_want_of": absent})
        x0, y0, x1, y1 = target
        answers[cid] = {"box": {"left": round(x0 / CW, 3), "top": round(y0 / CH, 3), "right": round(x1 / CW, 3),
                                "bottom": round(y1 / CH, 3), "confidence": 0.7},
                        "vis": 0.97, "absent_vis": 0.02}
    _write(EVALSETS / "crops" / "manifest.json", {
        "note": "Synthetic app mock-ups. hand is one author's crop in source px; want was written before scoring.",
        "focus_scale": [1.25, 2.0], "cases": cases,
        "gates": {"mean_iou_min": 0.68}, "local_reference": {"presence_auc": 1.0}})
    _write(ANSWERS / "crop_answers.json", {"source": NOTE, "cases": answers, "missing": []})


# ---------------------------------------------------------------------------------------------------- stat (STY-06)
LABELLED = [("l1", "Now it runs 100x faster.", False), ("l2", "The release notes list 14 new features.", True)]
CONTROLS = [
    ("In the vendor's own benchmark report, the new model cut error rates from 12 percent to 7 percent.", True),
    ("The launch announcement says nine more partners are building on it.", True),
    ("The billing report shows the team sent 48,000 invoices in the first week.", True),
    ("The changelog says the context window grew from 200 thousand to 1 million tokens.", True),
    ("In my test of 50 invoices, it got 47 right.", True),
    ("This makes you 100x more productive.", False),
    ("It's ten times better than anything else out there.", False),
    ("I have a million reasons to try this.", False),
    ("This will 10x your business.", False),
    ("It took me about a thousand tries to get this working.", False),
]


def make_stat():
    sentences, answers = {"labelled": [], "controls": []}, {}
    for sid, text, measured in LABELLED:
        sentences["labelled"].append({"id": sid, "text": text, "measured": measured})
        answers[text] = {"has_number": 0.95, "stat_is_measured": 0.86 if measured else 0.08}
    for i, (text, measured) in enumerate(CONTROLS):
        sentences["controls"].append({"text": text, "measured": measured})
        answers[text] = {"has_number": 0.97, "stat_is_measured": round((0.78 + 0.03 * (i % 4)) if measured
                                                                       else (0.06 + 0.04 * (i % 4)), 2)}
    sentences["note"] = "Synthetic sentences: `labelled` are checked against the stat_check bands, `controls` at 0.5."
    sentences["local_reference"] = {"controls_correct": len(CONTROLS)}
    _write(EVALSETS / "stat_check" / "sentences.json", sentences)
    _write(ANSWERS / "stat_answers.json", {"source": NOTE, "sentences": answers, "missing": []})


if __name__ == "__main__":
    make_qa()
    make_highlight()
    make_crops()
    make_stat()
    print(f"wrote the synthetic eval sets under {EVALSETS.name}/ and their answers under tests/fixtures/eval/")
