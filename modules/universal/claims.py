"""Claims: is a spoken number a measured or sourced stat (STY-06), and does a stat's quote appear in a source (QA-05).

STY-06 asks two Nouls per sentence that holds a number hint (question set `stat_check`).
Below `opinion_below` the number is an opinion or figure of speech (caption only); at or above `source_at` it needs a
source; in between the agent reads it. QA-05's code half then looks for the sentence in the spec's source texts. Both
decisions are never-auto (docs/never-auto.md). `decide()` reads answers only and never touches the network.
"""
import html
import re

from core.module import Module
from core.policy import decision_policy
from core.rows import code_evidence, error_route, evidence_from, make_row
from jevlib.paths import project_root
from jevlib.request import Request, load_qset, noul_p, thresholds_for

_NUMBER_WORDS = ("one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty sixty seventy "
                 "eighty ninety hundred thousand million billion percent half double twice").split()
_HINT = re.compile(r"\d|×|\b(?:" + "|".join(_NUMBER_WORDS) + r")\b", re.IGNORECASE)
_SOURCE_SUFFIXES = (".txt", ".md", ".html", ".htm")
_HTML_TAG = re.compile(r"<[^>]*>")
_UNIFY = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-", "−": "-", "×": "x"})


def has_number_hint(text):
    """True when a sentence holds a digit, a multiplication sign or a number word: a cheap filter before any request."""
    return _HINT.search(text) is not None


def stat_requests(sentences, qs=None):
    """One request per `(sentence id, text)` pair: both Nouls over `{"scene": {"words": text}}`, in one request."""
    qs = qs or load_qset(ClaimsModule.QSET)
    questions = {qid: qs.question(qid) for qid in ("has_number", "stat_is_measured")}
    return [Request(qs=qs, state={"scene": {"words": text}}, questions=dict(questions), tag=sid) for sid, text in sentences]


def normalise(text):
    """Lower-case, quotes and dashes unified, `x` for the multiplication sign, "<n> percent" as "<n>%", thousands commas
    dropped, other punctuation turned into spaces (keeping `%` and decimal points), whitespace collapsed."""
    s = text.lower().translate(_UNIFY)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = re.sub(r"(?<=\d)\s*percent\b", "%", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    s = re.sub(r"[^\w\s%.]", " ", s)
    return " ".join(s.split())


def numbers_in(text):
    """The digit-numbers of a text after `normalise`, in order, as strings ("12%" gives "12", "48,000" gives "48000")."""
    return re.findall(r"\d+(?:\.\d+)?", normalise(text))


def load_sources(spec):
    """`{asset id: text}` for each `source_text` asset that is a readable local `.txt`, `.md`, `.html` or `.htm` file
    (relative to the project root). Remote, missing and other-format assets are left out; HTML tags are stripped."""
    sources = {}
    for asset in spec.get("assets", []):
        uri = asset.get("uri", "")
        if asset.get("kind") != "source_text" or "://" in uri or not uri.lower().endswith(_SOURCE_SUFFIXES):
            continue
        path = project_root(spec) / uri
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if uri.lower().endswith((".html", ".htm")):
            text = html.unescape(_HTML_TAG.sub(" ", text))
        sources[asset["id"]] = text
    return sources


def quote_check(claim_text, sources):
    """Where a claim shows up in the sources: `{"match", "source", "numbers"}`. `quote_found`: the normalised claim is a
    substring of a source. `numbers_found`: every digit-number of the claim is in one source. `not_found` otherwise (the
    claim is not in the text, so it may be made up). `no_sources` when there is nothing to read."""
    numbers = numbers_in(claim_text)
    if not sources:
        return {"match": "no_sources", "source": None, "numbers": numbers}
    claim = normalise(claim_text)
    normalised = {sid: normalise(text) for sid, text in sources.items()}
    for sid, text in normalised.items():
        if claim and claim in text:
            return {"match": "quote_found", "source": sid, "numbers": numbers}
    if numbers:
        for sid, text in sources.items():
            found = set(numbers_in(text))
            if all(n in found for n in numbers):
                return {"match": "numbers_found", "source": sid, "numbers": numbers}
    return {"match": "not_found", "source": None, "numbers": numbers}


def _carry(own, answer):
    """A row's `route_reasons`: the module's own reasons, then the answer's (`backend_fallback`, say), de-duplicated."""
    return list(dict.fromkeys([*own, *answer.route_reasons]))


def _sentences(spec):
    """`(sentence dict, text)` for every sentence, the text joined from its words (ids `[first, last]`, inclusive)."""
    words = spec["transcript"]["words"]
    index = {w["id"]: i for i, w in enumerate(words)}
    out = []
    for s in spec["units"]["sentences"]:
        first, last = s["words"]
        out.append((s, " ".join(w["text"] for w in words[index[first]:index[last] + 1])))
    return out


class ClaimsModule(Module):
    id = "universal.claims"
    decisions = ("STY-06", "QA-05")
    QSET = "stat_check"

    def applies(self, spec, kit):
        return bool(spec.get("units", {}).get("sentences"))

    def _hinted(self, spec):
        return [(s, text) for s, text in _sentences(spec) if has_number_hint(text)]

    def requests(self, spec, kit, policy):
        return stat_requests([(s["id"], text) for s, text in self._hinted(spec)], load_qset(self.QSET))

    def decide(self, spec, kit, answers, policy):
        sty, qa = decision_policy(policy, "STY-06"), decision_policy(policy, "QA-05")
        sources = None
        rows = []
        for (sentence, text), ans in zip(self._hinted(spec), answers, strict=True):
            target = {"kind": "sentence", "id": sentence["id"], "t": [sentence["start"], sentence["end"]],
                      "words": list(sentence["words"])}
            if not ans.ok:
                lane, reasons = error_route(ans)
                rows.append(make_row(decision="STY-06", module=self.id, target=target, value={"class": None}, p=None,
                                     confidence=None, lane=lane, route_reasons=_carry(reasons, ans),
                                     evidence=evidence_from(ans, sty)))
                continue
            thr = thresholds_for(ans.request.qs, ans.backend)
            has_number, measured = noul_p(ans, "has_number"), noul_p(ans, "stat_is_measured")
            if has_number < thr["has_number"]:
                continue
            if measured < thr["opinion_below"]:
                cls, lane, reasons = "opinion", "agent", ["never_auto"]
            elif measured >= thr["source_at"]:
                cls, lane, reasons = "needs_source", "human", ["never_auto"]
            else:
                cls, lane, reasons = "unsure", "agent", ["in_review_band"]
            rows.append(make_row(decision="STY-06", module=self.id, target=target,
                                 value={"class": cls, "has_number": has_number, "stat_is_measured": measured},
                                 p=measured, confidence={"metric": "noul_p", "value": measured}, lane=lane,
                                 route_reasons=_carry(reasons, ans), evidence=evidence_from(ans, sty)))
            if cls == "needs_source":
                if sources is None:
                    sources = load_sources(spec)
                rows.append(make_row(decision="QA-05", module=self.id, target={**target, "kind": "claim"},
                                     value=quote_check(text, sources), p=None, confidence=None, lane="human",
                                     route_reasons=_carry(["never_auto"], ans), evidence=code_evidence(qa)))
        return rows
