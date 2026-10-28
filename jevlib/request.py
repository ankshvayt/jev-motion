"""Question sets, requests, answers and the accessors over them.

A question set is a versioned file `qsets/<id>@<version>.json`: wording, pinned models, thresholds per backend and
the route it may take. Editing any question bumps the version, because answers move when other questions in the same
request change. Modules build `Request`s; the client turns them into `Answer`s.
"""
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from jevlib.cache import file_sha256
from jevlib.paths import QSETS_DIR

SENSITIVITIES = ("public", "own_render", "private")
ROUTES = ("jev", "local", "either", "jev-for-speed")
REQUIRED_FIELDS = ("id", "version", "route", "backend", "models", "thresholds", "order_check", "batch",
                   "evidence", "questions")


@dataclass(frozen=True, kw_only=True)
class QuestionSet:
    id: str
    version: int
    route: str
    backend: str
    models: dict
    thresholds: dict
    calibrated: dict = field(default_factory=lambda: {"local": None})
    order_check: bool
    batch: int
    evidence: dict
    notes: str = ""
    questions: dict

    def question(self, template_id, **subs):
        """A deep copy of one question template with every `{name}` in every string replaced (keys included).
        Plain `str.replace`, never `str.format`, so any other brace in the wording survives."""
        def sub(obj):
            if isinstance(obj, str):
                for name, value in subs.items():
                    obj = obj.replace("{" + name + "}", str(value))
                return obj
            if isinstance(obj, dict):
                return {sub(k): sub(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sub(v) for v in obj]
            return obj
        return sub(copy.deepcopy(self.questions[template_id]))

    def has_local_calibration(self):
        """A set has one when it stores thresholds for the local backend (`thresholds.local` is not null)."""
        return self.thresholds.get("local") is not None


def load_qset(qset_id, version=None, root=QSETS_DIR):
    """Load `<root>/<qset_id>@<version>.json` (the highest version when `version` is None)."""
    root = Path(root)
    if version is None:
        found = {}
        for p in (root.iterdir() if root.is_dir() else ()):
            stem, _, ver = p.stem.rpartition("@")
            if stem == qset_id and p.suffix == ".json" and ver.isdigit():
                found[int(ver)] = p
        if not found:
            raise FileNotFoundError(f"no question set {qset_id!r} in {root}")
        path = found[max(found)]
    else:
        path = root / f"{qset_id}@{version}.json"
        if not path.is_file():
            raise FileNotFoundError(f"no question set {qset_id!r} version {version} in {root}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: a question set is a JSON object")
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"{path.name}: missing field(s) {', '.join(missing)}")
    if data["route"] not in ROUTES:
        raise ValueError(f"{path.name}: route {data['route']!r} is not one of {', '.join(ROUTES)}")
    optional = {k: data[k] for k in ("calibrated", "notes") if k in data}
    return QuestionSet(**{k: data[k] for k in REQUIRED_FIELDS}, **optional)


@dataclass(frozen=True)
class Request:
    qs: QuestionSet
    state: dict
    questions: dict
    images: dict = field(default_factory=dict)
    sensitivity: str = "public"
    tag: str = ""

    def has_box(self):
        return any(isinstance(q, dict) and q.get("type") == "box" for q in self.questions.values())

    def image_refs_in_order(self):
        """Image refs (keys of `images`) that are a whole string value in `state` then `questions`, depth-first in
        insertion order, each once. A ref inside a longer string is not a reference."""
        found = []

        def walk(obj):
            if isinstance(obj, str):
                if obj in self.images and obj not in found:
                    found.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)

        walk(self.state)
        walk(self.questions)
        return found


@dataclass
class Answer:
    request: Request
    answers: dict | None
    backend: str
    model: str
    model_resolved: str | None
    input_tokens: int
    latency_s: float
    cached: bool
    request_sha256: str
    error: str | None = None
    detail: str = ""
    route_reasons: list = field(default_factory=list)

    @property
    def ok(self):
        return self.error is None


def image_ref(path, tag="img"):
    """A reference string for an image file, stable per file content: `<<tag-sha12>>`."""
    return f"<<{tag}-{file_sha256(path)[:12]}>>"


def _answer_of(ans, qid):
    return (ans.answers or {}).get(qid)


def noul_p(ans, qid):
    a = _answer_of(ans, qid)
    return None if a is None else a.get("noul")


def choice_probs(ans, qid):
    a = _answer_of(ans, qid)
    return {} if a is None else {k: float(v) for k, v in (a.get("probabilities") or {}).items()}


def choice_top(ans, qid, exclude=("none",)):
    """The most probable option that is not in `exclude`, as `(option, p)`; `(None, 0.0)` when there is none."""
    best, best_p = None, 0.0
    for option, p in choice_probs(ans, qid).items():
        if option not in exclude and (best is None or p > best_p):
            best, best_p = option, p
    return best, best_p


def box_frac(ans, qid):
    """The Box answer as fractions of the image: `left, top, right, bottom`; None when absent."""
    a = _answer_of(ans, qid)
    return None if a is None else a.get("box")


def choice_conf(probs):
    """How far the top option stands above a uniform guess: 0.0 at uniform, 1.0 when certain (1.0 when n <= 1)."""
    n = len(probs)
    if n <= 1:
        return 1.0
    return (max(probs.values()) - 1 / n) / (1 - 1 / n)


def strictest(values):
    """The most private sensitivity among `values`; None (or an unknown value) counts as private."""
    rank = {s: i for i, s in enumerate(SENSITIVITIES)}
    top = max((rank.get(v, rank["private"]) for v in values), default=0)
    return SENSITIVITIES[top]


def thresholds_for(qs, backend):
    """The thresholds a set uses on `backend`; a local set with no local calibration uses Jev's."""
    t = qs.thresholds.get(backend)
    if t is None:
        t = qs.thresholds.get("jev") or {}
    return dict(t)
