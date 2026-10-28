"""A scripted backend for tests. Nothing here touches the network or the disk."""
import copy

from jevlib.backends.base import IMAGES_HEADER, BackendUnavailable


class FakeBackend:
    """`responder(req, model)` returns what `send` returns. `.calls` records every `(request, model)` sent.

    `healthy=False` makes `health()` and `send()` raise BackendUnavailable, as a stopped server would; `raises` is an
    exception raised by every `send`. Named "jev" it follows Jev's header rule: the images header on a request that
    carries an image and on no other.
    """

    def __init__(self, name, responder, *, version="", healthy=True, raises=None):
        self.name = name
        self.responder = responder
        self._version = version
        self.healthy = healthy
        self.raises = raises
        self.calls = []

    def send(self, req, model):
        self.calls.append((req, model))
        if not self.healthy:
            raise BackendUnavailable(f"fake backend {self.name!r} is down")
        if self.raises is not None:
            raise self.raises
        return self.responder(req, model)

    def header_flags(self, req):
        return [IMAGES_HEADER] if self.name == "jev" and req.images else []

    def version(self):
        return self._version

    def health(self):
        if not self.healthy:
            raise BackendUnavailable(f"fake backend {self.name!r} is down")
        return {"status": "ready", "images": True}


def noul(p):
    return {"noul": p}


def choice(probs):
    """A Choice answer: the most probable option (the first one on a tie), with its probability as confidence."""
    top = max(probs, key=probs.get)
    return {"choice": top, "confidence": probs[top], "probabilities": dict(probs)}


def box(left, top, right, bottom, confidence=0.6):
    return {"box": {"left": left, "top": top, "right": right, "bottom": bottom}, "confidence": confidence}


def by_qid(table, tokens=100):
    """A responder that answers exactly the qids the request asks (KeyError for a qid missing from `table`)."""
    def responder(req, model):
        return {"answers": {qid: copy.deepcopy(table[qid]) for qid in req.questions},
                "model": model, "input_tokens": tokens, "latency_s": 0.0}
    return responder
