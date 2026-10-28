"""The decision client (docs/fallback.md): picks a backend per request, falls back, caches, keeps a ledger.

A module hands the client a `Request`; the client sends it to Jev or to Decider and returns an `Answer`. Routing, in
order (`Client.route`):

1. `no_images` and the request carries an image: no backend (`jev_unavailable`, detail `no_images`).
2. `force_backend` set (parity runs): that backend, no fallback; Box on `local` has no backend (`box_on_local`).
3. Box goes to `jev`. Otherwise the set's `backend`, switched to `local` for a `private` request when the set's route
   is `local`, `either` or `jev-for-speed` and it has local thresholds.
4. `local` chosen but Decider is unhealthy, or its app version or model differs from the set's calibration: `jev`
   with `backend_fallback` (a version change also prints the re-calibration notice, once per set).
5. `jev` chosen but `no_jev` is set or Jev is down: `local` with `backend_fallback` when the set is calibrated for
   the running Decider; otherwise no backend (`jev_unavailable`). Jev is down when its `health()` raises (no key,
   offline) or a send raises `BackendUnavailable`; either lasts for the rest of the run.

`Client.ask` never raises for a backend failure: it returns an `Answer` whose `error` is `jev_unavailable`,
`jev_error` or `local_unavailable`, and the module routes that row to the agent. Error text goes through `redact`.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from jevlib import images, tokens
from jevlib.backends.base import BackendError, BackendUnavailable
from jevlib.cache import Cache, cache_key, file_sha256
from jevlib.env import load_key, redact
from jevlib.request import Answer

LOCAL_ROUTES = ("local", "either", "jev-for-speed")
DETAIL_MAX = 300


class Ledger:
    """Per-backend counters for one run: the `backends_used` block of decisions.json.

    A cache hit counts only as a hit; just live requests add tokens, dollars and serial seconds. Failed requests are
    not counted. Unused fields are `""` or 0 so the block always has the same keys.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._jev = {"text": set(), "image": set(), "requests": 0, "input_tokens": 0, "cache_hits": 0}
        self._local = {"models": set(), "app_version": "", "requests": 0, "serial_s": 0.0, "cache_hits": 0}

    def add(self, ans, app_version=""):
        """Count one answered request. `app_version` is the Decider version, recorded for a `local` answer."""
        if not ans.ok or ans.backend not in ("jev", "local"):
            return
        with self._lock:
            if ans.backend == "jev":
                c = self._jev
                c["image" if ans.request.images else "text"].add(ans.model)
                if ans.cached:
                    c["cache_hits"] += 1
                else:
                    c["requests"] += 1
                    c["input_tokens"] += ans.input_tokens
            else:
                c = self._local
                c["models"].add(ans.model)
                c["app_version"] = app_version
                if ans.cached:
                    c["cache_hits"] += 1
                else:
                    c["requests"] += 1
                    c["serial_s"] += ans.latency_s

    def to_dict(self):
        with self._lock:
            j, loc = self._jev, self._local
            return {
                "jev": {"models": {"text": ", ".join(sorted(j["text"])), "image": ", ".join(sorted(j["image"]))},
                        "requests": j["requests"], "input_tokens": j["input_tokens"],
                        "usd": round(tokens.usd(j["input_tokens"]), 6), "cache_hits": j["cache_hits"]},
                "local": {"model": ", ".join(sorted(loc["models"])), "app_version": loc["app_version"],
                          "requests": loc["requests"], "serial_s": round(loc["serial_s"], 3),
                          "cache_hits": loc["cache_hits"]},
            }


def _sent_dims(w, h):
    """About the size an image is sent to Jev at (`images.fit_for_jev`): short side up to 280, long side down to 1568."""
    k = max(images.MIN_SIDE / min(w, h), 1.0)
    if max(w, h) * k > images.MAX_SIDE:
        k = images.MAX_SIDE / max(w, h)
    return max(round(w * k), images.MIN_SIDE), max(round(h * k), images.MIN_SIDE)


class Client:
    def __init__(self, backends, cache, *, no_jev=False, no_images=False, force_backend=None,
                 jev_concurrency=6, log=print, secrets=()):
        """`backends` maps "jev" and "local" to Backend objects. `secrets` are strings (the key) that must never
        reach a log line or an `Answer.detail`."""
        if force_backend not in (None, "jev", "local"):
            raise ValueError(f"force_backend must be 'jev' or 'local', not {force_backend!r}")
        self.backends = backends
        self.cache = cache
        self.no_jev = no_jev
        self.no_images = no_images
        self.force_backend = force_backend
        self.jev_concurrency = jev_concurrency
        self.log = log
        self.ledger = Ledger()
        self._secrets = tuple(secrets)
        self._lock = threading.RLock()
        self._local_send = threading.Lock()   # Decider takes one request at a time, whoever asks
        self._local_ok = None                 # None: not checked yet; False: unhealthy for the rest of the run
        self._jev_checked = False
        self._jev_down = None                 # None, or the detail of the BackendUnavailable that took Jev down
        self._versions = {}
        self._said = set()

    # ------------------------------------------------------------ helpers
    def _say_once(self, key, message):
        with self._lock:
            if key in self._said:
                return
            self._said.add(key)
        self.log(message)

    def _detail(self, exc):
        return redact(str(exc), self._secrets)[:DETAIL_MAX]

    def _version(self, name):
        with self._lock:
            if name not in self._versions:
                self._versions[name] = self.backends[name].version()
            return self._versions[name]

    def _set_jev_down(self, exc):
        with self._lock:
            if self._jev_down is None:
                self._jev_down = self._detail(exc) or "jev_down"
        self._say_once("jev-down", f"Jev unavailable: {self._jev_down}")

    def _set_local_down(self, exc):
        with self._lock:
            self._local_ok = False
        self._say_once("local-down", f"Decider unavailable: {self._detail(exc)}; its sets run on Jev")

    def _jev_unavailable(self):
        """`no_jev`, or Jev cannot take a request: no key, offline, or refused earlier in the run. Jev's `health()`
        spends nothing; it is asked once per run."""
        if self.no_jev:
            return True
        with self._lock:
            if not self._jev_checked and self._jev_down is None:
                self._jev_checked = True
                jev = self.backends.get("jev")
                if jev is None:
                    self._jev_down = "jev backend missing"
                else:
                    try:
                        jev.health()
                    except (BackendUnavailable, BackendError) as exc:
                        self._set_jev_down(exc)
            return self._jev_down is not None

    def _local_healthy(self):
        """`GET /health` once per run; False for the rest of the run after any failure."""
        with self._lock:
            if self._local_ok is None:
                local = self.backends.get("local")
                if local is None:
                    self._local_ok = False
                else:
                    try:
                        local.health()
                        self._local_ok = True
                    except (BackendUnavailable, BackendError) as exc:
                        self._set_local_down(exc)
            return self._local_ok

    def _local_state(self, qs):
        """"ok", "down", "uncalibrated" (no calibration record) or "changed" (Decider no longer the calibrated one)."""
        if not self._local_healthy():
            return "down"
        cal = qs.calibrated.get("local")
        if not cal:
            return "uncalibrated"
        if self._version("local") != cal.get("app_version") or qs.models.get("local") != cal.get("model"):
            self._say_once(("changed", qs.id), f"Decider changed: run `jevm eval --backend local --qset {qs.id}`")
            return "changed"
        return "ok"

    def _local_can_stand_in(self, req):
        """Rule 5: Jev is out, so can this request be answered locally?"""
        return not req.has_box() and req.qs.has_local_calibration() and self._local_state(req.qs) == "ok"

    # ------------------------------------------------------------ routing
    def _route(self, req):
        if self.no_images and req.images:
            return "none", ["jev_unavailable"], "no_images"
        if self.force_backend:
            if self.force_backend == "local" and req.has_box():
                return "none", ["local_unavailable"], "box_on_local"
            if self.force_backend == "jev" and self.no_jev:
                return "none", ["jev_unavailable"], "no_jev"
            return self.force_backend, [], ""
        qs = req.qs
        chosen, reasons = "jev", []
        if not req.has_box():
            chosen = qs.backend
            if req.sensitivity == "private" and qs.route in LOCAL_ROUTES and qs.has_local_calibration():
                chosen = "local"
        if chosen == "local" and self._local_state(qs) != "ok":
            chosen, reasons = "jev", ["backend_fallback"]
        if chosen == "jev" and self._jev_unavailable():
            if reasons == [] and self._local_can_stand_in(req):
                return "local", ["backend_fallback"], ""
            return "none", ["jev_unavailable"], "no_jev" if self.no_jev else (self._jev_down or "jev backend missing")
        return chosen, reasons, ""

    def route(self, req):
        """`(backend, route_reasons)` for one request; backend is "jev", "local" or "none"."""
        name, reasons, _ = self._route(req)
        return name, reasons

    # ------------------------------------------------------------ asking
    def _key(self, req, name):
        be = self.backends[name]
        shas = [file_sha256(req.images[ref]) for ref in req.image_refs_in_order()]
        return cache_key(name, req.qs.models[name], self._version(name), be.header_flags(req),
                         req.state, req.questions, shas)

    def _fail(self, req, error, detail, reasons=()):
        return Answer(request=req, answers=None, backend="none", model="", model_resolved=None, input_tokens=0,
                      latency_s=0.0, cached=False, request_sha256="", error=error, detail=detail,
                      route_reasons=list(reasons))

    def _send(self, req, name, reasons):
        """One request on one backend, through the cache. Raises BackendUnavailable or BackendError on failure."""
        model = req.qs.models[name]
        key = self._key(req, name)
        hit = self.cache.get(key)
        if hit is not None and isinstance(hit.get("answers"), dict):
            ans = Answer(req, hit["answers"], name, model, hit.get("model"), hit.get("input_tokens", 0), 0.0, True,
                         key, route_reasons=list(reasons))
        else:
            be = self.backends[name]
            if name == "local":
                with self._local_send:
                    res = be.send(req, model)
            else:
                res = be.send(req, model)
            if not isinstance(res.get("answers"), dict):
                raise BackendError(f"{name} returned no answers")
            self.cache.put(key, {"answers": res["answers"], "model": res.get("model"),
                                 "input_tokens": res.get("input_tokens", 0), "latency_s": res.get("latency_s", 0.0)})
            ans = Answer(req, res["answers"], name, model, res.get("model"), res.get("input_tokens", 0),
                         res.get("latency_s", 0.0), False, key, route_reasons=list(reasons))
        got = ans.model_resolved
        if got and got != model:
            self._say_once(("drift", req.qs.id, got), f"model drift: {req.qs.id} pinned {model}, got {got}")
        self.ledger.add(ans, app_version=self._version("local") if name == "local" else "")
        return ans

    def ask(self, req):
        """The `Answer` for one request. Failures come back as an Answer with `error` set, never as an exception."""
        name, reasons, detail = self._route(req)
        if name == "none":
            return self._fail(req, reasons[0], detail, reasons[1:])
        forced = self.force_backend is not None
        try:
            return self._send(req, name, reasons)
        except (BackendUnavailable, BackendError) as exc:
            first = exc
        with_fallback = reasons if "backend_fallback" in reasons else reasons + ["backend_fallback"]

        if name == "local":
            if isinstance(first, BackendUnavailable):
                self._set_local_down(first)
            if forced or self._jev_unavailable():
                return self._fail(req, "local_unavailable", self._detail(first), reasons)
            try:
                return self._send(req, "jev", with_fallback)
            except BackendUnavailable as exc:
                self._set_jev_down(exc)
                return self._fail(req, "jev_unavailable", self._detail(exc), with_fallback)
            except BackendError as exc:
                return self._fail(req, "jev_error", self._detail(exc), with_fallback)

        if isinstance(first, BackendUnavailable):
            self._set_jev_down(first)
        error = "jev_unavailable" if isinstance(first, BackendUnavailable) else "jev_error"
        if forced or not self._local_can_stand_in(req):
            return self._fail(req, error, self._detail(first), reasons)
        try:
            return self._send(req, "local", with_fallback)
        except (BackendUnavailable, BackendError) as exc:
            if isinstance(exc, BackendUnavailable):
                self._set_local_down(exc)
            return self._fail(req, error, f"{self._detail(first)}; local: {self._detail(exc)}", with_fallback)

    def run_many(self, reqs):
        """Answers in the order of `reqs`. Jev requests run on a pool of `jev_concurrency`; Decider requests run one
        after another in this thread at the same time."""
        reqs = list(reqs)
        answers = [None] * len(reqs)
        jev_idx, local_idx = [], []
        for i, req in enumerate(reqs):
            (local_idx if self._route(req)[0] == "local" else jev_idx).append(i)
        with ThreadPoolExecutor(max_workers=max(1, self.jev_concurrency)) as pool:
            futures = [(i, pool.submit(self.ask, reqs[i])) for i in jev_idx]
            for i in local_idx:
                answers[i] = self.ask(reqs[i])
            for i, future in futures:
                answers[i] = future.result()
        return answers

    # ------------------------------------------------------------ cost
    def estimate(self, reqs):
        """What a run would send, from the routing and the cache, without sending: `jev` requests, cached, input
        tokens, dollars; `local` requests, cached, serial seconds; `none` requests that go to the agent. Tokens and
        seconds are counted for requests the cache does not already hold."""
        est = {"jev": {"requests": 0, "cached": 0, "input_tokens": 0, "usd": 0.0},
               "local": {"requests": 0, "cached": 0, "serial_s": 0.0},
               "none": {"requests": 0}}
        for req in reqs:
            name, _, _ = self._route(req)
            e = est[name]
            e["requests"] += 1
            if name == "none":
                continue
            if self.cache.get(self._key(req, name)) is not None:
                e["cached"] += 1
                continue
            refs = req.image_refs_in_order()
            if name == "jev":
                chars = len(json.dumps({"state": req.state, "questions": req.questions},
                                       separators=(",", ":"), ensure_ascii=False))
                dims = []
                for ref in refs:
                    try:
                        w, h, _ = images.image_size(req.images[ref])
                        dims.append(_sent_dims(w, h))
                    except (images.ImageLimitError, OSError):
                        pass
                e["input_tokens"] += tokens.request_tokens(chars, dims)
            else:
                e["serial_s"] += tokens.local_seconds(len(req.questions), len(refs))
        est["jev"]["usd"] = tokens.usd(est["jev"]["input_tokens"])
        return est

    def preflight(self, reqs, *, budget_usd, dry_run):
        """Print one line per backend before anything is spent. Returns 0 to stop on `dry_run`, 3 when the Jev
        estimate is over `budget_usd` (None: no limit), else None to go on. A dry run wins over an exceeded budget
        but still says so."""
        est = self.estimate(reqs)
        j, loc, none = est["jev"], est["local"], est["none"]
        spend = tokens.fmt_usd(j["usd"])
        self.log(f"jev: {j['requests']} requests ({j['cached']} cached), ~{j['input_tokens']} input tokens, "
                 f"{'' if spend.startswith('<') else '~'}{spend}")
        self.log(f"local: {loc['requests']} requests ({loc['cached']} cached), ~{loc['serial_s']:.1f} s serial")
        if none["requests"]:
            self.log(f"none: {none['requests']} requests have no backend; their rows go to the agent")
        over = budget_usd is not None and j["usd"] > budget_usd
        if over:
            self.log(f"over budget: {'' if spend.startswith('<') else '~'}{spend} against a {tokens.fmt_usd(budget_usd)} budget; "
                     + ("dry run, nothing sent" if dry_run else "stopping before anything is sent (pass --budget to raise it)"))
        if dry_run:
            if not over:
                self.log("dry run: nothing sent")
            return 0
        return 3 if over else None


def make_client(decide=None, *, cache_dir=None, no_jev=False, no_images=False, force_backend=None, log=print):
    """The production client: real Jev and Decider backends, the answer cache, settings from `spec.decide`.

    `cache_dir` None means `jevlib.paths.cache_dir()`: `JEVM_CACHE_DIR`, else `.jevm-cache/` in the current directory
    (the commands that have a spec pass `<project>/work/jevm/.jevm-cache/`).

    Decider's URL is `JEVM_DECIDER_URL` when set, else `decide.backends.local.url`, else the backend's default. Jev
    concurrency is `decide.backends.jev.concurrency`, default 6. The backends are imported here so tests of the
    client do not need them.
    """
    from jevlib.backends.jev import JevBackend
    from jevlib.backends.local import LocalBackend

    cfg = ((decide or {}).get("backends")) or {}
    url = os.environ.get("JEVM_DECIDER_URL") or (cfg.get("local") or {}).get("url") or None
    concurrency = (cfg.get("jev") or {}).get("concurrency") or 6
    return Client({"jev": JevBackend(), "local": LocalBackend(url=url)}, Cache(cache_dir),
                  no_jev=no_jev, no_images=no_images, force_backend=force_backend,
                  jev_concurrency=concurrency, log=log, secrets=[load_key()])
