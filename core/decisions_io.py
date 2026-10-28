"""The `decisions.json` / `decisions.md` writer and `jevm resolve` (docs/decisions.md).

`decisions.json` is the contract scripts read; `decisions.md` is what the agent reads. `merge` keeps a resolution through a
re-plan as long as the proposed `value` did not change. `resolve` works on any JSON with a `"decisions"` list, so it serves
`qa.json` too. Every write is atomic (temp file, then `os.replace`). Nothing here touches the network.

Filesystem errors: a library function that cannot read or write a file raises `ValueError` with a one-line message (the
path and the OS reason, never any file content). `resolve` and `cli_resolve` catch it, print it to stderr and return 2.
`merge` and the row formatting do no I/O.
"""
import contextlib
import copy
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from .policy import decision_policy, load_policy

CONTRACT_VERSION = "1"

ACTORS = ("agent", "human", "auto")
_UNANSWERED = ("jev_unavailable", "jev_error", "local_unavailable")
_IMAGE_DIRS = ("crops",)
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_MAX_IDS_SHOWN = 10


# ---------------------------------------------------------------------------------------------------------------- the doc

def _stamp(now=None) -> str:
    """`now` as `YYYY-MM-DDTHH:MM:SSZ` (UTC). A string passes through; a naive datetime counts as UTC."""
    if isinstance(now, str):
        return now
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is not None:
        now = now.astimezone(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_backends() -> dict:
    """The `backends_used` block with every counter at zero (the shape `Ledger.to_dict()` returns)."""
    return {"jev": {"models": {"text": "", "image": ""}, "requests": 0, "input_tokens": 0, "usd": 0.0, "cache_hits": 0},
            "local": {"model": "", "app_version": "", "requests": 0, "serial_s": 0.0, "cache_hits": 0}}


def new_doc(*, project_id, spec_sha256, kit_id, manifest_sha256, now=None) -> dict:
    return {"contract_version": CONTRACT_VERSION, "project_id": project_id, "spec_sha256": spec_sha256,
            "kit": {"id": kit_id, "manifest_sha256": manifest_sha256}, "created_at": _stamp(now),
            "backends_used": _empty_backends(), "decisions": []}


def _why(exc: Exception) -> str:
    """A short reason for a failed file operation: the OS's own words for an OSError, never any file content."""
    return (exc.strerror or type(exc).__name__) if isinstance(exc, OSError) else "invalid content"


def load(path) -> dict | None:
    """The document at `path`, or None when there is no such file. Raises ValueError for a file that cannot be read (parent
    is a file, permission denied, a folder), is not UTF-8 or JSON, or has no `decisions` list: a caller must not overwrite a
    file it could not read, because that would drop its resolutions."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        raise ValueError(f"{path}: not valid UTF-8 text") from None
    except (OSError, ValueError) as exc:                         # ValueError: an embedded null byte in the path
        raise ValueError(f"{path}: cannot read ({_why(exc)})") from exc
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("decisions"), list):
        raise ValueError(f"{path}: not a decisions file (no \"decisions\" list)")
    return doc


def _plain(row: dict) -> dict:
    """A copy of `row` through JSON, so tuples become lists and what is stored is what a reload gives back."""
    return json.loads(json.dumps(row))


def merge(doc: dict, rows: list[dict], *, modules: set[str] = frozenset(), ids: set[str] = frozenset()) -> dict:
    """Replace, in `doc` (in place, also returned), the rows whose `module` is in `modules` or whose `id` is in `ids` with
    `rows`. A fresh row also replaces an existing row with the same id, so ids stay unique. A replaced row whose `value` is
    unchanged keeps its `status` and `resolved`; a changed value resets them to `proposed` / `None`. A fresh row takes the
    place of the row it replaces; the rest are appended in the order given. `rows` are copied, never altered."""
    fresh: dict[str, dict] = {}
    for row in rows:
        row = _plain(row)
        if row["id"] in fresh:
            raise ValueError(f"two rows with the id {row['id']!r}")
        fresh[row["id"]] = row
    merged, placed = [], set()
    for old in doc.setdefault("decisions", []):
        rid = old.get("id")
        if not (old.get("module") in modules or rid in ids or rid in fresh):
            merged.append(old)
            continue
        new = fresh.get(rid)
        if new is None or rid in placed:
            continue                                             # no longer produced: dropped
        if old.get("value") == new["value"]:
            new["status"], new["resolved"] = old.get("status", "proposed"), old.get("resolved")
        merged.append(new)
        placed.add(rid)
    merged.extend(row for rid, row in fresh.items() if rid not in placed)
    doc["decisions"] = merged
    return doc


def _union(a: str, b: str) -> str:
    """Two comma-joined model lists as one sorted, comma-joined list without repeats."""
    names = {n.strip() for n in f"{a},{b}".split(",") if n.strip()}
    return ", ".join(sorted(names))


def set_backends(doc: dict, ledger_dict: dict) -> dict:
    """Add one run's `Ledger.to_dict()` counters into `doc["backends_used"]` (in place, also returned). Models are unioned;
    the Decider app version is the latest one seen."""
    used = _empty_backends()
    old = doc.get("backends_used") or {}
    used["jev"].update({k: v for k, v in (old.get("jev") or {}).items() if k != "models"})
    used["jev"]["models"].update((old.get("jev") or {}).get("models") or {})
    used["local"].update(old.get("local") or {})

    add_j = ledger_dict.get("jev") or {}
    jev = used["jev"]
    for key in ("requests", "input_tokens", "cache_hits"):
        jev[key] += add_j.get(key, 0)
    jev["usd"] = round(jev["usd"] + add_j.get("usd", 0), 6)
    for kind in ("text", "image"):
        jev["models"][kind] = _union(jev["models"].get(kind, ""), (add_j.get("models") or {}).get(kind, ""))

    add_l = ledger_dict.get("local") or {}
    local = used["local"]
    for key in ("requests", "cache_hits"):
        local[key] += add_l.get(key, 0)
    local["serial_s"] = round(local["serial_s"] + add_l.get("serial_s", 0), 3)
    local["model"] = _union(local["model"], add_l.get("model", ""))
    local["app_version"] = add_l.get("app_version") or local["app_version"]
    doc["backends_used"] = used
    return doc


# ----------------------------------------------------------------------------------------------------------- decisions.md

def _js(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))


def _cell(text: str) -> str:
    """Text safe inside a markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def _when(target: dict) -> str:
    """`12.44-13.40 s, w112-w118` from a row target; `-` when it carries neither."""
    parts = []
    t = target.get("t")
    if t:
        parts.append(f"{t[0]:.2f}-{t[-1]:.2f} s" if t[0] != t[-1] else f"{t[0]:.2f} s")
    words = target.get("words")
    if words:
        parts.append(f"{words[0]}-{words[-1]}" if words[0] != words[-1] else str(words[0]))
    return ", ".join(parts) or "-"


def _p(row: dict) -> str:
    return "p n/a" if row.get("p") is None else f"p {row['p']:.2f}"


def _ids_text(rows: list[dict]) -> str:
    ids = [r["id"] for r in rows[:_MAX_IDS_SHOWN]]
    more = len(rows) - len(ids)
    return ", ".join(ids) + (f" and {more} more" if more > 0 else "")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _is_open(row: dict) -> bool:
    return row.get("status", "proposed") == "proposed"


def _not_helping(rows: list[dict], policy: dict) -> tuple[int, int, bool]:
    """(rows in agent or human, rows counted, flagged) for one module. Only rows of `can_auto` decisions count: a decision
    that is agent's or the human's by design does not show the module failing."""
    counted = [r for r in rows if decision_policy(policy, r["decision"])["can_auto"]]
    bad = sum(1 for r in counted if r["lane"] in ("agent", "human"))
    share = policy.get("module_not_helping", 0.40)
    return bad, len(counted), bool(counted) and bad / len(counted) > share + 1e-9


def _todo(row: dict, flagged: bool) -> str:
    reasons = row.get("route_reasons", [])
    if row["lane"] == "auto":
        return "decide directly; the module is not helping on this video"
    if any(r in reasons for r in _UNANSWERED):
        return "decide directly; no model answered"
    if "never_auto" in reasons:
        return "decide directly; never applied automatically"
    if "order_flip" in reasons:
        return "check both readings; the answer changed when the options were reversed"
    if "in_review_band" in reasons:
        return "check; p is close to the threshold"
    if "budget_edge" in reasons:
        return "check; near the budget edge"
    if "presence_low" in reasons:
        return "check; presence is low, so likely drop"
    return "decide directly; the module is not helping on this video" if flagged else "accept or override"


def _answered_by(rows: list[dict]) -> str:
    counts = Counter()
    for r in rows:
        ev = r.get("evidence") or {}
        backend = ev.get("backend") or "none"
        counts[("code (no model)" if backend == "none" else f"{backend} {ev.get('model') or ''}".strip())] += 1
    return ", ".join(f"{name} x{n}" for name, n in sorted(counts.items()))


def _backend_lines(doc: dict) -> list[str]:
    used = doc.get("backends_used") or {}
    jev, local = used.get("jev") or {}, used.get("local") or {}
    models = jev.get("models") or {}
    jev_models = ", ".join(f"{kind} {models[kind]}" for kind in ("text", "image") if models.get(kind)) or "no model used"
    local_model = local.get("model") or "no model used"
    if local.get("app_version"):
        local_model += f" (Decider {local['app_version']})"
    return [
        f"- jev: {jev_models}; {_plural(jev.get('requests', 0), 'request')}; {jev.get('input_tokens', 0):,} input tokens; "
        f"${jev.get('usd', 0):.4f}; {_plural(jev.get('cache_hits', 0), 'cache hit')}",
        f"- local: {local_model}; {_plural(local.get('requests', 0), 'request')}; "
        f"{local.get('serial_s', 0):.1f} s serial; {_plural(local.get('cache_hits', 0), 'cache hit')}",
    ]


def _notices(rows: list[dict]) -> list[str]:
    out = []
    fell_back = [r for r in rows if "backend_fallback" in r.get("route_reasons", [])]
    if fell_back:
        out.append(f"- backend_fallback: {_plural(len(fell_back), 'row')} answered by Jev because Decider rejected or "
                   f"failed the request: {_ids_text(fell_back)}")
    for reason in _UNANSWERED:
        hit = [r for r in rows if reason in r.get("route_reasons", [])]
        if hit:
            out.append(f"- {reason}: {_plural(len(hit), 'row')} got no model answer, so the agent decides "
                       f"{'it' if len(hit) == 1 else 'them'}: {_ids_text(hit)}")
    return out


def _agent_table(rows: list[dict], flagged: bool) -> list[str]:
    lines = ["| id | time and words | proposal | top-3 with p | why routed | what to do |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        alts = "; ".join(f"{a.get('p', 0):.2f} `{_js(a.get('value'))}`" for a in (r.get("alternatives") or [])[:3]) or "none"
        reasons = list(r.get("route_reasons", []))
        if flagged and "module_not_helping" not in reasons:
            reasons.append("module_not_helping")
        lines.append("| " + " | ".join(_cell(c) for c in (
            r["id"], _when(r.get("target") or {}), f"`{_js(r.get('value'))}` ({_p(r)})", alts, ", ".join(reasons),
            _todo(r, flagged))) + " |")
    return lines


def _module_section(module: str, rows: list[dict], policy: dict) -> list[str]:
    bad, counted, flagged = _not_helping(rows, policy)
    lines = [f"## {module}", ""]
    if flagged:
        lines += [f"> **module_not_helping:** {bad} of {counted} rows from decisions it can settle on its own went to "
                  f"agent or human (more than {policy.get('module_not_helping', 0.40):.0%}). Decide this module's rows "
                  f"directly; they are listed below.", ""]
    lines += [f"Answered by: {_answered_by(rows)}.", ""]

    listed = [r for r in rows if _is_open(r) and (r["lane"] == "agent" or (flagged and r["lane"] == "auto"))]
    if listed:
        lines += [f"### Rows for the agent ({len(listed)})", ""] + _agent_table(listed, flagged) + [""]
    elif any(r["lane"] == "agent" for r in rows):
        lines += ["No open rows for the agent.", ""]

    asked = [r for r in rows if _is_open(r) and r["lane"] == "human"]
    if asked:
        lines += [f"{_plural(len(asked), 'row')} for the human reviewer: see the questions block below.", ""]

    auto_by_decision: dict[str, list[dict]] = {}
    for r in rows:
        if r["lane"] == "auto":
            auto_by_decision.setdefault(r["decision"], []).append(r)
    if auto_by_decision:
        lines += ["### Auto rows", ""]
        for decision, autos in auto_by_decision.items():
            edge = [r for r in rows if r["decision"] == decision and r["lane"] != "auto"
                    and "budget_edge" in r.get("route_reasons", [])]
            text = f"- {decision}: {_plural(len(autos), 'auto row')}"
            status = Counter(r.get("status", "proposed") for r in autos)
            status.pop("proposed", None)
            if status:
                text += " (" + ", ".join(f"{n} {s}" for s, n in sorted(status.items())) + ")"
            if edge:
                text += f"; {len(edge)} near the budget edge listed above"
            lines.append(text)
        lines.append("")

    done = [r for r in rows if not _is_open(r)]
    if done:
        lines += [f"Resolved ({len(done)}): " + "; ".join(
            f"{r['id']} {r['status']}" + (f" by {r['resolved']['by']}" if r.get("resolved") else "") for r in done) + ".", ""]
    return lines


def _question(n: int, row: dict) -> str:
    value = row.get("value")
    shown = _js(value["text"]) if isinstance(value, dict) and isinstance(value.get("text"), str) else _js(value)
    why = ", ".join(row.get("route_reasons", []))
    return (f"{n}. {row['id']} at {_when(row.get('target') or {})}: is {shown} right? "
            f"Reply yes, or give me the value you want." + (f" (asked because: {why})" if why else ""))


def _image_links(media_dir: Path) -> list[str]:
    links = []
    for sub in _IMAGE_DIRS:
        folder = Path(media_dir) / sub
        try:
            if not folder.is_dir():
                continue
            for f in sorted(folder.iterdir()):
                if f.is_file() and f.suffix.lower() in _IMAGE_SUFFIXES:
                    links.append(f"- [{sub}/{f.name}]({sub}/{quote(f.name)})")
        except OSError as exc:
            raise ValueError(f"{folder}: cannot list images ({_why(exc)})") from exc
    return links


def render_md(doc: dict, policy: dict, *, media_dir: Path | None = None) -> str:
    """The `decisions.md` text for `doc`: a header line per backend and the rows per lane, one section per
    module (agent rows in a table, auto rows as one line per decision, `module_not_helping` first when it applies), one
    block of questions for the human reviewer, links to the images under `media_dir/crops/`, and any backend
    fallback notice. A resolved row leaves the open tables and is named under "Resolved"."""
    rows = doc.get("decisions") or []
    kit = doc.get("kit") or {}
    lines = [f"# Decisions: {doc.get('project_id', '')}", "",
             f"Contract {doc.get('contract_version', CONTRACT_VERSION)}; created {doc.get('created_at', '')}; "
             f"spec {str(doc.get('spec_sha256', ''))[:12] or '-'}; kit {kit.get('id', '-')} "
             f"(manifest {str(kit.get('manifest_sha256', ''))[:12] or '-'}).", "",
             "## Backends", ""] + _backend_lines(doc) + [""]
    if not rows:
        return "\n".join(lines + ["No decisions yet.", ""])

    per_lane = Counter(r["lane"] for r in rows)
    open_rows = [r for r in rows if _is_open(r)]
    lines += [f"Rows by lane: auto {per_lane['auto']}, agent {per_lane['agent']}, human {per_lane['human']}. "
              f"Open: {sum(1 for r in open_rows if r['lane'] == 'agent')} for the agent, "
              f"{sum(1 for r in open_rows if r['lane'] == 'human')} for the human reviewer.", ""]
    if any(r["lane"] == "agent" for r in open_rows):
        lines += ["Resolve with `jevm resolve decisions.json ID[,ID...] (--accept | --value V) --by agent [--note \"...\"]`.", ""]
    notices = _notices(rows)
    if notices:
        lines += ["## Notices", ""] + notices + [""]

    by_module: dict[str, list[dict]] = {}
    for r in rows:
        by_module.setdefault(r["module"], []).append(r)
    for module, module_rows in by_module.items():
        lines += _module_section(module, module_rows, policy)

    asked = [r for r in open_rows if r["lane"] == "human"]
    if asked:
        lines += ["## Questions for the human reviewer", "",
                  "Send these in one message. Do not act on them before the reviewer answers.", "",
                  "```text"] + [_question(i, r) for i, r in enumerate(asked, 1)] + ["```", ""]
    links = _image_links(media_dir) if media_dir is not None else []
    if links:
        lines += ["## Images", ""] + links + [""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------------- files, resolve

def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` through a temp file; the parent folder is made if missing. Raises ValueError (path and OS
    reason) when it cannot, for instance when the parent is a file or the folder is not writable."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
    except OSError as exc:
        raise ValueError(f"{path}: cannot write ({_why(exc)})") from exc


def _write_files(json_path: Path, doc: dict, md: str | None) -> None:
    """The JSON file at `json_path`, then (when `md` is not None) `decisions.md` beside it. Each file is atomic; if the
    second cannot be written the message says the first was, so a caller never believes nothing changed."""
    _atomic_write(json_path, _json_text(doc))
    if md is not None:
        try:
            _atomic_write(json_path.with_name("decisions.md"), md)
        except ValueError as exc:
            raise ValueError(f"{json_path.name} was updated but decisions.md was not: {exc}") from exc


def _json_text(doc: dict) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def write(doc: dict, policy: dict, out_dir) -> tuple[Path, Path]:
    """Write `decisions.json` and `decisions.md` into `out_dir`, which is created (with its parents) when missing. Returns the
    two paths. Raises ValueError, writing nothing, when a file cannot be written: `out_dir` or a parent is a file, the
    folder is not writable, or `crops/` cannot be listed."""
    out_dir = Path(out_dir)
    json_path, md_path = out_dir / "decisions.json", out_dir / "decisions.md"
    md = render_md(doc, policy, media_dir=out_dir)
    _write_files(json_path, doc, md)
    return json_path, md_path


def _no_constant(name):
    raise ValueError(name)


def _parse_value(raw):
    """`--value V`: JSON when it parses (NaN and Infinity do not count), else the text as given."""
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw, parse_constant=_no_constant)
    except ValueError:
        return raw


def _default_policy() -> dict:
    try:
        return load_policy()
    except (OSError, ValueError) as exc:
        raise ValueError(f"policy file: cannot be read ({_why(exc)})") from exc


def _fail(message: str) -> int:
    print(f"jevm resolve: {message}", file=sys.stderr)
    return 2


def resolve(path, ids, *, accept: bool, value, by: str, note: str, now=None, policy: dict | None = None) -> int:
    """Resolve the rows `ids` in the file at `path` (any JSON with a `decisions` list). `accept` marks each row `accepted`
    with its own proposed value; `value` (a JSON string, else a plain string) marks it `overridden` with that value. The
    row's own `value` is kept as the proposal. Returns 0, or 2 with a one-line message on stderr and the file untouched: an
    unknown id, a missing, unreadable or unwritable file (any OSError), or bad arguments. A failed write of `decisions.md`
    comes after the JSON is written, and its message says so. Beside a `decisions.json` it re-renders `decisions.md`."""
    path = Path(path)
    ids = list(dict.fromkeys(ids))
    if not ids:
        return _fail("no row ids given")
    if accept == (value is not None):
        return _fail("give exactly one of --accept and --value")
    if by not in ACTORS:
        return _fail(f"--by must be one of {', '.join(ACTORS)}, not {by!r}")
    try:
        doc = load(path)
        if doc is None:
            return _fail(f"no such file: {path}")
        known = {r.get("id") for r in doc["decisions"]}
        unknown = [i for i in ids if i not in known]
        if unknown:
            return _fail(f"unknown id{'s' if len(unknown) > 1 else ''}: {', '.join(unknown)}; nothing changed")

        at = _stamp(now)
        override = None if accept else _parse_value(value)
        for row in doc["decisions"]:
            if row.get("id") in ids:
                row["status"] = "accepted" if accept else "overridden"
                row["resolved"] = {"value": copy.deepcopy(row.get("value")) if accept else copy.deepcopy(override),
                                   "by": by, "at": at, "changed": not accept, "reviewed": True, "note": note or ""}
        md = (render_md(doc, policy or _default_policy(), media_dir=path.parent)
              if path.name == "decisions.json" else None)
        _write_files(path, doc, md)
    except ValueError as exc:
        return _fail(str(exc))
    except OSError as exc:                                       # nothing above should leak one; this is the net
        return _fail(f"{path}: {_why(exc)}")
    return 0


def cli_resolve(path, ids_csv: str, accept: bool, value, by: str, note: str) -> int:
    """`jevm resolve PATH ID[,ID...] (--accept | --value V) --by WHO [--note TEXT]`. Returns the exit code: 0, or 2."""
    ids = [i.strip() for i in (ids_csv or "").split(",") if i.strip()]
    rc = resolve(path, ids, accept=accept, value=value, by=by, note=note)
    if rc == 0:
        print(f"resolved {_plural(len(set(ids)), 'row')} in {path}")
    return rc
