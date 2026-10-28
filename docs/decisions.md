# The decisions contract

`jevm plan` and `jevm crop` write two files into `<project>/work/jevm/`:

- `decisions.json`: the contract. Scripts and builds read it.
- `decisions.md`: the same rows rendered for the agent (or person) that resolves them.

`jevm qa` writes `work/jevm/qa/<render-stem>/qa.json` with the same row shape under `"decisions"`, so `jevm resolve` works on both files.

## The document

```json
{
  "contract_version": "1",
  "project_id": "my-short",
  "spec_sha256": "<sha256 of spec.json>",
  "kit": {"id": "example", "manifest_sha256": "<sha256 of the kit's manifest.json>"},
  "created_at": "2026-01-01T12:00:00Z",
  "backends_used": {
    "jev":   {"models": {"text": "jev-1.13.0", "image": "jev-1.14.0"}, "requests": 12, "input_tokens": 9000, "usd": 0.0004, "cache_hits": 3},
    "local": {"model": "", "app_version": "", "requests": 0, "serial_s": 0.0, "cache_hits": 0}
  },
  "decisions": []
}
```

`backends_used` adds up across runs: each `plan` or `crop` run adds its own counts. Model names are unioned.

## A row

```json
{
  "id": "CAP-03:cg4",
  "decision": "CAP-03",
  "module": "universal.captions",
  "target": {"kind": "caption_group", "id": "cg4", "t": [3.6, 5.1], "words": ["w12", "w16"]},
  "value": {"span": ["w14", "w15"], "text": "3 hours"},
  "alternatives": [{"value": {"span": ["w13", "w15"]}, "p": 0.18}],
  "p": 0.71,
  "confidence": {"metric": "choice_conf", "value": 0.64},
  "lane": "auto",
  "route_reasons": ["above_auto", "order_agree"],
  "status": "proposed",
  "resolved": null,
  "evidence": {"label": "Default", "ref": "docs/question-sets.md", "qset": "highlight", "qset_version": 1,
               "backend": "jev", "model": "jev-1.13.0", "request_sha256": "<cache key>", "order_check": "agree", "cached": false}
}
```

| Field | Meaning |
| --- | --- |
| `id` | `<decision>:<target id>`. Stable across re-runs of the same spec. |
| `decision` | The decision id: `CAP-02`, `CAP-03`, `STY-06`, `QA-05`, `AST-01` or `QA-01`. |
| `module` | The module that produced the row, for example `universal.captions`. |
| `target` | What the row is about: `kind`, `id`, and when known the time span `t` (seconds) and word span `words`. |
| `value` | The proposal. Its shape depends on the decision. |
| `alternatives` | Up to a few other values with their `p`, best first. |
| `p` | The model probability the routing used, or `null` when code decided or no model answered. |
| `confidence` | `{"metric", "value"}`: how sure the proposal is, in the decision's own metric. |
| `lane` | Who acts on the row: `auto`, `agent` or `human` (below). |
| `route_reasons` | Why the row is in its lane (below). |
| `status` | `proposed`, `accepted` or `overridden`. |
| `resolved` | `null`, or `{"value", "by", "at", "changed", "reviewed", "note"}` once someone resolved it. |
| `evidence` | Where the proposal came from: the policy's evidence label and ref, the question set and version, the backend and model, the cache key, the option-order check (`agree`, `flip` or `null`) and whether the answer was cached. Code-decided rows carry `backend: "none"`. |

## Lanes

| Lane | Meaning | Who acts |
| --- | --- | --- |
| `auto` | Reversible and cheap to redo, above the auto line, and both option orders agree. | The build uses it; the agent may override. |
| `agent` | Near a threshold (within the review band), an order flip, a decision whose policy says the agent reviews it, or any row no model answered. | The build agent accepts, picks an alternative, or overrides. |
| `human` | Claims, and every decision on the never-auto list ([never-auto.md](never-auto.md)) that the policy routes to a person. | A human reviewer. The agent sends all of them in one message and does not act on them before the reviewer answers. |

A never-auto decision can never be in `auto`: `core/rows.py` moves such a row to `agent` and adds `never_auto`, whatever the policy file says.

## Route reasons

`above_auto`, `in_review_band`, `order_flip`, `order_agree`, `never_auto`, `budget_edge`, `jev_unavailable`, `jev_error`, `local_unavailable`, `backend_fallback`, `quarantined`, `module_not_helping`, `presence_low`.

The three "no answer" reasons (`jev_unavailable`, `jev_error`, `local_unavailable`) always put the row in `agent`. `backend_fallback` marks a row that was answered, but by the other backend ([fallback.md](fallback.md)).

## Re-running and resolving

- Running `plan` again replaces a module's rows. A row whose proposed `value` did not change keeps its `status` and `resolved`; a changed value resets the row to `proposed`.
- `crop` replaces only its own `AST-01:<beat>` row.
- `jevm resolve decisions.json ID[,ID...] (--accept | --value V) --by agent|human [--note TEXT]` marks rows `accepted` (the proposal stands) or `overridden` (with `V`, parsed as JSON when it parses, else kept as text). The row's own `value` stays as the proposal beside the override. An unknown id exits 2 and changes nothing.
- Every write is atomic (temp file, then rename). A file jevm cannot read is never overwritten, so resolutions are never lost.

## decisions.md

One header with the contract version, spec and kit hashes, then:

1. One line per backend: models, requests, input tokens, dollars, cache hits.
2. Rows by lane, and how many are still open for the agent and for the human reviewer.
3. Notices: rows answered by the fallback backend, rows no model answered.
4. One section per module: the open `agent` rows in a table (time and words, proposal, top alternatives with p, why routed, what to do), the `auto` rows as one line per decision, and resolved rows by id. When more than 40% of a module's rows from decisions that could settle on their own land in `agent` or `human`, the section starts with `module_not_helping` and lists its auto rows too.
5. "Questions for the human reviewer": every open `human` row as one numbered question, in a single block to send in one message.
6. Links to the focus-crop overlays under `crops/`.
