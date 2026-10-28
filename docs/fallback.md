# Backends and fallback

jevm asks bounded questions of one of two decision models over HTTP:

- **Jev** (TypeSafe, cloud): `https://api.typesafe.ai/v1/systemone`. Needs a key in `JEV_API`. Models are pinned per question set (`jev-1.13.0` for text, `jev-1.14.0` for images and Box); an alias is never sent.
- **Decider** (local): a decision-model server on your machine that takes Jev's request shape. Default `http://127.0.0.1:8080`, or `JEVM_DECIDER_URL`, or `spec.decide.backends.local.url`. Optional: nothing needs it. It answers one request at a time, has no Box question, and costs nothing.

Each question set names its backend and route, pins its models, and stores thresholds per backend. A set has a local calibration when `thresholds.local` is set; `calibrated.local` records the Decider app version and model the local thresholds were set on.

## Routing, in order

1. `--no-images` and the request carries an image: no backend. The row goes to the `agent` lane with `jev_unavailable`.
2. A forced backend (`jevm eval --backend`): that backend only, no fallback. Box on `local` has no backend.
3. Box always goes to Jev. Any other request goes to its set's backend; a `private` request moves to Decider when the set's route allows it (`local`, `either`, `jev-for-speed`) and the set has a local calibration.
4. Decider chosen but it is down, or its app version or model differs from the set's calibration: the request goes to Jev and the row carries `backend_fallback`. A changed Decider also prints, once per set, the eval command that re-checks it.
5. Jev chosen but `--no-jev` is set, there is no key, or Jev is down: the request goes to Decider with `backend_fallback` when the set is calibrated for the running Decider; otherwise there is no backend and the row goes to the `agent` lane with `jev_unavailable`.

A request that still fails after retries (Jev retries 408, 429, 529 and 5xx with backoff) goes to the `agent` lane with `jev_error`; a Decider failure with no Jev to fall back to gives `local_unavailable`. Once a backend is down it stays down for the rest of the run.

Either way every command finishes. `plan` prints how many rows got no model answer, and `decisions.md` lists them under Notices. Nothing is ever applied automatically from a missing answer.

## Spending

`plan`, `crop`, `qa` and `eval` print one estimate line per backend before sending anything: requests, estimated input tokens, estimated dollars, cache hits. `--dry-run` stops there. `--budget USD` (default `spec.decide.budget_usd`; `eval` defaults to $0.10) is a hard stop: a run whose estimate is over it sends nothing and exits 3.

## The cache

Raw answers are cached on a key that keeps request order (backend, model, Decider version, header flags, state, questions, image hashes), so a re-run costs nothing and a request with its options reversed is a different entry. The cache folder is `JEVM_CACHE_DIR` when set; otherwise `<project>/work/jevm/.jevm-cache/` for `plan`, `crop` and `qa`, and `.jevm-cache/` in the current directory for `eval`. A record never holds headers or credentials.

## The key

`JEV_API` is read from the environment, else from a `.env` file in the current working directory (that one folder only; parent folders are not searched). It is read by code only and never printed, logged, cached or written; any error text that could carry it is redacted first. `JEVM_OFFLINE=1` makes both real backends refuse before any network call (the tests set it).
