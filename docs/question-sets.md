# Question sets

Every model question jevm asks lives in `qsets/<id>@<version>.json`: the wording, the pinned models, the thresholds per backend, the route and the evidence label. Modules build requests from these files; nothing in code repeats the wording. Editing any question bumps the version, because answers move when other questions in the same request change. The highest version on disk is used.

| Set | Decision | Asks | Backend |
| --- | --- | --- | --- |
| `highlight@1` | CAP-02, CAP-03 | Per caption group: `need`, `need2` (is there one phrase that is the point of the line?) and `pick` (a Choice over every 1-3 word span, then `none`). Asked forward and with the options reversed. | Jev |
| `stat_check@1` | STY-06 | Per sentence with a number hint: `has_number`, `stat_is_measured`. | Decider first, Jev fallback |
| `presence@1` | AST-01 | Is the wanted region visible in the image? | Decider first, Jev fallback |
| `crop_box@1` | AST-01 | A Box around the wanted region, plus the same presence question bundled. | Jev only |
| `qa_frame.short@1` | QA-01 | Seven defect questions (`text_occluded`, `cut_off_edge`, `caption_on_face`, `empty_region`, `overlapping_panels`, `overlay_on_controls`, `hard_to_read`) and two gates (`face_visible`, `caption_visible`), all in one request per frame, with the kit's `qa_context` as `context`. | Jev |

## Evidence labels

Every question set and every policy entry carries an evidence label, which jevm copies into each row's `evidence` block.

- **Default**: the wording and thresholds this repository ships with. They are a starting point, not a measurement on your videos: check them with `jevm eval` on eval sets frozen from your own work before you rely on the `auto` lane.

Use another label in your own question sets or policy (for example the name of the eval run that set the thresholds) when you have one; jevm passes it through unchanged.

## Checking a set on your own videos

`jevm eval` scores the sets against frozen eval sets in `evalsets/` and gates them. The shipped eval sets are synthetic (flat-colour mock-ups, invented lines, written by `evalsets/make_synthetic.py`), so a live run on them is a smoke test of your key and backends, not a measurement. To measure, freeze your own frames, caption groups, crops and sentences in the same layout and put your bars in each set's `gates` and `local_reference` blocks:

| Set file | Gates | `local_reference` (a local run passes within one item) |
| --- | --- | --- |
| `evalsets/qa_frame.short/manifest.json` | `own_flagged_min`, `clean_flagged_max` | `clean_flagged` |
| `evalsets/highlight/<name>.json` | `precision_at_k_min` and/or `hand_rank_max` | (span overlap against the hand groups) |
| `evalsets/crops/manifest.json` | `mean_iou_min` (and the centre inside the hand rectangle on every case) | `presence_auc` |
| `evalsets/stat_check/sentences.json` | every `labelled` sentence on its side of the bands, every control right at 0.5 | `controls_correct` |

`jevm eval --offline` replays the scripted answers in `tests/fixtures/eval/` through fake backends: no network, no key, nothing spent. It checks the harness, not the models.
