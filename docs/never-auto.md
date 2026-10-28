# The never-auto list

Some decisions are never applied automatically, whatever a model says and whatever the policy file sets. The list is code (`NEVER_AUTO` in `core/policy.py`), not policy data: a policy file can move thresholds, but it cannot route one of these decisions to the `auto` lane. A row asked for `auto` leaves in the `agent` lane with `never_auto` among its reasons.

| Id | Decision | Why it is never automatic |
| --- | --- | --- |
| `STY-06` | Is a spoken number a measured or sourced figure, or an opinion? | A claim. Code then decides what a figure needs: a source, or a caption only. |
| `QA-05` | Does an on-screen claim match its source? | A claim. The exact-quote half is code; the rest goes to a person. |
| `AST-01` | Where a window's focus crop sits on a screenshot or recording. | Someone looks at the overlay before the crop is used. |
| `QA-01` | Which frames of a render look wrong. | Triage only: a flag is a pointer for someone to look, never a verdict. |

The caption decisions (`CAP-02`, `CAP-03`) are not on the list: they are reversible and cheap to redo, so they may land in `auto` when the model is clear and both option orders agree.

In the shipped policy (`policies/default.json`), `QA-05` and the `needs_source` outcome of `STY-06` go to the `human` lane. Opinions and figures of speech found by `STY-06`, and the unsure ones between the two bands, stay in the `agent` lane, which keeps them as captions and never presents them as stats. `AST-01` and `QA-01` rows always go to the `agent` lane.
