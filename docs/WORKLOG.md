# Work log
Newest first. Each entry: what changed, how it was verified, what's next. Never edit or delete past entries.
Earlier history: `git log` (commit bodies explain each change) and `docs/archive/implementation-progress.md` (the 2026-09-07 implementation).

Earlier months: `docs/worklog/YYYY-MM.md`.

## 2026-11-01 - Prompt rules from the hova/keva team review; optional audience
Changed: merged the closed hova and keva team reviews (`review-merge`, then `reground` of team-edited rows; rows the KB cannot ground were dropped from the sets and archived in `dropped_ungrounded_team_questions.jsonl`). Compared every reviewer edit before/after and every rejection (keva rejections from the domain the owner asked to skip were excluded). Lessons now in the generation prompts: answers speak to the user and never cite "the document"; answer only what is asked (reviewers cut facts appended from a second document); state the apply channel and approver when the evidence has them; skip calendar/list/table trivia and compound questions; unanswerable answers refer the user to the responsible channel. New optional `generation.audience` keeps questions to one population (keva had hova-population questions). Intent duplicates reviewers flagged sit at cosine 0.82-0.91, inside the range of distinct variants, so the duplicate threshold is unchanged. Decisions: D13.

| Verification command | Result |
|---|---|
| `python -m pytest -q` | 342 passed |
| `python tools/docmap.py update && python tools/docmap.py check` | 0 errors; 1 warning (skill 5.4.0 available, upgrade not in scope) |

Next: generate the meitav set with these prompts and watch the review for document references.

## 2026-11-01 - Upgrade to project-foundations 5.3 (docmap 5.3.0)
Changed: replaced `tools/docmap.py` with the skill's 5.3.0 copy (adds `find`, `get`, `stats`, `docs`, the doc-lifecycle and FACTS checks, `rotate`). Gave every doc in `docs/` a Status line (the "verified" dates are the docs' last-edit date, 2026-10-23, not a fresh read; `theme-layer-ab-results.md` is `historical`). Rotated the work log (six entries moved to `docs/worklog/2026-10.md`). Added `[tool.docmap.facts]` (seven shipped defaults from `config.example.toml`, generated into `docs/FACTS.md`; existing docs still copy those values, to be replaced as they are touched) and raised `agents_max_lines` to 120. AGENTS.md gained the Scaffolding line, `find`/`get` at session start and five maintenance rows (Status lines, FACTS, superseding decisions, scaffolding feedback, milestone audit). Decisions: none.

| Verification command | Result |
|---|---|
| `python tools/docmap.py check` with the new script, before the changes | 0 errors, 24 warnings (23 missing Status lines, un-rotated work log) |
| `python tools/docmap.py update && python tools/docmap.py check` after | 0 errors, 0 warnings |
| `python tools/docmap.py stats` | session-start read about 5.3k tokens (was 6.7k before rotation), under the 8k limit |
| `python -m pytest -q` | 340 passed |

Next: run `python tools/docmap.py docs` at the next milestone and re-verify the design docs against the code; set `lifecycle_strict = true` once that is done.
