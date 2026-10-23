# Brief
Updated 2026-10-13. Drawn from README.md; items marked TBD (owner) need Sergiy's confirmation.

## Goal
Alloy (`chatbot_eval`) builds evidence-grounded Hebrew evaluation sets ("silver" question sets) from
a document corpus, judges internal knowledge chatbots against them at claim level, and helps managers
bootstrap a chatbot's Hebrew system prompt from the same documents.

## Non-goals (from README.md)
- Not ground truth: silver questions and references need domain review before use as a benchmark.
- Not a deployment or security layer: generated system prompts are starting points, never deployed
  automatically, and not security controls; ACLs, DLP/PII filtering and monitoring stay with the
  application.
- Not tied to the existing chatbot fleet: chatbots are reached only through the adapter boundary.
- No personal question types until an allowlisted, masked, auditable data interface exists.

## Users / stakeholders
- Domain reviewers: review exported question sets (`review-export` / `review-merge`).
- Managers of internal chatbots: evaluation reports and generated system prompts.
- Approvals: TBD (owner).

## Success criteria
- Generated questions pass deterministic validation (quote provenance, schema) before acceptance.
- Offline test suite green; CI green on `main`.
- Live: judge calibrated against a double-reviewed human sample before results gate a release
  (README, "Recommended production process").

## Constraints
- Corpora are internal IDF documents: never committed, never quoted in commits, docs or tests.
- Gemini quota and cost: optional cost-adding checks are off by default.

## Open questions (for the owner)
- [ ] `docs/theme-layer-ab-results.md` quotes Hebrew topic titles derived from the hova/keva corpora. Acceptable in the GitHub repo, or redact?
- [ ] Two interpreters were documented for the `army` env (`/Users/sergiyhoref/Programming/Army/army/bin/python` in AGENTS.md, `/opt/homebrew/Caskroom/miniforge/base/envs/army/bin/python` formerly in README). AGENTS.md keeps the first; confirm.
- [ ] Next priorities after user-facing mode (README lists incremental regeneration, a second-pass quality grader, judge calibration, persistent review workflows).
- [ ] `docs/RUNS.md` (committed) lists local run folders by path, e.g. `hova/outputs/oren-prompt`, whose name may be a person's first name. Acceptable, or should such folders be renamed or the path column dropped? - the next `docmap.py runs` re-adds every local folder.
- [ ] The untracked local folder `Claude outputs/` (agent prompts and the project-foundations skill) is read by `docmap.py`, which lists untracked files, and adds 4 local-only decision-ID warnings to `check`. Track it elsewhere, add it to `.gitignore`, or leave it? - CI and fresh clones are unaffected.
