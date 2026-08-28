# V13 healing pass — runbook

Written 2026-08-27 while the main pass was on Q09 (99 of 152 cards resolved). Every item here
is an instrument correction discovered during the run; none touches a card definition, a
criterion, or a lock-covered file. The scientific lock hashes the eleven substrate modules, the
card definitions, the cells template and the report interface — not the trunk instrument code.

## What is wrong, and the fix already on disk

| Card(s) | Symptom in the main pass | Cause | Fix (commit) |
|---|---|---|---|
| A02, A03, A12, A13 | ERROR at world 0/1: `KeyError('convention_obs')` | poe families emit no rich channels; the cue scrambler read them unguarded | guard the two reads (`999c8f6`) |
| A05 | ERROR: `bincount` refused negative goals | poe families carry `goal = -1` | drop sentinel goals from the concentration (`ee3a023`) |
| P04 | LANDED, but poe-family units scored `pred[-1]` (silent) | same sentinel used as an index | skip poe families as Q06 does (`ee3a023`) |
| P07 | LANDED; declared history level 8 never realized | code used histories (2, 5) against the definition's (2, 8) | histories (2, 8) in unit and reduce (this commit) |
| C06 | RESOURCE_BLOCKED: 1038 of 2560 units produced rows | harness samples makers independently of reader families; readers with no family-mates yield no rows | **not yet written** — must be a change inside `unit_C06` only (other C cards landed on the shared harness) |
| C02, C03, C05, C12, A06, O04, P01 | INSTRUMENT_FAILED (controls that held at the 6-world smoke) | under-powered or mis-set instruments at full n | none — curator decision whether to recalibrate and re-run |

Crashed cards (ERROR) are not a resolved state: the runner retries them on the next invocation
with no manual step. LANDED and RESOURCE_BLOCKED are resolved: a re-run needs the ledger entry
removed and the manifest status reset.

## Steps, in order, after `RUNNER_STATUS.json` reads `stage: idle`

1. Confirm the watchdog exited (`results/v13/logs/watchdog.log`) and no `WATCHDOG_HALTED.json`.
2. Write the C06 fix inside `unit_C06` (top up `fam_makers` for a reader whose family has fewer
   than two harness makers by computing the same likelihood cache for extra same-family makers;
   do not change `harness()`), smoke it at six worlds including a poe family.
3. Reset for re-run: P04, P07, C06 — delete `discovery:<id>` from `results/v13/COMPLETION.json`,
   set status `BUILT` in `results/v13/QUEUE_MANIFEST.json`, remove their checkpoint files.
4. `./.venv/Scripts/python.exe runners/run_v13.py --stage all` (detached, below-normal priority,
   watchdog alongside). Discovery retries A02 A03 A05 A12 A13 and the three reset cards; transfer
   and attacks skip ledger-resolved entries; confirmation adds newly promoted cards; bridge and
   report regenerate.
5. `runners/validate_v13_program.py` (full, not `--interim`) must return 0; then the fresh-clone
   receipt (`runners/fresh_clone_v13.py`).
6. Only then read results: Pass A of `RESULTS.md`, `FINDINGS.md` and `docs/theory/READING_INTENT.md`
   in the same pass, the Pass-B packet from `runners/report_v13.py`.
7. Decide with the curator: the seven instrument failures; moving per-unit residual lists
   (2.65 MB each in C04 and P01) to a gitignored sidecar with the aggregate kept in the verdict.

## Disclosure

Every re-run above replaces a verdict produced by an instrument that was wrong in a way the
6-world smoke could not see (no poe family was sampled). The direction of each correction is
fixed by the definition or by the generator's documented semantics, not by the outcome. The
superseded verdicts stay in git history.
