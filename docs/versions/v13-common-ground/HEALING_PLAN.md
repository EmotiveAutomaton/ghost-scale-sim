# V13 healing pass — runbook and record

Written 2026-08-27 while the main pass was on Q09; rewritten 2026-08-28 at launch of the healing
pass, after the main pass reported idle (17:57) with 228 ledger entries. Every item here is an
instrument correction discovered during the run; none touches a card definition, a criterion, or
a lock-covered file. The scientific lock hashes the eleven substrate modules, the card
definitions, the cells template and the report interface — not the trunk instrument code.

## Corrections, with the cause as actually established

| Card(s) | Symptom in the main pass | Cause | Fix (commit) |
|---|---|---|---|
| A02, A03, A12, A13 | ERROR at world 0/1: `KeyError('convention_obs')` | poe families emit no rich channels; the cue scrambler read them unguarded | guard the two reads (`999c8f6`); **healed in-line by the watchdog relaunches** (A03, A05, A12, A13 landed; A02 failed its positive control honestly) |
| A05 | ERROR: `bincount` refused negative goals | poe families carry `goal = -1` (`world.artifact`) | drop sentinel goals from the concentration (`ee3a023`); healed in-line |
| P04 | LANDED, but poe-family units scored `pred[-1]` (silent) | same sentinel used as an index | skip poe families as Q06 does (`ee3a023`); verdict retired, re-run |
| P07 | LANDED; declared history level 8 never realized (validator) | code used histories (2, 5) against the definition's (2, 8) | histories (2, 8) in unit and reduce (`863649f`); verdict retired, confirmation **superseded** by recorded amendment, re-run |
| C06 | RESOURCE_BLOCKED: receipt saw the sparsest cell in 1,038 of 2,560 units | **not** the harness (every unit had rows; the earlier diagnosis in this file was wrong): fixed typicality thresholds (0.85 / 0.93) realized the "low" bin in 40% of units, and the receipt requires every cell in every unit | bins are per-unit tertiles of typicality; the criterion (rank correlation of gain with typicality) is untouched; verdict retired, re-run |
| L04, L12 | INSTRUMENT_FAILED in 0.3 s: licensing cards "UNRUN" | the licensing gate read only discovery verdicts; the five transfer-only cards (A14, C16, G16, H16, Q12) resolve in the transfer lane | the gate reads the card's own lane; verdicts retired, re-run |
| A14, C16, G16, H16, Q12 | resolved in the transfer ledger, but the manifest and coverage called them BUILT | `run_lane` updated manifest status only for the discovery lane | a transfer-only card's transfer run owns its status; the five statuses backfilled from their transfer verdicts |
| C02, C03, C05, C12, A02, A06, O04, P01 (discovery); Q12 (transfer); A09, O11 (confirmation) | INSTRUMENT_FAILED | controls that held at the 6-world smoke and not at full n | none — curator decision whether to recalibrate and re-run |

## What the record holds

- Retired discovery verdicts: `results/validation/soundingline/v13/superseded/<card>.<stamp>.json`;
  the superseded P07 confirmation: `.../confirmation/superseded/P07.<stamp>.json`.
- Each retirement is an entry in `results/v13/AMENDMENTS.json` (kind `discovery_verdict:<card>`)
  with the retired verdict's state and hash; the P07 packet change is amendment 1 in
  `results/v13/CONFIRMATION.json` (`superseded`, `amendments`, original packet preserved).
- `runners/run_v13_confirmation.py --supersede <ids>` is the recorded inverse of an amendment:
  it removes a card and its hash from the frozen packet so the ordinary amendment path re-adds it
  with the new discovery hash. Tests: `tests/test_v13_confirmation_freeze.py` (three added).

## The healing pass itself (launched 2026-08-28 ~19:30)

`runners/run_v13.py --stage all` under a fresh `runners/watchdog_v13.py`. Discovery re-runs C06,
P04, P07 (ERROR cards were already retried in-line); transfer and attacks skip their ledger
entries; confirmation verifies the 70-card packet, then amends it with whatever the re-runs
promote (P07 re-enters with its new hash); bridge re-runs L04 and L12; the report regenerates.

## After it reports idle

1. `runners/validate_v13_program.py` (full) must return 0; then the fresh-clone receipt.
2. Only then read results: Pass A of `RESULTS.md`, `FINDINGS.md` and
   `docs/theory/READING_INTENT.md` in the same pass, the Pass-B packet from `runners/report_v13.py`.
3. Decide with the curator: the eleven instrument failures above; moving per-unit residual lists
   (2.65 MB each in C04 and P01) to a gitignored sidecar with the aggregate kept in the verdict.

## Disclosure

Every re-run above replaces a verdict produced by an instrument that was wrong in a way the
6-world smoke could not see (no poe family was sampled; receipts are not checked at smoke
scale). The direction of each correction is fixed by the definition, by the generator's
documented semantics, or by the receipt rule — not by the outcome. Superseded verdicts stay on
disk and in git history. The main pass died silently three times (22:25, 03:55, 06:33); no
traceback, no stderr, no fault event; the watchdog resumed each time with ledger growth.
