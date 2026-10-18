# N13: the falsifier's power requirement fails

Can the FX1 falsifier detect the planted minimum goal signal? N13 fails the power requirement: 2 of 16 law blocks clear zero at 81 records, versus 13 required. Habit-only worlds return 0 false SUPPORTED blocks, meeting that requirement. All 9,216 cases and full replay verify. This is a constructed-method limitation; FX1 remains unready to falsify the theory.

The primary power rate is 12.5%, against the frozen requirement of at least
81.25%. This is a scientific outcome, not an instrument failure: calibration,
known-answer recovery, score reconstruction and complete replay pass. No bar,
margin or sample count changed after seeing production. The passed null check
does not compensate for failed power. Because P3 fails, no setting can receive
FALSIFIED; all such would-be conclusions stay INCONCLUSIVE.

The table shows planted oracle strength (in units of the frozen margin), records
per maker, the margin in half-Brier units, goal-reader gain over records-only
with a descriptive 95% whole-law interval, blocks whose within-law lower bound
clears zero, and SUPPORTED/FALSIFIED/INCONCLUSIVE counts out of sixteen.

| Planted oracle strength | Records | Frozen margin | Mean gain (95% law interval) | Positive-lower-bound blocks | Verdict counts S / F / I |
|---|---:|---:|---|---:|---|
| 0m | 32 | 0.00371767 | 0.00000000 (0.00000000, 0.00000000) | roundoff only | 0 / 0 / 16 |
| 0m | 81 | 0.00127979 | 0.00000000 (-0.00000000, 0.00000000) | roundoff only | 0 / 0 / 16 |
| 0m | 250 | 0.00008219 | -0.00000000 (-0.00000000, 0.00000000) | roundoff only | 0 / 0 / 16 |
| 1m | 32 | 0.00367288 | 0.00199555 (0.00125903, 0.00276214) | 4 | 2 / 0 / 14 |
| 1m | 81 | 0.00129378 | 0.00063845 (0.00028496, 0.00103133) | 2 | 2 / 0 / 14 |
| 1m | 250 | 0.00008266 | 0.00000608 (-0.00009230, 0.00010703) | 0 | 0 / 0 / 16 |
| 2m | 32 | 0.00369949 | 0.00579245 (0.00447373, 0.00715029) | 8 | 8 / 0 / 8 |
| 2m | 81 | 0.00129275 | 0.00150926 (0.00104702, 0.00200950) | 3 | 3 / 0 / 13 |
| 2m | 250 | 0.00008393 | 0.00013874 (-0.00008093, 0.00037779) | 0 | 0 / 0 / 16 |

Habit-only gains are numerical zero (absolute means below 2e-17); signs of their
floating-point residue have no scientific interpretation. Their means never
reach the positive margin, so the false-SUPPORTED result is unaffected. Generic
validation-selected tempering chooses exponent one in all nine settings; both
reader comparisons against that rival are retained and do not add a new benefit.

The primary uses Sounding Line V5's delivered 42.1769% repeated-choice agreement,
matching it over the balanced sixteen laws. Goal and habit makers have exactly
equal consistency within each law. This matches one delivered aggregate, not the
full real-text population or its contexts. The three requested 70% sensitivities
are explicitly UNAVAILABLE by owner-approved interpretation: 70% is higher than
the delivered value and infeasible in some required uniform-noise laws.

At 81 records and strength m, the frozen margin is .001293781 half-Brier units.
The independent oracle calibration estimates .001086806, within the predeclared
calibration tolerance. The observed inferred-goal gain is .000638451. Knowing
the latent goal and inferring it from limited records are distinct tasks. At
250 records the records-only reader is near its personal floor, so the registered
headroom-scaled margin shrinks; that setting is not a simple larger-sample power
replication of the 81-record test. Neither more records nor a larger margin is
declared a proven remedy by these results.

Verification covers complete source and fit bindings, all 9,216 cases, full
fresh-process replay, independent proper scores and 171 across-law intervals.
A separate scalar half-Brier reviewer reconstructs 432 within-law gain/headroom/
pooled intervals and the primary counts from retained predictions and outcomes.
Within-law Student intervals are the declared finite-sample approximation;
across-law intervals bootstrap whole laws. No real-data FX1 headroom, reader
parity or theory precondition is established by this miniature.

Route to Sounding Line: **P3 fails power and passes the false-SUPPORTED check;
FX1 waits.** Constructed-method validation, miniature—architecture untested;
outside confirmation. [Protocol](N13_PROTOCOL.md),
[calibration](../../../results/v23/n13-calibration/CALIBRATION.json),
[complete summary](../../../results/v23/n13-decision-review/SUMMARY.json),
[independent decisions](../../../results/v23/n13-decision-review/DECISION_REVIEW.json),
[provenance](../../../results/v23/n13-decision-review/PROVENANCE.json).
