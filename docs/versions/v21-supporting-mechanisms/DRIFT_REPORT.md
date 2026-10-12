# V21 role changes and matched long-history support

We asked whether separating preference, skill and opportunity changes improves reconstruction. The one-role model has lower mean loss than unrestricted transitions in all 24 observed-change cells, but does not uniformly beat a static model. In the 96-episode, position-24 example it identifies skill changes correctly 67.2% of the time. This is a supplied-law constructed-method result, miniature — architecture untested; it establishes neither stable human values nor confirmation.

Eight reused development coefficient worlds supply 16 streams apiece. Each stream has binary preference, skill and opportunity-law states and a known endpoint channel. A single assigned role changes at absolute position 8, 16, 24, 48 or 72, or no role changes. The 32- and 96-episode readers consume exact prefixes of the same underlying stream at each fixed absolute position. The forecast is the next episode, so its target differs between horizons. Positions beyond the short prefix are future changes, not observed changes. Repeated no-change rows across positions are identical controls. Whole-world paired intervals use 4,096 resamples; cases, fits, horizons and repeated controls are not independent worlds. No model fits occur.

All five readers know the finite channel and state vocabulary. Static inference assumes no change; fast and slow filters reset all roles with hazards 1/8 and 1/64. The one-role mixture assigns prior .25 to no change and otherwise one uniformly located change across 95 gaps, flipping one role. The unrestricted chronological rival shares the change-time prior but can transition to any of the seven different states. This is unequal transition capacity and prior mass, not a matched learned architecture comparison. Neither true role nor change time enters any reader. Historical aliasing and supplied channel knowledge remain distinct from fitted capacity, validation convergence, support privilege and uncertainty about an unknown law.

The table fixes the shared absolute change position at 24 and the forecast horizon at 96. Rows identify the intervention. Differences are the one-role model minus the named rival in uncapped full-target natural-log loss, where a negative difference favors the one-role model. Brackets are descriptive paired-world 95% intervals. These are illustrative registered cells, not selected promotions.

| Intervention | Against static | Against unrestricted transitions |
|---|---:|---:|
| No change | +0.21625 [+0.14720, +0.27249] | +0.01597 [-0.00203, +0.03426] |
| Preference change | -1.63098 [-2.15140, -1.12163] | -0.03923 [-0.06740, -0.01425] |
| Skill change | -0.03048 [-0.15544, +0.07667] | -0.14120 [-0.20739, -0.08347] |
| Opportunity-law change | +0.01358 [+0.00005, +0.03655] | -0.01871 [-0.05846, +0.01950] |

Across all 24 cells with a change inside the observed prefix, the one-role model has lower mean joint loss than unrestricted transitions; 21 descriptive intervals exclude zero. Against the static model it has lower mean loss in 19 of 24, with 17 intervals below zero and five above zero. At the example above, the preference-change gain against unrestricted transitions is .03923 nats, but preference-state accuracy is 3.125 percentage points lower. Its full multiclass Brier gain of .00505 has an interval including zero. For skill change, the .14120-nat gain and .01787 Brier gain coexist with only a small, uncertain skill-state accuracy gain. Prior restrictions, proper losses and individual attribution must not be conflated.

The next table separates current-state recovery from identifying what changed, at the same position and horizon. Preference and skill columns give the one-role reader’s fraction of correctly recovered current binary states. The last column is the fraction correctly identifying no change or the changed role. All are finite descriptive means, not independent demonstrations of stable traits.

| Intervention | Preference state correct | Skill state correct | Change type correct |
|---|---:|---:|---:|
| No change | 96.88% | 90.62% | 60.94% |
| Preference change | 83.59% | 98.44% | 75.00% |
| Skill change | 93.75% | 92.19% | 67.19% |
| Opportunity-law change | 96.09% | 93.75% | 97.66% |

This miniature has no original-world changed-execution arm. Its relevant paired comparison is each reader’s role-change minus no-change loss. The table gives those differences at position 24 and horizon 96; negative means lower joint loss under that intervention, not learned transfer. All intervals, other positions/horizons, separate bit losses, future-artifact prediction and full Brier are retained in the [complete contrasts](../../../results/v21/drift/DRIFT.json).

| Reader | Preference change | Skill change | Opportunity-law change |
|---|---:|---:|---:|
| static | +1.80077 | +0.20091 | +0.02476 |
| all-fast | +0.12565 | +0.12165 | +0.06701 |
| all-slow | +0.14461 | +0.17245 | +0.04921 |
| role-mixture | -0.04646 | -0.04582 | -0.17791 |
| direct-chronological | +0.00873 | +0.11135 | -0.14322 |

All repeated no-change forecasts agree exactly across five positions: 5,120 duplicate-control forecast rows were checked and never counted as additional worlds. Every chronological observation is a distinct episode draw. This ruler has no duplicated product-of-factors reader to remove; earlier learned-reader duplicate controls remain separately bound. The source-bound [admission](B04_DRIFT_ADMISSION.json) specifies the changed-role prior, unavailable information and scope. No trained chronological model, law-marginal ceiling or original-world headroom is inherited. Stable-value attribution and architecture promotion remain unsupported.

The supplied support event completes the shifted counterpart of the prior 64-history, complete-current observation diagnostic, with inspection hidden in the current episode. Both unchanged direct fits are validation-stopped, nested in each of eight worlds. The shifted exact-law headroom lower bound is .29047 nats, above the frozen .02 gate; the native bound is .27899. This is a retained-predictor support intervention, not a method-by-support architecture comparison.

The support table reports changed-execution uncapped log loss, full multiclass Brier (the sum of squared probability errors) and the fraction of true histories removed. Lower proper losses are better; removing any true history yields infinite uncapped log loss.

| Support rule | Log loss | Full Brier | Truth removed |
|---|---:|---:|---:|
| none | 3.46802 | 0.97368 | 0.00% |
| physical | 3.38813 | 0.97899 | 0.00% |
| learned | infinite | 0.97797 | 3.34% |
| supplied-exact | 1.00716 | 0.67624 | 0.00% |

Learned support excludes 3.34% of shifted truths versus 16.43% natively: both losses are infinite, so subtracting them does not produce a finite shift benefit. Exact support lowers shifted log loss by 2.46086 nats and Brier by .29744; physical support improves log loss while worsening Brier. Supplied exact support is a deployment privilege. Fixed candidate omissions retain their own exclusions and infinite losses in the [support record](../../../results/v21/drift/SUPPORT.json).

Every mask’s matched changed-minus-native contrast appears in the [support gate and shift record](../../../results/v21/drift/SUPPORT_GATE.json), with fits averaged inside worlds. The same record includes every upstream reader and duplicate removal: changed-minus-native loss is −.00171 nats for the converged direct reader, −.08535 for naive Bayes, and +.23776 with its evidence factor removed. All three intervals include zero. These are new V21 development worlds, not fresh confirmation of reused V20 worlds.

All 244 immutable complete packets bind to their plans, frozen sources, archives, acceptance and environment: 8,812,544 forecasts and 53,006 retained files verify, reusing 240 exact score receipts. The original role-change block contributes 25,600 forecasts. Its complete frozen scientific replay matches 2,721 files, excluding only CPU timing in one summary. Independently reconstructed elementary channels and scalar chronological forecasts agree within 4.16e-16. A separate finite change-path enumeration checks all 10,240 retained role-probability rows within 7.05e-14. All 320 opaque reader files contain only their permitted prefixes; 2,560 paired streams verify.

The shifted support block contributes 32,768 forecasts; all masks, retained scores and supplied posteriors reconstruct, with maximum probability discrepancy 2.84e-14. The unchanged 129-file C03 scientific replay is reused by matching source and receipt hashes. Two scientific blocks are newly accepted, bringing the total to 81. Numerical packet counts are not counts of findings, and no confirmation world is exposed. [Validity and full replay](../../../results/v21/drift/VALIDITY.json).

The original role-change consumer records 68.078125 CPU seconds including generation, checks and exports; shared consumer timing does not establish isolated per-reader cost. B04’s prior export repair remains its only family repair, and its scientific source is unchanged. The first reviewer marginal summary lost tiny positive probabilities by subtracting a nearly unit mass; it is retained locally, and all marginal losses were rebuilt by summing the relevant saved joint log probabilities. Forecasts and scientific criteria did not change. The old supervisor’s temporary-file race was already repaired; this review verifies the same recovered native owner and original adopted workers before resolving its supplied failure event. No service or environment belonging to Sounding Line is changed.

The independent B05 memory branch is now implemented and admitted after three focused controls, two complete 85-file engineering consumers and a full-size one-world pilot at 1.15625 CPU seconds. Original plus full replay forecasts 27.75 CPU seconds with 50% margin, under an aggregate 1,800-second cap. One pre-dispatch schema repair separates its 512-episode memory length from the original-world history field; the rejected frozen source and all costs remain retained. No scientific identity was overwritten and no second B05 repair is available. [Frozen memory admission](B05_MEMORY_ADMISSION.json). Scientific acceptance awaits its full replay.

The memory ruler compares actual payload bytes for raw history, a sufficient maker posterior, an exact predictive head, a source-linked episode ledger and their hybrid. It separates future endpoint prediction from witnessed past order/inspection, with explicit codec and supplied-law privileges. It does not claim a learned compression architecture. B06/E04 and remaining candidate-omission, decisive-selection and retraction fixture categories remain useful conditional work. B05 admission is not a declaration that the finite forest is complete. Fixed discovery, confirmation, science cutoff and Friday reporting clocks remain unchanged.
