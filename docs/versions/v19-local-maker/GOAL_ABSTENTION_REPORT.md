# Goal-report abstention: independently verified coverage tradeoff

We tested whether declining uncertain local-goal reports reduces errors at useful coverage. At 2,048 labels and an abstention cost of 0.1, separate-step reporting retains 76.43% coverage and lowers the predictive bank's error among reported steps from 13.45% to 0.12%. Independent reconstruction, all 1,920 paired estimates and both complete replays verify this constructed-method tradeoff. The forecast is unchanged; historical goal correspondence and human intent remain unestablished.

Incorrect step reports cost one, correct reports cost zero, and declining a step
costs 0.1, 0.25 or 0.5. These are stipulated decision costs, not measured user
preferences. Every forecast chooses reports using its own probabilities. Native
posterior weights only score them. The three policies always report separate
coordinate modes, report a joint mode all-or-none, or decline steps independently.
Equality abstains; whole-path probability is never substituted for mean step error.

The table shows predictive-bank reports at 2,048 labels under the original
operation-restricted forecast. Coverage is the percentage of all step opportunities
reported. Conditional error is incorrect reports divided by reports made. Native
cost averages mistakes plus the stated cost of each abstained step; lower is better.
Every row averages the same sixteen development laws, two draws and five seeds.

| Abstention cost | Policy | Coverage, percent | Conditional error, percent | Native cost per step |
| ---: | --- | ---: | ---: | ---: |
| 0.1 | Joint all-or-none | 48.85734 | 1.19274 | 0.056970 |
| 0.1 | Separate-step abstention | 76.42785 | 0.11945 | 0.024485 |
| 0.25 | Joint all-or-none | 86.57646 | 9.20873 | 0.113285 |
| 0.25 | Separate-step abstention | 77.12483 | 0.46641 | 0.060785 |
| 0.5 | Joint all-or-none | 99.30396 | 13.05362 | 0.133108 |
| 0.5 | Separate-step abstention | 89.71921 | 7.84827 | 0.121818 |

Always reporting has 100% coverage, 13.45341% conditional error and cost 0.134534
at every abstention cost. At cost 0.1, step abstention reduces cost by 0.110049
against always reporting, with a paired 95% lineage interval of
[0.099164, 0.120438] for the reduction. Joint all-or-none reporting retains only 48.85734%
coverage and costs 0.056970. The step policy therefore preserves more useful
reports under this stipulated objective. It does not improve the 0.913367-nat
forecast loss, which remains identical under every reporting policy.

All four readouts and both forecast variants remain reported at every budget;
no architecture winner or practical-margin test is inferred from the illustrative
bank cell. Conditional-error ratios are pooled before division. Their uncertainty
is not supplied by the additive-metric bootstrap intervals. The bank's reports at
2,048 labels all have native support. Lower-budget reports can lack support:
at 32 labels, always reporting the bank's coordinate modes produces incompatible
whole reports on 60.57473% of native packet mass. At cost 0.1, separate-step
abstention removes those incompatible reports while retaining 27.42200% coverage.
Incompatibility is distinct from being wrong about a realized goal; a compatible
report does not establish which historical goal occurred. An initial draft
overextended the largest-budget zero-incompatibility result to the whole roster;
this paragraph corrects that scope without changing scores or frozen execution.

The complete roster contains 46,080 learned score strata, 144 native rows, 225,280
learned frame forecasts, 704 unchanged public packets and 5,120 parent prediction
and accuracy identities. Four readouts, two frozen forecasts, four nested budgets,
sixteen development lineages, two training draws and five fit seeds are inherited.
No new fit, sample, teacher, protected lineage or tiny setting was used.

Original, adjacent and extracted-source executions match all 668 deterministic
outputs. All source archives, extracted source maps, plans, resident environment,
input/output maps and separate timing hashes verify. The checker reconstructs
every report and score independently. Maximum score and parent identity error is
1.03e-14. Scalar grouping with bootstrap multiplicities verifies original and
reconstructed rows. A third direct-index bootstrap verifies all 1,536 budget
contrasts, 384 normalized log-budget areas and 288 means within 2.78e-15. Seed
191012 fixes 10,000 paired lineage resamples, retaining draws/seeds within each
lineage. Intervals condition on these fits; training variation stays separate.

Twenty-five isolated controls passed before checker dispatch, covering all 64
partial reports, sparse support, scalar native cost, zero coverage, exact/near
thresholds, modal ties, pooled ratios, full fixture execution and 17 corruption
patterns. Failed fixture attempts and all charges remain retained. Inherited
probabilities and marginals are independently checked before binary64 thresholds.
Native records lack saved marginals, so the projection arithmetic primitive is
shared and checked against scalar sums. This does not independently validate the
numerical library. Original training and target-law construction remain inherited
accepted inputs. This is exploratory miniature evidence, architecture untested
beyond the admitted roster, and not confirmation of the primary bank claim.

Reader exports contain public artifacts, context and operation witnesses only.
Forecasts, reports, target distributions and scores are scientific/evaluator
evidence in separate archives. Both shared tiny settings remain consumed.
Sixty-three scientific batches have independent acceptance; less specific goal
reports and retrospective updating remain independent alternatives. The 96-hour
total, protected 16-hour reserve and immutable reporting times remain unchanged.

[Review protocol](GOAL_ABSTENTION_REVIEW_PROTOCOL.md).
[Accepted reconstruction](../../../results/v19/G19-D-goal-abstention-1/FINAL_REVIEW.json).
[Full regroup](../../../results/v19/G19-D-goal-abstention-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-D-goal-abstention-1/EVIDENCE_ROLES.json).

Goal coarsening is source admitted after 39 isolated controls. The existing serial queue dispatched all three binary goal partitions and the fine/one-class endpoints, with both complete replays, at verified below-normal priority. Numerical acceptance awaits completion and independent reconstruction. Retrospective updating and fixed-bin goal-confidence calibration remain two prepared independent alternatives. No new fit or protected lineage is used.
