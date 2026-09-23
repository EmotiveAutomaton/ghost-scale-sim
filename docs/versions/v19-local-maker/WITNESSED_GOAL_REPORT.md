# Goal dependence under complete witnesses: independently verified reversal

We tested whether dependence among goals improves prediction after all operations are witnessed. Removing it lowers the predictive bank’s logarithmic loss by 0.14852 nats at 2,048 labels; matched frequencies improve by 0.15200. Independent reconstruction, all 20 paired estimates and both complete replays verify this constructed-method reversal. The native reference instead loses accuracy, so the learned gain does not establish historical goal correspondence or human intent.

Each original forecast covers 5,832 three-step goal/operation paths. Complete
public witnesses restrict it to 27 goal sequences sharing the observed operations.
The comparison multiplies the three conditional goal marginals from that restricted
forecast. Original, restricted and goal-product scores remain separate. There is
no new fit, sample, evidence selection or evaluator-support restriction.

The roster is sixteen development coefficient lineages, two training draws,
five paired feature seeds, budgets 32/128/512/2048 and four readouts. Every readout
keeps its original matched outcomes and frozen fit. Operations are supplied by
the evidence. Confirmation and test lineages remain untouched.

Logarithmic loss is the native weighted negative natural logarithm of forecast
probability, measured in nats; lower is better. The table reports goal-product
minus restricted loss at 2,048 labels and over the normalized log-budget learning
curve. Negative values favor removing goal dependence. Brackets are paired 95%
sixteen-lineage bootstrap intervals, conditional on these fits. Both draws and
five seeds are averaged inside lineage; their separate variation remains in the
full receipt. The last column is the resulting product loss at the largest budget.

| Readout | Change at 2,048 labels, nats | Learning-curve change, nats | Product loss |
| --- | ---: | ---: | ---: |
| Raw history | -0.15084 [-0.17778, -0.12450] | -0.27259 [-0.29730, -0.24716] | 0.77797 |
| Frozen latent state | -0.14858 [-0.17516, -0.12261] | -0.27724 [-0.30267, -0.25111] | 0.76506 |
| Predictive bank | -0.14852 [-0.17509, -0.12256] | -0.27734 [-0.30278, -0.25118] | 0.76484 |
| Matched frequencies | -0.15200 [-0.17937, -0.12530] | -0.28073 [-0.30680, -0.25398] | 0.75546 |

All sixteen per-budget intervals and four learning-area intervals favor removing
goal dependence. Every interval lies beyond the 0.02-nat practical margin in that
direction. This is a reversal for these frozen learned distributions, not a general
claim that goals are independent. Native joint and restricted loss are both 0.68049;
the native goal product raises loss to 0.68192, a 0.00143-nat penalty. The bank is
not uniquely advantaged: matched frequencies retain lower loss after factorization.
No primary bank-versus-history/latent comparison is promoted by this diagnostic.

At 2,048 labels the bank's restricted loss falls from 0.91337 to 0.76484 nats.
Native-compatible probability falls slightly from 0.999703 to 0.999475. The nominal
90% candidate set grows from 2.38000 to 2.50543 labels and native candidate coverage
rises from 0.95776 to 0.97165. Probability outside the training alphabet rises from
0.000301 to 0.011262. Abstention rises from 0.33190 to 0.34600. These separate
measures prevent a better proper score from being described as uniformly better
localization or certainty.

The three goal marginals are preserved mathematically, but goal accuracy measured
at the selected joint-mode path changes from 0.86630 to 0.86547 for the bank.
That path need not combine the separate marginal modes. Changing dependence can
therefore change this localization measure without changing the marginals. The
checker also reproduces executed binary64 values and exact-tie priority. Operation accuracy
remains one because the operations were supplied. Complete witnesses still do
not specify the historical goal sequence.

All 509 deterministic files match across original, adjacent-source and extracted-
source replay. Every producer/checker source, archive, plan, environment, input,
output and separate timing binding verifies (640 producer and 643 checker source
files; 328 and 512 inputs). Independent scalar reconstruction verifies 112,640
learned frame forecasts, 11,264 native frame/law combinations, 7,680 learned strata,
48 native strata and 704 packets. Maximum marginal/product error is 2.11e-15;
maximum score error is 5.12e-13. All 2,560 original cells reproduce within 6.54e-13
and accepted public-support scores within 3.91e-14. These pass the frozen 1e-12
probability and 1e-10 score tolerances.

A separate direct-index bootstrap of original rows reproduces the sixteen budget
contrasts, four normalized log-budget areas and 48 means against both checker
regroupings within 5.69e-14. There are 10,000 paired resamples, seed 191008.
Lineage intervals do not quantify general training-population uncertainty; only
two training draws exist. All 45 isolated controls passed before checker dispatch,
including known dependence/independence, zero/tiny conditioning mass, exact witness
support, ties and deliberate source, score, marginal and roster corruption.
The numerical checker imports no producer conditioning, product, score or regroup
routine. Saved binary64 values retain executed ties after independent agreement.
Original fitted forecasts and native targets remain inherited accepted inputs;
this diagnostic does not retrain or independently regenerate them.

Reader exports contain only 704 unchanged anonymous complete-witness packets.
Native targets, weights, forecasts, marginals, scores and raw regroupings stay in
scientific/evaluator exports with hashes. This is an exploratory constructed-method
reversal, miniature — architecture untested beyond admitted rosters. Historical
goal correspondence, selective causal access, human intent and confirmation are
unestablished. Both shared tiny settings are consumed.

Temporal alphabet-mass matching asks whether the prior temporal-product gain
comes from redistributing probability beyond the training alphabet. Retrospective
evidence updating remains independent. The unchanged 96-hour total, 16-hour reserve,
inclusive 1,800-second card cap and immutable reporting times remain binding.

[Independent review](WITNESSED_GOAL_REVIEW_PROTOCOL.md).
[Accepted reconstruction](../../../results/v19/G19-A-witnessed-goal-factorization-1/FINAL_REVIEW.json).
[Full regroup](../../../results/v19/G19-A-witnessed-goal-factorization-1/INDEPENDENT_REGROUP.json).
[Evidence roles](../../../results/v19/G19-A-witnessed-goal-factorization-1/EVIDENCE_ROLES.json).

The conditional-goal alphabet-mass control passes 61 isolated controls and is source admitted with both complete replays in the existing serial queue. Its conservative four-pass estimate is 721.09 CPU seconds within the inclusive 1,800-second card. The full temporal alphabet-mass comparison remains timing-blocked after one exact vectorization improvement. Retrospective updating and goal-decision decoding remain two independently prepared alternatives.

[Conditional mass contract](GOAL_MASS_CONTROL_ADMISSION_PROTOCOL.md). [Decision alternative](GOAL_DECISION_PROTOCOL.md). [Temporal timing blocker](../../../results/v19/G19-A-temporal-mass-control-blocker/DISPOSITION.json).
