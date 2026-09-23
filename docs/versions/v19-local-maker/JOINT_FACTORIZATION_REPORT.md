# Learned sequence dependence: independently verified

We tested whether learned dependence between goal and operation sequences improves joint-process prediction. With complete witnesses and 2,048 labels, removing that dependence increases the predictive bank’s logarithmic loss by 2.30965 nats; the matched-frequency baseline shows a similar penalty. Independent reconstruction, all 80 paired estimates and both complete replays verify this constructed-method advantage. The exact reference needs no such dependence under complete witnesses, so this does not establish recovered historical process or human intent.

Each frozen joint forecast is compared with the product of its own goal-sequence
and operation-sequence marginals. There are 27 three-goal sequences and 216
three-operation sequences, giving 5,832 joint labels. Dependence within each
sequence remains. Every unknown tail is expanded over unseen labels; no public
or evaluator support mask is applied. Products can create unsupported cross pairs.

The original roster remains four evidence tiers, budgets 32/128/512/2048, sixteen
development coefficient lineages, two training draws and five paired feature seeds.
All four readouts remain: raw history, frozen latent state, predictive bank and
matched frequencies. Native posteriors and their products are evaluator-only rulers.
No fit, new episode, protected lineage or confirmation enters this diagnostic.

Logarithmic loss is native weighted negative log forecast probability, in nats;
lower is better. The first table reports product-minus-original loss at 2,048
labels. A positive number favors the joint forecast. Brackets are paired 95%
sixteen-lineage bootstrap intervals, conditional on both saved draws and all five
seeds averaged within each lineage. Native penalty averages the separate supplied
reference comparison. These intervals do not capture general training uncertainty.

| Evidence supplied | Predictive bank penalty, nats | Matched-frequency penalty | Native reference penalty |
| --- | ---: | ---: | ---: |
| Final artifact | 2.31343 [2.21921, 2.40858] | 2.31875 [2.22401, 2.41422] | 2.35104 |
| Artifact and truthful context | 2.31321 [2.21899, 2.40835] | 2.31875 [2.22401, 2.41422] | 1.86740 |
| Context and first operation | 2.31197 [2.21779, 2.40706] | 2.31875 [2.22401, 2.41422] | 0.92121 |
| Complete operation witnesses | 2.30965 [2.21557, 2.40460] | 2.31875 [2.22401, 2.41422] | 0.00000 |

The second table averages the same loss difference over log label budget, using
normalized trapezoid area. Columns name the original readout; every value compares
that readout with its own marginal product. The complete receipt retains each
budget, both training-draw means and all five feature-seed means separately.

| Evidence supplied | Raw history | Frozen latent state | Predictive bank | Matched frequencies |
| --- | ---: | ---: | ---: | ---: |
| Final artifact | 1.18570 [1.05230, 1.32296] | 1.18307 [1.04962, 1.32039] | 1.18255 [1.04910, 1.31987] | 1.16218 [1.02622, 1.30200] |
| Artifact and truthful context | 1.18779 [1.05454, 1.32493] | 1.18286 [1.04941, 1.32019] | 1.18221 [1.04877, 1.31952] | 1.16218 [1.02622, 1.30200] |
| Context and first operation | 1.08050 [0.94904, 1.21591] | 1.17772 [1.04432, 1.31498] | 1.18019 [1.04677, 1.31748] | 1.16218 [1.02622, 1.30200] |
| Complete operation witnesses | 0.87375 [0.74477, 1.00666] | 1.16986 [1.03652, 1.30706] | 1.17718 [1.04384, 1.31443] | 1.16218 [1.02622, 1.30200] |

Every learning-area contrast favors the original joint forecast. This establishes
a conditional advantage over the deliberately factorized version of the same
forecast, not superiority over the other readouts. Matched frequencies retain
nearly the same benefit. With full witnesses, the native operation sequence is
fixed and its product is exactly unchanged. The learned penalty therefore exposes
useful structure in an imperfect forecast, rather than a need to infer an unknown
operation sequence in that evidence condition. It does not promote the primary
bank-versus-history/latent comparison or establish process correspondence.

At the largest budget with full witnesses, the bank's loss rises from 5.60252 to
7.91217 nats. Its probability on native-compatible processes falls from 0.01123 to
0.00135. The nominal 90% candidate set expands from 175.5 to 2,308.33 labels on
average; native candidate coverage barely changes (0.84190 to 0.84352). Per-step
goal and operation accuracies stay unchanged because the corresponding sequence
marginals are preserved. A broader candidate set is not stronger localization.

All 1,994 deterministic files match across original, adjacent-source and complete
extracted-source replay. All 619 producer and 622 checker source files, source
archives, plans, environments, input/output bindings and timing hashes verify.
The independent checker covers 145,920 learned frame forecasts, 14,592 native
frame/law combinations, 20,480 learned strata, 128 native rows and 912 packets.
Maximum marginal/product discrepancy is 5.00e-16; maximum score discrepancy is
5.92e-12, within the frozen 1e-10 score tolerance. All original readout metrics
reproduce within 6.54e-13. A separate direct-index regroup of original raw rows
verifies all 64 budget contrasts, 16 learning areas and 128 metric means against
both checker regroupings within 4.55e-13.

Thirty-three isolated controls passed before dispatch. Scalar integer label
memberships and accurate sums independently reconstruct marginals; every product
is checked. Saved binary64 marginal products are scored after that check to retain
executed ties. Independent lexicographic sorting retains original alphabet-first
priority for learned arms and full-label priority for native rulers; 90% sets stop
at their first cumulative crossing. All loss, squared error, compatible mass,
coverage/size, unsupported mode, confidence, abstention, unknown mass and per-step
goal/operation fields reconstruct. No producer factorization, score or regroup
routine is imported. Native targets and fitted forecasts remain accepted inputs;
this review is not a new refit or independent derivation of the generative policy.

Reader inputs are 912 unchanged anonymous evidence packets. Native truth, weights,
fitted forecasts, marginal arrays, scores and regroupings stay in separate
scientific/evaluator exports. This is an exploratory constructed-method result,
miniature — architecture untested beyond admitted rosters. No human intent,
causal access, confirmation or conditional A3/C2/F2 admission follows. Temporal
factorization removes between-step dependence and remains a distinct next test;
retrospective evidence updating is independent. Both tiny settings remain consumed.

The inclusive 1,800 CPU-second card and campaign clocks are unchanged. Raw data,
failed preparations, checks, source and replay are retained.

[Admission contract](JOINT_FACTORIZATION_ADMISSION_PROTOCOL.md).
[Independent review](JOINT_FACTORIZATION_REVIEW_PROTOCOL.md).
[Accepted reconstruction](../../../results/v19/G19-A-joint-factorization-1/FINAL_REVIEW.json).
[Full regroup](../../../results/v19/G19-A-joint-factorization-1/INDEPENDENT_REGROUP.json).
[Evidence exports](../../../results/v19/G19-A-joint-factorization-1/EVIDENCE_ROLES.json).
