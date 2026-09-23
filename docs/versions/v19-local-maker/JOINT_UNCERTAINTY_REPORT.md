# Operation and conditional-goal uncertainty: independently verified

We tested whether frozen joint-process forecast error comes from operations or from goals conditional on those operations. With complete witnesses and 2,048 labels, the predictive bank assigns 4.68916 nats of loss to operations already determined by the evidence; conditional-goal loss is 0.91337 nats against native uncertainty of 0.68049. Independent reconstruction, all 360 paired estimates and both complete replays verify this constructed-method diagnostic; it does not establish historical process correspondence or human intent.

The original joint-process comparison is frozen: four evidence tiers, four nested
label budgets (32, 128, 512 and 2,048), sixteen development coefficient lineages,
two training draws, five feature seeds, three readouts and matched frequencies.
No support restriction, new model, sampled episode or protected lineage enters.
Native frame probabilities weight cases within each lineage; lineages are equal.

Logarithmic loss is the native weighted negative natural logarithm of a forecast
probability, measured in nats. Joint loss separates into operation-sequence loss
and goal-sequence loss conditional on that operation. Native entropy measures
uncertainty in the supplied exact reference; loss minus entropy is excess loss.
The full unknown tail remains spread over every unseen label in the fixed
5,832-label universe. This diagnostic does not replace the primary joint outcome.

The table averages the largest-budget predictive bank over all sixteen laws,
both draws and all five seeds. Each operation or goal sequence covers three steps.
Entropy is the uncertainty remaining to the supplied native reference, rather
than additional error by the reader.

| Evidence supplied | Joint loss, nats | Operation loss | Operation entropy | Conditional-goal loss | Conditional-goal entropy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final artifact | 5.60865 | 4.69539 | 3.40840 | 0.91326 | 0.68176 |
| Artifact and truthful context | 5.60832 | 4.69506 | 2.79137 | 0.91326 | 0.68069 |
| Context and first operation | 5.60625 | 4.69295 | 1.35857 | 0.91331 | 0.68067 |
| Complete operation witnesses | 5.60252 | 4.68916 | 0.00000 | 0.91337 | 0.68049 |

Under complete witnesses, 4.68916 of the bank's 4.92203 nats of excess loss is
operation error. The reference has essentially zero operation uncertainty because
the input names every executed operation. Goal entropy remains 0.68049 nats, so
operation disclosure does not identify goals. Conditional-goal excess is 0.23287.
Larger excess in richer evidence tiers need not mean less information: reference
uncertainty drops much more than this reader's loss. The earlier support decoder
directly enforced public constraints; this diagnostic uses unrestricted forecasts.

The second table gives normalized area under each component's loss versus log
label budget, as predictive bank minus raw history or frozen latent state.
Negative values favor the bank. Brackets are paired 95% sixteen-lineage bootstrap
intervals, conditional on the saved fits after averaging two draws and five seeds
within each lineage. All per-budget, frequency, draw and seed results remain in
the complete regroup receipt; these intervals do not capture all training variation.

| Evidence supplied | Operation: bank minus history | Goal given operation: bank minus history | Operation: bank minus latent | Goal given operation: bank minus latent |
| --- | ---: | ---: | ---: | ---: |
| Final artifact | -0.00307 [-0.00323, -0.00293] | 0.00043 [0.00039, 0.00047] | -0.00065 [-0.00069, -0.00061] | 0.00008 [0.00008, 0.00009] |
| Artifact and truthful context | -0.00800 [-0.00829, -0.00772] | 0.00124 [0.00113, 0.00134] | -0.00086 [-0.00091, -0.00081] | 0.00011 [0.00010, 0.00012] |
| Context and first operation | 0.11443 [0.11090, 0.11822] | -0.02078 [-0.02237, -0.01910] | 0.00296 [0.00291, 0.00301] | -0.00065 [-0.00070, -0.00059] |
| Complete operation witnesses | 0.34008 [0.33074, 0.35005] | -0.06463 [-0.06953, -0.05947] | 0.00892 [0.00880, 0.00906] | -0.00191 [-0.00207, -0.00175] |

With complete witnesses the bank gains 0.06463 nats on conditional goals versus
raw history but loses 0.34008 on operations. The joint comparison therefore still
favors history. Against latent state, both component differences remain inside
the declared 0.02-nat practical band across the learning curve. A favorable
conditional component is neither the primary outcome nor causal correspondence.

All 1,289 deterministic output files match original, adjacent-source and complete
extracted-source replay. Source archives (611 producer and 614 checker files),
plans, environments, all input/output bindings and timings verify. The independent
scalar checker covers 145,920 frame forecasts, 14,592 native frame/law combinations,
912 public packets and all 10,240 score strata. Maximum score discrepancy is
2.67e-14 and chain-identity discrepancy 5.33e-15. A separate direct-index regroup
of original raw rows verifies all 288 budget contrasts, 72 learning areas and
64 means against both checker regroupings within 1.53e-14.

Thirty-three isolated controls passed before dispatch, including full synthetic
execution, unknown-only labels, reversed axes and deliberate score, entropy,
inherited-field, reference, reader, hash, row-roster and summary corruptions.
No producer component or grouping function is imported by the scalar checker.
Previously accepted fitted forecasts and native posterior masses remain inherited
inputs. Localization, compatibility, coverage and abstention fields are verified
unchanged, rather than independently re-derived by this diagnostic. Their prior
acceptance is not a new localization result here.

The reader archive remains 912 anonymous evidence packets. Forecasts, native
posteriors, components, scores and regroupings are separate scientific/evaluator
evidence. The result concerns a finite constructed method; the miniature remains
architecture untested beyond admitted rosters. No human intent, new causal access,
confirmation or conditional A3/C2/F2 admission follows. Pursuit remains active:
the prepared learned joint-factorization comparison tests whether dependence in
the frozen forecasts adds value, while retrospective updating is independent.

The 1,800 CPU-second card ceiling includes preparation, controls, failures,
execution, both replays and review. No resource ceiling or campaign clock changes.

[Admission contract](JOINT_UNCERTAINTY_ADMISSION_PROTOCOL.md).
[Independent review protocol](JOINT_UNCERTAINTY_REVIEW_PROTOCOL.md).
[Accepted reconstruction](../../../results/v19/G19-A-joint-uncertainty-1/FINAL_REVIEW.json).
[Full regroup](../../../results/v19/G19-A-joint-uncertainty-1/INDEPENDENT_REGROUP.json).
[Separate exports](../../../results/v19/G19-A-joint-uncertainty-1/EVIDENCE_ROLES.json).
