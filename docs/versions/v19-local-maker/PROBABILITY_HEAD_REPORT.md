# Probability-valid retained-feature decoding — 21 September 2026

Does probability-valid decoding preserve the earlier bank advantage? With unchanged data and representations, fixed softmax training reduces the old-world bank-minus-history loss difference to -0.00316 nats, with a 95% paired lineage interval of [-0.02721, 0.02473]. In the local world, the bank advantage is below the declared 0.02-nat margin for artifacts alone and added context; sparse and complete operation witnesses favor direct history by 0.13339 and 0.39751 nats. Residual optimization error remains unresolved. This is an exploratory constructed-method diagnostic, miniature — architecture untested; it establishes neither general access failure nor process correspondence or human intent.

The comparison freezes the original old-outcome models and all random feature
transformations. Four arms receive the same new training labels at each nested
budget. Each target category has 129 fitted coefficients; representation geometry
differs. Softmax heads start at zero and take exactly 200 full-batch gradient steps
of size 0.1, with non-intercept L2 penalty 0.01 and no checkpoint selection. The
old scout has budgets 32/128 and eight development lineages; local readouts use
32/128/512/2048 and sixteen development lineages. Both retain two feature seeds
and two training draws. Those are not five independently trained initializations.

Logarithmic loss measures forecast error in nats; lower is better. The normalized
area averages loss differences across logarithmic label budgets. Negative values
favor the bank. Intervals resample paired development lineages, keeping both draws,
feature seeds and repeated queries within each lineage. They do not estimate a
universal training-population interval or confirm a selected method. Equal-class
population results average the four old classes equally, including the history-
inert class; none is dropped or replaced by the favorable class.

The first table averages the three local-goal and three operation questions.
Rows specify visible evidence; columns compare the bank with each named rival.
Each entry gives the normalized loss difference and its 95% conditional interval.

| Evidence | Bank minus direct history | Bank minus frozen latent | Bank minus matched-label mean |
|---|---:|---:|---:|
| Artifact alone | -0.00163 [-0.00273, -0.00049] | 0.00010 [-0.00025, 0.00045] | -0.00510 [-0.00778, -0.00260] |
| Artifact and context | -0.00852 [-0.01134, -0.00562] | -0.00020 [-0.00040, 0.00001] | -0.00522 [-0.00788, -0.00274] |
| Sparse witnesses | 0.13339 [0.12748, 0.13938] | 0.00379 [0.00345, 0.00414] | -0.00786 [-0.01048, -0.00543] |
| Complete witnesses | 0.39751 [0.38827, 0.40695] | 0.01218 [0.01166, 0.01270] | -0.01169 [-0.01418, -0.00935] |

The artifact-only and contextual bank advantages over direct history are below
the frozen practical margin: their full conditional intervals are within
[-0.02, 0.02] nats. This supports practical equivalence for those particular
paired contrasts conditional on these fits, not general architectural equivalence.
Sparse and complete witnesses favor direct history beyond that margin. The
complete-witness operation task includes copying supplied operations. A prediction
score does not evaluate a historically correct joint goal/process account.

The next two tables keep the target roles separate. Rows retain the same evidence
conditions; columns retain the same three rivals; entries are normalized differences
and conditional 95% lineage intervals, in nats.

Local-goal questions:

| Evidence | Bank minus direct history | Bank minus frozen latent | Bank minus matched-label mean |
|---|---:|---:|---:|
| Artifact alone | -0.00180 [-0.00275, -0.00077] | -0.00029 [-0.00061, 0.00005] | -0.00018 [-0.00022, -0.00014] |
| Artifact and context | -0.00950 [-0.01291, -0.00613] | -0.00039 [-0.00063, -0.00016] | -0.00029 [-0.00036, -0.00023] |
| Sparse witnesses | 0.08729 [0.07963, 0.09509] | 0.00229 [0.00183, 0.00275] | -0.00269 [-0.00288, -0.00252] |
| Complete witnesses | 0.28523 [0.27237, 0.29835] | 0.00832 [0.00748, 0.00917] | -0.00634 [-0.00684, -0.00585] |

Operation questions:

| Evidence | Bank minus direct history | Bank minus frozen latent | Bank minus matched-label mean |
|---|---:|---:|---:|
| Artifact alone | -0.00146 [-0.00313, 0.00021] | 0.00049 [-0.00000, 0.00099] | -0.01002 [-0.01537, -0.00501] |
| Artifact and context | -0.00753 [-0.01030, -0.00473] | -0.00001 [-0.00021, 0.00020] | -0.01016 [-0.01550, -0.00516] |
| Sparse witnesses | 0.17949 [0.17466, 0.18450] | 0.00528 [0.00497, 0.00558] | -0.01303 [-0.01827, -0.00811] |
| Complete witnesses | 0.50979 [0.50266, 0.51726] | 0.01603 [0.01569, 0.01640] | -0.01703 [-0.02215, -0.01226] |

In the old-world equal-class population, the bank still beats the frozen latent
readout by 0.04386 nats [0.03714, 0.05157] and the matched-label mean by 0.08917
[0.07814, 0.10247]. Against direct history its interval spans effects larger than
the practical margin in both directions: that contrast is inconclusive, not an
equivalence result. At 128 labels, losses are bank 2.29293, direct history 2.24078,
frozen latent 2.33898, matched mean 2.36174 and privileged reference 0.78491.
The remaining reference gap is large. Every old-class contrast and budget remains
in the independent review, including a reversal in class zero (softmax with
endogenous choices off) and the history-inert satisficing class.

The previous ridge results remain intact. On the old equal-class log-budget
average, softmax improves direct history by 3.16807 nats and the bank by 1.29143,
largely removing that earlier relative bank advantage. The intervention changes
the loss, optimization and regularization as well as probability validity; it does
not isolate clipping as the cause. Ridge's fixed sum-of-squares penalty and this
mean-cross-entropy penalty also have different sample-size scaling. Local artifact-
only bank loss at 2048 labels is now 1.35006, versus matched mean 1.34718 and exact
reference 1.31137; its old ridge crossing of the reference-plus-margin does not
transfer to this optimizer/objective.

All 288 fits reduce their recorded training objective, but final gradient norms
range from 0.01869 to 0.26300. No convergence threshold was met or required by the
fixed schedule. These gradients leave optimization as a live rival; this result
cannot establish that the saved features lack usable information. A separately
frozen bounded convergent-solver comparison will keep the objective and penalty
unchanged, rather than tune them against these development scores.

There is a preserved scoring deviation: fitted softmax probabilities do not receive
the protocol's 1e-6 category floor, while ridge, means and references do. The minimum
fitted probability is 0.00125131. A separately labelled post-result sensitivity
calculation applies that floor to saved forecasts; the largest absolute change
to one history/query loss is 0.00025710 nats. Original scores stay unchanged. This
small bound does not explain the large witnessed-history gaps, but the deviation
remains in the evidence and near-zero differences should not be overinterpreted.

**Disposition:** conditional practical equivalence for bank/direct local artifact
and context areas; reversal with witnessed operations; inconclusive old-world
bank/direct contrast; inadequate evidence of a generally capable readout.
**Pursuit:** one bounded optimization comparison for the identified residual-gradient
cause, plus independent recursive-state work. Neither needs a bank win or tiny model.

**Validation and limits:** all 40,704 score rows reconstruct exactly from saved
forecasts. Final objectives/gradients and coefficient-derived forecasts reproduce
independently; normalized probabilities, nested means, equal-class weighting and
all source/input hashes pass. Adjacent and separately extracted-source complete
replays match every deterministic output. Full models, forecasts and execution
measurements stay retained; the reproducible input capsule is explicitly evaluator
material. Blind reader inputs refer to unchanged allowlisted parent exports.
Training local-goal labels are evaluator supervision supplied equally; exact-bank
and exact-reference arms have additional law access. No test/confirmation lineage
or tiny-model setting was consumed. Neither numerical verification nor coefficient
lineage variation establishes general architecture validity or human intent.

[Scientific evidence](../../../results/v19/G19-A-probability-head-1/EXPORT_MANIFEST.json).
