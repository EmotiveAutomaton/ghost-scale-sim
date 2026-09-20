# V18.4 paid source audits and truth decisions

Does paying to investigate copied sources improve truth decisions? Audit-only decision policies buy nothing throughout this finite comparison. Two-step planning sometimes combines audits with evidence: in one calibrated condition its net utility is 0.86359 versus 0.86341 for evidence alone, a small advantage that reverses when audit reliability is overstated. A graph-information policy buys almost two audits yet reaches only 0.75898 net utility there. This is a descriptive constructed-method result: learning source structure and improving a truth decision are different objectives, miniature — architecture untested.

Twenty fresh parameter draws cross two initial shared-error rates, five actual/
assumed audit channels, three decision stakes and two audit prices: 1,200 exact
models. Every six-report string and every terminal path of at most two purchases
is enumerated. Reports, graph conditions and policy paths are not independent
observations. Initial dependence includes four supplied graphs with equal prior
mass; the two-audit menu leaves the alternating and fully independent graphs
aliased. Purchased evidence and audit errors are conditionally independent.

The table shows the condition with no shared initial flip, true and assumed audit
accuracy 95%, unit decision stake, audit price 0.025 and evidence price 0.1.
Rows name inquiry policies; columns give truth logarithmic loss in nats (lower
better), correct-choice probability, net utility after purchase costs, and expected
audit/evidence counts. Values average the twenty parameter draws and all exact
report paths under equal graph mass. A probability score and a decision score
need not favor the same policy.

| Policy | Truth loss | Correct choice | Net utility | Audits | Evidence |
|---|---:|---:|---:|---:|---:|
| Stop | 0.42211 | 0.80752 | 0.80752 | 0.00000 | 0.00000 |
| Assume independence; stop | 0.62975 | 0.80752 | 0.80752 | 0.00000 | 0.00000 |
| Fixed audit then evidence | 0.22416 | 0.91175 | 0.78675 | 1.00000 | 1.00000 |
| Same trace; assume independence | 0.41125 | 0.88227 | 0.75727 | 1.00000 | 1.00000 |
| Evidence only | 0.27669 | 0.91372 | 0.86341 | 0.00000 | 0.50310 |
| Audits only | 0.42211 | 0.80752 | 0.80752 | 0.00000 | 0.00000 |
| One-step decision planning | 0.31169 | 0.89721 | 0.86176 | 0.00000 | 0.35450 |
| Two-step decision planning | 0.27870 | 0.91301 | 0.86359 | 0.01726 | 0.48993 |
| Graph information per price | 0.39764 | 0.81060 | 0.75898 | 1.97832 | 0.02168 |

The two-step benefit is 0.000172 utility units in this slice, not a significance
claim or a broad audit advantage. Its mean audit count is 0.01726; most decisions
use no audit. Its truth loss is slightly worse than evidence-only loss, retaining
the difference between optimizing a binary action and a proper probability score.
Audit-only and one-step decision policies choose no audits in every model. Across
the 60 parameter settings averaged over draws, two-step planning buys any audits
in six; that count includes calibration stress tests, not six positive findings.

The next table fixes no initial shared error, unit stake and the cheaper audit.
Rows change the actual audit accuracy while the reader assumes 95%; columns
compare two-step decision planning with evidence-only planning and the graph
information heuristic. Values are net utility after all purchase costs.

| Actual / assumed audit accuracy | Two-step | Evidence only | Graph information |
|---|---:|---:|---:|
| 0.95 / 0.95 | 0.86359 | 0.86341 | 0.75898 |
| 0.75 / 0.95 | 0.86306 | 0.86341 | 0.75885 |
| 0.50 / 0.95 | 0.86240 | 0.86341 | 0.75869 |

With a 20% initial shared flip, calibrated 95% audits and the same prices/stake,
two-step and evidence-only policies coincide: net utility 0.79350, expected
correct choice 0.88959 and mean evidence count 0.96086, with no audits. The graph
heuristic buys 1.99595 audits and achieves net utility 0.63489. It reduces graph
squared-probability error from 0.60767 at stopping to 0.27093; this statistic sums
four squared differences from the actual graph indicator, with zero best. That
structural improvement does not establish decision value. At high audit prices
and low stakes, the heuristic can spend more than its expected decision reward.

Common traces isolate the effect of the initial dependence model. The table uses
the first table's calibrated condition and the same fixed audit-then-evidence
trace for both readers. Rows identify the actual source graph; columns report
truth logarithmic loss and probability of assigning over 95% to the wrong truth.
Graph-conditional values average twenty draws and exact paths within that graph.

| Actual graph | Dependence-model loss | Naive loss | Dependence false confidence | Naive false confidence |
|---|---:|---:|---:|---:|
| One root | 0.28643 | 0.85127 | 0.02119 | 0.12412 |
| Two contiguous roots | 0.23300 | 0.33055 | 0.00576 | 0.03095 |
| Two alternating roots | 0.22767 | 0.33055 | 0.00935 | 0.03095 |
| Six independent roots | 0.14955 | 0.13264 | 0.00388 | 0.00521 |

Assuming independence hurts most under a single copied root; it has lower loss
on the actual independent graph. Thus the mixture's average protection is not
uniform conditional superiority. All remaining stakes, prices, channels and
graph-conditional scores are retained in the public rollups and raw policy paths.

Independent verification reconstructed 4,320 means and replayed eight complete
models exactly from extracted source. Every unit's likelihood table was checked
by independent latent-root enumeration and every terminal posterior/mass by scalar
path products; budget, coverage, costs and conditional graph scores reconstruct.
Brute-force contingent policies check decision recursion; a separate joint-only
fixture proves it can detect complementary inquiries. Source/input/raw hashes and
portable reassembly match. The scoped combined suite passes 23 controls, including
seven S1 controls. V15 C11/M01 remain failed instruments. This is not a test of
larger policy trees, graph misspecification, shared audit/evidence errors, native
construction, learned uptake, human source trust or general provenance recovery.

Science charged 339.718750 CPU seconds, verification 17.593750 including its
parent and child: 357.312500 total. Combined wall time was 366.156719 seconds,
6.10 minutes. Useful follow-ons should test whether the audit's limited decision
value changes under shared purchased-channel error, graph omission or a third
audit that separates the retained graph alias; repeating this roster is unnecessary.

[Portable evidence](../../results/v18/exploratory-loop/S1-paid-noisy-audits-1/SCIENTIFIC_MANIFEST.json).
