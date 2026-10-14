# V22 C1: Joint goal-tree inference

Does joint inference improve recovery across a goal hierarchy? Joint reading improves internal-node probability forecasts over the one-direction ablations, but bottom-up inference already matches it at the root. Removing distant evidence can hurt more than removing immediate-child evidence, and root accuracy is not monotone in every coverage series. These are finite constructed-tree results: marginal node scores do not establish a joint-prediction advantage or mathematical smoothness.

Trees have depth four and branching two or three, with binary states, three finite transition laws and noisy observed leaves. Library coverage exposes 25%, 50% or 75% of leaf evidence on average. The law posterior is weighted by each law's evidence likelihood. Both one-direction ablations receive those same full-evidence law weights, an explicit information privilege.

All scores are marginal node scores. The table shows joint inference at the root and its log-loss benefit over bottom-up inference at depth three. Rows name branching and leaf coverage. Root accuracy is correct modal classification; the final column is bottom-up minus joint loss, in nats, with a descriptive interval.

| Branching | Coverage | Root loss | Root accuracy | Depth-three gain over bottom-up |
|---:|---:|---:|---:|---:|
| 2 | 25% | 0.5538 | 71.80% | 0.0622 [0.0218, 0.0956] |
| 2 | 50% | 0.4742 | 76.45% | 0.0738 [0.0214, 0.1224] |
| 2 | 75% | 0.4293 | 77.08% | 0.0657 [0.0240, 0.1040] |
| 3 | 25% | 0.3591 | 83.72% | 0.1429 [0.0602, 0.2119] |
| 3 | 50% | 0.2977 | 82.27% | 0.1177 [0.0502, 0.1772] |
| 3 | 75% | 0.2949 | 82.44% | 0.0797 [0.0364, 0.1135] |

At depth three, joint gains over top-down at 50% and 75% coverage are 0.2498/0.3062 nats for binary branching and 0.3660/0.4029 for ternary branching. At the root, bottom-up and joint inference are identical because all available evidence arrives from descendants. Thus a blanket claim that joint beats both directions at every node is false. The complete five-depth results, including leaf ties and accuracy reversals, are retained.

Immediate-child evidence is useful, but not universally dominant. At ternary branching and 50% coverage, removing adjacent evidence increases focus-node loss by 0.0845 nats; removing distant leaves increases it by 0.1677. Their paired difference is -0.0832 [-0.1656, 0.0051], so the observed ordering is not decisive at this interval. Every adjacent-minus-distant interval crosses zero across the six settings. Many distant observations may outweigh a few noisy neighbors. Conditional Markov separation does not order these unequal noisy evidence sets, and this design does not directly test a separately observed root versus parent.

Root log loss improves across all three coverage points, but ternary root accuracy falls from 83.72% to 82.27% before reaching 82.44%. Full Brier also slightly worsens from 0.2027 to 0.2040 between 50% and 75% in that branch. Three points with different sampled cases cannot establish a mathematically smooth degradation curve.

All four direct fits stopped on validation. Fifteen of 30 high-label target/depth strata have exact headroom and 15 remain unresolved. For example, binary root at 50% has 0.0221 direct excess loss but 0.44 percentage points higher direct accuracy. No joint-distribution log loss, universal architectural superiority or human comprehension claim is made.

## Verification and scope

All 2,048 original case posteriors, 16,384 reader-case forecast bundles and 3,840 world/target score strata have independent numerical receipts. The complete 102-file scientific replay matches. This review rechecked every retained original/replay file hash, the plan, immutable source archive, environment declaration and receipt bindings; it reused the matching numerical work without rerunning science.

There are eight coefficient worlds, 256 development cases per world, two label budgets and two fits per budget. All four fits stopped on validation. Fits are averaged within worlds; intervals resample whole paired worlds 4,096 times. These descriptive 95% intervals do not adjust for the many comparisons and do not constitute confirmation. An original matched headroom interval must clear 0.02 nats; missing worlds or unresolved gates cannot support superiority. Report-stage intervals use a separate fixed bootstrap seed; immutable admission gates remain unchanged.

The independent checker establishes numerical correctness within this finite model, not adequacy of its mechanism assumptions. The parameter-law draw has only a small finite support; eight world identities need not be eight distinct law values. Miniature — architecture untested. No human conclusion follows.

[Public aggregates and evidence](../../../results/v22/c1/README.md) retain all predeclared target/control strata, fitting dispositions, matched headroom and metric reversals. Raw forecasts, training arrays, complete per-world details and original receipts remain retained under the provenance hashes; they are not duplicated into the public payload.
