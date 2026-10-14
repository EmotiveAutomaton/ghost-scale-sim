# V22 B1: Costly departures and set-point recovery

Do costly departures identify preferences more efficiently? Effort weighting narrows the constructed set-point intervals but lowers coverage relative to exact unit weighting. At 32 native records, coverage remains 98.26% and future-choice log loss improves by only 0.0090 nats. The direct reader already trails inside its training range and does not worsen after larger perturbations, so the proposed inside-parity/outside-failure account is not established. These are finite constructed-mechanism results.

The generator contains two to four internal variables, discrete set points, two drive geometries, uncertain choice precision and explicit action costs. The unit-weight exact likelihood already accounts for effort. Extra weighting of a departure from the historical modal action is a power-likelihood heuristic; it does not add observations.

The table reports native set-point intervals by number of records. Coverage is the fraction of true coordinate set points inside the nominal 90% marginal interval; width is measured in set-point units. Choice-loss difference is effort weighting minus exact unit weighting, averaged over native and larger current perturbations within each world.

| Records | Exact coverage / width | Effort coverage / width | Choice-loss difference, nats |
|---:|---:|---:|---:|
| 2 | 99.74% / 1.944 | 98.70% / 1.890 | +0.0092 [-0.0042, 0.0224] |
| 8 | 98.91% / 1.360 | 96.79% / 1.131 | +0.0237 [-0.0054, 0.0525] |
| 32 | 98.96% / 0.650 | 98.26% / 0.529 | -0.0090 [-0.0165, -0.0024] |

All three coverage reductions have paired intervals below zero, but all observed coverage levels remain above the nominal 90%. The discrete intervals are conservative. Narrower intervals therefore support a qualified fixed-count efficiency observation, not a demonstrated equivalent-coverage reduction in required episodes. The 32-record forecast gain is below the declared 0.02-nat practical margin; it also improves Brier by 0.00394 and accuracy by 1.04 percentage points. Earlier record counts do not establish a forecast gain.

Choice consistency is the largest true action probability, not repeated empirical agreement. Its within-world correlation with Euclidean distance is 0.167 [0.065, 0.275] in two dimensions, 0.254 [0.075, 0.450] in three, and 0.126 [-0.016, 0.257] in four. Random menus, geometry and precision vary too, so these descriptive correlations do not establish a universal causal distance law.

Every reader's change under larger current perturbations is shown below. Rows name readers; columns give native loss, shifted loss and their paired difference in nats. Histories retain the native perturbation range. Direct rows average both fits within worlds.

| Reader | Native | Larger perturbation | Shift minus native |
|---|---:|---:|---:|
| Exact law marginal | 0.8834 | 0.7623 | -0.1211 |
| Effort weighted | 0.8998 | 0.7619 | -0.1380 |
| Cost ignorant | 0.8998 | 0.7828 | -0.1170 |
| Frequency | 1.1614 | 1.2411 | +0.0797 |
| Supplied law | 0.8206 | 0.6925 | -0.1281 |
| Supplied law and set point | 0.7235 | 0.6297 | -0.0938 |
| Direct, 2,048 labels | 1.1155 | 1.1094 | -0.0061 |
| Direct, 8,192 labels | 1.1058 | 1.1013 | -0.0044 |

The direct reader is already worse in range; neither direct change establishes deterioration. The original parity-then-failure prediction is therefore not observed. Full intervals and every dimension/history/control change are in the supplement. Learned superiority is restricted by the original matched gates: only 26 of 72 high-label strata admit headroom, three show parity and 43 are unresolved.

Zero costs make weighting identical to exact inference. The independent-effort and nonidentifying-menu controls are retained at every record length; alias menus provide no set-point information. All reader/target proper scores and accuracy/confidence reversals remain available. No equivalent-coverage learning curve, new population or follow-up allocation is inferred.

The arm named independent effort randomizes costs independently of the action effects, but those costs still enter the maker's choice utility. It is therefore **not** the protocol's fully choice-independent costly-label null. That specific control is unexecuted as specified and receives an explicit unresolved disposition. Numerical validity of the saved forecasts does not repair this semantic gap; no claim that irrelevant costly labels are harmless is accepted, and no additional run is authorized by closure.

## Verification and scope

All 2,048 original case posteriors, 20,480 reader-case forecast bundles and 11,520 world/target score strata have independent numerical receipts. The complete 118-file scientific replay matches. This review rechecked every retained original/replay file hash, the plan, immutable source archive, environment declaration and receipt bindings; it reused the matching numerical work without rerunning science.

There are eight coefficient worlds, 256 development cases per world, two label budgets and two fits per budget. All four fits stopped on validation. Fits are averaged within worlds; intervals resample whole paired worlds 4,096 times. These descriptive 95% intervals do not adjust for the many comparisons and do not constitute confirmation. An original matched headroom interval must clear 0.02 nats; missing worlds or unresolved gates cannot support superiority. Report-stage intervals use a separate fixed bootstrap seed; immutable admission gates remain unchanged.

The independent checker establishes numerical correctness within this finite model, not adequacy of its mechanism assumptions. The parameter-law draw has only a small finite support; eight world identities need not be eight distinct law values. Miniature — architecture untested. No human conclusion follows.

[Public aggregates and evidence](../../../results/v22/b1/README.md) retain all predeclared target/control strata, fitting dispositions, matched headroom and metric reversals. Raw forecasts, training arrays, complete per-world details and original receipts remain retained under the provenance hashes; they are not duplicated into the public payload.
