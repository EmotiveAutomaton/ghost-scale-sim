# V22 B2: Acquisition routes and devaluation

Can acquisition route predict a choice after need satisfaction? The two-route reader beats both single-route assumptions on post-devaluation choice loss in every tested condition. Under broken contingency its loss is 0.456 nats, versus 0.926 for the model-based-only reader and 1.212 for association-only. Yet observational aliases keep route probability exactly at one half. This is a constructed-mechanism prediction result, not proof that the historical route is always identifiable.

Both routes acquire values from the same 32 experience records. The model-based route tracks the difference between action and alternative outcomes; the associative route retains the action's association. Four pre-intervention choices inform the route/law posterior. Need satisfaction changes the subsequent choice probability for the model-based route only. Those dynamics are planted assumptions being tested for readable consequences.

The table reports future-choice log loss after need satisfaction. Rows distinguish intact contingency, degraded contingency and pre-intervention observational aliasing. Lower is better, in natural-log units. Direct rows average its two high-label fits within each coefficient world.

| Acquisition evidence | Both routes | Model-based only | Association only | Prior mixture | Direct, 8,192 labels |
|---|---:|---:|---:|---:|---:|
| Intact contingency | 0.6496 | 0.9179 | 1.2535 | 0.6664 | 0.6991 |
| Degraded contingency | 0.4557 | 0.9264 | 1.2118 | 0.6626 | 0.5129 |
| Observational alias | 0.6967 | 0.8579 | 1.4785 | 0.6963 | 0.7203 |

The gain over the better single-route reader is 0.2683 [0.2160, 0.3275], 0.4706 [0.3635, 0.5746] and 0.1612 [0.1173, 0.2108] nats, respectively. Under degraded contingency the additional gain over the prior mixture is 0.2069 [0.1564, 0.2569]. Under intact contingency it is only 0.0168 [-0.0057, 0.0371], and in the alias control it is -0.00039 [-0.00151, 0.00061]. Modeling both possibilities can protect prediction without identifying which route happened.

In the alias condition every saved pre-intervention route posterior equals 0.5 within numerical roundoff. Single-route readers assign exactly zero probability to the other route and have infinite route-label loss; their choice losses above remain finite. No pooled two-head score is used to claim the choice result.

Observed action rates before/after need satisfaction are 90.35%/22.75% for model-based makers with intact contingency, versus 94.43%/92.90% for associative makers. With degraded contingency the rates are 34.52%/26.61% versus 92.14%/93.38%. These groups are different sampled cases paired only by coefficient world; the simulator's causal devaluation rule is known, but this table is not a same-case intervention contrast. Model-based choice does not drop literally to zero: its stipulated post-devaluation probability is about 0.269.

The supplied-law oracle can lose to the law mixture in a finite sample: intact-condition oracle loss is 0.6583 versus 0.6496, while it improves degraded loss to 0.4441. Modal accuracy and proper loss also differ; uncertainty and ties are retained. High-label excess losses over exact inference are 0.0495, 0.0571 and 0.0236 nats when pooled within these conditions, but those are not matching route-stratum admission gates. Eleven of 24 high-label target strata have headroom, six parity and seven unresolved. No universal learned superiority or historical-route identification follows.

## Verification and scope

All 2,048 original case posteriors, 18,432 reader-case forecast bundles and 3,456 world/target score strata have independent numerical receipts. The complete 110-file scientific replay matches. This review rechecked every retained original/replay file hash, the plan, immutable source archive, environment declaration and receipt bindings; it reused the matching numerical work without rerunning science.

There are eight coefficient worlds, 256 development cases per world, two label budgets and two fits per budget. All four fits stopped on validation. Fits are averaged within worlds; intervals resample whole paired worlds 4,096 times. These descriptive 95% intervals do not adjust for the many comparisons and do not constitute confirmation. An original matched headroom interval must clear 0.02 nats; missing worlds or unresolved gates cannot support superiority. Report-stage intervals use a separate fixed bootstrap seed; immutable admission gates remain unchanged.

The independent checker establishes numerical correctness within this finite model, not adequacy of its mechanism assumptions. The parameter-law draw has only a small finite support; eight world identities need not be eight distinct law values. Miniature — architecture untested. No human conclusion follows.

[Public aggregates and evidence](../../../results/v22/b2/README.md) retain all predeclared target/control strata, fitting dispositions, matched headroom and metric reversals. Raw forecasts, training arrays, complete per-world details and original receipts remain retained under the provenance hashes; they are not duplicated into the public payload.
