# V22 A2: Steering signatures and selective uptake

Does recognizing steering reduce content uptake while preserving technique? With a correctly mapped signature, the constructed full-library reader changes its content belief by 0.0167 on average versus 0.1606 for the missing-process reader, while technique-production probability is 89.94% versus 89.46%. But the content gap persists without the signature, so the proposed vanishing-gap control fails. This supports a qualified library-and-signature mechanism, not a clean signature-only explanation.

Artifacts contain a claim, a noisy own reading, eight selection-signature bits and eight operation witnesses. Source goal, technique and property are separate binary targets; the finite reliability law is unknown. A reader with the steering process removed is deliberately misspecified and cannot assign positive probability to steering. Its infinite steering-label loss is retained but is not the content or technique score.

The table concerns actual steering sources. Content uptake is absolute change in property probability from the reader's own reading; technique is probability assigned to producing the true operation. Correctly mapped full-library and missing-library readers see the same observations.

| Signature / mapping | Full content change | Missing content change | Full technique | Missing technique |
|---|---:|---:|---:|---:|
| Absent / ordinary | 0.1105 | 0.2407 | 89.59% | 89.58% |
| Absent / swapped | 0.1103 | 0.2361 | 89.05% | 89.03% |
| Present / ordinary | 0.0167 | 0.1606 | 89.94% | 89.46% |
| Present / swapped, map known | 0.0186 | 0.1562 | 88.91% | 88.33% |

With the ordinary signature, missing minus full absolute uptake is 0.1439; the full reader's mean steering posterior is 0.8993. Without the signature the gap remains 0.1302 and steering posterior is 0.5149. Signature removal removes its information but does not remove the readers' different hypothesis spaces. Content/own-reading disagreement and the prior possibility of steering still matter. Thus the proposed disappearance of the content gap is false in this implementation.

Ignoring the known swapped map reverses goal interpretation: mean steering posterior falls to 0.1133 and absolute content change rises to 0.2097, while technique remains 88.91%. This is a useful separation of the operation channel from goal classification. The full reader's content/technique contrast is not evidence that a missing-process model was an equally informed optimal rival.

All informing-source cells, signed changes, three target losses, Brier scores, confidence errors and fit-specific forecasts are retained. Eight of 24 high-label target strata show parity, only one admits headroom and 15 remain unresolved. A native no-signature informer cell illustrates a reversal: the direct reader's content loss is 0.0223 higher despite 1.26 percentage points higher accuracy. No blanket learned-reader advantage or real persuasion claim is accepted.

## Verification and scope

All 2,048 original case posteriors, 16,384 reader-case forecast bundles and 3,072 world/target score strata have independent numerical receipts. The complete 102-file scientific replay matches. This review rechecked every retained original/replay file hash, the plan, immutable source archive, environment declaration and receipt bindings; it reused the matching numerical work without rerunning science.

There are eight coefficient worlds, 256 development cases per world, two label budgets and two fits per budget. All four fits stopped on validation. Fits are averaged within worlds; intervals resample whole paired worlds 4,096 times. These descriptive 95% intervals do not adjust for the many comparisons and do not constitute confirmation. An original matched headroom interval must clear 0.02 nats; missing worlds or unresolved gates cannot support superiority. Report-stage intervals use a separate fixed bootstrap seed; immutable admission gates remain unchanged.

The independent checker establishes numerical correctness within this finite model, not adequacy of its mechanism assumptions. The parameter-law draw has only a small finite support; eight world identities need not be eight distinct law values. Miniature — architecture untested. No human conclusion follows.

[Public aggregates and evidence](../../../results/v22/a2/README.md) retain all predeclared target/control strata, fitting dispositions, matched headroom and metric reversals. Raw forecasts, training arrays, complete per-world details and original receipts remain retained under the provenance hashes; they are not duplicated into the public payload.
