# Support and composition holdout: exact propagation helps; sampling can reverse it

Does learned transition composition help on queries absent from training when every prediction has the same final smoothing? Exact learned propagation lowers withheld-composition loss by 0.16115 nats versus direct lookup; sixteen sampled rollouts instead raise it by 0.23095 nats. When all true primitive transitions were observed, both improve, but that diagnostic uses evaluator truth. Independent reconstruction and both complete replays verify this exploratory constructed-method result. Declared mechanics and proposed operations do not establish historical process correspondence or human intent.

Logarithmic loss measures the probability assigned to the correct final artifact,
in nats; lower is better. Every final forecast receives the same uniform mixture
of 1/32. Transition and direct-table unit pseudocounts are still different
factorizations. Equal final smoothing does not capacity-match those estimators.

The fixed holdout removes accept-tool followed by repair-evidence from the first
two operations of each training path. It retains 1,991 and 1,983 of the two
original 2,048-path selections. It therefore changes both support and sample size.
Eight development coefficient lineages supply 640 full forward queries each;
training draws are averaged within a paired lineage before 10,000 bootstrap
resamples (seed 190501). Intervals condition on those two training draws and this
generator. Their draw means remain separate; they do not measure universal
training uncertainty. No reserved confirmation lineage is used.

The first table gives the comparison after the declared holdout. Rows cross the
five predeclared query populations with each method; negative differences favor
that method over direct lookup. Columns report the difference, paired-lineage
95% interval, and the two training-draw averages in the same units.

| Query population | Method | Loss minus direct lookup, nats | 95% conditional interval | Training-draw means, nats |
|---|---|---:|---:|---:|
| All queries | Nearest-query retrieval | -0.01160 | [-0.01248, -0.01054] | -0.01140, -0.01179 |
| All queries | Exact learned propagation | -0.05562 | [-0.07342, -0.03702] | -0.05260, -0.05863 |
| All queries | One sampled rollout | +2.02924 | [+1.99437, +2.06444] | +2.10189, +1.95658 |
| All queries | Four sampled rollouts | +0.53498 | [+0.51463, +0.55438] | +0.56384, +0.50611 |
| All queries | Sixteen sampled rollouts | +0.02989 | [+0.01303, +0.04707] | +0.04777, +0.01200 |
| All queries | Exact supplied mechanics | -1.00052 | [-1.01431, -0.98392] | -0.99677, -1.00426 |
| Full query seen | Nearest-query retrieval | +0.00000 | [+0.00000, +0.00000] | +0.00000, +0.00000 |
| Full query seen | Exact learned propagation | -0.03274 | [-0.05038, -0.01477] | -0.02775, -0.03773 |
| Full query seen | One sampled rollout | +2.04012 | [+2.00413, +2.07622] | +2.12553, +1.95472 |
| Full query seen | Four sampled rollouts | +0.48601 | [+0.46470, +0.50627] | +0.52108, +0.45094 |
| Full query seen | Sixteen sampled rollouts | +0.02664 | [+0.00934, +0.04423] | +0.05307, +0.00021 |
| Full query seen | Exact supplied mechanics | -0.91019 | [-0.92189, -0.89638] | -0.90563, -0.91476 |
| Full query unseen | Nearest-query retrieval | -0.14588 | [-0.15250, -0.13811] | -0.14262, -0.14914 |
| Full query unseen | Exact learned propagation | -0.32159 | [-0.33363, -0.30732] | -0.34027, -0.30292 |
| Full query unseen | One sampled rollout | +1.90254 | [+1.88475, +1.92135] | +1.82945, +1.97563 |
| Full query unseen | Four sampled rollouts | +1.10616 | [+1.07905, +1.13979] | +1.06332, +1.14899 |
| Full query unseen | Sixteen sampled rollouts | +0.06813 | [+0.05477, +0.08356] | -0.01424, +0.15050 |
| Full query unseen | Exact supplied mechanics | -2.05172 | [-2.05172, -2.05172] | -2.05172, -2.05172 |
| Withheld operation prefix | Nearest-query retrieval | -0.02193 | [-0.02421, -0.01988] | -0.00470, -0.03916 |
| Withheld operation prefix | Exact learned propagation | -0.16115 | [-0.16744, -0.15428] | -0.15581, -0.16648 |
| Withheld operation prefix | One sampled rollout | +2.00492 | [+1.99394, +2.01494] | +2.72289, +1.28695 |
| Withheld operation prefix | Four sampled rollouts | +1.38697 | [+1.35729, +1.41928] | +2.32238, +0.45157 |
| Withheld operation prefix | Sixteen sampled rollouts | +0.23095 | [+0.21168, +0.25054] | +0.02810, +0.43380 |
| Withheld operation prefix | Exact supplied mechanics | -2.05172 | [-2.05172, -2.05172] | -2.05172, -2.05172 |
| Withheld prefix; all true transitions seen | Nearest-query retrieval | -0.12144 | [-0.12219, -0.12072] | -0.11385, -0.12903 |
| Withheld prefix; all true transitions seen | Exact learned propagation | -0.55587 | [-0.56023, -0.55171] | -0.55219, -0.55955 |
| Withheld prefix; all true transitions seen | One sampled rollout | +1.03291 | [+1.00051, +1.06695] | +0.57169, +1.49413 |
| Withheld prefix; all true transitions seen | Four sampled rollouts | +0.30581 | [+0.28735, +0.32180] | +0.41739, +0.19422 |
| Withheld prefix; all true transitions seen | Sixteen sampled rollouts | -0.37788 | [-0.39453, -0.36282] | -0.32319, -0.43258 |
| Withheld prefix; all true transitions seen | Exact supplied mechanics | -2.05172 | [-2.05172, -2.05172] | -2.05172, -2.05172 |

The exact learned propagation gain on the withheld prefix exceeds the frozen
0.02-nat practical margin throughout its conditional interval. The sixteen-rollout
reversal also exceeds that margin. On all queries exact propagation improves
0.05562 nats, whereas sixteen rollouts worsen 0.02989 nats. Retrieval improves
only 0.01160 nats overall. The support-conditioned subset is a diagnostic: it
consults the true intermediate path and cannot be used as a reader selection rule.

The next table retains original-training comparisons with the same row and column
definitions. On the prefix before its removal, learned propagation loses to
direct lookup by 0.08900 nats. Changing the support changes the rival's advantage;
this does not isolate a universal benefit from additional computation.

| Query population | Method | Loss minus direct lookup, nats | 95% conditional interval | Training-draw means, nats |
|---|---|---:|---:|---:|
| All queries | Nearest-query retrieval | -0.01100 | [-0.01191, -0.00994] | -0.01133, -0.01067 |
| All queries | Exact learned propagation | -0.05726 | [-0.07565, -0.03811] | -0.05427, -0.06025 |
| All queries | One sampled rollout | +2.01289 | [+1.97767, +2.04896] | +2.06483, +1.96095 |
| All queries | Four sampled rollouts | +0.50963 | [+0.49016, +0.52819] | +0.53251, +0.48675 |
| All queries | Sixteen sampled rollouts | +0.01100 | [-0.00532, +0.02770] | +0.03525, -0.01324 |
| All queries | Exact supplied mechanics | -0.97095 | [-0.98643, -0.95267] | -0.96832, -0.97357 |
| Full query seen | Nearest-query retrieval | +0.00000 | [+0.00000, +0.00000] | +0.00000, +0.00000 |
| Full query seen | Exact learned propagation | -0.03753 | [-0.05518, -0.01948] | -0.03245, -0.04262 |
| Full query seen | One sampled rollout | +2.02068 | [+1.98470, +2.05709] | +2.10304, +1.93832 |
| Full query seen | Four sampled rollouts | +0.48938 | [+0.46803, +0.50980] | +0.54246, +0.43630 |
| Full query seen | Sixteen sampled rollouts | +0.01461 | [-0.00241, +0.03185] | +0.04218, -0.01296 |
| Full query seen | Exact supplied mechanics | -0.90995 | [-0.92157, -0.89624] | -0.90699, -0.91292 |
| Full query unseen | Nearest-query retrieval | -0.20510 | [-0.20749, -0.20252] | -0.21058, -0.19962 |
| Full query unseen | Exact learned propagation | -0.40756 | [-0.41331, -0.40045] | -0.44156, -0.37357 |
| Full query unseen | One sampled rollout | +1.87432 | [+1.85741, +1.89264] | +1.38303, +2.36560 |
| Full query unseen | Four sampled rollouts | +0.86871 | [+0.85817, +0.88191] | +0.35553, +1.38188 |
| Full query unseen | Sixteen sampled rollouts | -0.05509 | [-0.05810, -0.05251] | -0.08739, -0.02280 |
| Full query unseen | Exact supplied mechanics | -2.05172 | [-2.05172, -2.05172] | -2.05172, -2.05172 |
| Withheld operation prefix | Nearest-query retrieval | -0.00698 | [-0.00796, -0.00588] | -0.00432, -0.00964 |
| Withheld operation prefix | Exact learned propagation | +0.08900 | [+0.08350, +0.09520] | +0.09763, +0.08037 |
| Withheld operation prefix | One sampled rollout | +2.49345 | [+2.48043, +2.50704] | +2.80698, +2.17992 |
| Withheld operation prefix | Four sampled rollouts | +0.84190 | [+0.83360, +0.85004] | +1.54920, +0.13459 |
| Withheld operation prefix | Sixteen sampled rollouts | +0.06092 | [+0.05636, +0.06630] | +0.03137, +0.09046 |
| Withheld operation prefix | Exact supplied mechanics | -0.94685 | [-0.95990, -0.93151] | -0.98920, -0.90450 |
| Withheld prefix; all true transitions seen | Nearest-query retrieval | -0.00293 | [-0.00329, -0.00251] | +0.00000, -0.00586 |
| Withheld prefix; all true transitions seen | Exact learned propagation | +0.09276 | [+0.08745, +0.09867] | +0.10191, +0.08361 |
| Withheld prefix; all true transitions seen | One sampled rollout | +2.48610 | [+2.47297, +2.49991] | +2.82445, +2.14775 |
| Withheld prefix; all true transitions seen | Four sampled rollouts | +0.80097 | [+0.79203, +0.80997] | +1.55054, +0.05139 |
| Withheld prefix; all true transitions seen | Sixteen sampled rollouts | +0.07204 | [+0.06829, +0.07630] | +0.03253, +0.11155 |
| Withheld prefix; all true transitions seen | Exact supplied mechanics | -0.91552 | [-0.92522, -0.90391] | -0.95512, -0.87592 |

Population mass is the native probability of a stratum, averaged over paired
lineages and draws. Query counts are distinct full queries in each training draw;
they are not independent worlds. The same queries can appear in nested strata.

| Training set | Query population | Mean native population mass | Query counts in the two draws |
|---|---|---:|---:|
| original | All queries | 1.000000 | [640] / [640] |
| original | Full query seen | 0.946479 | [545] / [552] |
| original | Full query unseen | 0.053521 | [95] / [88] |
| original | Withheld operation prefix | 0.026735 | [20] / [20] |
| original | Withheld prefix; all true transitions seen | 0.026002 | [17] / [18] |
| composition-holdout | All queries | 1.000000 | [640] / [640] |
| composition-holdout | Full query seen | 0.920803 | [528] / [536] |
| composition-holdout | Full query unseen | 0.079197 | [112] / [104] |
| composition-holdout | Withheld operation prefix | 0.026735 | [20] / [20] |
| composition-holdout | Withheld prefix; all true transitions seen | 0.006841 | [7] / [9] |

Independent reconstruction verifies every native trajectory and its probability,
all selected and removed training paths, fitted counts, tied nearest retrieval,
explicit three-step propagation, deterministic sampling uniforms, support labels
and 143,360 scored cases in 1,120 strata. The maximum discrepancy is 3.11e-15.
Known-answer and deliberately corrupted-forecast controls pass. Both full replays
match all 41 deterministic output files. Sources, plans, inputs, environments and
execution-specific timing receipts are bound separately. Earlier replay-only
receipts remain historical; they were not numerical acceptance.

The reader export contains 640 queries with declared skill/belief mechanics,
initial artifact and proposed operations. Hidden endpoints, actual goals,
transient states, training selections, model arrays and support diagnostics remain
in the separate scientific/evaluator export. These are forward queries with
privileged mechanics, not artifact-only reading. The controls validate arithmetic
and this finite construction, not human intent, inverse historical uniqueness,
architecture-wide generality or transfer to a changed law.

**Warrant:** exploratory method advantage for exact learned propagation under the
declared holdout; sampled-rollout reversal; support-specific qualification.
**Pursuit:** preserve the changed-law question as a separately specified challenge;
advance the independent routine/current-request revision branch. No extra seeds,
architecture settings or confirmation promotion are justified by these intervals.

[Frozen comparison](QUEUE_COMPLETENESS_PROTOCOL.md), [independent check](SUPPORT_REVIEW_PROTOCOL.md),
[evidence](../../../results/v19/G19-F-support-1/README.md).
