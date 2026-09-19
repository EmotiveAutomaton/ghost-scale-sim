# V18.2 first intervention wave

The first intervention wave tested whether readers can update preferences while keeping the maker's information separate from their own. Preference updating helped after a policy change, and treating a reader-only correction as the maker's belief badly distorted prediction. The missing-skill family was disproved in 42 of 128 histories. These are constructed mechanism results; the first learned models failed the held-out task-structure test, and the original selective-learning block is withheld pending its corrected comparison. Miniature — architecture untested.

The following table describes the original core wave only. Each predictive
comparison uses 128 independent maker/world lineages; repeated probes and
counterfactual siblings remain within that lineage. Expected log loss averages
negative log probability against the generator's entire action distribution,
rather than only the sampled action; lower is better.

| Comparison | What ran and what it establishes |
|---|---|
| Persistent policy change | Static posterior expected log loss 1.0216; preference-transition posterior 1.0052. A fixed transition prior helps in this particular changing-policy generator. |
| Reader-only correction | Perspective-separated expected log loss 0.9941; shared-world predictor 7.4598. The latter incorrectly overwrites the maker's stale signal. |
| Missing skill | 42 histories produce evidence outside the no-acquired-skill family. Bounded expansion restores nonempty support, while safe abstention remains explicit. Support repair alone is not proof of accurate later prediction. |
| Learned state | Both networks trained with 11,250 sampled target examples from 1,250 makers. Their best development epoch was the first, and generalization to the held-out incidence structure was poor. This is a representation/generalization failure, not an unavailable training path. |
| Selective uptake | WITHHELD: update weight initially changed goal learning but not repertoire learning. Original raw records remain, the coupling was repaired, and a zero-update control now covers both. |
| Individual versus population | With four people and 64 observations each, the individual interval width is 0.490 and population width 1.975. With 64 people and two observations each, population width is 0.600 and individual width 2.772. These intervals concern different estimands. Bias and duplicate-counting failures remain in the full cells. |

[Proof records](../../results/v18/maker-state/README.md) include independent
execution, direct reference checks and bounded source-frozen replay. The audited
learned summary groups the direct and network baselines on the identical probe
subsets; the first automatic summary had pooled four baseline probes against
three base network probes. Raw scores are unchanged and both summaries remain.
The early empirical predictor also performed unused posterior computation;
later code removes that overhead without changing its prediction rule.

The admitted follow-on wave covers unknown current state, longer histories,
novel goals, memory limits, uncertain access, paid observation execution,
repaired/dependent uptake, aligned coordinates and independent fit seeds.
The new assembly-family adapter is separately admitted. No confirmation is claimed.
