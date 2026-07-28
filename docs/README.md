# Build specs

These are the documents each version of the simulation was written against, before it was
written. The code and the results files cite them by section, so they're here to make those
citations resolve.

| file | what it is |
|---|---|
| [SPEC_V1.md](SPEC_V1.md) | The original build spec. Hypotheses H1 through H6, the null suite N1 through N7, the parameter table, the invariant tests. |
| [SPEC_V2.md](SPEC_V2.md) | Version 2. Adds the Learner observer, reader heterogeneity, biased generated content, and nulls N8 through N12. |
| [PLAN_V2.md](PLAN_V2.md) | The longer working plan behind version 2, including the reasoning that didn't survive into the spec. |
| [SPEC_V3.md](SPEC_V3.md) | Version 3. Written to repair E8 under a finite-sample diagnosis, which version 3's own gate then refuted. Adds nulls N13 through N15. |

Read them alongside the matching results file. The specs say what was predicted; the results say
what happened, and where the two disagree the results file records it as a deviation rather than
quietly revising the spec.

The specs are the record of what was predicted, so they aren't edited after a run. Where a spec is
now known to be wrong, the correction lives in the results file that found it, not here.
