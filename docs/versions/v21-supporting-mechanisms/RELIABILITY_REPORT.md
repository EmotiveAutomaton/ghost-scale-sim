# V21 fallible records: soft evidence versus hard rejection

Can a fallible inspection record help without excluding the true history? At 95% reliability, hard rejection excludes truth in 4.74% of native and changed-execution forecasts, making uncapped log loss infinite. Soft weighting excludes no truths and reduces loss by 0.4854 and 0.4836 nats, respectively. This is a verified constructed-method diagnostic with supplied reliability and support, miniature - architecture untested, and no confirmation.

We applied the same binary inspection witness to one retained converged direct forecast per fit. Four complete native past episodes and a context-visible current artifact remain fixed. Eight development worlds contain both fit seeds; 256 cases per world are averaged within fit and then within world. The changed condition alters current execution while keeping native history. Individual test cases differ across execution conditions; only worlds are paired. Source exact-law headroom exceeds the fixed margin and both native direct fits validation-stopped. No new architecture was fitted.

Rows below cross supplied record reliability with the three treatments. Natural-log loss in nats and full multiclass Brier are lower when forecasts are better. Exclusion is the percentage of true histories assigned zero probability; the same exclusion occurs in both execution conditions. A missing finite log score means genuine infinity, not capped loss.

| Reliability | Treatment | Native log loss | Changed log loss | Native Brier | Changed Brier | Truth excluded |
|---:|---|---:|---:|---:|---:|---:|
| 1.00 | no-record | 2.9010 | 3.0738 | 0.9215 | 0.9444 | 0.00% |
| 1.00 | soft-record | 2.2297 | 2.4035 | 0.8481 | 0.8880 | 0.00% |
| 1.00 | hard-record | 2.2297 | 2.4035 | 0.8481 | 0.8880 | 0.00% |
| 0.95 | no-record | 2.9010 | 3.0738 | 0.9215 | 0.9444 | 0.00% |
| 0.95 | soft-record | 2.4156 | 2.5902 | 0.8590 | 0.8958 | 0.00% |
| 0.95 | hard-record | infinite | infinite | 0.8620 | 0.9014 | 4.74% |
| 0.80 | no-record | 2.9010 | 3.0738 | 0.9215 | 0.9444 | 0.00% |
| 0.80 | soft-record | 2.7188 | 2.8911 | 0.8948 | 0.9237 | 0.00% |
| 0.80 | hard-record | infinite | infinite | 0.9173 | 0.9559 | 20.21% |
| 0.50 | no-record | 2.9010 | 3.0738 | 0.9215 | 0.9444 | 0.00% |
| 0.50 | soft-record | 2.9010 | 3.0738 | 0.9215 | 0.9444 | 0.00% |
| 0.50 | hard-record | infinite | infinite | 1.0185 | 1.0557 | 49.85% |

At 95% reliability, soft weighting reduces native loss by 0.4854 nats (descriptive 95% whole-world interval 0.4587–0.5138), and changed-execution loss by 0.4836 (0.4554–0.5141). At 80%, reductions are 0.1822 (0.1601–0.2018) and 0.1827 (0.1579–0.2056). All intervals use 4,096 whole-world bootstrap draws after nesting both fits; these reused development worlds are not confirmation. With perfectly reliable records, soft and hard treatments agree. At chance reliability, soft evidence leaves forecasts unchanged. The hard treatment still excludes 49.85% of truths. No completely empty-support fallback occurred.

The native-to-changed log-loss increase is 0.1728 nats without the record at every reliability. For soft weighting it is 0.1738, 0.1746, 0.1723 and 0.1728 at reliability 1, .95, .8 and .5. Hard weighting increases finite loss by 0.1738 at reliability 1; differences between infinite losses at noisy rates are undefined. Full-Brier increases are retained for every treatment and reliability in [the aggregate record](../../../results/v21/reliability/ANALYSIS.json). Thus using the witness helps within both conditions without removing the execution-shift penalty.

Proper-loss improvement does not improve every score: at 95% reliability, soft weighting raises incorrect answers made with at least 90% confidence from zero to 0.220% natively and 0.024% under changed execution. Hard weighting raises them to 0.293% and 0.049%. The full record retains modal and per-bit accuracy, top-eight coverage, below-floor mass and capped compatibility loss separately.

Each treatment applies exactly one new report. Removing duplicate report identities is therefore an identity control, not a new independent arm. The repeated no-record outputs across reliability settings are identical controls and are never counted as independent observations. The upstream native/shift direct fits share training inputs; earlier naive-Bayes duplicate-removal findings remain separate. B02 now explicitly manipulates copied historical episodes.

Verification binds 195 immutable complete packets, 6,483,968 saved forecasts and 27,137 retained files, reusing 192 matching score receipts. The new three packets contribute 147,456 scored forecasts including the replay. The complete frozen native design reproduces all 330 scientific files. Independent scalar probability-domain reconstruction agrees within 2.23e-16. Twelve false-context cases exercise the unchanged canonical ProcessCase and offline exporter; truth and exact probabilities remain evaluator-only. These are 12 unique fixtures, not 24 replications because they also replay. No browser or live reader inference is claimed.

Reliability is correctly supplied, not learned. Exact current target support is supplied equally to all treatments. The witness is independently generated conditional on true inspection; ordinary semantic context need not obey this channel. Its probability weighting tests constraint handling, not a new law-marginal ceiling or a general maker model. Fit capacity, validation convergence, support privilege, law uncertainty and historical aliasing remain separate. Stop hard-rejection deployment for this stipulated noisy field; retain the finite result without promoting a universal claim.

The next admitted branch is [B02 dependence](B02_DEPENDENCE_ADMISSION.json): grouped episode likelihoods at three powers, a power chosen only on validation laws, pooled empirical and latent-state readers, the frozen direct rival, and duplicate-removal controls. Two focused controls, two complete 141-file engineering consumers and 12 copying fixtures pass. One canonical context-ID construction failure was retained and repaired before scientific admission. A full-size one-world engineering pilot used 6.40625 CPU seconds for its consumer; the conservative full-block estimate is 153.75 seconds. The diagnostic plus full replay reserve at most 3,600 CPU seconds through the original queue. New copied-cell exact-law headroom and direct-fit convergence are required before any superiority screen. Wide/shifted and calibration-transfer extensions, B03 and C02 remain conditional scope. No executed identity is repeated.
