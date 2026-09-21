# Equal-size predictive portfolios: learned reversal and exact-bank benefit

Does choosing which questions to preserve improve a fixed-size predictive bank? No learned-bank improvement was found. Targeted selection recovered the original five questions in every fit; random and diversity choices worsened loss by 0.15928 and 0.23028 nats, while improving the exact-bank diagnostic. Independent reconstruction and both full replays verify this exploratory constructed-method result; it does not establish local process correspondence or human intent.

Logarithmic loss measures the probability assigned to the outcome distribution,
in nats; lower is better. The table reports each alternative minus the original
portfolio, equally weighting four old-world classes. Brackets are 95% paired
lineage bootstrap intervals, conditional on the four retained fits.

| Bank | Random minus original | Diversity minus original | Targeted minus original |
|---|---:|---:|---:|
| Learned | +0.15928 [+0.10748, +0.21966] | +0.23028 [+0.15819, +0.30973] | -0.00000 [-0.00000, -0.00000] |
| Exact conditional (oracle) | -0.53834 [-0.62980, -0.45414] | -0.70367 [-0.81447, -0.60337] | -0.00000 [-0.00000, +0.00000] |
| Perturbed exact (oracle) | -0.33720 [-0.36096, -0.31055] | -0.49668 [-0.53921, -0.44784] | +0.00000 [-0.00000, +0.00000] |

Targeted selection chooses the original set in a different order in all four
fits. Its tiny signed differences are floating-point ordering effects, not an
improvement or statistically meaningful discovery. The other two learned-bank
pooled reversals exceed the declared 0.02-nat practical margin throughout their
conditional intervals. Exact and perturbed-exact advantages do not demonstrate
usable learned access: those arms receive evaluator-law predictions as inputs.

This table retains each learned-bank class. Rows identify the stipulated decision
rule and whether goal-dependent choice is active; columns give loss differences
with the same conditional intervals. Targeted selection ties in every class.

| Population | Random minus original | Diversity minus original |
|---|---:|---:|
| Satisficing, endogenous off | +0.21994 [+0.10656, +0.35212] | +0.24238 [+0.13056, +0.37626] |
| Satisficing, endogenous on | +0.26013 [+0.16896, +0.34405] | +0.37667 [+0.20725, +0.54653] |
| Softmax, endogenous off | +0.11657 [+0.07498, +0.17196] | +0.19216 [+0.10610, +0.30394] |
| Softmax, endogenous on | +0.04047 [-0.06051, +0.14304] | +0.10992 [-0.01997, +0.25931] |

The softmax class with endogenous choice has intervals spanning zero for both
alternative portfolios; that class alone is inconclusive. The other three show
reversals. The history-inert satisficing class remains included: a cross-world
learned decoder can fail there even when additional history has no conditional
value. No favorable-class population replaces the original equal-class result.

The next table gives absolute loss, its range across two draws crossed with two
feature seeds, and the fraction of raw forecasts requiring the frozen probability
repair. A range of four fits describes those fits; it is not a population interval.

| Bank | Portfolio | Mean loss | Range across four fits | Invalid raw distributions |
|---|---|---:|---:|---:|
| Learned | Original | 2.34368 | 2.24459–2.42397 | 94.58% |
| Learned | Random | 2.50296 | 2.39674–2.59913 | 97.75% |
| Learned | Diversity | 2.57396 | 2.52147–2.63015 | 98.19% |
| Learned | Targeted | 2.34368 | 2.24459–2.42397 | 94.58% |
| Exact conditional (oracle) | Original | 1.80302 | 1.76353–1.84251 | 93.55% |
| Exact conditional (oracle) | Random | 1.26469 | 1.10303–1.44829 | 97.46% |
| Exact conditional (oracle) | Diversity | 1.09935 | 1.09568–1.10303 | 99.80% |
| Exact conditional (oracle) | Targeted | 1.80302 | 1.76353–1.84251 | 93.55% |
| Perturbed exact (oracle) | Original | 1.99408 | 1.94390–2.08170 | 96.83% |
| Perturbed exact (oracle) | Random | 1.65688 | 1.45604–1.87813 | 99.41% |
| Perturbed exact (oracle) | Diversity | 1.49740 | 1.46152–1.51832 | 99.71% |
| Perturbed exact (oracle) | Targeted | 1.99408 | 1.94390–2.08170 | 96.83% |

Every portfolio preserves five 16-category distributions from an eight-question
menu. The two farther targets are outside the menu. Training splits contain 64
lineages for fitting, 32 for portfolio selection and 32 for ridge selection:
1,024/512/512 histories per draw, with no refit on selection data. Eight original
development lineages retain all four classes, both draws and both feature seeds.
The 10,000 bootstrap draws resample the eight paired lineage means after averaging
questions, draws and seeds. Test and confirmation lineages remain untouched.

Distribution teachers are simulator supervision shared by all selectors. The
shared acquisition head has 16,512 coefficients plus the fixed random basis; each
target head has 2,592. Targeted selection evaluates 30 candidates per fit and
each portfolio/bank validates three penalties. All arms incur eight-question
acquisition before five-question selection; deployed target width is five, but
this implementation still computes the shared eight-question acquisition head.
No end-to-end speed or acquisition-cost advantage is established. Component CPU
times are retained inside each execution's native charge, never added twice.

The Gaussian perturbation uses fitting-set per-entry learned-error variance before
repair. Mean squared error is about 0.02274–0.02286 for learned banks there, while
the perturbed exact bank after clipping/normalization is only 0.01000–0.01023.
These are not matched post-repair errors. The oracle/learned reversal therefore
does not isolate error direction, numerical conditioning or selection quality.
Numerical spectra remain diagnostics, not exact nonidentification certificates.

All eight checker controls pass. Independent centered ridge reconstruction checks
4,608 histories/references, four acquisition heads, all selector and penalty choices,
48 target heads, 3,072 strata and 2,048 spectra. Maximum saved-forecast discrepancy
is 4.54e-15. A second regroup reproduces all 45 contrasts and their intervals within
1.04e-14. Original, adjacent and extracted-source executions match all 125
deterministic files; each distinct timing receipt verifies separately. Sources,
plans, input hashes and full raw records remain immutable.

Anonymous artifact/context/source histories and feature arrays are reader inputs.
Training teachers have a separate declared archive. Identifiers, laws, raw truth,
fitted models, forecasts and verification are scientific/evaluator evidence.
The prior numerical-pending receipts are retained as historical verification states.

**Warrant:** exploratory constructed-method reversal for learned random/diversity
portfolios, numerical identity for targeted selection, and separate oracle
advantages; miniature — architecture untested. No held-out-law confirmation,
universal training uncertainty, local-process correspondence or human intent claim.

**Pursuit:** no learned portfolio winner advances to transfer on this result.
C1's saved-reader change/stay fixture audit is the next independent implementation;
F1's matched-computation local rollouts remain prepared. The primary local learning
comparison and the research week remain open. The shared two model settings are
consumed; no additional tiny fit is authorized by this result.

[Frozen protocol](PORTFOLIO_PROTOCOL.md), [independent verification](PORTFOLIO_REVIEW_PROTOCOL.md),
[numerical evidence](../../../results/v19/G19-A2-portfolio-1/INDEPENDENT_REVIEW.json),
and [role-separated exports](../../../results/v19/G19-A2-portfolio-1/EXPORT_MANIFEST.json).
