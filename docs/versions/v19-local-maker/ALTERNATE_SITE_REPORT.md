# One alternative transformer access site

Does moving the intervention before final normalization make factor access practically selective? No practically meaningful benefit was found for the learned direction: its 32-observation purpose and belief gains are 0.00281 and 0.00623 nats, below the declared 0.02-nat margin. The class-mean belief direction still damages stay cases by 0.21473 nats. All 28,672 scores and both full replays verify this exploratory constructed-method result; joint selective access and human intent are not established.

Logarithmic loss measures forecast error in nats; lower is better. This table
shows learned rank-one replacement loss minus unchanged-recipient loss against
the same known-maker hybrid target. Rows separate observation prefix and factor.
Change donors flip the target factor and skill; stay donors change skill alone.
Brackets are 95% paired coefficient-lineage bootstrap intervals, conditional on
the four retained model fits. Negative differences improve forecasts.

| Observations | Factor | Change loss difference | Stay loss difference |
|---|---|---:|---:|
| 8 | Purpose | -0.00189 [-0.00247, -0.00124] | -0.00011 [-0.00020, -0.00003] |
| 8 | Belief | -0.00493 [-0.00555, -0.00434] | -0.00009 [-0.00023, +0.00004] |
| 32 | Purpose | -0.00281 [-0.00361, -0.00201] | -0.00001 [-0.00005, +0.00005] |
| 32 | Belief | -0.00623 [-0.00671, -0.00571] | -0.00020 [-0.00034, -0.00006] |

Every change interval remains inside the declared 0.02-nat practical threshold.
There are small directional improvements, but no practical selective-access win
for this probe. The longer-prefix factor accuracies are 80.27% for purpose and
80.47% for belief, where accuracy is the fraction of balanced labels correctly
classified and chance is 50%. These accuracies do not establish a causal role.

This table compares the intervention effect at the new site with that at the
original normalized site, paired within lineage, maker, draw and seed. It is the
difference between two intervention-minus-unchanged differences. Negative values
favor the new site; interval and row definitions match the preceding table.

| Observations | Factor | Change: new minus old site | Stay: new minus old site |
|---|---|---:|---:|
| 8 | Purpose | -0.00007 [-0.00023, +0.00008] | -0.00000 [-0.00004, +0.00003] |
| 8 | Belief | -0.00019 [-0.00028, -0.00012] | +0.00001 [-0.00002, +0.00004] |
| 32 | Purpose | -0.00008 [-0.00017, -0.00000] | -0.00006 [-0.00009, -0.00003] |
| 32 | Belief | -0.00019 [-0.00028, -0.00008] | +0.00003 [+0.00001, +0.00005] |

At 32 observations the additional change gains are only 0.000084 and 0.000191
nats. A detectable site difference does not turn a sub-margin intervention into
a practical result. No additional layer search is authorized by this result.

The next table gives learned-direction change loss minus each named control at
32 observations. Random means one frozen equal-rank direction per model/factor,
not an estimated distribution over arbitrary directions. Negative favors learned.

| Factor | Random direction | Shuffled labels | Wrong factor | Class-mean direction |
|---|---:|---:|---:|---:|
| Purpose | -0.00311 [-0.00436, -0.00173] | -0.00283 [-0.00366, -0.00202] | -0.00214 [-0.00299, -0.00129] | +0.04572 [+0.03620, +0.05611] |
| Belief | -0.00033 [-0.00102, +0.00062] | -0.00612 [-0.00658, -0.00562] | -0.00597 [-0.00664, -0.00519] | +0.03867 [+0.02774, +0.04923] |

The class-mean direction uses the same privileged factor labels but a different
fixed estimator. Its results below retain all prefixes and both change/stay
conditions, with the same loss and interval conventions.

| Observations | Factor | Change loss difference | Stay loss difference |
|---|---|---:|---:|
| 8 | Purpose | -0.04362 [-0.05968, -0.02750] | +0.00275 [+0.00051, +0.00529] |
| 8 | Belief | -0.04177 [-0.05670, -0.02398] | +0.12028 [+0.10148, +0.14030] |
| 32 | Purpose | -0.04853 [-0.05962, -0.03833] | +0.00940 [+0.00756, +0.01128] |
| 32 | Belief | -0.04490 [-0.05580, -0.03385] | +0.21473 [+0.19384, +0.23379] |

The longer-prefix purpose effect remains a limited candidate: change improvement
0.04853 nats with stay cost 0.00940. Belief still incurs 0.21473 nats of stay
damage. A purpose candidate and damaging belief edit cannot jointly establish
selective purpose/belief access. C2 composition is not admitted here.

The following table gives observed learned-direction loss-difference ranges
across the four combinations of two training draws and two saved seeds at 32
observations. These are fit ranges, not confidence intervals or independent worlds.

| Factor | Change range across four fits | Stay range across four fits |
|---|---:|---:|
| Purpose | [-0.00539, -0.00125] | [-0.00003, +0.00002] |
| Belief | [-0.01243, -0.00308] | [-0.00040, -0.00005] |

Each bootstrap averages makers, draws and seeds within eight coefficient
lineages before 10,000 resamples with seed 190501. Full per-fit site comparisons,
all seven arms, shorter-prefix controls and forecast movement are preserved in
the numerical exports. Two seeds and two training draws do not quantify universal
training uncertainty. There is no untouched confirmation or multiple-testing claim.

The site is the feedforward residual before the transformer's final normalization.
Every altered state passes through the unchanged normalization and head. The four
saved transformer models remain frozen; the recurrent reader has no corresponding
site and is not included. Eight auxiliary probes fit signed purpose/belief labels
at prefix 32 using the original closed-form ridge penalty, with no development
tuning or intervention-loss optimization. Prefix eight is transfer. No sequence
weights change and no additional shared model setting is consumed.

A separate four-head attention implementation and Gaussian-CDF activation rebuild
all pre-normalization training/development states within 4.78e-15. Normalized
states and original predictions agree within 2.00e-15. Augmented least squares
rebuilds every probe/control direction within 3.80e-13; all forecasts agree within
6.17e-15 and all 28,672 finite score rows within 5.33e-15. All 128 probe rows agree.
All 958 deterministic files match original, adjacent and extracted-source replays;
execution measurements are checked separately. Native targets rely on the prior
fully verified fixture and are not another independent world implementation.

The first review attempt failed because its output decoder accepted only a matrix
while the new state identity check supplied a full sequence. That failed reviewer
and its cost are retained. The corrected shape-general decoder completed the
entire reconstruction. No scientific producer, scoring rule or frozen source changed.

Blind reader exports contain anonymous observed codes and novelty flags. Factor
labels and probe directions are auxiliary-training evidence. True makers, laws,
hidden states, donor/hybrid pairing and scores remain evaluator/reproduction
evidence. Known-maker intervention targets have more information than the
history-conditional capability targets. Neither is historical local-process recovery.

**Warrant:** practically small learned-direction effects at the sole alternate
site; limited purpose-only class-mean candidate and nonselective belief control.
Exploratory constructed method, miniature — architecture untested. No human intent,
joint selective mediation or preceding-process correspondence follows.

**Pursuit:** continue the independently specified [in-alphabet missing-tool
experiment](MISSING_TOOL_PROTOCOL.md). F1 learned rollouts and a joint-process
G-P1 readout remain prepared alternatives requiring their own admission. The
primary comparison, confirmation and research week remain open.

[Frozen protocol](ALTERNATE_SITE_PROTOCOL.md),
[independent reconstruction](../../../results/v19/G19-C1-alternate-site-1/INDEPENDENT_REVIEW.json),
[role-separated exports](../../../results/v19/G19-C1-alternate-site-1/EXPORT_MANIFEST.json).
