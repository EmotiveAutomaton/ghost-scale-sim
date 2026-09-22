# Checkpoint mixture compression: prediction survives lost maker support

Can a smaller set of complete maker/change hypotheses preserve useful prediction and uncertainty? Retaining 64 or 256 hypotheses keeps every individual final-checkpoint endpoint-loss interval within the declared 0.02-nat margin, but truncation can discard the true maker and sharply worsen maker-state loss. Independent reconstruction of 225,280 rows, all 19,800 paired estimates and both complete replays verifies this constructed-method tradeoff under supplied laws; recursive sufficiency, calibrated uncertainty, historical process correspondence and human intent remain unestablished.

The frozen comparison keeps 1, 16, 64, 256 or all complete maker/change hypotheses.
The full roster has 560 hypotheses at 32 observations and 3,632 at 128. All eight
development coefficient lineages, both saved observation draws and sixteen
equally weighted makers remain, with identity-aware and identity-omitted inputs,
independent/copied evidence, stationary/changing makers, purpose/skill changes
and all original checkpoints. No capacity was chosen after seeing outcomes.

The table reports ranges of mean changes across all 32 named final-checkpoint
strata, subtracting the full mixture in the same stratum. Endpoint logarithmic
loss scores future-artifact probabilities; maker-state logarithmic loss scores
the actual current maker, both in nats (lower is better). Inclusion is the
fraction of true makers in a tie-inclusive set with at least 90% posterior mass.
The interval column counts individual paired 95% lineage intervals strictly
inside the practical endpoint-loss margin. These are not simultaneous guarantees;
stationary purpose/skill aliases and independent-source presentation identities
do not add independent replication. Component bytes count float64 weights and
int32 selected indices, excluding shared roster metadata, runtime objects,
initial acquisition and selection work.

| Retained hypotheses | Endpoint loss change, nats | Individual endpoint intervals within ±0.02 | Maker-state loss change, nats | Inclusion change, percentage points | Component bytes at 128 observations |
|---|---:|---:|---:|---:|---:|
| 1 | -0.00271 to +3.34918 | 12/32 | +182.74425 to +540.59618 | -70.70312 to -25.00000 | 12 |
| 16 | -0.00198 to +1.11082 | 29/32 | +5.25694 to +104.72387 | -12.10938 to -0.78125 | 192 |
| 64 | -0.00112 to +0.00124 | 32/32 | -0.00565 to +24.14087 | -5.07812 to -0.78125 | 768 |
| 256 | -0.00005 to +0.00005 | 32/32 | -0.00249 to +2.68796 | -0.78125 to +0.00000 | 3,072 |

The full mixture uses 29,056 weight bytes at 128 observations, with implicit
indices. The current sixteen-state marginal uses 128 weight bytes and exactly
reproduces same-time endpoint forecasts. The latter says nothing about future
changes. Therefore selected-component storage savings do not establish either
the smallest sufficient state or an end-to-end computational advantage.

At 128 observations, with copied evidence and source identity omitted after a
purpose change, retaining 64 hypotheses changes endpoint loss by +0.00034 nats
[+0.00006, +0.00061], but worsens current-maker loss by +24.14087 nats
[+8.05482, +40.25967]. True-maker inclusion decreases 3.51563 percentage points
[1.56250, 5.46875 points lost]. This illustrates the distinction rather than
selecting a best capacity. All four fixed capacities and all strata remain.

The unchanged 1e-300 logarithm floor assigns about 690.78 nats when the true
maker receives zero probability. Hard truncation can remove that support even
when future endpoints remain similar under different makers. This explains large
maker-state penalties without weakening the scoring rule. Keeping 256 hypotheses
has much smaller average endpoint effects, but some maker-state intervals still
fail the ±0.02-nat equivalence criterion. A good predictive score cannot certify
preserved uncertainty, nominal coverage calibration or process correspondence.

All 880 means and 11,880 metric contrasts remain in VERIFICATION.zip, including
earlier checkpoints, removed mass, true-maker probability, set size, storage,
and omitted-versus-aware effects at every matched capacity. Nine measures yield
19,800 total estimates. Makers average within draw, then both draws within each
paired coefficient lineage. All draw means and lineage values are retained;
10,000 percentile resamples use seed 190501. Uncertainty conditions on these
saved draws, with no new observations, fits or protected confirmation lineages.

The independent checker reconstructs all 225,280 selections/projections/scores
and 14,080 maker-averaged cells, with maximum discrepancy 2.85e-14. All 45,056
full-mixture parent identities, 45,056 same-time marginal controls and 56,320
independent-source identities verify. A separate event review recombines every
estimate, paired draw/lineage value and interval without importing the producer
or checker regroup functions. Parent laws/posteriors inherit prior validation;
this review does not re-audit their entire likelihood construction.

Original, adjacent and extracted-source executions match all 111 deterministic
outputs. Source, plan, input, environment, output and separate timing hashes
verify. The checker passed nine mandatory controls, including complete native
reconstruction and deliberate corruption. The earlier withdrawn JSON-defect
checker and failed floating-point assertion remain retained and charged. The
assertion-only repair used exact 3/8 with one binary64-step tolerance; no
scientific selection, arithmetic, population, margin or score changed.

Reader inputs are the two unchanged bound parent exports. Scientific archives
and VERIFICATION.zip contain reproducibility and evaluator material; sixteen
large parent joint arrays remain retained with public hashes and byte lengths.
Supplied laws, true makers, schedules and scores never enter reader inputs.

**Warrant:** scoped endpoint similarity at fixed capacities alongside possible
loss of true-maker support; constructed method, miniature — architecture untested.
**Pursuit:** the current-marginal transition diagnostic tests future sufficiency
under fixed schedules. Bounded primitive feedback and a lossless future-schedule
quotient remain prepared alternatives. Neither depends on compression winning.
The research week and protected confirmation/reporting reserve remain active.

[Experiment](MIXTURE_COMPRESSION_PROTOCOL.md),
[independent review](MIXTURE_COMPRESSION_REVIEW_PROTOCOL.md),
[marginal transitions](MARGINAL_TRANSITION_PROTOCOL.md),
[primitive feedback](PRIMITIVE_FEEDBACK_PROTOCOL.md),
[future-schedule quotient](FUTURE_SCHEDULE_QUOTIENT_PROTOCOL.md).
