# V28: finite descriptive maps

The owner commissioned [the packet](PACKET.md) on November 11. There are four
items: one V27 retrospective page, three family reports and one theory return.
No texture, follow-up, precision or registered scientific claim is admitted.
The twelve-hour CPU ceiling includes all preparation and failed work. Gear 2
permits four single-thread below-normal workers under the inherited host-load
backoff. Original last-start and drain endpoints remain November 15, 20:00 and
22:00 Pacific. K3 and all closed scientific records stay closed.

Current endpoint: all three maps are complete and verified, across 128 settings. G1 uses the separately frozen numerical repair, with the original failure retained. [Final theory return](THEORY_RETURN.md), [G1](reports/G1.md), [G2](reports/G2.md), [G3](reports/G3.md), and [V27 retrospective](V27_MAP.md). All 34 known answers pass, G1's ten were repeated after repair, and nine targeted checks pass. The finite queue has no scientific job remaining.

G1 numerical repair: exact integer-count symmetries receive equal goal probabilities and lowest-index tie breaking. Token marks are represented at twelve decimal places before midranking, with a ten-place resolution check. The original grid and seeds were replayed, all 292 pre-flight comparisons rechecked, and the full map independently reconstructed with unchanged numerical gates. See [repair protocol](../../../results/v28/G1/REPAIR_PROTOCOL.json) and [retained original](../../../results/v28/G1/retained-original/).

## Review and fixed implementation choices

- Owner-approved G1 comparison: use the larger of the original tolerance and
  three combined library-block standard errors. All 176 supplied score means
  reproduce to their printed precision from the original scripts and seeds;
  all 292 applicable comparisons pass. Sixteen old-setting AUC comparisons
  have no current-setting counterpart and are explicitly excluded; eight
  matching low-overlap AUC points use the supplied original output instead of
  its inaccurate summary range. Exact known-answer checks remain unchanged.
- Owner-approved G3 admission: use the exact retained-path match and preserve
  the failed fresh-sample comparison. Every censored-path classification agrees;
  maximum final log-odds discrepancy is 1.71e-13. Scientific initial stocks
  remain uniform over 0–30, separate from the fixed-5 pre-flight fixture.

- G1: history lengths share the same retained page, maker, library and nested
  history within each block. This gives the history-blind reader exactly flat
  rows and paired comparisons. Intervals use library blocks, including a
  delete-one-block jackknife for pooled AUC with exact midranks. Full-library
  beam equality uses eight types in the small world and 32 in the larger one.
- G2: the single unspecified 1,024-step N cell is ONE(12,1), chosen before
  outcomes because the packet names ONE's near-alias with slow dulling. The
  other 27 cells retain 256 steps. Moving references use the previous
  pre-action stock; counters include the current observation. Action, loss,
  windfall and clipping follow the packet's order exactly.
- G3: inflow is capped at the declared maximum of 30 before choice. The
  2,000-step weight control is the mean true-weight posterior across 100
  excluded worlds for each of seven classes in the plentiful regime.
- Both separation tables retain the literal zero-based last-below step plus
  one. A zero means above threshold at the first observed step, not inference
  without data. A terminal below-threshold path is censored to infinity.
  "Aliased" means unresolved within this horizon when over half are censored;
  the positive-rate grids do not establish exact infinite-history aliasing.
- Equal action laws imply zero evidence increments conditional on a common
  parameter/weight value. After parameter posteriors diverge, marginal type
  odds can still change. G3's surplus identity is conditional in this sense.
  No-need observations remain exactly uninformative for every model.
- G2's event table includes the otherwise missing case: D and W both separate
  a pair while N does not. It does not force that case into "only D" or "only W".

Each family requires known-answer checks, the delivered pre-flight comparison,
one original whole-grid run, one full seed replay and one independently coded
whole-grid reconstruction. Raw observations and process state remain local;
public results contain aggregates and hash bindings. The pre-flight scripts
were first read after the independently implemented checks passed. The source theory
handoff is retained outside the public repository because it includes private
correspondence notes; the reports cite only the relevant theory rules.

Current execution is recorded in `results/v28/CURRENT_STATUS.json`. A finite
endpoint receives its final reports and retires the monitor. An idle machine
then remains idle.
