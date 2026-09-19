# V18.3 H: Finite compression and changed questions

A memory optimized for one question need not preserve the best answers to another. With two memory symbols, the exact old-question optimum had logarithmic loss 1.18834 versus 1.18900 for the best observation-bit code, but new-question loss was worse: 1.13447 versus 1.12202. At four symbols the new-question disadvantage was 0.01559. These are descriptive constructed-method results from exhaustive finite codebooks, not evidence that a role-organized neural memory is uniquely necessary.

The table averages within twenty coefficient draws across the sixteen declared cells. Lower logarithmic loss is better. Codes are selected only for the old question; a generator-informed decoder measures the information they retain for new questions. This is not a learned decoder-transfer result.

| Maximum stored bits | Flat old loss | Observation-bit old loss | Flat new loss | Observation-bit new loss |
|---:|---:|---:|---:|---:|
| 0 | 1.47962 | 1.47962 | 1.16829 | 1.16829 |
| 1 | 1.18834 | 1.18900 | 1.13447 | 1.12202 |
| 2 | 1.18206 | 1.18532 | 1.12270 | 1.10710 |
| 3 | 1.18054 | 1.18054 | 1.05147 | 1.05147 |

All 4,140 partitions of eight histories were enumerated, with optima certified separately at cardinalities one, two, four and eight. Equal maximum storage does not mean equal code entropy: at two symbols, entropy was 0.48314 versus 0.52740 nats. At zero and three bits the methods coincide. Across cells, the two-symbol new-question difference has both signs and ties; the pooled advantage is not universal. The acquisition-interference contrast is zero here because the balanced latent-state mixture preserves the relevant observable distribution. Declared factor settings are not sixteen independent architectures.

All 320 retained units and 512 packet means passed checks; eight fixed whole-unit replays from extracted source passed. An earlier discarded pilot had an inactive shared-noise factor and an unwanted constant query-label entropy in the new loss. Both were corrected before this discovery packet. The independent/shared-noise construction now changes the joint while preserving its marginals. Full raw blocks, source, certificate inputs and replay bindings are in `results/v18/research-extension/H-core/`.
