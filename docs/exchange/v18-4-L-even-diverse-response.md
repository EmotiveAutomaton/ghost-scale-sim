# V18.4 learned memory: diverse questions on even support

Does adding composed questions to training repair learned-memory transfer? It repairs those now-covered questions, but farther-question prediction worsens. Across the two maker-combination halves, mean logarithmic prediction loss, where lower is better, is 0.85285 for flat memory and 0.83334 for split memory versus 0.87369 for direct history on the added compositions. On the untouched farther questions, losses are 3.82158, 3.45505 and 2.66348, all worse than their matched old-question-training counterparts. Question coverage explains a repair within the trained menu; it does not establish systematic transfer beyond that menu. These are descriptive constructed-method results, miniature — architecture untested.

The first diverse-question regime holds maker support, 512 training histories per
cell, 128 development histories, 96 test histories, width 48, 64 epochs and three
fit seeds fixed against the even-support old-question regime. Five of eight
question forms rotate per history, so the total label count stays fixed while
exposure to each original question falls. The three added forms are covered in
both training and development. Two farther forms remain untouched. Test worlds
and histories are shared: this is a paired supervision intervention, not an
independent replication or proof of a general composition mechanism.

The table gives mean logarithmic prediction loss in nats; lower is better. Rows
describe actual training exposure. Columns compare direct history, flat recurrent
memory, three recurrent slots and the supplied-law exact reference. The opening
paragraph averages the equally sized combination halves.

| Actual test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained combinations, original questions | 1.36652 | 0.90364 | 0.86402 | 0.67637 |
| Untrained combinations, original questions | 1.35963 | 0.89848 | 0.86069 | 0.67039 |
| Trained combinations, now-covered compositions | 0.87305 | 0.85589 | 0.83841 | 0.64611 |
| Untrained combinations, now-covered compositions | 0.87432 | 0.84982 | 0.82827 | 0.64068 |
| Both combination halves, untouched farther questions | 2.66348 | 3.82158 | 3.45505 | 0.75339 |

The old-question-trained counterparts' losses on the now-covered compositions
were 1.70594 direct,
2.97044 flat and
3.02711 split. Their farther-question losses
were 1.72506, 3.02999
and 2.92077. Repair also occurs for untrained maker
combinations, but farther questions worsen for all three architectures. Lower
per-question exposure, finite optimization and decoder extrapolation remain
competing explanations. Complementary/full maker-support diversity packets,
the frozen-state readout ladder and the supplied-law bank are already queued.

No learning-control failures were recorded. Independent extracted-source
verification reconstructed all 90 means and replayed 2,880 fixed forecasts with
zero maximum absolute discrepancy. Source, input, raw-data and selected-weight
hashes match; portable export reassembly verified every exported scientific file.
This is bounded forecast replay, not independent retraining or random-architecture
severity. V15 C11/M01 instrument failures remain unchanged.

Science used 2,877.1875 charged CPU seconds and 2,946.8097 wall seconds;
adjacent verification added 5.21875 charged CPU seconds. This first diversity fit
took 49.11 wall minutes, longer than the 42.41-minute mean of the preceding three
main regimes. The remaining two diversity fits therefore project about 98 minutes
from 03:15 UTC, before U2, L3 and L2a; timing remains a forecast.

[Portable scientific evidence](../../results/v18/exploratory-loop/L-even-diverse-1-r4/SCIENTIFIC_MANIFEST.json).
