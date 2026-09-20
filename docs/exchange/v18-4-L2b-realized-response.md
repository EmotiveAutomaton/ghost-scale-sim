# V18.4 realized-target control

Does learned history transfer beyond the original questions when training states are sampled independently and selection sees only those questions? No in this realized-target control: farther mean logarithmic loss, where lower is better, is 1.70223 for direct history, 2.91068 for flat memory and 3.02246 for split memory, versus 1.57940 without history. Recurrent memory still helps on the original questions. The conditional-target comparison remains pending, so this arm does not establish the effect of changing training targets. This is a descriptive constructed-method result, miniature — architecture untested.

This first L2b arm fits twelve selected readers: four architectures and three
initialization seeds, width 48, 64 epochs and batch size 128. Each of sixteen
cells supplies 768 training histories with five original-question targets per
history and 384 development histories. Hidden states are sampled independently
from the sixteen permitted nonzero-skill states. Targets are exact response
distributions at the sampled state, not single sampled artifacts. Development
uses the same realized-state target and original menu. Both compositions and
farther forms are absent from fitting and selection.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows distinguish the original, composed and farther question sets. Columns show
the no-history neural rival, three learned history readers and supplied-law exact
inference. The first two rows average equally sized maker halves. The test set
contains 96 coefficient lineages with paired cells, queries and fit seeds; those
repeated observations are not independent training-dataset replications.

| Question set | Question/world only | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|---:|
| Original | 1.48711 | 1.31306 | 0.85567 | 0.80949 | 0.67447 |
| Untaught compositions | 1.55192 | 1.64328 | 2.95921 | 3.07101 | 0.64288 |
| Farther | 1.57940 | 1.70223 | 2.91068 | 3.02246 | 0.75870 |

The recurrent gains remain specific to the original questions. On compositions
and farther questions, every learned history reader is worse than the no-history
rival. Farther paired excess losses over that rival are 0.12282 for direct,
1.33128 for flat and 1.44306 for split; their descriptive lineage-bootstrap
intervals are respectively [0.09327, 0.15506], [1.18192, 1.47814] and
[1.23814, 1.64622]. These are descriptive, within-dataset comparisons, not
architecture-wide confidence statements or independent fit replications.

This is the baseline for the already running conditional-target arm. Its new
state sampling, lineage and old-only development differ from L4, so cross-study
score differences do not isolate any of those changes. Only the two L2b arms
form the controlled target comparison. The exact reference still uses its
stipulated 24-state prior and is not claimed Bayes-optimal for the balanced test
allocation. A supplied-law bank decoder is a separate diagnostic of access.

All learning controls pass. Independent extracted-source verification reconstructs
all 105 means and 3,840 fixed forecasts across 60 reader/test combinations exactly.
The separate scalar target audit checks 160 training/development rows; maximum
teacher discrepancy is 3.33e-16. The same audit records state counts without
requiring random balance. Plan, source, input, raw forecast and selected-weight
hashes match, and portable reassembly validates every scientific file. Replay is
bounded forecast reconstruction, not full retraining or architecture severity.
Historical V15 C11/M01 remain failed instruments; no human-intent claim follows.

Science and adjacent verification charged 4305.359375 CPU seconds and took 75.84 wall minutes. Science alone charged 4285.937500 seconds; verification charged 19.421875. Native CPU receipts take precedence when larger than self-reported parent time. The fixed campaign budget and clocks are unchanged.

[Portable evidence](../../results/v18/exploratory-loop/L2b-realized-targets-1/SCIENTIFIC_MANIFEST.json).
