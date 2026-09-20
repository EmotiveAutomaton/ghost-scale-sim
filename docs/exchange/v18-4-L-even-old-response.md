# V18.4 learned memory: first main regime

Does learned memory fail on new maker combinations or on new questions? With old-question training on half the maker combinations, both recurrent readers retain their advantage on the other combinations: logarithmic prediction loss, where lower is better, is 0.88207 for flat memory and 0.84053 for split memory versus 1.35942 for direct history. On new questions about familiar combinations, the ordering reverses: 2.97557 and 3.04533 versus 1.71226. This isolates a question-transfer failure in the tested regime; it does not yet distinguish missing supervision from a deficient memory or decoder. These are descriptive constructed-method results, miniature — architecture untested.

This is the first main L1 regime: even-combination support, five old questions,
three models and three fitting seeds, width 48, 64 epochs. There are 512 training,
128 development and 96 test histories per fixed cell; the 16 cells are paired
within coefficient-draw lineage. Positive-skill states only are used. Fits and
repeated queries are not independent observations. Development selects the saved
weights; test results did not choose capacities or checkpoints.

The table reports mean logarithmic prediction loss in nats; lower is better.
Rows distinguish actual combination and question exposure for this even/old
training regime. Columns are direct history, one recurrent state, three recurrent
slots, and the separately supplied-law exact reference.

| Test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Familiar combinations, old questions | 1.36277 | 0.89072 | 0.84726 | 0.67637 |
| Other combinations, old questions | 1.35942 | 0.88207 | 0.84053 | 0.67039 |
| Familiar combinations, new composed questions | 1.71226 | 2.97557 | 3.04533 | 0.64611 |
| Other combinations, new composed questions | 1.69962 | 2.96531 | 3.00888 | 0.64068 |
| Farther composition absent from all training menus | 1.72506 | 3.02999 | 2.92077 | 0.75339 |

The larger trained regime preserves the predecessor's reversal. Missing latent
combinations alone does not explain it: the same old questions transfer across
the held-out combinations. This does not establish that the recurrent states
discarded the needed information. Diverse-question supervision is already queued;
an observable predictive-bank objective and frozen-state decoder comparison remain
distinct next tests. The exact reference and reconstructed intervention summary
use the finite law, so their advantage does not establish learned access.

Validation found no learning-control failures. It reconstructed 180 scalar scores
and 288 target/exact references. Independent extracted-source verification
reconstructed all 90 retained means and replayed 64 fixed forecast rows in each of
45 model-seed/test files (2,880 forecasts), with maximum absolute error zero.
Input, source, raw-data and selected-weight hashes match; export reassembly checks
every portable scientific file. This is bounded forecast replay, not independent
retraining or a random-architecture severity test. Historical V15 C11/M01 failed
instruments remain unchanged.

The science attempt used 2,428.28125 charged CPU seconds (219.859375 native parent
plus 2,208.421875 child) and 2,492.94 wall seconds; queued verification used
4.515625 charged CPU seconds. Individual direct/flat/split fits cost about
38/259/436 CPU seconds. Five remaining matched packets imply approximately 3.4
CPU hours or 3.5 wall hours at this measured rate, with a 25% planning margin;
this is a forecast, not fulfillment of the exploration window.

[Portable scientific evidence](../../results/v18/exploratory-loop/L-even-old-1-r4/SCIENTIFIC_MANIFEST.json).
