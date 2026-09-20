# V18.4 learned memory: full maker support

Does training on every maker combination remove the failure on new questions? It does not. Across the two combination halves, mean logarithmic prediction loss, where lower is better, is 0.88336 for flat memory and 0.83404 for split memory versus 1.35010 for direct history on old questions. On new composed questions, losses reverse to 2.84959 and 2.98549 versus 1.71639. Missing maker support alone therefore cannot explain this failure at the tested training budget; question supervision, objectives and decoding remain unresolved. These are descriptive constructed-method results, miniature — architecture untested.

This third main L1 regime trains on all positive-skill maker combinations while
retaining 512 training, 128 development and 96 test histories per cell, width 48,
64 epochs, three reader kinds and three fit seeds. Total histories and label rows
match the half-support regimes; each individual combination consequently receives
fewer examples. The five training question forms are unchanged. Test worlds and
histories are shared across these regimes: the support comparison is paired,
not another independent replication. Both test combination halves are now trained.

The table reports mean logarithmic prediction loss in nats; lower is better.
Rows identify actual exposure. Columns compare direct history, a flat recurrent
state, three recurrent slots and the supplied-law exact reference. The opening
paragraph averages the equally sized combination halves, not independent studies.

| Test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained even combinations, old questions | 1.35632 | 0.88693 | 0.83583 | 0.67637 |
| Trained odd combinations, old questions | 1.34388 | 0.87979 | 0.83226 | 0.67039 |
| Trained even combinations, new composed questions | 1.72093 | 2.85837 | 2.99784 | 0.64611 |
| Trained odd combinations, new composed questions | 1.71184 | 2.84082 | 2.97315 | 0.64068 |
| Farther question family, both trained halves | 1.80569 | 2.98964 | 3.06669 | 0.75339 |

Full maker support preserves both recurrent old-question gains and new-question
failure. A matched per-combination sample increase could test an interaction with
data density, but missing question supervision and decoder access have more direct
pending comparisons. Three diverse-question regimes and the frozen-state readout
ladder remain queued. A supplied-law predictive-bank diagnostic is the next
distinct access test; it must report inversion conditioning and invalid raw
probabilities rather than let clipping disguise failure.

No learning-control failures were recorded. The evaluator reconstructed 180 scalar
scores and 288 target/exact references. The independent extracted-source verifier
reconstructed all 90 means and replayed 2,880 fixed forecasts with maximum absolute
error zero. Source, input, raw-data and selected-weight hashes match. The portable
export was reassembled and every scientific file checked. This is bounded forecast
replay, not independent retraining, random-architecture severity or human evidence.
The retained V15 C11/M01 instrument failures remain unchanged.

Science used 2,653.3125 charged CPU seconds and 2,725.0692 wall seconds; adjacent
verification added 5.265625 charged CPU seconds. The three completed main regimes
average about 2,477.86 science CPU seconds and 42.41 wall minutes. Three remaining
main regimes project 2.12 wall hours from the 02:25 UTC boundary, with a 25% planning
margin, before the already admitted U2 and L3 diagnostics. These forecasts do not
fulfill the minimum exploration window.

[Portable scientific evidence](../../results/v18/exploratory-loop/L-all-old-1-r4/SCIENTIFIC_MANIFEST.json).
