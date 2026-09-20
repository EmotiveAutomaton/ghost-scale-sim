# V18.4 learned memory: complementary support

Does swapping the trained maker combinations remove the failure on new questions? It does not. With old-question training on the complementary half, logarithmic prediction loss, where lower is better, is 0.88296 for flat memory and 0.83120 for split memory versus 1.36712 for direct history on unseen combinations. On new questions about trained combinations, the losses reverse to 2.86223 and 2.99240 versus 1.67857. The failure therefore survives this support swap; missing question supervision, the learned state and its decoder remain competing explanations. These are descriptive constructed-method results, miniature — architecture untested.

The second main L1 regime swaps even for odd training support while retaining
512 training, 128 development and 96 test histories per cell, width 48, 64 epochs,
three models and three fitting seeds. The five old training questions are unchanged.
The test worlds and histories are shared with the first regime, so this is a paired
support manipulation, not a new independent replication. The legacy raw condition
names have their exposure meaning reversed for odd-support training.

The table reports mean logarithmic prediction loss in nats; lower is better.
Rows identify actual combination and question exposure. Columns compare direct
history, one recurrent state, three recurrent slots and the supplied-law reference.

| Test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained odd combinations, old questions | 1.36726 | 0.88050 | 0.83518 | 0.67039 |
| Unseen even combinations, old questions | 1.36712 | 0.88296 | 0.83120 | 0.67637 |
| Trained odd combinations, new composed questions | 1.67857 | 2.86223 | 2.99240 | 0.64068 |
| Unseen even combinations, new composed questions | 1.67388 | 2.86232 | 3.02345 | 0.64611 |
| Farther question family, both combination halves | 1.76186 | 2.89984 | 2.93177 | 0.75339 |

Both support halves retain the old-question advantage and the new-question reversal.
This weakens a one-sided missing-combination explanation without choosing between
objective, question supervision and decoder limitations. The queued full-support
and diverse-question regimes address coverage. A frozen-state decoder ladder is
the next distinct diagnostic; additional labels for its readouts must be declared,
and successful decoding does not establish that the original model used that information.

Validation found no learning-control failures. The evaluator reconstructed 180
scalar scores and 288 target/exact references. The independent extracted-source
pass reconstructed all 90 means and replayed 2,880 fixed forecasts with maximum
absolute difference zero. Source, input, raw-data and selected-weight hashes match;
the portable export was reassembled and every scientific file checked. This is
bounded forecast replay, not independent retraining, a random-architecture severity
test or evidence about human mechanisms. V15 C11/M01 failures remain unchanged.

Science cost was 2,351.984375 charged CPU seconds and 2,416.56 wall seconds;
adjacent verification added 4.421875 CPU seconds. The two completed main packets
average 2,390.13 science CPU seconds and 40.91 wall minutes. Four remaining main
packets project about 2.73 wall hours from this completion, plus the admitted
matched-prefix diagnostic; allow a 25% planning margin. This forecast does not
fulfill the minimum exploration window.

[Portable scientific evidence](../../results/v18/exploratory-loop/L-odd-old-1-r4/SCIENTIFIC_MANIFEST.json).
