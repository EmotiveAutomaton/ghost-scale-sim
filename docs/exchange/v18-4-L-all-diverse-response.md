# V18.4 learned memory: diverse questions with full maker support

Does teaching composed questions generalize beyond the taught menu when all maker combinations are represented? It does not in this comparison. On the now-covered compositions, mean logarithmic prediction loss, where lower is better, is 0.84787 for flat memory and 0.83017 for split memory versus 0.87602 for direct history. Untouched farther-question losses are 3.73355, 3.37085 and 2.81020, all worse than their paired old-question-trained counterparts. Full maker support preserves the specific-question repair and farther deterioration seen on both support halves. These are descriptive constructed-method results, miniature — architecture untested.

The full-support regime matches its old-question counterpart: 512 training
histories per cell, 128 development histories, 96 test histories, width 48,
64 epochs and three fit seeds. Five of eight questions rotate per history,
preserving total labels while lowering exposure per original question. All maker
combinations and the added compositions occur in training and development; only
the farther forms remain question-family holdouts. The same world draws and test
histories are reused. This is a paired supervision intervention, not independent
test-world replication. Full support has fewer examples per combination than
the half-support regimes at this fixed total sample budget.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows state actual training exposure; columns compare direct history, flat recurrent
memory, three recurrent slots and the supplied-law exact reference. The opening
paragraph averages the equally sized maker-combination halves.

| Actual test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Trained even combinations, original questions | 1.36571 | 0.91002 | 0.85893 | 0.67637 |
| Trained odd combinations, original questions | 1.34674 | 0.90130 | 0.85340 | 0.67039 |
| Trained even combinations, now-covered compositions | 0.87771 | 0.85058 | 0.83463 | 0.64611 |
| Trained odd combinations, now-covered compositions | 0.87433 | 0.84516 | 0.82571 | 0.64068 |
| Both trained halves, untouched farther questions | 2.81020 | 3.73355 | 3.37085 | 0.75339 |

Old-question-trained composition losses were 1.71639
for direct history, 2.84959 for flat memory and
2.98549 for split memory. Their farther losses were
1.80569, 2.98964 and
3.06669. The specific coverage intervention repairs the
added compositions in all three support regimes. It does not isolate reduced
supervision density, optimization, or the original decoder's extrapolation as the
cause of farther deterioration. The frozen-state readout and supplied-law bank
diagnostics remain admitted to distinguish access from information loss.

No learning-control failures occurred. Independent extracted-source verification
reconstructed all 90 means and replayed 2,880 fixed forecasts with zero maximum
absolute discrepancy. Source, input, raw-data and selected-weight hashes match;
portable reassembly checked every exported scientific file. This is bounded
forecast replay, not retraining or random-architecture severity. V15 C11/M01
remain recorded instrument failures.

Science cost 2,555.90625 charged CPU seconds and 2,622.591923 wall seconds
(43.71 minutes). Adjacent verification cost 4.5625 charged CPU seconds. The three
diversity fits average 45.71 wall minutes. The initial six main learning regimes
are now verified; the adaptive campaign remains open, with further diagnostic
science and replay already queued. Completion does not fulfill the minimum window.

[Portable scientific evidence](../../results/v18/exploratory-loop/L-all-diverse-1-r4/SCIENTIFIC_MANIFEST.json).
