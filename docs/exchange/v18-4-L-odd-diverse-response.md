# V18.4 learned memory: diverse questions on complementary support

Does the repair from teaching composed questions survive swapping the maker combinations used for training? It does: on the now-covered compositions, mean logarithmic prediction loss, where lower is better, is 0.85764 for flat memory and 0.83488 for split memory versus 0.87301 for direct history. Untouched farther-question losses are 3.61553, 3.45373 and 2.75825, again worse for every reader than with old-question training. The paired support swap strengthens the specific-supervision explanation; it does not establish systematic transfer beyond the taught menu. These are descriptive constructed-method results, miniature — architecture untested.

The complementary maker-support regime matches its old-question counterpart in
512 training histories per cell, 128 development histories, 96 test histories,
width 48, 64 epochs and three fit seeds. Five of eight questions rotate per
history, preserving total labels but reducing exposure per question. The added
compositions are covered in training and development; the two farther forms are
untouched. Shared world draws and test histories make this a paired supervision
intervention and support swap, not independent test-world replication.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify actual training exposure; columns compare direct history, flat
recurrent memory, three recurrent slots and the supplied-law exact reference.
The opening paragraph averages the equally sized maker-combination halves.

| Actual test exposure | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|
| Untrained even combinations, original questions | 1.36816 | 0.90644 | 0.85836 | 0.67637 |
| Trained odd combinations, original questions | 1.36839 | 0.89472 | 0.85840 | 0.67039 |
| Untrained even combinations, now-covered compositions | 0.86912 | 0.85767 | 0.83407 | 0.64611 |
| Trained odd combinations, now-covered compositions | 0.87691 | 0.85761 | 0.83569 | 0.64068 |
| Both combination halves, untouched farther questions | 2.75825 | 3.61553 | 3.45373 | 0.75339 |

On these same compositions, old-question-trained losses were
1.67623 direct,
2.86228 flat and
3.00793 split. Farther-question counterparts
were 1.76186, 2.89984
and 2.93177. Both trained and untrained maker
combinations benefit on the newly covered forms. Original-question losses rise
slightly; supervision density, optimization and decoder extrapolation remain
competing explanations. Full-support diversity, frozen-state readouts and finite
predictive-bank access are admitted and still pending.

No learning-control failures occurred. Independent extracted-source verification
reconstructed all 90 means and replayed 2,880 fixed forecasts with zero maximum
absolute discrepancy. Source, input, raw-data and selected-weight hashes match;
portable reassembly checked every exported scientific file. This is bounded
forecast replay, not retraining or random-architecture severity. V15 C11/M01
remain recorded instrument failures.

Science cost 2,591.484375 charged CPU seconds and 2,657.9505615 wall seconds
(44.30 minutes). Adjacent verification cost 4.765625 charged CPU seconds.
The first two diverse fits average 46.71 wall minutes. This updates the final
diversity fit's forecast to about 04:46 UTC, with a 25% margin to about 04:58 UTC;
neither estimate is a completion receipt. Further useful designs remain open.

[Portable scientific evidence](../../results/v18/exploratory-loop/L-odd-diverse-1-r4/SCIENTIFIC_MANIFEST.json).
