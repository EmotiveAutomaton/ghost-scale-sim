# V18.4 restored original-question exposure

Does restoring original-question exposure prevent diverse training from harming farther predictions? It does not in this comparison: farther mean logarithmic loss, where lower is better, rises from 1.67326 to 2.70708 for direct history, 2.29268 to 3.55791 for flat memory, and 2.01584 to 3.77260 for split memory. The no-history reader also worsens, from 1.61451 to 3.08139, while now-trained compositions improve. Reduced original-question exposure alone therefore cannot explain the boundary; the matched-total control remains pending. This is a descriptive constructed-method result, miniature — architecture untested.

The paired arms share nested history draws and exactly 480 labels per original
question in each architecture cell. Diverse training restores that exposure by
using 768 histories and 3,840 labels per cell, compared with 480 histories and
2,400 labels in the original-menu control. Every state/length/original-question
cell has ten labels in both arms. The added compositions each receive 480 labels.
Three seeds, width 48 and 64 epochs are fixed. Both arms select using the same
diverse development set and score the same 96 coefficient lineages. Compositions
are absent from original-menu training but present in development for both arms;
only farther forms are absent from both training and selection.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify the training allocation and scored question set. Columns compare
four learned readers and the supplied-law exact posterior. Equal-size maker
halves are averaged; queries, seeds and architecture cells are paired within lineage.

| Training arm and question set | Question/world only | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|---:|
| Original menu, 480 histories; original | 1.49986 | 1.37708 | 1.02764 | 1.19240 | 0.68016 |
| Original menu, 480 histories; composed | 1.51787 | 1.58302 | 2.17733 | 1.89814 | 0.64746 |
| Original menu, 480 histories; farther | 1.61451 | 1.67326 | 2.29268 | 2.01584 | 0.75365 |
| Diverse menu, 768 histories; original | 1.49335 | 1.31951 | 0.86544 | 0.82689 | 0.68016 |
| Diverse menu, 768 histories; composed | 0.88942 | 0.85576 | 0.82872 | 0.81183 | 0.64746 |
| Diverse menu, 768 histories; farther | 3.08139 | 2.70708 | 3.55791 | 3.77260 | 0.75365 |

Restored exposure retains covered improvement and farther deterioration in all
four learned readers. In the diverse arm, direct history now beats its no-history
rival on farther forms; both recurrent readers remain worse than that rival.
The no-history deterioration means this failure cannot be attributed solely to
recurrent memory. It remains compatible with fitting the new query distribution,
decoder extrapolation and optimization or selection effects. The two arms change
history count and total updates, so this is not an isolated question-diversity
effect. The queued 768-history original-menu arm supplies the matched-total
contrast. No partial running effects enter this report.

All learning controls pass. Independent extracted-source verification reconstructs
105 means and 3,840 fixed forecasts across 60 reader/test combinations with zero
maximum discrepancy. All retained-file, source, plan and selected-weight hashes
match; portable reassembly checks every scientific file. Verification covers
bounded forecast replay, not full retraining or architecture-wide severity.
No human-intent conclusion follows. V15 C11/M01 remain failed instruments.

Science charged 4,795.796875 CPU seconds and took 4,945.508667 wall seconds;
adjacent verification charged 6.062500 CPU seconds and took 8.294156 wall seconds.
Together they used 4,801.859375 CPU seconds and 82.56 wall minutes, within the
admitted 4,150–5,650 CPU-second science estimate. The campaign remains active.

[Portable evidence](../../results/v18/exploratory-loop/L4-diverse-restored-1/SCIENTIFIC_MANIFEST.json).
