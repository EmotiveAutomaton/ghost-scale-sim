# V18.4 matched total training comparison

Does diverse training still harm farther prediction when histories and update counts match? Yes: farther mean logarithmic loss, where lower is better, rises from 1.68038 to 2.70708 for direct history, 2.10836 to 3.55791 for flat memory, and 2.15324 to 3.77260 for split memory. The no-history reader also worsens, from 1.58653 to 3.08139, while taught compositions improve. Together with the matched-original-exposure comparison, this rules out either reduced original-question exposure or total training volume alone as an explanation. Query allocation, decoder extrapolation and the resulting optimization path remain unresolved. This is a descriptive constructed-method result, miniature — architecture untested.

The matched-total pair uses the same 768 histories and 3,840 target rows per cell,
initialization seeds, batch size and 64 epochs. Only question allocation changes:
the original-menu arm has 768 labels per original question, whereas the diverse
arm has 480 per original or composed question. The earlier 480-history arm matches
that original-question count separately. No single comparison holds both original
exposure and total labels fixed. Both 768-history arms select on the same diverse
development set; compositions are exposed to selection even in the old-only arm.
Farther forms remain absent from fitting and selection. The tests share 96
coefficient lineages across sixteen cells, with seeds and queries paired within them.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify training allocation and scored question set; columns identify four
learned readers and the supplied-law exact posterior. The original and composition
rows average equally sized maker halves. Farther tests use their retained allocation.

| Training arm and question set | Question/world only | Direct history | Flat memory | Split memory | Exact reference |
|---|---:|---:|---:|---:|---:|
| Original menu, 480 histories; original | 1.49986 | 1.37708 | 1.02764 | 1.19240 | 0.68016 |
| Original menu, 480 histories; composed | 1.51787 | 1.58302 | 2.17733 | 1.89814 | 0.64746 |
| Original menu, 480 histories; farther | 1.61451 | 1.67326 | 2.29268 | 2.01584 | 0.75365 |
| Original menu, 768 histories; original | 1.49299 | 1.33843 | 1.10288 | 1.08275 | 0.68016 |
| Original menu, 768 histories; composed | 1.53564 | 1.60926 | 2.04502 | 2.02509 | 0.64746 |
| Original menu, 768 histories; farther | 1.58653 | 1.68038 | 2.10836 | 2.15324 | 0.75365 |
| Diverse menu, 768 histories; original | 1.49335 | 1.31951 | 0.86544 | 0.82689 | 0.68016 |
| Diverse menu, 768 histories; composed | 0.88942 | 0.85576 | 0.82872 | 0.81183 | 0.64746 |
| Diverse menu, 768 histories; farther | 3.08139 | 2.70708 | 3.55791 | 3.77260 | 0.75365 |

Increasing original-menu training from 480 to 768 histories does not reproduce
the diverse arm's farther deterioration: the no-history and flat readers improve
slightly, while direct and split readers worsen much less. At matched total
histories, diverse training improves taught compositions but substantially raises
farther loss for every reader. This triangulates against two single-factor
explanations; it does not isolate a universal causal effect of diversity or exclude
interactions with optimizer dynamics and development selection. The no-history
reversal again shows that recurrent memory is not required for the failure.

Ten isolated publication controls and all learning controls pass. Extracted-source verification independently rebuilds
105 means and 3,840 fixed forecasts across 60 reader/test combinations with zero
maximum discrepancy. Source, plan, retained input and selected-weight hashes match;
portable reassembly validates every exported scientific file. This is bounded
forecast replay, not full retraining or architecture-wide severity. Fit seeds,
queries and cells are not independent training-dataset replications. V15 C11/M01
remain failed instruments; this result makes no human-intent claim.

Science charged 4,461.421875 CPU seconds and took 4,604.257853 wall seconds;
adjacent replay charged 5.656250 CPU seconds and took 7.262147 wall seconds.
Together they used 4,467.078125 CPU seconds and 76.86 wall minutes. This lies within
the admitted science estimate of 4,150–5,650 CPU seconds. Campaign clocks remain
unchanged. Conditional-target learning, feasible bank decoding and distinct-law
mixtures are separate admitted experiments, not established results here.

[Portable evidence](../../results/v18/exploratory-loop/L4-old-total-1/SCIENTIFIC_MANIFEST.json).
