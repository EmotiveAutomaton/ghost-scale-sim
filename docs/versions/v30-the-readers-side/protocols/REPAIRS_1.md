# First bounded repairs, November 15

Failed capsules and charges are retained. Each family uses its one repair
allowance; no scientific threshold or grid is changed.

P: independent fits differ by at most 1.11e-13 in the failing HALF cell, but
decimal rounding placed exactly tied fitted labels on opposite sides of a bin.
COUNT now ranks exact integer counts. PERFECT and HALF corrected ranks also
use integer-count ordering, which follows algebraically from their permutation
symmetry. Other fitted ranks and estimates remain unchanged. Recheck every
retained pilot/main pair, retain any corrected summaries separately, then run
a complete fresh excluded fixture of the failed setting before resuming the
uncompleted grid. No valid completed simulation is resampled.

K: replace the compiler-unsupported float(boolean) with an explicit 1.0/0.0
conditional. Full trajectory controls, replay and independent reconstruction
must pass before admitting the payoff map.

L: the independent observed-event reference used 1-exp(-activation), which loses
precision at tiny positive activation. Replace it with the scalar math.expm1
identity. The primary stable expression, gates, grid and retained seeds are
unchanged. Rerun the entire excluded baseline and resolution validation.
The first attempt already met its 2,048-interval grid and exhaustive short-path
gates; only the independent exact-event arithmetic gate failed.
