# V18.4 feasible old-bank decoding

Do learned old-question predictions retain useful history when decoded as valid mixtures of supplied finite states? Yes in this comparison: farther mean logarithmic loss, where lower is better, is 0.81764 for flat memory and 0.83576 for split memory, versus 0.93884 for the no-history bank and 0.93101 for the same-world prior. Both memory readers improve over clipped inversion, which gives 0.91168 and 0.97634. Useful history therefore survives this feasible decoder, but the decoder is given the world law and does not identify unique maker roles. This is a descriptive constructed-method result, miniature — architecture untested.

Reuse 1,536 fixed histories from the verified 480-history original-menu L4 fits:
48 indices from each of sixteen cells and both maker halves, balanced across
8/16/32-step lengths. Each supplies twelve selected learned banks: four readers
and three fits. Concatenate five old-question distributions over sixteen artifacts
into eighty coordinates. Fit nonnegative weights summing to one over the supplied
24 state-specific banks, then apply those weights to composition and farther
answers. This uses supplied law access, with no new training or test-driven
model selection. Seeds and histories are paired reused evidence, not independent
replication of the parent learning experiment.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows identify learned inputs and decoding method. Columns separate composed
questions from farther forms. Both reference rows retain the same public world;
the history posterior additionally conditions on observed history. All fit seeds
are averaged within each retained history before aggregating equally sized cells.

| Reader and decoder | Composed questions | Farther forms |
|---|---:|---:|
| Question/world bank, feasible | 0.87989 | 0.93884 |
| Direct-history bank, feasible | 0.79682 | 0.86033 |
| Flat-memory bank, feasible | 0.75050 | 0.81764 |
| Split-memory bank, feasible | 0.77099 | 0.83576 |
| Question/world bank, clipped inverse | 1.34289 | 0.94636 |
| Direct-history bank, clipped inverse | 1.48912 | 0.92748 |
| Flat-memory bank, clipped inverse | 1.47163 | 0.91168 |
| Split-memory bank, clipped inverse | 1.51861 | 0.97634 |
| Same-world prior | 0.87227 | 0.93101 |
| History posterior | 0.64775 | 0.75805 |

All three history readers beat the no-history learned bank and same-world prior
on both question groups after feasible decoding. The exact history posterior
remains better. Projection also improves all four readers relative to the fixed
clipped inverse in this aggregate. This establishes useful history in the retained
old-answer predictions under this supplied-law decoder; it does not establish a
learned law, unique roles, recursive state closure or superiority across architectures.
The inverse's raw probability violations affect 99.90–100% of mapped question
vectors, pooled over all five decoded questions. Clipping makes scored outputs
valid, but does not make their bank attainable by the declared finite state family.

The question/world-only bank has a smaller Euclidean projection residual than the
history banks but worse prediction. Residual is therefore a geometric diagnostic,
not a scientific success criterion. A feasible bank also need not determine unique
state weights; the alias control retains this limitation. Direct farther forecasts
from the original fit use a different test-state allocation and are not compared
as a within-history decoder intervention. The prior and posterior references use
a stipulated uniform 24-state prior, not the deterministic allocation's exact prior.

All 18,432 learned banks pass independent feasibility certificates; the maximum
simplex duality gap is 1.75658099e-14, below the unchanged 1e-7 bound. The gap
bounds remaining convex objective improvement; it is not prediction loss. The
measured affine ranks span 3–23 across the retained worlds.

Every unit passed scalar mixture, probability, score, prior/posterior and gap
reconstruction. All 3,200 aggregate means and eight whole-unit extracted-source
replays agree. Source, plan, parent input/forecast and portable-file hashes match;
eleven isolated bank/runtime controls pass. Verification is bounded replay,
not fresh-world confirmation or architecture-wide severity. The earlier numerical
fixture failure remains retained; its repair preceded admission and did not relax
the gap threshold. V15 C11/M01 remain failed instruments.

Science charged 261.765625 CPU seconds and adjacent verification 115.062500,
totaling 376.828125. Combined wall time was 385.151485 seconds (6.42 minutes),
consistent with the 180–500 CPU-second packet forecast. The already admitted
L2d pair applies this same decoder to matched-total old versus diverse training;
conditional-target fits independently examine how the old-answer bank is learned.
These follow-ons remain queued or running, not established by this result.

[Portable evidence](../../results/v18/exploratory-loop/L2c-feasible-bank-1/SCIENTIFIC_MANIFEST.json).
