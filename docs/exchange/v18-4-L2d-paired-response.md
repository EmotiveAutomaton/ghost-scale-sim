# V18.4 matched-training bank comparison

Does diverse training also damage farther prediction when both training arms use the same supplied-law decoder of their old-answer banks? No: farther mean logarithmic loss, where lower is better, falls from 0.82531 to 0.80272 for flat memory and from 0.82492 to 0.79555 for split memory; direct-history banks improve too. Useful bank information therefore survives despite the earlier deterioration in direct farther answers. This supports a decoder-access explanation within the supplied law, not learned law access. This is a descriptive constructed-method result, miniature — architecture untested.

Both source fits used 768 histories per cell and matched total histories, labels,
initializations and updates. Diverse training restored original-question exposure
relative to the smaller exposure control while adding composed questions; the
larger original-menu arm matches its total training allocation. Development is
shared across these two L4 arms. This diagnostic uses the same 1,536 selected
world/history/state identities and twelve banks per history in each arm. It adds
the same supplied finite-law decoder without retraining, changing selected weights
or exposing new training targets. All selected roster identities match exactly.

The table reports mean logarithmic prediction loss in nats, lower being better.
Rows name reader/decoder and question group. Columns compare original-menu with
diverse training and their paired difference; negative differences favor diverse
training. Seeds average within each reused history and repeated cells are paired,
not independent replication of the learning experiment.

| Reader and decoder | Questions | Original-menu training | Diverse training | Diverse minus original |
|---|---|---:|---:|---:|
| Question/world bank, feasible | Composition | 0.87740 | 0.87651 | -0.00090 |
| Question/world bank, feasible | Farther | 0.93778 | 0.93751 | -0.00027 |
| Direct-history bank, feasible | Composition | 0.78926 | 0.78022 | -0.00904 |
| Direct-history bank, feasible | Farther | 0.85703 | 0.84836 | -0.00867 |
| Flat-memory bank, feasible | Composition | 0.75956 | 0.72402 | -0.03554 |
| Flat-memory bank, feasible | Farther | 0.82531 | 0.80272 | -0.02260 |
| Split-memory bank, feasible | Composition | 0.76351 | 0.71503 | -0.04848 |
| Split-memory bank, feasible | Farther | 0.82492 | 0.79555 | -0.02937 |
| Question/world bank, clipped inverse | Composition | 1.31327 | 1.43431 | +0.12104 |
| Question/world bank, clipped inverse | Farther | 0.94478 | 0.95284 | +0.00806 |
| Direct-history bank, clipped inverse | Composition | 1.46900 | 1.45745 | -0.01155 |
| Direct-history bank, clipped inverse | Farther | 0.94494 | 0.95771 | +0.01277 |
| Flat-memory bank, clipped inverse | Composition | 1.46900 | 1.39887 | -0.07012 |
| Flat-memory bank, clipped inverse | Farther | 0.91218 | 0.85966 | -0.05252 |
| Split-memory bank, clipped inverse | Composition | 1.49058 | 1.37216 | -0.11842 |
| Split-memory bank, clipped inverse | Farther | 0.94349 | 0.83640 | -0.10709 |
| Same-world prior | Composition | 0.87227 | 0.87227 | +0.00000 |
| Same-world prior | Farther | 0.93101 | 0.93101 | +0.00000 |
| History posterior | Composition | 0.64775 | 0.64775 | +0.00000 |
| History posterior | Farther | 0.75805 | 0.75805 | +0.00000 |

Every feasible history bank improves on both question groups under diverse
training; all three beat its no-history bank and same-world prior. The no-history
bank changes little. Poorer information accessible through this old bank is
therefore insufficient as a sole account of diverse training's direct-answer
deterioration. Supplied-law decoding changes access to the information; it does
not prove unique state recovery or that a reader can learn this law. Optimization,
calibration and decoder behavior remain possible contributors.

The earlier direct farther test uses a different state allocation. Its published
means provide context, not a paired direct-versus-bank effect. The present two
bank arms do share every selected history and state. Both prior/posterior references
use a stipulated uniform 24-state prior; no test-allocation optimality is claimed.
Probability feasibility and projection residuals remain distinct from usefulness.

All 1,536 units passed scalar score, mixture and certificate checks. All 3,200
means and eight extracted-source whole-unit replays agree, as did the first arm's
verification. Source, plan, parent forecast and portable evidence hashes match.
Twenty-one isolated target/generation/bank/runtime controls pass. This is bounded
replay, not full retraining or architecture severity; V15 C11/M01 stay failed.
No human inference, unrestricted transfer or predictive-state closure is established.

All 18,432 bank certificates pass; maximum simplex gap 2.4236689e-14 is below 1e-7, with affine ranks 3–23. This second arm charged 258.593750 science CPU seconds and 110.765625 replay seconds, totaling 369.359375; combined wall time was 6.32 minutes.

[Portable evidence](../../results/v18/exploratory-loop/L2d-diverse-bank-1/SCIENTIFIC_MANIFEST.json).
