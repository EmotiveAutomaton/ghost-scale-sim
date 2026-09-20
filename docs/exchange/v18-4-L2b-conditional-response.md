# V18.4 conditional-target comparison

Does replacing realized-state training targets with exact history-conditional targets repair prediction on untaught questions? It does not repair farther transfer: mean logarithmic loss, where lower is better, rises from 1.70223 to 1.73769 for direct history, 2.91068 to 2.93959 for flat memory and 3.02246 to 3.15191 for split memory. Original-question memory scores improve, but all three history readers still lose to the no-history rival on untaught forms. This is a descriptive constructed-method result, miniature — architecture untested.

Both arms use the same independently sampled sixteen-state training population,
public inputs, initialization seeds, batch order, 64 epochs, width 48 and twelve
selected fits. Each of sixteen cells supplies 768 training and 384 development
histories. Conditional targets are posterior-weighted response distributions
given visible history and supplied world law. Realized targets are response
distributions at the sampled hidden state, not single sampled artifacts. Shared
development targets remain realized-state distributions and use original questions
only. Thus the intervention changes training targets without changing selection
targets or exposing composed/farther questions to training or development.

The table reports mean logarithmic loss in nats; lower is better. Each row names
a question group and reader. Columns compare the paired training targets and their
difference; positive differences mean conditional training is worse. The original
and composition groups each average two equally sized maker halves. Evaluation
uses 96 coefficient lineages; cells, fit seeds and queries are paired, not new
independent replications of training-data generation.

| Questions | Reader | Realized-state targets | Conditional targets | Conditional minus realized |
|---|---|---:|---:|---:|
| Original | Question/world only | 1.48711 | 1.48669 | -0.00042 |
| Original | Direct history | 1.31306 | 1.31077 | -0.00229 |
| Original | Flat memory | 0.85567 | 0.84266 | -0.01301 |
| Original | Split memory | 0.80949 | 0.79258 | -0.01691 |
| Original | Exact reference | 0.67447 | 0.67447 | +0.00000 |
| Untaught compositions | Question/world only | 1.55192 | 1.55749 | +0.00557 |
| Untaught compositions | Direct history | 1.64328 | 1.69771 | +0.05443 |
| Untaught compositions | Flat memory | 2.95921 | 2.77391 | -0.18530 |
| Untaught compositions | Split memory | 3.07101 | 3.10044 | +0.02943 |
| Untaught compositions | Exact reference | 0.64288 | 0.64288 | +0.00000 |
| Farther | Question/world only | 1.57940 | 1.58305 | +0.00365 |
| Farther | Direct history | 1.70223 | 1.73769 | +0.03546 |
| Farther | Flat memory | 2.91068 | 2.93959 | +0.02891 |
| Farther | Split memory | 3.02246 | 3.15191 | +0.12945 |
| Farther | Exact reference | 0.75870 | 0.75870 | +0.00000 |

The flat reader improves on untaught compositions while direct and split readers
worsen there. Farther scores worsen for all four learned readers, including the
no-history rival. Conditional averaging therefore does not remove the transfer
boundary at this finite training budget. This is consistent with remaining
optimization, representation and direct-decoder limitations; it does not identify
which one causes the failure. The population cross-entropy identity between the
two targets is not a guarantee about finite-budget learning or extrapolation.

All retained means and 3,840 fixed forecasts reconstruct exactly; 160 independently
audited training/development target rows agree to 3.33e-16. Twenty-one isolated
controls pass. Shared public arrays, development/test targets and complete world,
history and state rosters match across arms; only training targets differ. Raw
data, selected weights, source, plans and exact replay are exported and reassembled
with matching hashes. Verification does not retrain the models or establish
architecture-wide severity. The exact test reference retains its stipulated
24-state prior, distinct from the sixteen-state training teacher. V15 C11/M01
remain failed instruments; no human inference or learned-law claim follows.

The next supplied-law bank comparison asks whether useful target changes are
accessible in old-answer predictions despite direct transfer failure. Its realized
arm is already queued; the conditional arm requires verified-parent input binding,
the same roster, source freeze and an adjacent replay. No further fit seeds are
needed for that diagnostic.

Science charged 4322.125000 CPU seconds and adjacent replay 18.468750; combined wall time was 74.77 minutes. Charges use the larger native parent receipt plus recorded child work. Fixed campaign clocks and budget are unchanged.

[Portable evidence](../../results/v18/exploratory-loop/L2b-conditional-targets-1/SCIENTIFIC_MANIFEST.json).
