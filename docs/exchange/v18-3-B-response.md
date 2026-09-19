# V18.3 B: Changing maker state

What should a reader preserve when a maker changes? After an unannounced goal change, updating goal and belief while retaining the slower roles reduced future logarithmic loss from 3.299 for a static reader to 1.152; lower loss means better probabilistic prediction. The same rule lost when nothing changed, 1.023 versus 0.904, and when acquired skill changed, 1.058 versus 0.942. This is a descriptive constructed-mechanism result: selective updating helps when its timescale assumptions fit the change, and can hurt when they do not. These inference comparisons do not establish better craft uptake, because all readers shared the same separate learning routine.

The table reports future logarithmic loss after the possible change point, averaged within each of 20 coefficient draws across 16 architecture cells. Lower is better. The known-change reader receives the true change time and is a privileged timing comparison.

| What changes | Static history | Selective fast-role update | Update every role | Known change time |
|---|---:|---:|---:|---:|
| stationary | 0.90364 | 1.02325 | 1.04605 | 0.90364 |
| goal | 3.29933 | 1.15235 | 1.18820 | 0.94900 |
| belief | 2.95316 | 1.10247 | 1.15925 | 0.90112 |
| skill | 0.94206 | 1.05753 | 1.04452 | 0.91082 |
| tool | 0.52077 | 0.63100 | 0.64973 | 0.52077 |
| opportunity | 2.85215 | 3.00221 | 2.52897 | 2.85215 |
| gradual | 3.10481 | 1.16896 | 1.21272 | 1.56120 |
| return | 2.04603 | 1.23059 | 1.29380 | 0.95091 |

Eight packets retain 2,560 trajectories, each with 32 time steps. Time steps are not independent samples. Windowed and discounted readers, stationary and return controls, false reset counts, retained skill probability and identical native learning executions remain in the raw record. The selective reader models goal/belief transitions at a fixed rate; it is not a fully fitted Bayesian change-point detector. Its reset diagnostic does not feed a separate undisclosed controller. When opportunities change outside the fitted family, selective updating can lose to broader forgetting. The shared native learning routine prevents attributing execution differences to these inference methods. All retained-unit checks and aggregate means passed; eight source-extracted replays per packet give 64 bounded whole-trajectory replays.

[Complete current record](../versions/v18-selective-acquisition/research-extension/RESULTS.md). V18.3 remains active.
