# H2-X02 exploratory map

With deliberation probability 0.6, how do observation dose and difficulty burden change the value and misreading of pauses? The separate exploratory map retains 12 settings and 64 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.00695672 to 0.0463694. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested. Reported per-event means average whole-world gain/event-count ratios only over worlds with at least one corresponding event; zero-event worlds are unavailable, not zero-valued. The featured cells use 64–64 of 64 sampled worlds each. This is neither an all-world average nor a ratio of pooled gains to pooled event counts; nominal whole-world sample size does not establish precision for this conditional statistic.

Claim `G-V26-FR-H2X02-1`. Saved numerical results and finite closeouts remain unchanged; the interpretation was qualified on November 6.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.04636943615860019 | 0.005733548203952757 | 64 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.037927182726058065 | 0.0056523604846006196 | 64 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.038793145019581865 | 0.007509071114661252 | 64 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.03240623178721821 | 0.005295103287226226 | 64 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.021726285826422186 | 0.002648191513999576 | 64 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.01833717057463266 | 0.003040310128484262 | 64 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.023510851629559393 | 0.0030711899771755946 | 64 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.019190718123942054 | 0.0029129600733757695 | 64 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.009014148799884231 | 0.0013074598006690664 | 64 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.006956719481752022 | 0.0012283456155205033 | 64 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.010316116778728023 | 0.001561668509136513 | 64 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.008826719313651941 | 0.0015591768790535045 | 64 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.6,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.6,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.6,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.6,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.6,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.6,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.6,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.6,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.6,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.6,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.6,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.6,
      "hard": 3,
      "q": 0.5,
      "units": 64
    }
  ],
  "case": "At least 25% of excluded worlds remain uncertain under text-only reading at each setting; both deliberation and difficulty have positive probability."
}
```

Validity: all original/replay raw bytes match; scalar posterior reconstruction and separate decimal whole-world aggregation pass. The declared generative model and some likelihood kernels are shared. Known answers validate equations, not real-world mechanisms. Independent settings are not paired causal contrasts. No simultaneous uncertainty guarantee or confirmation is asserted.

Construction and exclusions: Makers are nested within worlds. Reader comparisons share observations within a setting. Different setting cells use independent worlds; no paired causal claim. All unavailable outputs remain explicit.

Sounding Line S5/S6: goal evidence in pauses after modelling choice difficulty.
ToMpathy current-attention marks require an explicit difficulty model.

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x02-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x02-study/INDEPENDENT_REVIEW.json).
