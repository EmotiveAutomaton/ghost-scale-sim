# H2-X01 exploratory map

With deliberation probability 0.3, how do observation dose and difficulty burden change the value and misreading of pauses? The separate exploratory map retains 12 settings and 64 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.0224003 to 0.0617249. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested.

Claim `G-V26-FR-H2X01-1`. Earlier finite results and closeouts remain unchanged.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.061724929531340816 | 0.008129570689133434 | 64 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.034222689950131065 | 0.010575018852474596 | 64 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.05268424418945147 | 0.009056568816084177 | 64 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.03709072037923691 | 0.009895240823402703 | 64 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.04669993087443028 | 0.006941209798251577 | 64 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.03477630643974801 | 0.005976891399890853 | 64 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.05116297930034682 | 0.006856495036246039 | 64 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.026600453102806513 | 0.005153690271314816 | 64 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.03890413565340806 | 0.0033477367611771626 | 64 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.022400265132274288 | 0.0031313446139805516 | 64 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.036297109638512744 | 0.0035565835315809406 | 64 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.023781575101953986 | 0.003640779331939663 | 64 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.3,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.3,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.3,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.3,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.3,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.3,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.3,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.3,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.3,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.3,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.3,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.3,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x01-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x01-study/INDEPENDENT_REVIEW.json).
