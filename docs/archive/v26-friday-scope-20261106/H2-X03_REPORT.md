# H2-X03 exploratory map

With deliberation probability 0.1, how do observation dose and difficulty burden change the value and misreading of pauses? The separate exploratory map retains 12 settings and 64 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.0102657 to 0.0830167. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested.

Claim `G-V26-FR-H2X03-1`. Earlier finite results and closeouts remain unchanged.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.07093903877209705 | 0.026590143211158133 | 63 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.0705667923799013 | 0.015458216471815875 | 63 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.05668032536602851 | 0.026486874836051708 | 64 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.010265704807413071 | 0.02592388318361052 | 64 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.04051228270177377 | 0.014490472151474657 | 64 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.03670193361766598 | 0.012790378299856313 | 64 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.08301665426129629 | 0.018469576259128834 | 64 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.030830405524181693 | 0.014171664811155053 | 64 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.06587272582329222 | 0.011412328483495703 | 64 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.04474510375146975 | 0.008092659877828152 | 64 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.0752118926368311 | 0.009158333245937284 | 64 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.04489384336601945 | 0.007436234886695857 | 64 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.1,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.1,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.1,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.1,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.1,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.1,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.1,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.1,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.1,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.1,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.1,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.1,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x03-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x03-study/INDEPENDENT_REVIEW.json).
