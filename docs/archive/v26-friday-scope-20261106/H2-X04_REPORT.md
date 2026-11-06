# H2-X04 exploratory map

With deliberation probability 0.05, does the higher and noisier per-deliberation value seen at 0.1 persist across observation dose and difficulty burden? The separate exploratory map retains 12 settings and 128 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.0371864 to 0.123624. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested.

Claim `G-V26-FR-H2X04-1`. Earlier finite results and closeouts remain unchanged.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.08716488108418532 | 0.02465897429335894 | 116 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.03718636726750899 | 0.021895922063685194 | 115 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.12362433923498618 | 0.024972187844919076 | 115 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.043043998450328635 | 0.02346176235587546 | 116 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.06763521647039122 | 0.017455089870310962 | 128 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.05086777180947187 | 0.016727625036146696 | 127 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.09616774004034932 | 0.01983431525658062 | 127 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.0714703467807204 | 0.014847006181565831 | 128 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.07852679063789135 | 0.011420783812074235 | 128 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.06318966726355188 | 0.009098222301315595 | 128 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.06368099275883513 | 0.010301279655848454 | 128 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.056970754648591804 | 0.00972969569262062 | 128 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.05,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.05,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.05,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.05,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.05,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.05,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.05,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.05,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.05,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.05,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.05,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.05,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x04-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x04-study/INDEPENDENT_REVIEW.json).
