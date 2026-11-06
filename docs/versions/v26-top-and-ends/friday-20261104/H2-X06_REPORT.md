# H2-X06 exploratory map

With deliberation probability 0.0125, does the positive per-deliberation value at 0.025 persist across the complete observation-dose and difficulty grid? The separate exploratory map retains 12 settings and 512 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.0544643 to 0.107968. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested. Reported per-event means average whole-world gain/event-count ratios only over worlds with at least one corresponding event; zero-event worlds are unavailable, not zero-valued. The featured cells use 224–469 of 512 sampled worlds each. This is neither an all-world average nor a ratio of pooled gains to pooled event counts; nominal whole-world sample size does not establish precision for this conditional statistic.

Claim `G-V26-FR-H2X06-1`. Saved numerical results and finite closeouts remain unchanged; the interpretation was qualified on November 6.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.1022171905455776 | 0.02537498599081296 | 233 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.05446425035431175 | 0.020803406864824004 | 228 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.06096630568752887 | 0.024509560154905387 | 239 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.08848790372958708 | 0.022830002758309424 | 224 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.08063906707075713 | 0.017982735243780103 | 379 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.0683928566676639 | 0.014866256335061332 | 379 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.10796833692870482 | 0.01794042379415825 | 366 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.06754756831018169 | 0.016401397745203748 | 367 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.08939217421430411 | 0.013297064117172377 | 468 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.07345149730288433 | 0.010593965831195868 | 464 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.100306039845295 | 0.013558009385853489 | 469 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.06948442920270334 | 0.01206759211694843 | 455 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.0125,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.0125,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.0125,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.0125,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.0125,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.0125,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.0125,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.0125,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.0125,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.0125,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.0125,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.0125,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x06-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x06-study/INDEPENDENT_REVIEW.json).
