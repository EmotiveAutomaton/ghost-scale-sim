# H2-X08 exploratory map

With deliberation probability 0.003125, does the positive per-deliberation value at 0.00625 persist across the complete observation-dose and difficulty grid? The separate exploratory map retains 12 settings and 2048 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.0279076 to 0.100883. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested. Reported per-event means average whole-world gain/event-count ratios only over worlds with at least one corresponding event; zero-event worlds are unavailable, not zero-valued. The featured cells use 265–957 of 2048 sampled worlds each. This is neither an all-world average nor a ratio of pooled gains to pooled event counts; nominal whole-world sample size does not establish precision for this conditional statistic.

Claim `G-V26-FR-H2X08-1`. Saved numerical results and finite closeouts remain unchanged; the interpretation was qualified on November 6.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.06600844512415312 | 0.02467667992048548 | 296 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.027907630013347475 | 0.02242012409511415 | 265 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.0778519783111147 | 0.019928002710224012 | 307 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.05437691327106944 | 0.020835431332383386 | 311 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.10088325099029868 | 0.015992247744933036 | 598 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.06256951137068636 | 0.014975756381564778 | 541 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.0878159027151549 | 0.016646565973738474 | 540 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.08818464400134579 | 0.014333048293919164 | 547 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.10066556451220424 | 0.012891340535598499 | 929 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.06313711912765625 | 0.010357135426690377 | 957 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.08454360370718328 | 0.013083400710695326 | 904 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.06201790152960754 | 0.010624547170587344 | 934 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.003125,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.003125,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.003125,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.003125,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.003125,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.003125,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.003125,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.003125,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.003125,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.003125,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.003125,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.003125,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x08-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x08-study/INDEPENDENT_REVIEW.json).
