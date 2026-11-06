# H2-X07 exploratory map

With deliberation probability 0.00625, does the positive per-deliberation value at 0.0125 persist across the complete observation-dose and difficulty grid? The separate exploratory map retains 12 settings and 1024 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.0294321 to 0.127691. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested. Reported per-event means average whole-world gain/event-count ratios only over worlds with at least one corresponding event; zero-event worlds are unavailable, not zero-valued. The featured cells use 246–733 of 1024 sampled worlds each. This is neither an all-world average nor a ratio of pooled gains to pooled event counts; nominal whole-world sample size does not establish precision for this conditional statistic.

Claim `G-V26-FR-H2X07-1`. Saved numerical results and finite closeouts remain unchanged; the interpretation was qualified on November 6.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.12769060261551812 | 0.022643083160373303 | 288 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.07549842665339454 | 0.020557295912978526 | 246 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.0654889107300601 | 0.02480475153997859 | 263 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.02943206208850215 | 0.021212927592505558 | 266 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.06546159127020916 | 0.01685530679949184 | 495 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.05013377692701982 | 0.014220135464377882 | 477 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.08627398634017082 | 0.01747121746253451 | 481 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.04869733901550828 | 0.013907639721345623 | 475 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.10575550023362049 | 0.01242423184962911 | 733 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.06416346160409475 | 0.010684718400880143 | 711 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.08409240478913742 | 0.012864957652658887 | 728 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.06874594269899033 | 0.010566527763043518 | 717 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.00625,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.00625,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.00625,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.00625,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.00625,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.00625,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.00625,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.00625,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.00625,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.00625,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.00625,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.00625,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x07-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x07-study/INDEPENDENT_REVIEW.json).
