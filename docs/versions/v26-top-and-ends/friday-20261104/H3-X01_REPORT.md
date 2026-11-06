# H3-X01 exploratory map

With revision rate 0.3, how do incomplete revision and discovery timing change initial versus settled-goal recovery? The separate exploratory map retains 12 settings and 64 independent worlds per setting. Featured cell means (nats per revision) span: primary 0.179084 to 0.541975. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested. Reported per-event means average whole-world gain/event-count ratios only over worlds with at least one corresponding event; zero-event worlds are unavailable, not zero-valued. The featured cells use 62–64 of 64 sampled worlds each. This is neither an all-world average nor a ratio of pooled gains to pooled event counts; nominal whole-world sample size does not establish precision for this conditional statistic.

Claim `G-V26-FR-H3X01-1`. Saved numerical results and finite closeouts remain unchanged; the interpretation was qualified on November 6.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 1, "reader": "contrasts"} | primary | 0.4583081987855356 | 0.05614293398329994 | 63 |
| 10 | {"history": 1, "reader": "contrasts"} | primary | 0.5088800691170078 | 0.05184993769396594 | 64 |
| 11 | {"history": 1, "reader": "contrasts"} | primary | 0.19064364395412292 | 0.01231285894246541 | 64 |
| 1 | {"history": 1, "reader": "contrasts"} | primary | 0.22054044615993826 | 0.022887801159086948 | 64 |
| 2 | {"history": 1, "reader": "contrasts"} | primary | 0.4840105190398621 | 0.039801272180658286 | 64 |
| 3 | {"history": 1, "reader": "contrasts"} | primary | 0.17908353117615086 | 0.013961553421190674 | 64 |
| 4 | {"history": 1, "reader": "contrasts"} | primary | 0.5275941062686498 | 0.0491304520114617 | 64 |
| 5 | {"history": 1, "reader": "contrasts"} | primary | 0.21339874430852496 | 0.014636570207392764 | 64 |
| 6 | {"history": 1, "reader": "contrasts"} | primary | 0.48518145768572113 | 0.04458950695381931 | 64 |
| 7 | {"history": 1, "reader": "contrasts"} | primary | 0.20859227810937456 | 0.011890551100683672 | 64 |
| 8 | {"history": 1, "reader": "contrasts"} | primary | 0.5419753293057468 | 0.052515241631841486 | 62 |
| 9 | {"history": 1, "reader": "contrasts"} | primary | 0.20440884719847313 | 0.01560212175797888 | 64 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "discovery_fraction": 0.25,
      "partial": 0.0,
      "r": 0.3,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.0,
      "r": 0.3,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.0,
      "r": 0.3,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.0,
      "r": 0.3,
      "units": 48
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.5,
      "r": 0.3,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.5,
      "r": 0.3,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.5,
      "r": 0.3,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.5,
      "r": 0.3,
      "units": 48
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.75,
      "r": 0.3,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.75,
      "r": 0.3,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.75,
      "r": 0.3,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.75,
      "r": 0.3,
      "units": 48
    }
  ],
  "case": "Changed-goal cases have positive probability .5; at least 25% of excluded changed-goal documents remain uncertain under the final-text reader at each setting."
}
```

Validity: all original/replay raw bytes match; scalar posterior reconstruction and separate decimal whole-world aggregation pass. The declared generative model and some likelihood kernels are shared. Known answers validate equations, not real-world mechanisms. Independent settings are not paired causal contrasts. No simultaneous uncertainty guarantee or confirmation is asserted.

Construction and exclusions: Makers are nested within worlds. Reader comparisons share observations within a setting. Different setting cells use independent worlds; no paired causal claim. All unavailable outputs remain explicit.

Sounding Line S2/S3: additional goal evidence in revision history per event.
ToMpathy expertise/habit panel: constructed support for realigned spans, not a real-record validation.

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h3-x01-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h3-x01-study/INDEPENDENT_REVIEW.json).
