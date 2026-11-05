# E1-X04 exploratory map

At learning rate 0.005 and specialist share 0.5, do one-nat crossings persist, and how often do fixed-value makers cross falsely? The separate exploratory map retains 4 settings and 64 independent worlds per setting. Featured cell means (probabilities) span: ever crossed 0.265625 to 1; final above 0 to 1; recrossed below 0.265625 to 0.875. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested.

Claim `G-V26-FR-E1X04-1`. Earlier finite results and closeouts remain unchanged.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | ever_crossed | 0.390625 | 0.06146842128667525 | 64 |
| 1 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | ever_crossed | 0.328125 | 0.05915529526875285 | 64 |
| 2 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | ever_crossed | 0.265625 | 0.05564461487585784 | 64 |
| 3 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | ever_crossed | 0.453125 | 0.06271663843120986 | 64 |
| 0 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | final_above | 0.0 | 0.0 | 64 |
| 1 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | final_above | 0.0 | 0.0 | 64 |
| 2 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | final_above | 0.0 | 0.0 | 64 |
| 3 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | final_above | 0.0 | 0.0 | 64 |
| 0 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | recrossed_below | 0.390625 | 0.06146842128667525 | 64 |
| 1 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | recrossed_below | 0.328125 | 0.05915529526875285 | 64 |
| 2 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | recrossed_below | 0.265625 | 0.05564461487585784 | 64 |
| 3 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "A"} | recrossed_below | 0.453125 | 0.06271663843120986 | 64 |
| 0 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | ever_crossed | 1.0 | 0.0 | 64 |
| 1 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | ever_crossed | 1.0 | 0.0 | 64 |
| 2 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | ever_crossed | 1.0 | 0.0 | 64 |
| 3 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | ever_crossed | 1.0 | 0.0 | 64 |
| 0 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | final_above | 1.0 | 0.0 | 64 |
| 1 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | final_above | 1.0 | 0.0 | 64 |
| 2 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | final_above | 1.0 | 0.0 | 64 |
| 3 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | final_above | 1.0 | 0.0 | 64 |
| 0 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | recrossed_below | 0.875 | 0.041666666666666664 | 64 |
| 1 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | recrossed_below | 0.828125 | 0.04753177456453702 | 64 |
| 2 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | recrossed_below | 0.8125 | 0.0491747370293402 | 64 |
| 3 | {"history": 2048, "reader": "B-over-A-persistence", "truth": "B"} | recrossed_below | 0.75 | 0.05455447255899809 | 64 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "environment": "stationary",
      "eta": 0.005,
      "evidence_path": true,
      "histories": [
        64,
        128,
        256,
        512,
        1024,
        2048
      ],
      "max_history": 2048,
      "specialists": 0.5,
      "theta": 0.6,
      "truth_types": [
        "A",
        "B"
      ]
    },
    {
      "environment": "forced",
      "eta": 0.005,
      "evidence_path": true,
      "histories": [
        64,
        128,
        256,
        512,
        1024,
        2048
      ],
      "max_history": 2048,
      "specialists": 0.5,
      "theta": 0.6,
      "truth_types": [
        "A",
        "B"
      ]
    },
    {
      "environment": "trade",
      "eta": 0.005,
      "evidence_path": true,
      "histories": [
        64,
        128,
        256,
        512,
        1024,
        2048
      ],
      "max_history": 2048,
      "specialists": 0.5,
      "theta": 0.6,
      "truth_types": [
        "A",
        "B"
      ]
    },
    {
      "environment": "shift",
      "eta": 0.005,
      "evidence_path": true,
      "histories": [
        64,
        128,
        256,
        512,
        1024,
        2048
      ],
      "max_history": 2048,
      "specialists": 0.5,
      "theta": 0.6,
      "truth_types": [
        "A",
        "B"
      ]
    }
  ],
  "case": "Every menu condition has differing fixed/circulating policies after an informative outcome in at least 25% of deterministic grid fixtures. Shift is tested only where it changes menus; threshold theta is not a B-truth intervention."
}
```

Validity: all original/replay raw bytes match; scalar posterior reconstruction and separate decimal whole-world aggregation pass. The declared generative model and some likelihood kernels are shared. Known answers validate equations, not real-world mechanisms. Independent settings are not paired causal contrasts. No simultaneous uncertainty guarantee or confirmation is asserted.

Construction and exclusions: Makers are nested within worlds. Reader comparisons share observations within a setting. Different setting cells use independent worlds; no paired causal claim. All unavailable outputs remain explicit.

Sounding Line V3/V4: forced-use analogues and longitudinal records needed.
ToMpathy core values: distinguish fixed weights from drifting usage ratios under explicit assumptions.

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-e1-x04-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-e1-x04-study/INDEPENDENT_REVIEW.json).
