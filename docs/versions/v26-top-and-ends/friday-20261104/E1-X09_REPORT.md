# E1-X09 exploratory map

At learning rate 0.02 and specialist share 1.0, do one-nat crossings persist, and how often do fixed-value makers cross falsely? The separate exploratory map retains 3 settings and 64 independent worlds per setting. Featured cell means (probabilities) span: ever crossed 0.203125 to 1; final above 0 to 1; recrossed below 0.203125 to 0.859375. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested.

Claim `G-V26-FR-E1X09-1`. Earlier finite results and closeouts remain unchanged.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | ever_crossed | 0.234375 | 0.053369535239372906 | 64 |
| 1 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | ever_crossed | 0.203125 | 0.050688158720985414 | 64 |
| 2 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | ever_crossed | 0.28125 | 0.05664543544843536 | 64 |
| 0 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | final_above | 0.0 | 0.0 | 64 |
| 1 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | final_above | 0.0 | 0.0 | 64 |
| 2 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | final_above | 0.0 | 0.0 | 64 |
| 0 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | recrossed_below | 0.234375 | 0.053369535239372906 | 64 |
| 1 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | recrossed_below | 0.203125 | 0.050688158720985414 | 64 |
| 2 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "A"} | recrossed_below | 0.28125 | 0.05664543544843536 | 64 |
| 0 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | ever_crossed | 1.0 | 0.0 | 64 |
| 1 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | ever_crossed | 1.0 | 0.0 | 64 |
| 2 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | ever_crossed | 1.0 | 0.0 | 64 |
| 0 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | final_above | 1.0 | 0.0 | 64 |
| 1 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | final_above | 1.0 | 0.0 | 64 |
| 2 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | final_above | 1.0 | 0.0 | 64 |
| 0 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | recrossed_below | 0.828125 | 0.04753177456453702 | 64 |
| 1 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | recrossed_below | 0.859375 | 0.043797805514170944 | 64 |
| 2 | {"history": 1024, "reader": "B-over-A-persistence", "truth": "B"} | recrossed_below | 0.765625 | 0.053369535239372906 | 64 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "environment": "stationary",
      "eta": 0.02,
      "evidence_path": true,
      "histories": [
        64,
        128,
        256,
        512,
        1024
      ],
      "max_history": 1024,
      "specialists": 1.0,
      "theta": 0.6,
      "truth_types": [
        "A",
        "B"
      ]
    },
    {
      "environment": "forced",
      "eta": 0.02,
      "evidence_path": true,
      "histories": [
        64,
        128,
        256,
        512,
        1024
      ],
      "max_history": 1024,
      "specialists": 1.0,
      "theta": 0.6,
      "truth_types": [
        "A",
        "B"
      ]
    },
    {
      "environment": "trade",
      "eta": 0.02,
      "evidence_path": true,
      "histories": [
        64,
        128,
        256,
        512,
        1024
      ],
      "max_history": 1024,
      "specialists": 1.0,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-e1-x09-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-e1-x09-study/INDEPENDENT_REVIEW.json).
