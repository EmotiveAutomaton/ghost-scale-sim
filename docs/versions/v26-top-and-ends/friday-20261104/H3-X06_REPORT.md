# H3-X06 exploratory map

With revision rate 0.025, how does the cell-to-cell sign variation seen at 0.05 behave at the complete and mostly incomplete revision endpoints across discovery timing and document length? The separate exploratory map retains 8 settings and 256 independent worlds per setting. Featured cell means (nats per revision) span: primary -1.69433 to 0.466893. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested. Reported per-event means average whole-world gain/event-count ratios only over worlds with at least one corresponding event; zero-event worlds are unavailable, not zero-valued. The featured cells use 98–238 of 256 sampled worlds each. This is neither an all-world average nor a ratio of pooled gains to pooled event counts; nominal whole-world sample size does not establish precision for this conditional statistic. The generating revision rates [0.025] are absent from the reader prior [0.3, 0.6, 0.9]. This is a misspecified-reader stress test; its negative gains do not establish that revision history is intrinsically harmful or identify a transition for a correctly specified reader.

Claim `G-V26-FR-H3X06-1`. Saved numerical results and finite closeouts remain unchanged; the interpretation was qualified on November 6.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 1, "reader": "contrasts"} | primary | 0.40259890585542757 | 0.1354345966674769 | 98 |
| 1 | {"history": 1, "reader": "contrasts"} | primary | -0.42432954655802074 | 0.1689651884651301 | 211 |
| 2 | {"history": 1, "reader": "contrasts"} | primary | 0.23138334818247555 | 0.150141857485459 | 139 |
| 3 | {"history": 1, "reader": "contrasts"} | primary | -1.5328950706737534 | 0.22242676253826157 | 238 |
| 4 | {"history": 1, "reader": "contrasts"} | primary | 0.4668927079783814 | 0.1390675350151124 | 101 |
| 5 | {"history": 1, "reader": "contrasts"} | primary | -1.1851830381939852 | 0.25010813986845565 | 204 |
| 6 | {"history": 1, "reader": "contrasts"} | primary | -0.2122907351720684 | 0.15618699421476323 | 128 |
| 7 | {"history": 1, "reader": "contrasts"} | primary | -1.694325536728082 | 0.2105374951733314 | 232 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "discovery_fraction": 0.25,
      "partial": 0.0,
      "r": 0.025,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.0,
      "r": 0.025,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.0,
      "r": 0.025,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.0,
      "r": 0.025,
      "units": 48
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.75,
      "r": 0.025,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.75,
      "r": 0.025,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.75,
      "r": 0.025,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.75,
      "r": 0.025,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h3-x06-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h3-x06-study/INDEPENDENT_REVIEW.json).
