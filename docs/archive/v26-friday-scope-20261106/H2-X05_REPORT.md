# H2-X05 exploratory map

With deliberation probability 0.025, does the positive but noisy per-deliberation value at 0.05 persist across the complete observation-dose and difficulty grid? The separate exploratory map retains 12 settings and 256 independent worlds per setting. Featured cell means (nats per deliberation) span: primary 0.0447569 to 0.104966. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested.

Claim `G-V26-FR-H2X05-1`. Earlier finite results and closeouts remain unchanged.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 16, "reader": "contrasts"} | primary | 0.09972592261773323 | 0.026561479175868805 | 166 |
| 1 | {"history": 16, "reader": "contrasts"} | primary | 0.06768024147133446 | 0.017858110093363203 | 182 |
| 6 | {"history": 16, "reader": "contrasts"} | primary | 0.06107682839486673 | 0.025982375597898125 | 175 |
| 7 | {"history": 16, "reader": "contrasts"} | primary | 0.07199191790146096 | 0.01899033015121626 | 195 |
| 2 | {"history": 32, "reader": "contrasts"} | primary | 0.1049659893641663 | 0.020047288493038204 | 232 |
| 3 | {"history": 32, "reader": "contrasts"} | primary | 0.06528633707334384 | 0.015374006955992397 | 243 |
| 8 | {"history": 32, "reader": "contrasts"} | primary | 0.06463284537156296 | 0.02071231911780937 | 223 |
| 9 | {"history": 32, "reader": "contrasts"} | primary | 0.0785851564805144 | 0.015946810484042968 | 233 |
| 10 | {"history": 64, "reader": "contrasts"} | primary | 0.1005550425033829 | 0.012073685750325221 | 254 |
| 11 | {"history": 64, "reader": "contrasts"} | primary | 0.0447568946528078 | 0.009789953192299805 | 255 |
| 4 | {"history": 64, "reader": "contrasts"} | primary | 0.08875158187618049 | 0.01339677659788574 | 256 |
| 5 | {"history": 64, "reader": "contrasts"} | primary | 0.062185158571346204 | 0.01127736534553196 | 253 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "d": 0.025,
      "hard": 1,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.025,
      "hard": 3,
      "q": 0.2,
      "units": 16
    },
    {
      "d": 0.025,
      "hard": 1,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.025,
      "hard": 3,
      "q": 0.2,
      "units": 32
    },
    {
      "d": 0.025,
      "hard": 1,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.025,
      "hard": 3,
      "q": 0.2,
      "units": 64
    },
    {
      "d": 0.025,
      "hard": 1,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.025,
      "hard": 3,
      "q": 0.5,
      "units": 16
    },
    {
      "d": 0.025,
      "hard": 1,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.025,
      "hard": 3,
      "q": 0.5,
      "units": 32
    },
    {
      "d": 0.025,
      "hard": 1,
      "q": 0.5,
      "units": 64
    },
    {
      "d": 0.025,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x05-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h2-x05-study/INDEPENDENT_REVIEW.json).
