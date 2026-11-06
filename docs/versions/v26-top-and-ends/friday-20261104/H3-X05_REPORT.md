# H3-X05 exploratory map

With revision rate 0.05, does the larger per-revision value seen at 0.1 persist across revision completeness, discovery timing and document length? The separate exploratory map retains 12 settings and 128 independent worlds per setting. Featured cell means (nats per revision) span: primary -0.235126 to 0.499156. These ranges describe different cells, not an uncertainty interval or a selected winning setting. Constructed-method evidence; miniature, architecture untested. Reported per-event means average whole-world gain/event-count ratios only over worlds with at least one corresponding event; zero-event worlds are unavailable, not zero-valued. The featured cells use 69–128 of 128 sampled worlds each. This is neither an all-world average nor a ratio of pooled gains to pooled event counts; nominal whole-world sample size does not establish precision for this conditional statistic. The generating revision rates [0.05] are absent from the reader prior [0.3, 0.6, 0.9]. This is a misspecified-reader stress test; its negative gains do not establish that revision history is intrinsically harmful or identify a transition for a correctly specified reader.

Claim `G-V26-FR-H3X05-1`. Saved numerical results and finite closeouts remain unchanged; the interpretation was qualified on November 6.

The table lists frozen setting indices, observation strata and featured means. Standard errors use independent whole worlds; makers are averaged within worlds. Means from distinct cells have not been pooled. All cells, missing values and whole-world estimates remain in the aggregate JSON.

| Setting | Observation stratum | Measure | Mean | Standard error | Available worlds |
|---|---|---|---:|---:|---:|
| 0 | {"history": 1, "reader": "contrasts"} | primary | 0.31322229643176047 | 0.13819000507735477 | 83 |
| 10 | {"history": 1, "reader": "contrasts"} | primary | 0.11828898813337906 | 0.12423873933999921 | 92 |
| 11 | {"history": 1, "reader": "contrasts"} | primary | -0.19652155042317934 | 0.16888653769039247 | 128 |
| 1 | {"history": 1, "reader": "contrasts"} | primary | 0.219719750796242 | 0.1402328103853515 | 120 |
| 2 | {"history": 1, "reader": "contrasts"} | primary | 0.26141605169316906 | 0.1756652280698316 | 96 |
| 3 | {"history": 1, "reader": "contrasts"} | primary | -0.09022155113458251 | 0.15718928057178677 | 128 |
| 4 | {"history": 1, "reader": "contrasts"} | primary | 0.4991557755592305 | 0.12759060608617934 | 75 |
| 5 | {"history": 1, "reader": "contrasts"} | primary | 0.3162828817988121 | 0.11066236633035588 | 121 |
| 6 | {"history": 1, "reader": "contrasts"} | primary | -0.11753763881676997 | 0.17690728938617825 | 89 |
| 7 | {"history": 1, "reader": "contrasts"} | primary | -0.12066685282821148 | 0.1295930280051158 | 126 |
| 8 | {"history": 1, "reader": "contrasts"} | primary | 0.3594617683679243 | 0.1267935782658736 | 69 |
| 9 | {"history": 1, "reader": "contrasts"} | primary | -0.23512638342125905 | 0.14974481436932272 | 123 |

Settings and prospective case requirement:

```json
{
  "settings": [
    {
      "discovery_fraction": 0.25,
      "partial": 0.0,
      "r": 0.05,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.0,
      "r": 0.05,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.0,
      "r": 0.05,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.0,
      "r": 0.05,
      "units": 48
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.5,
      "r": 0.05,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.5,
      "r": 0.05,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.5,
      "r": 0.05,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.5,
      "r": 0.05,
      "units": 48
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.75,
      "r": 0.05,
      "units": 12
    },
    {
      "discovery_fraction": 0.25,
      "partial": 0.75,
      "r": 0.05,
      "units": 48
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.75,
      "r": 0.05,
      "units": 12
    },
    {
      "discovery_fraction": 0.75,
      "partial": 0.75,
      "r": 0.05,
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

[Complete aggregate](../../../../results/v26/friday-20261104/v26-friday-20261104-h3-x05-study/SUMMARY.json). [Independent reconstruction](../../../../results/v26/friday-20261104/v26-friday-20261104-h3-x05-study/INDEPENDENT_REVIEW.json).
