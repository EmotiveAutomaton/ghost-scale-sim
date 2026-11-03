# R13: frozen V26 protocol

W13 REPAIR. How much does a misleading cue have to mislead before persistence and proxy come apart? (ceiling 1.5 of the shared 3)

- W13's design, with its two uninformative shifts (restoration without the cue, and restoration delayed after the cue) replaced by shifts in which the cue misleads by degrees: the cue goes with a drop in an internal variable on a fraction m of records (0.1, 0.25, 0.5).
- **Case and check:** in every replacement shift, at every positive learning rate, the learner's trust in the cue changes after at least one record.
- **Known-answer:** W13's own: in training all types give identical choice likelihoods, and at learning rate 0 the learner's likelihood equals the proxy's on every shifted record.
- **Shed order:** m = 0.1.
- **Primary:** shifted records to 1 nat between learner and proxy, by m and learning rate.
- **Route:** to Sounding Line, the episodes a real-record test of "persistence at the top" would need; to ToMpathy, none this week.


## Declared implementation choices

W13 full type roster and softmax/lapse dynamics retained. Each positive shift record independently misleading with probability m; negative restoration changes trust only after the choice. Case check enumerates the positive-probability harmful branch at every positive rate. Finite trajectories with no harm remain in science; they are not redrawn.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[]
```
