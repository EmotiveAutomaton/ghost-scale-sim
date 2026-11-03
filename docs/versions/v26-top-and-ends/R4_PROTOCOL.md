# R4: frozen V26 protocol

W4 PRECISION. Is W4's fall in detection between 512 and 1,024 records real? (ceiling 1.5 of the shared 3)

- **The case:** at 1,024 records W4's matched threshold rounds to 1 (tie mass 0.40), so detection there is set by tie credit. Check: arm (a) reproduces that threshold and tie mass.
- Rerun W4's records-needed curve on W4's own worlds and seeds, individual reader, α = 10. Regenerate each world at W4's own record levels (8, 32, 128, 512 and 1,024), so its counts at 128, 512 and 1,024 records are W4's exactly, and make the counts at 192, 256, 384 and 768 records by splitting W4's increments with multivariate hypergeometric draws under a separate declared seed (records within an increment are exchangeable). Score 128, 192, 256, 384, 512, 768 and 1,024 records in three arms:
  - (a) W4's rule unchanged: eight-decimal half-even rounding with the exact-arithmetic fallback (section 0.5), near-mean calibration panel;
  - (b) the same panel, with makers ranked by the exact complementary tail probability (1 minus the flag), computed in log space from integer counts and checked at 50 and 80 digits, without rounding;
  - (c) rule (b), calibrated on the exact-mean panel instead.
- **Primary:** detection at matched 5% false alarms at 1,024 records, arm (b) minus arm (a). The fall is a rounding artifact if arm (b) at 1,024 records is no more than 0.05 below arm (b) at 512.
- **Known-answer:** arm (a) reproduces W4's 512- and 1,024-record values exactly; with the uniform prior in place of the estimated one, the estimated-population flags equal the individual reader's in every arm (W4's own check).
- **Shed order:** arm (c); the 192, 384 and 768 levels.
- **Route:** to ToMpathy, its records-needed figure (its B15 record) is replaced by this curve, or confirmed; to Sounding Line, none this week.


## Declared implementation choices

Retained W4 alpha10, non-placebo setting0, sixteen original world identities. Original8/32/128/512/1024 increments regenerated unchanged. Extra levels split increments with a separately seeded multivariate hypergeometric bridge. Individual reader only; arm a original half-even rule with boundary fallback; b/c rank -log complementary tail checked at50/80 decimal digits. These are repairs on retained worlds, not fresh independent confirmation.

All choices are provisional scientific modelling choices, frozen before the excluded cost probe. No observed study result selected them. Sixteen independently seeded whole-world blocks per setting; check-only and cost-probe identities are excluded. Full byte replay and separate scalar posterior/decimal aggregation are required; some likelihood kernels remain shared. Known-answer fixtures independently verify the critical equations.

Texture variants are finite one-factor settings in the order below. Each receives its own excluded measured probe, complete replay and reconstruction. No texture result upgrades the primary.

```json
[]
```
