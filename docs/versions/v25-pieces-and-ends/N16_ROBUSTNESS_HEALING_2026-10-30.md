# N16 robustness: excluded normalization-guard repair

The first excluded validation attempt stopped after retaining 88 cases, before
any complete balanced block or scientific admission. A true-label posterior at
one N9 case was `1.0000000000000002`. The new extension's exact `<= 1` check
incorrectly rejected this floating-point normalization residual. The original
confirmation's implementation and accepted result were unaffected.

The separately frozen repair checks the complete probability vector for finite,
nonnegative entries and normalization within `1e-12`, with the same tolerance
above one. It does not clip, round or replace a probability or score. Comparator
support must remain positive; a zero candidate probability is valid because
the declared 0.1 mixture retains comparator support. All full-trace independent
reconstruction tolerances and scientific definitions remain unchanged.

The new tests cover the exact overshoot without modifying its input, rejection
of an actually invalid vector, and equality with the previously corrected N16
stress-map estimand. Fourteen targeted tests pass after repair. The earlier
temporary-directory fixture failure is also retained; that test passed with a
workspace-local scratch directory.

The failed capsule, logs, 88-case raw prefix, terminal and full 1,810-second
reservation remain retained. The repair keeps the same excluded seed identity
and must reproduce every retained raw prefix row before scientific admission.
It has a new source capsule and accounting root. Every old attempt charge is
carried forward under both the extension's original 24-hour sublimit and N16's
original 64-hour total. No clock, sample size or scientific setting was reset.

[Frozen extension protocol](N16_ROBUSTNESS_PROTOCOL_2026-10-30.md);
[repair bindings](../../../results/v25/n16-robustness-20261030/REPAIR.json).
