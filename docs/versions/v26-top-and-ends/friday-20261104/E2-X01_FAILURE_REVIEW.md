# E2-X01 verification resource failure

Corrections first: this packet produced no scientific result and changes no prior
claim. Its complete original pass remains retained but unverified.

The 128-world-by-12-setting original pass completed in 125.96875 process CPU
seconds and wrote all 1,536 raw worlds, its aggregate and its raw hash index. The
cost probe had made replay look slightly faster than original, so the prospective
replay reservation was 121.6796875 CPU seconds. Replay reached that reservation
during the final setting and the runner stopped it. The packet is therefore
`FAILED`; it has no replay receipt, independent review, aggregate export, report or
claim. Its 411.4609375-second failed reservation remains charged.

This is an engineering resource-estimation failure, not a control failure or a
scientific null. The original and replay execute the same frozen worlds, so future
plans now reserve both against the slower of their measured probe estimates. The
three later E2 probes are held during this separate review rather than being used
as blind retries.

A first healing packet was justified. It bound and copied the failed packet's full
original raw index, aggregate, cost record and failed status; it does not redraw or
repeat the original pass. It receives a new immutable source capsule and a measured
shared-pool reservation for deterministic replay and independent reconstruction.
Only byte-identical replay plus scalar and decimal reconstruction can turn that
retained original into a reportable E2-X01 result. The failure and healing costs
remain separate, and the healing cannot promote any earlier E2 claim.

The healing completed. Replay used 137.046875 CPU seconds and matched every retained
raw byte; independent review used 99.4375 CPU seconds and passed scalar posterior
and separate decimal aggregation checks. The healing used 241.015625 CPU seconds
against its 448.90625-second reservation. It produced claim
`G-V26-FR-E2X01-1`; the failed packet and its original charge remain preserved.

[Failed packet status](../../../../results/v26/friday-20261104/CURRENT_STATUS.json).
[Verified healed result](E2-X01_REPORT.md).
