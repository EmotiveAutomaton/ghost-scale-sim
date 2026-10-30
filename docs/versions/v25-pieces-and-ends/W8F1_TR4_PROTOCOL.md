# W8F1: TR4-calibrated short-history dose check

Frozen before constructed outcomes, 27 October 2026.

Sounding Line's completed TR4 aggregate found no adaptive-forecast advantage in
fourteen supported writers and 200 final-third sessions, but its temporal-order
reference failed its bias gate. The human result is descriptive only. It supplies
the reason and record-scale calibration for this constructed follow-up; it does
not supply a transferable drift rate or a human-effect target.

This one-generation W8 follow-up asks whether W8's exact adaptive reader improves
on its whole-history reader at short histories, and how that contrast changes
across a low-to-high constructed drift dose. It retains W8's finite simplex,
softmax choice model, exact adaptive library, controls, prefix histories, full
replay and independent reconstruction. Four walk settings use rates 0, 0.002,
0.01 and 0.1. They are ordinal points already present in W8's reader library,
selected to span stationary, low, middle and high constructed drift. They are not
mapped numerically to TR4's additive-log-ratio process variances.

The primary report is adaptive minus whole-history natural-log loss reduction per
record at 16 and 32 records, with 64 records retained as a short-history
sensitivity. Positive values favor adaptation. The dose at which the mean gain
first exceeds 0.01 nats per record is descriptive; no confirmation threshold is
created. The 128- and 512-record outputs remain retained for continuity but do
not change the primary. Sixteen independent whole-world blocks run per setting,
with eight makers inside each world.

The packet is generation one from `v25-w8-study-1`, capped at 3,600 CPU seconds,
half W8's family ceiling and within the remaining follow-up share. It is a
constructed method check, miniature, architecture untested. It cannot repair
TR4's biased reference, establish preference drift in people, or license a
Sounding Line claim.
