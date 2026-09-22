# Compression checker with exact fixture reference

The previous gate stopped on a one-step floating-point discrepancy from the exact 3/8 reference. This separately frozen replacement uses an equality check allowing one binary64 step. All nine fixtures run before numerical reconstruction. Scientific selection, scores and thresholds are unchanged; no test pass or numerical result is claimed by admission.

[Protocol](../../../docs/versions/v19-local-maker/MIXTURE_COMPRESSION_REVIEW_PROTOCOL.md).

The separately frozen assertion repair has now passed all nine mandatory controls
without skips, including the complete native fixture and deliberate corruption.
Independent campaign reconstruction was observed running afterward. This gate
verifies the repaired checker apparatus; it does not accept the campaign scores.
