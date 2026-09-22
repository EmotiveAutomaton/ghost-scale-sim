# Failed compression checker: exact-reference fixture repaired

The mandatory gate passed eight tests and failed one rounding-sensitive lower
bound before campaign reconstruction. The computed value is one binary64 step
below the exact 3/8 reference. The separately frozen replacement uses a tighter
equality assertion with one-step tolerance. No scientific definition changed.
Original sources, gate, log, generated fixtures and 80.171875 native CPU seconds
remain retained.

[Replacement](../G19-B-mixture-compression-1-verify-3/README.md).
