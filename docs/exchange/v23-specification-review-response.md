# V23 specification review

This is V23. The review found issues that prevent immediate launch: flooring
negative score differences invalidates the proposed confirmation guarantee for
raw gains, several declared controls are empirical predictions, and the required
ToMpathy ledger schema is not implemented in the inspected checkout.

The [review](../versions/v23-research-division/IMPLEMENTATION_REVIEW.md) gives a
concrete correction package. Exact rational arithmetic verifies that the proposed
flooring can make a one-step evidence factor average 1.375 under a true null,
where a valid e-value must average at most one. This is a method-validity finding,
not a finding about makers or people. The e-LOND formula matches its primary
source; its input-validity requirements remain essential.

The original sheet is preserved unchanged, and the
[work order](../versions/v23-research-division/WORK_ORDER.json) records all sixteen
families and twins in priority order, including the optional play family, with
the 90-CPU-hour ceiling and 18 protected hours. The 1,234 specified grid cells
include optional N4b and N8, before twins and additional controls. None is yet
implemented or admitted by this preparation.

Owner approval of the documented corrections is pending. Execution must also
wait for V21's published final report and verified closure, the pinned canonical
schema, and the corrected calibration, conformance and cost gates. V21 continues
its existing three-worker final diagnostic and completion review independently.
