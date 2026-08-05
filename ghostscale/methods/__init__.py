"""The methodology layer: controls, provenance, and measures that are not experiments.

WHAT THIS DIRECTORY IS, AND WHY IT IS NOT AN EXPERIMENT DIRECTORY.

``ghostscale/v1`` through ``v10`` answer questions about readers. ``ghostscale/validation/``
answers questions another project asked. Neither of them is the right home for the machinery that
decides whether an answer is trustworthy at all, because that machinery has to be shared, stable,
and older than any single result that depends on it.

So: gates, positive and negative controls, provenance stamps, an overlap coefficient, and wrappers
for the measures that need a third-party library. Nothing here computes a finding. Everything here
computes a reason to believe or disbelieve one.

TWO RULES.

  1  NOTHING HERE MAY BE REQUIRED TO REPRODUCE A PUBLISHED NUMBER. Every module that leans on an
     optional dependency degrades to a recorded skip rather than an exception, and the base
     install of this package still runs every closed version unchanged. See the note in
     ``pyproject.toml`` about why the dependency list is the reproducibility contract.

  2  A GATE RECORDS; A TEST FAILS. Modules write their gate results into their verdict JSON and
     keep going, so a long exploratory sweep is never killed at hour two by a control that was
     expected to fail. ``tests/test_gates.py`` then walks every committed verdict and fails the
     suite if any gate failed. That puts the hard stop at the point where a result would become
     public, which is where S-2 needed one and did not have one.

WHY GATES EXIST AT ALL, stated once so the practice does not decay into ritual. Two defects
shipped in the first Sounding Line batch. S-2's manipulation never reached the reader -- the
feature streams were bit-identical with it switched off. S-3's detector threshold was fitted on
the labelled test data. Neither was a statistics error and neither would have been caught by a
larger sample. Both were "the measurement is not attached to the thing", and both are catchable by
a control that costs a few seconds:

  PLACEBO           a manipulation set to zero must reproduce the control EXACTLY, not merely
                    within an interval. This is the single most productive check in the
                    repository: it caught a shared RNG stream and a ``1/3`` that is not uniform in
                    floating point, both of which moved a headline number.
  POSITIVE CONTROL  a manipulation whose answer is known by construction must return that answer,
                    through the full stack. This is the one the repository did not have.
  LIVE MANIPULATION a manipulation set to maximum must CHANGE the output. The one-line version of
                    ``scripts/audit_s2_mixture.py``, run before the fact instead of after.
  ANALYTIC IDENTITY a quantity with a provable symmetry or bound must satisfy it. T-1's
                    mutual-information symmetry check was this before the category had a name.
  NO-ORACLE         a statistic must not change when a label it is not supposed to see is
                    permuted. S-3's fitted threshold fails this.
"""
from __future__ import annotations

from .gates import Gate, GateReport, gate_block
from .provenance import produced_by, stamp

__all__ = ["Gate", "GateReport", "gate_block", "produced_by", "stamp"]
