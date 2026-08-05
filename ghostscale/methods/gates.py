"""Standing controls, recorded in every verdict and asserted by the test suite.

THE FIVE KINDS, and the real defect each one is aimed at. None of these is hypothetical; every one
of them corresponds to something that actually went wrong in this repository or in the project
this repository answers questions for.

  ``placebo``       A manipulation set to zero must reproduce the control EXACTLY. Not "within an
                    interval" -- exactly, to floating-point tolerance. Aimed at: a side channel
                    drawing from the rollout's RNG stream, which moved depth recovery 0.163 ->
                    0.282 on a channel carrying no information; and a ``1/3`` whose diagonal and
                    off-diagonal differ in the last bits, which flipped an argmax on a near-tied
                    posterior.
  ``live``          A manipulation set to maximum must CHANGE the output. Aimed at: S-2, whose
                    per-position goal mixture was drawn and discarded because
                    ``V5Environment.sample_feature`` ignores ``artifact.goal`` once a creator is
                    bound. The feature streams were bit-identical with the mixture off.
  ``positive``      A manipulation whose answer is known BY CONSTRUCTION must return that answer,
                    through the whole stack. Aimed at: the gap this repository had. A negative
                    control (null N28) tells you the instrument does not fire on nothing. Only a
                    positive control tells you it fires on something.
  ``identity``      A quantity with a provable symmetry, bound, or closed form must satisfy it.
                    Aimed at: nothing that broke -- this one has always passed -- which is exactly
                    why it is worth keeping. T-1's goal->depth and depth->goal edges must agree
                    because both reduce to a symmetric conditional mutual information, and they
                    agree to 3e-16.
  ``no_oracle``     A statistic must not move when a label it is not supposed to see is permuted
                    or withheld. Aimed at: S-3's detector threshold, fitted on the pooled
                    ground-truth-labelled divergences and re-fitted per cell; and E45's efficiency
                    result, where the simulating reader held the world's own emission map.

A gate carries the numbers it judged on, not just a boolean. A gate that fails and cannot say by
how much is a gate nobody will trust enough to fix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

KINDS = ("placebo", "live", "positive", "identity", "no_oracle")

#: Default tolerance for exactness gates. Chosen at 1e-12 rather than at machine epsilon because
#: a posterior that has been through pymdp's normalisation accumulates a few ulps legitimately;
#: the deviations these gates are aimed at are 1e-2 and larger, so there is four orders of
#: magnitude of daylight either side of this number.
EXACT_TOL = 1e-12


@dataclass
class Gate:
    """One control, its verdict, and the numbers it judged on.

    ``expected_to_fail`` is for a gate that DOCUMENTS A KNOWN FLAW rather than guarding against a
    new one. S-2's live gate is the case: its manipulation provably never reaches the reader, and
    running the gate anyway keeps the evidence attached to the result instead of leaving it in a
    commit message. S-3's no-oracle gate is the other: its detector threshold is fitted on the
    labelled test data, which is a real defect that the module is kept for anyway because its
    other findings stand.

    A gate marked this way failing is not news. A gate marked this way PASSING is news -- it means
    somebody fixed the underlying defect and did not update the gate, and the test suite reports
    it as an unexpected pass rather than quietly accepting it.
    """

    name: str
    kind: str
    passed: bool
    detail: str
    observed: float | None = None
    expected: float | None = None
    tolerance: float | None = None
    skipped: str | None = None       # reason, if the gate could not run at all
    expected_to_fail: bool = False

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown gate kind {self.kind!r}; expected one of {KINDS}")

    @property
    def is_problem(self) -> bool:
        """True when this gate's result is something a human needs to look at."""
        if self.skipped:
            return False
        return self.passed if self.expected_to_fail else (not self.passed)

    def to_dict(self) -> dict:
        d = {"name": self.name, "kind": self.kind, "passed": bool(self.passed),
             "detail": self.detail}
        for k in ("observed", "expected", "tolerance"):
            v = getattr(self, k)
            if v is not None:
                d[k] = float(v)
        if self.expected_to_fail:
            d["expected_to_fail"] = True
        if self.skipped:
            d["skipped"] = self.skipped
        return d


@dataclass
class GateReport:
    """Every gate a module ran, plus the summary the test suite reads.

    ``all_passed`` treats a SKIPPED gate as passing, and that is deliberate: a gate skipped
    because an optional dependency is absent must not fail a base install. ``skipped`` is reported
    separately so a skip can never be mistaken for a pass by a human reading the file.
    """

    gates: list = field(default_factory=list)

    def add(self, gate: Gate) -> Gate:
        self.gates.append(gate)
        return gate

    # -- the five constructors ---------------------------------------------------------------
    def placebo(self, name: str, observed_max_deviation: float, tol: float = EXACT_TOL,
                detail: str = "", expected_to_fail: bool = False) -> Gate:
        """A zero-strength manipulation must reproduce the control to ``tol``."""
        ok = bool(math.isfinite(observed_max_deviation) and abs(observed_max_deviation) <= tol)
        return self.add(Gate(
            name=name, kind="placebo", passed=ok, observed=observed_max_deviation,
            expected=0.0, tolerance=tol,
            expected_to_fail=expected_to_fail,
            detail=detail or ("a manipulation at zero strength reproduces the control exactly; "
                              "any deviation is the harness, not the manipulation")))

    def live(self, name: str, observed_change: float, min_change: float,
             detail: str = "", expected_to_fail: bool = False) -> Gate:
        """A full-strength manipulation must change the output by at least ``min_change``."""
        ok = bool(math.isfinite(observed_change) and abs(observed_change) >= min_change)
        return self.add(Gate(
            name=name, kind="live", passed=ok, observed=observed_change, expected=min_change,
            expected_to_fail=expected_to_fail,
            detail=detail or ("a manipulation at full strength changes the output; if it does "
                              "not, the manipulation is not reaching the measurement (S-2)")))

    def positive(self, name: str, observed: float, expected: float, tol: float,
                 detail: str = "", expected_to_fail: bool = False) -> Gate:
        """A known-answer task must return its known answer."""
        ok = bool(math.isfinite(observed) and abs(observed - expected) <= tol)
        return self.add(Gate(
            name=name, kind="positive", passed=ok, observed=observed, expected=expected,
            tolerance=tol,
            expected_to_fail=expected_to_fail,
            detail=detail or "a task whose answer is known by construction returns that answer"))

    def identity(self, name: str, lhs: float, rhs: float, tol: float = 1e-9,
                 detail: str = "", expected_to_fail: bool = False) -> Gate:
        """Two quantities that must be equal for an analytic reason must be equal."""
        ok = bool(math.isfinite(lhs) and math.isfinite(rhs) and abs(lhs - rhs) <= tol)
        return self.add(Gate(
            name=name, kind="identity", passed=ok, observed=abs(lhs - rhs), expected=0.0,
            tolerance=tol,
            expected_to_fail=expected_to_fail,
            detail=detail or "an analytic identity the harness has to satisfy"))

    def no_oracle(self, name: str, observed_change: float, tol: float,
                  detail: str = "", expected_to_fail: bool = False) -> Gate:
        """A statistic must not move when a label it should not see is permuted."""
        ok = bool(math.isfinite(observed_change) and abs(observed_change) <= tol)
        return self.add(Gate(
            name=name, kind="no_oracle", passed=ok, observed=observed_change, expected=0.0,
            tolerance=tol,
            expected_to_fail=expected_to_fail,
            detail=detail or ("the statistic does not move when a label it is not supposed to "
                              "see is permuted; if it does, it is an oracle statistic (E45)")))

    def skip(self, name: str, kind: str, reason: str) -> Gate:
        """A gate that could not run. Passes, but is counted separately and visibly."""
        return self.add(Gate(name=name, kind=kind, passed=True, detail="not run",
                             skipped=reason))

    # -- reporting ----------------------------------------------------------------------------
    def to_dict(self) -> dict:
        ran = [g for g in self.gates if not g.skipped]
        failed = [g for g in ran if not g.passed and not g.expected_to_fail]
        documented = [g for g in ran if not g.passed and g.expected_to_fail]
        surprises = [g for g in ran if g.passed and g.expected_to_fail]
        return {
            "all_passed": len(failed) == 0 and len(surprises) == 0,
            "n_gates": len(self.gates),
            "n_ran": len(ran),
            "n_skipped": len(self.gates) - len(ran),
            "n_failed": len(failed),
            "failed_names": [g.name for g in failed],
            "documented_failures": [g.name for g in documented],
            "unexpected_passes": [g.name for g in surprises],
            "gates": [g.to_dict() for g in self.gates],
            "how_to_read": (
                "a gate RECORDS here and does not raise, so an exploratory sweep is never killed "
                "mid-run. tests/test_gates.py walks every committed verdict and fails the suite "
                "on anything in failed_names or unexpected_passes, which puts the hard stop where "
                "a result would become public. THREE categories, and they are not the same "
                "thing: failed_names is a control that broke; documented_failures is a gate "
                "marked expected_to_fail, which records a KNOWN defect so the evidence travels "
                "with the result (S-2's dead manipulation, S-3's fitted threshold); "
                "unexpected_passes is a documented defect that has started passing, which means "
                "somebody fixed it without updating the gate and is the one nobody would "
                "otherwise notice. A skipped gate counts as passing but is reported separately: "
                "it means an optional dependency was absent, never that a check succeeded."),
        }


def gate_block(report: GateReport | None) -> dict:
    """The block every verdict JSON should carry under the key ``gates``."""
    return (report or GateReport()).to_dict()
