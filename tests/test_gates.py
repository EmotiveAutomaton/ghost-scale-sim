"""Every committed verdict must carry provenance and must pass its own gates.

THIS IS WHERE THE HARD STOP LIVES, and putting it here rather than inside the modules is the whole
design. A gate that raises mid-run kills a two-hundred-second sweep over a control that may have
been expected to fail, which trains people to switch gates off. A gate that only prints gets
ignored, which is how S-2 shipped. So gates RECORD during a run and this file FAILS at the point a
result would become public.

Reads committed JSON rather than re-running anything, so it is fast enough to be part of the
ordinary test cycle. That also means it checks what is actually on disk and about to be pushed,
not what a fresh run would produce -- which is the thing being guarded.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SL = REPO / "results" / "validation" / "soundingline"

#: Modules with gates. A module absent from this set is allowed to have none -- S-1, S-4/5 and S-6
#: audit somebody else's statistic and have no manipulation of their own to gate -- but a module
#: IN this set that loses its gates is a regression, and ``test_gated_modules_still_have_gates``
#: is what catches a retrofit being silently reverted.
GATED = {
    "s2_flattened_intent", "s3_two_channels",
    "t1_triangle", "t2_automaticity", "t3_countability", "t4_uncertain_reader", "t5_detection",
    "t6_information_budget", "t7_posthoc", "t8_multivariate_detection", "t9_concentration",
    "t10_boundary_recovery", "s11_component_count",
}


def verdicts():
    if not SL.exists():
        return []
    out = []
    for p in sorted(SL.glob("*.json")):
        if p.name == "summary.json":
            continue
        try:
            out.append((p.stem, json.loads(p.read_text(encoding="utf-8"))))
        except json.JSONDecodeError:
            out.append((p.stem, None))
    return out


ALL = verdicts()
pytestmark = pytest.mark.skipif(not ALL, reason="no soundingline verdicts on disk yet")


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_verdict_is_valid_json(name, v):
    assert v is not None, f"{name}.json does not parse"


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_verdict_carries_provenance(name, v):
    """Which module, at which content, produced this."""
    if v is None:
        pytest.skip("unparseable")
    p = v.get("produced_by")
    assert p, f"{name} has no produced_by block; add methods.provenance.stamp(...)"
    assert p.get("module", "").endswith(".py"), f"{name} produced_by.module is not a file"
    assert p.get("sha256"), f"{name} produced_by has no source hash"


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_gated_modules_still_have_gates(name, v):
    if v is None:
        pytest.skip("unparseable")
    if name not in GATED:
        pytest.skip("module has no manipulation of its own to gate")
    g = v.get("gates")
    assert g, f"{name} is in GATED but its verdict has no gates block"
    assert g["n_ran"] >= 1, f"{name} recorded gates but ran none of them"


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_no_control_broke(name, v):
    """A gate that was supposed to pass and did not. This is the one that stops a release."""
    if v is None or not v.get("gates"):
        pytest.skip("no gates")
    g = v["gates"]
    failed = g.get("failed_names") or []
    if failed:
        detail = "\n".join(
            f"    {gg['name']} ({gg['kind']}): observed={gg.get('observed')} "
            f"expected={gg.get('expected')} tol={gg.get('tolerance')}\n      {gg['detail']}"
            for gg in g["gates"] if gg["name"] in failed)
        pytest.fail(f"{name}: {len(failed)} control(s) failed\n{detail}")


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_no_documented_defect_started_passing(name, v):
    """A gate marked expected_to_fail that now passes.

    Nobody would otherwise notice this, and it means one of two things, both of which need a
    human: the underlying defect was fixed and the gate was not updated, or the gate stopped
    measuring what it was pointed at. S-2's dead manipulation passing would mean the environment
    changed under it.
    """
    if v is None or not v.get("gates"):
        pytest.skip("no gates")
    surprises = v["gates"].get("unexpected_passes") or []
    assert not surprises, (
        f"{name}: gate(s) {surprises} are marked expected_to_fail but PASSED. Either the defect "
        f"they document was fixed (update the gate and the module's withdrawal note) or the gate "
        f"has stopped measuring what it was aimed at.")


@pytest.mark.parametrize("name,v", ALL, ids=[n for n, _ in ALL])
def test_documented_failures_are_explained(name, v):
    """A module carrying a known defect has to say so in the verdict, not only in a gate."""
    if v is None or not v.get("gates"):
        pytest.skip("no gates")
    documented = v["gates"].get("documented_failures") or []
    if not documented:
        return
    assert v.get("withdrawn") is True or v.get("QUALIFIED") or v.get("WITHDRAWN"), (
        f"{name} has documented gate failures {documented} but the verdict has no `withdrawn` "
        f"flag, `WITHDRAWN` note or `QUALIFIED` note. A known defect must be legible to someone "
        f"reading the top of the file, not only to someone who scrolls to the gates block.")


def test_no_oversized_committed_csv():
    """The naming convention IS the size policy; this is what enforces it.

    Per-rollout data goes to ``*_points.csv``, which .gitignore excludes. The T-series originally
    shipped its per-rollout data under bare names and committed 17.5 MB to a repository whose
    entire library is 2 MB. Anything large enough to be per-rollout data and not named for it is
    almost certainly the same mistake again.
    """
    if not SL.exists():
        pytest.skip("no soundingline results yet")
    big = [(p.name, p.stat().st_size)
           for p in SL.glob("*.csv")
           if not p.name.endswith("_points.csv") and p.stat().st_size > 2_000_000]
    assert not big, (
        "committed CSVs over 2 MB that are not named *_points.csv: "
        + ", ".join(f"{n} ({s/1048576:.1f} MB)" for n, s in big)
        + ". Per-rollout data belongs in *_points.csv (gitignored); commit a *_summary.csv.")
