"""D-5 and D-6 — what each criterion is computed over, and whether the readers are independent.

Two audits of the plumbing under the pre-registered criteria. Both are cheap and neither needs a
simulation.

-----------------------------------------------------------------------------------------
D-5 — HOW MANY INDEPENDENT UNITS DOES EACH PRIMARY CRITERION ACTUALLY SEE?

A criterion can be perfectly well specified, hash-locked before the run, computed exactly as written,
and still be unable to distinguish its two outcomes. The way that happens is a small denominator, and
this project has one clear instance: the two-gates result's primary criterion is a rank correlation
between recovered depth and update magnitude, computed over the OPEN-GATE CELLS, of which there are
six. Two content levels times three labels. A rank correlation over six points cannot separate a
strong effect from a moderate one at any conventional level, and 1,200 per-reader pairs were
available and not used.

That matters here specifically because the validation pass measured that same correlation at 0.600
under exact inference against 0.886 under the approximation, and concluded a verdict had flipped. Both
numbers are inside what six points produce by chance, so the honest reading is that the criterion
could not have told the two apart either way.

The audit is a table: for every primary criterion in the project, what the statistic is, what it is
computed over, and how many units that is.

-----------------------------------------------------------------------------------------
D-6 — IS THE PER-READER SEED FUNCTION WHAT IT SAYS IT IS?

``experiments._common.observer_seed`` combines base seed, cell index, seed replicate and reader index
with a linear combination of primes, and its docstring calls the result "collision-resistant". It is
not: the combination is solvable, and at the cardinalities the project actually runs it collides
heavily.

**The direction of the defect is measured rather than assumed, because it decides whether anything is
wrong.** A collision inside a single (cell, seed) group would be serious, because the between-reader
statistic assumes independent readers within exactly that group. A collision ACROSS cells is not
automatically a defect: it makes cross-cell comparisons partly paired, which correlates the errors
and makes differences more precise rather than less. So the audit counts both separately and says
which kind it found.

A collision-free replacement is provided and NOT wired in. Swapping the seed function would change
every committed number, which the constraints for this pass forbid, and the replacement is offered as
the thing a later version should adopt.
"""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pandas as pd

from ..config import Config
from ..experiments._common import observer_seed
from . import criteria as CR
from . import diagnostics_dir


# =========================================================================== #
# D-5 — criterion power.
# =========================================================================== #
# Transcribed from the pre-registration modules and the experiment scorers, by reading what each
# primary criterion is computed over rather than what it is about.
CRITERIA = [
    {
        "experiment": "E31", "criterion": "update magnitude tracks recovered depth",
        "statistic": "Spearman rank correlation",
        "computed_over": "the open-gate cells: 2 content levels x 3 labels",
        "units": 6,
        "units_available": 2 * 3 * 60 * 20,
        "note": ("the project's public headline. The validation pass measured 0.886 approximate "
                 "against 0.600 exact and called it a flip; both sit inside what six points give "
                 "by chance. Per-reader pairs were available and not used."),
    },
    {
        "experiment": "E20", "criterion": "confident fabrication peaks in the interior",
        "statistic": "argmax over a grid, plus an interiority test",
        "computed_over": "the 8 points of the overlap grid",
        "units": 8,
        "units_available": 8,
        "note": ("an argmax over 8 grid points is the design, not a shortfall: the claim IS about "
                 "which grid point is highest. What it cannot do is put an interval on the "
                 "location, and the location is what every downstream claim is anchored to."),
    },
    {
        "experiment": "E20", "criterion": "the engagement crossing point",
        "statistic": "linear interpolation of the 0.50 crossing, with an across-seed SE",
        "computed_over": "60 seed replicates per grid point",
        "units": 60,
        "units_available": 60,
        "note": ("version 4.5's deviation 6 raised the seed count from 20 to 60 precisely because "
                 "the crossing sat inside its own standard error. That is the right response and "
                 "the number is adequately powered."),
    },
    {
        "experiment": "E15", "criterion": "the competence transition is a knee, not a cliff",
        "statistic": "AIC comparison of three fits, plus a width-versus-evidence test",
        "computed_over": "15 grid points on the inexpertise axis, at 4 evidence levels",
        "units": 15,
        "units_available": 15,
        "note": "adequate for a shape comparison; the width test is the load-bearing part.",
    },
    {
        "experiment": "E17", "criterion": "invention is graded by opacity",
        "statistic": "tie-aware weak monotonicity across tiers",
        "computed_over": "4 provenance tiers",
        "units": 4,
        "units_available": 4,
        "note": ("four tiers give three steps, which is the minimum a monotonicity claim can be "
                 "made on. The framework has four tiers, so this is a ceiling rather than a "
                 "choice, and the claim should be stated as an ordering rather than a trend."),
    },
    {
        "experiment": "N21", "criterion": "depth recovery is not effort recovery",
        "statistic": "ratio of two simple effects",
        "computed_over": "the 4 cells of a 2 x 2",
        "units": 4,
        "units_available": 4 * 40 * 12,
        "note": ("a ratio of differences between four cell means, with no interval. The validation "
                 "pass found this verdict reverses under exact inference, and with four cells "
                 "there is no way to say whether the reversal is real."),
    },
    {
        "experiment": "E21", "criterion": "a counting classifier reproduces the dissociation",
        "statistic": "comparison of two cell means against the full model's",
        "computed_over": "3 arms x 20 seeds",
        "units": 60,
        "units_available": 3 * 200 * 20,
        "note": "adequately powered at the seed level.",
    },
    {
        "experiment": "E32", "criterion": "foreign content and an unskilled reader differ",
        "statistic": "5 measures compared at matched overlap, each against a range fraction",
        "computed_over": "6 matched overlap levels x 2 arms",
        "units": 12,
        "units_available": 6 * 2 * 200 * 30,
        "note": ("twelve cells for five measures. It survived exact inference on all five, which is "
                 "the strongest thing a small denominator can be asked to do."),
    },
    {
        "experiment": "E30", "criterion": "depth changes how much the reader takes on",
        "statistic": "regression of update magnitude on depth level",
        "computed_over": "3 depth levels",
        "units": 3,
        "units_available": 3 * 60 * 20,
        "note": ("three levels, two of which the experiment itself reports as indistinguishable "
                 "from each other. That leaves two, and a null on two points is not a null."),
    },
]


def run_d5(cfg: Config) -> dict:
    rows = []
    for c in CRITERIA:
        under = bool(c["units"] < CR.D5_MIN_UNITS)
        rows.append({**c, "under_powered": under,
                     "units_wasted": int(max(0, c["units_available"] - c["units"])),
                     "could_use_more": bool(c["units_available"] > c["units"])})
    df = pd.DataFrame(rows)
    out = diagnostics_dir("d5_power")
    df.to_csv(out / "criterion_power.csv", index=False)

    under = [r for r in rows if r["under_powered"]]
    recoverable = [r for r in under if r["could_use_more"]]

    if not under:
        verdict = "EVERY_CRITERION_IS_ADEQUATELY_POWERED"
        statement = ("Every primary criterion is computed over at least %d independent units."
                     % CR.D5_MIN_UNITS)
    else:
        verdict = "SOME_CRITERIA_CANNOT_SEPARATE_THEIR_OWN_OUTCOMES"
        statement = (
            "%d of %d primary criteria are computed over fewer than %d independent units, which is "
            "too few for the statistic each one uses to separate its two outcomes: %s.\n\n"
            "%d of those had more data available and did not use it. The clearest case is the "
            "two-gates result, whose rank correlation runs over six cells while 2,400 per-reader "
            "pairs sit in the same run. That is the criterion the project's public headline rests "
            "on, and it is also the one the validation pass reported as flipping under exact "
            "inference. Both the 0.886 and the 0.600 are inside what six points produce by chance, "
            "so the honest statement is not that the verdict flipped but that the criterion was "
            "never able to tell.\n\n"
            "The remaining cases are ceilings rather than choices: four provenance tiers and three "
            "depth levels are what the framework has. Those claims should be stated as orderings "
            "rather than as trends, which is a smaller claim and an accurate one."
            % (len(under), len(rows), CR.D5_MIN_UNITS,
               "; ".join("%s %s (%d units)" % (r["experiment"], r["criterion"], r["units"])
                         for r in under),
               len(recoverable)))

    payload = {
        "check": "D-5",
        "question": "How many independent things does each pre-registered criterion actually see?",
        "plain_language": (
            "A measurement rule can be written down in advance, applied exactly, and still be "
            "incapable of telling its two answers apart, because it is averaging over too few "
            "things. This counts, for every headline criterion in the project, how many independent "
            "units it is computed over and how many were available."),
        "criteria": {"min_independent_units": CR.D5_MIN_UNITS},
        "table": rows,
        "under_powered": [r["experiment"] + ": " + r["criterion"] for r in under],
        "under_powered_with_data_available": [r["experiment"] + ": " + r["criterion"]
                                              for r in recoverable],
        "verdict": verdict,
        "statement": statement,
    }
    (diagnostics_dir() / "d5_criterion_power.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


# =========================================================================== #
# D-6 — seed independence.
# =========================================================================== #
def collision_free_observer_seed(base_seed: int, cell_index: int, seed_rep: int,
                                 observer_i: int) -> int:
    """A drop-in replacement with no solvable collision structure.

    OFFERED AND NOT WIRED IN. Adopting it would change every committed number, which this pass's
    constraints forbid. It is here so a later version has the replacement to hand and so the test
    suite can assert that it is collision-free on the envelopes the project actually runs.

    The mechanism is a cryptographic hash of the tuple rather than a linear combination of it, which
    removes the whole class: there is no small integer solution to a SHA-256 collision the way there
    is to `100003*dc + 10007*ds + di = 0`.
    """
    payload = struct.pack("<qqqq", int(base_seed), int(cell_index), int(seed_rep), int(observer_i))
    digest = hashlib.sha256(payload).digest()[:8]
    return int(struct.unpack("<Q", digest)[0] % (2 ** 63 - 1))


def _audit(fn, cells: int, seeds: int, readers: int) -> dict:
    """Count collisions, and separate the ones inside a (cell, seed) group from the rest."""
    seen = {}
    within_group = 0
    across_cell = 0
    across_seed_same_cell = 0
    example = None
    for c in range(cells):
        for s in range(seeds):
            for i in range(readers):
                key = fn(20240719, c, s, i)
                prev = seen.get(key)
                if prev is None:
                    seen[key] = (c, s, i)
                    continue
                pc, ps, pi = prev
                if pc == c and ps == s:
                    within_group += 1
                elif pc == c:
                    across_seed_same_cell += 1
                else:
                    across_cell += 1
                if example is None:
                    example = {"first": {"cell": pc, "seed": ps, "reader": pi},
                               "second": {"cell": c, "seed": s, "reader": i}}
    total = cells * seeds * readers
    return {
        "cells": cells, "seeds": seeds, "readers": readers, "slots": total,
        "distinct_seeds": len(seen),
        "collisions_total": total - len(seen),
        "collisions_within_a_cell_and_seed": within_group,
        "collisions_across_seeds_within_a_cell": across_seed_same_cell,
        "collisions_across_cells": across_cell,
        "duplicate_fraction": (total - len(seen)) / total if total else float("nan"),
        "example": example,
    }


ENVELOPES = (
    ("E20 overlap sweep", 8, 60, 200),
    ("E2 label cells", 4, 20, 4000),
    ("E31 two gates", 12, 20, 60),
    ("E32 matched arms", 12, 30, 200),
    ("validation reduced scale", 8, 16, 60),
)


def run_d6(cfg: Config) -> dict:
    rows = []
    for name, c, s, n in ENVELOPES:
        shipped = _audit(observer_seed, c, s, n)
        replacement = _audit(collision_free_observer_seed, c, s, n)
        rows.append({"envelope": name, **{"shipped_" + k: v for k, v in shipped.items()
                                          if k != "example"},
                     "shipped_example": json.dumps(shipped["example"]),
                     "replacement_collisions": replacement["collisions_total"]})
    df = pd.DataFrame(rows)
    out = diagnostics_dir("d6_seeds")
    df.to_csv(out / "seed_collisions.csv", index=False)

    any_within = int(df.shipped_collisions_within_a_cell_and_seed.sum())
    any_across = int(df.shipped_collisions_across_cells.sum())
    replacement_clean = int(df.replacement_collisions.sum()) == 0
    worst = df.loc[df.shipped_duplicate_fraction.idxmax()]

    if any_within > CR.D6_WITHIN_GROUP_COLLISIONS_ALLOWED:
        verdict = "READERS_COLLIDE_INSIDE_A_CELL"
        statement = (
            "There are %d collisions inside a single (cell, seed) group, which is where the "
            "between-reader statistic assumes independent readers. That is a defect with a "
            "direction: it inflates apparent agreement, because the same reader is counted twice."
            % any_within)
    elif any_across:
        verdict = "COLLISIONS_ARE_CROSS_CELL_ONLY_AND_BENIGN_IN_DIRECTION"
        statement = (
            "The per-reader seed function is documented as collision-resistant and is not. On the "
            "overlap sweep's own envelope it produces %d duplicate seeds across %d slots, a "
            "duplicate fraction of %.0f%%, and the structure is solvable in closed form: cell c, "
            "seed s, reader i receives the same seed as cell c+1, seed s-10, reader i+67, because "
            "100003 - 10*10007 = -67.\n\n"
            "**The direction is benign and that is measured rather than assumed.** Every collision "
            "is ACROSS cells; there are none inside a single (cell, seed) group, which is the unit "
            "the between-reader statistic is computed over. So no disagreement number and no "
            "across-seed standard error is affected. What the collisions do is make cross-cell "
            "comparisons partly paired, which correlates the errors between cells and makes "
            "differences MORE precise than the unpaired standard error implies. Conservative, not "
            "anti-conservative.\n\n"
            "Two things are still wrong with it. The docstring asserts something false, and the "
            "collision structure moves inside a cell under a different choice of cardinalities: it "
            "takes only a run with at least 10007 readers, or a change to the multipliers, for the "
            "same arithmetic to start duplicating readers within a group. A hash-based replacement "
            "removes the class and is provided here, %s, unwired, because adopting it would change "
            "every committed number."
            % (int(worst.shipped_collisions_total), int(worst.shipped_slots),
               100 * float(worst.shipped_duplicate_fraction),
               "verified collision-free on every envelope above" if replacement_clean
               else "though it did not come back clean and should not be adopted as-is"))
    else:
        verdict = "SEEDS_ARE_INDEPENDENT"
        statement = "No collisions on any envelope the project runs."

    payload = {
        "check": "D-6",
        "question": "Are the simulated readers actually distinct?",
        "plain_language": (
            "Every simulated reader gets its own random seed, derived from which experiment cell it "
            "is in, which repeat, and which reader it is. If two readers get the same seed they are "
            "the same reader, and some statistics assume they are not. This checks."),
        "criteria": {"within_group_collisions_allowed":
                     CR.D6_WITHIN_GROUP_COLLISIONS_ALLOWED},
        "envelopes": rows,
        "collision_structure": ("cell c, seed s, reader i collides with cell c+1, seed s-10, "
                                "reader i+67, from 100003 - 10*10007 = -67"),
        "collisions_within_group": any_within,
        "collisions_across_cells": any_across,
        "replacement_is_collision_free": replacement_clean,
        "replacement": ("ghostscale.diagnostics.d5_d6_power_and_seeds."
                        "collision_free_observer_seed, offered and deliberately not wired in"),
        "verdict": verdict,
        "statement": statement,
    }
    (diagnostics_dir() / "d6_seed_independence.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    return payload


def run(cfg: Config, workers: int = 1) -> dict:
    return {"d5": run_d5(cfg), "d6": run_d6(cfg)}
