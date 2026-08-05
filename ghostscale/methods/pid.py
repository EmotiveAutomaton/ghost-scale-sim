"""Partial information decomposition: what the artifact carries about each latent, and jointly.

WHY THIS IS HERE AND NOT IN AN EXPERIMENT. T-1 asked whether the goal-process-depth structure is a
chain or a triangle by SUPPLYING each vertex to a reader and measuring recovery at another. That
is a question about the reader. There is a prior question about the WORLD -- what a single emission
carries about each latent at all -- and it has an exact answer that needs no rollouts, no reader
and no sampling, because ``world.subsig`` IS the emission likelihood.

T-1's superadditivity test (does supplying two vertices beat the sum of supplying each?) is a
hand-rolled synergy measure. PID is the principled version, and it decomposes the joint information
into four parts that the pairwise test collapses into one:

    redundant   present in either source alone. Either vertex tells you this.
    unique      present in one source and not the other.
    synergistic present only in the two together. Neither vertex alone tells you any of it.

WHAT IT RETURNED THE FIRST TIME IT WAS RUN, because it is the reason this module exists::

    depth     total  redundant  uniq GOAL  uniq MODE    SYNERGY
    mu=1     1.4521     0.0000     1.4521     0.0000    -0.0000
    mu=2     1.8783     0.2966     1.1555     0.0000     0.4262
    mu=3     1.8783     0.2706     1.1816     0.0000     0.4262

Two things fall out. At mu = 1 it returns exactly zero mode information and exactly zero synergy,
which is null N28's design guarantee recovered independently from the likelihood alone -- so this
doubles as an ``identity`` gate. And unique mode information is EXACTLY ZERO at every depth:
everything the execution mode contributes to an emission is either redundant with the goal or
readable only jointly with it. That is a structural statement about why T-1's edges came out the
way they did, and no measure previously in this repository could produce it.

DEGRADES GRACEFULLY. Without ``dit`` installed this returns a recorded skip. It is a diagnostic,
not a finding, and no published number depends on it.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-15


def available() -> tuple:
    """``(ok, reason)``. Import is attempted once, here, so callers never see an ImportError."""
    try:
        import dit                                       # noqa: F401
        return True, ""
    except Exception as exc:                             # noqa: BLE001
        return False, f"dit not installed ({type(exc).__name__})"


def decompose(joint_probs: np.ndarray) -> dict:
    """PID of two discrete sources about one discrete target.

    ``joint_probs`` is ``p[source_a, source_b, target]`` and need not be normalised.

    Returns bits, and the atom names are spelled out rather than left as dit's lattice keys --
    ``((0,), (1,))`` meaning "redundant" is not something a reader should have to look up.
    """
    ok, reason = available()
    if not ok:
        return {"skipped": reason}
    import dit
    from dit.pid import PID_WB

    p = np.asarray(joint_probs, dtype=float)
    if p.ndim != 3:
        raise ValueError(f"expected p[a, b, target], got shape {p.shape}")
    p = p / p.sum()
    outcomes, probs = [], []
    for a in range(p.shape[0]):
        for b in range(p.shape[1]):
            for t in range(p.shape[2]):
                if p[a, b, t] > _EPS:
                    outcomes.append((a, b, t))
                    probs.append(float(p[a, b, t]))
    if not outcomes:
        return {"skipped": "joint distribution is empty"}
    d = dit.Distribution(outcomes, probs)
    atoms = PID_WB(d, [[0], [1]], [2])._pis
    red = float(atoms[((0,), (1,))])
    ua = float(atoms[((0,),)])
    ub = float(atoms[((1,),)])
    syn = float(atoms[((0, 1),)])
    return {
        "total_mutual_information_bits": red + ua + ub + syn,
        "redundant_bits": red,
        "unique_source_a_bits": ua,
        "unique_source_b_bits": ub,
        "synergistic_bits": syn,
        "estimator": "PID_WB (Williams-Beer I_min)",
        "how_to_read": (
            "redundant = either source alone tells you this. unique = one source and not the "
            "other. synergistic = only the two together. A source whose UNIQUE term is zero "
            "carries nothing the other source does not also carry or that is not readable only "
            "jointly -- which is a stronger statement than a low mutual information."),
    }


def emission_pid(subsig: np.ndarray, mu_index: int) -> dict:
    """PID of {goal, execution mode} about one emitted feature, at one depth. Exact.

    ``subsig`` is ``world.subsig`` with shape ``(n_mu, n_goals, n_subgoals, n_features)``. Goal and
    mode are given a uniform prior, which is what the world does: ``build_subgoal_chains`` is
    doubly stochastic with a uniform stationary distribution precisely so that no goal spends more
    time in any mode than any other.
    """
    s = np.asarray(subsig, dtype=float)
    _n_mu, ng, n_sub, _nf = s.shape
    block = s[int(mu_index)]                                  # (ng, n_sub, nf)
    block = block / block.sum(axis=-1, keepdims=True)
    joint = block / float(ng * n_sub)                         # uniform over (goal, mode)
    out = decompose(joint)
    out["mu_index"] = int(mu_index)
    return out


def n28_identity_from_pid(subsig: np.ndarray, mu_shallow_index: int = 0,
                          tol: float = 1e-9) -> dict:
    """Null N28, recovered from the likelihood alone.

    At the shallowest depth every execution mode emits the goal signature exactly, so an emission
    can carry NO information about the mode -- unique and synergistic terms must both be zero.
    That is checkable without running a reader, and it is the cheapest positive control in the
    repository: it validates the depth construction and the PID wiring in one call.
    """
    r = emission_pid(subsig, mu_shallow_index)
    if "skipped" in r:
        return r
    worst = max(abs(r["unique_source_b_bits"]), abs(r["synergistic_bits"]))
    return {
        "unique_mode_bits": r["unique_source_b_bits"],
        "synergistic_bits": r["synergistic_bits"],
        "worst_abs_deviation": float(worst),
        "tolerance": float(tol),
        "holds": bool(worst <= tol),
        "what_it_checks": (
            "at the shallowest depth every mode emits the goal signature exactly, so an emission "
            "cannot carry mode information. Both the unique-mode and synergistic terms must be "
            "zero. Recovers null N28 from the likelihood alone, with no rollouts."),
    }
