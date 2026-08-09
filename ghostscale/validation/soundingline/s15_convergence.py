"""S-15 — does recovery error shrink with more artifacts by one maker, and what is left?

THE QUESTION, batch four's highest surviving priority. Armstrong & Mindermann prove a policy
cannot be uniquely decomposed into planner and reward, and that observing more environments does
not resolve it; Skalse and Cao show rewards stay partially identifiable in the infinite-data
limit. The project does not dispute the proofs. It disputes that their conditions describe a
human reading a human artifact, and that position is a claim about a CONVERGENCE RATE and a
RESIDUAL — which the theorems constrain and nobody has measured.

THE DESIGN (SPEC §2). Sixty persistent makers, each a value profile from a named six-member
family; fifty artifacts each; recover the profile from n = 1..50 artifacts and plot error
against n. Report the slope AND the asymptote. Then remove the theory's assumptions one at a
time and price them:

    bounded_clean       the theory's case — bounded family, expert reader, clean tier
    bounded_noisy       CURATOR tier, inexpert reader: the corpus-pricing arm (C4)
    unbounded_family    64 random profiles, truth not included: convergent midbrains removed
    wrong_expertise     reader at d = 0.5: the known-transition-model assumption removed
    construction_B      conjunctive-satisfaction emitter: the S-13 discrimination arm (C3)
    shuffled            artifacts permuted across makers: the null

Criteria C1–C4 are pre-registered in ``prereg_v11.py`` and hash-locked before this module ran.
"""
from __future__ import annotations

import json

import numpy as np

from ... import constants as K
from ...config import Config
from ...methods import gates as G
from ...methods import provenance as PROVENANCE
from ...prereg_v11 import (evaluate_c1, evaluate_c2, evaluate_c3, lock_status)
from ...v11 import seed as v11_seed
from ...v11.maker import (Reader, build_maker_world, build_reader, maker_artifacts_A,
                          maker_artifacts_B, posterior_from_logliks, profile_family,
                          profile_loglik_A, profile_loglik_B, random_family, readout)
from . import sl_dir

N_GRID = (1, 2, 3, 5, 8, 12, 20, 30, 50)
N_MAKERS = 60
N_ARTIFACTS = 50
N_STEPS = 24


def _make_makers(family: dict, per_profile: int, rng) -> list[tuple[str, np.ndarray]]:
    makers = []
    for name, w in family.items():
        for i in range(per_profile):
            makers.append((name, np.asarray(w, dtype=float)))
    rng.shuffle(makers)
    return makers


def _arm(world, reader: Reader, makers, tier: int, hypothesis_family: dict,
         construction: str, rng, shuffle: bool = False) -> dict:
    """One arm: every maker's artifacts, cumulative profile posterior, curve over N_GRID."""
    all_arts, truths, names = [], [], []
    for name, w in makers:
        if construction == "A":
            arts, _ = maker_artifacts_A(world, w, tier, N_ARTIFACTS, N_STEPS, rng)
        else:
            arts = maker_artifacts_B(world, w, tier, N_ARTIFACTS, N_STEPS, rng)
        all_arts.append(arts)
        truths.append(w)
        names.append(name)

    if shuffle:
        # The no-oracle null: pool every artifact and deal them back, so each pseudo-maker's
        # evidence is everybody's. Recovery of the pseudo-maker's own profile must collapse.
        pool = np.concatenate(all_arts, axis=0)
        perm = rng.permutation(len(pool))
        pool = pool[perm]
        all_arts = [pool[i * N_ARTIFACTS:(i + 1) * N_ARTIFACTS] for i in range(len(makers))]

    acc_by_n = {n: [] for n in N_GRID}
    l1_by_n = {n: [] for n in N_GRID}
    for arts, w_true, true_name in zip(all_arts, truths, names):
        cum = (profile_loglik_A(reader, arts, tier, hypothesis_family) if construction == "A"
               else profile_loglik_B(reader, arts, tier, hypothesis_family))
        for n in N_GRID:
            post = posterior_from_logliks(cum, n)
            best, l1 = readout(post, hypothesis_family, w_true)
            acc_by_n[n].append(1.0 if best == true_name else 0.0)
            l1_by_n[n].append(l1)
    return {
        "accuracy_by_n": {str(n): float(np.mean(acc_by_n[n])) for n in N_GRID},
        "l1_by_n": {str(n): float(np.mean(l1_by_n[n])) for n in N_GRID},
        "n_makers": len(makers), "tier": K.PROVENANCE_NAMES[tier],
        "construction": construction,
    }


def run(cfg: Config, n_obs: int | None = None) -> dict:
    """``n_obs`` is accepted for runner uniformity and ignored: the design is fixed by the
    pre-registration (60 makers x 50 artifacts), and shrinking it would change what the
    committed criteria mean."""
    world = build_maker_world(cfg)
    fam = profile_family(cfg.cardinalities.num_goals)
    rng = np.random.default_rng(v11_seed("s15"))

    expert = build_reader(world, cfg, d=0.0)
    inexpert = build_reader(world, cfg, d=0.25,
                            rng=np.random.default_rng(v11_seed("s15-reader-d025")))
    wrong = build_reader(world, cfg, d=0.5,
                         rng=np.random.default_rng(v11_seed("s15-reader-d05")))
    big_family = random_family(64, cfg.cardinalities.num_goals,
                               np.random.default_rng(v11_seed("s15-bigfam")))

    makers = _make_makers(fam, N_MAKERS // len(fam), np.random.default_rng(v11_seed("s15-mk")))

    arms = {}
    arms["bounded_clean"] = _arm(world, expert, makers, K.CREATOR, fam, "A",
                                 np.random.default_rng(v11_seed("s15-clean")))
    arms["bounded_noisy"] = _arm(world, inexpert, makers, K.CURATOR, fam, "A",
                                 np.random.default_rng(v11_seed("s15-noisy")))
    arms["unbounded_family"] = _arm(world, expert, makers, K.CREATOR, big_family, "A",
                                    np.random.default_rng(v11_seed("s15-unb")))
    arms["wrong_expertise"] = _arm(world, wrong, makers, K.CREATOR, fam, "A",
                                   np.random.default_rng(v11_seed("s15-wrong")))
    arms["construction_B"] = _arm(world, expert, makers, K.CREATOR, fam, "B",
                                  np.random.default_rng(v11_seed("s15-B")))
    arms["shuffled"] = _arm(world, expert, makers, K.CREATOR, fam, "A",
                            np.random.default_rng(v11_seed("s15-shuf")), shuffle=True)

    # The all-uniform placebo world for the gate: the estimator must call uniform, uniformly.
    uniform_makers = [("uniform", fam["uniform"])] * 12
    arm_uniform = _arm(world, expert, uniform_makers, K.CREATOR, fam, "A",
                       np.random.default_rng(v11_seed("s15-unifworld")))

    # ---- criteria, from the hash-locked card ---------------------------------------------------
    c1 = evaluate_c1(arms["bounded_clean"]["accuracy_by_n"])
    c2 = evaluate_c2(arms["bounded_clean"]["l1_by_n"]["50"],
                     arms["unbounded_family"]["l1_by_n"]["50"],
                     arms["wrong_expertise"]["l1_by_n"]["50"])
    c3 = evaluate_c3(arms["construction_B"]["accuracy_by_n"]["1"],
                     arms["bounded_clean"]["accuracy_by_n"]["1"])
    noisy_acc = arms["bounded_noisy"]["accuracy_by_n"]
    n_star = next((n for n in N_GRID if noisy_acc[str(n)] >= 0.90), None)
    c4 = {"n_star_accuracy_0.90": n_star, "curve": noisy_acc,
          "how_to_read": ("the smallest number of artifacts by one maker at which the CURATOR-"
                          "tier inexpert reader identifies the profile 90% of the time. This is "
                          "the power analysis for the sibling's follower-corpus design: sourced "
                          "works per maker below this number cannot separate values at this "
                          "noise, whatever the instrument.")}

    # ---- gates ---------------------------------------------------------------------------------
    gr = G.GateReport()
    gr.positive("bounded_clean_converges",
                observed=arms["bounded_clean"]["accuracy_by_n"]["50"], expected=1.0, tol=0.05,
                detail="with the theory's assumptions intact, fifty artifacts must all but "
                       "settle the profile; if the ceiling arm cannot converge, nothing "
                       "downstream means anything.")
    gr.no_oracle("shuffled_makers_read_nothing",
                 observed_change=arms["shuffled"]["accuracy_by_n"]["50"] - 1.0 / len(fam),
                 tol=0.12,
                 detail="artifacts permuted across makers: every pseudo-maker's evidence is the "
                        "population mixture, so profile recovery must sit at chance. If it does "
                        "not, the estimator is reading something other than the evidence.")
    gr.positive("uniform_world_reads_uniform",
                observed=arm_uniform["accuracy_by_n"]["50"], expected=1.0, tol=0.05,
                detail="twelve makers all carrying the uniform profile must all be read as "
                       "uniform: no spurious individuality from noise (noise in, zero out).")
    gr.live("construction_gap_is_real", observed_change=c3["gap"], min_change=0.25,
            detail="the A/B construction switch is the manipulation; if single-artifact "
                   "recovery does not separate the two emitters, the discrimination arm is not "
                   "reaching the measurement.")

    verdict = {
        "test": "S-15 — does recovery error shrink with more artifacts, and at what rate?",
        "for": ("Sounding Line, batch four; the impossibility-literature disagreement stated as "
                "a convergence rate and a residual"),
        "prereg": lock_status(),
        "severity_note": ("runs on the shared V11 maker world — architecture untested pending "
                          "SV-T; treat reproduction rates as unknown, not as high"),
        "design": {"n_makers": N_MAKERS, "n_artifacts": N_ARTIFACTS, "n_steps": N_STEPS,
                   "n_grid": list(N_GRID), "profile_family": sorted(profile_family(4).keys()),
                   "reader": "exact forced-deep (the ceiling); see SPEC §1"},
        "arms": arms,
        "uniform_placebo_world": arm_uniform,
        "criteria": {"C1_convergence": c1, "C2_assumption_price": c2,
                     "C3_construction_gap": c3, "C4_corpus_price": c4},
        "what_would_have_falsified_the_claim": (
            "a flat accuracy curve in bounded_clean (the theorems bite at practical n), or a "
            "bounded-family asymptote no better than the unbounded one (the convergent-midbrains "
            "assumption buys nothing)."),
        "what_must_hold_in_the_real_environment": (
            "a maker's profile is stable on the timescale of the works sampled; the reader's "
            "channel family overlaps the maker's (the convergent-midbrains premise — priced, not "
            "assumed, by the unbounded arm); and artifacts reach the reader with tier-like "
            "transmission rather than adversarial rewriting."),
        "what_this_cannot_show": (
            "that human values are profiles over four channels, or that any real corpus behaves "
            "like construction A or B. The A/B discrimination borrows its real-world force from "
            "measurements the sibling has already made (single-artifact failures, multi-work "
            "successes, no stacking acceleration), and only travels as far as those do."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "s15_convergence.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
