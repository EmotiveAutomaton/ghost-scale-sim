"""S-14 — the motivational aperture: is an ABSENT drive recoverable from the work?

THE CLAIM, batch four's S-14, in the curator's example: forced to design something abhorrent, he
could tap a need for efficiency but not the cruelty a willing designer would have — it would not
be there to optimise. You can only route attention onto drives you possess, so the drive a maker
LACKS constrains what they produce, and the absence is as informative as a presence. The only
proposal in the program that treats an absence as a measurable, and a candidate mechanism for
work that reads as made-under-duress.

THE CONSTRUCTION (SPEC §§1, 3). Instruction amplifies MULTIPLICATIVELY — w' ∝ w·exp(A·e_k) — so
an absent channel (w[k] = 0) cannot be amplified and the maker routes around it, while a
present-but-unused trace (w[k] = ε) is amplified to dominance. Commissioned artifacts carry a
compliance channel (λ·sig[k]) both maker types deliver identically; the difference lives ONLY in
how the remainder is pursued. Three maker kinds per target channel:

    masked    w[k] = 0,    rest uniform     the aperture
    unused    w[k] = 0.02, rest uniform     the trace that instruction can amplify
    present   w[k] = 0.70                   the easy known answer, for the positive gate

TWO EVIDENCE REGIMES, because the prediction is a CONTRAST (C5): in spontaneous standing work
the ε-trace is nearly invisible, so masked and unused read almost alike; under commission the
amplification separates them. The aperture is revealed by commission, not by observation. And
the λ=1 arm (pure compliance, no pursuit channel) must collapse to chance (C6) — the
discriminator is the how-channel and nothing else.
"""
from __future__ import annotations

import json

import numpy as np

from ... import constants as K
from ...config import Config
from ...methods import gates as G
from ...methods import provenance as PROVENANCE
from ...prereg_v11 import evaluate_c5, evaluate_c6, lock_status
from ...v11 import seed as v11_seed
from ...v11.maker import (build_maker_world, build_reader, commissioned_emission, draw_artifact,
                          poe, tier_mix)
from . import sl_dir

EPS_UNUSED = 0.02
A_AMP = 4.0
LAM = 0.5
N_PER_KIND = 20
N_ARTIFACTS = 12
N_STEPS = 24
TIER = None  # set to K.CREATOR at run time; module constant kept symbolic for the docstring


def _profiles(k: int, ng: int) -> dict[str, np.ndarray]:
    masked = np.full(ng, 1.0 / (ng - 1))
    masked[k] = 0.0
    unused = np.full(ng, (1.0 - EPS_UNUSED) / (ng - 1))
    unused[k] = EPS_UNUSED
    present = np.full(ng, 0.10)
    present[k] = 0.70
    return {"masked": masked, "unused": unused, "present": present}


def _emission(world, w, k, regime: str, lam: float):
    if regime == "commissioned":
        return commissioned_emission(world, w, k, K.CREATOR, A_AMP, lam)
    # Spontaneous standing work: the conjunctive emission of the standing profile. The ε-trace
    # enters only through an exponent of ε, which is the construction of near-invisibility.
    return tier_mix(world, poe(world.sig, w), K.CREATOR)


def _reader_dist(reader, w_hyp, k, regime: str, lam: float):
    if regime == "commissioned":
        from ...v11.maker import amplify
        how = poe(reader.sig_r, amplify(w_hyp, k, A_AMP))
        dist = lam * reader.sig_r[k] + (1.0 - lam) * how
    else:
        dist = poe(reader.sig_r, w_hyp)
    a = float(reader.alpha[K.CREATOR])
    out = a * dist + (1.0 - a) * reader.synth
    return out / out.sum()


def _discriminate(world, reader, k, kinds: tuple[str, str], regime: str, lam: float,
                  rng) -> float:
    """Accuracy of the two-hypothesis exact reader over N_PER_KIND makers of each kind."""
    profs = _profiles(k, world.sig.shape[0])
    hyp = {name: np.log(np.maximum(_reader_dist(reader, profs[name], k, regime, lam), 1e-300))
           for name in kinds}
    correct = 0
    for truth in kinds:
        dist = _emission(world, profs[truth], k, regime, lam)
        for _ in range(N_PER_KIND):
            arts = np.stack([draw_artifact(dist, N_STEPS, rng) for _ in range(N_ARTIFACTS)])
            scores = {name: float(hyp[name][arts].sum()) for name in kinds}
            if max(scores, key=scores.get) == truth:
                correct += 1
    return correct / (2.0 * N_PER_KIND)


def run(cfg: Config, n_obs: int | None = None) -> dict:
    """``n_obs`` accepted for runner uniformity and ignored; the design is pre-registered."""
    world = build_maker_world(cfg)
    reader = build_reader(world, cfg, d=0.0)
    ng = world.sig.shape[0]

    by_k = {}
    for k in range(ng):
        rng = np.random.default_rng(v11_seed(f"s14-k{k}"))
        by_k[k] = {
            "commissioned": _discriminate(world, reader, k, ("masked", "unused"),
                                          "commissioned", LAM, rng),
            "spontaneous": _discriminate(world, reader, k, ("masked", "unused"),
                                         "spontaneous", LAM, rng),
            "commissioned_lambda1": _discriminate(world, reader, k, ("masked", "unused"),
                                                  "commissioned", 1.0, rng),
            "present_vs_masked_commissioned": _discriminate(
                world, reader, k, ("masked", "present"), "commissioned", LAM, rng),
        }

    mean = {stat: float(np.mean([by_k[k][stat] for k in by_k]))
            for stat in next(iter(by_k.values()))}
    c5 = evaluate_c5(mean["commissioned"], mean["spontaneous"])
    c6 = evaluate_c6(mean["commissioned_lambda1"])

    gr = G.GateReport()
    gr.positive("present_vs_masked_is_easy",
                observed=mean["present_vs_masked_commissioned"], expected=1.0, tol=0.05,
                detail="a drive at 0.70 against a drive at zero, under commission, is the known "
                       "answer by construction; an instrument that cannot see THAT sees nothing.")
    gr.placebo("pure_compliance_reads_nothing",
               observed_max_deviation=mean["commissioned_lambda1"] - 0.5, tol=0.10,
               detail="at lambda = 1 the pursuit channel is removed and both maker kinds emit "
                      "identically; discrimination must collapse to chance, or the reader is "
                      "reading something other than how the goal is pursued.")
    gr.live("commission_reveals_the_aperture",
            observed_change=mean["commissioned"] - mean["spontaneous"], min_change=0.20,
            detail="the commission is the manipulation: amplification separates a zero from an "
                   "epsilon. If commissioned and spontaneous discrimination do not separate, the "
                   "mechanism is not reaching the measurement.")

    verdict = {
        "test": "S-14 — the motivational aperture: is an absent drive recoverable?",
        "for": "Sounding Line, batch four; the only absence-as-measurable proposal",
        "prereg": lock_status(),
        "severity_note": ("runs on the shared V11 maker world — architecture untested pending "
                          "SV-T"),
        "design": {"epsilon_unused": EPS_UNUSED, "amplification": A_AMP,
                   "compliance_lambda": LAM, "n_per_kind": N_PER_KIND,
                   "n_artifacts": N_ARTIFACTS, "n_steps": N_STEPS,
                   "mechanism": "w' ∝ w·exp(A·e_k): attention can only amplify what exists"},
        "by_target_channel": by_k,
        "means": mean,
        "criteria": {"C5_aperture": c5, "C6_how_channel": c6},
        "what_would_have_falsified_the_claim": (
            "commissioned discrimination at chance (the absence leaves no readable mark even "
            "under amplification), or spontaneous discrimination as good as commissioned (the "
            "aperture is visible without instruction, and the amplification story is not what "
            "carries it)."),
        "what_must_hold_in_the_real_environment": (
            "commissioned work with a known brief exists and is identifiable; instruction acts "
            "multiplicatively on standing drives (the theory's own amplification mechanism); "
            "and compliance and pursuit are at least partly separable in real artifacts. The "
            "made-under-duress reading additionally needs real briefs to demand surfaces their "
            "makers' drives cannot supply."),
        "what_this_cannot_show": (
            "that any real maker's drive set is expressible as a profile with hard zeros, or "
            "that real commissions amplify rather than replace. The mechanism is the theory's; "
            "this establishes it is COHERENT and measurable in a world that has it, not that "
            "the world outside does."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "s14_aperture.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
