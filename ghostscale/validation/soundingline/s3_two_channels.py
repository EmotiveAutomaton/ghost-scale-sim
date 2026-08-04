"""S-3 — do two affect layers produce a measurable divergence, and does the shield give it away?

**THIS ONE DOES NOT USE THE V5 WORLD, AND THAT IS DELIBERATE.** Every other test here reuses the
simulation's own creator and reader because the question is about them. This question is not: it
asks whether a two-channel emission -- one involuntary, one chosen -- can be inverted at all. The
V5 creator has a single emission channel, so answering this inside it would mean adding a second
one to the shared model, which is new generative machinery in a repository that is otherwise
closed. A minimal purpose-built construction answers the mechanism question without touching it.

Read the result as: *this is what a two-channel emitter would do*. Not: *this is what the Ghost
Scale reader does*.

THE CONSTRUCTION. One creator state s, two emissions over one alphabet:

    leak     drawn from L(.|s). Not under the creator's control.
    display  drawn from D(.|s, condition).
                 candid      D tracks the leak's own tendency
                 concealing  D is pushed AWAY from the leak, and AMPLIFIED

The curator's correction is the part that matters, and it is counter-intuitive:

    Leaked greater than emblematic doesn't even count as concealment. If anything the emblematic
    would get LARGER. You perform louder to cover up. THE SHIELD MATCHES THE LEAK.

So concealment is modelled as opposition WITH amplification, and the sharp prediction is that
amplifying the shield makes concealment MORE detectable, not less -- because a louder display is a
more precise readout of the thing it is opposing.

THREE QUESTIONS: can the state be recovered from the leak alone; does channel divergence identify
the concealing condition; and does amplification help or hurt the concealer.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ...config import Config
from ...prereg_v6 import BOOTSTRAP_DRAWS, percentile_interval
from ...v6 import SEED_OFFSET
from . import sl_dir

_EPS = 1e-12
N_STATES = 4
N_SYMBOLS = 8
AMPLIFICATIONS = (1.0, 2.0, 4.0, 8.0)


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=float) / max(temp, _EPS)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _build(rng) -> tuple:
    """One affinity matrix per state, shared by both channels: the same body, two behaviours."""
    aff = rng.normal(size=(N_STATES, N_SYMBOLS))
    leak = _softmax(aff, temp=1.0)
    return aff, leak


def _display(aff: np.ndarray, s: int, condition: str, amp: float) -> np.ndarray:
    if condition == "candid":
        return _softmax(aff[s] * amp)
    # Concealing: push away from what the state would leak, and perform it louder.
    return _softmax(-aff[s] * amp)


def _posterior(obs: np.ndarray, like: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(like, _EPS, None))[:, obs].sum(axis=1)
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def run(cfg: Config, n_obs: int = 400, n_emissions: int = 12) -> dict:
    rng = np.random.default_rng(SEED_OFFSET + 90_300)
    aff, leak = _build(rng)
    rows = []

    for condition in ("candid", "concealing"):
        for amp in AMPLIFICATIONS:
            disp = np.stack([_display(aff, s, condition, amp) for s in range(N_STATES)])
            for i in range(int(n_obs)):
                s = int(rng.integers(N_STATES))
                lk = rng.choice(N_SYMBOLS, size=n_emissions, p=leak[s])
                dp = rng.choice(N_SYMBOLS, size=n_emissions, p=disp[s])

                post_leak = _posterior(lk, leak)
                post_disp = _posterior(dp, leak)          # read the display AS IF it were a leak
                # Divergence between what the two channels say about the state. A reader has both.
                div = float(np.sum(post_leak * np.log(np.clip(post_leak, _EPS, None)
                                                      / np.clip(post_disp, _EPS, None))))
                rows.append({
                    "condition": condition, "amp": amp, "i": i, "true_state": s,
                    "leak_correct": int(int(np.argmax(post_leak)) == s),
                    "display_correct": int(int(np.argmax(post_disp)) == s),
                    "divergence": div,
                })

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "s3_two_channels.csv", index=False)

    leak_readable = float(df.leak_correct.mean())
    chance = 1.0 / N_STATES

    by_amp = {}
    for amp in AMPLIFICATIONS:
        a = df[(df.condition == "concealing") & (df.amp == amp)].divergence.to_numpy()
        b = df[(df.condition == "candid") & (df.amp == amp)].divergence.to_numpy()
        draws = [float(np.mean(rng.choice(a, a.size, replace=True))
                       - np.mean(rng.choice(b, b.size, replace=True)))
                 for _ in range(BOOTSTRAP_DRAWS)]
        lo, hi = percentile_interval(draws)
        # A single-threshold detector on divergence, scored as balanced accuracy.
        thr = float(np.median(np.concatenate([a, b])))
        acc = 0.5 * (float((a > thr).mean()) + float((b <= thr).mean()))
        by_amp[float(amp)] = {"concealing": float(a.mean()), "candid": float(b.mean()),
                              "separation": float(a.mean() - b.mean()), "interval": [lo, hi],
                              "excludes_zero": bool(np.isfinite(lo) and lo > 0),
                              "detector_balanced_accuracy": acc}

    seps = [by_amp[a]["separation"] for a in AMPLIFICATIONS]
    accs = [by_amp[a]["detector_balanced_accuracy"] for a in AMPLIFICATIONS]
    louder_is_worse = bool(accs[-1] > accs[0])

    verdict = {
        "test": "S-3 — two affect layers, and whether a louder shield gives the concealer away",
        "for": "Sounding Line, family_v3.yaml leaked and emblematic channels",
        "IMPORTANT": (
            "this test does NOT use the V5 world. It is a minimal purpose-built two-channel "
            "emitter, because the shared model has one emission channel and adding a second is "
            "new generative machinery. Read it as what a two-channel emitter does, not as what "
            "the Ghost Scale reader does."),
        "construction": {"n_states": N_STATES, "n_symbols": N_SYMBOLS,
                         "emissions_per_artifact": int(n_emissions),
                         "amplifications": list(AMPLIFICATIONS),
                         "concealment": "display pushed AWAY from the leak and amplified"},
        "is_the_leak_readable_at_all": {
            "accuracy": leak_readable, "chance": chance,
            "verdict": ("LEAK_IS_READABLE" if leak_readable > chance + 0.10
                        else "LEAK_IS_NOT_READABLE_IN_PRINCIPLE")},
        "divergence_by_amplification": by_amp,
        "does_a_louder_shield_help_the_concealer": {
            "balanced_accuracy_at_lowest_amplification": accs[0],
            "balanced_accuracy_at_highest": accs[-1],
            "separation_at_lowest": seps[0], "separation_at_highest": seps[-1],
            "verdict": ("THE_SHIELD_GIVES_THEM_AWAY" if louder_is_worse
                        else "AMPLIFYING_THE_SHIELD_HELPS_THE_CONCEALER"),
            "how_to_read": (
                "the curator's claim is that performing louder to cover up makes concealment MORE "
                "detectable. That predicts detector accuracy RISING with amplification.")},
        "what_would_have_falsified_it": (
            "divergence failing to separate the two conditions, or separation shrinking as the "
            "shield gets louder. Either would make the leaked layer a dead end."),
    }
    (sl_dir() / "s3_two_channels.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
