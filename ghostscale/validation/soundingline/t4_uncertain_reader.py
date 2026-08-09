"""T-4 — does leaked/emblematic divergence survive a reader that does not know the channel?

S-3 found the two-channel emitter works: state recovers from the leak at 0.899, divergence
identifies concealment, and amplifying the shield makes concealment MORE detectable. All of that
was measured with a reader holding the world's exact emission model.

WHY THIS MATTERS: Sounding Line's version reads a language model, which has no correct prior over
a human's emission channels at all. If divergence collapses under mild reader uncertainty the
whole leaked-layer programme is a simulation result that cannot be transported.

**A PROBLEM IN S-3 ITSELF, FOUND WHILE READING IT, AND FIXED HERE FIRST.** S-3's detector sets its
threshold as

    thr = float(np.median(np.concatenate([a, b])))

where ``a`` is the concealing divergences and ``b`` the candid ones, at the amplification being
scored. That threshold is fitted on ground-truth-labelled data from both conditions, and re-fitted
at every amplification. No reader can do that. So ``detector_balanced_accuracy`` is an upper
bound, and the headline -- accuracy RISING 0.845 -> 0.9625 with amplification -- is partly a
threshold that moves with the data it is being scored on. ``frozen`` re-scores the whole sweep
with the threshold fixed at the lowest amplification, which is what a reader who calibrated once
and then met a louder concealer would actually have.

FOUR DEGRADATIONS, and one construction change.

  ``prior_noise``   the reader's emission model is mixed toward uniform. It knows roughly what
                    states look like and not exactly.
  ``channel_swap``  with probability p the reader has the two channels the wrong way round, so
                    it reads the performance as the leak.
  ``no_concealment_model``  already the baseline: the detector is a bare threshold on divergence
                    with no representation of concealment. Kept explicit so the report can say so.
  ``wrong_cardinality``  the reader models the wrong number of states.

  PARTIAL CONCEALMENT. S-3's concealer emits ``softmax(-aff * amp)`` -- an EXACT negation, which
  is the most detectable concealment there is, because a perfect opposite carries as much
  information as a perfect copy. Real concealment is partial. ``theta`` rotates the display away
  from the leak by a controlled fraction, and the honest question is where along that axis the
  divergence signal dies.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ...config import Config
from ...v6 import SEED_OFFSET
from ...methods import gates as G
from ...methods import provenance as PROV
from . import sl_dir
from . import t_common as T

_EPS = 1e-12
N_STATES = 4
N_SYMBOLS = 8
AMPLIFICATIONS = (1.0, 2.0, 4.0, 8.0)
#: 1.0 is S-3's exact negation; 0.0 is candid. Everything between is partial concealment.
THETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=float) / max(temp, _EPS)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _build(rng):
    aff = rng.normal(size=(N_STATES, N_SYMBOLS))
    return aff, _softmax(aff, temp=1.0)


def _display(aff: np.ndarray, s: int, theta: float, amp: float) -> np.ndarray:
    """Push the display away from the leak by ``theta``. theta = 0 candid, 1 = S-3's negation."""
    return _softmax(aff[s] * (1.0 - 2.0 * float(theta)) * amp)


def _degrade(leak: np.ndarray, kind: str, level: float, rng) -> np.ndarray:
    """The reader's BELIEVED emission model, which is no longer the world's."""
    L = np.asarray(leak, dtype=float).copy()
    if kind == "prior_noise":
        u = np.full_like(L, 1.0 / L.shape[1])
        L = (1.0 - level) * L + level * u
    elif kind == "dirichlet":
        # A reader whose model is wrong in a structured way rather than smoothly flattened.
        L = L + level * rng.dirichlet(np.ones(L.shape[1]), size=L.shape[0])
    return L / L.sum(axis=1, keepdims=True)


def _posterior(obs: np.ndarray, like: np.ndarray) -> np.ndarray:
    lp = np.log(np.clip(like, _EPS, None))[:, obs].sum(axis=1)
    lp -= lp.max()
    p = np.exp(lp)
    return p / p.sum()


def _divergence(pa: np.ndarray, pb: np.ndarray) -> float:
    return float(np.sum(pa * np.log(np.clip(pa, _EPS, None) / np.clip(pb, _EPS, None))))


def _balanced_accuracy(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    return 0.5 * (float((a > thr).mean()) + float((b <= thr).mean()))


def run(cfg: Config, n_obs: int = 400, n_emissions: int = 12) -> dict:
    # S-3'S OWN SEED AND ITS OWN FIRST DRAW, so the affinity matrix here IS S-3's affinity
    # matrix. Rule 2 of this package: re-scoring another module's result means reproducing its
    # number first, not running something similar. Built at 90_300 and consumed in the same order
    # ``s3_two_channels.run`` consumes it; ``reproduces_s3`` checks the reproduction held.
    rng = np.random.default_rng(SEED_OFFSET + 90_300)
    aff, leak = _build(rng)
    rows = []

    def block(tag, theta, amp, degrade_kind, degrade_level, swap_p, model_states):
        # The reader's model. ``model_states`` < N_STATES is a reader with the wrong cardinality.
        L = _degrade(leak, degrade_kind, degrade_level, rng)
        Lm = L[:int(model_states)] if int(model_states) != N_STATES else L
        Lm = Lm / Lm.sum(axis=1, keepdims=True)
        disp = np.stack([_display(aff, s, theta, amp) for s in range(N_STATES)])
        for i in range(int(n_obs)):
            s = int(rng.integers(N_STATES))
            lk = rng.choice(N_SYMBOLS, size=n_emissions, p=leak[s])
            dp = rng.choice(N_SYMBOLS, size=n_emissions, p=disp[s])
            if swap_p > 0 and rng.random() < swap_p:
                lk, dp = dp, lk           # the reader has the channels the wrong way round
            p_leak = _posterior(lk, Lm)
            p_disp = _posterior(dp, Lm)
            rows.append({
                "tag": tag, "theta": float(theta), "amp": float(amp),
                "degrade_kind": degrade_kind, "degrade_level": float(degrade_level),
                "swap_p": float(swap_p), "model_states": int(model_states),
                "concealing": int(theta > 0.0), "i": i,
                "leak_correct": int(int(np.argmax(p_leak)) == s and int(model_states) == N_STATES),
                "divergence": _divergence(p_leak, p_disp),
            })

    # 1. S-3's own design, reproduced, plus partial concealment.
    for theta in THETAS:
        for amp in AMPLIFICATIONS:
            block("theta_sweep", theta, amp, "none", 0.0, 0.0, N_STATES)
    # 2. Reader degradations, at S-3's full negation, at a partial one, AND at theta = 0 -- the
    # degraded CANDID baseline. The first run omitted the 0.0 arm, so the scoring loop's fairness
    # fallback ("candid arm must be degraded the same way") silently compared every degraded
    # concealing reader against an UNdegraded baseline, which inflates the separation wherever
    # degradation raises divergence in both arms.
    for theta in (1.0, 0.5, 0.0):
        for amp in AMPLIFICATIONS:
            for lvl in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95):
                block("prior_noise", theta, amp, "prior_noise", lvl, 0.0, N_STATES)
            for lvl in (0.5, 1.0, 2.0, 4.0):
                block("dirichlet", theta, amp, "dirichlet", lvl, 0.0, N_STATES)
            for sp in (0.0, 0.1, 0.25, 0.5):
                block("channel_swap", theta, amp, "none", 0.0, sp, N_STATES)
            for ms in (2, 3, 4):
                block("wrong_cardinality", theta, amp, "none", 0.0, 0.0, ms)

    df = pd.DataFrame(rows)
    df.to_csv(sl_dir() / "t4_uncertain_reader_points.csv", index=False)
    (df.groupby(["tag", "theta", "amp", "degrade_level", "swap_p", "model_states"])
       [["divergence", "leak_correct"]].agg(["mean", "std", "count"])
       .to_csv(sl_dir() / "t4_uncertain_reader_summary.csv"))

    def pair(d: pd.DataFrame, theta: float, amp: float):
        a = d[(d.theta == theta) & (d.amp == amp)].divergence.to_numpy()
        b = d[(d.theta == 0.0) & (d.amp == amp)].divergence.to_numpy()
        return a, b

    # ---- S-3 reproduced, and then re-scored with a FROZEN threshold ---------------------------
    ts = df[df.tag == "theta_sweep"]
    fitted, frozen = {}, {}
    a1, b1 = pair(ts, 1.0, AMPLIFICATIONS[0])
    thr_frozen = float(np.median(np.concatenate([a1, b1])))
    for amp in AMPLIFICATIONS:
        a, b = pair(ts, 1.0, amp)
        thr_fit = float(np.median(np.concatenate([a, b])))
        sep = T.boot_diff(a, b, rng)
        fitted[str(amp)] = {"separation": sep["difference"], "interval": sep["interval"],
                            "threshold": thr_fit,
                            "balanced_accuracy": _balanced_accuracy(a, b, thr_fit)}
        frozen[str(amp)] = {"threshold": thr_frozen,
                            "balanced_accuracy": _balanced_accuracy(a, b, thr_frozen)}
    fit_rise = fitted[str(AMPLIFICATIONS[-1])]["balanced_accuracy"] \
        - fitted[str(AMPLIFICATIONS[0])]["balanced_accuracy"]
    frz_rise = frozen[str(AMPLIFICATIONS[-1])]["balanced_accuracy"] \
        - frozen[str(AMPLIFICATIONS[0])]["balanced_accuracy"]

    # ---- partial concealment ------------------------------------------------------------------
    partial = {}
    for theta in THETAS:
        if theta == 0.0:
            continue
        row = {}
        for amp in AMPLIFICATIONS:
            a, b = pair(ts, theta, amp)
            thr = float(np.median(np.concatenate([a, b])))
            sep = T.boot_diff(a, b, rng)
            row[str(amp)] = {"separation": sep["difference"], "interval": sep["interval"],
                             "excludes_zero": sep["excludes_zero"],
                             "balanced_accuracy": _balanced_accuracy(a, b, thr),
                             "balanced_accuracy_frozen_at_amp1": _balanced_accuracy(
                                 a, b, thr_frozen)}
        partial[str(theta)] = row

    # ---- degradations --------------------------------------------------------------------------
    degraded = {}
    for tag, col, levels in (("prior_noise", "degrade_level", (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)),
                             ("dirichlet", "degrade_level", (0.5, 1.0, 2.0, 4.0)),
                             ("channel_swap", "swap_p", (0.0, 0.1, 0.25, 0.5)),
                             ("wrong_cardinality", "model_states", (2, 3, 4))):
        d = df[df.tag == tag]
        block_out = {}
        for theta in (1.0, 0.5):
            for lvl in levels:
                dd = d[d[col] == lvl]
                per_amp = {}
                for amp in AMPLIFICATIONS:
                    a = dd[(dd.theta == theta) & (dd.amp == amp)].divergence.to_numpy()
                    b = ts[(ts.theta == 0.0) & (ts.amp == amp)].divergence.to_numpy()
                    # candid arm must be degraded the same way, or the contrast is unfair
                    bb = dd[(dd.theta == 0.0) & (dd.amp == amp)].divergence.to_numpy()
                    b_use = bb if bb.size else b
                    if a.size == 0 or b_use.size == 0:
                        continue
                    thr = float(np.median(np.concatenate([a, b_use])))
                    sep = T.boot_diff(a, b_use, rng)
                    per_amp[str(amp)] = {
                        "separation": sep["difference"], "interval": sep["interval"],
                        "excludes_zero": sep["excludes_zero"],
                        "balanced_accuracy": _balanced_accuracy(a, b_use, thr),
                        "balanced_accuracy_frozen_at_amp1": _balanced_accuracy(
                            a, b_use, thr_frozen),
                    }
                if per_amp:
                    block_out[f"theta{theta}_{col}{lvl}"] = per_amp
        degraded[tag] = block_out

    # ---- where does it die? ---------------------------------------------------------------------
    breakdown = {}
    for tag, col in (("prior_noise", "degrade_level"), ("channel_swap", "swap_p"),
                     ("wrong_cardinality", "model_states")):
        alive = []
        for k, per_amp in degraded[tag].items():
            if not k.startswith("theta1.0"):
                continue
            ok = all(v["excludes_zero"] and v["separation"] > 0 for v in per_amp.values())
            alive.append((k, ok, float(np.mean([v["balanced_accuracy"]
                                                for v in per_amp.values()]))))
        breakdown[tag] = [{"level": k, "separation_survives_at_all_amplifications": ok,
                           "mean_balanced_accuracy": acc} for k, ok, acc in alive]

    # ---- standing gates ---------------------------------------------------------------------
    gr = G.GateReport()
    _leak = float(df[(df.tag == "theta_sweep") & (df.model_states == N_STATES)].leak_correct.mean())
    gr.positive("reproduces_s3_leak_accuracy", _leak, 0.8990625, 0.02,
                detail="built from S-3's own seed and affinity matrix, so the leak must be "
                       "readable at S-3's own rate. Rule 2 of this package: re-scoring another "
                       "module's result means reproducing its number first.")
    gr.positive("candid_display_is_not_concealment",
                float(ts[(ts.theta == 0.0)].divergence.mean()), 0.0, 1.5,
                detail="at theta = 0 the display tracks the leak, so channel divergence must sit "
                       "near zero. The zero-strength end of the manipulation.")
    gr.live("concealment_reaches_the_divergence",
            float(ts[(ts.theta == 1.0)].divergence.mean()
                  - ts[(ts.theta == 0.0)].divergence.mean()), 1.0,
            detail="full concealment must move the divergence. The manipulation reaches the "
                   "measurement.")
    gr.no_oracle("frozen_threshold_does_not_collapse_the_result",
                 abs(float(fit_rise) - float(frz_rise)), 0.15,
                 detail="S-3 fitted its detector threshold on the pooled labelled test data and "
                        "re-fitted it per cell. This bounds how much of the headline that "
                        "bought: if fitted and frozen diverge by more than this, the result is "
                        "the threshold rather than the effect.")

    verdict = {
        "test": "T-4 — does leaked/emblematic divergence survive an uncertain reader?",
        "for": "Sounding Line, whether the leaked-layer programme transports off the simulator",
        "reproduces_s3": {
            "s3_leak_accuracy": 0.8990625,
            "here_leak_accuracy": float(df[(df.tag == "theta_sweep")
                                           & (df.model_states == N_STATES)].leak_correct.mean()),
            "note": ("built from S-3's own seed and affinity matrix. The leak accuracy is the "
                     "cheapest check that the construction is the same one."),
        },
        "S3_THRESHOLD_PROBLEM": {
            "what": ("S-3 sets its detector threshold to the median of the pooled concealing and "
                     "candid divergences AT THE AMPLIFICATION BEING SCORED. That is fitted on "
                     "ground-truth-labelled test data and re-fitted per cell, which no reader "
                     "can do."),
            "fitted_per_amplification": fitted,
            "frozen_at_lowest_amplification": frozen,
            "accuracy_rise_with_fitted_threshold": float(fit_rise),
            "accuracy_rise_with_frozen_threshold": float(frz_rise),
            "the_shield_still_gives_them_away_under_a_frozen_threshold": bool(frz_rise > 0),
        },
        "partial_concealment": partial,
        "reader_degradations": degraded,
        "where_it_breaks": breakdown,
        "construction": {"n_states": N_STATES, "n_symbols": N_SYMBOLS,
                         "emissions_per_artifact": int(n_emissions),
                         "amplifications": list(AMPLIFICATIONS), "thetas": list(THETAS),
                         "theta_meaning": "0 candid, 1 = S-3's exact negation, between = partial"},
        "what_would_have_falsified_it": (
            "divergence surviving every degradation unchanged, which would mean the signal does "
            "not depend on the reader's model at all and S-3 transports for free; or dying at the "
            "first perturbation, which would mean it is an oracle result."),
        "what_this_cannot_show": (
            "anything about real text, and nothing about the Ghost Scale reader. This is the "
            "minimal two-channel emitter S-3 built, degraded. The V5 world has one emission "
            "channel."),
    }
    PROV.stamp(verdict, __file__, gr)
    (sl_dir() / "t4_uncertain_reader.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
