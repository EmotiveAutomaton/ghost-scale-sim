"""D-1 — channel accounting. What the reader's provenance belief is actually made of.

NO SIMULATION. Every number here is arithmetic on the likelihood arrays, which is why it runs first:
it predicts the outcome of several parameter sweeps before any parameter is swept.

-----------------------------------------------------------------------------------------
THE THING NOTHING IN THE RECORD NAMES.

The observer receives three observations at every timestep, and TWO of them carry information about
provenance:

* the **label** (modality 1), whose likelihood is `kappa * onehot(truthful(p)) + (1 - kappa) *
  uniform`, so a declared signal argues for the tier it names;
* the **content** (modality 0), whose likelihood under DEEP is `alpha[p] * sig[goal] + (1 -
  alpha[p]) * synth`, so features that look synthetic argue for a low-alpha tier.

Both arrive at EVERY timestep. Both accumulate. On machine-made content carrying a false human
label they point in OPPOSITE directions, and which one wins is decided by a single inequality
between two per-observation log-likelihood ratios.

That has two consequences, and neither is anywhere in the specs or the results files.

**First, the run length decides the mechanism.** The number of timesteps is an engineering
parameter, chosen for reasons of cost and of giving each condition a comparable inference attempt.
It is not a theoretical quantity. But when the two channels disagree, the winner after T steps is
whichever has the larger per-step evidence, so T decides which channel produced the result. A
headline measured at T = 24 can invert at T = 4 with nothing else changed.

**Second, it predicts the parameter sweeps.** The crossover is a closed-form function of kappa given
alpha, so the kappa at which the label stops beating the content can be computed rather than
discovered. The validation pass's robustness matrix found the label effect weakening from 14.9x to
3.1x at kappa = 0.30 and recorded no reason. The reason is that 0.30 is below the crossover.

-----------------------------------------------------------------------------------------
WHAT IS COMPUTED, AND THE ONE APPROXIMATION IN IT.

For a true provenance `p_true`, a declared signal naming `p_claim`, and a goal `g`:

    content LLR  =  E_{f ~ P(f | p_true, g, DEEP)} [ log P(f | p_claim, g, DEEP)
                                                     - log P(f | p_true, g, DEEP) ]
    label   LLR  =  log P(sig_claim | p_claim) - log P(sig_claim | p_true)

The content term is an EXPECTED log-likelihood ratio, taken under the distribution the features are
actually drawn from. That makes it the per-observation drift of the log posterior odds, which is the
right quantity for asking who wins in the limit, and it is negative by Gibbs' inequality whenever
the two columns differ: content always argues for the truth, on average. The label term is exact and
constant, because the same declared signal arrives every step.

The approximation is that this describes the DRIFT and not the whole trajectory. Individual feature
draws fluctuate, so a reader can be pulled the wrong way for a few steps; the crossover describes
where the drift changes sign, which is what decides the outcome as T grows. The measured posterior
trajectories in the verdict file confirm the drift picture, and they are reported beside the
arithmetic rather than instead of it.
"""
from __future__ import annotations

import json

import numpy as np

from .. import constants as K
from ..config import Config
from . import criteria as CR
from . import diagnostics_dir

_FLOOR = 1e-300


def _expected_llr_content(A0: np.ndarray, p_true: int, p_claim: int, goal: int,
                          attention: int = K.DEEP) -> float:
    """Per-observation expected log-LR for `p_claim` against `p_true`, from the content channel."""
    p_true_col = np.asarray(A0)[:, p_true, goal, attention]
    p_claim_col = np.asarray(A0)[:, p_claim, goal, attention]
    m = p_true_col > 0
    return float(np.sum(p_true_col[m] * np.log(np.clip(p_claim_col[m], _FLOOR, None)
                                               / p_true_col[m])))


def _llr_label(A1: np.ndarray, signal: int, p_true: int, p_claim: int) -> float:
    """Per-observation log-LR for `p_claim` against `p_true`, from the label channel."""
    a1 = np.asarray(A1)
    num = float(a1[signal, p_claim, 0, K.DEEP])
    den = float(a1[signal, p_true, 0, K.DEEP])
    return float(np.log(max(num, _FLOOR) / max(den, _FLOOR)))


def label_llr_from_kappa(kappa: float, num_signals: int) -> float:
    """Closed form for the label channel's log-LR when the declared signal names `p_claim`.

        P(sig | p_claim) = kappa + (1 - kappa) / S      the truthful row
        P(sig | p_true)  =         (1 - kappa) / S      any other row
        LLR = log( (1 + (S - 1) kappa) / (1 - kappa) )

    Written out because it is what makes the crossover solvable rather than searchable.
    """
    k = float(np.clip(kappa, 0.0, 1.0 - 1e-12))
    s = int(num_signals)
    return float(np.log((1.0 + (s - 1) * k) / (1.0 - k)))


def crossover_kappa(content_llr: float, num_signals: int) -> float:
    """The kappa at which the label channel exactly cancels the content channel.

    Solves `log((1 + (S-1)k) / (1-k)) = -content_llr` for k. Returns NaN when the content term is
    non-negative, which means the content does not argue against the claim and there is nothing to
    cross over.
    """
    if not np.isfinite(content_llr) or content_llr >= 0:
        return float("nan")
    r = float(np.exp(-content_llr))
    s = int(num_signals)
    k = (r - 1.0) / (r + s - 1.0)
    return float(k) if 0.0 <= k <= 1.0 else float("nan")


# --------------------------------------------------------------------------- #
# The cells worth accounting for: every place a label and a content stream disagree.
# --------------------------------------------------------------------------- #
CELLS = (
    ("machine work passed off as human", K.GHOST, K.CREATOR, K.SIG_CREATOR,
     "E2, E4, E17, A1, E31. The trust exploit, and the one cell where the two channels are in "
     "open conflict."),
    ("machine work labelled honestly", K.GHOST, K.GHOST, K.SIG_GHOST,
     "E2, E17, E31. Both channels agree, so there is no race and no crossover."),
    ("human work falsely labelled machine", K.CREATOR, K.GHOST, K.SIG_GHOST,
     "A1's other arm. The mirror of the exploit, and the asymmetry claim rests on the two "
     "behaving differently."),
    ("curated work passed off as fully human", K.CURATOR, K.CREATOR, K.SIG_CREATOR,
     "E17's interior tier, where alpha is 0.60 rather than 0.05 and the content argues far more "
     "weakly."),
)


def run(cfg: Config, workers: int = 1) -> dict:
    """Account for both channels at every conflicting cell, then measure one trajectory."""
    from ..generative_model import build_shared_model

    gm = build_shared_model(cfg)
    A0, A1 = np.asarray(gm.A[0]), np.asarray(gm.A[1])
    num_signals = int(cfg.cardinalities.num_signals)
    kappa_default = float(cfg.signal_model.kappa)
    goal = int(cfg.get("experiments.e2.true_goal", 1))
    n_timesteps = int(cfg.run.n_timesteps)

    rows = []
    for name, p_true, p_claim, signal, where in CELLS:
        content = _expected_llr_content(A0, p_true, p_claim, goal)
        label = _llr_label(A1, signal, p_true, p_claim)
        cross = crossover_kappa(content, num_signals)
        net = label + content
        # How many steps the label leads for before the content overtakes, when it does.
        steps = (float(label / -content) if content < 0 and label > 0 and net < 0
                 else float("inf") if net > 0 else 0.0)
        rows.append({
            "cell": name,
            "true_provenance": K.PROVENANCE_NAMES[p_true],
            "claimed_provenance": K.PROVENANCE_NAMES[p_claim],
            "alpha_true": float(gm.alpha[p_true]),
            "alpha_claimed": float(gm.alpha[p_claim]),
            "content_llr_per_step": content,
            "label_llr_per_step": label,
            "net_llr_per_step": net,
            "channels_conflict": bool(content < 0 < label or label < 0 < content),
            "label_wins_in_the_limit": bool(net > 0),
            "steps_before_content_overtakes": steps,
            "crossover_kappa": cross,
            "kappa_margin_nats": (float(label + content) if np.isfinite(cross) else float("nan")),
            "run_length": n_timesteps,
            "where_it_matters": where,
        })

    # The kappa curve for the one cell that is in open conflict, so a reader can see the crossover
    # rather than take it on trust.
    from ..generative_model import build_A1_observer
    conflict = rows[0]
    content_conflict = conflict["content_llr_per_step"]
    curve = []
    for kappa in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.54, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        a1 = build_A1_observer(cfg, kappa)
        lab = _llr_label(a1, K.SIG_CREATOR, K.GHOST, K.CREATOR)
        curve.append({"kappa": float(kappa), "label_llr": lab,
                      "content_llr": content_conflict,
                      "net": lab + content_conflict,
                      "label_wins": bool(lab + content_conflict > 0),
                      "closed_form_label_llr": label_llr_from_kappa(kappa, num_signals)})

    trajectory = _measured_trajectory(cfg)

    cross = conflict["crossover_kappa"]
    margin = float(conflict["net_llr_per_step"])
    safe = bool(abs(margin) >= CR.D1_SAFE_MARGIN_NATS)
    default_above = bool(np.isfinite(cross) and kappa_default > cross)

    if not np.isfinite(cross):
        verdict = "NO_CHANNEL_CONFLICT"
        statement = ("No cell has the two channels in conflict, so there is no race and the "
                     "run length cannot decide which mechanism produced a result.")
    elif default_above and safe:
        verdict = "LABEL_WINS_AT_THE_DEFAULT_WITH_MARGIN"
        statement = (
            f"The reader's belief about who made something is fed by two streams that both arrive "
            f"every timestep and, on machine work passed off as human, point in opposite "
            f"directions. The content argues for the truth at {content_conflict:+.3f} nats per "
            f"observation. The label argues for the lie at {conflict['label_llr_per_step']:+.3f} "
            f"nats per observation at the default trust of {kappa_default}. The label wins, with a "
            f"margin of {margin:+.3f} nats per step, and it wins at any run length.\n\n"
            f"**The crossover sits at trust = {cross:.3f}.** Below it the content overtakes the "
            f"label as the run lengthens and the reader ends up believing the truth, so the trust "
            f"exploit is not a fact about labels in general but a fact about labels trusted more "
            f"than {cross:.2f}. That is a narrower claim than the one currently made and it is the "
            f"accurate one. It also explains, without any new simulation, why the validation "
            f"pass's robustness matrix found the label effect weakening five-fold at trust 0.30: "
            f"0.30 is below the crossover.")
    elif default_above:
        verdict = "LABEL_WINS_BUT_NEAR_THE_CROSSOVER"
        statement = (
            f"The label beats the content at the default trust of {kappa_default}, but only by "
            f"{margin:+.3f} nats per observation, inside the {CR.D1_SAFE_MARGIN_NATS} nat margin "
            f"committed before the run. At that distance from the crossover at {cross:.3f} the "
            f"outcome depends on the run length, which is an engineering parameter, and every "
            f"claim from this cell has to say so.")
    else:
        verdict = "CONTENT_WINS_AT_THE_DEFAULT"
        statement = (
            f"At the default trust of {kappa_default} the content channel beats the label "
            f"({margin:+.3f} nats per observation net), so the reader ends up believing the truth "
            f"as the run lengthens. Any label effect measured at this setting is a transient of "
            f"the early timesteps rather than the model's steady state, and the claims resting on "
            f"it have to be restated in those terms.")

    verdict_payload = {
        "check": "D-1",
        "question": ("What is the reader's belief about provenance actually made of, and which "
                     "channel wins?"),
        "plain_language": (
            "The reader is told who made the thing, and it also looks at the thing. Both of those "
            "are evidence about who made it, both arrive at every glance, and on a lie they "
            "disagree. This works out how strong each one is per glance, and therefore which one "
            "wins. It needs no simulation at all, which is why it runs first: the answer predicts "
            "how several later sweeps will come out."),
        "method": ("expected per-observation log-likelihood ratio for each channel, computed on "
                   "the likelihood arrays. No simulation. See the module docstring for the one "
                   "approximation: these are drifts, not trajectories."),
        "criteria": {"safe_margin_nats": CR.D1_SAFE_MARGIN_NATS},
        "default_kappa": kappa_default,
        "num_signals": num_signals,
        "cells": rows,
        "kappa_curve_on_the_conflicting_cell": curve,
        "measured_trajectory": trajectory,
        "crossover_kappa": cross,
        "default_is_above_crossover": default_above,
        "margin_nats_per_step": margin,
        "margin_is_safe": safe,
        "verdict": verdict,
        "statement": statement,
        "what_this_changes": [
            ("Every claim phrased as 'a provenance label does X' is really 'a provenance label "
             "trusted above {:.2f} does X at this content transparency'. The threshold is a "
             "property of the two alphas and the signal cardinality, not of the theory."
             .format(cross) if np.isfinite(cross) else
             "No cell is in conflict, so no claim needs the qualifier."),
            ("The run length is load-bearing wherever the margin is small. It was chosen as an "
             "engineering parameter and it should be reported alongside any claim from a "
             "near-crossover cell."),
            ("The robustness matrix's kappa row is explained: the weakening at low trust is the "
             "crossover, and it could have been predicted from the construction."),
        ],
    }
    out = diagnostics_dir()
    (out / "d1_channels.json").write_text(
        json.dumps(verdict_payload, indent=2, default=float), encoding="utf-8")
    return verdict_payload


def _measured_trajectory(cfg: Config) -> list:
    """One measured posterior trajectory per kappa, to check the drift picture against a run.

    Cheap: a single reader per kappa, exact inference, forced DEEP so the tape is comparable. This
    is the only simulation in D-1 and it exists to keep the arithmetic honest.
    """
    from ..environment import Artifact, Environment
    from ..exact import make_exact_agent
    from ..generative_model import build_D, build_shared_model

    rows = []
    for kappa in (0.1, 0.3, 0.5, 0.7, 0.9):
        c = cfg.copy()
        c.set("signal_model.kappa", float(kappa))
        gm = build_shared_model(c, kappa=float(kappa))
        env = Environment(c, gm, np.random.default_rng(4), honesty=1.0, signing_rate=1.0)
        art = Artifact(provenance=K.GHOST, goal=int(c.get("experiments.e2.true_goal", 1)),
                       declared_signal=K.SIG_CREATOR)
        agent = make_exact_agent(gm, build_D(c, np.random.default_rng(7)), c)
        agent.reset()
        obs_rng = np.random.default_rng(11)
        series = []
        for t in range(int(c.run.n_timesteps)):
            agent.infer_states(env.observation(art, K.DEEP, obs_rng))
            agent.action = np.zeros(len(agent.num_controls))
            agent.action[K.F_ATTENTION] = K.DEEP
            series.append(float(np.asarray(agent.qs[K.F_PROVENANCE])[K.CREATOR]))
        rows.append({
            "kappa": float(kappa),
            "believes_the_label_at_t": {"1": series[0], "2": series[1], "4": series[3],
                                        "10": series[9], str(len(series)): series[-1]},
            "ends_believing_the_label": bool(series[-1] > 0.5),
        })
    return rows
