"""Reader-side machinery for strategic sources (spec §6, trunk S).

The generator lives in ``world_communication``. This module is what a *reader* brings: an audience
model that may be wrong, a recursive listener that may over-apply itself, a selection-policy
posterior, and an uptake gate that has to stay factored under contradiction.

The three things V14 got for free and V15 must earn
---------------------------------------------------
1. **The audience model.** V14's recursive listener knew the maker's audience model by oracle.
   Here the reader carries its own, and S06 crosses *strategy* against *model match*: recursive
   reading should help against a steering maker with a matched model and hurt with a wrong one.
2. **The private readout.** Removed at the generator (see ``world_communication``). What is left is
   a probe the reader must *choose to buy*, and S03 asks which one is worth buying.
3. **The uptake gate.** V14 showed a factored gate moves policy uptake without moving belief. That
   only means anything if the two can come apart under contradiction, so S08 contradicts the
   source's content and its record separately and reports the full side-effect matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import common as C
from . import world_communication as WC

PROBES = ("audience_persuaded", "private_cost", "correction", "evidence_choice")
#: The five owners S08 must keep separate under contradiction.
OWNERS = ("source_motive", "content_truth", "reliability_history", "induced_response", "uptake")


@dataclass
class AudienceModel:
    """What the reader thinks the maker thinks the audience believes."""

    belief: np.ndarray
    match: float = 1.0                      # 1 correct, 0 unrelated to the true audience

    @staticmethod
    def make(world, true_belief: np.ndarray, match: float, rng) -> "AudienceModel":
        wrong = C.normalize(rng.random(true_belief.size))
        return AudienceModel(belief=C.normalize(match * true_belief + (1 - match) * wrong),
                             match=float(match))


def literal_reader(world, ep, upto: int, budget=None) -> np.ndarray:
    """Face value: the assertion means what it says, and evidence is a random sample."""
    toks = ep.routes.get("assertion", [])[:upto]
    counts = np.full(world.n_g, 0.5)
    for t in toks:
        counts[t % world.n_g] += 1.0
    if budget is not None:
        budget.lik(1)
    return C.normalize(counts)


def audience_aware_reader(world, ep, upto: int, audience: AudienceModel, budget=None) -> np.ndarray:
    """A recursive listener: what would a maker steering *this* audience have said?

    The reader inverts the selection policy under its own audience model. When the model is right
    this pays; when it is wrong it is confidently wrong, which is the whole content of S06.
    """
    shown = ep.meta.get("actions", [])[:upto]
    lg = np.zeros(world.n_g)
    for g in range(world.n_g):
        acc = 0.0
        for p in range(world.n_p):
            sel = WC._selection_distribution(world, p, g, ep.context)
            steer = C.normalize(sel * (1.0 + 1.6 * audience.belief[:sel.size]
                                       if audience.belief.size >= sel.size else sel))
            acc += float(np.log(np.maximum(steer[np.asarray(shown, int)], 1e-300)).sum()) \
                if shown else 0.0
        lg[g] = acc / max(world.n_p, 1)
    if budget is not None:
        budget.lik(world.n_g * world.n_p)
    return C.softmax(lg)


def selection_policy_posterior(world, ep, upto: int, budget=None) -> dict:
    """S07: was this evidence *selected*, or is it a random sample?

    The reader scores the observed evidence stream under each selection policy including the
    null one (``sample_all``), so "selected" is a comparison and not an assumption.
    """
    shown = ep.meta.get("actions", [])[:upto]
    lg = np.zeros(world.n_p)
    for p in range(world.n_p):
        acc = 0.0
        for g in range(world.n_g):
            sel = WC._selection_distribution(world, p, g, ep.context)
            acc += float(np.exp(np.log(np.maximum(sel[np.asarray(shown, int)], 1e-300)).sum())) \
                if shown else 1.0
        lg[p] = float(np.log(max(acc / max(world.n_g, 1), 1e-300)))
    if budget is not None:
        budget.lik(world.n_p * world.n_g)
    post = C.softmax(lg)
    names = [WC.SELECTION[i % len(WC.SELECTION)] for i in range(world.n_p)]
    out = {n: float(v) for n, v in zip(names, post)}
    out["_selected_mass"] = float(sum(v for n, v in out.items()
                                      if not n.startswith("_") and n != "sample_all"))
    return out


def probe_value(world, latent, prior_post: np.ndarray, probe: str, rng, budget=None) -> dict:
    """S03: what one counterfactual probe is worth, in expected information about the motive."""
    before = WC.motive_posterior(world, prior_post)
    out = WC.counterfactual_probe(world, latent, probe, rng)
    lg = np.log(np.maximum(prior_post, 1e-300))
    for t in world.latent_space():
        lg[t] += WC.probe_likelihood(world, t, probe, out["draw"], budget)
    after_post = C.softmax(lg.ravel()).reshape(prior_post.shape)
    after = WC.motive_posterior(world, after_post)
    keys = sorted(before)
    return {"probe": probe, "outcome": out["draw"],
            "information_gain": C.kl(np.array([after[k] for k in keys]),
                                     np.array([before[k] for k in keys])),
            "posterior": after_post, "motive_after": after}


def buy_probes(world, latent, post: np.ndarray, probes, rng, budget=None) -> dict:
    """Buy a sequence of probes and report the trajectory, not only the endpoint."""
    cur = post
    trace = []
    for pr in probes:
        r = probe_value(world, latent, cur, pr, rng, budget)
        cur = r["posterior"]
        trace.append({"probe": pr, "information_gain": r["information_gain"],
                      "motive": r["motive_after"]})
    return {"posterior": cur, "trace": trace,
            "motive": WC.motive_posterior(world, cur)}


# --------------------------------------------------------------------------- #
# Uptake (S08, S09).
# --------------------------------------------------------------------------- #
@dataclass
class UptakeState:
    """Five quantities that must be able to move independently."""

    source_motive: float = 0.5
    content_truth: float = 0.5
    reliability_history: float = 0.5
    induced_response: float = 0.0
    uptake: float = 0.5
    meta: dict = field(default_factory=dict)

    def vector(self) -> np.ndarray:
        return np.array([self.source_motive, self.content_truth, self.reliability_history,
                         self.induced_response, self.uptake])


def factored_uptake(state: UptakeState, evidence: dict, gate: str = "factored") -> UptakeState:
    """Update uptake from a message.

    ``factored``  belief in the content moves with the evidence; *policy uptake* is gated by the
                  source's motive and record, and the two are not the same number.
    ``scalar``    one trust number multiplies everything, so distrusting a source also stops the
                  reader believing true things it says. That is the failure S09 measures as
                  negative transfer.
    """
    s = UptakeState(**{k: v for k, v in state.__dict__.items() if k != "meta"})
    support = float(evidence.get("support", 0.0))
    strategic = float(evidence.get("strategic", 0.0))
    corrected = float(evidence.get("corrected", 0.0))
    intensity = float(evidence.get("intensity", 0.0))

    if gate == "scalar":
        trust = float(np.clip(1.0 - strategic, 0.05, 1.0))
        s.content_truth = float(np.clip(s.content_truth + trust * 0.35 * support, 0.0, 1.0))
        s.uptake = float(np.clip(s.uptake + trust * 0.35 * support, 0.0, 1.0))
        s.source_motive = float(np.clip(s.source_motive + 0.3 * strategic, 0.0, 1.0))
        s.reliability_history = float(np.clip(s.reliability_history + 0.3 * corrected, 0.0, 1.0))
        s.induced_response = float(np.clip(s.induced_response + 0.4 * intensity, 0.0, 1.0))
        return s

    s.content_truth = float(np.clip(s.content_truth + 0.35 * support, 0.0, 1.0))
    s.source_motive = float(np.clip(s.source_motive + 0.3 * strategic, 0.0, 1.0))
    s.reliability_history = float(np.clip(s.reliability_history + 0.3 * corrected, 0.0, 1.0))
    s.induced_response = float(np.clip(s.induced_response + 0.4 * intensity, 0.0, 1.0))
    discount = float(np.clip(1.0 - 0.8 * s.source_motive * (1.0 - s.reliability_history),
                             0.05, 1.0))
    s.uptake = float(np.clip(s.uptake + 0.35 * support * discount, 0.0, 1.0))
    return s


def side_effect_matrix(gate: str = "factored") -> dict:
    """S08: intervene on each owner and record what else moved.

    A factored gate should show a near-diagonal matrix; a scalar gate should show belief moving
    when only trust was touched.
    """
    # Interventions happen against a STANDING MESSAGE. With no message present the scalar
    # gate's whole failure mode -- distrust suppressing belief in true content -- has nothing
    # to act on, and both gates report a perfectly diagonal matrix.
    message = {"support": 0.8}
    # Both arms start from a FRESH state and differ only in the perturbation. Comparing an
    # already-updated baseline against a second update reports a second dose of the same
    # evidence as though it were a side-effect.
    base = factored_uptake(UptakeState(), message, gate)
    out = {}
    for owner, ev in (("content_truth", {"support": 1.0}),
                      ("source_motive", {"strategic": 1.0}),
                      ("reliability_history", {"corrected": 1.0}),
                      ("induced_response", {"intensity": 1.0})):
        after = factored_uptake(UptakeState(), {**message, **ev}, gate)
        out[owner] = {k: float(getattr(after, k) - getattr(base, k))
                      for k in ("source_motive", "content_truth", "reliability_history",
                                "induced_response", "uptake")}
    off = [abs(v[k]) for owner, v in out.items() for k in v
           if k != owner and k != "uptake"]
    return {"gate": gate, "matrix": out, "max_off_diagonal": float(max(off) if off else 0.0)}


def selective_uptake(rng, n: int = 120, gate: str = "factored", strategic_rate: float = 0.5,
                     true_rate: float = 0.5) -> dict:
    """S09: does correct motive inference improve *selective* uptake?

    Messages are crossed: true or false content, strategic or sincere source. A reader with blanket
    distrust rejects true content from strategic sources; a copier accepts false content from them.
    Both costs are reported.
    """
    rows = []
    for _ in range(int(n)):
        strategic = float(rng.random() < strategic_rate)
        true_content = float(rng.random() < true_rate)
        # The reader cannot see truth. It sees evidence support, which is higher for true
        # content but overlaps -- and which a strategic source inflates by selecting. That
        # inflation is exactly what motive inference is supposed to discount, and passing a
        # constant support (as this did) makes every gate equally unselective by construction.
        support = float(np.clip(rng.normal(0.72 if true_content > 0.5 else 0.34, 0.22)
                                + 0.34 * strategic, 0.0, 1.0))
        st = factored_uptake(UptakeState(), {"support": support, "strategic": strategic,
                                             "corrected": 1.0 - strategic * 0.6}, gate)
        rows.append({"strategic": strategic, "true_content": true_content,
                     "support": support, "uptake": st.uptake, "belief": st.content_truth})
    def m(sel, key):
        v = [r[key] for r in rows if sel(r)]
        return float(np.mean(v)) if v else float("nan")
    return {"gate": gate,
            "true_uptake": m(lambda r: r["true_content"] > 0.5, "uptake"),
            "false_uptake": m(lambda r: r["true_content"] < 0.5, "uptake"),
            "true_belief": m(lambda r: r["true_content"] > 0.5, "belief"),
            "negative_transfer": m(lambda r: r["true_content"] > 0.5 and r["strategic"] > 0.5,
                                   "uptake"),
            "n": len(rows)}
