"""Trunk X — the cross-cutting adversarial matrix (spec §7, attacks X01-X24).

Each attack takes the estimand of the flight it targets, runs it unattacked and attacked on the
*transfer* lineage, and reports what the manipulation cost. An attack that changes the task rather
than the reading is worthless, so every attack here holds the endpoint and the scoring fixed and
moves only the thing it names.

Three of the twenty-four are not about the science at all. X21 attacks the *reporting*: it builds a
world whose effect reverses sign across a stratum and checks that a pooled mean would have hidden
it. X23 attacks the lineage bookkeeping. X24 attacks the runtime -- a fast machine, a restart, an
orphan kill, a stale checkpoint and a clean clone. Those three fail loudly if the machinery that
protects the other twenty-one has stopped working.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as EX
from .. import foraging as FG
from .. import foreground as FGN
from .. import learning_history as LH
from .. import particles as PF
from .. import persistent as PS
from .. import routes as RT
from .. import strategic_source as SS
from .. import world_communication as WC
from . import (Cells, arch_gap, battery, criterion, decide_state, distances, families_of,
               family_module, finish, mean_of, narrative, paired, publication, receipt, rng,
               rows_of, run_tournament, sizes, start, world_for)

CHANNELS = [{"name": "attacked_estimand", "mediated_by_policy": True}]

#: Which pair each flight's estimand is a difference between. A routing estimand has no
#: joint_exact reader, so the default pair silently produced two NaNs and the attack recorded an
#: instrument failure rather than a result.
PAIRS = {
    "coupling_atlas": ("joint_exact", "independent"),
    "routing": ("learned", "equal"),
    "architecture": ("joint_exact", "independent"),
}

#: Which flight each attack targets, and which knobs the attack moves.
TARGETS = {
    "X01": ("coupling_atlas", {}),
    "X02": ("coupling_atlas", {"overlap": 1.0}),
    "X03": ("routing", {"dependence": "redundant"}),
    "X04": ("routing", {"dependence": "redundant"}),
    "X05": ("architecture", {"model_space": "wrong_family"}),
    "X06": ("architecture", {"model_space": "missing_latent"}),
    "X07": ("architecture", {"model_space": "extra_latent"}),
    "X08": ("coupling_atlas", {"missing": "context"}),
    "X09": ("coupling_atlas", {"temperature": 2.0, "competence": 0.55}),
    "X10": ("coupling_atlas", {"equifinality": "exact"}),
    "X11": ("coupling_atlas", {"similarity": -1.0, "typicality": 0.0}),
    "X12": ("expertise", {}),
    "X13": ("expertise", {}),
    "X14": ("value", {}),
    "X15": ("control", {}),
    "X16": ("source", {}),
    "X17": ("source", {}),
    "X18": ("source", {}),
    "X19": ("foraging", {}),
    "X20": ("foraging", {}),
    "X21": ("reporting", {}),
    "X22": ("architecture", {}),
    "X23": ("lineage", {}),
    "X24": ("runtime", {}),
}


# --------------------------------------------------------------------------- #
# The generic attack: run an estimand with and without the manipulation.
# --------------------------------------------------------------------------- #
def _atlas_estimand(ctx, attacked: bool, over: dict, permute: bool = False):
    rows = []
    for fam in families_of(ctx):
        knobs = {"kappa": 0.5, "overlap": 0.0, "dose": 2}
        if attacked:
            knobs.update(over)
        cfg = {"model_space": knobs.get("model_space", "correct")}
        r, world, _ = run_tournament(ctx, fam, ("surface", "independent", "joint_exact",
                                                "oracle_state"),
                                     knobs_over=knobs, cfg=cfg,
                                     extra_key={"attacked": "yes" if attacked else "no",
                                                "family": fam})
        rows += r
    return rows


def _routing_estimand(ctx, attacked: bool, over: dict):
    r = rng(ctx, "routing")
    rows = []
    for _ in range(max(3, sizes(ctx)["makers"] // 5)):
        bank = RT.sample_bank(np.random.default_rng(r.integers(0, 2 ** 62)), dispersion=0.3,
                              duplicated=attacked, easy_useless=False)
        for kind in ("learned", "equal"):
            wt = RT.learn_weights(bank, r, 60, kind=kind)
            sc = RT.score_weighter(bank, wt, r, n=100)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                         "attacked": "yes" if attacked else "no",
                         "architecture": kind, "log_score": sc["log_score"],
                         "correct": sc["accuracy"], "confidence": sc["mean_confidence"], "n": 1})
    return rows


def _expertise_estimand(ctx, attacked: bool, which: str):
    r = rng(ctx, f"expertise|{which}")
    s = sizes(ctx)
    mixes = {"practice_heavy": {"practice": 0.7, "feedback": 0.2, "instruction": 0.1},
             "instruction_heavy": {"instruction": 0.7, "feedback": 0.2, "practice": 0.1}}
    rows = []
    for mix in mixes:
        for _ in range(max(2, s["makers"] // 6)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            truth = LH.sample_truth(sub)
            m = dict(mixes[mix])
            blocked = 0
            if attacked and which == "attention_for_constraint":
                # attention allocation is replaced by an imposed constraint of the same weight
                m = {"practice": 0.2, "instruction": 0.2, "feedback": 0.2, "constraint": 0.4}
                blocked = 3
            elif attacked and which == "randomized_path":
                # the mixture is scrambled toward uniform while the skill target is unchanged, so
                # the path carries less and less signature and the skill match still holds
                m = {k: 0.25 for k in LH.SOURCES if k != "constraint"}
            lr, cur, info = LH.train_to_skill(truth, m, sub, blocked_k=blocked)
            obs = LH.observe(lr, sub, n=max(24, s["makers"] * 3))
            post = LH.history_posterior(obs, truth, mixes, sub, n_sim=s["sims"])
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                         "attacked": "yes" if attacked else "no", "architecture": "history_reader",
                         "log_score": float(np.log(max(post[mix], 1e-12))),
                         "correct": float(max(post, key=post.get) == mix),
                         "final_skill": info["final_skill"], "n": 1})
    return rows


def _value_estimand(ctx, attacked: bool):
    r = rng(ctx, "value")
    rows = []
    for _ in range(max(3, sizes(ctx)["makers"] // 5)):
        w = PS.sample_value_world(np.random.default_rng(r.integers(0, 2 ** 62)))
        rv = PS.make_rivals(w, r)
        if attacked:                       # preference reversed while the residue persists
            rv = {"changed_preference": rv["changed_preference"],
                  "stale_residue": rv["stale_residue"].copy_with(
                      preference=-w.preference)}
        else:
            rv = {"changed_preference": rv["changed_preference"],
                  "stale_residue": rv["stale_residue"]}
        for name in rv:
            obs = [PS.choose(rv[name], r, public=True)["choice"] for _ in range(12)]
            post = PS.rival_posterior(obs, rv, r, n_sim=sizes(ctx)["sims"])
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                         "attacked": "yes" if attacked else "no", "architecture": "rival_reader",
                         "log_score": float(np.log(max(post[name], 1e-12))),
                         "correct": float(max(post, key=post.get) == name), "n": 1})
    return rows


def _control_estimand(ctx, attacked: bool):
    r = rng(ctx, "control")
    s = sizes(ctx)
    cw = FGN.collision_world(r)
    w = cw["world"]
    rows = []
    for arch in FGN.ARCHITECTURES:
        for _ in range(max(3, s["makers"] // 3)):
            at = (s["steps"] // 2) if attacked else None
            ep = FGN.rollout(w, arch, r, s["steps"], interrupt_at=at)
            post = FGN.architecture_posterior(ep, w, r, n_sim=s["sims"])
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                         "attacked": "yes" if attacked else "no",
                         "architecture": "control_reader",
                         "log_score": float(np.log(max(post[arch], 1e-12))),
                         "correct": float(max(post, key=post.get) == arch), "n": 1})
    return rows


def _source_estimand(ctx, attacked: bool, which: str):
    w = world_for(ctx, "communication", kappa=0.0, dose=8)
    r = rng(ctx, f"source|{which}")
    s = sizes(ctx)
    probes = (("private_cost", "correction", "evidence_choice") if which == "audience"
              else SS.PROBES)
    rows = []
    for _ in range(max(3, s["makers"] // 3)):
        lat = WC.sample_latent(w, r)
        ep = WC.rollout(w, lat, r, s["steps"])
        post = EX.joint_posterior(WC, w, ep, min(8, s["steps"]))
        klass = WC.collision_class(lat.tendency)
        truth = WC.motive_of(lat.tendency)
        use = probes
        if attacked and which == "audience":
            use = ("audience_persuaded",)          # the audience is already persuaded or absent
        cur = post
        for pr in use:
            out = SS.probe_value(w, lat, cur, pr, r)
            if attacked and which == "noise":
                cur = C.softmax((0.4 * np.log(np.maximum(out["posterior"], 1e-300))
                                 + 0.6 * np.log(np.maximum(cur, 1e-300))).ravel()
                                ).reshape(cur.shape)
            else:
                cur = out["posterior"]
        mp = WC.motive_posterior(w, cur)
        inside = {m: mp[m] for m in klass}
        if attacked and which == "audience_model":
            true_aud = C.normalize(np.abs(w.evidence_support[:, lat.goal]) + 0.1)
            am = SS.AudienceModel.make(w, true_aud, 0.0, r)
            aw = SS.audience_aware_reader(w, ep, 4, am)
            sc = C.log_score(aw, lat.goal)
        else:
            sc = float(np.log(max(mp[truth], 1e-12)))
        rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                     "attacked": "yes" if attacked else "no", "architecture": "source_reader",
                     "log_score": sc,
                     "correct": float(max(inside, key=inside.get) == truth), "n": 1})
    return rows


def _foraging_estimand(ctx, attacked: bool, which: str):
    r = rng(ctx, f"foraging|{which}")
    s = sizes(ctx)
    rows = []
    for pol in ("progress", "gain_per_cost", "changepoint"):
        for _ in range(max(2, s["makers"] // 6)):
            sub = np.random.default_rng(r.integers(0, 2 ** 62))
            if which == "changepoint":
                eco = "silent_change" if attacked else "learnable"
            else:
                eco = "noise" if attacked else "learnable"
            items = FG.make_ecology(eco, sub, n_items=9)
            out = FG.forage(items, pol, sub, steps=60 if ctx.get("smoke") else 120)
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"],
                         "attacked": "yes" if attacked else "no", "architecture": pol,
                         "log_score": out["held_out_gain"],
                         "correct": 1.0 - out["fraction_on_noise"], "n": 1})
    return rows


def _architecture_estimand(ctx, attacked: bool, over: dict, impoverish: bool = False):
    rows = []
    for fam in families_of(ctx):
        knobs = {"kappa": 0.5, "overlap": 0.33, "dose": 4}
        cfg = {"model_space": "correct"}
        if attacked:
            knobs.update(over)
            cfg["model_space"] = knobs.get("model_space", "correct")
            if impoverish:
                cfg["n_particles"] = 12                     # particle impoverishment
        r, _, _ = run_tournament(ctx, fam,
                                 ("surface", "independent", "joint_exact", "particle",
                                  "oracle_model_space", "oracle_state"),
                                 knobs_over=knobs, cfg=cfg,
                                 extra_key={"attacked": "yes" if attacked else "no",
                                            "family": fam})
        rows += r
    return rows


ESTIMANDS = {
    "coupling_atlas": lambda ctx, a, over: _atlas_estimand(ctx, a, over),
    "routing": lambda ctx, a, over: _routing_estimand(ctx, a, over),
    "architecture": lambda ctx, a, over: _architecture_estimand(ctx, a, over),
}


def _generic_unit(ctx, attack_id: str, **kw):
    flight, over = TARGETS[attack_id]
    fn = ESTIMANDS.get(flight)
    if fn is None:
        raise KeyError(flight)
    return {"rows": fn(ctx, False, over) + fn(ctx, True, over), "flight": flight}


def _attack_reduce(ctx, units, what, *, pair=("joint_exact", "independent"), value="log_score",
                   claim="METHOD", extra=None):
    """The shared reduce: attacked minus unattacked, in the flight's own units."""
    card = ctx["card"]
    rows = rows_of(units)
    a, b = pair
    v = start(card, ctx, f"the targeted result survives: {card.construction}", claim)
    gr = G.GateReport()

    def gap(att):
        return (mean_of(rows, value,
                        lambda r, att=att: r.get("attacked") == att
                        and r.get("architecture") == a)
                - mean_of(rows, value,
                          lambda r, att=att: r.get("attacked") == att
                          and r.get("architecture") == b))
    un, at = gap("no"), gap("yes")
    loss = un - at
    battery(gr, live={"name": "the_attack_reaches_the_measurement", "observed": abs(loss)},
            placebo={"name": "the_unattacked_arm_is_the_control", "observed": 0.0, "tol": 0.0},
            positive={"name": "both_arms_produced_an_estimand",
                      "observed": float(un == un and at == at), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "the_attack_did_not_change_the_endpoint", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_hidden_event_is_the_same_one", "observed": abs(at)})
    criterion(v, card.id, at, card.sesoi, "greater", card.sesoi_basis,
              detail="under the attack the targeted advantage still clears the flight's own bar")
    criterion(v, f"{card.id}_loss", loss, 10.0, "less",
              "how much the attack removed, in the flight's own units",
              detail="and the attack's cost is reported whether or not the result survives")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["estimand"] = {"unattacked": un, "attacked": at, "loss": loss,
                                "pair": [a, b], "value": value}
    v["results"]["by_architecture"] = {
        nm: {att: mean_of(rows, value,
                          lambda r, nm=nm, att=att: r.get("architecture") == nm
                          and r.get("attacked") == att)
             for att in ("no", "yes")}
        for nm in sorted({r.get("architecture") for r in rows if r.get("architecture")})}
    narrative(v, what.format(un=un, at=at, loss=loss),
              "an attack that costs nothing was not an attack")
    distances(v, card.id, CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# X01-X11: the atlas, routing and architecture attacks.
# --------------------------------------------------------------------------- #
def unit_X01(ctx):
    """Surface names permuted. The estimand must be invariant, not merely survive."""
    rows = _atlas_estimand(ctx, False, {})
    att = _atlas_estimand(ctx, True, {})
    for r in att:
        r["attacked"] = "yes"
    return {"rows": rows + att, "flight": "coupling_atlas"}


def reduce_X01(units, ctx):
    return _attack_reduce(ctx, units,
                          "renaming the surface leaves the joint advantage at {at:+.4f} nats "
                          "against {un:+.4f} unattacked",
                          extra=[("X01_invariance", abs(
                              mean_of(rows_of(units), "log_score",
                                      lambda r: r.get("attacked") == "yes")
                              - mean_of(rows_of(units), "log_score",
                                        lambda r: r.get("attacked") == "no")), 0.35, "less",
                              "score movement a pure relabelling may cause",
                              "a permutation of names is not supposed to move the score at all")])


for _xid in ("X02", "X03", "X04", "X05", "X06", "X07", "X08", "X09", "X10", "X11"):
    def _mk(xid=_xid):
        def unit(ctx):
            return _generic_unit(ctx, xid)

        def reduce_(units, ctx):
            flight = TARGETS[xid][0]
            return _attack_reduce(
                ctx, units,
                "under the attack the targeted advantage is {at:+.4f} nats against {un:+.4f} "
                "unattacked, a loss of {loss:+.4f}",
                pair=PAIRS[flight])
        return unit, reduce_
    globals()[f"unit_{_xid}"], globals()[f"reduce_{_xid}"] = _mk()


# --------------------------------------------------------------------------- #
# X12-X20: the trunk-specific attacks.
# --------------------------------------------------------------------------- #
def unit_X12(ctx):
    return {"rows": _expertise_estimand(ctx, False, "attention_for_constraint")
            + _expertise_estimand(ctx, True, "attention_for_constraint"),
            "flight": "expertise"}


def reduce_X12(units, ctx):
    return _attack_reduce(ctx, units,
                          "swapping attention for an imposed constraint leaves history recovery at "
                          "{at:+.4f} against {un:+.4f}",
                          pair=("history_reader", "history_reader"), value="correct")


def unit_X13(ctx):
    return {"rows": _expertise_estimand(ctx, False, "randomized_path")
            + _expertise_estimand(ctx, True, "randomized_path"), "flight": "expertise"}


def reduce_X13(units, ctx):
    return _attack_reduce(ctx, units,
                          "randomizing the training path with skill rematched leaves history "
                          "recovery at {at:+.4f} against {un:+.4f}",
                          pair=("history_reader", "history_reader"), value="correct")


def unit_X14(ctx):
    return {"rows": _value_estimand(ctx, False) + _value_estimand(ctx, True), "flight": "value"}


def reduce_X14(units, ctx):
    return _attack_reduce(ctx, units,
                          "reversing the preference while the residue persists leaves rival "
                          "discrimination at {at:+.4f} against {un:+.4f}",
                          pair=("rival_reader", "rival_reader"), value="correct")


def unit_X15(ctx):
    return {"rows": _control_estimand(ctx, False) + _control_estimand(ctx, True),
            "flight": "control"}


def reduce_X15(units, ctx):
    return _attack_reduce(ctx, units,
                          "interrupting and restoring the foreground goal leaves architecture "
                          "recovery at {at:+.4f} against {un:+.4f}",
                          pair=("control_reader", "control_reader"), value="correct")


def unit_X16(ctx):
    return {"rows": _source_estimand(ctx, False, "audience")
            + _source_estimand(ctx, True, "audience"), "flight": "source"}


def reduce_X16(units, ctx):
    return _attack_reduce(ctx, units,
                          "an already-persuaded audience leaves motive separation at {at:+.4f} "
                          "against {un:+.4f}",
                          pair=("source_reader", "source_reader"), value="correct")


def unit_X17(ctx):
    return {"rows": _source_estimand(ctx, False, "noise")
            + _source_estimand(ctx, True, "noise"), "flight": "source"}


def reduce_X17(units, ctx):
    return _attack_reduce(ctx, units,
                          "making the private action noisy and costly leaves motive separation at "
                          "{at:+.4f} against {un:+.4f}",
                          pair=("source_reader", "source_reader"), value="correct")


def unit_X18(ctx):
    return {"rows": _source_estimand(ctx, False, "audience_model")
            + _source_estimand(ctx, True, "audience_model"), "flight": "source"}


def reduce_X18(units, ctx):
    return _attack_reduce(ctx, units,
                          "a strategic speaker with the wrong assumed audience model leaves the "
                          "reader at {at:+.4f} against {un:+.4f}",
                          pair=("source_reader", "source_reader"), value="log_score")


def unit_X19(ctx):
    return {"rows": _foraging_estimand(ctx, False, "changepoint")
            + _foraging_estimand(ctx, True, "changepoint"), "flight": "foraging"}


def reduce_X19(units, ctx):
    return _attack_reduce(ctx, units,
                          "a silent changepoint inside a settled item leaves realized gain at "
                          "{at:+.4f} against {un:+.4f}",
                          pair=("changepoint", "progress"), value="log_score")


def unit_X20(ctx):
    return {"rows": _foraging_estimand(ctx, False, "noise")
            + _foraging_estimand(ctx, True, "noise"), "flight": "foraging"}


def reduce_X20(units, ctx):
    return _attack_reduce(ctx, units,
                          "unlearnable noise with high surprise leaves gain-per-cost ahead of "
                          "progress by {at:+.4f} against {un:+.4f}",
                          pair=("gain_per_cost", "progress"), value="log_score")


# --------------------------------------------------------------------------- #
# X21 — the attack on the reporting.
# --------------------------------------------------------------------------- #
def unit_X21(ctx):
    """Build a world whose effect reverses sign across a stratum, then check the pooled mean."""
    r = rng(ctx, "X21")
    s = sizes(ctx)
    rows = []
    for stratum in ("low_dose", "high_dose"):
        dose = 1 if stratum == "low_dose" else 16
        for fam in families_of(ctx)[:1]:
            # the unattacked arm is the same design reported conditionally; the attacked arm is
            # the same numbers pooled. Both levels of the declared factor have to be realized or
            # the card blocks its own receipt.
            for attacked in ("no", "yes"):
                rr, _, _ = run_tournament(ctx, fam, ("independent", "joint_exact"),
                                          knobs_over={"kappa": 1.0, "overlap": 0.0, "dose": dose},
                                          extra_key={"stratum": stratum, "attacked": attacked})
                for row in rr:
                    row["stratum"] = stratum
                rows += rr
    # deliberately unequal stratum sizes, which is how a pooled mean hides a reversal
    small = [r_ for r_ in rows if r_["stratum"] == "low_dose"][: max(2, len(rows) // 8)]
    big = [r_ for r_ in rows if r_["stratum"] == "high_dose"]
    return {"rows": rows, "weighted": small + big, "flight": "reporting"}


def reduce_X21(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    weighted = rows_of(units, "weighted")
    v = start(card, ctx, "a pooled mean can hide a sign reversal that the conditional report shows",
              "METHOD")
    gr = G.GateReport()

    def gap(rs, st=None):
        sel = [r for r in rs if st is None or r.get("stratum") == st]
        return (mean_of(sel, "log_score", lambda r: r.get("architecture") == "joint_exact")
                - mean_of(sel, "log_score", lambda r: r.get("architecture") == "independent"))
    lo, hi = gap(rows, "low_dose"), gap(rows, "high_dose")
    pooled = gap(weighted)
    reverses = float((lo > 0) != (hi > 0))
    hidden = float(reverses and abs(pooled) < max(abs(lo), abs(hi)) / 2)
    battery(gr, live={"name": "the_strata_disagree", "observed": abs(lo - hi)},
            placebo={"name": "both_strata_were_measured_the_same_way", "observed": 0.0, "tol": 0.0},
            positive={"name": "the_conditional_report_shows_both",
                      "observed": float(lo == lo and hi == hi), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "the_stratum_is_a_declared_factor", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_same_hidden_event_in_both_strata",
                        "observed": abs(lo - hi)})
    criterion(v, "X21", abs(lo - hi), card.sesoi, "greater", card.sesoi_basis,
              detail="the two strata disagree by at least the bar, so a pooled headline would be "
                     "reporting a number that describes neither of them")
    v["conditional_matrix"] = {"axis_rows": "stratum",
                               "surface": {"low_dose": lo, "high_dose": hi},
                               "pooled_mean_if_reported": pooled,
                               "pooled_headline": "REFUSED",
                               "sign_reverses": bool(reverses),
                               "pooled_would_have_hidden_it": bool(hidden)}
    narrative(v, f"the advantage is {lo:+.4f} nats at one observation and {hi:+.4f} at sixteen; a "
                 f"size-weighted pooled mean would have reported {pooled:+.4f}",
              "the rule against pooled headlines is enforced against a case built to break it")
    distances(v, "X21", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# X22 — solver approximation and particle impoverishment.
# --------------------------------------------------------------------------- #
def unit_X22(ctx):
    return {"rows": _architecture_estimand(ctx, False, {})
            + _architecture_estimand(ctx, True, {}, impoverish=True), "flight": "architecture"}


def reduce_X22(units, ctx):
    rows = rows_of(units)
    part = {att: mean_of(rows, "log_score",
                         lambda r, att=att: r.get("architecture") == "particle"
                         and r.get("attacked") == att)
            for att in ("no", "yes")}
    v = _attack_reduce(ctx, units,
                       "with the particle count starved the exact reader's advantage is {at:+.4f} "
                       "against {un:+.4f}",
                       extra=[("X22_particle_cost", part["no"] - part["yes"], 0.0, "greater",
                               "score the particle reader loses when its population is starved",
                               "starving the filter costs it measurably, so the attack reached "
                               "the thing it was aimed at")])
    v["results"]["particle_score"] = part
    return v


# --------------------------------------------------------------------------- #
# X23 — the lineage attack.
# --------------------------------------------------------------------------- #
def unit_X23(ctx):
    """Try to make a confirmation object share an ancestor with a discovery object."""
    from ..schemas import TIERS
    t = ctx.get("tier") or TIERS["T0"]
    ids = {ln: C.lane_ids(ln, t) for ln in ("discovery", "transfer", "confirmation", "coverage")}
    disjoint = C.lineage_disjoint(ids)
    # the attack: ask for a discovery world id under the confirmation lane
    refused = False
    try:
        C.world_seed("confirmation", ids["discovery"][0])
    except AssertionError:
        refused = True
    # and check the seeds themselves differ even for the same integer id
    same_id = ids["discovery"][0]
    s_disc = C.seed(f"world|discovery|{same_id}")
    s_conf = C.seed(f"world|confirmation|{same_id}")
    rows = [{"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "no", "check": "ranges_disjoint",
             "ok": float(disjoint), "n": 1},
            {"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "yes", "check": "cross_lane_id_refused",
             "ok": float(refused), "n": 1},
            {"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "yes", "check": "seeds_differ_by_lane",
             "ok": float(s_disc != s_conf), "n": 1}]
    return {"rows": rows, "flight": "lineage", "ids": {k: [v_[0], v_[-1]] for k, v_ in ids.items()}}


def reduce_X23(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a lineage or seed swap is refused rather than silently accepted",
              "METHOD")
    gr = G.GateReport()
    worst = min((r["ok"] for r in rows), default=0.0)
    battery(gr, positive={"name": "every_lineage_check_holds", "observed": worst, "expected": 1.0,
                          "tol": 1e-9},
            placebo={"name": "the_lane_ranges_do_not_overlap",
                     "observed": float(1.0 - rows[0]["ok"]), "tol": 0.0},
            live={"name": "a_cross_lane_request_is_refused",
                  "observed": float(rows[1]["ok"])})
    criterion(v, "X23", worst, 1.0, "greater", "exact: every lineage check must hold",
              detail="the lane id ranges are disjoint, a cross-lane world id is refused, and the "
                     "same integer id seeds differently in different lanes")
    v["results"]["checks"] = {r["check"]: r["ok"] for r in rows}
    v["results"]["lane_ranges"] = units[0]["ids"]
    narrative(v, "the lane ranges are disjoint, a cross-lane world id raises, and the same id "
                 "produces different seeds per lane",
              "a confirmation cannot be run on a discovery object by accident")
    distances(v, "X23", [{"name": "lineage_bookkeeping", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# X24 — runtime failure injection.
# --------------------------------------------------------------------------- #
def unit_X24(ctx):
    from .. import runtime_contract as RC
    rows = []
    # fast machine: the guard must refuse a queue that a fast machine would empty
    g_fast = RC.opening_guard(core_upper_h=40, core_lower_h=12, coverage_lower_h=90,
                              confirmation_worker_h=30, hashed=True, recovery_tests=True)
    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "yes", "check": "fast_machine_refused",
                 "ok": float(not g_fast["may_open"]), "n": 1})
    # restart: the deadline is inherited, not reset
    w1 = RC.window()
    inherited = True
    if w1:
        w2 = RC.open_window()                       # a restart must not move the deadline
        inherited = (w2["deadline"] == w1["deadline"])
    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "yes", "check": "deadline_inherited",
                 "ok": float(inherited), "n": 1})
    # orphan kill: the runner must be launched as a MODULE. The sibling project's orphan sweeper
    # kills any python whose command line matches a runners/run_ script path, which killed the V14
    # runner seven times and cost that program its first window. A module-form launch does not
    # match. This checks the launcher on disk rather than trusting the memory of a fix.
    from .. import REPO
    ps1 = REPO / "runners" / "run_v15_wrapped.ps1"
    launch_ok = bool(ps1.exists() and "-m" in ps1.read_text(encoding="utf-8")
                     and "runners.run_v15" in ps1.read_text(encoding="utf-8"))
    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "yes", "check": "module_form_launch",
                 "ok": float(launch_ok), "n": 1})
    # stale checkpoint: a checkpoint whose source hash differs is refused
    stale = C.load_ckpt("smoke", "X24", 0, 0, "not-the-right-hash")
    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "yes", "check": "stale_checkpoint_refused",
                 "ok": float(stale is None), "n": 1})
    # runtime failure is expressible and cannot be softened
    occ = RC.Occupancy()
    occ.note_queue_empty()
    failed, reasons = occ.runtime_failed()
    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "yes",
                 "check": "runtime_failure_is_expressible",
                 "ok": float(failed and bool(reasons)), "n": 1})
    rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "attacked": "no", "check": "healthy_guard_admits",
                 "ok": float(RC.opening_guard(core_upper_h=90, core_lower_h=30,
                                              coverage_lower_h=400, confirmation_worker_h=30,
                                              hashed=True, recovery_tests=True)["may_open"]),
                 "n": 1})
    return {"rows": rows, "flight": "runtime"}


def reduce_X24(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the runtime machinery refuses every injected failure", "METHOD")
    gr = G.GateReport()
    worst = min((r["ok"] for r in rows), default=0.0)
    battery(gr, positive={"name": "every_injected_failure_is_refused", "observed": worst,
                          "expected": 1.0, "tol": 1e-9},
            placebo={"name": "a_healthy_queue_is_still_admitted",
                     "observed": float(1.0 - [r["ok"] for r in rows
                                              if r["check"] == "healthy_guard_admits"][0]),
                     "tol": 0.0},
            live={"name": "the_guard_distinguishes_the_cases",
                  "observed": float(len({r["ok"] for r in rows}) >= 1)})
    criterion(v, "X24", worst, 1.0, "greater", "exact: every injected failure must be refused",
              detail="a fast-machine queue is refused, a restart inherits the deadline, the runner "
                     "launches as a module, a stale checkpoint is recomputed, and a runtime "
                     "failure is expressible")
    v["results"]["checks"] = {r["check"]: r["ok"] for r in rows}
    narrative(v, "fast machine refused, deadline inherited, module-form launch present, stale "
                 "checkpoint recomputed, runtime failure expressible",
              "the machinery that protects the other attacks is itself attacked")
    distances(v, "X24", [{"name": "runtime_machinery", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
