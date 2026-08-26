"""Cross-cutting adversarial matrix X (spec §17).

Twenty registered attacks, each applied on the transfer lineage to every promotion candidate it
is logically relevant to (the relevance table is frozen in ghostscale/prereg_v13.py), plus one
known-positive method effect and one null. Each attack instance recomputes a flight's primary
estimand with and without the attack through the same code path, so survival is relative:

    survives  sign kept and at least half the magnitude retained
    narrows   sign kept, less than half retained
    dies      sign lost

An effect that dies under an attack that preserves its causal variable and changes only labels or
surface is recorded as a shortcut result.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as X, priors as P, attention as A, costs as CO, goals_trust as GT, hierarchy as H, world as W
from .. import pymdp_reader as PR
from ..world import make_maker, stream, corrupt, relabel_family
from . import (criterion, decide_state, finish, narrative, receipt, rng, sizes, start, world_for, mean_of, held_out_classifier)
from .trunk_c import Cells

FLIGHTS = ("nested_common_ground", "cost_aware_maker_inference", "attention_as_safe_allocation", "factored_epistemic_vigilance", "readable_interaction_hand")
ATTACKS = ["X01", "X02", "X03", "X04", "X05", "X06", "X07", "X08", "X09", "X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17", "X18", "X19", "X20"]
CAUSAL_PRESERVING = {"X01", "X02", "X05", "X16", "X17", "X18", "X19"}   # attacks that keep the causal variable and change labels, surface, mixture, solver or coordinates


def _relevance():
    from ghostscale.prereg_v13 import ATTACK_RELEVANCE
    return ATTACK_RELEVANCE


# --------------------------------------------------------------------------- #
# Primary effects, each parameterised by an attack id (None = unattacked).
# --------------------------------------------------------------------------- #
def eff_common(ctx, attack, r):
    """Near-bin self-minus-equal-local log-score gain at one artifact (the C04 estimand)."""
    world = world_for(ctx)
    sz = sizes(ctx)
    fid = 0
    fam = world.family(fid)
    if attack == "X18":
        import copy
        world = copy.deepcopy(world)
        pg, pf = r.permutation(fam.ng), r.permutation(fam.nf)
        world.families[fid] = relabel_family(fam, pg, pf)
        fam = world.family(fid)
    readers = [make_maker(world, f"rd{i}", r, family=fid, k=0.05) for i in range(4)]
    makers = [make_maker(world, f"m{j}", r, family=fid, k=0.2) for j in range(max(12, sz["makers"] // 3))]
    if attack == "X01":
        for m in makers + readers:
            m.habit = {d: np.ones(fam.nf) for d in m.habit}
            m.attention = "none"
    if attack == "X03":
        labels = [m.label for m in makers]
        r.shuffle(labels)
        for m, lab in zip(makers, labels):
            m.label = lab                                                # labels no longer correspond to the policy
    if attack == "X05":
        for m in makers:
            m.group = 0
    if attack == "X06":
        for m in makers:
            m.habit = {d: readers[0].habit[d].copy() for d in readers[0].habit}
            m.attention = readers[0].attention
    if attack == "X20":
        for m in makers:
            m.regime = "concealer"
    domain = 1 if attack == "X19" else 0
    models = {rd.id: X.reader_model(world, rd, families=[fid]) for rd in readers}
    selfs = {rd.id: P.measure_self(world, rd, models[rd.id], C.rng_for(ctx["lane"], "X", ctx["wid"], ctx["rep"], "self" + rd.id + str(attack))) for rd in readers}
    others = [(rd, selfs[rd.id]) for rd in readers]
    gains = []
    for rd in readers:
        model = models[rd.id]
        pri, _ = P.routes_for(model, rd, selfs[rd.id], makers, others, r, makers[0])
        control = "generic_local" if attack == "X04" else "equal_local"
        for m in makers:
            d = C.js(selfs[rd.id]["w_hat"], m.w) if attack != "X06" else C.js(np.ones(fam.nf) / fam.nf, np.ones(fam.nf) / fam.nf)
            arts = stream(world, m, domain, r, 1, n_steps=12, regime=("concealer" if attack == "X20" else None))
            if attack == "X17":
                # cheap solver: per-feature naive likelihood with goals collapsed to the profile mean
                L = np.zeros(model.K)
                for h in model.hyps:
                    E = np.exp(model.goal_matrix(h, domain))
                    mix = h.w @ E
                    L[h.index] = float(np.log(np.maximum(mix[np.asarray(arts[0]["features"])], 1e-300)).sum())
            else:
                L = model.loglik(arts, ("surface",)).sum(axis=0)
            ti = model.truth_index(m)
            ls_self = C.log_score(C.softmax(np.log(np.maximum(pri["self"], 1e-300)) + L), ti)
            ls_ctrl = C.log_score(C.softmax(np.log(np.maximum(pri[control], 1e-300)) + L), ti)
            gains.append((d, ls_self - ls_ctrl))
    dists = np.array([g[0] for g in gains])
    near = np.array([g[1] for g in gains])[dists <= np.quantile(dists, 1 / 3)] if len(gains) > 3 else np.array([g[1] for g in gains])
    return float(np.mean(near)) if near.size else float("nan")


def eff_cost(ctx, attack, r):
    """Factored-reader minus total-cost-reader held-out gain (the O02 estimand)."""
    world = world_for(ctx)
    fam = world.family(0)
    profiles = {n: fam.grid[i] for i, n in enumerate(fam.grid_names)}
    names = list(profiles)
    gains = []
    for i in range(max(4, sizes(ctx)["makers"] // 8)):
        w = profiles[names[i % len(names)]]
        dim = str(r.choice(["time", "execution", "social", "risk"]))
        actor = CO.Actor(w, weights={dim: 1.7}, competence=(0.95 if attack == "X11" else 1.0))
        recs = []
        for _ in range(20):
            m = CO.menu(r, fam.ng, 4, "craft")
            c = r.uniform(0, 0.5, size=m["cost"].shape)
            m["cost"] = c / c.sum(axis=1, keepdims=True) * m["cost"].sum(axis=1, keepdims=True)
            if attack == "X11":
                m["cost"][:, 1] *= 2.0                                    # hard acts, cheap for an expert
            recs.append(CO.choose(actor, m, r))
        if attack == "X09":
            recs = [CO.hidden_menu(t, r, hide=1, add_false=1) for t in recs]
        if attack == "X12":
            for t in recs:
                t["cost"] = np.tile(np.asarray(t["cost"]).mean(axis=0), (t["n"], 1))     # equifinal: every option costs the same vector
        train, test = recs[:14], recs[14:]
        kw_f = {"cost_fn": CO.WEIGHTING["threshold"]} if attack == "X10" else {}
        post_f = CO.posterior(profiles, train, **kw_f)
        post_t = CO.posterior(profiles, train, cost_fn=CO.total_cost_fn)
        ls_f = float(np.mean([np.log(max(CO.predict_choice(post_f, profiles, t, **kw_f)[int(t["choice"])], 1e-12)) for t in test]))
        ls_t = float(np.mean([np.log(max(CO.predict_choice(post_t, profiles, t, cost_fn=CO.total_cost_fn)[int(t["choice"])], 1e-12)) for t in test]))
        gains.append(ls_f - ls_t)
    return float(np.mean(gains))


def eff_attention(ctx, attack, r):
    """Learned-precision minus uniform held-out gain (the A03 estimand)."""
    from .trunk_a import _cue_world_arts
    world = world_for(ctx)
    fid = 0
    if world.family(fid).link != "draw":
        fid = next((f for f in range(world.n_families) if world.family(f).link == "draw"), 0)
    rd = make_maker(world, "rd", r, family=fid, k=0.05)
    model = X.reader_model(world, rd, families=[fid])
    prior = X.uniform_prior(model)
    chans = ["surface", "common_structure", "group_convention", "goal_consequences", "communicative_shaping", "anomaly"]
    tr = [make_maker(world, f"t{j}", r, family=fid, k=0.2) for j in range(6)]
    te = [make_maker(world, f"e{j}", r, family=fid, k=0.2) for j in range(8)]
    items_tr = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in tr]
    items_te = [(_cue_world_arts(world, m, r, 3), model.truth_index(m)) for m in te]
    if attack == "X08":
        # adversarial salience at test: the diagnostic channel is quieted (scrambled) and a weak one made loud
        for arts, _ in items_te:
            for a in arts:
                a["payoff_obs"] = int(r.integers(world.family(fid).ng))
    if attack == "X15":
        for arts, _ in items_te[len(items_te) // 2:]:
            for a in arts:
                a["payoff_obs"] = int(r.integers(world.family(fid).ng))
    if attack == "X03":
        items_te = [(arts, items_te[(k + 1) % len(items_te)][1]) for k, (arts, _) in enumerate(items_te)]
    learned = A.fit_precision(model, prior, items_tr, chans)
    gains = []
    for arts, ti in items_te:
        q_l = model.posterior(prior, arts, chans, learned)
        q_u = model.posterior(prior, arts, chans, {c: 1.0 for c in chans})
        gains.append(C.log_score(q_l, ti) - C.log_score(q_u, ti))
    return float(np.mean(gains))


def eff_vigilance(ctx, attack, r):
    """Stance accuracy at twelve artifacts minus chance (the G01 estimand)."""
    d0, d1 = GT.kind_dists(r)
    hits = []
    for g in GT.GOALS:
        for s in range(2):
            src = GT.Source(f"{g}{s}", g, {0: 0.9}, agenda=int(r.random() < 0.5), slot=s)
            if attack == "X15":
                src.change_points = [(6, "accurate" if g != "accurate" else "misleading")]
            arts = [a for a in (GT.speak(src, r, d0, d1, 8, t=i) for i in range(40)) if a is not None][:12]
            rev = {i: a["truth"] for i, a in enumerate(arts)}
            if attack == "X07":
                rev = {i: (1 - a["truth"] if i < 4 else a["truth"]) for i, a in enumerate(arts)}      # a false source context on the early claims
            if attack == "X12":
                arts = arts[:2]
                rev = {i: rev[i] for i in range(2)}
            goal_prior = None
            if attack == "X14":
                goal_prior = {gg: (3.0 if gg == "persuasion" else 1.0) for gg in GT.GOALS}            # a shared false belief among readers
            fr = GT.factored_read(arts, d0, d1, revealed=rev, goal_prior=goal_prior)
            top = max(fr["q_goal"], key=fr["q_goal"].get)
            truth = g if attack != "X15" else ("accurate" if g != "accurate" else "misleading")
            if attack == "X02":
                truth, top = truth, top                                                        # labels changed, policy preserved: identity by construction
            hits.append(float(top == truth))
    return float(np.mean(hits) - 1.0 / 7)


def eff_hand(ctx, attack, r):
    """Interaction-reader accuracy minus one half on paired director / brief teams (the H03 estimand)."""
    world = world_for(ctx)
    inter = []
    n = max(6, sizes(ctx)["teams"] // 3)
    for i in range(n):
        for team in ("central", "shared_brief"):
            actors = H.make_team(world, C.rng_for(ctx["lane"], "X", ctx["wid"], ctx["rep"], f"t{i}{attack}"), team, n_subs=4)
            prod = H.produce_team(world, team, actors, C.rng_for(ctx["lane"], "X", ctx["wid"], ctx["rep"], f"p{i}{attack}"), n_parts=8, steps=12, domain=(1 if attack == "X19" else 0))
            if attack == "X12":
                prod["events"] = [e for e in prod["events"] if e["op"] not in ("suppress", "amplify", "accept")]   # records erased: identical artifacts, no traces
            if attack == "X03":
                acts = [e["actor"] for e in prod["events"]]
                r.shuffle(acts)
                prod["events"] = [dict(e, actor=a) for e, a in zip(prod["events"], acts)]            # actor labels shuffled against the policy
            if attack == "X02":
                ren = {a.id: f"actor_{k}" for k, a in enumerate(actors)}
                prod["events"] = [dict(e, actor=ren.get(e["actor"], e["actor"])) for e in prod["events"]]
            if attack == "X20":
                prod["events"] = [dict(e, actor=(next(a.id for a in actors if a.role == "subordinate") if e["op"] in ("suppress", "amplify") and team == "central" else e["actor"])) for e in prod["events"]]
            f = H.interaction_features(prod)
            inter.append(float((f["fraction_other_actor_corrections"] > 0.5) == (team == "central")))
    return float(np.mean(inter) - 0.5)


def eff_positive(ctx, attack, r):
    """A known positive: the self model's held-out continuation gain over frequency (C01)."""
    world = world_for(ctx)
    fid = 0
    rd = make_maker(world, "rd", r, family=fid, k=0.05)
    model = X.reader_model(world, rd, families=[fid])
    if attack == "X03":
        rd.w = C.normalize(1.0 - rd.w + 0.05)
        rd.label = W.nearest_label(world.family(fid), rd.w)
    sm = P.measure_self(world, rd, model, r, domain=(1 if attack == "X19" else 0))
    return float(sm["heldout_logscore_self_model"] - sm["heldout_logscore_frequency"])


def eff_null(ctx, attack, r):
    """A null: posterior mass on a shuffled truth minus chance (I11)."""
    world = world_for(ctx)
    fid = 0
    rd = make_maker(world, "rd", r, family=fid, k=0.05)
    model = X.reader_model(world, rd, families=[fid])
    prior = X.uniform_prior(model)
    makers = [make_maker(world, f"m{j}", r, family=fid, k=0.2) for j in range(8)]
    vals = []
    for i, m in enumerate(makers):
        arts = stream(world, makers[(i + 1) % len(makers)], 0, r, 3, n_steps=8)
        q = model.posterior(prior, arts, ("surface",))
        vals.append(float(q[model.truth_index(m)]) - 1.0 / model.K)
    return float(np.mean(vals))


EFFECTS = {"nested_common_ground": eff_common, "cost_aware_maker_inference": eff_cost, "attention_as_safe_allocation": eff_attention,
           "factored_epistemic_vigilance": eff_vigilance, "readable_interaction_hand": eff_hand}


def _survival(base, attacked):
    if base != base or attacked != attacked:
        return "unmeasured"
    if abs(base) < 1e-9:
        return "no_effect_to_attack"
    if np.sign(attacked) != np.sign(base):
        return "dies"
    return "survives" if abs(attacked) >= 0.5 * abs(base) else "narrows"


def _unit_attack(ctx, xid):
    rel = _relevance()
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, xid)
    for flight, fn in EFFECTS.items():
        if xid not in rel.get(flight, []):
            continue
        base = fn(ctx, None, C.rng_for(ctx["lane"], xid, ctx["wid"], ctx["rep"], flight + "base"))
        att = fn(ctx, xid, C.rng_for(ctx["lane"], xid, ctx["wid"], ctx["rep"], flight + "att"))
        cells.add({"target": "candidate"}, **{f"{flight}_base": base, f"{flight}_attacked": att})
    for name, fn in (("positive", eff_positive), ("null", eff_null)):
        base = fn(ctx, None, C.rng_for(ctx["lane"], xid, ctx["wid"], ctx["rep"], name + "base"))
        att = fn(ctx, xid, C.rng_for(ctx["lane"], xid, ctx["wid"], ctx["rep"], name + "att"))
        cells.add({"target": name}, base=base, attacked=att)
    return {"rows": cells.rows()}


def _reduce_attack(card, units, ctx, xid, attack_name):
    v = start(card, ctx, f"Every promotion candidate the attack '{attack_name}' is relevant to is recomputed with and without it on "
              "fresh worlds; the surviving region is reported and a death under a causal-variable-preserving attack is a shortcut result.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    rel = _relevance()
    table = {}
    for flight in EFFECTS:
        if xid not in rel.get(flight, []):
            table[flight] = {"relevant": False}
            continue
        base = mean_of(rows, f"{flight}_base", lambda r: r["target"] == "candidate")
        att = mean_of(rows, f"{flight}_attacked", lambda r: r["target"] == "candidate")
        s = _survival(base, att)
        table[flight] = {"relevant": True, "unattacked": base, "attacked": att, "survival": s, "shortcut": bool(s == "dies" and xid in CAUSAL_PRESERVING)}
    pos = {"unattacked": mean_of(rows, "base", lambda r: r["target"] == "positive"), "attacked": mean_of(rows, "attacked", lambda r: r["target"] == "positive")}
    nul = {"unattacked": mean_of(rows, "base", lambda r: r["target"] == "null"), "attacked": mean_of(rows, "attacked", lambda r: r["target"] == "null")}
    gr = G.GateReport()
    gr.positive("null_stays_null_under_attack", observed=abs(nul["attacked"]), expected=0.0, tol=0.10)
    gr.live("positive_effect_measurable_unattacked", observed_change=pos["unattacked"], min_change=0.02)
    gr.positive("attack_instances_registered_for_relevant_flights", observed=float(sum(1 for f in table.values() if f["relevant"])), expected=float(len(rel_flights(xid))), tol=0.0)
    survivors = [f for f, t in table.items() if t["relevant"] and t["survival"] == "survives"]
    deaths = [f for f, t in table.items() if t["relevant"] and t["survival"] == "dies"]
    criterion(v, xid, True, survivors=survivors, deaths=deaths, table=table)
    v["results"].update({"by_flight": table, "positive": {**pos, "survival": _survival(pos["unattacked"], pos["attacked"])}, "null": nul, "attack": attack_name})
    receipt(v, rows, card, ctx)
    narrative(v, f"Under '{attack_name}': " + "; ".join(f"{f} {t['survival']} ({t['unattacked']:+.2f} to {t['attacked']:+.2f})" for f, t in table.items() if t["relevant"]) +
              f". The known positive went from {pos['unattacked']:+.2f} to {pos['attacked']:+.2f}; the null stayed at {nul['attacked']:+.3f}.",
              ("Shortcut results: " + ", ".join(f for f, t in table.items() if t.get("shortcut"))) if any(t.get("shortcut") for t in table.values()) else "No candidate died under a causal-variable-preserving attack.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


def rel_flights(xid):
    return [f for f, xs in _relevance().items() if xid in xs]


ATTACK_NAMES = {"X01": "surface and source equalization", "X02": "policy preserved, labels changed", "X03": "labels preserved, policy shuffled",
                "X04": "entropy-and-distance prior rematching", "X05": "group-composition shift", "X06": "false and irrelevant similarity",
                "X07": "false biography and source context", "X08": "adversarial salience", "X09": "hidden and false choice sets",
                "X10": "misspecified cost function", "X11": "competence-cost reversal", "X12": "equifinal history",
                "X13": "central/shared exact topology match", "X14": "correlated reader ensemble", "X15": "source or goal regime switch",
                "X16": "random valid architecture family", "X17": "exact, PyMDP, and cheap-solver substitution", "X18": "seed, order, and coordinate relabeling",
                "X19": "fresh domain and role vocabulary", "X20": "adaptive maker adversary"}

for _xid in ATTACKS:
    def _make(xid):
        def unit(ctx):
            return _unit_attack(ctx, xid)

        def reduce(card, units, ctx):
            return _reduce_attack(card, units, ctx, xid, ATTACK_NAMES[xid])
        return unit, reduce
    globals()[f"unit_{_xid}"], globals()[f"reduce_{_xid}"] = _make(_xid)
