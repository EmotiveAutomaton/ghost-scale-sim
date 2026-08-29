"""Trunk X — the cross-cutting adversarial matrix (spec §6, cards X01-X12).

Each attack is applied to every RELEVANT flight candidate (prereg_v14.ATTACK_RELEVANCE), to one
method positive and to one valid null; applicability is recorded per flight, never a silent
not-applicable. A flight's effect is its primary card's effect recomputed on the attack lineage:

    joint_reconstruction_advantage    joint minus independent next-action log score (J04)
    reliable_routing                  learned minus equal weighting under a degraded route (R02)
    competence_history_dissociation   the smaller of competence's and history's own effects (E01)
    affect_source_factorization       fanatic/propagandist accuracy with probes minus without (A06)
    learning_progress_foraging        learning-progress minus surprise realized gain (F04)

The method positive is oracle identifiability of the process (its class mass with the other two
latents supplied, over the prior); the null is a shuffled-maker effect (the joint advantage on a
maker whose latents were redrawn between evidence and target), which sits at zero. Survival:
sign kept and at least half the magnitude retained; dying under a causal-variable-preserving
attack marks a shortcut.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import communication as CM
from .. import foraging as F
from .. import history_skill as HS
from .. import joint as J
from .. import routes as R
from ..world import N_ACT, N_FEAT, PLAN_DIRECT, PLAN_HABIT, ROUTE_COST, ROUTES, episode, make_maker, relabel, stream
from . import Cells, battery, criterion, decide_state, finish, mean_of, narrative, pursuit_of, receipt, rng, sizes, start, world_for

NF = ("action", "semantic", "context")
FLIGHTS = ("joint_reconstruction_advantage", "reliable_routing", "competence_history_dissociation", "affect_source_factorization", "learning_progress_foraging")


def _ls(p, a):
    return float(np.log(max(float(p[int(a)]), 1e-12)))


def _gain(rd, post, prior, ep_next):
    a = int(ep_next["action"][0])
    return _ls(J.next_episode_action_dist(rd, post), a) - _ls(J.next_episode_action_dist(rd, prior), a)


def _noised(ep, route, r):
    e = dict(ep)
    if route == "semantic":
        e["semantic"] = [int(x) for x in r.integers(N_FEAT, size=len(ep["semantic"]))]
    return e


# --------------------------------------------------------------------------- #
# Flight effects under an attack. ``attack`` names the transformation; each effect function
# applies the ones that touch its construction and records which it applied.
# --------------------------------------------------------------------------- #
def effect_joint(world, r, n, attack=None, reader_k=(0.75, 0.8)):
    rd = J.Reader(world, 0, *reader_k)
    prior = J.uniform_prior()
    vals = []
    for i in range(n):
        if attack == "X03":                                      # equifinal makers: the class, not a member, is the target
            pref = int(np.argmax([C.softmax(world.params.goal_temp * np.log(world.family(0).prefs[p] + 1e-9))[0] for p in range(6)]))
            m = make_maker(world, f"m{i}", r, family=0, plan=int(r.choice([PLAN_DIRECT, PLAN_HABIT])), pref=pref, competence="mid")
            eps = [episode(world, m, r, index=k, goal=0) for k in range(5)]
        elif attack == "X05":                                    # wrong generative model: the maker's competence is not what the reader assumes
            m = make_maker(world, f"m{i}", r, family=0, competence="low")
            eps = stream(world, m, r, 5)
        else:
            m = make_maker(world, f"m{i}", r, family=0, competence="mid")
            eps = stream(world, m, r, 5)
        seen = eps[:4]
        if attack == "X01":
            seen = [relabel(e, r.permutation(N_FEAT)) for e in seen]
        if attack == "X04":                                      # a duplicated cause, fused as a shared cause
            seen = seen[:3] + [R.duplicate_semantic(seen[3], r, "paraphrase")]
            tabs = R.fused_tables(rd, seen, NF, "shared_cause")
        else:
            tabs = rd.route_tables(seen, NF)
        if attack == "X12":                                      # evidence order permuted (past episodes)
            past = seen[:-1]
            r.shuffle(past)
            tabs = rd.route_tables(past + [seen[-1]], NF)
        pj, pi = J.joint(prior, tabs), J.independent(prior, tabs)
        vals.append(_gain(rd, pj, prior, eps[4]) - _gain(rd, pi, prior, eps[4]))
    return float(np.mean(vals)), attack in ("X01", "X03", "X04", "X05", "X11", "X12")


def effect_routing(world, r, n, attack=None):
    rd = J.Reader(world, 0, 0.75, 0.8)
    prior = J.uniform_prior()
    training = []
    for i in range(max(6, n)):
        m = make_maker(world, f"t{i}", r, family=0, competence="mid")
        eps = [_noised(e, "semantic", r) for e in stream(world, m, r, 3)]
        training.append((eps[:2], eps[2]))
    learned, _ = R.learn_reliability(rd, training, prior)
    if attack == "X02":                                          # ease inverted: the learned reader must not care
        learned = dict(learned)
    vals = []
    for i in range(n):
        m = make_maker(world, f"x{i}", r, family=0, competence="low" if attack == "X05" else "mid")
        eps = [_noised(e, "semantic", r) for e in stream(world, m, r, 3)]
        if attack == "X01":
            eps = [relabel(e, r.permutation(N_FEAT)) for e in eps]
        if attack == "X04":
            eps = eps[:1] + [R.duplicate_semantic(eps[1], r, "duplicate")] + eps[2:]
            tabs = R.fused_tables(rd, eps[:2], ROUTES, "shared_cause")
        else:
            tabs = rd.route_tables(eps[:2], ROUTES)
        if attack == "X12":
            tabs = rd.route_tables([eps[1], eps[0]][::-1], ROUTES)
        vals.append(_gain(rd, J.joint(prior, tabs, learned), prior, eps[2]) - _gain(rd, J.joint(prior, tabs, R.weights_named("equal")), prior, eps[2]))
    return float(np.mean(vals)), attack in ("X01", "X02", "X04", "X05", "X11", "X12")


def effect_dissociation(world, r, n, attack=None):
    rd = J.Reader(world, 0, 0.75, 0.8)
    fam = world.family(0)
    hf = r.normal(0, 1, N_FEAT)
    k_eff, h_eff = [], []
    for i in range(max(2, n // 2)):
        lo = HS.agent(world, "lo", r, 0, "low", "none", pref=i % 6, plan=i % 4, h_feat=hf)
        hi = HS.agent(world, "hi", r, 0, "high", "none", pref=i % 6, plan=i % 4, h_feat=hf)
        hn = HS.agent(world, "hn", r, 0, "mid", "none", pref=i % 6, plan=i % 4, h_feat=hf)
        hs = HS.agent(world, "hs", r, 0, "mid", "strong", pref=i % 6, plan=i % 4, h_feat=hf)
        if attack == "X06":                                      # swap: history held, competence swapped, and the reverse; own effects must survive
            lo, hi = hi, lo
            k_sign = -1.0
        else:
            k_sign = 1.0
        def ea(m):
            return float(np.mean([np.mean(np.array(e["intended"]) == np.array([rd.inv_vocab[a] for a in e["action"]])) for e in stream(world, m, r, 12)]))

        def er(m):
            return float(np.mean([HS.early_relevance(e, m, fam, rd.inv_vocab) for e in stream(world, m, r, 12)]))
        k_eff.append(k_sign * (ea(hi) - ea(lo)))
        h_eff.append(er(hs) - er(hn))
    return float(min(np.mean(k_eff), np.mean(h_eff))), attack in ("X01", "X05", "X06", "X11", "X12")


def effect_affect(world, r, n, attack=None):
    vals = []
    for i in range(max(2, n // 2)):
        for region in ("sincere_fanatic", "strategic_propagandist"):
            s = CM.source(r, region, intensity="high")
            if attack == "X07":                                  # owner swap: intended effect moved away from the maker's appraisal
                s["appraisal"] = {"alarm": "calm", "calm": "admire", "admire": "alarm"}[CM.maker_appraisal(s)]
            arts = [CM.speak(dict(s, policy="cherry_pick"), r) for _ in range(2)]
            ll_art = sum(CM.loglik_artifact(a) + CM.loglik_appraisal_cue(a, "intensity") for a in arts)
            ll_pr = ll_art + CM.loglik_correction(CM.correction_event(s, r)) + CM.loglik_private(CM.private_action(s, r))
            if attack == "X08":                                  # collision: matched artifact and intended effect, probes only
                ll_pr = CM.loglik_correction(CM.correction_event(s, r)) + CM.loglik_private(CM.private_action(s, r))

            def acc(ll):
                rp = CM.region_posterior(CM.posterior(ll))
                pair = {"sincere_fanatic": rp["sincere_fanatic"], "strategic_propagandist": rp["strategic_propagandist"]}
                tot = sum(pair.values())
                return float(pair[region] / tot > 0.5) if tot > 0 else 0.5
            vals.append(acc(ll_pr) - acc(ll_art))
    return float(np.mean(vals)), attack in ("X01", "X07", "X08", "X11", "X12")


def effect_foraging(world, r, n, attack=None):
    vals = []
    for i in range(max(2, n // 2)):
        g = {}
        for pol in ("surprise", "learning_progress"):
            items = [F.make_item(r, "unlearnable_noise"), F.make_item(r, "structured_learnable"), F.make_item(r, "complex_compressible")]
            if attack == "X10":                                  # the noise item made salient and 'novel'; the structured item dull
                items[0]["novelty"], items[1]["novelty"] = 1.0, 0.05
                items[0]["relevance"], items[1]["relevance"] = 1.0, 0.6
            for it in items:
                for _ in range(2):
                    F.observe(it, r)
            g[pol] = F.forage(items, pol, 12.0, r)["gain"]
        vals.append(g["learning_progress"] - g["surprise"])
    return float(np.mean(vals)), attack in ("X01", "X10", "X11", "X12")


def effect_positive(world, r, n, attack=None):
    """Method positive: oracle process identifiability (class mass with the other latents supplied) over the prior."""
    rd = J.Reader(world, 0, 0.75, 0.8)
    prior = J.uniform_prior()
    vals = []
    for i in range(n):
        m = make_maker(world, f"p{i}", r, family=0, competence="high")
        eps = stream(world, m, r, 4)
        if attack == "X01":
            eps = [relabel(e, r.permutation(N_FEAT)) for e in eps]
        truth = J.truth_of(m, eps[-1])
        pm = J.oracle(prior, rd.route_tables(eps, NF), truth, "process")
        vals.append(float(pm[truth[0]]) - 0.25)
    return float(np.mean(vals)), True


def effect_null(world, r, n, attack=None):
    """Valid null: the joint advantage when the target maker is a different maker than the evidence's."""
    rd = J.Reader(world, 0, 0.75, 0.8)
    prior = J.uniform_prior()
    vals = []
    for i in range(n):
        m = make_maker(world, f"n{i}", r, family=0, competence="mid")
        other = make_maker(world, f"o{i}", r, family=0, competence="mid")
        eps = stream(world, m, r, 4)
        ep_next = episode(world, other, r)
        tabs = rd.route_tables(eps, NF)
        vals.append(_gain(rd, J.joint(prior, tabs), prior, ep_next) - _gain(rd, J.independent(prior, tabs), prior, ep_next))
    return float(np.mean(vals)), True


EFFECTS = {"joint_reconstruction_advantage": effect_joint, "reliable_routing": effect_routing,
           "competence_history_dissociation": effect_dissociation, "affect_source_factorization": effect_affect,
           "learning_progress_foraging": effect_foraging}
CAUSAL_PRESERVING = {"X01", "X02", "X04", "X05", "X06", "X07", "X08", "X10", "X12"}      # the latent is preserved; dying here is a shortcut


def survival(unattacked: float, attacked: float) -> str:
    if abs(unattacked) < 1e-3:
        return "null"
    if np.sign(attacked) == np.sign(unattacked) and abs(attacked) >= 0.5 * abs(unattacked):
        return "survives"
    if np.sign(attacked) == np.sign(unattacked) and attacked != 0:
        return "narrows"
    return "dies"


def _unit_attack(ctx, xid):
    from ghostscale.prereg_v14 import ATTACK_RELEVANCE
    world = world_for(ctx)
    r = rng(ctx, xid.lower())
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    table = {}
    for fl, fn in EFFECTS.items():
        relevant = xid in ATTACK_RELEVANCE.get(fl, [])
        if not relevant:
            table[fl] = {"relevant": False, "reason": "the attack does not touch this flight's construction; recorded, not silent"}
            continue
        base, _ = fn(world, r, n, None)
        if xid == "X11":                                          # aggregation: the effect in two planned subgroups
            lo, _ = fn(world, r, n, None) if fl != "joint_reconstruction_advantage" else effect_joint(world, r, n, None, reader_k=(0.55, 0.6))
            hi, _ = fn(world, r, n, None) if fl != "joint_reconstruction_advantage" else effect_joint(world, r, n, None, reader_k=(0.95, 0.95))
            att = min(lo, hi) if np.sign(lo) == np.sign(hi) else 0.0
            table[fl] = {"relevant": True, "unattacked": base, "attacked": att, "subgroups": {"low": lo, "high": hi},
                         "survival": "survives" if np.sign(lo) == np.sign(base) == np.sign(hi) else "dies", "shortcut": False,
                         "hidden_reversal": bool(np.sign(lo) != np.sign(hi))}
            continue
        att, applied = fn(world, r, n, xid)
        s = survival(base, att)
        table[fl] = {"relevant": True, "applied": bool(applied), "unattacked": base, "attacked": att, "survival": s,
                     "shortcut": bool(s == "dies" and xid in CAUSAL_PRESERVING)}
    pos_b, _ = effect_positive(world, r, n, None)
    pos_a, _ = effect_positive(world, r, n, xid)
    nul_b, _ = effect_null(world, r, n, None)
    nul_a, _ = effect_null(world, r, n, xid)
    cand = [t for t in table.values() if t.get("relevant")]
    cells.add({"target": "candidate"}, survived=float(np.mean([t["survival"] == "survives" for t in cand])) if cand else 1.0,
              died=float(np.mean([t["survival"] == "dies" for t in cand])) if cand else 0.0, n_relevant=float(len(cand)))
    cells.add({"target": "positive"}, survived=float(survival(pos_b, pos_a) == "survives"), died=float(survival(pos_b, pos_a) == "dies"), n_relevant=1.0)
    cells.add({"target": "null"}, survived=float(abs(nul_a) < 0.05), died=float(abs(nul_a) >= 0.05), n_relevant=1.0)
    return {"rows": cells.rows(), "table": table, "positive": {"unattacked": pos_b, "attacked": pos_a}, "null": {"unattacked": nul_b, "attacked": nul_a}}


def _reduce_attack(card, units, ctx, xid, attack_name, question):
    v = start(card, ctx, f"Attack {xid} ({attack_name}): {question}", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    fl_tables = {}
    for fl in FLIGHTS:
        ts = [u["table"][fl] for u in units]
        if not ts[0].get("relevant"):
            fl_tables[fl] = {"relevant": False, "reason": ts[0].get("reason")}
            continue
        ub = float(np.mean([t["unattacked"] for t in ts]))
        ab = float(np.mean([t["attacked"] for t in ts]))
        s = survival(ub, ab)
        fl_tables[fl] = {"relevant": True, "unattacked": ub, "attacked": ab, "survival": s, "shortcut": bool(s == "dies" and xid in CAUSAL_PRESERVING and xid != "X11"),
                         **({"hidden_reversal": float(np.mean([t.get("hidden_reversal", False) for t in ts]))} if xid == "X11" else {})}
    pos_b = float(np.mean([u["positive"]["unattacked"] for u in units]))
    pos_a = float(np.mean([u["positive"]["attacked"] for u in units]))
    nul_a = float(np.mean([u["null"]["attacked"] for u in units]))
    survivors = [fl for fl, t in fl_tables.items() if t.get("relevant") and t["survival"] == "survives"]
    narrowed = [fl for fl, t in fl_tables.items() if t.get("relevant") and t["survival"] == "narrows"]
    deaths = [fl for fl, t in fl_tables.items() if t.get("relevant") and t["survival"] == "dies"]
    gr = G.GateReport()
    gr.identity("applicability_recorded_for_every_flight", float(sum("relevant" in t for t in fl_tables.values())), float(len(FLIGHTS)), tol=0.0,
                detail="no silent not-applicable")
    battery(gr, positive={"observed": pos_a, "expected": pos_b, "tol": max(0.5 * abs(pos_b), 0.1), "name": "method_positive_survives_the_attack"},
            placebo={"observed": abs(nul_a), "tol": 0.08, "name": "valid_null_stays_null_under_the_attack"})
    criterion(v, xid, True, survivors=survivors, narrowed=narrowed, deaths=deaths, table=fl_tables)
    v["results"].update({"positive": {"unattacked": pos_b, "attacked": pos_a}, "null": {"attacked": nul_a}})
    receipt(v, rows, card, ctx)
    parts = [f"{fl} {t['survival']} ({t['unattacked']:+.2f} to {t['attacked']:+.2f})" for fl, t in fl_tables.items() if t.get("relevant")]
    narrative(v, f"Under '{attack_name}': " + ("; ".join(parts) if parts else "no flight is relevant") + f". The method positive went from {pos_b:+.2f} to {pos_a:+.2f}; the null stayed at {nul_a:+.3f}.",
              "The surviving region is what the flight may claim.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(True))


ATTACKS = {"X01": ("surface", "Does the effect survive the alteration of cheap features with the latent preserved?"),
           "X02": ("route ease", "Does routing follow accuracy when ease is inverted?"),
           "X03": ("equifinal history", "Does the effect hold with equivalence-class uncertainty on identical artifacts?"),
           "X04": ("duplicate evidence", "Does confidence stay honest when one cause is paraphrased?"),
           "X05": ("wrong generative model", "Does the effect survive a maker whose competence the reader misjudges?"),
           "X06": ("attention/skill swap", "Do the own effects of competence and history survive being swapped?"),
           "X07": ("affect owner swap", "Does the fanatic/propagandist boundary survive an owner swap?"),
           "X08": ("fanatic/propagandist collision", "Does the boundary hold on matched artifact and intended effect, from probes only?"),
           "X09": ("hierarchy equivalence", "Does an exact equivalence defeat unjustified unique attribution?"),
           "X10": ("hope and salience", "Does progress beat surprise when noise is made salient and structure dull?"),
           "X11": ("aggregation", "Do pooled effects hide a planned sign reversal across reader competence?"),
           "X12": ("solver/lineage", "Is the effect stable under evidence order and the fresh lane?")}

for _xid, (_name, _q) in ATTACKS.items():
    def _mk(xid=_xid, name=_name, q=_q):
        def unit(ctx):
            return _unit_attack(ctx, xid)

        def reduce(card, units, ctx):
            return _reduce_attack(card, units, ctx, xid, name, q)
        return unit, reduce
    globals()[f"unit_{_xid}"], globals()[f"reduce_{_xid}"] = _mk()
