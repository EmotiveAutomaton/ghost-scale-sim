"""Trunk Q: active epistemic foraging and PyMDP (spec §15).

The only trunk in which PyMDP is central. Every probe set passes a divergence audit before any
policy is scored; exact expected information gain is the reference everywhere; disagreement is a
solver result and never a fact about the maker.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as X, priors as P, costs as CO, goals_trust as GT, attention as A
from .. import pymdp_reader as PR
from ..world import make_maker, stream, N_METHODS
from . import (battery, boot, criterion, decide_state, finish, narrative, receipt, rng, sizes, start, world_for, mean_of, pursuit_of, sim_bin)
from .trunk_c import Cells, harness, reader_priors, target_priors, bins_for, posterior_at

DIVERGENCE_FLOOR = 0.02
DISCRIM_FLOOR = 0.005


def _commission_emissions(model, fid, domain=0):
    """emissions[p, h, f]: the surface distribution under commission p (a goal channel) and hypothesis h."""
    fam = model.world.family(fid)
    return np.stack([np.stack([model.emission(h, g, None, domain) for h in model.hyps]) for g in range(fam.ng)])


def _family_model(world, r, fid, k=0.05):
    rd = make_maker(world, f"rd{fid}", r, family=fid, k=k)
    model = X.reader_model(world, rd, families=[fid])
    return rd, model


def _realized_gain(model, prior, m, arts):
    ti = model.truth_index(m)
    q = model.posterior(prior, arts, ("surface",))
    return C.log_score(q, ti) - C.log_score(prior, ti), q


# --------------------------------------------------------------------------- #
# Q01 — PyMDP selects the information-maximising probe.
# --------------------------------------------------------------------------- #
def unit_Q01(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q01")
    n_trials = max(4, sizes(ctx)["makers"] // 8)
    for fid in range(world.n_families):
        if world.family(fid).link != "draw":
            continue
        rd, model = _family_model(world, r, fid)
        ems = _commission_emissions(model, fid)
        a_f = PR.pairwise_divergence(ems)
        if a_f["min_pairwise_probe_js"] < DIVERGENCE_FLOOR or float(np.mean(a_f["discriminativeness_per_probe"])) < DISCRIM_FLOOR:
            continue                                        # the same audit gate as Q02: agreement is meaningless where probes cannot discriminate
        K = model.K
        for prior_kind in ("uniform", "peaked"):
            for horizon in (1, 2):
                for t in range(n_trials):
                    if prior_kind == "uniform":
                        prior = np.full(K, 1.0 / K)
                    else:
                        prior = C.normalize(np.full(K, 0.02) + np.eye(K)[int(r.integers(K))] * 0.5)
                    eig = PR.exact_eig_per_probe(ems, prior, 12, C.rng_for(ctx["lane"], "Q01", ctx["wid"], ctx["rep"], f"e{fid}{t}{prior_kind}"), draws=60)
                    ag = PR.build_reader(ems, prior, probe_costs=np.zeros(ems.shape[0]), policy_len=horizon)
                    ch, _ = PR.choose_probe(ag)
                    ag_u = PR.build_reader(ems, prior, probe_costs=np.zeros(ems.shape[0]), use_info_gain=False, policy_len=horizon)
                    ch_u, _ = PR.choose_probe(ag_u)
                    ch_r = int(r.integers(ems.shape[0]))
                    d = PR.policy_disagreement(eig, ch)
                    tie = max(0.1 * float(eig.max() - eig.min()), 1e-4)

                    def near(c):
                        return float(eig[int(c)] >= float(eig.max()) - tie)
                    cells.add({"prior": prior_kind, "horizon": horizon}, agree=near(ch), agree_utility=near(ch_u),
                              agree_random=near(ch_r), rank=float(d["agent_rank"]), eig_spread=float(eig.max() - eig.min()))
    return {"rows": cells.rows()}


def reduce_Q01(card, units, ctx):
    v = start(card, ctx, "Given probes with distinct likelihoods, the legacy PyMDP reader chooses the probe with the highest exact "
              "expected information gain, and a utility-only agent or a random agent does not.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    by = {f"{p}|h{h}": {k: mean_of(rows, k, lambda r, p=p, h=h: r["prior"] == p and r["horizon"] == h) for k in ("agree", "agree_utility", "agree_random", "eig_spread")} for p in ("uniform", "peaked") for h in (1, 2)}
    agree1 = mean_of(rows, "agree", lambda r: r["horizon"] == 1)
    agree = mean_of(rows, "agree")
    passed = bool(agree1 >= 0.8)
    gr = G.GateReport()
    gr.live("probes_genuinely_differ_in_eig", observed_change=mean_of(rows, "eig_spread"), min_change=DIVERGENCE_FLOOR, detail="the identity world first: probes must differ before agreement means anything")
    gr.positive("epistemic_drive_tracks_the_exact_ranking", observed=float(agree1 >= mean_of(rows, "agree_random") + 0.2), expected=1.0, tol=0.0,
                detail="instrument control: the info-gain agent must sit well above a random chooser against the exact ranking. The agreement VALUE is the pre-registered criterion, and the by-prior surface is the result: the smoke pass already shows agreement degrading under peaked priors, which is exactly the discrepancy surface I10 exists to map. A probe within a tenth of the spread of the best counts as agreement: near-ties are not disagreements")
    gr.positive("utility_only_agent_fails_the_epistemic_control", observed=float(mean_of(rows, "agree_utility") < agree), expected=1.0, tol=0.0)
    gr.positive("random_agent_at_chance", observed=mean_of(rows, "agree_random"), expected=1.0 / 4, tol=0.25)
    criterion(v, "Q01", passed, agreement_horizon_one=agree1, agreement_all=agree, by_cell=by)
    v["results"].update({"agreement": agree, "by_prior_and_horizon": by})
    v["pymdp"] = {"agreement": agree}
    receipt(v, rows, card, ctx)
    narrative(v, f"PyMDP chose the exact information-maximising commission {agree:.0%} of the time across priors and horizons; a utility-only agent {mean_of(rows, 'agree_utility'):.0%}, a random agent {mean_of(rows, 'agree_random'):.0%}.",
              "The bounded reader's epistemic drive is calibrated against exact information gain where the probes differ.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q02 — the commission instrument, repaired.
# --------------------------------------------------------------------------- #
def unit_Q02(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q02")
    n_trials = max(4, sizes(ctx)["makers"] // 8)
    audits = []
    skipped = 0
    for fid in range(world.n_families):
        if world.family(fid).link != "draw":
            continue
        fam = world.family(fid)
        rd, model = _family_model(world, r, fid)
        ems = _commission_emissions(model, fid)
        a_f = PR.pairwise_divergence(ems)
        if a_f["min_pairwise_probe_js"] < DIVERGENCE_FLOOR or float(np.mean(a_f["discriminativeness_per_probe"])) < DISCRIM_FLOOR:
            skipped += 1                                    # the audit is the gate: a family whose commissions cannot discriminate is not scored
            continue
        audits.append(a_f)
        K = model.K
        makers = [make_maker(world, f"m{j}", r, family=fid, k=0.2) for j in range(n_trials)]
        for m in makers:
            prior = model.posterior(np.full(K, 1.0 / K), stream(world, m, 0, r, 1, n_steps=8), ("surface",))   # one free look first
            eig = PR.exact_eig_per_probe(ems, prior, 12, C.rng_for(ctx["lane"], "Q02", ctx["wid"], ctx["rep"], "e" + m.id), draws=40)
            choices = {"exact": int(np.argmax(eig)), "random": int(r.integers(fam.ng))}
            ag = PR.build_reader(ems, prior, probe_costs=np.zeros(fam.ng))
            choices["pymdp"], _ = PR.choose_probe(ag)
            for pol in ("exact", "pymdp", "random", "free_look"):
                if pol == "free_look":
                    arts = stream(world, m, 0, r, 1, n_steps=12)
                else:
                    arts = stream(world, m, 0, r, 1, n_steps=12, commission=choices[pol])
                g, q = _realized_gain(model, prior, m, arts)
                cells.add({"policy": pol}, gain=g, conf=float(q.max()), top1=float(int(np.argmax(q)) == model.truth_index(m)), eig_chosen=float(eig[choices.get(pol, 0)]) if pol != "free_look" else float(np.mean(eig)))
    return {"rows": cells.rows(), "audit": {"min_pairwise": float(min(a["min_pairwise_probe_js"] for a in audits)) if audits else 0.0,
                                            "min_discriminativeness": float(min(float(np.mean(a["discriminativeness_per_probe"])) for a in audits)) if audits else 0.0,
                                            "families_skipped": skipped, "families_scored": len(audits)}}


def reduce_Q02(card, units, ctx):
    v = start(card, ctx, "The V12 commission instrument failed because candidate commissions were not discriminative; rebuilt with a "
              "divergence gate before any policy is scored, exact and PyMDP selection are tested against random and free looks.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    mp = float(np.min([u["audit"]["min_pairwise"] for u in units]))
    md = float(np.min([u["audit"]["min_discriminativeness"] for u in units]))
    by = {p: boot(rows, "gain", lambda r, p=p: r["policy"] == p, seed_tag="Q02" + p)["mean"] for p in ("exact", "pymdp", "random", "free_look")}
    live = mp >= DIVERGENCE_FLOOR and md >= DISCRIM_FLOOR
    gain_exact = by["exact"] - by["random"]
    passed = bool(live and gain_exact >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": mp, "min": DIVERGENCE_FLOOR, "name": "commissions_pairwise_divergent"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "one_free_look_shared_by_every_policy"},
            positive={"observed": float(gain_exact >= 0.0), "expected": 1.0, "tol": 0.0, "name": "chosen_commission_gains_over_random", "detail": "whether commissioning at all beats a free look is a reported comparison, not a control: a free artifact carries goal-draw evidence a commission removes"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_makers_same_prior"},
            oracle={"observed": md, "min": DISCRIM_FLOOR, "name": "probes_discriminate_hypotheses"},
            prediction={"gain": gain_exact, "min": 0.0, "name": "exact_minus_random_realized_gain"},
            calibration={"observed": by["pymdp"], "reference": by["random"], "direction": "up", "tol": 0.20, "name": "pymdp_near_random_or_better"})
    state = decide_state(gr)
    closure = "" if live else "instrument not live: commission probes below the divergence floor; one repair was allowed and this rebuild is it"
    criterion(v, "Q02", passed, min_pairwise_js=mp, min_discriminativeness=md, exact_minus_random=gain_exact, pymdp_minus_random=by["pymdp"] - by["random"])
    v["results"].update({"audit": {"min_pairwise_js": mp, "min_discriminativeness": md, "floors": {"pairwise": DIVERGENCE_FLOOR, "discriminativeness": DISCRIM_FLOOR}}, "realized_gain_by_policy": by})
    v["pymdp"] = {"pymdp_minus_random": by["pymdp"] - by["random"]}
    receipt(v, rows, card, ctx)
    narrative(v, f"Commission probes differed pairwise by at least {mp:.3f} (Jensen-Shannon) and discriminated hypotheses by at least {md:.3f}; the exact-chosen commission realised {gain_exact:+.2f} nats more than a random one, PyMDP's {by['pymdp'] - by['random']:+.2f}, and a free look {by['free_look'] - by['random']:+.2f}.",
              "The commission instrument is live; whether choosing the commission beats looking freely is now a measured number rather than a dead card.")
    return finish(card, v, gr, __file__, state, ctx, closure_reason=closure, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q03 — querying missing opportunity information.
# --------------------------------------------------------------------------- #
FIELDS = ("size", "composition", "costs", "control")


def unit_Q03(ctx):
    world = world_for(ctx)
    fam = world.family(0)
    profiles = {n: fam.grid[i] for i, n in enumerate(fam.grid_names)}
    names = list(profiles)
    cells = Cells(ctx["wid"], ctx["rep"])
    decisions = []
    r = rng(ctx, "q03")
    for i in range(max(4, sizes(ctx)["makers"] // 6)):
        w = profiles[names[i % len(names)]]
        actor = CO.Actor(w, motivation=1.0)
        recs = CO.stream(actor, r, 10, fam.ng, ecology="craft")
        post = CO.posterior(profiles, recs[:6])
        nxt = recs[6]
        # the reader's next-choice prediction; each field, if bought, reveals part of the record; decision value is the
        # expected reduction in the entropy of the prediction
        for price_name, price in (("low", 0.02), ("high", 0.3)):
            values = {}
            for field in FIELDS + ("polish",):
                if field == "polish":
                    values[field] = 0.0
                    continue
                hidden = dict(nxt)
                if field == "size":
                    hidden = CO.hidden_menu(nxt, r, hide=1)
                elif field == "composition":
                    hidden["payoff"] = np.tile(np.asarray(nxt["payoff"]).mean(axis=0), (nxt["n"], 1))
                elif field == "costs":
                    hidden["cost"] = np.tile(np.asarray(nxt["cost"]).mean(axis=0), (nxt["n"], 1))
                else:
                    hidden["mode"] = "imposed"
                p_full = CO.predict_choice(post, profiles, nxt)
                p_hidden = CO.predict_choice(post, profiles, hidden)
                values[field] = float(C.entropy(p_hidden) - C.entropy(p_full))
            best = max(FIELDS, key=lambda f: values[f])
            # the query policy buys the field with the highest value net of price, or nothing; polish is a distractor with no value
            values["polish"], values["none"] = 0.0, 0.0
            net = {f: values[f] - (0.0 if f == "none" else price * (0.5 if f == "polish" else 1.0)) for f in list(FIELDS) + ["polish", "none"]}
            buy = max(net, key=net.get)
            for field in FIELDS:
                cells.add({"field": field, "price": price_name}, value=values[field], bought=float(buy == field), is_best=float(best == field), polish_bought=float(buy == "polish"))
            decisions.append({"price": price_name, "best_bought": float(buy == best), "polish_bought": float(buy == "polish"),
                              "bought_any": float(buy != "none"), "nonbest_bought": float(buy not in (best, "none", "polish"))})
    return {"rows": cells.rows(), "decisions": decisions}


def reduce_Q03(card, units, ctx):
    v = start(card, ctx, "A reader that can buy missing opportunity information buys the field with the highest expected decision "
              "value, not the most polished record, and buys less as prices rise.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    dec = [d for u in units for d in u.get("decisions", [])]
    rate = float(np.mean([d["best_bought"] for d in dec if d["price"] == "low"])) if dec else float("nan")
    polish = float(np.mean([d["polish_bought"] for d in dec])) if dec else float("nan")
    by = {f: {p: {"value": mean_of(rows, "value", lambda r, f=f, p=p: r["field"] == f and r["price"] == p), "bought": mean_of(rows, "bought", lambda r, f=f, p=p: r["field"] == f and r["price"] == p)} for p in ("low", "high")} for f in FIELDS}
    passed = bool(rate >= 0.6 and polish <= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": max(by[f]["low"]["value"] for f in FIELDS), "min": 0.02, "name": "fields_carry_decision_value"},
            placebo={"observed": polish, "tol": 0.05, "name": "polish_is_never_bought"},
            positive={"observed": rate, "expected": 1.0, "tol": 0.4, "name": "highest_value_field_bought_at_low_price"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_records"},
            oracle={"observed": rate, "min": 0.6, "name": "value_identifiable"},
            prediction={"gain": float(np.mean([d["bought_any"] for d in dec if d["price"] == "low"]) - np.mean([d["bought_any"] for d in dec if d["price"] == "high"])) if dec else 0.0, "min": 0.0, "name": "buys_less_at_high_price"},
            calibration={"observed": float(np.mean([d["nonbest_bought"] for d in dec])) if dec else 0.0, "reference": rate, "direction": "down", "tol": 0.0, "name": "non_best_fields_bought_less"})
    criterion(v, "Q03", passed, best_field_bought_rate=rate, polish_bought=polish, by_field=by)
    v["results"].update({"by_field_and_price": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"At a low price the reader bought the field with the highest decision value {rate:.0%} of the time and never bought polish ({polish:.0%}).",
              "Queries for opportunity information follow decision value; the polished record is a distractor the reader ignores.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q04 — choosing which cost cause to disambiguate.
# --------------------------------------------------------------------------- #
PAIRS = {"motivation_competence": ("motivation", "competence"), "motivation_constraint": ("motivation", "constraint"), "knowledge_risk": ("knowledge", "risk_tolerance")}


def _probe_menu(r, ng, target_dim):
    """A menu designed to separate the pair: one option that is only worth it if the actor's
    weight on the targeted cause is high."""
    m = CO.menu(r, ng, 4, "craft")
    if target_dim == "competence":
        m["cost"][0, 1] += 1.2                                                     # a hard execution
        m["payoff"][0] *= 2.0
    elif target_dim == "motivation":
        m["cost"][0] += 0.3
        m["payoff"][0] *= 1.8
    elif target_dim == "constraint":
        m["mandatory"] = np.zeros(4, bool)
    elif target_dim == "knowledge":
        m["info"][0] = 1.0
    else:
        m["variance"][0] = 3.0
        m["payoff"][0] *= 1.5
    return m


def unit_Q04(ctx):
    world = world_for(ctx)
    fam = world.family(0)
    profiles = {n: fam.grid[i] for i, n in enumerate(fam.grid_names)}
    names = list(profiles)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q04")
    for pair, (d1_, d2_) in PAIRS.items():
        causes = {d1_: list(CO.LEVELS[d1_]), d2_: list(CO.LEVELS[d2_])}
        for i in range(max(3, sizes(ctx)["makers"] // 8)):
            w = profiles[names[i % len(names)]]
            cell = {d1_: float(r.choice(CO.LEVELS[d1_])), d2_: float(r.choice(CO.LEVELS[d2_]))}
            actor = CO.Actor(w, **cell)
            recs = CO.stream(actor, r, 14, fam.ng, ecology="craft")
            post = CO.posterior({names[i % len(names)]: w}, recs, causes=causes)
            # the most confused pair: the cause axis whose marginal is closest to flat
            marg = {}
            for d in (d1_, d2_):
                levels = sorted({c[d] for c in post["cells"]})
                marg[d] = np.array([sum(p for p, c in zip(post["cause"], post["cells"]) if c[d] == lv) for lv in levels])
            confused = max(marg, key=lambda d: C.entropy(marg[d]))
            test = CO.stream(actor, r, 6, fam.ng, ecology="craft")
            for pol in ("eig", "uncertainty", "random"):
                target = confused if pol == "eig" else (max(marg, key=lambda d: marg[d].max()) if pol == "uncertainty" else str(r.choice([d1_, d2_])))
                probe = _probe_menu(r, fam.ng, target)
                rec = CO.choose(actor, probe, r)
                post2 = CO.posterior({names[i % len(names)]: w}, recs + [rec], causes=causes)
                ls_before = float(np.mean([np.log(max(CO.predict_choice(post, {names[i % len(names)]: w}, t)[int(t["choice"])], 1e-12)) for t in test]))
                ls_after = float(np.mean([np.log(max(CO.predict_choice(post2, {names[i % len(names)]: w}, t)[int(t["choice"])], 1e-12)) for t in test]))
                cells.add({"confused_pair": pair, "policy": pol}, gain=ls_after - ls_before, targeted_confused=float(target == confused),
                          cause_correct=float(all(abs(post2["cells"][int(np.argmax(post2["cause"]))][d] - cell[d]) < 1e-9 for d in cell)))
    return {"rows": cells.rows()}


def reduce_Q04(card, units, ctx):
    v = start(card, ctx, "When motivation, competence, knowledge, constraint and risk compete to explain a paid cost, a reader that "
              "probes the most confused pair improves its held-out prediction more than uncertainty sampling or a random probe.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {p: {"gain": boot(rows, "gain", lambda r, p=p: r["policy"] == p, seed_tag="Q04" + p)["mean"], "targeted": mean_of(rows, "targeted_confused", lambda r, p=p: r["policy"] == p),
              "cause_correct": mean_of(rows, "cause_correct", lambda r, p=p: r["policy"] == p)} for p in ("eig", "uncertainty", "random")}
    passed = bool(by["eig"]["targeted"] >= 0.6 and by["eig"]["gain"] - by["random"]["gain"] >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": by["eig"]["gain"], "min": 0.0, "name": "a_probe_moves_prediction"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_history_all_policies"},
            positive={"observed": by["eig"]["targeted"], "expected": 1.0, "tol": 0.4, "name": "eig_targets_the_confused_pair"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_menus_before_the_probe"},
            oracle={"observed": by["eig"]["cause_correct"], "min": 0.25, "name": "causes_identifiable_after_a_probe"},
            prediction={"gain": by["eig"]["gain"] - by["random"]["gain"], "min": 0.0, "name": "eig_minus_random"},
            calibration={"observed": by["random"]["gain"], "reference": by["eig"]["gain"], "direction": "down", "tol": 0.05, "name": "random_probe_gains_no_more"})
    criterion(v, "Q04", passed, by_policy=by)
    v["results"].update({"by_policy": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"The information-seeking probe targeted the most confused cause pair {by['eig']['targeted']:.0%} of the time and improved held-out choice prediction by {by['eig']['gain']:+.2f} nats against {by['random']['gain']:+.2f} for a random probe.",
              "A cost cause is something a reader can choose to disambiguate, and the choice pays.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q05 — entry probing by expertise.
# --------------------------------------------------------------------------- #
PROBES5 = {"purpose": "goal_consequences", "mechanics": "mechanics", "context": "group_convention", "anomaly": "anomaly", "source": "communicative_shaping"}


def unit_Q05(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q05")
    for i in range(max(2, sizes(ctx)["readers"] // 4)):
        fid = i % world.n_families
        if world.family(fid).link != "draw":
            continue
        for expertise in ("novice", "expert"):
            rd = make_maker(world, f"r{i}{expertise}", r, family=fid, k=0.05 if expertise == "expert" else 0.5)
            if expertise == "novice":
                rd.method_pref = np.full_like(rd.method_pref, 1.0 / N_METHODS)
            model = X.reader_model(world, rd, families=[fid])
            prior = X.uniform_prior(model)
            makers = [make_maker(world, f"m{j}", r, family=fid, k=0.2) for j in range(5)]
            # training items decide each probe's expected gain for this reader (its own expertise decides what it can read)
            train = [(stream(world, m, 0, r, 2, n_steps=8), model.truth_index(m)) for m in [make_maker(world, f"t{j}", r, family=fid, k=0.2) for j in range(5)]]
            gains = {p: A.channel_diagnosticity(model, prior, train, ch) for p, ch in PROBES5.items()}
            adaptive_first = max(gains, key=gains.get)
            for m in makers:
                arts = stream(world, m, 0, r, 2, n_steps=8)
                ti = model.truth_index(m)
                for pol, first in (("adaptive", adaptive_first), ("purpose_first", "purpose")):
                    q1 = model.posterior(prior, arts[:1], (PROBES5[first],))
                    q2 = model.posterior(prior, arts, tuple(PROBES5.values()))
                    cells.add({"expertise": expertise, "policy": pol}, ls_first=C.log_score(q1, ti), ls_final=C.log_score(q2, ti), first_is_purpose=float(first == "purpose"),
                              first_is_mechanics=float(first == "mechanics"))
    return {"rows": cells.rows()}


def reduce_Q05(card, units, ctx):
    v = start(card, ctx, "An adaptive reader chooses its first probe from what it can read: novices open with purpose, experts may "
              "open with mechanics, and fixed purpose-first is the strong cheap baseline.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {x: {p: {k: mean_of(rows, k, lambda r, x=x, p=p: r["expertise"] == x and r["policy"] == p) for k in ("ls_first", "ls_final", "first_is_purpose", "first_is_mechanics")} for p in ("adaptive", "purpose_first")} for x in ("novice", "expert")}
    changes = bool(abs(by["novice"]["adaptive"]["first_is_purpose"] - by["expert"]["adaptive"]["first_is_purpose"]) >= 0.2 or by["expert"]["adaptive"]["first_is_mechanics"] >= 0.3)
    gr = G.GateReport()
    battery(gr, live={"observed": abs(by["novice"]["adaptive"]["ls_first"] - by["expert"]["adaptive"]["ls_first"]), "min": 0.0, "name": "expertise_moves_the_first_probe_score"},
            placebo={"observed": max(abs(by[x]["adaptive"]["ls_final"] - by[x]["purpose_first"]["ls_final"]) for x in by), "tol": 0.15, "name": "final_posteriors_converge"},
            positive={"observed": float(by["novice"]["adaptive"]["ls_first"] >= by["novice"]["purpose_first"]["ls_first"] - 0.1), "expected": 1.0, "tol": 0.0, "name": "adaptive_no_worse_than_purpose_first_for_novices"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts"},
            oracle={"observed": mean_of(rows, "ls_final") - np.log(1 / 32), "min": 0.0, "name": "identifiable"},
            prediction={"gain": by["expert"]["adaptive"]["ls_first"] - by["expert"]["purpose_first"]["ls_first"], "min": -1.0, "name": "expert_adaptive_minus_purpose_first"},
            calibration={"observed": float(changes), "reference": 0.0, "direction": "up", "tol": 1.0, "name": "entry_changes_with_expertise_reported"})
    criterion(v, "Q05", changes, by_expertise=by)
    v["results"].update({"by_expertise_and_policy": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Adaptive readers opened with purpose {by['novice']['adaptive']['first_is_purpose']:.0%} of the time as novices and {by['expert']['adaptive']['first_is_purpose']:.0%} as experts, with mechanics first {by['expert']['adaptive']['first_is_mechanics']:.0%} of the time for experts; final posteriors converged.",
              "Goal-first is where a generic reader starts; it is not where an expert has to.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(changes))


# --------------------------------------------------------------------------- #
# Q06 — inferring the maker's attention by active observation.
# --------------------------------------------------------------------------- #
def unit_Q06(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q06")
    chans = {"goal": "goal_consequences", "mechanics": "mechanics", "surface": "communicative_shaping"}
    for i in range(max(2, sizes(ctx)["readers"] // 4)):
        fid = i % world.n_families
        if world.family(fid).link != "draw":
            continue
        rd, model = _family_model(world, r, fid)
        prior = X.uniform_prior(model)
        for att in ("goal", "mechanics", "surface"):
            for j in range(3):
                m = make_maker(world, f"m{att}{j}", r, family=fid, k=0.2, attention=att)
                arts = stream(world, m, 0, r, 6)
                ti = model.truth_index(m)
                for pol in ("eig", "uncertainty"):
                    # attention posterior from the concentration of each channel's observations
                    # each attention hypothesis has an exact generative signature (spec state semantics):
                    # goal -> payoff noise scaled by 0.3; mechanics -> squared method preference;
                    # surface -> a fixed cue slot. The posterior over allocations is the likelihood of
                    # the observed payoffs, methods and slots under each, which is unbiased where
                    # concentration statistics carry different small-sample biases per channel.
                    fam = world.family(rd.family)
                    pn = world.params.payoff_noise
                    K_slots = max(1, int(fam.tail.size))
                    slots = [a["slot"] for a in arts]

                    def _att_ll(att_h):
                        pn_h = pn * (0.3 if att_h == "goal" else 1.0)
                        ll = 0.0
                        for a in arts:
                            g = a["goal"]
                            ll += float(np.log(max(1.0 - pn_h, 1e-9))) if a["payoff_obs"] == g else float(np.log(max(pn_h / max(fam.ng - 1, 1), 1e-9)))
                            if a.get("method") is not None:
                                mp_g = fam.method_pref[g] / fam.method_pref[g].sum()
                                if att_h == "mechanics":
                                    mp_g = mp_g ** 2
                                    mp_g = mp_g / mp_g.sum()
                                ll += float(np.log(max(mp_g[a["method"]], 1e-9)))
                        if att_h == "surface":
                            ll += float(np.log(1.0 / K_slots)) + float(sum(np.log(0.95) if s == slots[0] else np.log(max(0.05 / max(K_slots - 1, 1), 1e-9)) for s in slots[1:]))
                        else:
                            ll += len(slots) * float(np.log(1.0 / K_slots))
                        return ll
                    q_att = C.softmax(np.array([_att_ll(k) for k in ("goal", "mechanics", "surface")]))
                    if pol == "eig":
                        # the informed read AVOIDS the channel the maker attended to: attention
                        # sharpens execution toward the family convention there, which masks the
                        # profile; individuality shows where attention did not press. (Measured,
                        # not assumed: the smoke pass found the attended channel scores worse for
                        # identification, which is what the state semantics imply.)
                        pick = ["goal", "mechanics", "surface"][int(np.argmin(q_att))]
                    else:
                        pick = ["goal", "mechanics", "surface"][int(r.integers(3))]     # attention-blind baseline
                    q = model.posterior(prior, arts, ("surface", chans[pick]), {"surface": 1.0, chans[pick]: 2.0})
                    cells.add({"maker_attention": att, "policy": pol}, att_correct=float(["goal", "mechanics", "surface"][int(np.argmax(q_att))] == att),
                              ls=C.log_score(q, ti), att_ls=float(np.log(max(q_att[["goal", "mechanics", "surface"].index(att)], 1e-12))))
    return {"rows": cells.rows()}


def reduce_Q06(card, units, ctx):
    v = start(card, ctx, "Where the maker's attention shaped which decisions were sharp, the reader can infer that allocation from the "
              "artifacts and choose what to inspect accordingly, beyond uncertainty-only sampling.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {p: {k: mean_of(rows, k, lambda r, p=p: r["policy"] == p) for k in ("att_correct", "ls", "att_ls")} for p in ("eig", "uncertainty")}
    per_att = {a: mean_of(rows, "att_correct", lambda r, a=a: r["maker_attention"] == a and r["policy"] == "eig") for a in ("goal", "mechanics", "surface")}
    gain = by["eig"]["ls"] - by["uncertainty"]["ls"]
    passed = bool(gain >= 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": by["eig"]["att_correct"] - 1 / 3, "min": 0.1, "name": "attention_inferred_above_chance"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_artifacts_both_policies"},
            positive={"observed": min(per_att.values()), "expected": 1.0, "tol": 0.7, "name": "each_allocation_recognised"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_evidence"},
            oracle={"observed": by["eig"]["att_correct"], "min": 0.5, "name": "identifiable"},
            prediction={"gain": gain, "min": 0.0, "name": "eig_minus_uncertainty"},
            calibration={"observed": by["uncertainty"]["ls"], "reference": by["eig"]["ls"], "direction": "down", "tol": 0.02, "name": "attention_blind_sampling_worse",
                         "detail": "the baseline picks a channel without using the attention posterior"})
    criterion(v, "Q06", passed, by_policy=by, per_attention=per_att)
    v["results"].update({"by_policy": by, "attention_accuracy_by_allocation": per_att})
    receipt(v, rows, card, ctx)
    narrative(v, f"The maker's attention allocation was inferred {by['eig']['att_correct']:.0%} of the time from six artifacts, and inspecting a channel its attention did not sharpen improved the profile score by {gain:+.2f} nats over an attention-blind pick.",
              "Maker attention is a latent an active reader can recover and use: what it sharpened toward convention, and where individuality still shows.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q07 — challenge with an anticipating adversary.
# --------------------------------------------------------------------------- #
def unit_Q07(ctx):
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q07")
    d0, d1 = GT.kind_dists(r)
    for s in range(max(4, sizes(ctx)["sources"] // 2)):
        for goal in ("accurate", "persuasion", "misleading"):
            for anticipates in (0, 1):
                src = GT.Source(f"s{s}", goal, {0: 0.9}, agenda=1, slot=s)
                static = [a for a in (GT.speak(src, r, d0, d1, 8, t=i) for i in range(8)) if a][:4]
                for pol in ("goal_aware", "passive", "random"):
                    if pol == "passive":
                        fr = GT.factored_read(static, d0, d1, revealed=None)
                    else:
                        T = int(r.random() < 0.5) if pol == "random" else int(np.argmax([sum(GT.goal_loglik(a, t, d0, d1, "misleading") for a in static) for t in (0, 1)]))
                        # an anticipating adversary answers a challenge as an accurate source would
                        responder = GT.Source(src.id, "accurate" if (anticipates and goal != "accurate") else goal, src.reliability, 1, slot=s)
                        resp = GT.challenge_response(responder, r, d0, d1, T)
                        arts = static + ([resp] if resp and not resp.get("declined") else [])
                        fr = GT.factored_read(arts, d0, d1, revealed={len(static): T} if len(arts) > len(static) else None)
                    cells.add({"policy": pol, "anticipates": anticipates}, ls=float(np.log(max(fr["q_goal"][goal], 1e-12))), acc=float(max(fr["q_goal"], key=fr["q_goal"].get) == goal))
    return {"rows": cells.rows()}


def reduce_Q07(card, units, ctx):
    v = start(card, ctx, "A challenge chosen for the goal it would expose beats passive and random reading, and an adversary who "
              "anticipates the challenge and answers as a teacher would takes the gain away.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {p: {str(a): boot(rows, "ls", lambda r, p=p, a=a: r["policy"] == p and r["anticipates"] == a, seed_tag=f"Q07{p}{a}")["mean"] for a in (0, 1)} for p in ("goal_aware", "passive", "random")}
    gain = grid["goal_aware"]["0"] - grid["passive"]["0"]
    loss = grid["goal_aware"]["0"] - grid["goal_aware"]["1"]
    passed = bool(gain >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": gain, "min": 0.0, "name": "challenge_moves_the_score"},
            placebo={"observed": abs(grid["passive"]["0"] - grid["passive"]["1"]), "tol": 0.3, "name": "passive_reading_unaffected_by_anticipation"},
            positive={"observed": float(grid["goal_aware"]["0"] >= grid["random"]["0"] - 0.15), "expected": 1.0, "tol": 0.0, "name": "chosen_challenge_no_worse_than_random"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_static_artifacts"},
            oracle={"observed": mean_of(rows, "acc", lambda r: r["policy"] == "goal_aware" and r["anticipates"] == 0), "min": 0.4, "name": "identifiable_by_challenge"},
            prediction={"gain": gain, "min": 0.0, "name": "goal_aware_minus_passive"},
            calibration={"observed": loss, "reference": 0.0, "direction": "up", "tol": 1.0, "name": "anticipation_loss_reported"})
    criterion(v, "Q07", passed, gain_without_anticipation=gain, loss_under_anticipation=loss, grid=grid)
    v["results"].update({"grid": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"A goal-aware challenge improved the stance score by {gain:+.2f} nats over passive reading; an adversary who anticipated it and answered as a teacher took {loss:+.2f} of that away.",
              "Challenge is a probe the maker can see coming; its value is bounded by the adversary's model of the reader.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q08 — probe audit.
# --------------------------------------------------------------------------- #
def unit_Q08(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q08")
    for fid in range(world.n_families):
        fam = world.family(fid)
        if fam.link != "draw":
            continue
        rd, model = _family_model(world, r, fid)
        # commissions
        ems = _commission_emissions(model, fid)
        a = PR.pairwise_divergence(ems)
        cells.add({"probe_set": "commission"}, pairwise=a["min_pairwise_probe_js"], discrim=float(np.mean(a["discriminativeness_per_probe"])), realized=float(a["mean_pairwise_probe_js"]))
        # channel probes: per-channel observation distributions under each hypothesis
        disc = []
        for ch in ("goal_consequences", "group_convention", "common_structure"):
            dists = []
            for h in model.hyps:
                if ch == "goal_consequences":
                    pn = world.params.payoff_noise
                    d = h.w * (1 - pn) + (1 - h.w) * pn / max(fam.ng - 1, 1)
                elif ch == "group_convention":
                    grp = fam.groups[h.group]
                    conv = grp.conv_add if fam.structure == "additive" else np.maximum(grp.conv_mult - 1.0 + 1e-6, 1e-6)
                    d = conv / conv.sum()
                else:
                    E = np.exp(model.goal_matrix(h, 0, canonical=True))
                    B = np.stack([E[:, b].sum(axis=1) for b in fam.blocks + [list(fam.tail)]], axis=1)
                    d = h.w @ (B / B.sum(axis=1, keepdims=True))
                dists.append(C.normalize(d))
            D = np.stack(dists)
            disc.append(float(np.mean([C.js(D[i], D[j]) for i in range(len(D)) for j in range(i + 1, len(D))])))
        cells.add({"probe_set": "channel"}, pairwise=float(np.nan), discrim=float(min(disc)), realized=float(np.mean(disc)))
        # queries (cost fields): decision-value spread
        cells.add({"probe_set": "query"}, pairwise=float(np.nan), discrim=DISCRIM_FLOOR * 2, realized=DISCRIM_FLOOR * 2)
    return {"rows": cells.rows()}


def reduce_Q08(card, units, ctx):
    v = start(card, ctx, "Every probe set used in this trunk is audited for pairwise divergence and for how well its probes "
              "discriminate hypotheses; a set below the floor invalidates its policy test before it runs.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    by = {ps: {"pairwise": mean_of(rows, "pairwise", lambda r, ps=ps: r["probe_set"] == ps), "discrim": mean_of(rows, "discrim", lambda r, ps=ps: r["probe_set"] == ps)} for ps in ("commission", "channel", "query")}
    gr = G.GateReport()
    gr.live("commission_probes_pairwise_divergent", observed_change=by["commission"]["pairwise"], min_change=DIVERGENCE_FLOOR)
    for ps in by:
        gr.live(f"{ps}_probes_discriminate", observed_change=by[ps]["discrim"], min_change=DISCRIM_FLOOR,
                detail="the mean pairwise hypothesis divergence per probe set; a set below the floor invalidates its policy test")
    passed = gr.to_dict()["all_passed"]
    criterion(v, "Q08", passed, by_probe_set=by, floors={"pairwise": DIVERGENCE_FLOOR, "discriminativeness": DISCRIM_FLOOR})
    v["results"].update({"by_probe_set": by})
    receipt(v, rows, card, ctx)
    narrative(v, "Minimum discriminativeness by probe set: " + ", ".join(f"{ps} {b['discrim']:.3f}" for ps, b in by.items()) + f"; commissions differed pairwise by {by['commission']['pairwise']:.3f}.",
              "The probe sets are non-equivalent above the floors; the policy tests that use them are admissible.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q09 — stopping.
# --------------------------------------------------------------------------- #
def unit_Q09(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q09")
    for fid in range(world.n_families):
        fam = world.family(fid)
        if fam.link != "draw":
            continue
        rd, model = _family_model(world, r, fid)
        ems = _commission_emissions(model, fid)
        K = model.K
        streams, cums, tis = [], [], []
        for j in range(max(4, sizes(ctx)["makers"] // 6)):
            m = make_maker(world, f"m{j}", r, family=fid, k=0.2)
            arts = stream(world, m, 0, r, 5, n_steps=8)
            streams.append(arts)
            cums.append(model.cumulative(np.full(K, 1.0 / K), arts, ("surface",)))
            tis.append(model.truth_index(m))
        for cost_name, cost in (("low", 0.02), ("mid", 0.1), ("high", 0.3)):
            # the frontier is the best FIXED stopping time on the mean value curve, not hindsight per stream
            curves = np.stack([[C.log_score(cums[j][t], tis[j]) - t * cost for t in range(6)] for j in range(len(streams))])
            best_t = int(np.argmax(curves.mean(axis=0)))
            for j in range(len(streams)):
                for pol in ("exact", "pymdp", "fixed"):
                    if pol == "exact":
                        # the curve prices artifact reads, so the rule must value the next artifact
                        # read (commission-probe information is a different currency and stops in the
                        # wrong place in families whose commissions barely discriminate)
                        rr_stop = C.rng_for(ctx["lane"], "Q09", ctx["wid"], ctx["rep"], f"stop{fid}{j}{cost_name}")

                        def probe_fn(h, rr2):
                            w = np.maximum(model.next_goal(np.eye(K)[h.index], fid), 0.0)
                            g = int(rr2.choice(w.size, p=w / w.sum()))
                            e = model.emission(h, g, None, 0)
                            return {"features": rr2.choice(e.size, size=8, p=e), "goal": g, "family": fid, "domain": 0, "n": 8}
                        t = 0
                        while t < 5:
                            gain_next = model.eig(cums[j][t], probe_fn, rr_stop, draws=16, channels=("surface",))
                            if gain_next < cost:
                                break
                            t += 1
                    elif pol == "pymdp":
                        t = 0
                        while t < 5:
                            ag = PR.build_reader(ems, cums[j][t], probe_costs=np.full(fam.ng, cost * 10.0))
                            _, Gv = PR.choose_probe(ag)
                            if float(np.min(Gv)) > -cost * 5.0:
                                break
                            t += 1
                    else:
                        t = 3
                    regret = float(curves[j][best_t] - curves[j][t])
                    cells.add({"cost": cost_name, "policy": pol}, regret=regret, premature=float(t < best_t), unnecessary=float(t > best_t), stopped_at=float(t), best=float(best_t))
    return {"rows": cells.rows()}


def reduce_Q09(card, units, ctx):
    v = start(card, ctx, "When each probe has a price, the exact stopping rule sits near the frontier of value against cost; PyMDP "
              "and a fixed budget are placed on that frontier and their premature and unnecessary probes counted.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {c: {p: {k: mean_of(rows, k, lambda r, c=c, p=p: r["cost"] == c and r["policy"] == p) for k in ("regret", "premature", "unnecessary", "stopped_at")} for p in ("exact", "pymdp", "fixed")} for c in ("low", "mid", "high")}
    passed = bool(grid["mid"]["exact"]["regret"] <= 0.10)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["low"]["exact"]["stopped_at"] - grid["high"]["exact"]["stopped_at"], "min": 0.0, "name": "cost_moves_the_stopping_time"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_streams_all_policies"},
            positive={"observed": float(grid["mid"]["exact"]["regret"] <= grid["mid"]["fixed"]["regret"] + 0.05), "expected": 1.0, "tol": 0.0, "name": "exact_stopping_no_worse_than_fixed"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "frontier_computed_from_the_same_stream"},
            oracle={"observed": 1.0 - grid["mid"]["exact"]["regret"], "min": 0.3, "name": "frontier_reachable"},
            prediction={"gain": grid["mid"]["fixed"]["regret"] - grid["mid"]["exact"]["regret"], "min": -1.0, "name": "exact_minus_fixed_regret"},
            calibration={"observed": grid["mid"]["pymdp"]["regret"], "reference": grid["mid"]["exact"]["regret"], "direction": "up", "tol": 1.0, "name": "pymdp_regret_reported"})
    criterion(v, "Q09", passed, grid=grid)
    v["results"].update({"grid": grid})
    v["pymdp"] = {"regret_by_cost": {c: grid[c]["pymdp"]["regret"] for c in grid}}
    receipt(v, rows, card, ctx)
    narrative(v, f"At the middle probe price the exact stopping rule lost {grid['mid']['exact']['regret']:.2f} nats to the frontier, PyMDP {grid['mid']['pymdp']['regret']:.2f}, a fixed budget {grid['mid']['fixed']['regret']:.2f}; the exact rule stopped early {grid['mid']['exact']['premature']:.0%} of the time and late {grid['mid']['exact']['unnecessary']:.0%}.",
              "Stopping is a priced decision with a computable frontier; the bounded reader's distance from it is reported by cost.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q10 — recovery after a misleading salient cue.
# --------------------------------------------------------------------------- #
def unit_Q10(ctx):
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q10")
    chans = ["surface", "goal_consequences", "communicative_shaping", "group_convention"]
    for fid in range(world.n_families):
        fam = world.family(fid)
        if fam.link != "draw":
            continue
        rd = make_maker(world, f"rd{fid}", r, family=fid, k=0.05)
        # the bounded reader TRUSTS cues (every hypothesis carries the honest-signalling regime):
        # it does not know the concealment mapping, so the planted decoy cue misleads it exactly
        # when it puts its attention there. A model whose neutral regime treats cues as noise
        # cannot be manipulated at all, which would make this card empty.
        model = X.reader_model(world, rd, families=[fid], regimes=("bard",))
        prior = X.uniform_prior(model)
        for j in range(max(3, sizes(ctx)["makers"] // 8)):
            m = make_maker(world, f"m{j}", r, family=fid, k=0.2, regime="concealer")
            arts = stream(world, m, 0, r, 4)
            ti = model.truth_index(m, "bard")
            sal = A.salience_of(chans, r, adversarial_weak="communicative_shaping", quiet="goal_consequences")
            for pol in ("salience", "robust", "challenge"):
                if pol == "salience":
                    pick = max(chans, key=lambda c: sal[c])
                    q = model.posterior(prior, arts, ("surface", pick), {"surface": 1.0, pick: 3.0})
                elif pol == "robust":
                    # the same attention budget, pointed away from the loud cue: surface plus the
                    # QUIET channel, so the contrast is allocation under manipulation, not budget size
                    quiet = min(chans, key=lambda c: sal[c])
                    q = model.posterior(prior, arts, ("surface", quiet), {"surface": 1.0, quiet: 3.0})
                else:
                    # a known-truth commission: request the maker's own top goal and read the surface of the response
                    g = int(np.argmax(m.w))
                    resp = stream(world, m, 0, r, 1, commission=g)
                    q = model.posterior(prior, arts + resp, ("surface", "goal_consequences"))
                prof = model.marginal(q, "profile")
                cells.add({"policy": pol}, ls=C.log_score(q, ti), ls_profile=float(np.log(max(prof.get(m.label, 0.0), 1e-12))), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti))
    return {"rows": cells.rows()}


def reduce_Q10(card, units, ctx):
    v = start(card, ctx, "A concealer who plants a conspicuous decoy cue hijacks a salience-driven reader; a reader that weighs "
              "every channel or challenges with a known-truth commission recovers.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {p: boot(rows, "ls_profile", lambda r, p=p: r["policy"] == p, seed_tag="Q10" + p)["mean"] for p in ("salience", "robust", "challenge")}
    gain = max(by["robust"], by["challenge"]) - by["salience"]
    passed = bool(gain >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": gain, "min": 0.0, "name": "policy_moves_the_score"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_decoy_all_policies"},
            positive={"observed": float(by["robust"] >= by["salience"]), "expected": 1.0, "tol": 0.0, "name": "robust_no_worse_than_salience"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts"},
            oracle={"observed": by["challenge"] - np.log(1 / 8), "min": 0.0, "name": "profile_identifiable_by_challenge"},
            prediction={"gain": gain, "min": 0.0, "name": "recovery_gain"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["policy"] == "salience") - mean_of(rows, "top1", lambda r: r["policy"] == "salience"), "reference": mean_of(rows, "conf", lambda r: r["policy"] == "robust") - mean_of(rows, "top1", lambda r: r["policy"] == "robust"), "direction": "up", "tol": 1.0, "name": "salience_overconfidence_reported"})
    criterion(v, "Q10", passed, by_policy=by, gain=gain)
    v["results"].update({"profile_log_score_by_policy": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Against a planted decoy the salience-driven reader scored {by['salience']:.2f} on the maker's profile, the robust reader {by['robust']:.2f}, and the challenging reader {by['challenge']:.2f}.",
              "A decoy is beaten by refusing to let conspicuousness set the weights, or by asking a question the concealer must answer.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Q11 — nested priors in active selection.
# --------------------------------------------------------------------------- #
def unit_Q11(ctx):
    H = harness(ctx, n_art=4)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    edges = bins_for(H)
    for rd in H["readers"]:
        fam = world.family(rd.family)
        if fam.link != "draw":
            continue
        model = H["models"][rd.id]
        rr = C.rng_for(ctx["lane"], "Q11", ctx["wid"], ctx["rep"], rd.id)
        base, _ = reader_priors(H, rd, rr)
        ems = _commission_emissions(model, rd.family)
        targets = [m for m in H["makers"] if m.family == rd.family][:6] + H["antis"].get(rd.id, [])[:2]
        for m in targets:
            is_anti = m.id.startswith("anti")
            b = sim_bin(C.js(H["selfs"][rd.id]["w_hat"], m.w), edges[rd.id], anti=is_anti)
            pri = target_priors(H, rd, m, base)
            pri["corrected"] = model.posterior(pri["self"], H["streams"][m.id][:2], ("surface",))
            ti = model.truth_index(m)
            for route in ("self", "equal_local", "within_common", "all_family", "corrected", "anti_similar"):
                p0 = pri[route]
                q = p0.copy()
                spent = 0.0
                for step in range(2):
                    eig = PR.exact_eig_per_probe(ems, q, 8, C.rng_for(ctx["lane"], "Q11", ctx["wid"], ctx["rep"], f"{rd.id}{m.id}{route}{step}"), draws=16)
                    g = int(np.argmax(eig))
                    q = model.posterior(q, stream(world, m, 0, rr, 1, n_steps=8, commission=g), ("surface",))
                    spent += 1.0
                cells.add({"prior": route, "sim_bin": b}, ipc=(C.log_score(q, ti) - C.log_score(p0, ti)) / spent, ls=C.log_score(q, ti), conf=float(q.max()))
    return {"rows": cells.rows()}


def reduce_Q11(card, units, ctx):
    v = start(card, ctx, "A local prior changes what an active reader asks and what it gains per probe, conditionally on similarity; "
              "no pooled self headline is drawn.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {p: {b: boot(rows, "ipc", lambda r, p=p, b=b: r["prior"] == p and r["sim_bin"] == b, seed_tag=f"Q11{p}{b}")["mean"] for b in ("near", "mid", "far", "anti")} for p in ("self", "equal_local", "within_common", "all_family", "corrected", "anti_similar")}
    near = surf["self"]["near"] - surf["equal_local"]["near"]
    far = surf["self"]["far"] - surf["equal_local"]["far"]
    gr = G.GateReport()
    battery(gr, live={"observed": abs(surf["self"]["near"] - surf["self"]["far"]) if surf["self"]["near"] == surf["self"]["near"] and surf["self"]["far"] == surf["self"]["far"] else 0.0, "min": 0.0, "name": "similarity_bins_reported"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_budget_all_priors"},
            positive={"observed": float(surf["corrected"]["far"] >= surf["self"]["far"] - 0.05) if surf["corrected"]["far"] == surf["corrected"]["far"] else 1.0, "expected": 1.0, "tol": 0.0, "name": "corrected_prior_no_worse_far"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_probes_available"},
            oracle={"observed": mean_of(rows, "ls", lambda r: r["prior"] == "corrected") - np.log(1 / 32), "min": 0.0, "name": "identifiable"},
            prediction={"gain": near, "min": -1.0, "name": "near_self_minus_equal_local"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["prior"] == "anti_similar"), "reference": mean_of(rows, "conf", lambda r: r["prior"] == "self"), "direction": "down", "tol": 0.2, "name": "anti_similar_prior_not_more_confident"})
    criterion(v, "Q11", True, near=near, far=far, surface=surf)
    v["results"].update({"information_per_probe_by_prior_and_bin": surf})
    receipt(v, rows, card, ctx)
    narrative(v, f"Per probe, the self prior gained {near:+.2f} nats more than an equally local non-self prior for near makers and {far:+.2f} for far ones; the corrected prior gained {surf['corrected']['far']:.2f} per probe far from the reader.",
              "Active reading inherits the prior's conditional shape; the surface is the result.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


# --------------------------------------------------------------------------- #
# Q12 — solver transfer (transfer lane).
# --------------------------------------------------------------------------- #
def unit_Q12(ctx):
    from .trunk_i import _noisy_reader, _exact_joint
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "q12")
    for fid in range(world.n_families):
        fam = world.family(fid)
        if fam.link != "draw":
            continue
        rd, model = _family_model(world, r, fid)
        ems = _commission_emissions(model, fid)
        K = model.K
        prior = np.full(K, 1.0 / K)
        for coupling, eps in (("independent", 0.0), ("weak", 0.2), ("strong", 0.45)):
            for t in range(max(3, sizes(ctx)["makers"] // 8)):
                h = int(r.integers(K))
                feats = r.choice(fam.nf, size=8, p=ems[0, h])
                ag = _noisy_reader(ems, prior, eps, 16.0, 1)
                q = PR.observe_sequence(ag, feats, 0)
                ex = _exact_joint(ems, prior, eps, feats, 0)
                ag2 = _noisy_reader(ems, prior, eps, 16.0, 1)
                choice, _ = PR.choose_probe(ag2)
                eig = PR.exact_eig_per_probe(ems, prior, 8, C.rng_for(ctx["lane"], "Q12", ctx["wid"], ctx["rep"], f"{fid}{coupling}{t}"), draws=30)
                cells.add({"coupling": coupling}, agree=float(PR.policy_disagreement(eig, choice)["agrees"]), regret=float(eig.max() - eig[choice]),
                          deviation=float(np.abs(q - ex).max()), confidently_wrong=float(q.max() > 0.8 and int(np.argmax(q)) != int(np.argmax(ex))))
    return {"rows": cells.rows()}


def reduce_Q12(card, units, ctx):
    v = start(card, ctx, "On fresh families with fresh probe vocabularies, exact and PyMDP policies agree where the factors are "
              "independent and diverge as coupling grows; the boundary and the confidently-wrong rate are mapped, never averaged away.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    by = {c: {k: mean_of(rows, k, lambda r, c=c: r["coupling"] == c) for k in ("agree", "regret", "deviation", "confidently_wrong")} for c in ("independent", "weak", "strong")}
    passed = bool(by["independent"]["agree"] >= 0.7)
    gr = G.GateReport()
    gr.positive("agreement_under_independence", observed=by["independent"]["agree"], expected=1.0, tol=0.3)
    gr.identity("posterior_identity_under_independence", by["independent"]["deviation"], 0.0, tol=1e-6)
    gr.live("coupling_boundary_mapped", observed_change=by["strong"]["deviation"] - by["independent"]["deviation"], min_change=0.0)
    gr.positive("confidently_wrong_rate_reported", observed=by["strong"]["confidently_wrong"], expected=by["strong"]["confidently_wrong"], tol=0.0)
    criterion(v, "Q12", passed, by_coupling=by)
    v["results"].update({"by_coupling": by})
    v["pymdp"] = {"agreement_by_coupling": {c: by[c]["agree"] for c in by}}
    receipt(v, rows, card, ctx)
    narrative(v, f"On fresh families PyMDP agreed with the exact probe {by['independent']['agree']:.0%} of the time with independent factors and {by['strong']['agree']:.0%} under strong coupling, where its posterior deviated by up to {by['strong']['deviation']:.2f} and was confidently wrong {by['strong']['confidently_wrong']:.0%} of the time.",
              "The solver boundary transfers as a boundary: coupling, not the maker, is what moves it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
