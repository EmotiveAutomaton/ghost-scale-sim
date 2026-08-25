"""Trunk Q: active reading (spec section 8).

Probes are commissions: the reader asks the maker for an artifact with one goal amplified, or
takes a free look. Every policy is scored against the exact Monte-Carlo expected information
gain about the maker's profile. The legacy PyMDP agent supplies its own probe choice through a
controllable probe factor and is read against card I03's mean-field map.
"""
from __future__ import annotations

import itertools

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import make_maker, population, emission
from .. import exact as X, self_other as SO, pymdp_reader as PR, opportunities as OP
from . import finish, worlds_for, decide_state

A_AMP = 4.0
N_STEPS = 24


def _cw(w, p, ng):
    v = np.asarray(w, float).copy()
    if p is not None and 0 <= p < ng:
        v[p] *= A_AMP
    return v / v.sum()


def probe_emissions(world, template, tier="CREATOR", assumption="plain", family=None):
    """E[p, k, f]: the predicted surface under hypothesis k when the reader commissions goal p.
    The last probe is the free look (no commission)."""
    fam = world.family if family is None else family
    names = list(fam)
    P = world.ng + 1
    E = np.zeros((P, len(names), world.nf))
    for p in range(P):
        for k, n in enumerate(names):
            E[p, k] = X.reader_emission(world, template, None, _cw(fam[n], p, world.ng), 0, tier, assumption, n)
    return E, names


def eig_by_probe(E, q, names, n_steps, rng, draws=300):
    prior = dict(zip(names, np.asarray(q, float)))
    return np.array([X.expected_information_gain(prior, {n: E[p, k] for k, n in enumerate(names)}, n_steps, rng, draws)
                     for p in range(E.shape[0])])


def probe_artifact(world, m, p, rng, n_steps=N_STEPS, goal_w=None):
    w = _cw(m.w if goal_w is None else goal_w, p, world.ng)
    g = int(rng.choice(world.ng, p=w))
    dist, _ = emission(world, m, g, 0, rng)
    return {"features": rng.choice(world.nf, size=int(n_steps), p=dist), "goals": [g], "domain": 0, "probe": int(p)}


def exact_post(E, q, arts):
    ll = np.log(np.maximum(np.asarray(q, float), 1e-300)).copy()
    for a in arts:
        ll += np.log(np.maximum(E[a["probe"]][:, a["features"]], 1e-300)).sum(axis=1)
    out = np.exp(ll - ll.max())
    return out / out.sum()


def H(q):
    q = np.asarray(q, float)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


def uncertainty_probe(E, q):
    """Classic uncertainty sampling: probe where the predictive distribution is most uncertain."""
    return int(np.argmax([H(np.asarray(q) @ E[p]) for p in range(E.shape[0])]))


def pymdp_probe(ag):
    ag.infer_policies()
    return int(ag.sample_action()[1])


def _episode(world, E, names, m, policy, rng, n_probes=4, q0=None, ag=None, n_steps=N_STEPS):
    K = len(names)
    q = np.full(K, 1 / K) if q0 is None else np.asarray(q0, float).copy()
    arts, ig, probes = [], [], []
    for _ in range(n_probes):
        if policy == "exact_eig":
            p = int(np.argmax(eig_by_probe(E, q, names, n_steps, rng, 200)))
        elif policy == "uncertainty":
            p = uncertainty_probe(E, q)
        elif policy == "random":
            p = int(rng.integers(E.shape[0]))
        elif policy == "free_look":
            p = E.shape[0] - 1
        elif policy == "pymdp":
            p = pymdp_probe(ag)
        else:
            raise ValueError(policy)
        a = probe_artifact(world, m, p, rng, n_steps)
        h0 = H(q)
        q = exact_post(E, q, [a])
        ig.append(h0 - H(q))
        arts.append(a)
        probes.append(p)
        if policy == "pymdp":
            PR.observe_sequence(ag, a["features"], p)
    return {"q": q, "ig": ig, "probes": probes, "arts": arts}


# --------------------------------------------------------------------------- #
def run_Q01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The legacy PyMDP reader's chosen probe is the probe with the largest "
                    "exact expected information gain about the maker's profile at the agent's one-step "
                    "horizon; a utility-only agent has no such preference.", "METHOD")
    rows, spreads, placebo = [], [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            E, names = probe_emissions(world, world.sig)
            K = len(names)
            for s in range(3):
                rng = C.rng_for("Q01", wid, s)
                for t in range(6):
                    kind = ("uniform", "self_first", "peaked")[t % 3]
                    if kind == "uniform":
                        prior = np.full(K, 1 / K)
                    elif kind == "self_first":
                        sp = SO.self_first_prior(world, world.family[names[int(rng.integers(K))]])
                        prior = np.array([sp[n] for n in names])
                    else:
                        prior = np.full(K, 0.04)
                        prior[int(rng.integers(K))] += 1.0 - 0.04 * K
                    ag = PR.build_reader(E, prior, probe_costs=np.zeros(E.shape[0]))
                    q = np.asarray(ag.infer_states([0, 0])[0], float)      # the agent's own belief after its dummy look
                    probe = pymdp_probe(ag)
                    ut = PR.build_reader(E, prior, probe_costs=np.zeros(E.shape[0]), use_info_gain=False)
                    ut.infer_states([0, 0])
                    probe_u = pymdp_probe(ut)
                    eig1 = eig_by_probe(E, q, names, 1, rng, 600)
                    eig8 = eig_by_probe(E, q, names, 8, rng, 300)
                    d1, d8 = PR.policy_disagreement(eig1, probe), PR.policy_disagreement(eig8, probe)
                    rows.append({"wid": wid, "prior": kind, "agree_1": float(d1["agrees"]),
                                 "agree_1_within_tol": float(eig1[probe] >= eig1.max() - 0.02),
                                 "rank_1": d1["agent_rank"], "agree_8": float(d8["agrees"]),
                                 "utility_only_agree": float(PR.policy_disagreement(eig1, probe_u)["agrees"])})
                    spreads.append(float(eig1.max() - eig1.min()))
            flat = np.repeat(E[-1:], E.shape[0], axis=0)
            placebo.append(float(np.ptp(eig_by_probe(flat, np.full(K, 1 / K), names, 1, C.rng_for("Q01", wid, 9), 600))))
    agree = float(np.mean([r["agree_1_within_tol"] for r in rows]))
    gr = G.GateReport()
    gr.placebo("identical_probes_have_no_eig_spread", observed_max_deviation=float(max(placebo)), tol=0.03,
               detail="with every probe emitting the same surface the exact EIG spread is Monte-Carlo noise only")
    gr.live("probes_differ_in_information", observed_change=float(np.mean(spreads)), min_change=0.02)
    gr.positive("agent_prefers_the_informative_probe", observed=agree, expected=1.0, tol=0.2,
                detail="the criterion: PyMDP agrees with the exact one-step EIG-best probe (within 0.02 nats) on at least 80 percent of decisions")
    v["results"] = {"agreement_exact_top1": float(np.mean([r["agree_1"] for r in rows])), "agreement_within_tol": agree,
                    "agreement_at_8_steps": float(np.mean([r["agree_8"] for r in rows])),
                    "mean_rank": float(np.mean([r["rank_1"] for r in rows])),
                    "utility_only_rival_agreement": float(np.mean([r["utility_only_agree"] for r in rows])),
                    "by_prior": {k: float(np.mean([r["agree_1_within_tol"] for r in rows if r["prior"] == k])) for k in ("uniform", "self_first", "peaked")},
                    "eig_spread_mean": float(np.mean(spreads)), "criterion_C_Q01": {"passed": bool(agree >= 0.8)}}
    v["what_must_hold_outside_the_simulation"] = "nothing; a solver-agreement check"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "A reader choosing commissions by exact expected information gain "
                    "captures more information per probe than random or uncertainty sampling, and the "
                    "PyMDP reader does the same until probe cost dominates.", "CONSTRUCTED_MECHANISM")
    pols = ("exact_eig", "pymdp", "uncertainty", "random", "free_look")
    res = {p: {"ig": [], "ls": [], "by_world": {}} for p in pols}
    cost_pref = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            E, names = probe_emissions(world, world.sig)
            K = len(names)
            for s in range(3):
                rng = C.rng_for("Q02", wid, s)
                for m in population(world, 8, rng, k_choices=(0.0, 0.3), prefix=f"m{s}"):
                    truth = names.index(m.profile)
                    for p in pols:
                        ag = PR.build_reader(E, np.full(K, 1 / K), probe_costs=np.zeros(E.shape[0])) if p == "pymdp" else None
                        ep = _episode(world, E, names, m, p, C.rng_for("Q02", wid, s, m.id + p), ag=ag)
                        res[p]["ig"].append(float(np.mean(ep["ig"])))
                        res[p]["ls"].append(float(np.log(max(ep["q"][truth], 1e-12))))
                        res[p]["by_world"].setdefault(wid, []).append(float(np.mean(ep["ig"])))
                    for c in (0.0, 0.5, 2.0):
                        costs = np.full(E.shape[0], c)
                        costs[-1] = 0.0
                        ag = PR.build_reader(E, np.full(K, 1 / K), probe_costs=costs)
                        ep = _episode(world, E, names, m, "pymdp", C.rng_for("Q02", wid, s, m.id + f"c{c}"), ag=ag)
                        cost_pref.setdefault(str(c), []).append(float(np.mean([pp < E.shape[0] - 1 for pp in ep["probes"]])))
    diff = {w: [a - b for a, b in zip(res["exact_eig"]["by_world"][w], res["random"]["by_world"][w])] for w in res["random"]["by_world"]}
    boot = C.hboot(diff, np.random.default_rng(C.seed("Q02:boot")), draws=500)
    gr = G.GateReport()
    gr.live("commissions_change_information", observed_change=float(np.mean(res["exact_eig"]["ig"]) - np.mean(res["free_look"]["ig"])), min_change=0.01,
            detail="choosing commissions must move realised information relative to the free look")
    gr.positive("probe_cost_suppresses_commissions", observed=float(np.mean(cost_pref["2.0"]) <= np.mean(cost_pref["0.0"])), expected=1.0, tol=0.0,
                detail="a known answer: a costly commission is chosen no more often than a free one")
    v["results"] = {"realized_ig_per_probe": {p: float(np.mean(r["ig"])) for p, r in res.items()},
                    "final_log_score": {p: float(np.mean(r["ls"])) for p, r in res.items()},
                    "exact_minus_random_ig": boot, "commission_share_by_cost": {k: float(np.mean(x)) for k, x in cost_pref.items()},
                    "criterion_C_Q02": {"passed": bool(boot["mean"] >= 0.05)}}
    v["what_must_hold_outside_the_simulation"] = "a maker complies with a commission in the way the reader models"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Choosing which two of six episodes to inspect under the self-first prior "
                    "predicts the seventh episode's goal at least as well as uncertainty sampling.",
                    "CONSTRUCTED_MECHANISM")
    rows = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            E, names = probe_emissions(world, world.sig)
            K = len(names)
            rng = C.rng_for("Q03", wid, 0)
            readers = [make_maker(world, f"reader{i}", n, rng, k=0.05) for i, n in enumerate(names)]
            selfs = {r.id: SO.measure_self(world, r, C.rng_for("Q03", wid, 1, r.id)) for r in readers}
            makers = population(world, 24, rng, k_choices=(0.0, 0.3))
            pairs = list(itertools.combinations(range(6), 2))
            for r in readers:
                sp = SO.self_first_prior(world, selfs[r.id]["w_hat"])
                spv = np.array([sp[n] for n in names])
                uni = np.full(K, 1 / K)
                for m in makers:
                    mrng = C.rng_for("Q03", wid, 2, r.id + m.id)
                    tags = [int(mrng.integers(world.ng + 1)) for _ in range(6)]
                    eps = [probe_artifact(world, m, p, mrng) for p in tags]
                    target = probe_artifact(world, m, world.ng, mrng)
                    g_next = target["goals"][0]

                    def pick(prior_vec):
                        e = eig_by_probe(E, prior_vec, names, N_STEPS, mrng, 150)
                        sc = np.array([e[tags[i]] + e[tags[j]] for i, j in pairs])
                        return pairs[int(np.argmax(sc + 1e-9 * mrng.random(len(pairs))))]

                    def score(pair, prior_vec):
                        q = exact_post(E, prior_vec, [eps[pair[0]], eps[pair[1]]])
                        pred = sum(q[k] * world.family[n] for k, n in enumerate(names))
                        return float(np.log(max(pred[g_next] / pred.sum(), 1e-12))), float(np.log(max(q[names.index(m.profile)], 1e-12)))

                    sel = {"self_first": pick(spv), "uncertainty": pick(uni), "random": pairs[int(mrng.integers(len(pairs)))]}
                    sel["oracle_pair"] = max(pairs, key=lambda pr: score(pr, spv)[0])
                    row = {"wid": wid, "dist": SO.js(selfs[r.id]["w_hat"], m.w)}
                    for k, pr in sel.items():
                        row[k + "_matched"], row[k + "_profile"] = score(pr, spv)
                    row["uncertainty_own"] = score(sel["uncertainty"], uni)[0]
                    rows.append(row)
    by_world = {}
    for r in rows:
        by_world.setdefault(r["wid"], []).append(r["self_first_matched"] - r["uncertainty_matched"])
    boot = C.hboot(by_world, np.random.default_rng(C.seed("Q03:boot")), draws=500)
    gr = G.GateReport()
    gr.positive("oracle_pair_is_the_ceiling", observed=float(np.mean([r["oracle_pair_matched"] >= r["self_first_matched"] - 1e-9 for r in rows])), expected=1.0, tol=0.0)
    gr.live("selection_matters", observed_change=float(np.mean([r["oracle_pair_matched"] - r["random_matched"] for r in rows])), min_change=0.02,
            detail="if the best pair does not beat a random pair, episode selection has nothing to select")
    v["results"] = {"self_minus_uncertainty_matched_prior": boot,
                    "means": {k: float(np.mean([r[k] for r in rows])) for k in ("self_first_matched", "uncertainty_matched", "uncertainty_own", "random_matched", "oracle_pair_matched")},
                    "criterion_C_Q03": {"passed": bool(boot["mean"] >= 0.0)}}
    v["what_must_hold_outside_the_simulation"] = "episode summaries are available before inspection"
    return finish(card, v, gr, __file__, decide_state(gr))


ITEMS = (("artifact", 1.0, 0.3), ("biography", 0.5, 0.9), ("prior_work", 1.0, 0.5), ("tool_records", 0.8, 0.2), ("reputation", 0.3, 1.0))


def _item_channels(world, E_free, names, rng):
    """Per item: an outcome channel P(outcome | hypothesis) in one padded observation space."""
    K = len(names)
    O = max(world.nf, K + 1, 4)
    ch = {}
    a = np.zeros((K, O))
    a[:, :world.nf] = E_free
    ch["artifact"] = a
    b = np.zeros((K, O))
    for k in range(K):
        b[k, :K] = 0.3 / (K - 1)
        b[k, k] = 0.7
    ch["biography"] = b
    perm = np.asarray(world.domains[1].perm)
    pw = np.zeros((K, O))
    pw[:, perm] = E_free
    ch["prior_work"] = pw
    cw = OP.ChoiceWorld(world.ng, world.family, names)
    menu = OP.menu(cw, rng)
    t = np.zeros((K, O))
    for k, n in enumerate(names):
        t[k, :4] = np.exp([OP.choice_loglik(cw, world.family[n], menu, aa) for aa in range(4)])
    ch["tool_records"] = t
    r = np.zeros((K, O))
    r[:, :2] = 0.5
    ch["reputation"] = r
    return ch, O


def _channel_eig(ch, q):
    q = np.asarray(q, float)
    po = q @ ch
    h_prior = H(q)
    out = 0.0
    for o in range(ch.shape[1]):
        if po[o] <= 0:
            continue
        post = q * ch[:, o] / po[o]
        out += po[o] * (h_prior - H(post))
    return float(out)


def run_Q04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Readers that buy context buy what discriminates makers per unit cost, "
                    "not what is polished; the PyMDP buyer does so through its epistemic term.",
                    "CONSTRUCTED_MECHANISM")
    purchases = {"exact": [], "pymdp": [], "polish": []}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            E, names = probe_emissions(world, world.sig)
            K = len(names)
            for s in range(3):
                rng = C.rng_for("Q04", wid, s)
                ch, O = _item_channels(world, E[-1], names, rng)
                stack = np.stack([ch[i] for i, _, _ in ITEMS])
                costs = np.array([c for _, c, _ in ITEMS])
                polish = np.array([p for _, _, p in ITEMS])
                for m in population(world, 8, rng, prefix=f"m{s}"):
                    truth = names.index(m.profile)
                    for buyer in purchases:
                        q = np.full(K, 1 / K)
                        ag = PR.build_reader(stack, q, probe_costs=costs) if buyer == "pymdp" else None
                        for step in range(3):
                            eig = np.array([_channel_eig(stack[i], q) for i in range(len(ITEMS))])
                            if buyer == "exact":
                                i = int(np.argmax(eig / costs))
                            elif buyer == "polish":
                                i = int(np.argmax(polish / costs))
                            else:
                                i = pymdp_probe(ag)
                            o = int(rng.choice(O, p=stack[i][truth] / stack[i][truth].sum()))
                            purchases[buyer].append({"wid": wid, "item": ITEMS[i][0], "eig": float(eig[i]),
                                                     "eig_rank": int((eig / costs > eig[i] / costs[i]).sum()) + 1,
                                                     "polish": float(polish[i]), "max_eig_per_cost": float((eig / costs).max())})
                            q = q * stack[i][:, o]
                            q = q / q.sum()
                            if buyer == "pymdp":
                                PR.observe_sequence(ag, np.array([o]), i)
    summ = {}
    for buyer, rows in purchases.items():
        summ[buyer] = {"mean_eig_bought": float(np.mean([r["eig"] for r in rows])), "mean_polish_bought": float(np.mean([r["polish"] for r in rows])),
                       "share_top_discriminating": float(np.mean([r["eig_rank"] == 1 for r in rows])),
                       "item_shares": {i: float(np.mean([r["item"] == i for r in rows])) for i, _, _ in ITEMS}}
    gr = G.GateReport()
    gr.positive("exact_buyer_buys_top_discrimination", observed=summ["exact"]["share_top_discriminating"], expected=1.0, tol=1e-9,
                detail="by construction the exact buyer ranks by discrimination per cost; this gate checks the harness")
    gr.positive("reputation_is_never_bought_by_the_exact_buyer", observed=summ["exact"]["item_shares"]["reputation"], expected=0.0, tol=1e-9)
    gr.live("polish_rival_differs", observed_change=summ["polish"]["mean_polish_bought"] - summ["exact"]["mean_polish_bought"], min_change=0.05)
    v["results"] = {"buyers": summ, "criterion_C_Q04": {"pymdp_share_top": summ["pymdp"]["share_top_discriminating"], "pymdp_polish": summ["pymdp"]["mean_polish_bought"],
                                                          "passed": bool(summ["pymdp"]["share_top_discriminating"] > summ["polish"]["share_top_discriminating"])},
                    "note": "the PyMDP buyer evaluates one outcome per purchase; the exact buyer scores the full item channel"}
    v["what_must_hold_outside_the_simulation"] = "context items have a knowable discriminating value before purchase"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Stopping when the next probe's expected information gain falls below its "
                    "cost is near the hindsight optimum across costs; premature and unnecessary probing "
                    "rates are reported for exact and PyMDP readers.", "CONSTRUCTED_MECHANISM")
    out = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            E, names = probe_emissions(world, world.sig)
            K = len(names)
            E_aug = np.concatenate([E, np.repeat(world.synth[None, None, :], K, axis=1)], axis=0)
            for c in (0.0, 0.01, 0.03, 0.1, 0.3):
                cell = out.setdefault(str(c), {"regret": [], "premature": [], "unnecessary": [], "n_probes": [], "pymdp_n_probes": [], "pymdp_regret": []})
                rng = C.rng_for("Q05", wid, 0, f"c{c}")
                for m in population(world, 10, rng, k_choices=(0.0, 0.3)):
                    truth = names.index(m.profile)
                    q = np.full(K, 1 / K)
                    traj = [float(np.log(max(q[truth], 1e-12)))]
                    unnecessary = 0
                    for n in range(8):
                        e = eig_by_probe(E, q, names, N_STEPS, rng, 150)
                        if e.max() < c:
                            break
                        if int(np.argmax(q)) == truth and q.max() >= 0.9:
                            unnecessary += 1
                        a = probe_artifact(world, m, int(np.argmax(e)), rng)
                        q = exact_post(E, q, [a])
                        traj.append(float(np.log(max(q[truth], 1e-12))))
                    n_used = len(traj) - 1
                    values = [ls - c * i for i, ls in enumerate(traj)]
                    qq = q.copy()
                    for extra in range(n_used, 8):
                        e = eig_by_probe(E, qq, names, N_STEPS, rng, 100)
                        a = probe_artifact(world, m, int(np.argmax(e)), rng)
                        qq = exact_post(E, qq, [a])
                        values.append(float(np.log(max(qq[truth], 1e-12))) - c * (extra + 1))
                    cell["regret"].append(float(max(values) - values[n_used]))
                    cell["premature"].append(float(int(np.argmax(q)) != truth))
                    cell["unnecessary"].append(float(unnecessary))
                    cell["n_probes"].append(float(n_used))
                    costs = np.full(E_aug.shape[0], 10.0 * c)
                    costs[-1] = 0.0
                    ag = PR.build_reader(E_aug, np.full(K, 1 / K), probe_costs=costs)
                    qp = np.full(K, 1 / K)
                    used = 0
                    for n in range(8):
                        p = pymdp_probe(ag)
                        if p == E_aug.shape[0] - 1:
                            break
                        a = probe_artifact(world, m, p, rng)
                        PR.observe_sequence(ag, a["features"], p)
                        qp = exact_post(E, qp, [a])
                        used += 1
                    cell["pymdp_n_probes"].append(float(used))
                    cell["pymdp_regret"].append(float(max(values) - (np.log(max(qp[truth], 1e-12)) - c * used)))
    table = {c: {k: float(np.mean(x)) for k, x in d.items()} for c, d in out.items()}
    gr = G.GateReport()
    gr.positive("free_probes_are_never_declined", observed=table["0.0"]["n_probes"], expected=8.0, tol=1e-9,
                detail="at zero cost the exact stopping rule never stops early: the known answer")
    gr.live("cost_shortens_inspection", observed_change=table["0.0"]["n_probes"] - table["0.3"]["n_probes"], min_change=1.0)
    v["results"] = {"by_cost": table, "note": "PyMDP costs enter its preference vector at ten times the log-score cost; the mapping is reported, not fitted"}
    v["what_must_hold_outside_the_simulation"] = "inspection cost and log-score value share a scale"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Against a concealer that plants its decoy's commissioned behaviour, "
                    "information-driven probing helps only a reader that models the planting; the naive "
                    "active reader walks into planted evidence.", "BOUNDARY")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            E_plain, names = probe_emissions(world, world.sig, assumption="plain")
            K = len(names)
            E_aware = np.zeros_like(E_plain)
            for p in range(E_plain.shape[0]):
                for k, n in enumerate(names):
                    d = world.decoy_of[n]
                    E_aware[p, k] = X.reader_emission(world, world.sig, None, _cw(world.family[d], p, world.ng), 0, "CREATOR", "concealer", n)
            rng = C.rng_for("Q06", wid, 0)
            for m in population(world, 12, rng, regimes=("concealer",)):
                truth = names.index(m.profile)
                decoy_w = world.family[world.decoy_of[m.profile]]
                for reader, E in (("naive", E_plain), ("aware", E_aware)):
                    for pol in ("exact_eig", "uncertainty", "random"):
                        q = np.full(K, 1 / K)
                        prng = C.rng_for("Q06", wid, 1, m.id + reader + pol)
                        for _ in range(4):
                            if pol == "exact_eig":
                                p = int(np.argmax(eig_by_probe(E, q, names, N_STEPS, prng, 150)))
                            elif pol == "uncertainty":
                                p = uncertainty_probe(E, q)
                            else:
                                p = int(prng.integers(E.shape[0]))
                            a = probe_artifact(world, m, p, prng, goal_w=decoy_w)
                            q = exact_post(E, q, [a])
                        res.setdefault(reader, {}).setdefault(pol, {}).setdefault(wid, []).append(float(np.log(max(q[truth], 1e-12))))
    table = {r: {p: C.hboot(d, np.random.default_rng(C.seed("Q06" + r + p)), draws=400) for p, d in pols.items()} for r, pols in res.items()}
    gains = {r: float(table[r]["exact_eig"]["mean"] - table[r]["uncertainty"]["mean"]) for r in table}
    gr = G.GateReport()
    gr.live("planting_hurts_the_naive_reader", observed_change=float(table["aware"]["random"]["mean"] - table["naive"]["random"]["mean"]), min_change=0.1,
            detail="the aware model must score higher than the naive one on planted evidence, or the planting is not reaching the reader")
    v["results"] = {"log_score_by_reader_and_policy": table, "active_gain_vs_uncertainty": gains,
                    "criterion_C_Q06": {"passed": bool(max(gains.values()) > 0.0), "closure_if_never_better": bool(max(gains.values()) <= 0.0)}}
    v["what_must_hold_outside_the_simulation"] = "a concealer knows which goal was commissioned"
    state = decide_state(gr)
    if state == "LANDED" and max(gains.values()) <= 0.0:
        return finish(card, v, gr, __file__, "SCIENTIFIC_CLOSED", "active selection never beats uncertainty sampling under the adaptive adversary")
    return finish(card, v, gr, __file__, state)
