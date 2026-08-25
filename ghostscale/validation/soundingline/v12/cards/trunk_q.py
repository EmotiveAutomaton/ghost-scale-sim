"""Trunk Q: active reading (spec section 8).

Probes are commissions: the reader asks the maker for an artifact with one goal amplified, or
takes a free look. The reader's exact path scores an artifact with the artifact-structured
likelihood (one goal per artifact drawn from the commissioned profile, then features given the
goal). The legacy PyMDP agent carries a per-feature mixture emission as its A matrix, which is its
own approximation; Q01 compares the two.

Deviation D-V12-1 (recorded in RESULTS.md). The first discovery run scored the "exact" path with
the per-feature mixture likelihood, which treats every feature as an independent draw from the
profile's goal blend. Over twenty-four near-identical draws that likelihood is overconfident by
construction: it put Q02 at ceiling (each artifact resolved the posterior completely) and B04's
regime posterior below the uniform prior. The Q cards and B04 were re-run on the structured
likelihood; the first verdicts are kept under attempt1/ beside the new ones.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import logsumexp

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import make_maker, population, emission
from .. import exact as X, self_other as SO, pymdp_reader as PR, opportunities as OP
from . import finish, worlds_for, decide_state

A_AMP = 4.0
N_STEPS = 24
DEVIATION = ("D-V12-1: the exact path is scored with the artifact-structured likelihood (one goal per "
             "artifact); the first discovery verdict, scored with a per-feature mixture likelihood, is "
             "kept as attempt1")


def _cw(w, p, ng):
    v = np.asarray(w, float).copy()
    if p is not None and 0 <= p < ng:
        v[p] *= A_AMP
    return v / v.sum()


def _onehot(g, ng):
    v = np.zeros(ng)
    v[g] = 1.0
    return v


def H(q):
    q = np.asarray(q, float)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


class Model:
    """The reader's model of commissioned artifacts. Hypotheses are (label, profile name, regime
    assumption, base profile weights). G[h, g, f] is P(feature | goal g) under hypothesis h's
    regime cue; W[p, h, g] the goal weights under probe p; the goal mixture is the PyMDP agent's
    A matrix. The last probe is the free look."""

    def __init__(self, world, template, tier, hyps):
        self.world = world
        self.labels = [h[0] for h in hyps]
        K, ng, nf = len(hyps), world.ng, world.nf
        P = ng + 1
        self.K, self.P = K, P
        self.G = np.zeros((K, ng, nf))
        self.W = np.zeros((P, K, ng))
        for k, (label, name, assumption, base_w) in enumerate(hyps):
            for g in range(ng):
                self.G[k, g] = X.reader_emission(world, template, None, _onehot(g, ng), 0, tier, assumption, name)
            for p in range(P):
                self.W[p, k] = _cw(base_w, p, ng)
        self.logG = np.log(np.maximum(self.G, 1e-300))
        self.logW = np.log(np.maximum(self.W, 1e-300))
        self.mixture = np.einsum("pkg,kgf->pkf", self.W, self.G)

    def loglik(self, p, feats):
        per_goal = self.logG[:, :, np.asarray(feats)].sum(axis=2)          # [K, ng]
        return logsumexp(self.logW[p] + per_goal, axis=1)

    def posterior(self, q, arts):
        ll = np.log(np.maximum(np.asarray(q, float), 1e-300)).copy()
        for a in arts:
            ll += self.loglik(a["probe"], a["features"])
        out = np.exp(ll - ll.max())
        return out / out.sum()

    def sample(self, k, p, n_steps, rng):
        g = int(rng.choice(self.world.ng, p=self.W[p, k]))
        return rng.choice(self.world.nf, size=int(n_steps), p=self.G[k, g])

    def eig(self, q, p, n_steps, rng, draws=150, groups=None):
        """Monte-Carlo expected information gain (nats) about the hypothesis, or about the group
        marginal when ``groups`` maps hypotheses to groups, from one artifact under probe p."""
        q = np.asarray(q, float)

        def ent(qq):
            return H(np.bincount(groups, weights=qq) if groups is not None else qq)
        h0 = ent(q)
        posts = []
        for _ in range(int(draws)):
            k = int(rng.choice(self.K, p=q))
            f = self.sample(k, p, n_steps, rng)
            posts.append(ent(self.posterior(q, [{"probe": p, "features": f}])))
        return float(h0 - np.mean(posts))

    def eig_all(self, q, n_steps, rng, draws=150, groups=None):
        return np.array([self.eig(q, p, n_steps, rng, draws, groups) for p in range(self.P)])


def plain_model(world, template=None, tier="CREATOR", family=None):
    fam = world.family if family is None else family
    return Model(world, world.sig if template is None else template, tier, [(n, n, "plain", fam[n]) for n in fam])


def probe_artifact(world, m, p, rng, n_steps=N_STEPS, goal_w=None):
    """The maker's actual artifact under commission p (the free look when p is the last probe)."""
    w = _cw(m.w if goal_w is None else goal_w, p, world.ng)
    g = int(rng.choice(world.ng, p=w))
    dist, _ = emission(world, m, g, 0, rng)
    return {"features": rng.choice(world.nf, size=int(n_steps), p=dist), "goals": [g], "domain": 0, "probe": int(p)}


def mixture_eig(E, q, names, n_steps, rng, draws=300):
    """EIG under the per-feature mixture model: the PyMDP agent's own objective."""
    prior = dict(zip(names, np.asarray(q, float)))
    return np.array([X.expected_information_gain(prior, {n: E[p, k] for k, n in enumerate(names)}, n_steps, rng, draws)
                     for p in range(E.shape[0])])


def uncertainty_probe(E, q):
    """Classic uncertainty sampling: the probe whose next feature is least predictable."""
    return int(np.argmax([H(np.asarray(q) @ E[p]) for p in range(E.shape[0])]))


def pymdp_probe(ag):
    ag.infer_policies()
    return int(ag.sample_action()[1])


def _episode(world, model, m, policy, rng, n_probes=4, q0=None, ag=None, n_steps=N_STEPS):
    q = np.full(model.K, 1 / model.K) if q0 is None else np.asarray(q0, float).copy()
    arts, ig, probes = [], [], []
    for _ in range(n_probes):
        if policy == "exact_eig":
            p = int(np.argmax(model.eig_all(q, n_steps, rng, 120)))
        elif policy == "uncertainty":
            p = uncertainty_probe(model.mixture, q)
        elif policy == "random":
            p = int(rng.integers(model.P))
        elif policy == "free_look":
            p = model.P - 1
        elif policy == "pymdp":
            p = pymdp_probe(ag)
        else:
            raise ValueError(policy)
        a = probe_artifact(world, m, p, rng, n_steps)
        h0 = H(q)
        q = model.posterior(q, [a])
        ig.append(h0 - H(q))
        arts.append(a)
        probes.append(p)
        if policy == "pymdp":
            PR.observe_sequence(ag, a["features"], p)
    return {"q": q, "ig": ig, "probes": probes, "arts": arts}


# --------------------------------------------------------------------------- #
def run_Q01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "The legacy PyMDP reader's chosen probe is the probe with the largest "
                    "expected information gain about the maker's profile: under its own per-feature "
                    "mixture model by construction, and under the artifact-structured model as the "
                    "question; a utility-only agent has no such preference.", "METHOD")
    rows, spreads, placebo = [], [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            model = plain_model(world)
            E, names, K = model.mixture, model.labels, model.K
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
                    e_agent = mixture_eig(E, q, names, 1, rng, 600)
                    e_struct = model.eig_all(q, N_STEPS, rng, 200)
                    rows.append({"wid": wid, "prior": kind,
                                 "agree_agent_model": float(e_agent[probe] >= e_agent.max() - 0.02),
                                 "agree_structured": float(e_struct[probe] >= e_struct.max() - 0.02),
                                 "rank_structured": PR.policy_disagreement(e_struct, probe)["agent_rank"],
                                 "utility_only_agree_structured": float(e_struct[probe_u] >= e_struct.max() - 0.02),
                                 "structured_best_is_agent_model_best": float(int(np.argmax(e_struct)) == int(np.argmax(e_agent)))})
                    spreads.append(float(e_struct.max() - e_struct.min()))
            flat = np.repeat(E[-1:], E.shape[0], axis=0)
            placebo.append(float(np.ptp(mixture_eig(flat, np.full(K, 1 / K), names, 1, C.rng_for("Q01", wid, 9), 600))))
    agree_s = float(np.mean([r["agree_structured"] for r in rows]))
    agree_a = float(np.mean([r["agree_agent_model"] for r in rows]))
    gr = G.GateReport()
    gr.placebo("identical_probes_have_no_eig_spread", observed_max_deviation=float(max(placebo)), tol=0.03,
               detail="with every probe emitting the same surface the EIG spread is Monte-Carlo noise only")
    gr.live("probes_differ_in_structured_information", observed_change=float(np.mean(spreads)), min_change=0.02)
    gr.positive("agent_implements_eig_under_its_own_model", observed=agree_a, expected=1.0, tol=0.2,
                detail="the solver check: the agent's probe is within 0.02 nats of the best probe under the agent's own one-step mixture model")
    v["results"] = {"agreement_with_structured_eig": agree_s, "agreement_with_agent_model_eig": agree_a,
                    "mean_rank_structured": float(np.mean([r["rank_structured"] for r in rows])),
                    "utility_only_rival_agreement": float(np.mean([r["utility_only_agree_structured"] for r in rows])),
                    "structured_and_agent_model_share_best_probe": float(np.mean([r["structured_best_is_agent_model_best"] for r in rows])),
                    "by_prior": {k: float(np.mean([r["agree_structured"] for r in rows if r["prior"] == k])) for k in ("uniform", "self_first", "peaked")},
                    "structured_eig_spread_mean": float(np.mean(spreads)),
                    "criterion_C_Q01": {"agreement": agree_s, "passed": bool(agree_s >= 0.8)}, "deviation": DEVIATION}
    v["what_must_hold_outside_the_simulation"] = "nothing; a solver-agreement check"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "A reader choosing commissions by expected information gain captures "
                    "more information per probe than random or uncertainty sampling, and the PyMDP "
                    "reader does the same until probe cost dominates.", "CONSTRUCTED_MECHANISM")
    pols = ("exact_eig", "pymdp", "uncertainty", "random", "free_look")
    res = {p: {"ig": [], "ls": [], "by_world": {}} for p in pols}
    cost_pref = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            model = plain_model(world)
            E, names, K = model.mixture, model.labels, model.K
            for s in range(3):
                rng = C.rng_for("Q02", wid, s)
                for m in population(world, 8, rng, k_choices=(0.0, 0.3), prefix=f"m{s}"):
                    truth = names.index(m.profile)
                    for p in pols:
                        ag = PR.build_reader(E, np.full(K, 1 / K), probe_costs=np.zeros(E.shape[0])) if p == "pymdp" else None
                        ep = _episode(world, model, m, p, C.rng_for("Q02", wid, s, m.id + p), ag=ag)
                        res[p]["ig"].append(float(np.mean(ep["ig"])))
                        res[p]["ls"].append(float(np.log(max(ep["q"][truth], 1e-12))))
                        res[p]["by_world"].setdefault(wid, []).append(float(np.mean(ep["ig"])))
                    for c in (0.0, 0.5, 2.0):
                        costs = np.full(E.shape[0], c)
                        costs[-1] = 0.0
                        ag = PR.build_reader(E, np.full(K, 1 / K), probe_costs=costs)
                        ep = _episode(world, model, m, "pymdp", C.rng_for("Q02", wid, s, m.id + f"c{c}"), ag=ag)
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
                    "criterion_C_Q02": {"passed": bool(boot["mean"] >= 0.05)}, "deviation": DEVIATION}
    v["what_must_hold_outside_the_simulation"] = "a maker complies with a commission in the way the reader models"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Choosing which two of six episodes to inspect under the self-first prior "
                    "predicts the seventh episode's goal at least as well as uncertainty sampling.",
                    "CONSTRUCTED_MECHANISM")
    rows = []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            model = plain_model(world)
            names, K = model.labels, model.K
            rng = C.rng_for("Q03", wid, 0)
            readers = [make_maker(world, f"reader{i}", n, rng, k=0.05) for i, n in enumerate(names)]
            selfs = {r.id: SO.measure_self(world, r, C.rng_for("Q03", wid, 1, r.id)) for r in readers}
            makers = population(world, 24, rng, k_choices=(0.0, 0.3))
            pairs = list(itertools.combinations(range(6), 2))
            for r in readers:
                sp = SO.self_first_prior(world, selfs[r.id]["w_hat"])
                spv = np.array([sp[n] for n in names])
                uni = np.full(K, 1 / K)
                e_self = model.eig_all(spv, N_STEPS, C.rng_for("Q03", wid, 4, r.id), 120)
                e_uni = model.eig_all(uni, N_STEPS, C.rng_for("Q03", wid, 5, r.id), 120)
                for m in makers:
                    mrng = C.rng_for("Q03", wid, 2, r.id + m.id)
                    tags = [int(mrng.integers(world.ng + 1)) for _ in range(6)]
                    eps = [probe_artifact(world, m, p, mrng) for p in tags]
                    target = probe_artifact(world, m, world.ng, mrng)
                    g_next = target["goals"][0]

                    def pick(e):
                        sc = np.array([e[tags[i]] + e[tags[j]] for i, j in pairs])
                        return pairs[int(np.argmax(sc + 1e-9 * mrng.random(len(pairs))))]

                    def score(pair, prior_vec):
                        q = model.posterior(prior_vec, [eps[pair[0]], eps[pair[1]]])
                        pred = sum(q[k] * world.family[n] for k, n in enumerate(names))
                        return float(np.log(max(pred[g_next] / pred.sum(), 1e-12))), float(np.log(max(q[names.index(m.profile)], 1e-12)))

                    sel = {"self_first": pick(e_self), "uncertainty": pick(e_uni), "random": pairs[int(mrng.integers(len(pairs)))]}
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
                    "criterion_C_Q03": {"passed": bool(boot["mean"] >= 0.0)}, "deviation": DEVIATION}
    v["what_must_hold_outside_the_simulation"] = "episode summaries are available before inspection"
    return finish(card, v, gr, __file__, decide_state(gr))


ITEMS = (("artifact", 1.0, 0.3), ("biography", 0.5, 0.9), ("prior_work", 1.0, 0.5), ("tool_records", 0.8, 0.2), ("reputation", 0.3, 1.0))


def _label_channels(world, names, rng):
    """Outcome channels P(outcome | hypothesis) for the label-type items, in one padded space."""
    K = len(names)
    O = max(world.nf, K + 1, 4)
    ch = {}
    b = np.zeros((K, O))
    for k in range(K):
        b[k, :K] = 0.3 / (K - 1)
        b[k, k] = 0.7
    ch["biography"] = b
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
        out += po[o] * (h_prior - H(q * ch[:, o] / po[o]))
    return float(out)


def run_Q04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Readers that buy context buy what discriminates makers per unit cost, "
                    "not what is polished; the PyMDP buyer does so through its epistemic term.",
                    "CONSTRUCTED_MECHANISM")
    purchases = {"exact": [], "pymdp": [], "polish": []}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            model = plain_model(world)
            names, K = model.labels, model.K
            free = model.P - 1
            for s in range(3):
                rng = C.rng_for("Q04", wid, s)
                ch, O = _label_channels(world, names, rng)
                art = np.zeros((K, O))
                art[:, :world.nf] = model.mixture[free]
                stack = np.stack([art, ch["biography"], art, ch["tool_records"], ch["reputation"]])    # PyMDP A per item
                costs = np.array([c for _, c, _ in ITEMS])
                polish = np.array([p for _, _, p in ITEMS])
                for m in population(world, 8, rng, prefix=f"m{s}"):
                    truth = names.index(m.profile)
                    for buyer in purchases:
                        q = np.full(K, 1 / K)
                        ag = PR.build_reader(stack, q, probe_costs=costs) if buyer == "pymdp" else None
                        for step in range(3):
                            e_art = model.eig(q, free, N_STEPS, rng, 100)
                            eig = np.array([e_art, _channel_eig(stack[1], q), e_art, _channel_eig(stack[3], q), _channel_eig(stack[4], q)])
                            if buyer == "exact":
                                i = int(np.argmax(eig / costs))
                            elif buyer == "polish":
                                i = int(np.argmax(polish / costs))
                            else:
                                i = pymdp_probe(ag)
                            purchases[buyer].append({"wid": wid, "item": ITEMS[i][0], "eig": float(eig[i]),
                                                     "eig_rank": int((eig / costs > eig[i] / costs[i] + 1e-12).sum()) + 1,
                                                     "polish": float(polish[i]), "max_eig_per_cost": float((eig / costs).max())})
                            if ITEMS[i][0] in ("artifact", "prior_work"):
                                a = probe_artifact(world, m, free, rng)
                                q = model.posterior(q, [a])
                                if buyer == "pymdp":
                                    PR.observe_sequence(ag, a["features"], i)
                            else:
                                o = int(rng.choice(O, p=stack[i][truth] / stack[i][truth].sum()))
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
                    "note": "artifact items deliver a full artifact scored with the structured likelihood; label items deliver one outcome. "
                            "The PyMDP buyer evaluates its one-step mixture information for every item", "deviation": DEVIATION}
    v["what_must_hold_outside_the_simulation"] = "context items have a knowable discriminating value before purchase"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Stopping when the next probe's expected information gain falls below its "
                    "cost is near the hindsight optimum across costs; premature and unnecessary probing "
                    "rates are reported for exact and PyMDP readers.", "CONSTRUCTED_MECHANISM")
    out = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            model = plain_model(world)
            E, names, K = model.mixture, model.labels, model.K
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
                        e = model.eig_all(q, N_STEPS, rng, 100)
                        if e.max() < c:
                            break
                        if int(np.argmax(q)) == truth and q.max() >= 0.9:
                            unnecessary += 1
                        a = probe_artifact(world, m, int(np.argmax(e)), rng)
                        q = model.posterior(q, [a])
                        traj.append(float(np.log(max(q[truth], 1e-12))))
                    n_used = len(traj) - 1
                    values = [ls - c * i for i, ls in enumerate(traj)]
                    qq = q.copy()
                    for extra in range(n_used, 8):
                        e = model.eig_all(qq, N_STEPS, rng, 80)
                        a = probe_artifact(world, m, int(np.argmax(e)), rng)
                        qq = model.posterior(qq, [a])
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
                        qp = model.posterior(qp, [a])
                        used += 1
                    cell["pymdp_n_probes"].append(float(used))
                    cell["pymdp_regret"].append(float(max(values) - (np.log(max(qp[truth], 1e-12)) - c * used)))
    table = {c: {k: float(np.mean(x)) for k, x in d.items()} for c, d in out.items()}
    gr = G.GateReport()
    gr.positive("free_probes_are_never_declined", observed=table["0.0"]["n_probes"], expected=8.0, tol=1e-9,
                detail="at zero cost the exact stopping rule never stops early: the known answer")
    gr.live("cost_shortens_inspection", observed_change=table["0.0"]["n_probes"] - table["0.3"]["n_probes"], min_change=1.0)
    v["results"] = {"by_cost": table, "note": "PyMDP costs enter its preference vector at ten times the log-score cost; the mapping is reported, not fitted",
                    "deviation": DEVIATION}
    v["what_must_hold_outside_the_simulation"] = "inspection cost and log-score value share a scale"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_Q06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Against a concealer that plants its decoy's commissioned behaviour, "
                    "information-driven probing helps only a reader that models the planting; the naive "
                    "active reader walks into planted evidence.", "BOUNDARY")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            names = world.family_names
            naive = plain_model(world)
            aware = Model(world, world.sig, "CREATOR", [(n, n, "concealer", world.family[world.decoy_of[n]]) for n in names])
            K = naive.K
            rng = C.rng_for("Q06", wid, 0)
            for m in population(world, 12, rng, regimes=("concealer",)):
                truth = names.index(m.profile)
                decoy_w = world.family[world.decoy_of[m.profile]]
                for reader, model in (("naive", naive), ("aware", aware)):
                    for pol in ("exact_eig", "uncertainty", "random"):
                        q = np.full(K, 1 / K)
                        prng = C.rng_for("Q06", wid, 1, m.id + reader + pol)
                        for _ in range(4):
                            if pol == "exact_eig":
                                p = int(np.argmax(model.eig_all(q, N_STEPS, prng, 100)))
                            elif pol == "uncertainty":
                                p = uncertainty_probe(model.mixture, q)
                            else:
                                p = int(prng.integers(model.P))
                            a = probe_artifact(world, m, p, prng, goal_w=decoy_w)
                            q = model.posterior(q, [a])
                        res.setdefault(reader, {}).setdefault(pol, {}).setdefault(wid, []).append(float(np.log(max(q[truth], 1e-12))))
    table = {r: {p: C.hboot(d, np.random.default_rng(C.seed("Q06" + r + p)), draws=400) for p, d in pols.items()} for r, pols in res.items()}
    gains = {r: float(table[r]["exact_eig"]["mean"] - table[r]["uncertainty"]["mean"]) for r in table}
    gr = G.GateReport()
    gr.live("planting_hurts_the_naive_reader", observed_change=float(table["aware"]["random"]["mean"] - table["naive"]["random"]["mean"]), min_change=0.1,
            detail="the aware model must score higher than the naive one on planted evidence, or the planting is not reaching the reader")
    v["results"] = {"log_score_by_reader_and_policy": table, "active_gain_vs_uncertainty": gains,
                    "criterion_C_Q06": {"passed": bool(max(gains.values()) > 0.0), "closure_if_never_better": bool(max(gains.values()) <= 0.0)},
                    "deviation": DEVIATION}
    v["what_must_hold_outside_the_simulation"] = "a concealer knows which goal was commissioned"
    state = decide_state(gr)
    if state == "LANDED" and max(gains.values()) <= 0.0:
        return finish(card, v, gr, __file__, "SCIENTIFIC_CLOSED", "active selection never beats uncertainty sampling under the adaptive adversary")
    return finish(card, v, gr, __file__, state)
