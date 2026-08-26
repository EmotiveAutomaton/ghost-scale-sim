"""Trunk C: common prior and nested similarity (spec §9).

One harness serves the sixteen cards. Per unit (world, repeat): a population of makers spread
over families, groups and ecologies; a set of readers with measured self-models; every reader's
own likelihood model (its templates and habit); every maker's artifact stream; and for each
same-family reader x maker pair the per-artifact channel log-likelihoods. Priors differ by route;
likelihoods are shared, so a route's gain is the prior's gain and nothing else. Rows are
aggregated to unit-level cell means before they leave the worker.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as X, priors as P, attention as A, projection as PJ
from ..world import make_maker, population, similarity, stream, histogram, N_METHODS
from . import (battery, boot, ci_abs, ci_pos, criterion, decide_state, finish, held_out_classifier, narrative, receipt, rng, sizes,
               start, world_for, mean_of, pursuit_of, edges_for, sim_bin)

DOSES = (1, 2, 4, 8, 16)
N_ART = 18
CH = ("surface",)


# --------------------------------------------------------------------------- #
# Harness.
# --------------------------------------------------------------------------- #
class Cells:
    """Accumulates metric sums per cell; ``rows()`` returns one row per cell with means and n."""

    def __init__(self, wid, rep):
        self.wid, self.rep, self.acc = wid, rep, {}

    def add(self, factors: dict, **metrics):
        key = tuple(sorted((k, str(v)) for k, v in factors.items()))
        d = self.acc.setdefault(key, {"_n": 0, "_f": dict(factors)})
        d["_n"] += 1
        for m, x in metrics.items():
            if x is None or (isinstance(x, float) and x != x):
                continue
            d[m] = d.get(m, 0.0) + float(x)

    def rows(self) -> list:
        out = []
        for d in self.acc.values():
            n = d["_n"]
            row = {"wid": self.wid, "rep": self.rep, "n": n, **d["_f"]}
            for m, x in d.items():
                if not m.startswith("_"):
                    row[m] = x / n
            out.append(row)
        return out


def _reader_kinds(world, fam_id, i, r, kind="practitioner"):
    if kind == "practitioner":
        return make_maker(world, f"reader{i}", r, family=fam_id, k=0.05)
    m = make_maker(world, f"reader{i}", r, family=fam_id, k=0.05 if kind == "familiar" else 0.5)
    m.method_pref = np.full_like(m.method_pref, 1.0 / N_METHODS)
    return m


def harness(ctx, n_art: int = N_ART, n_steps: int | None = None, channels=CH, anti: bool = True,
            with_others: bool = True, k_makers=(0.0, 0.3), tag: str = "C") -> dict:
    world = world_for(ctx)
    sz = sizes(ctx)
    r = C.rng_for(ctx["lane"], tag, int(ctx["wid"]), int(ctx["rep"]), "pop")
    nf = world.n_families
    makers = population(world, sz["makers"], r, k=None)
    for i, m in enumerate(makers):
        m.k = float(k_makers[i % len(k_makers)])
        from ..world import corrupt
        m.template = corrupt(world.family(m.family).methods, m.k, r)
    readers = []
    for i in range(sz["readers"]):
        readers.append(make_maker(world, f"reader{i}", r, family=i % nf, k=0.05))
    # planted anti-similar makers per reader (inverted profile in the reader's family and group)
    antis = {}
    if anti:
        for rd in readers:
            fam = world.family(rd.family)
            lst = []
            for j in range(3):
                w = C.normalize(1.0 - rd.w + 0.05)
                m = make_maker(world, f"anti{rd.id}_{j}", r, family=rd.family, group=rd.group, w=w, k=float(k_makers[j % len(k_makers)]))
                lst.append(m)
            antis[rd.id] = lst
    steps = world.params.n_steps if n_steps is None else int(n_steps)
    streams = {}
    for m in makers + [a for lst in antis.values() for a in lst]:
        streams[m.id] = stream(world, m, 0, C.rng_for(ctx["lane"], tag, int(ctx["wid"]), int(ctx["rep"]), "art|" + m.id), n_art, n_steps=steps)
    models, selfs = {}, {}
    for rd in readers:
        models[rd.id] = X.reader_model(world, rd, families=[rd.family])
        selfs[rd.id] = P.measure_self(world, rd, models[rd.id], C.rng_for(ctx["lane"], tag, int(ctx["wid"]), int(ctx["rep"]), "self|" + rd.id))
    L = {}
    for rd in readers:
        model = models[rd.id]
        for m in makers + antis.get(rd.id, []):
            if m.family != rd.family:
                continue
            L[(rd.id, m.id)] = model.loglik(streams[m.id], channels)
    return {"world": world, "makers": makers, "readers": readers, "antis": antis, "streams": streams, "models": models,
            "selfs": selfs, "L": L, "sz": sz, "steps": steps, "channels": channels}


def reader_priors(H, rd, rr, regime="plain"):
    """Reader-level routes (target-independent) and the matching report."""
    model = H["models"][rd.id]
    fam_makers = [m for m in H["makers"] if m.family == rd.family]
    others = [(r2, H["selfs"][r2.id]) for r2 in H["readers"] if r2.family == rd.family and r2.id != rd.id]
    dummy = fam_makers[0]
    priors, rep = P.routes_for(model, rd, H["selfs"][rd.id], fam_makers, others, rr, dummy, regime=regime)
    priors.pop("oracle", None)
    priors["within_common"] = P.population_prior(model, fam_makers, family=rd.family)
    return priors, rep


def target_priors(H, rd, m, base: dict, history=None, channels=CH):
    model = H["models"][rd.id]
    fam_makers = [x for x in H["makers"] if x.family == rd.family]
    out = dict(base)
    out["oracle"] = P.oracle(model, m)
    out["within_common"] = P.population_prior(model, fam_makers, family=rd.family, exclude=m.id)
    if history:
        out["target_learned"] = model.posterior(out["within_common"], history, channels)
    return out


def posterior_at(model, prior, L, n):
    lp = np.log(np.maximum(prior, 1e-300)) + L[:n].sum(axis=0)
    return C.softmax(lp)


def pair_iter(H, include_anti=True):
    for rd in H["readers"]:
        targets = [m for m in H["makers"] if m.family == rd.family]
        for m in targets:
            yield rd, m, False
        if include_anti:
            for m in H["antis"].get(rd.id, []):
                yield rd, m, True


def bins_for(H):
    """Similarity-bin edges per reader from the individual-level distance over its targets."""
    edges = {}
    for rd in H["readers"]:
        d = [C.js(H["selfs"][rd.id]["w_hat"], m.w) for m in H["makers"] if m.family == rd.family]
        edges[rd.id] = edges_for(d)
    return edges


def hidden_goal_ls(model, post, fam, art):
    if art["goal"] < 0:
        return None
    w = model.next_goal(post, fam)
    return float(np.log(max(w[art["goal"]], 1e-12)))


# --------------------------------------------------------------------------- #
# C01 — measure the self-model.
# --------------------------------------------------------------------------- #
def unit_C01(ctx):
    world = world_for(ctx)
    sz = sizes(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "readers")
    diag = {"label": [], "placebo": [], "oracle_gain": [], "conf": [], "correct": [], "len_corr": []}
    for i in range(sz["readers"]):
        rd = make_maker(world, f"reader{i}", r, family=i % world.n_families, k=0.05)
        model = X.reader_model(world, rd, families=[rd.family])
        for d in range(2):
            sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "C01", ctx["wid"], ctx["rep"], f"{rd.id}|{d}"), domain=d)
            cells.add({"domain": d, "baseline": "frequency"}, gain=sm["heldout_logscore_self_model"] - sm["heldout_logscore_frequency"],
                      trans_gain=sm["heldout_transition_self"] - sm["heldout_transition_frequency"])
            cells.add({"domain": d, "baseline": "population"}, gain=sm["heldout_logscore_self_model"] - sm["heldout_logscore_population"],
                      trans_gain=sm["heldout_transition_self"] - sm["heldout_transition_frequency"])
            if d == 0:
                diag["label"].append(sm["label_correct"])
                post = sm["posterior"]
                diag["conf"].append(float(post.max()))
                diag["correct"].append(sm["label_correct"])
                # placebo: the same artifacts in permuted order give the same estimate (exact inference)
                arts = stream(world, rd, 0, C.rng_for(ctx["lane"], "C01", ctx["wid"], ctx["rep"], f"pl|{rd.id}"), 12)
                q1 = model.posterior(X.uniform_prior(model), arts, CH)
                q2 = model.posterior(X.uniform_prior(model), arts[::-1], CH)
                diag["placebo"].append(float(np.abs(q1 - q2).max()))
                # oracle: the true profile supplied instead of measured
                orc = P.oracle(model, rd)
                hold = stream(world, rd, 0, C.rng_for(ctx["lane"], "C01", ctx["wid"], ctx["rep"], f"or|{rd.id}"), 6)
                from ..exact import prefix_continuation, frequency_continuation
                g_or = np.mean([prefix_continuation(model, orc, a, 6) - frequency_continuation(arts, a, 6, world.family(rd.family).nf) for a in hold])
                diag["oracle_gain"].append(float(g_or))
    return {"rows": cells.rows(), "diag": {k: (float(np.mean(v)) if v else None) for k, v in diag.items() if k not in ("conf", "correct")},
            "conf": diag["conf"], "correct": diag["correct"]}


def reduce_C01(card, units, ctx):
    v = start(card, ctx, "A reader can measure its own production model from its own artifacts well enough to predict "
              "its held-out continuations and method choices above frequency and family-population baselines.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    res = {}
    for d in (0, 1):
        for b in ("frequency", "population"):
            res[f"gain_domain{d}_vs_{b}"] = boot(rows, "gain", lambda r, d=d, b=b: r["domain"] == d and r["baseline"] == b, seed_tag=f"C01{d}{b}")
    trans = boot(rows, "trans_gain", lambda r: r["baseline"] == "frequency", seed_tag="C01t")
    conf = [c for u in units for c in u["conf"]]
    corr = [c for u in units for c in u["correct"]]
    diag = {k: float(np.nanmean([u["diag"][k] for u in units if u["diag"].get(k) is not None])) for k in ("label", "placebo", "oracle_gain")}
    g0, g1 = res["gain_domain0_vs_frequency"]["mean"], res["gain_domain1_vs_frequency"]["mean"]
    passed = bool(g0 >= 0.05 and g1 >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": g0, "min": 0.05, "name": "self_model_beats_frequency"},
            placebo={"observed": diag["placebo"], "tol": 1e-9, "name": "artifact_order_irrelevant"},
            positive={"observed": diag["label"], "expected": 1.0, "tol": 0.35, "name": "planted_label_recovered"},
            surface={"accuracy": abs(g0 - g1), "chance": 0.0, "tol": 0.15, "name": "gain_not_a_domain_surface_effect"},
            oracle={"observed": diag["oracle_gain"], "min": 0.03, "name": "true_profile_predicts_continuation"},
            prediction={"gain": trans["mean"], "min": 0.0, "name": "held_out_transitions"},
            calibration={"observed": C.ece(conf, corr), "reference": 0.25, "direction": "down", "name": "label_confidence_ece"})
    criterion(v, "C01", passed, gain_domain0=g0, gain_domain1=g1)
    v["results"].update({"gains": res, "transition_gain": trans, "diagnostics": diag, "label_ece": C.ece(conf, corr)})
    rec = receipt(v, rows, card, ctx)
    narrative(v, f"Readers predicted the rest of their own held-out artifacts {g0:+.2f} nats per feature better than pooled frequency "
                 f"in their native convention and {g1:+.2f} in the second; their own method choices were predicted {trans['mean']:+.2f} "
                 f"nats better than method frequency; the planted profile label was recovered {diag['label']:.0%} of the time.",
              "The measured self-model is a usable object in every family: the first prior of trunk C rests on it." if passed else
              "The self-model does not beat frequency at this construction; trunk C's self route is built on a weaker object than intended.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C02 — nested similarity ruler.
# --------------------------------------------------------------------------- #
def unit_C02(ctx):
    from scipy.stats import spearmanr
    world = world_for(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "ruler")
    surface_js = []
    for fid in range(world.n_families):
        fam = world.family(fid)
        rd = make_maker(world, "ref", r, family=fid, group=0, ecology=0, k=0.0, pressure=0.0, attention="goal", habit_strength=0.0)
        model_all = X.reader_model(world, rd, families=None)
        prior_all = X.uniform_prior(model_all)

        RULER_CH = ("surface", "group_convention", "goal_consequences")
        def measured(m, arts):
            # the individual ruler reads the payoff record directly (continuous), the others the grid posterior
            pn = world.params.payoff_noise
            pays = [a["payoff_obs"] for a in arts if "payoff_obs" in a and int(a["family"]) == fid and 0 <= a["payoff_obs"] < fam.ng]
            w_emp = (np.bincount(pays, minlength=fam.ng) + 0.5) / (len(pays) + 0.5 * fam.ng) if pays else np.full(fam.ng, 1.0 / fam.ng)
            q = model_all.posterior(prior_all, arts, RULER_CH)
            goal_conc = 0.0
            if int(arts[0]["family"]) == fid and fam.link == "draw":
                hist = np.zeros(fam.ng)
                for a in arts:
                    LM = model_all.goal_matrix(model_all.hyps[model_all.by_family[fid][0]], a["domain"])
                    hist += C.softmax(LM[:, np.asarray(a["features"])].sum(axis=1))
                goal_conc = -C.entropy(hist / hist.sum())
            fam_mass = sum(q[i] for i in model_all.by_family[fid])
            qf = np.zeros(model_all.K)
            for i in model_all.by_family[fid]:
                qf[i] = q[i]
            qf = C.normalize(qf) if qf.sum() > 0 else prior_all
            w_hat = model_all.profile_mean(qf, fid)
            grp = model_all.marginal(qf, "group")
            g_hat = np.array([grp.get(g, 0.0) for g in range(len(fam.groups))])
            fit = float(np.mean(model_all.loglik(arts, ("surface",)).max(axis=1)))
            same = int(arts[0]["family"]) == fid
            surf = C.js(histogram(np.concatenate([a["features"] for a in arts]), fam.nf),
                        histogram(np.concatenate([a["features"] for a in stream(world, rd, 0, r, 4)]), fam.nf)) if same else 1.0
            g_rel = float(g_hat[1] / (g_hat[0] + g_hat[1])) if (g_hat.size > 1 and g_hat[0] + g_hat[1] > 1e-12) else 0.5
            return {"common": 1.0 - fam_mass, "group": g_rel, "individual": C.js(rd.w, w_emp),
                    "expertise": -fit, "state": goal_conc, "surface": surf}
        planted, meas = {lv: [] for lv in ("common", "group", "expertise", "individual", "state", "surface")}, {lv: [] for lv in ("common", "group", "expertise", "individual", "state", "surface")}
        # individual: profile ladder (payoff draws are one per artifact, so the ruler needs many artifacts)
        for t in np.linspace(0, 1, 8):
            w = C.normalize((1 - t) * rd.w + t * C.normalize(1.0 - rd.w + 0.05))
            m = make_maker(world, "m", r, family=fid, group=0, ecology=0, w=w, k=0.0, habit_strength=0.0)
            arts = stream(world, m, 0, r, 64, n_steps=4)
            planted["individual"].append(C.js(rd.w, w))
            meas["individual"].append(measured(m, arts)["individual"])
        # group: convention ladder between group 0 and group 1, measured as the continuous per-artifact
        # log-likelihood difference between the two group hypotheses on the convention channel (a
        # posterior saturates into a step; a log-likelihood ratio stays graded). Additive families
        # carry their convention in conv_add, the rest in conv_mult.
        mk0 = make_maker(world, "g0", r, family=fid, group=0, ecology=0, w=rd.w, k=0.0, habit_strength=0.0)
        mk1 = make_maker(world, "g1", r, family=fid, group=1, ecology=0, w=rd.w, k=0.0, habit_strength=0.0)
        h0, h1 = model_all.truth_index(mk0), model_all.truth_index(mk1)
        g0, g1 = fam.groups[0], fam.groups[1]
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            m = make_maker(world, "m", r, family=fid, group=0, ecology=0, w=rd.w, k=0.0, habit_strength=0.0)
            if fam.structure == "additive":
                saved = g0.conv_add.copy()
                g0.conv_add = (1 - t) * saved + t * g1.conv_add
            else:
                saved = g0.conv_mult.copy()
                g0.conv_mult = (1 - t) * saved + t * g1.conv_mult
            arts = stream(world, m, 0, r, 32)
            if fam.structure == "additive":
                g0.conv_add = saved
            else:
                g0.conv_mult = saved
            LL = model_all.loglik(arts, ("group_convention",))
            planted["group"].append(float(t))
            meas["group"].append(float(np.mean(LL[:, h1] - LL[:, h0])))
        # expertise: corruption ladder
        for k in (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.75, 0.9):
            m = make_maker(world, "m", r, family=fid, group=0, ecology=0, w=rd.w, k=k, habit_strength=0.0)
            arts = stream(world, m, 0, r, 12)
            planted["expertise"].append(k)
            meas["expertise"].append(measured(m, arts)["expertise"])
        # state: pressure ladder (sharper goal draws); rungs separated so adjacent levels do not saturate
        for pr in (0.0, 0.15, 0.4, 0.8, 1.5):
            m = make_maker(world, "m", r, family=fid, group=0, ecology=0, w=rd.w, k=0.0, pressure=pr, attention="none", habit_strength=0.0)
            arts = stream(world, m, 0, r, 96, n_steps=4)
            planted["state"].append(pr)
            meas["state"].append(measured(m, arts)["state"])
        # surface: a ladder of partial feature relabelling, read as native
        own_h = histogram(np.concatenate([a["features"] for a in stream(world, rd, 0, r, 24)]), fam.nf)
        perm = fam.domains[1].perm
        for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            m = make_maker(world, "m", r, family=fid, group=0, ecology=0, w=rd.w, k=0.0, habit_strength=0.0)
            arts = stream(world, m, 0, r, 24)
            feats = np.concatenate([a["features"] for a in arts])
            swap = r.random(feats.size) < t
            feats = np.where(swap, perm[feats], feats)
            planted["surface"].append(float(t))
            meas["surface"].append(C.js(histogram(feats, fam.nf), own_h))
        # common: other families
        for f2 in range(world.n_families):
            m = make_maker(world, "m", r, family=f2, group=0, ecology=0, k=0.0, habit_strength=0.0)
            arts = stream(world, m, 0, r, 6)
            planted["common"].append(float(f2 != fid) * (1 + float(world.family(f2).structure != fam.structure)))
            meas["common"].append(measured(m, arts)["common"])
        for lv in planted:
            x, y = np.asarray(planted[lv]), np.asarray(meas[lv])
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                rho = 0.0
            else:
                rho = float(spearmanr(x, y).statistic)
            cells.add({"level": lv}, spearman=rho)
        surface_js.append(meas["surface"][-1])
    return {"rows": cells.rows(), "surface_js": float(np.mean(surface_js))}


def reduce_C02(card, units, ctx):
    v = start(card, ctx, "Measured distances at each nested level recover their planted orderings, and surface "
              "similarity is identified separately from every other level.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    by = {lv: boot(rows, "spearman", lambda r, lv=lv: r["level"] == lv, seed_tag="C02" + lv) for lv in ("common", "group", "expertise", "individual", "state", "surface")}
    worst = min(b["mean"] for b in by.values())
    passed = bool(all(b["mean"] >= 0.8 for b in by.values()))
    gr = G.GateReport()
    for lv, b in by.items():
        gr.positive(f"{lv}_recovers_planted_ordering", observed=b["mean"], expected=1.0, tol=0.2)
    gr.identity("surface_separately_identified", float(by["surface"]["mean"] > 0.5), 1.0, tol=0.0,
                detail="relabelling features alone moves the surface ruler monotonically")
    criterion(v, "C02", passed, worst_level_spearman=worst)
    v["results"].update({"spearman_by_level": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Each planted level's ordering was recovered from artifacts alone with rank correlations from {worst:.2f} up; "
                 f"surface convention moved only the surface ruler.",
              "The nested ruler is validated: every C card that bins makers by similarity bins them by a measured quantity, not a label.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# Shared scoring over routes x bins x doses.
# --------------------------------------------------------------------------- #
def route_cells(ctx, H, routes, doses=DOSES, hidden=True, tag="C"):
    cells = Cells(ctx["wid"], ctx["rep"])
    edges = bins_for(H)
    reports = []
    extra = {"cheap_near": [], "cheap_far": []}
    for rd in H["readers"]:
        rr = C.rng_for(ctx["lane"], tag, ctx["wid"], ctx["rep"], "routes|" + rd.id)
        base, rep = reader_priors(H, rd, rr)
        reports.append(rep)
        model = H["models"][rd.id]
        fam = rd.family
        r_arts = stream(H["world"], rd, 0, rr, 4)
        for m, is_anti in [(m, a) for r2, m, a in pair_iter(H) if r2.id == rd.id]:
            priors = target_priors(H, rd, m, base)
            L = H["L"][(rd.id, m.id)]
            d = C.js(H["selfs"][rd.id]["w_hat"], m.w)
            b = sim_bin(d, edges[rd.id], anti=is_anti)
            cheap = C.js(histogram(np.concatenate([a["features"] for a in H["streams"][m.id][:2]]), H["world"].family(fam).nf),
                         histogram(np.concatenate([a["features"] for a in r_arts]), H["world"].family(fam).nf))
            arts = H["streams"][m.id]
            for route in routes:
                if route not in priors:
                    continue
                for n in doses:
                    post = posterior_at(model, priors[route], L, n)
                    sc = X.score_rows(model, post, m)
                    hg = hidden_goal_ls(model, post, fam, arts[n]) if (hidden and n < len(arts)) else None
                    cells.add({"route": route, "sim_bin": b, "dose": n}, ls=sc["ls"], ls_profile=sc["ls_profile"], top1=sc["top1"],
                              conf=sc["conf"], brier=sc["brier"], l1=sc["l1"], hidden=hg, dist=d, cheap=cheap,
                              self_mass=(float(post[model.truth_index(rd)]) if rd.label != m.label or rd.group != m.group else None))
    return cells, reports


def gain(rows, a, b, where=None, key="ls", tag=""):
    from . import paired
    ra = [r for r in rows if r["route"] == a and (where is None or where(r))]
    rb = [r for r in rows if r["route"] == b and (where is None or where(r))]
    A_ = C.by_unit(ra, key)
    B_ = C.by_unit(rb, key)
    return C.paired_hboot(A_, B_, np.random.default_rng(C.seed("gain|" + tag)), 500)


def matching_summary(reports: list) -> dict:
    out = {}
    for k in ("equal_local", "generic_local"):
        gaps = [r[k]["residual_divergence_gap"] for r in reports if r.get(k)]
        out[k] = {"mean_residual_divergence_gap": float(np.mean(gaps)) if gaps else None, "max": float(np.max(gaps)) if gaps else None}
    ent = {}
    for r in reports:
        for k, val in r["entropy_by_route"].items():
            ent.setdefault(k, []).append(val - r["self_entropy"])
    out["entropy_gap_by_route"] = {k: float(np.max(np.abs(v))) for k, v in ent.items()}
    div = {}
    for r in reports:
        for k, val in r["expected_divergence_by_route"].items():
            div.setdefault(k, []).append(val - r["self_expected_divergence"])
    out["expected_divergence_gap_by_route"] = {k: float(np.mean(v)) for k, v in div.items()}
    return out


# --------------------------------------------------------------------------- #
# C03 — within-common vs all-family.
# --------------------------------------------------------------------------- #
def unit_C03(ctx):
    H = harness(ctx, anti=False)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    edges = bins_for(H)
    for rd in H["readers"]:
        model_all = X.reader_model(world, rd, families=None)
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        pri = {"within_common": P.population_prior(model_all, H["makers"], family=rd.family),
               "all_family": P.all_family_prior(model_all, H["makers"])}
        for m in fam_makers:
            L = model_all.loglik(H["streams"][m.id], CH)
            d = C.js(H["selfs"][rd.id]["w_hat"], m.w)
            b = sim_bin(d, edges[rd.id])
            for route, prior in pri.items():
                for n in DOSES:
                    post = posterior_at(model_all, prior, L, n)
                    sc = X.score_rows(model_all, post, m)
                    cells.add({"route": route, "dose": n}, ls=sc["ls"], top1=sc["top1"], conf=sc["conf"], level=b == "near")
    return {"rows": cells.rows()}


def reduce_C03(card, units, ctx):
    v = start(card, ctx, "A prior restricted to the maker's common substrate saves evidence over a broad all-family "
              "prior at the same likelihood and compute; the saving is the value of common structure.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by_dose = {str(n): gain(rows, "within_common", "all_family", lambda r, n=n: r["dose"] == n, tag=f"C03{n}") for n in DOSES}
    g1 = by_dose["1"]["mean"]
    means = {rt: {str(n): mean_of(rows, "ls", lambda r, rt=rt, n=n: r["route"] == rt and r["dose"] == n) for n in DOSES} for rt in ("within_common", "all_family")}
    target = means["within_common"]["1"]
    saving = next((n for n in DOSES if means["all_family"][str(n)] >= target), None)
    conf_c = mean_of(rows, "conf", lambda r: r["route"] == "within_common" and r["dose"] == 1)
    top_c = mean_of(rows, "top1", lambda r: r["route"] == "within_common" and r["dose"] == 1)
    conf_a = mean_of(rows, "conf", lambda r: r["route"] == "all_family" and r["dose"] == 1)
    top_a = mean_of(rows, "top1", lambda r: r["route"] == "all_family" and r["dose"] == 1)
    passed = bool(g1 >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": means["within_common"]["16"] - means["within_common"]["1"], "min": 0.1, "name": "evidence_moves_the_posterior"},
            placebo={"observed": abs(means["within_common"]["16"] - means["all_family"]["16"]), "tol": 0.15, "name": "priors_converge_at_sixteen"},
            positive={"observed": top_c, "expected": 1.0, "tol": 0.6, "name": "within_common_above_floor"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_likelihood_same_compute", "detail": "both routes share the likelihood and the compute by construction"},
            oracle={"observed": means["within_common"]["16"] - np.log(1 / 40), "min": 0.5, "name": "identifiable_with_evidence"},
            prediction={"gain": g1, "min": 0.0, "name": "first_artifact_gain"},
            calibration={"observed": abs(conf_c - top_c), "reference": abs(conf_a - top_a), "direction": "down", "tol": 0.05, "name": "confidence_tracks_accuracy"})
    criterion(v, "C03", passed, gain_at_1=g1, evidence_savings_artifacts=saving)
    v["results"].update({"gain_by_dose": by_dose, "mean_log_score_by_route_and_dose": means, "dose_at_which_all_family_matches": saving})
    receipt(v, rows, card, ctx)
    narrative(v, f"Knowing the maker's common substrate was worth {g1:+.2f} nats on its profile after one artifact against a prior spread over "
                 f"every family; the broad prior needed {saving if saving else 'more than sixteen'} artifacts to reach what the substrate prior had at one.",
              "Common structure buys evidence in this construction; the question of trunk C is what self adds on top of it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C04 — self vs equally local non-self.
# --------------------------------------------------------------------------- #
ROUTES_C04 = ("self", "equal_local", "generic_local", "permuted_self", "random_local", "oracle", "within_common")


def unit_C04(ctx):
    H = harness(ctx)
    cells, reports = route_cells(ctx, H, ROUTES_C04, tag="C04")
    return {"rows": cells.rows(), "matching": matching_summary(reports)}


def reduce_C04(card, units, ctx):
    v = start(card, ctx, "A measured self prior beats an equally local non-self prior only if it carries information "
              "beyond locality; if the two tie, self is one cheap way to be local, not a privileged route.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {}
    for b in ("near", "mid", "far", "anti"):
        surf[b] = {str(n): {"self_minus_equal_local": gain(rows, "self", "equal_local", lambda r, b=b, n=n: r["sim_bin"] == b and r["dose"] == n, tag=f"C04e{b}{n}"),
                            "self_minus_generic_local": gain(rows, "self", "generic_local", lambda r, b=b, n=n: r["sim_bin"] == b and r["dose"] == n, tag=f"C04g{b}{n}"),
                            "self_minus_permuted": gain(rows, "self", "permuted_self", lambda r, b=b, n=n: r["sim_bin"] == b and r["dose"] == n, tag=f"C04p{b}{n}")}
                   for n in DOSES}
    near = surf["near"]["1"]["self_minus_equal_local"]["mean"]
    far = surf["far"]["1"]["self_minus_equal_local"]["mean"]
    anti = surf["anti"]["1"]["self_minus_equal_local"]["mean"] if surf["anti"]["1"]["self_minus_equal_local"]["n_units"] else float("nan")
    pooled = gain(rows, "self", "equal_local", lambda r: r["dose"] == 1, tag="C04pool")
    hidden_near = gain(rows, "self", "equal_local", lambda r: r["sim_bin"] == "near" and r["dose"] == 1, key="hidden", tag="C04h")
    tie = bool(all(abs(surf[b]["1"]["self_minus_equal_local"]["mean"]) < 0.05 for b in ("near", "mid", "far") if surf[b]["1"]["self_minus_equal_local"]["n_units"]))
    privilege = bool(near >= 0.05 and far <= 0.0)
    matching = [u["matching"] for u in units]
    resid = float(np.nanmean([m["equal_local"]["mean_residual_divergence_gap"] for m in matching if m["equal_local"]["mean_residual_divergence_gap"] is not None]))
    gen_resid = float(np.nanmean([m["generic_local"]["mean_residual_divergence_gap"] or 0 for m in matching]))
    near_gen = surf["near"]["1"]["self_minus_generic_local"]["mean"]
    slope = (near - near_gen) / (resid - gen_resid) if (resid == resid and abs(resid - gen_resid) > 1e-6) else float("nan")
    bound = abs(slope * resid) if slope == slope else float("nan")
    orc = mean_of(rows, "top1", lambda r: r["route"] == "oracle" and r["dose"] == 1)
    orc_ls = mean_of(rows, "ls", lambda r: r["route"] == "oracle" and r["dose"] == 1)
    self_ls = mean_of(rows, "ls", lambda r: r["route"] == "self" and r["dose"] == 1)
    perm_vs_rand = abs(mean_of(rows, "ls", lambda r: r["route"] == "permuted_self" and r["dose"] == 1) - mean_of(rows, "ls", lambda r: r["route"] == "random_local" and r["dose"] == 1))
    ece_self = abs(mean_of(rows, "conf", lambda r: r["route"] == "self" and r["dose"] == 1 and r["sim_bin"] == "near") - mean_of(rows, "top1", lambda r: r["route"] == "self" and r["dose"] == 1 and r["sim_bin"] == "near"))
    ece_eq = abs(mean_of(rows, "conf", lambda r: r["route"] == "equal_local" and r["dose"] == 1 and r["sim_bin"] == "near") - mean_of(rows, "top1", lambda r: r["route"] == "equal_local" and r["dose"] == 1 and r["sim_bin"] == "near"))
    gr = G.GateReport()
    battery(gr, live={"observed": orc_ls - self_ls, "min": 0.05, "name": "oracle_beats_self_so_the_prior_matters"},
            placebo={"observed": perm_vs_rand, "tol": 0.5, "name": "content_free_priors_agree", "detail": "permuted self and random local carry no correspondence; they must score alike up to their shape"},
            positive={"observed": orc, "expected": 1.0, "tol": 0.15, "name": "oracle_is_the_ceiling"},
            surface={"accuracy": min(abs(resid) if resid == resid else 0.0, abs(bound) if bound == bound else 0.0), "chance": 0.0, "tol": 0.25, "name": "residual_or_its_bound_within_tolerance", "detail": "the equal-local control's expected divergence gap to the self prior, or the gain that gap could manufacture (slope of gain on divergence times the gap), whichever is smaller"},
            oracle={"observed": orc_ls - np.log(1 / 40), "min": 1.0, "name": "identifiable_with_correct_variables"},
            prediction={"gain": hidden_near["mean"], "min": -0.5, "name": "near_bin_hidden_goal_reported"},
            calibration={"observed": ece_self, "reference": ece_eq, "direction": "down", "tol": 0.10, "name": "self_route_no_worse_calibrated_near"})
    criterion(v, "C04", privilege, near_gain=near, far_gain=far, anti_gain=anti, tie=tie, pooled_gain=pooled["mean"])
    v["results"].update({"conditional_surface": surf, "pooled_after_conditional": pooled, "hidden_goal_near": hidden_near,
                         "oracle_top1": orc, "reading": "self_privilege" if privilege else ("locality" if tie else "mixed")})
    v["matching_residuals"] = {"per_unit": matching, "mean_equal_local_residual": resid, "mean_generic_local_residual": gen_resid,
                               "sensitivity_slope_nats_per_nat_of_residual": slope, "bound_on_near_gain_from_residual": bound if bound == bound else None}
    receipt(v, rows, card, ctx)
    narrative(v, f"Against a non-self prior matched on entropy and on expected distance to the truth, the self prior scored {near:+.2f} nats "
                 f"on the nearest makers, {far:+.2f} on the farthest and {anti:+.2f} on planted anti-similar makers after one artifact; "
                 f"pooled over everything the difference was {pooled['mean']:+.2f}.",
              ("Self carries something beyond locality in this construction." if privilege else
               "Self is one cheap way to be local, not a privileged route: an equally local non-self prior does what self does." if tie else
               "Neither privilege nor a clean tie: the conditional surface, not a headline, is the result."),
              rival="locality: any prior as close to the truth as self does as well")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(privilege or tie))


# --------------------------------------------------------------------------- #
# C05 — self vs within-common by reader type.
# --------------------------------------------------------------------------- #
def unit_C05(ctx):
    H = harness(ctx)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    # reader type from typicality: distance of the reader's profile from its family's population mean
    for rd in H["readers"]:
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        mean_w = np.mean([m.w for m in fam_makers], axis=0)
        d_typ = C.js(rd.w, mean_w)
        ds = sorted(C.js(m.w, mean_w) for m in fam_makers)
        rtype = "near" if d_typ < ds[len(ds) // 4] else ("typical" if d_typ < ds[len(ds) // 2] else "atypical")
        rr = C.rng_for(ctx["lane"], "C05", ctx["wid"], ctx["rep"], rd.id)
        base, _ = reader_priors(H, rd, rr)
        model = H["models"][rd.id]
        for m, is_anti in [(m, a) for r2, m, a in pair_iter(H) if r2.id == rd.id]:
            if is_anti:
                continue
            priors = target_priors(H, rd, m, base)
            L = H["L"][(rd.id, m.id)]
            for n in DOSES:
                for route in ("self", "within_common", "anti_similar"):
                    post = posterior_at(model, priors[route], L, n)
                    sc = X.score_rows(model, post, m)
                    t = rtype if route != "anti_similar" else "anti"
                    cells.add({"reader_type": t, "dose": n, "route": "self" if route == "anti_similar" else route}, ls=sc["ls"], top1=sc["top1"], conf=sc["conf"])
                if rtype != "anti":
                    post = posterior_at(model, priors["within_common"], L, n)
                    sc = X.score_rows(model, post, m)
                    cells.add({"reader_type": "anti", "dose": n, "route": "within_common"}, ls=sc["ls"], top1=sc["top1"], conf=sc["conf"])
    return {"rows": cells.rows()}


def reduce_C05(card, units, ctx):
    v = start(card, ctx, "One personal sample improves on the common-substrate population prior when the reader is typical "
              "of its substrate and worsens it when the reader is atypical or anti-similar.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {t: {str(n): gain(rows, "self", "within_common", lambda r, t=t, n=n: r["reader_type"] == t and r["dose"] == n, tag=f"C05{t}{n}") for n in DOSES}
            for t in ("near", "typical", "atypical", "anti")}
    near, anti = surf["near"]["1"]["mean"], surf["anti"]["1"]["mean"]
    passed = bool(near >= 0.05 and anti <= 0.0)
    conv = abs(surf["near"]["16"]["mean"])
    gr = G.GateReport()
    battery(gr, live={"observed": near - anti if near == near and anti == anti else 0.0, "min": 0.05, "name": "reader_type_moves_the_gain"},
            placebo={"observed": conv, "tol": 0.15, "name": "routes_converge_at_sixteen"},
            positive={"observed": float(anti <= 0.0) if anti == anti else 1.0, "expected": 1.0, "tol": 0.0, "name": "anti_similar_self_does_not_gain"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_likelihood", "detail": "routes share likelihood and compute"},
            oracle={"observed": mean_of(rows, "top1", lambda r: r["route"] == "within_common" and r["dose"] == 16), "min": 0.5, "name": "identifiable_with_evidence"},
            prediction={"gain": near, "min": -0.5, "name": "near_reader_gain_reported"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["route"] == "self" and r["reader_type"] == "anti" and r["dose"] == 1),
                         "reference": mean_of(rows, "conf", lambda r: r["route"] == "self" and r["reader_type"] == "near" and r["dose"] == 1),
                         "direction": "down", "tol": 0.10, "name": "anti_similar_reader_not_more_confident"})
    criterion(v, "C05", passed, near_reader_gain=near, anti_reader_gain=anti)
    v["results"].update({"conditional_surface": surf})
    receipt(v, rows, card, ctx)
    narrative(v, f"For readers typical of their substrate the self prior beat the common population prior by {near:+.2f} nats after one artifact; "
                 f"for anti-similar readers it lost {anti:+.2f}.",
              "A personal sample is worth more than the population only to readers who are typical of it; the sign flips with typicality.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C06 — typicality law and the optimal common mixture.
# --------------------------------------------------------------------------- #
def unit_C06(ctx):
    H = harness(ctx, anti=False)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    from scipy.stats import spearmanr
    typs, gains = [], []
    for rd in H["readers"]:
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        mean_w = np.mean([m.w for m in fam_makers], axis=0)
        typ = 1.0 - C.js(rd.w, mean_w)
        rr = C.rng_for(ctx["lane"], "C06", ctx["wid"], ctx["rep"], rd.id)
        base, _ = reader_priors(H, rd, rr)
        model = H["models"][rd.id]
        half = len(fam_makers) // 2
        train, test = fam_makers[:half], fam_makers[half:]
        # optimal mixture weight fitted on the training half
        best_a, best_s = 0.0, -np.inf
        for a in np.linspace(0, 1, 11):
            s = 0.0
            for m in train:
                pri = target_priors(H, rd, m, base)
                mix = PJ.mixed_prior(pri["self"], pri["within_common"], a)
                s += C.log_score(posterior_at(model, mix, H["L"][(rd.id, m.id)], 1), model.truth_index(m))
            if s > best_s:
                best_a, best_s = a, s
        g = []
        for m in test:
            pri = target_priors(H, rd, m, base)
            L = H["L"][(rd.id, m.id)]
            ls_self = C.log_score(posterior_at(model, pri["self"], L, 1), model.truth_index(m))
            ls_mix = C.log_score(posterior_at(model, PJ.mixed_prior(pri["self"], pri["within_common"], best_a), L, 1), model.truth_index(m))
            ls_pop = C.log_score(posterior_at(model, pri["within_common"], L, 1), model.truth_index(m))
            g.append(ls_self - ls_pop)
            tb = "low" if typ < 0.85 else ("mid" if typ < 0.93 else "high")
            cells.add({"typicality_bin": tb, "route": "self"}, ls=ls_self, gain=ls_self - ls_pop, alpha=best_a, typ=typ)
            cells.add({"typicality_bin": tb, "route": "optimal_mix"}, ls=ls_mix, gain=ls_mix - ls_pop, alpha=best_a, typ=typ)
        typs.append(typ)
        gains.append(float(np.mean(g)))
    rho = float(spearmanr(typs, gains).statistic) if len(typs) > 2 and np.std(typs) > 1e-9 else 0.0
    return {"rows": cells.rows(), "rho": rho}


def reduce_C06(card, units, ctx):
    v = start(card, ctx, "The value of projecting oneself rises with the reader's typicality, and an optimally weighted "
              "mixture of self and common prior bounds what any fixed self weight can do.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    rho = float(np.nanmean([u["rho"] for u in units]))
    by = {tb: {rt: boot(rows, "gain", lambda r, tb=tb, rt=rt: r["typicality_bin"] == tb and r["route"] == rt, seed_tag=f"C06{tb}{rt}") for rt in ("self", "optimal_mix")}
          for tb in ("low", "mid", "high")}
    alpha = mean_of(rows, "alpha")
    passed = bool(rho >= 0.3)
    mix_gain = mean_of(rows, "gain", lambda r: r["route"] == "optimal_mix") - mean_of(rows, "gain", lambda r: r["route"] == "self")
    gr = G.GateReport()
    battery(gr, live={"observed": abs(by["high"]["self"]["mean"] - by["low"]["self"]["mean"]) if by["high"]["self"]["n_units"] and by["low"]["self"]["n_units"] else 0.0, "min": 0.0, "name": "typicality_bins_reported"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "mixture_weight_fitted_on_training_half_only"},
            positive={"observed": float(mix_gain >= -0.02), "expected": 1.0, "tol": 0.0, "name": "optimal_mixture_no_worse_than_self"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_likelihood"},
            oracle={"observed": abs(alpha - 0.5), "min": 0.0, "name": "fitted_weight_reported"},
            prediction={"gain": mix_gain, "min": -0.5, "name": "held_out_half_gain"},
            calibration={"observed": rho, "reference": 0.0, "direction": "up", "tol": 1.0, "name": "spearman_reported"})
    criterion(v, "C06", passed, spearman_gain_vs_typicality=rho, optimal_alpha=alpha)
    v["results"].update({"gain_by_typicality_and_route": by, "spearman": rho, "fitted_self_weight": alpha, "mixture_minus_self": mix_gain})
    receipt(v, rows, card, ctx)
    narrative(v, f"The self gain over the common prior rose with reader typicality (rank correlation {rho:+.2f}); the best fixed mixture put "
                 f"{alpha:.0%} of its weight on self and beat pure self by {mix_gain:+.2f} nats on held-out makers.",
              "Typicality governs projection's value; the common prior is not replaced by self but weighted against it.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C07 — group matching: true versus claimed.
# --------------------------------------------------------------------------- #
def unit_C07(ctx):
    H = harness(ctx, anti=False)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    for rd in H["readers"]:
        model = H["models"][rd.id]
        fam = world.family(rd.family)
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        rr = C.rng_for(ctx["lane"], "C07", ctx["wid"], ctx["rep"], rd.id)
        for m in fam_makers:
            true_match = int(m.group == rd.group)
            claimed_match = int(rr.random() < 0.5)
            claimed_group = rd.group if claimed_match else (rd.group + 1 + int(rr.integers(len(fam.groups) - 1))) % len(fam.groups)
            L = H["L"][(rd.id, m.id)]
            pri_claim = P.population_prior(model, fam_makers, family=rd.family, group=claimed_group, exclude=m.id)
            pri_true = P.population_prior(model, fam_makers, family=rd.family, group=m.group, exclude=m.id)
            pri_common = P.population_prior(model, fam_makers, family=rd.family, exclude=m.id)
            ti = model.truth_index(m)
            for n in (1, 4):
                cells.add({"true_match": true_match, "claimed_match": claimed_match, "dose": n},
                          gain_claim=C.log_score(posterior_at(model, pri_claim, L, n), ti) - C.log_score(posterior_at(model, pri_common, L, n), ti),
                          gain_true=C.log_score(posterior_at(model, pri_true, L, n), ti) - C.log_score(posterior_at(model, pri_common, L, n), ti))
    return {"rows": cells.rows()}


def reduce_C07(card, units, ctx):
    v = start(card, ctx, "A group prior helps when the maker actually shares the convention and not when it only claims to; "
              "the gain follows the production convention, not the label.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    cell = {}
    for t in (0, 1):
        for c in (0, 1):
            cell[f"true{t}_claimed{c}"] = boot(rows, "gain_claim", lambda r, t=t, c=c: r["true_match"] == t and r["claimed_match"] == c and r["dose"] == 1, seed_tag=f"C07{t}{c}")
    true_gain = boot(rows, "gain_true", lambda r: r["dose"] == 1, seed_tag="C07true")
    # gain under a true convention match (claimed prior applied where the claim is true and the group is the reader's)
    g_true = cell["true1_claimed1"]["mean"]
    g_claim_only = cell["true0_claimed1"]["mean"]
    gap = g_true - g_claim_only
    passed = bool(gap >= 0.05)
    true_claim_gain = mean_of(rows, "gain_true", lambda r: r["true_match"] == 1 and r["claimed_match"] == 1 and r["dose"] == 1)
    gr = G.GateReport()
    battery(gr, live={"observed": true_gain["mean"], "min": 0.02, "name": "true_group_prior_gains"},
            placebo={"observed": abs(g_true - true_claim_gain), "tol": 1e-9, "name": "claimed_prior_equals_true_prior_when_the_claim_is_true"},
            positive={"observed": float(g_true > g_claim_only), "expected": 1.0, "tol": 0.0, "name": "true_match_beats_claimed_only"},
            surface={"accuracy": max(g_claim_only, 0.0), "chance": 0.0, "tol": 0.10, "name": "label_alone_buys_nothing"},
            oracle={"observed": true_gain["mean"], "min": 0.02, "name": "true_group_supplied_gains"},
            prediction={"gain": gap, "min": 0.0, "name": "convention_minus_label"},
            calibration={"observed": cell["true0_claimed1"]["mean"], "reference": 0.0, "direction": "down", "tol": 0.0, "name": "a_false_claim_is_a_cost_not_a_prior"})
    criterion(v, "C07", passed, true_match_gain=g_true, claimed_only_gain=g_claim_only, gap=gap)
    v["results"].update({"cells": cell, "true_group_prior_gain": true_gain})
    receipt(v, rows, card, ctx)
    narrative(v, f"Using the maker's claimed group as a prior was worth {g_true:+.2f} nats when the claim was true and the convention shared, "
                 f"and {g_claim_only:+.2f} when only the claim matched.",
              "Group matching pays through the shared convention; a label without the convention is a cost, not a prior.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C08 — expertise matching.
# --------------------------------------------------------------------------- #
def unit_C08(ctx):
    world = world_for(ctx)
    sz = sizes(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "pop")
    makers = population(world, sz["makers"], r)
    streams = {m.id: stream(world, m, 0, C.rng_for(ctx["lane"], "C08", ctx["wid"], ctx["rep"], m.id), 8) for m in makers}
    for fid in range(world.n_families):
        fam = world.family(fid)
        base_rd = make_maker(world, "base", C.rng_for(ctx["lane"], "C08", ctx["wid"], ctx["rep"], f"rd|{fid}"), family=fid, k=0.05)
        for kind in ("practitioner", "familiar", "novice"):
            rd = _reader_kinds(world, fid, 0, C.rng_for(ctx["lane"], "C08", ctx["wid"], ctx["rep"], f"rd|{fid}|{kind}"), kind)
            rd.w, rd.label, rd.group, rd.habit = base_rd.w, base_rd.label, base_rd.group, base_rd.habit
            model = X.reader_model(world, rd, families=[fid])
            prior = X.uniform_prior(model)
            for m in [x for x in makers if x.family == fid]:
                arts = streams[m.id]
                post = model.posterior(prior, arts[:4], ("surface",))
                sc = X.score_rows(model, post, m)
                # method target: predict the maker's method given its goal under the reader's own preference
                mp = model.method_prefs[fid]
                ls_m = float(np.mean([np.log(max(mp[a["goal"], a["method"]], 1e-12)) for a in arts[4:] if a.get("method") is not None and a["goal"] >= 0]))
                cells.add({"reader_kind": kind, "target": "profile"}, ls=sc["ls_profile"], top1=sc["top1"], conf=sc["conf"], prof_sim=C.js(rd.w, m.w))
                cells.add({"reader_kind": kind, "target": "method"}, ls=ls_m, top1=sc["top1"], conf=sc["conf"], prof_sim=C.js(rd.w, m.w))
    return {"rows": cells.rows()}


def reduce_C08(card, units, ctx):
    v = start(card, ctx, "Practitioners, who share the maker's transition model, read the maker's methods better than "
              "observationally familiar readers and novices, separately from any similarity of preference.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {t: {k: boot(rows, "ls", lambda r, t=t, k=k: r["target"] == t and r["reader_kind"] == k, seed_tag=f"C08{t}{k}") for k in ("practitioner", "familiar", "novice")} for t in ("profile", "method")}
    gap_m = by["method"]["practitioner"]["mean"] - by["method"]["novice"]["mean"]
    gap_p = by["profile"]["practitioner"]["mean"] - by["profile"]["novice"]["mean"]
    fam_gap = by["method"]["familiar"]["mean"] - by["method"]["novice"]["mean"]
    passed = bool(gap_m >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": gap_m, "min": 0.02, "name": "expertise_moves_method_reading"},
            placebo={"observed": abs(mean_of(rows, "prof_sim", lambda r: r["reader_kind"] == "practitioner") - mean_of(rows, "prof_sim", lambda r: r["reader_kind"] == "novice")), "tol": 0.05, "name": "preference_similarity_matched_across_kinds"},
            positive={"observed": float(by["method"]["practitioner"]["mean"] >= by["method"]["familiar"]["mean"] - 0.02), "expected": 1.0, "tol": 0.0, "name": "practitioner_no_worse_than_familiar_on_methods"},
            surface={"accuracy": fam_gap, "chance": 0.0, "tol": 0.10, "name": "template_familiarity_alone_does_not_read_methods"},
            oracle={"observed": by["method"]["practitioner"]["mean"] - np.log(0.5), "min": 0.02, "name": "practitioner_above_uniform_methods"},
            prediction={"gain": gap_m, "min": 0.0, "name": "held_out_method_prediction"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["reader_kind"] == "novice" and r["target"] == "profile"),
                         "reference": mean_of(rows, "conf", lambda r: r["reader_kind"] == "practitioner" and r["target"] == "profile"), "direction": "down", "tol": 0.10, "name": "novice_not_more_confident"})
    criterion(v, "C08", passed, practitioner_minus_novice_method=gap_m, practitioner_minus_novice_profile=gap_p, familiar_minus_novice_method=fam_gap)
    v["results"].update({"by_target_and_kind": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Practitioners predicted makers' method choices {gap_m:+.2f} nats better than novices and read profiles {gap_p:+.2f} better; "
                 f"readers who knew the templates but not the transitions gained {fam_gap:+.2f} on methods.",
              "Transition-model access is a separable component of similarity: it reads mechanics, and it is not preference similarity.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C09 — individual history.
# --------------------------------------------------------------------------- #
def unit_C09(ctx):
    H = harness(ctx, anti=False, n_art=14)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    from ..exact import prefix_continuation
    for rd in H["readers"]:
        model = H["models"][rd.id]
        fam_makers = [m for m in H["makers"] if m.family == rd.family]
        by_label = {}
        for m in fam_makers:
            by_label.setdefault((m.group, m.label), []).append(m)
        for m in fam_makers:
            arts = H["streams"][m.id]
            twins = [x for x in by_label[(m.group, m.label)] if x.id != m.id] or [x for x in fam_makers if x.group == m.group and x.id != m.id]
            pri_group = P.population_prior(model, fam_makers, family=rd.family, group=m.group, exclude=m.id)
            for hist in (0, 2, 8):
                for control in ("real", "relabelled"):
                    src = arts if control == "real" else (H["streams"][twins[0].id] if twins else arts)
                    prior = model.posterior(pri_group, src[:hist], CH) if hist else pri_group
                    # the current goals are fresh: score the hidden goals of artifacts 10..13 and their continuation
                    hg = float(np.mean([hidden_goal_ls(model, prior, rd.family, a) or 0.0 for a in arts[10:14]]))
                    cont = float(np.mean([prefix_continuation(model, prior, a, 2) for a in arts[10:14]]))
                    cells.add({"history": hist, "control": control}, ls=hg, cont=cont, top1=float(int(np.argmax(prior)) == model.truth_index(m)))
    return {"rows": cells.rows()}


def reduce_C09(card, units, ctx):
    v = start(card, ctx, "A target's own earlier history improves held-out continuation beyond group and expertise, and "
              "the improvement is target-specific rather than a memorised source identity.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {c: {str(h): boot(rows, "ls", lambda r, c=c, h=h: r["control"] == c and r["history"] == h, seed_tag=f"C09{c}{h}") for h in (0, 2, 8)} for c in ("real", "relabelled")}
    gain8 = by["real"]["8"]["mean"] - by["real"]["0"]["mean"]
    shortcut = by["relabelled"]["8"]["mean"] - by["relabelled"]["0"]["mean"]
    passed = bool(gain8 >= 0.05 and (gain8 - shortcut) >= -0.02 and shortcut <= gain8 + 0.02)
    gr = G.GateReport()
    battery(gr, live={"observed": gain8, "min": 0.02, "name": "history_moves_continuation"},
            placebo={"observed": abs(by["real"]["0"]["mean"] - by["relabelled"]["0"]["mean"]), "tol": 0.05, "name": "no_history_no_difference"},
            positive={"observed": float(by["real"]["8"]["mean"] >= by["real"]["2"]["mean"] - 0.02), "expected": 1.0, "tol": 0.0, "name": "more_history_no_worse"},
            surface={"accuracy": shortcut, "chance": 0.0, "tol": max(0.10, gain8), "name": "same_label_history_is_not_the_target", "detail": "a same-label twin's history may carry label-level information but not the target's own"},
            oracle={"observed": gain8, "min": 0.02, "name": "target_history_supplied"},
            prediction={"gain": gain8, "min": 0.0, "name": "held_out_continuation"},
            calibration={"observed": mean_of(rows, "top1", lambda r: r["control"] == "real" and r["history"] == 8), "reference": mean_of(rows, "top1", lambda r: r["control"] == "real" and r["history"] == 0), "direction": "up", "tol": 0.0, "name": "history_raises_top1"})
    criterion(v, "C09", passed, gain_at_8=gain8, relabelled_gain_at_8=shortcut)
    v["results"].update({"by_control_and_history": by, "target_specific_share": (gain8 - shortcut) / gain8 if gain8 else None})
    receipt(v, rows, card, ctx)
    narrative(v, f"Eight artifacts of the target's own history improved continuation of its fresh work by {gain8:+.2f} nats; the same amount of "
                 f"a same-label twin's history gave {shortcut:+.2f}, so {((gain8 - shortcut) / gain8 if gain8 else 0):.0%} of the gain was the target's own.",
              "Individual history adds something after group and expertise, and most of it is the individual.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C10 — axis x target map.
# --------------------------------------------------------------------------- #
AXES = ("observation", "transition", "profile", "policy", "cost_model", "surface")
TARGETS = ("goal", "profile", "method", "cost_weights")


def unit_C10(ctx):
    H = harness(ctx, anti=False, n_art=6)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "cost")
    # cost weights: a persistent per-maker tradeoff vector
    for m in H["makers"] + H["readers"]:
        m.cost = {"weights": r.dirichlet(np.ones(4) * (2.0 + m.group))}
    Xs, Ys = [], {t: [] for t in TARGETS}
    for rd in H["readers"]:
        model = H["models"][rd.id]
        rr = C.rng_for(ctx["lane"], "C10", ctx["wid"], ctx["rep"], rd.id)
        base, _ = reader_priors(H, rd, rr)
        fam = world.family(rd.family)
        e_r = np.concatenate([a["features"] for a in stream(world, rd, 0, rr, 3)])
        for m in [m for m in H["makers"] if m.family == rd.family]:
            pri = target_priors(H, rd, m, base)
            L = H["L"][(rd.id, m.id)]
            arts = H["streams"][m.id]
            post_s = posterior_at(model, pri["self"], L, 1)
            post_c = posterior_at(model, pri["within_common"], L, 1)
            ax = {"observation": float(np.mean([C.js(rd.template[g, j], m.template[g, j]) for g in range(fam.ng) for j in range(N_METHODS)])),
                  "transition": float(np.abs(rd.method_pref - m.method_pref).mean()),
                  "profile": C.js(rd.w, m.w),
                  "policy": C.js(histogram(e_r, fam.nf), histogram(np.concatenate([a["features"] for a in arts[:3]]), fam.nf)),
                  "cost_model": float(np.abs(rd.cost["weights"] - m.cost["weights"]).sum()),
                  "surface": float(abs(len(e_r) / 3 - len(arts[0]["features"])) / max(len(arts[0]["features"]), 1))}
            Xs.append([ax[a] for a in AXES])
            Ys["goal"].append((hidden_goal_ls(model, post_s, rd.family, arts[1]) or 0.0) - (hidden_goal_ls(model, post_c, rd.family, arts[1]) or 0.0))
            Ys["profile"].append(C.log_score(post_s, model.truth_index(m)) - C.log_score(post_c, model.truth_index(m)))
            mp_r = rd.method_pref
            Ys["method"].append(float(np.mean([np.log(max(mp_r[a["goal"], a["method"]], 1e-12)) - np.log(0.5) for a in arts[1:4] if a.get("method") is not None and a["goal"] >= 0])))
            Ys["cost_weights"].append(-float(np.abs(rd.cost["weights"] - m.cost["weights"]).sum()) + 1.0)
    Xm = np.array(Xs)
    Xd = np.column_stack([np.ones(len(Xm)), Xm])

    def r2(y, cols):
        beta, *_ = np.linalg.lstsq(Xd[:, cols], y, rcond=None)
        res = y - Xd[:, cols] @ beta
        return 1 - res.var() / max(y.var(), 1e-12)
    for t in TARGETS:
        y = np.array(Ys[t])
        if y.var() < 1e-12:
            for a in AXES:
                cells.add({"axis": a, "target": t}, partial_r2=0.0)
            continue
        full = r2(y, list(range(Xd.shape[1])))
        for i, a in enumerate(AXES):
            cells.add({"axis": a, "target": t}, partial_r2=float(full - r2(y, [0] + [j + 1 for j in range(len(AXES)) if j != i])), full_r2=float(full))
    return {"rows": cells.rows()}


def reduce_C10(card, units, ctx):
    v = start(card, ctx, "Different similarity axes matter to different inference targets: no universal similarity scalar "
              "exists until the axis-by-target map is reported.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {t: {a: boot(rows, "partial_r2", lambda r, a=a, t=t: r["axis"] == a and r["target"] == t, seed_tag=f"C10{a}{t}")["mean"] for a in AXES} for t in TARGETS}
    best = {t: max(grid[t], key=grid[t].get) for t in TARGETS}
    passed = bool(all(max(grid[t].values()) >= 0.05 for t in TARGETS))
    distinct = len(set(best.values()))
    gr = G.GateReport()
    battery(gr, live={"observed": max(max(g.values()) for g in grid.values()), "min": 0.05, "name": "some_axis_explains_some_target"},
            placebo={"observed": grid["profile"]["surface"], "tol": 0.10, "name": "surface_length_explains_no_profile_gain"},
            positive={"observed": grid["profile"]["profile"], "expected": max(grid["profile"].values()), "tol": 0.10, "name": "profile_axis_leads_profile_target"},
            surface={"accuracy": max(grid[t]["surface"] for t in TARGETS), "chance": 0.0, "tol": 0.10, "name": "surface_axis_weak_everywhere"},
            oracle={"observed": grid["cost_weights"]["cost_model"], "min": 0.05, "name": "cost_axis_explains_cost_target"},
            prediction={"gain": float(distinct), "min": 2.0, "name": "at_least_two_distinct_leading_axes"},
            calibration={"observed": float(distinct), "reference": 1.0, "direction": "up", "tol": 0.0, "name": "no_universal_scalar"})
    criterion(v, "C10", passed, leading_axis_by_target=best, distinct_leaders=distinct)
    v["results"].update({"partial_r2": grid, "leading_axis_by_target": best})
    receipt(v, rows, card, ctx)
    narrative(v, "The axis that best explained each target's self-route gain was " + ", ".join(f"{t}: {a}" for t, a in best.items()) + ".",
              "Similarity is target-specific in this construction: the map replaces the scalar.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C11 — false projection from irrelevant similarity.
# --------------------------------------------------------------------------- #
def unit_C11(ctx):
    world = world_for(ctx)
    sz = sizes(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "pop")
    for i in range(sz["readers"]):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"reader{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "C11", ctx["wid"], ctx["rep"], rd.id))
        sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        for nuisance in (0, 1):
            for conflict in (0, 1):
                for j in range(4):
                    w = C.normalize(1.0 - rd.w + 0.05) if conflict else rd.w
                    m = make_maker(world, f"m{nuisance}{conflict}{j}", r, family=fid, group=rd.group, w=w, k=0.3,
                                   habit_strength=(rd.habit_strength if nuisance else 0.25), attention=(rd.attention if nuisance else "none"))
                    if nuisance:
                        m.habit = {d: rd.habit[d].copy() for d in rd.habit}
                    arts = stream(world, m, 0, C.rng_for(ctx["lane"], "C11", ctx["wid"], ctx["rep"], m.id), 2)
                    post = model.posterior(sp, arts, CH)
                    sc = X.score_rows(model, post, m)
                    self_mass = float(post[model.truth_index(rd)]) if conflict else None
                    cells.add({"nuisance_match": nuisance, "relevant_conflict": conflict}, conf=sc["conf"], top1=sc["top1"], ls=sc["ls"], self_mass=self_mass)
    return {"rows": cells.rows()}


def reduce_C11(card, units, ctx):
    v = start(card, ctx, "High similarity on nuisance dimensions with conflict on the decision-relevant one raises the "
              "reader's confidence and self weight without raising its accuracy, if the reader is vulnerable.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    cell = {f"nuisance{n}_conflict{c}": {k: mean_of(rows, k, lambda r, n=n, c=c: r["nuisance_match"] == n and r["relevant_conflict"] == c) for k in ("conf", "top1", "ls", "self_mass")} for n in (0, 1) for c in (0, 1)}
    conf_rise = cell["nuisance1_conflict1"]["conf"] - cell["nuisance0_conflict1"]["conf"]
    acc_change = cell["nuisance1_conflict1"]["top1"] - cell["nuisance0_conflict1"]["top1"]
    self_rise = (cell["nuisance1_conflict1"]["self_mass"] or 0) - (cell["nuisance0_conflict1"]["self_mass"] or 0)
    vulnerable = bool(conf_rise >= 0.05 and acc_change <= 0.0)
    gr = G.GateReport()
    battery(gr, live={"observed": cell["nuisance1_conflict0"]["conf"] - cell["nuisance0_conflict0"]["conf"] + conf_rise, "min": -1.0, "name": "nuisance_match_effect_reported"},
            placebo={"observed": abs(cell["nuisance1_conflict0"]["top1"] - cell["nuisance0_conflict0"]["top1"]), "tol": 0.5, "name": "nuisance_match_alone_barely_moves_accuracy"},
            positive={"observed": cell["nuisance1_conflict0"]["top1"], "expected": 1.0, "tol": 0.6, "name": "compatible_maker_read"},
            surface={"accuracy": acc_change, "chance": 0.0, "tol": 0.20, "name": "nuisance_match_adds_no_accuracy"},
            oracle={"observed": cell["nuisance0_conflict1"]["ls"] - np.log(1 / 40), "min": 0.0, "name": "conflict_still_readable_without_nuisance"},
            prediction={"gain": self_rise, "min": -1.0, "name": "self_mass_rise_reported"},
            calibration={"observed": conf_rise, "reference": acc_change, "direction": "up", "tol": 1.0, "name": "confidence_vs_accuracy_reported"})
    criterion(v, "C11", vulnerable, confidence_rise=conf_rise, accuracy_change=acc_change, self_mass_rise=self_rise)
    v["results"].update({"cells": cell, "vulnerable": vulnerable})
    receipt(v, rows, card, ctx)
    narrative(v, f"When a maker matched the reader on habit and attention but held an inverted profile, the reader's confidence moved {conf_rise:+.2f} "
                 f"and its accuracy {acc_change:+.2f} relative to the same conflict without the nuisance match; mass on the reader's own profile moved {self_rise:+.2f}.",
              "The exact reader is " + ("vulnerable to false projection from irrelevant similarity in this construction; trunk P tests the correction." if vulnerable else "not fooled by nuisance similarity here: its likelihood does not read habit as profile."))
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit="OPENED")


# --------------------------------------------------------------------------- #
# C12 — anti-similarity as a learned transform.
# --------------------------------------------------------------------------- #
def unit_C12(ctx):
    world = world_for(ctx)
    sz = sizes(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "pop")
    import itertools
    for i in range(sz["readers"]):
        fid = i % world.n_families
        fam = world.family(fid)
        rd = make_maker(world, f"reader{i}", r, family=fid, k=0.05, label=f"peaked_{i % world.family(fid).ng}")
        model = X.reader_model(world, rd, families=[fid])
        sm = P.measure_self(world, rd, model, C.rng_for(ctx["lane"], "C12", ctx["wid"], ctx["rep"], rd.id))
        shift = 1                                                   # the systematic transform: the peak moves one channel up
        perm = np.roll(np.arange(fam.ng), shift)
        makers = [make_maker(world, f"t{j}", r, family=fid, group=rd.group, w=C.normalize(rd.w[perm] + 0.02 * r.dirichlet(np.ones(fam.ng))), k=0.2) for j in range(8)]
        train, test = makers[:4], makers[4:]
        # learn the transform: the rotation that carries the reader's peak onto the training makers' inferred peak
        shifts = []
        for m in train:
            arts = stream(world, m, 0, C.rng_for(ctx["lane"], "C12", ctx["wid"], ctx["rep"], "tr" + m.id), 12)
            q = model.posterior(X.uniform_prior(model), arts, CH)
            shifts.append((int(np.argmax(model.profile_mean(q, fid))) - int(np.argmax(sm["w_hat"]))) % fam.ng)
        best_shift = int(np.bincount(shifts, minlength=fam.ng).argmax())
        best_p = list(np.roll(np.arange(fam.ng), best_shift))
        pri_raw = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
        pri_diff = P.local_prior(model, fid, sm["w_hat"][best_p], sm["group_hat"])
        pri_all = P.population_prior(model, [], family=fid, group=rd.group)
        for m in test:
            arts = stream(world, m, 0, C.rng_for(ctx["lane"], "C12", ctx["wid"], ctx["rep"], "te" + m.id), 17)
            L = model.loglik(arts, CH)
            for n in DOSES:
                for name, pri in (("raw_self", pri_raw), ("difference_from_self", pri_diff), ("all_family", pri_all)):
                    post = posterior_at(model, pri, L, n)
                    sc = X.score_rows(model, post, m)
                    cells.add({"model": name, "dose": n}, ls=sc["ls"], top1=sc["top1"], conf=sc["conf"], learned_correct=float(best_shift == shift))
    return {"rows": cells.rows()}


def reduce_C12(card, units, ctx):
    v = start(card, ctx, "When makers are systematic transforms of the reader, raw self-projection is wrong but a learned "
              "difference-from-self model beats a broad population prior while keeping self and other apart.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {m: {str(n): boot(rows, "ls", lambda r, m=m, n=n: r["model"] == m and r["dose"] == n, seed_tag=f"C12{m}{n}") for n in DOSES} for m in ("raw_self", "difference_from_self", "all_family")}
    g16 = by["difference_from_self"]["16"]["mean"] - by["all_family"]["16"]["mean"]
    g1 = by["difference_from_self"]["1"]["mean"] - by["all_family"]["1"]["mean"]
    raw_bad = by["raw_self"]["1"]["mean"] - by["all_family"]["1"]["mean"]
    learned = mean_of(rows, "learned_correct")
    passed = bool(g16 >= 0.05 or g1 >= 0.05)
    gr = G.GateReport()
    battery(gr, live={"observed": by["difference_from_self"]["1"]["mean"] - by["raw_self"]["1"]["mean"], "min": 0.05, "name": "transform_moves_the_score"},
            placebo={"observed": abs(by["raw_self"]["16"]["mean"] - by["all_family"]["16"]["mean"]), "tol": 1.0, "name": "priors_wash_out_with_evidence"},
            positive={"observed": learned, "expected": 1.0, "tol": 0.5, "name": "transform_learned_from_training_makers"},
            surface={"accuracy": max(raw_bad, 0.0), "chance": 0.0, "tol": 0.10, "name": "raw_self_does_not_gain_over_the_group_prior"},
            oracle={"observed": g1, "min": -1.0, "name": "difference_model_first_artifact_reported"},
            prediction={"gain": g16, "min": 0.0, "name": "held_out_transformed_makers"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["model"] == "raw_self" and r["dose"] == 1), "reference": mean_of(rows, "conf", lambda r: r["model"] == "difference_from_self" and r["dose"] == 1), "direction": "down", "tol": 0.15, "name": "raw_self_not_more_confident_while_wrong"})
    criterion(v, "C12", passed, gain_at_16=g16, gain_at_1=g1, raw_self_gain_at_1=raw_bad, transform_learned=learned)
    v["results"].update({"by_model_and_dose": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Raw self-projection scored {raw_bad:+.2f} nats against the broad prior on makers who were rotations of the reader; the difference-from-self "
                 f"model, having learned the rotation from four training makers ({learned:.0%} correctly), scored {g1:+.2f} at one artifact and {g16:+.2f} at sixteen.",
              "Anti-similarity is usable when it is systematic: the reader keeps its own model and learns the map.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C13 — focus on common structure by causal world.
# --------------------------------------------------------------------------- #
FOCUS_W = {"common": {"common_structure": 3.0, "surface": 0.5, "group_convention": 0.5, "goal_consequences": 0.5},
           "surface": {"common_structure": 0.5, "surface": 3.0, "group_convention": 0.5, "goal_consequences": 0.5},
           "uniform": {"common_structure": 1.0, "surface": 1.0, "group_convention": 1.0, "goal_consequences": 1.0}}
CH4 = ("surface", "common_structure", "group_convention", "goal_consequences")


def _causal_arts(world, m, rng_, causal, n):
    """Arrange which channel carries the goal: common (structure only), group (convention only),
    individual (payoff only), nuisance (surface permuted, others scrambled)."""
    arts = stream(world, m, 0, rng_, n)
    fam = world.family(m.family)
    if fam.link != "draw" or "structure_obs" not in arts[0]:
        return arts
    for a in arts:
        if causal == "common":
            a["features"] = rng_.integers(0, fam.nf, size=len(a["features"]))
            a["payoff_obs"] = int(rng_.integers(fam.ng))
            a["convention_obs"] = rng_.integers(0, fam.nf, size=len(a["convention_obs"])).tolist()
        elif causal == "group":
            a["features"] = rng_.integers(0, fam.nf, size=len(a["features"]))
            a["structure_obs"] = rng_.integers(0, len(fam.blocks) + 1, size=len(a["structure_obs"])).tolist()
            a["payoff_obs"] = int(rng_.integers(fam.ng))
        elif causal == "individual":
            a["features"] = rng_.integers(0, fam.nf, size=len(a["features"]))
            a["structure_obs"] = rng_.integers(0, len(fam.blocks) + 1, size=len(a["structure_obs"])).tolist()
            a["convention_obs"] = rng_.integers(0, fam.nf, size=len(a["convention_obs"])).tolist()
        elif causal == "nuisance":
            a["structure_obs"] = rng_.integers(0, len(fam.blocks) + 1, size=len(a["structure_obs"])).tolist()
            a["payoff_obs"] = int(rng_.integers(fam.ng))
            a["convention_obs"] = rng_.integers(0, fam.nf, size=len(a["convention_obs"])).tolist()
            a["features"] = rng_.integers(0, fam.nf, size=len(a["features"]))
    return arts


def unit_C13(ctx):
    world = world_for(ctx)
    sz = sizes(ctx)
    cells = Cells(ctx["wid"], ctx["rep"])
    r = rng(ctx, "pop")
    diag_w = {"common": "common_structure", "group": "group_convention", "individual": "goal_consequences", "nuisance": None}
    for i in range(sz["readers"]):
        fid = i % world.n_families
        rd = make_maker(world, f"reader{i}", r, family=fid, k=0.05)
        model = X.reader_model(world, rd, families=[fid])
        prior = X.uniform_prior(model)
        makers = [make_maker(world, f"m{j}", r, family=fid, k=0.3) for j in range(6)]
        for causal in ("common", "group", "individual", "nuisance"):
            for m in makers:
                arts = _causal_arts(world, m, C.rng_for(ctx["lane"], "C13", ctx["wid"], ctx["rep"], f"{causal}|{m.id}|{rd.id}"), causal, 5)
                hg_prior = hidden_goal_ls(model, prior, fid, arts[4])
                for focus in ("common", "surface", "uniform", "diagnostic"):
                    w = dict(FOCUS_W["uniform"]) if focus == "diagnostic" else dict(FOCUS_W[focus])
                    if focus == "diagnostic":
                        w = {c: 0.5 for c in CH4}
                        if diag_w[causal] is not None:
                            w[diag_w[causal]] = 3.0
                        else:
                            w = {c: 0.0 for c in CH4}
                    post = model.posterior(prior, arts[:4], CH4, w)
                    hg = hidden_goal_ls(model, post, fid, arts[4])
                    cells.add({"focus": focus, "causal_world": causal}, hidden=hg, gain_vs_prior=(hg - hg_prior) if hg is not None else None,
                              ls=C.log_score(post, model.truth_index(m)), conf=float(post.max()))
    return {"rows": cells.rows()}


def reduce_C13(card, units, ctx):
    v = start(card, ctx, "Focusing attention on common structure helps recover common-shaped goals only when common axes are "
              "causally relevant; maker-diagnostic focus is the ceiling everywhere.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    grid = {cw: {f: boot(rows, "hidden", lambda r, cw=cw, f=f: r["causal_world"] == cw and r["focus"] == f, seed_tag=f"C13{cw}{f}")["mean"] for f in ("common", "surface", "uniform", "diagnostic")} for cw in ("common", "group", "individual", "nuisance")}
    g_common = grid["common"]["common"] - grid["common"]["uniform"]
    g_nuis = grid["nuisance"]["common"] - grid["nuisance"]["uniform"]
    g_group = grid["group"]["common"] - grid["group"]["uniform"]
    passed = bool(g_common >= 0.03 and g_nuis <= 0.03)
    gr = G.GateReport()
    battery(gr, live={"observed": grid["common"]["diagnostic"] - grid["common"]["surface"], "min": 0.02, "name": "focus_moves_goal_recovery"},
            placebo={"observed": max(ci_pos(rows, "gain_vs_prior", lambda r, f=f: r["causal_world"] == "nuisance" and r["focus"] == f, seed_tag="c13" + f) for f in ("common", "surface", "uniform")), "tol": 0.10, "name": "no_focus_gains_over_the_prior_in_the_nuisance_world", "detail": "one-sided: a focus that hurts on scrambled channels is not a gain"},
            positive={"observed": float(all(grid[cw]["diagnostic"] >= grid[cw]["uniform"] - 0.02 for cw in grid)), "expected": 1.0, "tol": 0.0, "name": "diagnostic_focus_is_the_ceiling"},
            surface={"accuracy": grid["nuisance"]["surface"] - grid["nuisance"]["uniform"], "chance": 0.0, "tol": 0.15, "name": "surface_focus_no_gain_in_nuisance_world"},
            oracle={"observed": grid["common"]["diagnostic"] - np.log(1 / 4), "min": 0.0, "name": "goal_recoverable_with_the_right_channel"},
            prediction={"gain": g_common, "min": -1.0, "name": "hidden_goal_common_world"},
            calibration={"observed": mean_of(rows, "conf", lambda r: r["causal_world"] == "nuisance" and r["focus"] == "common"), "reference": mean_of(rows, "conf", lambda r: r["causal_world"] == "nuisance" and r["focus"] == "uniform"), "direction": "down", "tol": 0.25, "name": "nuisance_confidence_reported"})
    criterion(v, "C13", passed, common_world_gain=g_common, group_world_gain=g_group, nuisance_world_gain=g_nuis)
    v["results"].update({"hidden_goal_by_world_and_focus": grid})
    receipt(v, rows, card, ctx)
    narrative(v, f"Weighting common structure improved the hidden-goal score by {g_common:+.2f} nats where common axes carried the goal, {g_group:+.2f} where the "
                 f"group convention carried it, and {g_nuis:+.2f} where nothing but surface differed.",
              "Common-structure focus is conditionally useful; the ceiling is focus on whatever channel is diagnostic of this maker.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C14 — locality predicts a hidden continuation (by bin, before pooling).
# --------------------------------------------------------------------------- #
def unit_C14(ctx):
    H = harness(ctx, n_art=14)
    cells, reports = route_cells(ctx, H, ("self", "equal_local", "generic_local", "within_common"), doses=(1, 4, 12), tag="C14")
    return {"rows": cells.rows(), "matching": matching_summary(reports)}


def reduce_C14(card, units, ctx):
    v = start(card, ctx, "A local prior's gain on a maker's hidden next action is positive near the reader and negative or absent "
              "far from it; a pooled mean would hide the reversal.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {b: {str(n): gain(rows, "self", "equal_local", lambda r, b=b, n=n: r["sim_bin"] == b and r["dose"] == n, key="hidden", tag=f"C14{b}{n}") for n in (1, 4, 12)} for b in ("near", "mid", "far", "anti")}
    near, far = surf["near"]["1"]["mean"], surf["far"]["1"]["mean"]
    pooled = gain(rows, "self", "equal_local", lambda r: r["dose"] == 1, key="hidden", tag="C14pool")
    vs_common = gain(rows, "self", "within_common", lambda r: r["sim_bin"] == "near" and r["dose"] == 1, key="hidden", tag="C14c")
    passed = bool(near >= 0.03 and far <= 0.0)
    unif = np.log(1 / 4)
    gr = G.GateReport()
    battery(gr, live={"observed": mean_of(rows, "hidden", lambda r: r["route"] == "self" and r["dose"] == 12) - unif, "min": 0.05, "name": "continuation_predictable_at_twelve"},
            placebo={"observed": abs(surf["near"]["12"]["mean"]), "tol": 0.15, "name": "routes_converge_at_twelve"},
            positive={"observed": float(near >= far), "expected": 1.0, "tol": 0.0, "name": "near_gain_at_least_far_gain"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_likelihood"},
            oracle={"observed": mean_of(rows, "hidden", lambda r: r["route"] == "within_common" and r["dose"] == 12) - unif, "min": 0.0, "name": "population_route_predicts_at_twelve"},
            prediction={"gain": near, "min": -1.0, "name": "near_bin_hidden_gain"},
            calibration={"observed": abs(pooled["mean"]), "reference": abs(near), "direction": "down", "tol": 0.0, "name": "pooled_smaller_than_conditional"})
    criterion(v, "C14", passed, near_gain=near, far_gain=far, pooled=pooled["mean"], near_vs_common=vs_common["mean"])
    v["results"].update({"conditional_surface": surf, "pooled_after_conditional": pooled, "near_self_minus_within_common": vs_common})
    v["matching_residuals"] = {"per_unit": [u["matching"] for u in units]}
    receipt(v, rows, card, ctx)
    narrative(v, f"On the maker's hidden next goal the self prior beat the equally local non-self prior by {near:+.2f} nats for near makers and {far:+.2f} "
                 f"for far makers; pooled, the difference was {pooled['mean']:+.2f}.",
              "Locality predicts the continuation conditionally; the sign reversal is the finding and the pool is not.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C15 — reader plurality.
# --------------------------------------------------------------------------- #
def unit_C15(ctx):
    H = harness(ctx, anti=False, n_art=4)
    world = H["world"]
    cells = Cells(ctx["wid"], ctx["rep"])
    for fid in range(world.n_families):
        readers = [rd for rd in H["readers"] if rd.family == fid]
        extra = make_maker(world, f"extra{fid}", C.rng_for(ctx["lane"], "C15", ctx["wid"], ctx["rep"], f"x{fid}"), family=fid, k=0.05)
        H["models"][extra.id] = X.reader_model(world, extra, families=[fid])
        H["selfs"][extra.id] = P.measure_self(world, extra, H["models"][extra.id], C.rng_for(ctx["lane"], "C15", ctx["wid"], ctx["rep"], f"s{fid}"))
        for m in [m for m in H["makers"] if m.family == fid]:
            H["L"][(extra.id, m.id)] = H["models"][extra.id].loglik(H["streams"][m.id], H["channels"])
        readers = readers + [extra]
        if len(readers) < 2:
            continue
        makers = [m for m in H["makers"] if m.family == fid]
        fam = world.family(fid)
        for m in makers:
            posts_ind, posts_cor, singles = [], [], []
            for rd in readers:
                model = H["models"][rd.id]
                sm = H["selfs"][rd.id]
                sp = P.local_prior(model, fid, sm["w_hat"], sm["group_hat"])
                L = H["L"][(rd.id, m.id)]
                q = posterior_at(model, sp, L, 2)
                posts_ind.append(q)
                singles.append(C.log_score(q, model.truth_index(m)))
                # correlated: every reader also believes a shared false group note
                false_group = (m.group + 1) % len(fam.groups)
                note = PJ.evidence_loglik(model, "group_label", false_group, 0.9)
                posts_cor.append(C.softmax(np.log(np.maximum(sp, 1e-300)) + note + L[:2].sum(axis=0)))
            ti = H["models"][readers[0].id].truth_index(m)
            L_shared = H["L"][(readers[0].id, m.id)][:2].sum(axis=0)
            for corr, posts in (("independent", posts_ind), ("correlated", posts_cor)):
                for method in ("vote", "bayes"):
                    if method == "bayes":
                        # shared evidence read once: the pooled prior times one likelihood
                        pri = [P.local_prior(H["models"][rd.id], fid, H["selfs"][rd.id]["w_hat"], H["selfs"][rd.id]["group_hat"]) for rd in readers]
                        note = PJ.evidence_loglik(H["models"][readers[0].id], "group_label", (m.group + 1) % len(fam.groups), 0.9) if corr == "correlated" else 0.0
                        q = C.softmax(np.log(np.maximum(PJ.ensemble(pri, "mean"), 1e-300)) + note + L_shared)
                    else:
                        q = PJ.ensemble(posts, method)
                    cells.add({"correlation": corr, "method": method}, ls=C.log_score(q, ti), conf=float(q.max()), top1=float(int(np.argmax(q)) == ti), best_single=max(singles))
                cells.add({"correlation": corr, "method": "single"}, ls=max(singles) if corr == "independent" else max(C.log_score(p, ti) for p in posts), conf=float(np.max([p.max() for p in posts])), top1=float(np.mean([int(np.argmax(p)) == ti for p in posts])), best_single=max(singles), mean_single=float(np.mean(singles)))
    return {"rows": cells.rows()}


def reduce_C15(card, units, ctx):
    v = start(card, ctx, "An ensemble of readers with different self priors corrects projection only when their errors are "
              "diverse; a shared false belief makes agreement look like confirmation.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    by = {c: {m: boot(rows, "ls", lambda r, c=c, m=m: r["correlation"] == c and r["method"] == m, seed_tag=f"C15{c}{m}") for m in ("vote", "bayes", "single")} for c in ("independent", "correlated")}
    g_ind = by["independent"]["bayes"]["mean"] - by["independent"]["single"]["mean"]
    g_cor = by["correlated"]["bayes"]["mean"] - by["correlated"]["single"]["mean"]
    conf_cor = mean_of(rows, "conf", lambda r: r["correlation"] == "correlated" and r["method"] == "bayes")
    acc_cor = mean_of(rows, "top1", lambda r: r["correlation"] == "correlated" and r["method"] == "bayes")
    conf_ind = mean_of(rows, "conf", lambda r: r["correlation"] == "independent" and r["method"] == "bayes")
    acc_ind = mean_of(rows, "top1", lambda r: r["correlation"] == "independent" and r["method"] == "bayes")
    passed = bool(g_ind >= 0.02 and g_cor < g_ind)
    gr = G.GateReport()
    battery(gr, live={"observed": g_ind - g_cor, "min": 0.02, "name": "correlation_changes_the_ensemble_gain"},
            placebo={"observed": abs(by["independent"]["vote"]["mean"] - by["independent"]["bayes"]["mean"]), "tol": 3.0, "name": "vote_vs_bayes_reported"},
            positive={"observed": float(by["independent"]["bayes"]["mean"] >= mean_of(rows, "mean_single", lambda r: r["correlation"] == "independent" and r["method"] == "single") - 0.10), "expected": 1.0, "tol": 0.0, "name": "pooled_prior_no_worse_than_the_mean_single_reader", "detail": "the shared-evidence pool (mean prior, one likelihood) must sit at or above the average reader; the best single reader may beat it for makers near that reader"},
            surface={"accuracy": conf_cor - acc_cor, "chance": conf_ind - acc_ind, "tol": 2.0, "name": "correlated_overconfidence_reported"},
            oracle={"observed": by["independent"]["bayes"]["mean"] - np.log(1 / 40), "min": 0.5, "name": "ensemble_identifies"},
            prediction={"gain": g_ind, "min": -1.0, "name": "independent_gain"},
            calibration={"observed": conf_cor - acc_cor, "reference": conf_ind - acc_ind, "direction": "up", "tol": 0.0, "name": "correlated_readers_more_overconfident"})
    criterion(v, "C15", passed, independent_gain=g_ind, correlated_gain=g_cor, correlated_overconfidence=conf_cor - acc_cor, independent_overconfidence=conf_ind - acc_ind)
    v["results"].update({"by_correlation_and_method": by})
    receipt(v, rows, card, ctx)
    narrative(v, f"Pooling four readers' posteriors gained {g_ind:+.2f} nats over the best single reader when their errors were independent and {g_cor:+.2f} when all "
                 f"shared a false group note; the correlated pool's confidence exceeded its accuracy by {conf_cor - acc_cor:+.2f} against {conf_ind - acc_ind:+.2f}.",
              "Plurality corrects projection only with diverse errors; correlated false similarity stays visible as overconfidence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# C16 — transfer of the nested-prior law (transfer lane).
# --------------------------------------------------------------------------- #
def unit_C16(ctx):
    H = harness(ctx)
    cells, reports = route_cells(ctx, H, ("self", "equal_local", "within_common", "oracle"), doses=(1,), tag="C16")
    return {"rows": cells.rows(), "matching": matching_summary(reports)}


def reduce_C16(card, units, ctx):
    v = start(card, ctx, "The nested-prior interaction found in discovery keeps its qualitative shape on fresh common-substrate "
              "families, groups, expertise ecologies and maker distributions.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    surf = {b: {"self_minus_equal_local": gain(rows, "self", "equal_local", lambda r, b=b: r["sim_bin"] == b, tag=f"C16e{b}"),
                "self_minus_within_common": gain(rows, "self", "within_common", lambda r, b=b: r["sim_bin"] == b, tag=f"C16c{b}")} for b in ("near", "mid", "far", "anti")}
    near, far = surf["near"]["self_minus_equal_local"]["mean"], surf["far"]["self_minus_equal_local"]["mean"]
    disc = C.load_verdict("C04", "discovery")
    d_near = d_far = None
    if disc:
        d_near = disc["results"]["criterion_C04"]["near_gain"]
        d_far = disc["results"]["criterion_C04"]["far_gain"]
    same_sign = bool(d_near is not None and np.sign(near) == np.sign(d_near) and (far <= 0.02) == (d_far <= 0.02))
    gr = G.GateReport()
    battery(gr, live={"observed": abs(near) + abs(far), "min": 1e-4, "name": "fresh_worlds_move_the_estimate"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "transfer_lineage_untouched_by_discovery"},
            positive={"observed": mean_of(rows, "top1", lambda r: r["route"] == "oracle"), "expected": 1.0, "tol": 0.15, "name": "oracle_ceiling_on_fresh_worlds"},
            surface={"accuracy": float(np.nanmean([u["matching"]["equal_local"]["mean_residual_divergence_gap"] or 0 for u in units])), "chance": 0.0, "tol": 0.25, "name": "matching_holds_on_fresh_families"},
            oracle={"observed": mean_of(rows, "ls", lambda r: r["route"] == "oracle") - np.log(1 / 40), "min": 1.0, "name": "identifiable"},
            prediction={"gain": near, "min": -1.0, "name": "near_gain_on_fresh"},
            calibration={"observed": float(same_sign), "reference": 1.0, "direction": "up", "tol": 1.0, "name": "sign_agreement_with_discovery_reported"})
    criterion(v, "C16", same_sign, near_gain=near, far_gain=far, discovery_near=d_near, discovery_far=d_far)
    v["results"].update({"conditional_surface": surf})
    v["matching_residuals"] = {"per_unit": [u["matching"] for u in units]}
    receipt(v, rows, card, ctx)
    narrative(v, f"On fresh families the self-minus-equal-local gain was {near:+.2f} nats near and {far:+.2f} far, against {d_near if d_near is None else round(d_near, 2)} and "
                 f"{d_far if d_far is None else round(d_far, 2)} in discovery.",
              "The nested-prior law " + ("transfers in sign." if same_sign else "does not transfer in sign; the discovery shape is family-specific."))
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(same_sign))
