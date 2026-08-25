"""Trunk D: many hands, one artifact (spec section 13).

Multi-part artifacts under six control ecologies with full decision logs, so causal reach is
measured by intervention and attribution is scored against what actually happened. The reader
sees the artifact; the logs are the ground truth it is scored against.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from ..schemas import new_verdict
from ..world import make_maker, population
from ..hierarchy import ECOLOGIES, produce, part_goal_posteriors, coherence_features, _part_emission
from . import finish, worlds_for, decide_state


def _hist(feats, nf):
    h = np.bincount(np.asarray(feats), minlength=nf).astype(float)
    return h / h.sum()


def _js(p, q):
    m = 0.5 * (p + q)

    def kl(a, b):
        s = a > 0
        return float((a[s] * np.log(a[s] / b[s])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _part_ll(world, template, part):
    """log P(part | goal) for each goal under a template (alpha-mixed base signature)."""
    a = world.alpha["CREATOR"]
    out = np.zeros(world.ng)
    for g in range(world.ng):
        e = a * template[g] + (1 - a) * world.synth
        out[g] = np.log(np.maximum(e[part] / e.sum(), 1e-300)).sum()
    return out


def _lse(x):
    x = np.asarray(x, float)
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


def _redraw_part(world, contributor, g, slot, rng, steps):
    e = _part_emission(world, contributor.template, g, slot)
    a = world.alpha[contributor.tier]
    e = a * e + (1 - a) * world.synth
    return rng.choice(world.nf, size=int(steps), p=e / e.sum())


def run_D01(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Artifacts can be produced under six control ecologies matched on part "
                    "quality, counts and surface style, so that a cheap surface classifier cannot tell "
                    "the ecologies apart.", "METHOD")
    quality, hists, labels = {}, [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("D01", wid, 0)
            contributors = population(world, 4, rng, k_choices=(0.2,))
            for eco in ECOLOGIES:
                for i in range(40):
                    dg = (i % world.ng) if eco in ("director", "shared_brief", "editor", "ratifier") else None
                    art = produce(world, eco, contributors, rng, director_goal=dg)   # goals stratified so surfaces match by construction
                    quality.setdefault(eco, []).append(coherence_features(world, world.sig, art["parts"])["mean_part_confidence"])
                    hists.append(_hist(art["features"], world.nf))
                    labels.append(eco)
    Hs, L = np.array(hists), np.array(labels)
    cents = {e: Hs[L == e].mean(axis=0) for e in ECOLOGIES}
    js_max = max(_js(cents[a], cents[b]) for a in ECOLOGIES for b in ECOLOGIES if a < b)
    idx = np.arange(len(Hs))
    train = (idx % 2 == 0)          # held-out half; leave-one-out centroids sit below chance by construction
    c = {e: Hs[train & (L == e)].mean(axis=0) for e in ECOLOGIES}
    correct = [min(c, key=lambda e: _js(Hs[i], c[e])) == L[i] for i in idx[~train]]
    q = {e: float(np.mean(x)) for e, x in quality.items()}
    gr = G.GateReport()
    gr.placebo("surface_style_matched", observed_max_deviation=float(js_max), tol=0.02)
    gr.positive("quality_matched", observed=float(max(q.values()) - min(q.values())), expected=0.0, tol=0.15)
    gr.positive("cheap_classifier_near_chance", observed=float(np.mean(correct)), expected=1 / 6, tol=0.15)
    v["results"] = {"quality_by_ecology": q, "max_style_js": float(js_max), "surface_classifier_accuracy": float(np.mean(correct))}
    v["what_must_hold_outside_the_simulation"] = "nothing; a construction check"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_D02(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "An intervention at the director level reaches more of the artifact than "
                    "an intervention at a local part; reach is measured as the fraction of parts that "
                    "change under the same random stream.", "CONSTRUCTED_MECHANISM")
    reach = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            contributors = population(world, 4, C.rng_for("D02", wid, 0), k_choices=(0.2,))
            levels = (("director", "director_goal", {"director_goal": 1}), ("shared_brief", "brief", {"brief": 1}),
                      ("director", "secondary_goal", {"secondary_goal": 3}), ("director", "local", {"local_part": 0, "local_slot": 5}),
                      ("ratifier", "ratification", {"no_veto": True}))
            for i in range(20):
                for eco, level, iv in levels:
                    s = C.seed(f"D02:{wid}:{i}:{level}")
                    base = produce(world, eco, contributors, np.random.default_rng(s), director_goal=0)
                    iv = dict(iv)
                    if level == "local":
                        iv["local_slot"] = (base["log"]["parts"][0]["slot"] + 1) % len(world.family_names)
                    alt = produce(world, eco, contributors, np.random.default_rng(s), director_goal=0, intervene=iv)
                    # reach is measured on the decisions (goal, slot) each part was produced under; the surface
                    # difference is reported beside it, because small cue shifts leave many draws unchanged
                    changed = [a["goal"] != b["goal"] or a["slot"] != b["slot"] for a, b in zip(base["log"]["parts"], alt["log"]["parts"])]
                    surface = [not np.array_equal(a, b) for a, b in zip(base["parts"], alt["parts"])]
                    reach.setdefault(level, []).append(float(np.mean(changed)))
                    reach.setdefault(level + "_surface", []).append(float(np.mean(surface)))
    table = {k: float(np.mean(x)) for k, x in reach.items()}
    gr = G.GateReport()
    gr.positive("local_intervention_reaches_one_part", observed=table["local"], expected=0.25, tol=1e-9,
                detail="a local slot change alters exactly the part it targets: the known answer under a shared random stream")
    gr.positive("director_goal_reaches_every_part", observed=table["director_goal"], expected=1.0, tol=1e-9,
                detail="every part's goal is the director's or its successor; changing the director's goal changes all of them")
    v["results"] = {"reach_by_level": table, "criterion_C_D02": {"passed": bool(table["director_goal"] > table["local"])}}
    v["what_must_hold_outside_the_simulation"] = "interventions on a production process are possible"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_D03(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "With coherence matched (three of four parts on one goal in both ecologies), "
                    "the exact structure reader separates a central director from a shared brief where a "
                    "coherence baseline sits at chance.", "CONSTRUCTED_MECHANISM")
    acc, base_acc = [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("D03", wid, 0)
            contributors = population(world, 4, rng, k_choices=(0.2,))
            coh = {"director": [], "shared_brief": []}
            items = []
            for i in range(40):
                eco = "director" if i % 2 == 0 else "shared_brief"
                art = produce(world, eco, contributors, rng)
                parts = list(art["parts"])
                if eco == "shared_brief":
                    b = art["log"]["brief"]
                    g_dev = int(rng.choice([g for g in range(world.ng) if g != b]))
                    parts[-1] = _redraw_part(world, contributors[3 % len(contributors)], g_dev, int(rng.integers(len(world.family_names))), rng, len(parts[-1]))
                ll = np.array([_part_ll(world, world.sig, p) for p in parts])
                n = len(parts)
                l_dir = _lse([ll[:-1, dg].sum() + ll[-1, (dg + 1) % world.ng] for dg in range(world.ng)])
                l_brief = _lse([ll[:-1, b].sum() + _lse([ll[-1, g] for g in range(world.ng) if g != b]) - np.log(world.ng - 1) for b in range(world.ng)])
                pred = "director" if l_dir > l_brief else "shared_brief"
                acc.append(float(pred == eco))
                c = coherence_features(world, world.sig, parts)["share_dominant_goal"]
                coh[eco].append(c)
                items.append((c, eco))
            med = float(np.median([c for c, _ in items]))
            base_acc.append(float(np.mean([(("director" if c >= med else "shared_brief") == e) for c, e in items])))
    gr = G.GateReport()
    gr.positive("coherence_is_matched_by_construction", observed=float(np.mean(base_acc)), expected=0.5, tol=0.15,
                detail="the coherence baseline must sit at chance; if it does not, the construction leaked coherence")
    gr.live("structure_reader_moves", observed_change=float(np.mean(acc) - 0.5), min_change=0.1)
    v["results"] = {"structure_reader_accuracy": float(np.mean(acc)), "coherence_baseline_accuracy": float(np.mean(base_acc))}
    v["what_must_hold_outside_the_simulation"] = "a director allocates a structured secondary goal rather than a random deviation"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_D04(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Attribution is level-specific: the director level is attributed through "
                    "the primary goal, the local level through each part's template; a token-share "
                    "baseline attributes neither.", "CONSTRUCTED_MECHANISM")
    res = {"director_level": [], "local_level": [], "token_share_director": []}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("D04", wid, 0)
            names = world.family_names
            contributors = [make_maker(world, f"c{i}", f"peaked_{i}", rng, k=0.5) for i in range(4)]
            for i in range(40):
                d = int(rng.integers(4))
                art = produce(world, "director", contributors, rng, director_goal=int(np.argmax(contributors[d].w)))
                P = part_goal_posteriors(world, world.sig, art["parts"])
                primary = int(np.bincount(P.argmax(axis=1), minlength=world.ng).argmax())
                cand = [j for j, c in enumerate(contributors) if int(np.argmax(c.w)) == primary]
                res["director_level"].append(float(d in cand and len(cand) == 1))
                for part, entry in zip(art["parts"], art["log"]["parts"]):
                    lls = [max(_part_ll(world, c.template, part)) for c in contributors]
                    res["local_level"].append(float(contributors[int(np.argmax(lls))].id == entry["contributor"]))
                counts = np.bincount([int(e["contributor"][1:]) for e in art["log"]["parts"]], minlength=4)
                res["token_share_director"].append(float(int(np.argmax(counts)) == d))
    table = {k: float(np.mean(x)) for k, x in res.items()}
    gr = G.GateReport()
    gr.positive("token_share_is_chance_for_the_director", observed=table["token_share_director"], expected=0.25, tol=0.15,
                detail="each contributor writes one part; token share carries nothing about who directed")
    gr.live("local_attribution_above_chance", observed_change=float(table["local_level"] - 0.25), min_change=0.2)
    v["results"] = {"accuracy": table}
    v["what_must_hold_outside_the_simulation"] = "contributors' templates and profiles are known to the reader"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_D05(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Under a rewrite ladder, local template attribution dies first and the "
                    "director's goal structure survives longest: upstream reach outlives style.",
                    "CONSTRUCTED_MECHANISM")
    res = {}
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("D05", wid, 0)
            contributors = [make_maker(world, f"c{i}", f"peaked_{i}", rng, k=0.5) for i in range(4)]
            rewriter = make_maker(world, "rw", "uniform", rng, k=0.5)
            for i in range(30):
                d = int(rng.integers(4))
                dg = int(np.argmax(contributors[d].w))
                art = produce(world, "director", contributors, rng, director_goal=dg)
                for r in (0.0, 0.25, 0.5, 0.75, 1.0):
                    parts = []
                    for part, entry in zip(art["parts"], art["log"]["parts"]):
                        p = part.copy()
                        n_rw = int(round(r * len(p)))
                        if n_rw:
                            idx = rng.choice(len(p), size=n_rw, replace=False)
                            p[idx] = _redraw_part(world, rewriter, entry["goal"], entry["slot"], rng, n_rw)
                        parts.append(p)
                    P = part_goal_posteriors(world, world.sig, parts)
                    primary = int(np.bincount(P.argmax(axis=1), minlength=world.ng).argmax())
                    cell = res.setdefault(str(r), {"director": [], "local": []})
                    cell["director"].append(float(primary == dg))
                    for part, entry in zip(parts, art["log"]["parts"]):
                        lls = [max(_part_ll(world, c.template, part)) for c in contributors]
                        cell["local"].append(float(contributors[int(np.argmax(lls))].id == entry["contributor"]))
    table = {r: {k: float(np.mean(x)) for k, x in d.items()} for r, d in res.items()}
    gr = G.GateReport()
    gr.live("full_rewrite_erases_local_attribution", observed_change=float(table["0.0"]["local"] - table["1.0"]["local"]), min_change=0.2)
    v["results"] = {"attribution_by_rewrite_strength": table, "criterion_C_D05": {"director_survives_full_rewrite": table["1.0"]["director"]}}
    v["what_must_hold_outside_the_simulation"] = "a rewrite preserves the goal it was rewriting toward"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_D06(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "A director's next intervention (the next artifact's primary goal) is "
                    "predicted above a frequency baseline from the profile inferred over earlier artifacts.",
                    "CONSTRUCTED_MECHANISM")
    gains, ls_model, ls_base = {}, [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("D06", wid, 0)
            names = world.family_names
            contributors = population(world, 4, rng, k_choices=(0.2,))
            for i in range(30):
                w_d = world.family[names[i % len(names)]]
                goals = [int(rng.choice(world.ng, p=w_d)) for _ in range(7)]
                soft = []
                for t in range(6):
                    art = produce(world, "director", contributors, rng, director_goal=goals[t])
                    P = part_goal_posteriors(world, world.sig, art["parts"])
                    soft.append(P[:-1].mean(axis=0))
                ll = np.array([sum(np.log(max(float(s @ world.family[n]), 1e-12)) for s in soft) for n in names])
                q = np.exp(ll - ll.max())
                q /= q.sum()
                pred = sum(q[k] * world.family[n] for k, n in enumerate(names))
                freq = np.sum(soft, axis=0) + 0.5
                freq = freq / freq.sum()
                a, b = float(np.log(max(pred[goals[6]], 1e-12))), float(np.log(max(freq[goals[6]], 1e-12)))
                ls_model.append(a)
                ls_base.append(b)
                gains.setdefault(wid, []).append(a - b)
    boot = C.hboot(gains, np.random.default_rng(C.seed("D06")), draws=300)
    gr = G.GateReport()
    gr.live("prediction_beats_uniform", observed_change=float(np.mean(ls_model) - np.log(1 / 4)), min_change=0.05)
    v["results"] = {"model_log_score": float(np.mean(ls_model)), "frequency_log_score": float(np.mean(ls_base)), "gain": boot,
                    "criterion_C_D06": {"passed": bool(boot["mean"] > 0)}}
    v["what_must_hold_outside_the_simulation"] = "a director's goals across artifacts are drawn from one standing profile"
    return finish(card, v, gr, __file__, decide_state(gr))


def run_D07(card, cfg, workers=1, lane="both"):
    v = new_verdict(card, lane, "Ratified and merely-unnoticed artifacts can be identical; the artifact-only "
                    "reader abstains on them, the record reader separates them at once, and later artifacts "
                    "separate them for the artifact-only reader too.", "CONSTRUCTED_MECHANISM")
    abst, rec_acc, later = [], [], []
    with C.timed(v):
        for wid, world in worlds_for(cfg, lane):
            rng = C.rng_for("D07", wid, 0)
            contributors = population(world, 4, rng, k_choices=(0.2,))
            for i in range(30):
                dg = int(rng.integers(world.ng))
                # ratified: the ratifier vetoes the off-goal part and redraws it under dg
                rat = produce(world, "ratifier", contributors, rng, director_goal=dg)
                # unnoticed: no ratifier; the contributor happened to be on-goal already
                unn = produce(world, "ratifier", contributors, rng, director_goal=dg, intervene={"secondary_goal": dg, "no_veto": True})
                for art, truth in ((rat, "ratified"), (unn, "unnoticed")):
                    P = part_goal_posteriors(world, world.sig, art["parts"])
                    on_goal = float((P.argmax(axis=1) == dg).mean())
                    # artifact-only: both histories predict all parts on-goal; the likelihood ratio is one
                    abst.append(1.0 if on_goal == 1.0 else 0.0)
                    rec_acc.append(float(("ratified" if art["log"]["vetoed"] else "unnoticed") == truth))
                # later: six more artifacts from each history; the unnoticed source shows off-goal parts
                offs = {"ratified": 0, "unnoticed": 0}
                for t in range(6):
                    a1 = produce(world, "ratifier", contributors, rng, director_goal=dg)
                    iv = {"no_veto": True} | ({"secondary_goal": dg} if rng.random() < 0.7 else {})
                    a2 = produce(world, "ratifier", contributors, rng, director_goal=dg, intervene=iv)
                    for art, key in ((a1, "ratified"), (a2, "unnoticed")):
                        P = part_goal_posteriors(world, world.sig, art["parts"])
                        offs[key] += int((P.argmax(axis=1) != dg).sum())
                later.append(float(offs["unnoticed"] > offs["ratified"]))
    gr = G.GateReport()
    gr.positive("record_reader_separates_identical_artifacts", observed=float(np.mean(rec_acc)), expected=1.0, tol=1e-9,
                detail="the log says whether a veto happened; a record reader cannot be wrong here")
    gr.live("later_artifacts_separate", observed_change=float(np.mean(later) - 0.5), min_change=0.2)
    v["results"] = {"artifact_only_abstention_rate": float(np.mean(abst)), "record_reader_accuracy": float(np.mean(rec_acc)),
                    "later_separation_rate": float(np.mean(later))}
    v["what_must_hold_outside_the_simulation"] = "histories continue; the reader gets more than one artifact"
    return finish(card, v, gr, __file__, decide_state(gr))
