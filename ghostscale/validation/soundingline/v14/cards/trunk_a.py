"""Trunk A — affect ownership and strategic communication (spec §5, cards A01-A10).
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import communication as CM
from . import Cells, battery, criterion, decide_state, extra_gate, finish, held_out_classifier, mean_of, narrative, pursuit_of, receipt, rng, sizes, start, world_for

REGION_NAMES = ("honest_warning", "sincere_fanatic", "strategic_propagandist", "neutral_report")


def _full_ll(s, art, r, with_correction=True, with_private=True):
    ll = CM.loglik_artifact(art) + CM.loglik_appraisal_cue(art, "intensity")
    if with_correction:
        ll = ll + CM.loglik_correction(CM.correction_event(s, r))
    if with_private:
        ll = ll + CM.loglik_private(CM.private_action(s, r))
    return ll


def _region_acc(post, region):
    rp = CM.region_posterior(post)
    return float(max(rp, key=rp.get) == region), float(rp[region])


# --------------------------------------------------------------------------- #
# A01 — owners independently live.
# --------------------------------------------------------------------------- #
def unit_A01(ctx):
    r = rng(ctx, "a01")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["makers"] // 8)
    owners = ("reader_response", "maker_appraisal", "intended_effect", "content", "goal", "reliability", "uptake")
    for i in range(n):
        # a contradicting pool shown in full, no alarm sought: every owner's change is visible in
        # what it owns (on a supporting pool, selection collides with full showing by construction)
        base = CM.source(r, support="contradicts", policy="full", appraisal="admire")
        arts = [CM.speak(base, r) for _ in range(3)]
        ll0 = sum(_full_ll(base, a, r) for a in arts)
        post0 = CM.posterior(ll0)
        m0 = {"belief": CM.marginal(post0, "belief"), "appraisal": CM.marginal(post0, "appraisal"), "support": CM.marginal(post0, "support"), "policy": CM.marginal(post0, "policy")}
        resp0 = float(np.mean([CM.reader_response(a) for a in arts]))
        content_p = float(m0["support"].max())                   # the reader's belief in its reading of the content
        up0 = CM.uptake(post0, base["reliability"], content_p)
        for owner in owners:
            s = dict(base)
            moved = {"reader_response": 0.0, "maker_appraisal": 0.0, "intended_effect": 0.0, "content": 0.0, "goal": 0.0, "reliability": 0.0, "uptake": 0.0}
            if owner == "reader_response":
                arts2 = [dict(a, intensity=(1 if a["intensity"] == 3 else 3)) for a in arts]         # the reader's sensitivity, not the artifact's content
                resp1 = float(np.mean([CM.reader_response(a, sensitivity=2.5) for a in arts]))
                moved["reader_response"] = abs(resp1 - resp0)
                post1 = CM.posterior(sum(_full_ll(base, a, r) for a in arts))
            elif owner == "maker_appraisal":
                s["belief"] = {"threat": "benefit", "benefit": "neutral", "neutral": "threat"}[base["belief"]]
                arts2 = [CM.speak(s, r) for _ in range(3)]
                post1 = CM.posterior(sum(_full_ll(s, a, r) for a in arts2))
                moved["maker_appraisal"] = C.js(m0["belief"], CM.marginal(post1, "belief"))
            elif owner == "intended_effect":
                s["appraisal"] = {"alarm": "calm", "calm": "admire", "admire": "alarm"}[base["appraisal"]]
                arts2 = [CM.speak(s, r) for _ in range(3)]
                post1 = CM.posterior(sum(_full_ll(s, a, r) for a in arts2))
                moved["intended_effect"] = C.js(m0["appraisal"], CM.marginal(post1, "appraisal"))
            elif owner == "content":
                s["support"] = {"supports": "contradicts", "contradicts": "none", "none": "supports"}[base["support"]]
                arts2 = [CM.speak(s, r) for _ in range(3)]
                post1 = CM.posterior(sum(_full_ll(s, a, r) for a in arts2))
                moved["content"] = C.js(m0["support"], CM.marginal(post1, "support"))
            elif owner == "goal":
                s["policy"] = {"full": "fabricate", "cherry_pick": "full", "fabricate": "cherry_pick"}[base["policy"]]
                arts2 = [CM.speak(s, r) for _ in range(3)]
                post1 = CM.posterior(sum(_full_ll(s, a, r) for a in arts2))
                moved["goal"] = C.js(m0["policy"], CM.marginal(post1, "policy"))
            elif owner == "reliability":
                up1 = CM.uptake(post0, 0.1 if base["reliability"] > 0.5 else 0.9, content_p)
                moved["reliability"] = abs(up1["policy_uptake"] - up0["policy_uptake"])
                post1 = post0
            else:
                up1 = CM.uptake(post0, base["reliability"], content_p, gate="suppress")
                moved["uptake"] = abs(up1["policy_uptake"] - up0["policy_uptake"])
                post1 = post0
            # leak: the reader-response owner must not move any posterior; reliability/uptake must not move the goal posterior
            leak = 0.0
            if owner in ("reader_response", "reliability", "uptake"):
                leak = C.js(m0["policy"], CM.marginal(post1, "policy")) + C.js(m0["belief"], CM.marginal(post1, "belief"))
            cells.add({"owner": owner}, own=moved[owner], leak=leak)
    return {"rows": cells.rows()}


def reduce_A01(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A01"]
    v = start(card, ctx, "Reader response, maker appraisal, intended audience effect, content support, communicative goal, source reliability and uptake are separate owners: each moves its own quantity and none is written into another.", "METHOD")
    rows = [r for u in units for r in u["rows"]]
    owners = ("reader_response", "maker_appraisal", "intended_effect", "content", "goal", "reliability", "uptake")
    own = {o: mean_of(rows, "own", lambda r, o=o: r["owner"] == o) for o in owners}
    leak = {o: mean_of(rows, "leak", lambda r, o=o: r["owner"] == o) for o in owners}
    passed = bool(min(own.values()) >= cr["min_own"] and max(leak.values()) <= cr["max_leak"])
    gr = G.GateReport()
    battery(gr, live={"observed": min(own.values()), "min": cr["min_own"], "name": "every_owner_moves_its_own_quantity"},
            placebo={"observed": max(leak.values()), "tol": cr["max_leak"], "name": "reader_response_reliability_and_uptake_leave_the_source_posteriors"},
            positive={"observed": own["content"], "expected": max(own["content"], cr["min_own"]), "tol": 0.0, "name": "content_support_readable"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "intensity_matched_where_owners_swap"},
            oracle={"observed": own["maker_appraisal"], "min": cr["min_own"], "name": "maker_belief_readable"},
            prediction={"gain": own["intended_effect"], "min": 0.0, "name": "intended_effect_readable"},
            calibration={"observed": max(leak.values()), "reference": min(own.values()), "direction": "down", "tol": 0.0, "name": "leaks_below_own_effects"})
    criterion(v, "A01", passed, own=own, leak=leak)
    receipt(v, rows, card, ctx)
    narrative(v, f"Each owner moved its own quantity by at least {min(own.values()):.2f} (Jensen-Shannon or absolute) and the reader's response, the reliability and the uptake gate moved the source posteriors by at most {max(leak.values()):.3f}.",
              "The factorization V13 established survives the addition of appraisal owners and an audience.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A02 — the reader's own response as a prior for intended effect.
# --------------------------------------------------------------------------- #
def unit_A02(ctx):
    r = rng(ctx, "a02")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(8, sizes(ctx)["makers"] // 2)
    for i in range(n):
        s = CM.source(r)
        art = CM.speak(s, r)
        truth = CM.APPRAISALS.index(s["appraisal"])
        base = CM.posterior(CM.loglik_artifact(art))
        for similarity in ("similar", "dissimilar"):
            # the audience the source models: similar readers respond like it; dissimilar readers respond inversely
            # the reader's own response reads the intensity; a similar reader projects the planted
            # intensity law onto the intended effect, a dissimilar reader (responding inversely) projects it backwards
            resp = CM.reader_response(art, sensitivity=2.5)
            high = resp > 0.75
            if similarity == "dissimilar":
                high = not high
            proj = np.array([CM.P_HIGH[a] if high else 1.0 - CM.P_HIGH[a] for a in CM.APPRAISALS]); proj = proj / proj.sum()
            for reader in ("projecting", "neutral"):
                if reader == "projecting":
                    prior = np.ones(CM.N_SOURCE_STATES)
                    for k, st in enumerate(CM.SOURCE_STATES):
                        prior[k] = proj[st[2]]
                    prior = prior / prior.sum()
                    post = CM.posterior(CM.loglik_artifact(art), prior)
                else:
                    post = base
                am = CM.marginal(post, "appraisal")
                cells.add({"similarity": similarity, "reader": reader}, ls=float(np.log(max(am[truth], 1e-12))), correct=float(int(np.argmax(am)) == truth), conf=float(am.max()))
    return {"rows": cells.rows()}


def reduce_A02(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A02"]
    v = start(card, ctx, "The reader's own induced response is a useful prior for the intended audience effect only when the reader responds like the audience the source modelled; otherwise projecting it costs.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {s: {k: mean_of(rows, "ls", lambda r, s=s, k=k: r["similarity"] == s and r["reader"] == k) for k in ("projecting", "neutral")} for s in ("similar", "dissimilar")}
    gain_sim = ls["similar"]["projecting"] - ls["similar"]["neutral"]
    gain_dis = ls["dissimilar"]["projecting"] - ls["dissimilar"]["neutral"]
    passed = bool(gain_sim >= cr["min_gain"] and gain_dis <= 0.0)
    gr = G.GateReport()
    battery(gr, live={"observed": gain_sim - gain_dis, "min": 0.05, "name": "similarity_moves_the_value_of_projection"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "neutral_reader_identical_across_similarity"},
            positive={"observed": gain_sim, "expected": max(gain_sim, cr["min_gain"]), "tol": 0.0, "name": "projection_helps_when_similar"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifact_both_readers"},
            oracle={"observed": ls["similar"]["projecting"] - np.log(1 / 3), "min": 0.0, "name": "intended_effect_above_chance"},
            prediction={"gain": gain_sim, "min": cr["min_gain"], "name": "similar_projection_gain"},
            calibration={"observed": gain_dis, "reference": 0.0, "direction": "down", "tol": 0.0, "name": "dissimilar_projection_does_not_gain"})
    criterion(v, "A02", passed, gain_similar=gain_sim, gain_dissimilar=gain_dis, scores=ls)
    receipt(v, rows, card, ctx)
    narrative(v, f"Using its own response as a prior on the intended effect gained the reader {gain_sim:+.3f} nats when it responded like the audience and {gain_dis:+.3f} when it did not.",
              "Self-projection of affect is a similarity bet, exactly like self-projection of a profile.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A03 / A04 — partial identifiability of intended effect and maker appraisal.
# --------------------------------------------------------------------------- #
def unit_A03(ctx):
    r = rng(ctx, "a03")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        s = CM.source(r)
        arts = [CM.speak(s, r) for _ in range(2)]
        truth_eff = CM.APPRAISALS.index(s["appraisal"])
        truth_bel = CM.BELIEFS.index(s["belief"])
        for known in ("known", "uncertain"):
            if known == "known":
                ll = sum(CM.loglik_artifact(a) + CM.loglik_appraisal_cue(a, "intensity") for a in arts)
                ll = ll + CM.loglik_private(CM.private_action(s, r)) + CM.loglik_correction(CM.correction_event(s, r))
            else:                                                       # the assertion is withheld: the appraisal owner stays uncertain
                ll = sum(CM.loglik_appraisal_cue(a, "intensity") + _evidence_only_ll(a) for a in arts)
            post = CM.posterior(ll)
            am, bm = CM.marginal(post, "appraisal"), CM.marginal(post, "belief")
            cells.add({"maker_appraisal": known}, effect_correct=float(int(np.argmax(am)) == truth_eff), effect_p=float(am[truth_eff]), belief_entropy=C.entropy(bm),
                       belief_correct=float(int(np.argmax(bm)) == truth_bel))
    return {"rows": cells.rows()}


def _evidence_only_ll(art):
    """log P(evidence tokens | state) with the assertion withheld."""
    out = np.zeros(CM.N_SOURCE_STATES)
    for i, (b, sup, ap, pol, co, pv) in enumerate(CM.SOURCE_STATES):
        p = CM.emission_policy(CM.SUPPORT[sup], CM.POLICIES[pol])
        out[i] = sum(np.log(p[t]) for t in art["evidence"])
    return out


def reduce_A03(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A03"]
    v = start(card, ctx, "The intended audience effect can be recovered from intensity and evidence selection while the maker's own appraisal stays uncertain when its assertion is withheld.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    eff = {k: mean_of(rows, "effect_correct", lambda r, k=k: r["maker_appraisal"] == k) for k in ("known", "uncertain")}
    ent = {k: mean_of(rows, "belief_entropy", lambda r, k=k: r["maker_appraisal"] == k) for k in ("known", "uncertain")}
    passed = bool(eff["uncertain"] >= cr["min_intended"] and ent["uncertain"] >= cr["min_uncertainty"])
    gr = G.GateReport()
    battery(gr, live={"observed": ent["uncertain"] - ent["known"], "min": 0.1, "name": "withholding_the_assertion_leaves_appraisal_uncertain"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_artifacts_both_conditions"},
            positive={"observed": eff["known"], "expected": 1.0, "tol": 0.5, "name": "effect_recovered_with_full_evidence"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "intensity_is_the_only_effect_cue"},
            oracle={"observed": eff["known"] - 1 / 3, "min": 0.1, "name": "effect_identifiable"},
            prediction={"gain": eff["uncertain"] - 1 / 3, "min": 0.0, "name": "effect_above_chance_under_uncertainty"},
            calibration={"observed": ent["known"], "reference": ent["uncertain"], "direction": "down", "tol": 0.0, "name": "appraisal_entropy_falls_with_the_assertion"})
    criterion(v, "A03", passed, effect_accuracy=eff, belief_entropy=ent)
    receipt(v, rows, card, ctx)
    narrative(v, f"With the assertion withheld the intended effect was still named {eff['uncertain']:.0%} of the time while the maker's belief kept {ent['uncertain']:.2f} nats of entropy; with it, {eff['known']:.0%} and {ent['known']:.2f}.",
              "What a source wants its audience to feel and what it believes are read from different evidence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_A04(ctx):
    r = rng(ctx, "a04")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        base = CM.source(r)
        for swap in ("same", "swapped"):
            s = dict(base)
            if swap == "swapped":                                       # the intended effect differs from what the maker's belief would produce
                s["appraisal"] = {"alarm": "calm", "calm": "admire", "admire": "alarm"}[CM.maker_appraisal(s)]
            arts = [CM.speak(s, r) for _ in range(2)]
            ll = sum(CM.loglik_artifact(a) + CM.loglik_appraisal_cue(a, "intensity") for a in arts) + CM.loglik_private(CM.private_action(s, r))
            post = CM.posterior(ll)
            bm = CM.marginal(post, "belief")
            truth_bel = CM.BELIEFS.index(s["belief"])
            pv = CM.marginal(post, "private")
            priv_truth = CM.PRIVATE.index(s["private"])
            cells.add({"owner_swap": swap}, appraisal_correct=float(int(np.argmax(bm)) == truth_bel), private_correct=float(int(np.argmax(pv)) == priv_truth), conf=float(bm.max()))
    return {"rows": cells.rows()}


def reduce_A04(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A04"]
    v = start(card, ctx, "The maker's own appraisal is recovered from its assertion and private action even when the effect it intends on its audience points elsewhere.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ac = {s: mean_of(rows, "appraisal_correct", lambda r, s=s: r["owner_swap"] == s) for s in ("same", "swapped")}
    pc = {s: mean_of(rows, "private_correct", lambda r, s=s: r["owner_swap"] == s) for s in ("same", "swapped")}
    passed = bool(ac["swapped"] >= cr["min_owner"] and pc["swapped"] >= cr["min_owner"])
    gr = G.GateReport()
    battery(gr, live={"observed": ac["swapped"] - 1 / 3, "min": 0.1, "name": "appraisal_read_above_chance_under_swap"},
            placebo={"observed": abs(ac["same"] - ac["swapped"]), "tol": 0.25, "name": "swap_leaves_appraisal_recovery"},
            positive={"observed": ac["same"], "expected": 1.0, "tol": 0.5, "name": "appraisal_recovered_unswapped"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "intensity_matched"},
            oracle={"observed": pc["swapped"] - 0.5, "min": 0.0, "name": "private_action_predicted"},
            prediction={"gain": pc["swapped"] - 0.5, "min": 0.0, "name": "off_audience_action"},
            calibration={"observed": abs(mean_of(rows, "conf", lambda r: r["owner_swap"] == "swapped") - ac["swapped"]), "reference": 0.25, "direction": "down", "tol": 0.0, "name": "appraisal_confidence_tracks_accuracy"})
    criterion(v, "A04", passed, appraisal_accuracy=ac, private_accuracy=pc)
    receipt(v, rows, card, ctx)
    narrative(v, f"With the intended effect swapped away from the maker's own appraisal, the appraisal was still recovered {ac['swapped']:.0%} of the time and the private action predicted {pc['swapped']:.0%}.",
              "The maker's appraisal is owned by the maker and read from what it asserts and does off-audience, not from what it wants felt.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A05 / A06 — regions without labels; the fanatic/propagandist boundary.
# --------------------------------------------------------------------------- #
def unit_A05(ctx):
    r = rng(ctx, "a05")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["makers"] // 8)
    X, y = [], []
    for i in range(n):
        for region in REGION_NAMES:
            s = CM.source(r, region)
            arts = [CM.speak(s, r) for _ in range(2)]
            ll_art = sum(CM.loglik_artifact(a) + CM.loglik_appraisal_cue(a, "intensity") for a in arts)
            ll_cf = ll_art + CM.loglik_correction(CM.correction_event(s, r)) + CM.loglik_private(CM.private_action(s, r))
            for ev, ll in (("artifact_only", ll_art), ("counterfactual", ll_cf)):
                acc, mass = _region_acc(CM.posterior(ll), region)
                cells.add({"region": region, "evidence": ev}, correct=acc, mass=mass)
            X.append(np.bincount([t for a in arts for t in a["evidence"]], minlength=3) / (2 * CM.N_EVID))
            y.append(region)
    surf = held_out_classifier(np.array(X), np.array(y), r, metric="l2")
    return {"rows": cells.rows(), "surface_acc": surf}


def reduce_A05(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A05"]
    v = start(card, ctx, "Honest warning, sincere fanatic, strategic propagandist and neutral report are regions of one factorial, separated by counterfactual evidence (correction, private action) and not by the artifact alone.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    acc = {ev: mean_of(rows, "correct", lambda r, ev=ev: r["evidence"] == ev) for ev in ("artifact_only", "counterfactual")}
    by_region = {reg: mean_of(rows, "correct", lambda r, reg=reg: r["region"] == reg and r["evidence"] == "counterfactual") for reg in REGION_NAMES}
    surf = float(np.nanmean([u["surface_acc"] for u in units]))
    passed = bool(acc["counterfactual"] >= cr["min_separated"] and acc["artifact_only"] <= cr["max_artifact_only"] + 0.2)
    gr = G.GateReport()
    extra_gate(gr, "surface_collision", "artifact_does_not_separate_regions", acc["artifact_only"], cr["max_artifact_only"] + 0.2, "max", "region accuracy from the artifact alone")
    battery(gr, live={"observed": acc["counterfactual"] - acc["artifact_only"], "min": 0.1, "name": "counterfactual_evidence_separates"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_artifacts_both_evidence_levels"},
            positive={"observed": acc["counterfactual"], "expected": 1.0, "tol": 1.0 - cr["min_separated"], "name": "regions_separated_with_counterfactuals"},
            surface={"accuracy": surf, "chance": 0.25, "tol": 0.3, "name": "evidence_polarity_histogram_weak"},
            oracle={"observed": acc["counterfactual"] - 0.25, "min": 0.2, "name": "regions_identifiable"},
            prediction={"gain": acc["counterfactual"] - acc["artifact_only"], "min": 0.0, "name": "counterfactuals_predict_region"},
            calibration={"observed": acc["artifact_only"], "reference": acc["counterfactual"], "direction": "down", "tol": 0.0, "name": "artifact_only_below_counterfactual"})
    criterion(v, "A05", passed, accuracy=acc, by_region=by_region, surface=surf)
    receipt(v, rows, card, ctx)
    narrative(v, f"From the artifact alone the four regions were named {acc['artifact_only']:.0%} of the time; with the correction and private-action probes {acc['counterfactual']:.0%} (chance 25%).",
              "Region names are labels for what a source would do when contradicted or unobserved, not templates in its artifacts.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


def unit_A06(ctx):
    r = rng(ctx, "a06")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(3, sizes(ctx)["makers"] // 8)
    for i in range(n):
        for region in ("sincere_fanatic", "strategic_propagandist"):
            s = CM.source(r, region, intensity="high")
            arts = [CM.speak(dict(s, policy="cherry_pick"), r) for _ in range(2)]        # matched artifact: both cherry-pick with high intensity
            ll_art = sum(CM.loglik_artifact(a) + CM.loglik_appraisal_cue(a, "intensity") for a in arts)
            ll_c = CM.loglik_correction(CM.correction_event(s, r))
            ll_p = CM.loglik_private(CM.private_action(s, r))
            for ev, ll in (("artifact_only", ll_art), ("plus_correction", ll_art + ll_c), ("plus_private", ll_art + ll_p), ("both", ll_art + ll_c + ll_p)):
                post = CM.posterior(ll)
                rp = CM.region_posterior(post)
                pair = {"sincere_fanatic": rp["sincere_fanatic"], "strategic_propagandist": rp["strategic_propagandist"]}
                tot = sum(pair.values())
                pf = pair[region] / tot if tot > 0 else 0.5
                correct = 0.5 if abs(pf - 0.5) < 1e-9 else float(pf > 0.5)      # an exact tie is chance, not an error
                cells.add({"evidence": ev}, correct=correct, p_true=pf, max_pair=float(max(pair.values()) / tot) if tot > 0 else 0.5)
    return {"rows": cells.rows()}


def reduce_A06(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A06"]
    v = start(card, ctx, "A sincere fanatic and a strategic propagandist with the same artifact and the same intended effect are told apart only by what they do when contradicted and when unobserved; absent those, the reader abstains.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    acc = {ev: mean_of(rows, "correct", lambda r, ev=ev: r["evidence"] == ev) for ev in ("artifact_only", "plus_correction", "plus_private", "both")}
    mp = {ev: mean_of(rows, "max_pair", lambda r, ev=ev: r["evidence"] == ev) for ev in ("artifact_only", "plus_correction", "plus_private", "both")}
    passed = bool(acc["artifact_only"] <= cr["max_artifact_only"] and acc["both"] >= cr["min_with_probes"] and mp["artifact_only"] <= cr["max_abstain_mass"])
    gr = G.GateReport()
    extra_gate(gr, "surface_collision", "artifact_at_chance_for_the_pair", acc["artifact_only"], cr["max_artifact_only"], "max", "pairwise accuracy from the matched artifact alone")
    battery(gr, live={"observed": acc["both"] - acc["artifact_only"], "min": 0.2, "name": "probes_separate_the_pair"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_matched_artifacts_every_evidence_level"},
            positive={"observed": acc["both"], "expected": 1.0, "tol": 1.0 - cr["min_with_probes"], "name": "both_probes_separate"},
            surface={"accuracy": acc["artifact_only"], "chance": 0.5, "tol": cr["max_artifact_only"] - 0.5, "name": "matched_surface_at_chance"},
            oracle={"observed": acc["both"] - 0.5, "min": 0.2, "name": "pair_identifiable_with_counterfactuals"},
            prediction={"gain": max(acc["plus_correction"], acc["plus_private"]) - 0.5, "min": 0.0, "name": "one_probe_already_helps"},
            calibration={"observed": mp["artifact_only"], "reference": cr["max_abstain_mass"], "direction": "down", "tol": 0.0, "name": "abstains_without_a_discriminator"})
    criterion(v, "A06", passed, accuracy=acc, max_pair_mass=mp)
    receipt(v, rows, card, ctx)
    narrative(v, f"From the matched artifact the fanatic and the propagandist were told apart {acc['artifact_only']:.0%} of the time with the reader's larger of the two masses at {mp['artifact_only']:.2f}; the correction probe alone gave {acc['plus_correction']:.0%}, the private-action probe {acc['plus_private']:.0%}, both {acc['both']:.0%}.",
              "The boundary between sincerity and strategy is a counterfactual, and the reader refuses to draw it from the artifact.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A07 — the inverse-inverse reader.
# --------------------------------------------------------------------------- #
def unit_A07(ctx):
    r = rng(ctx, "a07")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        s = CM.source(r, support="contradicts", appraisal="alarm", policy="full")
        truth_sup = CM.SUPPORT.index(s["support"])
        for maker in ("plain", "audience_modelling"):
            art = CM.speak(s, r) if maker == "plain" else CM.audience_modelling_speak(s, r)
            for reader in ("plain", "audience_aware"):
                # the plain reader takes evidence at face value (every source read as full selection); the
                # aware reader keeps the selection policies in its hypothesis space
                ll_r = CM.loglik_artifact(art, assume_full=(reader == "plain"))
                post = CM.posterior(ll_r)
                sm = CM.marginal(post, "support")
                cells.add({"maker": maker, "reader": reader}, ls=float(np.log(max(sm[truth_sup], 1e-12))), effect_conf=float(sm.max()))
    return {"rows": cells.rows()}


def reduce_A07(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A07"]
    v = start(card, ctx, "A reader that models the maker as modelling it gains only when the maker really selects evidence to steer it.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ls = {m: {k: mean_of(rows, "ls", lambda r, m=m, k=k: r["maker"] == m and r["reader"] == k) for k in ("plain", "audience_aware")} for m in ("plain", "audience_modelling")}
    gain_vs_modelling = ls["audience_modelling"]["audience_aware"] - ls["audience_modelling"]["plain"]
    gain_vs_plain = ls["plain"]["audience_aware"] - ls["plain"]["plain"]
    interaction = gain_vs_modelling - gain_vs_plain
    passed = bool(interaction >= cr["min_interaction"])
    gr = G.GateReport()
    battery(gr, live={"observed": interaction, "min": 0.0, "name": "maker_mechanism_moves_the_value_of_awareness"},
            placebo={"observed": max(0.0, gain_vs_plain), "tol": 0.05, "name": "awareness_buys_nothing_against_a_plain_maker"},
            positive={"observed": gain_vs_modelling, "expected": max(gain_vs_modelling, 0.0), "tol": 0.0, "name": "awareness_helps_against_a_modelling_maker"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_source_state_both_makers"},
            oracle={"observed": ls["plain"]["plain"] - np.log(1 / 3), "min": 0.0, "name": "belief_readable_from_a_plain_maker"},
            prediction={"gain": interaction, "min": cr["min_interaction"], "name": "interaction"},
            calibration={"observed": gain_vs_plain, "reference": gain_vs_modelling, "direction": "down", "tol": 0.0, "name": "awareness_gain_ordered_by_maker"})
    criterion(v, "A07", passed, interaction=interaction, gain_vs_modelling_maker=gain_vs_modelling, gain_vs_plain_maker=gain_vs_plain, scores=ls)
    receipt(v, rows, card, ctx)
    narrative(v, f"Against a maker that selects evidence to steer its audience, an audience-aware reader read the maker's belief {gain_vs_modelling:+.3f} nats better; against a plain maker {gain_vs_plain:+.3f}.",
              "Modelling the modeller pays exactly when there is a modeller.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A08 — awareness against suppression.
# --------------------------------------------------------------------------- #
def unit_A08(ctx):
    r = rng(ctx, "a08")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        for truth in ("true", "false"):
            # true content: an honest warning (a supporting pool shown in full); false content: a strategic
            # propagandist (a contradicting pool with fabricated support) - both from the population the aware prior describes
            s = CM.source(r, "honest_warning" if truth == "true" else "strategic_propagandist")
            arts = [CM.speak(s, r) for _ in range(2)]
            ll = sum(CM.loglik_artifact(a) for a in arts)
            # the aware reader knows the source's record: two contradictions met, one off-audience action
            ll_probe = ll + sum(CM.loglik_correction(CM.correction_event(s, r)) for _ in range(2)) + CM.loglik_private(CM.private_action(s, r))
            for reader in ("aware", "unaware", "suppress"):
                if reader == "aware":
                    post = CM.posterior(ll_probe, CM.region_prior())   # awareness: the source population is structured, so the probe bears on the content
                    p_support = float(CM.marginal(post, "support")[0])
                    rel = float(CM.marginal(post, "policy")[0])
                    up = CM.uptake(post, rel, p_support, gate="factored")["policy_uptake"]
                elif reader == "unaware":
                    post = CM.posterior(sum(CM.loglik_artifact(a, assume_full=True) for a in arts))
                    up = float(CM.marginal(post, "support")[0])         # face value
                else:
                    up = 0.0
                cells.add({"reader": reader, "truth": truth}, uptake=up)
    return {"rows": cells.rows()}


def reduce_A08(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A08"]
    v = start(card, ctx, "Influence awareness improves discrimination between true and false content while keeping uptake of the true; blanket suppression loses both.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    up = {k: {t: mean_of(rows, "uptake", lambda r, k=k, t=t: r["reader"] == k and r["truth"] == t) for t in ("true", "false")} for k in ("aware", "unaware", "suppress")}
    disc = {k: up[k]["true"] - up[k]["false"] for k in up}
    passed = bool(disc["aware"] >= cr["min_discrimination"] and up["aware"]["true"] >= cr["min_true_uptake"])
    gr = G.GateReport()
    battery(gr, live={"observed": disc["aware"] - disc["unaware"], "min": 0.0, "name": "awareness_improves_discrimination"},
            placebo={"observed": disc["suppress"], "tol": 0.0, "name": "suppression_discriminates_nothing"},
            positive={"observed": up["aware"]["true"], "expected": max(up["aware"]["true"], cr["min_true_uptake"]), "tol": 0.0, "name": "true_content_still_taken_up"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_artifacts_every_reader"},
            oracle={"observed": disc["aware"], "min": cr["min_discrimination"], "name": "discrimination_above_bar"},
            prediction={"gain": disc["aware"], "min": 0.0, "name": "selective_uptake"},
            calibration={"observed": up["aware"]["false"], "reference": up["unaware"]["false"], "direction": "down", "tol": 0.0, "name": "false_uptake_below_unaware"})
    criterion(v, "A08", passed, uptake=up, discrimination=disc)
    receipt(v, rows, card, ctx)
    narrative(v, f"The aware reader took up true content at {up['aware']['true']:.2f} and false at {up['aware']['false']:.2f}; the unaware reader {up['unaware']['true']:.2f} and {up['unaware']['false']:.2f}; suppression took up nothing.",
              "Recognizing influence is not the same as resisting everything.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A09 — habituation of the acute response with growing uptake.
# --------------------------------------------------------------------------- #
def unit_A09(ctx):
    r = rng(ctx, "a09")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        s = CM.source(r, "honest_warning")
        ll = np.zeros(CM.N_SOURCE_STATES)
        for k in range(1, 9):
            art = CM.speak(s, r)
            ll = ll + CM.loglik_artifact(art)
            if k % 2 == 0:
                ll = ll + CM.loglik_correction(CM.correction_event(s, r))     # over exposures the source's record accrues: every second one meets a contradiction
            if k in (1, 2, 4, 8):
                post = CM.posterior(ll, CM.region_prior())         # content is identifiable only against the population
                cells.add({"exposure": k}, acute=CM.habituated_response(k - 1) * CM.reader_response(art), belief=float(CM.marginal(post, "support")[0]))
    return {"rows": cells.rows()}


def reduce_A09(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A09"]
    v = start(card, ctx, "The reader's acute response to a repeated source habituates while its belief about the content keeps accumulating.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    ac = {k: mean_of(rows, "acute", lambda r, k=k: r["exposure"] == k) for k in (1, 2, 4, 8)}
    be = {k: mean_of(rows, "belief", lambda r, k=k: r["exposure"] == k) for k in (1, 2, 4, 8)}
    share = ac[8] / ac[1] if ac[1] > 1e-9 else 0.0
    passed = bool(share <= cr["max_habituated_share"] and be[8] >= be[1])
    gr = G.GateReport()
    battery(gr, live={"observed": ac[1] - ac[8], "min": 0.1, "name": "response_habituates"},
            placebo={"observed": 0.0, "tol": 0.0, "name": "same_source_throughout"},
            positive={"observed": be[8] - be[1], "expected": max(0.0, be[8] - be[1]), "tol": 0.0, "name": "belief_accumulates"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "intensity_constant"},
            oracle={"observed": be[8], "min": 0.5, "name": "content_readable_by_eight"},
            prediction={"gain": be[8] - be[1], "min": 0.0, "name": "slow_channel_rises"},
            calibration={"observed": ac[8], "reference": ac[1], "direction": "down", "tol": 0.0, "name": "fast_channel_falls"})
    criterion(v, "A09", passed, acute=ac, belief=be, habituated_share=share)
    receipt(v, rows, card, ctx)
    narrative(v, f"The acute response fell from {ac[1]:.2f} at the first exposure to {ac[8]:.2f} at the eighth while belief in the content rose from {be[1]:.2f} to {be[8]:.2f}.",
              "Feeling less and believing more are two trajectories, and the construction keeps them apart.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))


# --------------------------------------------------------------------------- #
# A10 — the uptake gate.
# --------------------------------------------------------------------------- #
def unit_A10(ctx):
    r = rng(ctx, "a10")
    cells = Cells(ctx["wid"], ctx["rep"])
    n = max(4, sizes(ctx)["makers"] // 4)
    for i in range(n):
        s = CM.source(r, support="supports")
        arts = [CM.speak(s, r) for _ in range(2)]
        post = CM.posterior(sum(CM.loglik_artifact(a) for a in arts))
        p_support = float(CM.marginal(post, "support")[0])
        goal = CM.marginal(post, "policy")
        for reliability in ("reliable", "unreliable"):
            rel = 0.9 if reliability == "reliable" else 0.2
            for gate in ("factored", "scalar", "suppress"):
                up = CM.uptake(post, rel, p_support, gate=gate)
                cells.add({"gate": gate, "reliability": reliability}, policy=up["policy_uptake"], belief=up["belief_update"], goal_js=C.js(goal, np.array(up["goal_posterior"])[:3] if len(up["goal_posterior"]) >= 3 else goal) if False else 0.0)
    return {"rows": cells.rows()}


def reduce_A10(card, units, ctx):
    from ghostscale.prereg_v14 import CRITERIA
    cr = CRITERIA["A10"]
    v = start(card, ctx, "Factored trust gates what the reader adopts as policy after it has reconstructed the source, without moving what it believes about the content or what it inferred the source wanted.", "CONSTRUCTED_MECHANISM")
    rows = [r for u in units for r in u["rows"]]
    pol = {g: {rl: mean_of(rows, "policy", lambda r, g=g, rl=rl: r["gate"] == g and r["reliability"] == rl) for rl in ("reliable", "unreliable")} for g in ("factored", "scalar", "suppress")}
    bel = {g: {rl: mean_of(rows, "belief", lambda r, g=g, rl=rl: r["gate"] == g and r["reliability"] == rl) for rl in ("reliable", "unreliable")} for g in ("factored", "scalar", "suppress")}
    gate_effect = pol["factored"]["reliable"] - pol["factored"]["unreliable"]
    side = abs(bel["factored"]["reliable"] - bel["factored"]["unreliable"])
    scalar_side = abs(bel["scalar"]["reliable"] - bel["scalar"]["unreliable"])
    passed = bool(gate_effect >= cr["min_gate_effect"] and side <= cr["max_side_effect"])
    gr = G.GateReport()
    battery(gr, live={"observed": gate_effect, "min": cr["min_gate_effect"], "name": "reliability_moves_policy_uptake"},
            placebo={"observed": side, "tol": cr["max_side_effect"], "name": "reliability_leaves_content_belief"},
            positive={"observed": scalar_side, "expected": max(scalar_side, 0.1), "tol": 0.0, "name": "scalar_gate_moves_belief_the_planted_failure"},
            surface={"accuracy": 0.0, "chance": 0.0, "tol": 0.0, "name": "same_posterior_every_gate"},
            oracle={"observed": pol["factored"]["reliable"], "min": 0.3, "name": "reliable_source_adopted"},
            prediction={"gain": gate_effect, "min": 0.0, "name": "policy_tracks_reliability"},
            calibration={"observed": pol["suppress"]["reliable"], "reference": 0.0, "direction": "down", "tol": 0.0, "name": "suppression_adopts_nothing"})
    criterion(v, "A10", passed, gate_effect=gate_effect, belief_side_effect=side, scalar_belief_side_effect=scalar_side, policy=pol, belief=bel)
    receipt(v, rows, card, ctx)
    narrative(v, f"Under the factored gate policy uptake moved {gate_effect:+.2f} between a reliable and an unreliable source while content belief moved {side:.3f}; the scalar gate moved belief {scalar_side:.2f}.",
              "Trust is a gate on action, downstream of reconstruction, and it should not reach back into the evidence.")
    return finish(card, v, gr, __file__, decide_state(gr), ctx, pursuit=pursuit_of(passed))
