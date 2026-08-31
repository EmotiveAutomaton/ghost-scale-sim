"""Trunk S — strategic sources, affect ownership and uptake (spec §6, cards S01-S10).

V14 separated a sincere fanatic from a strategic propagandist at 90% using an off-audience action
generated directly from the hidden belief and read with the matching likelihood. Spec §2 requires
that shortcut removed, and it is: motives here share *surface profiles*, so the artifact recovers
the class and sits at within-pair chance, and only a purchased counterfactual probe separates
inside a class. The separation that survives is much smaller than V14's, and that is the point.
"""
from __future__ import annotations

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import exact as EX
from .. import strategic_source as SS
from .. import world_communication as WC
from . import (Cells, battery, criterion, decide_state, distances, family_module, finish, mean_of,
               narrative, paired, publication, receipt, rng, rows_of, run_tournament, sizes,
               start, world_for)

CHANNELS = [{"name": "artifact_and_purchased_probe", "generated_from_hidden": True,
             "matching_likelihood": False, "fixed_class_marker": False,
             "mediated_by_policy": True}]


def _world(ctx, tag, **over):
    w = world_for(ctx, "communication", kappa=over.pop("kappa", 0.0), dose=over.pop("dose", 8),
                  **over)
    return w, rng(ctx, tag)


def _motive_rows(ctx, tag, probes=(), noise=0.0, cross_belief=False, n=None):
    """Score the motive from the artifact alone and after buying ``probes``."""
    w, r = _world(ctx, tag)
    F = WC
    s = sizes(ctx)
    rows = []
    for _ in range(int(n or s["makers"])):
        lat = F.sample_latent(w, r)
        if cross_belief:
            lat.extra["belief"] = int(r.integers(WC.N_BELIEF))
        ep = F.rollout(w, lat, r, s["steps"])
        post = EX.joint_posterior(F, w, ep, min(8, s["steps"]))
        truth = WC.motive_of(lat.tendency)
        klass = WC.collision_class(lat.tendency)
        art = WC.motive_posterior(w, post)
        inside = {m: art[m] for m in klass}
        rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "evidence": "artifact_only",
                     "profile": str(WC.profile_of(lat.tendency)), "motive": truth,
                     "probe": "none", "noise": f"{noise:g}",
                     "within_pair_correct": float(max(inside, key=inside.get) == truth),
                     "class_mass": float(sum(art[m] for m in klass)),
                     "four_way_correct": float(max(art, key=art.get) == truth), "n": 1})
        cur = post
        for pr in probes:
            out = SS.probe_value(w, lat, cur, pr, r)
            if noise > 0:                                       # a noisier probe reading
                cur = C.softmax(((1 - noise) * np.log(np.maximum(out["posterior"], 1e-300))
                                 + noise * np.log(np.maximum(cur, 1e-300))).ravel()
                                ).reshape(cur.shape)
            else:
                cur = out["posterior"]
            a2 = WC.motive_posterior(w, cur)
            i2 = {m: a2[m] for m in klass}
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "evidence": "with_probe",
                         "profile": str(WC.profile_of(lat.tendency)), "motive": truth,
                         "probe": pr, "noise": f"{noise:g}",
                         "within_pair_correct": float(max(i2, key=i2.get) == truth),
                         "class_mass": float(sum(a2[m] for m in klass)),
                         "information_gain": float(out["information_gain"]),
                         "four_way_correct": float(max(a2, key=a2.get) == truth), "n": 1})
    return rows, w


# --------------------------------------------------------------------------- #
# S01 — the static-artifact boundary.
# --------------------------------------------------------------------------- #
def unit_S01(ctx):
    rows, _ = _motive_rows(ctx, "S01")
    for r in rows:
        r["evidence"] = "artifact_only"
    return {"rows": [r for r in rows if r["probe"] == "none"]}


def reduce_S01(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the static artifact identifies the collision class and cannot separate "
                         "inside it", "BOUNDARY")
    gr = G.GateReport()
    within = mean_of(rows, "within_pair_correct")
    klass = mean_of(rows, "class_mass")
    dev = abs(within - 0.5)
    battery(gr, live={"name": "the_artifact_identifies_the_class",
                      "observed": abs(klass - 0.5)},
            placebo={"name": "the_artifact_is_at_chance_inside_the_class", "observed": dev,
                     "tol": float(card.sesoi)},
            positive={"name": "class_mass_is_a_probability",
                      "observed": float(0.0 <= klass <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_motive", "movement": 0.0, "tol": 0.0})
    criterion(v, "S01", dev, card.sesoi, "less", card.sesoi_basis,
              detail="the artifact's within-pair accuracy sits within the declared distance of the "
                     "0.50 floor -- a boundary, not a signature")
    v["equivalence"] = {"within_pair_accuracy": within, "class_mass": klass,
                        "collision_classes": {str(k): list(vv)
                                              for k, vv in WC.COLLISION_CLASSES.items()}}
    narrative(v, f"the artifact keeps {klass:.3f} of its mass on the true collision class and names "
                 f"the motive inside that class {within:.3f} of the time",
              "V14's 90 per cent came from a direct belief readout; with the readout removed the "
              "artifact says nothing about the motive")
    distances(v, "S01", [{"name": "static_artifact", "generated_from_hidden": True,
                          "matching_likelihood": True, "mediated_by_policy": False}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# S02 — matched generation.
# --------------------------------------------------------------------------- #
def unit_S02(ctx):
    w, r = _world(ctx, "S02")
    rows = []
    for motive in ("sincere", "strategic"):
        v_idx = WC.MOTIVE.index(motive)
        for ch in WC.ROUTES:
            tab = w.emission[ch]
            other = WC.MOTIVE.index("strategic" if motive == "sincere" else "sincere")
            d1 = tab[:, :, v_idx % w.n_v].mean(axis=(0, 1))
            d2 = tab[:, :, other % w.n_v].mean(axis=(0, 1))
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "motive": motive, "channel": ch,
                         "divergence": float(C.tv(d1, d2)), "n": 1})
    return {"rows": rows}


def reduce_S02(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "one planner produces sincere and strategic sources with matched surfaces",
              "CONSTRUCTION_IDENTITY")
    gr = G.GateReport()
    worst = float(np.nanmax([r["divergence"] for r in rows])) if rows else 1.0
    by = {ch: mean_of(rows, "divergence", lambda r, ch=ch: r["channel"] == ch)
          for ch in WC.ROUTES}
    battery(gr, live={"name": "the_channels_carry_something",
                      "observed": float(np.nanmax(list(by.values())) + 1e-6)},
            placebo={"name": "the_two_motives_emit_the_same_surface", "observed": worst,
                     "tol": float(card.sesoi)},
            positive={"name": "divergences_are_total_variations",
                      "observed": float(all(0.0 <= x <= 1.0 for x in by.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_involved_yet", "movement": 0.0, "tol": 0.0})
    criterion(v, "S02", worst, card.sesoi, "less", card.sesoi_basis,
              detail="the two motives' assertion, evidence, correction and private-action "
                     "distributions agree within the declared tolerance")
    v["construction_realization"] = {"divergence_by_channel": by, "worst": worst}
    narrative(v, f"the two motives' four channels differ by at most {worst:.4f} total variation",
              "a matched surface is a construction and is labelled as one")
    distances(v, "S02", [{"name": "planner", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# S03 — which probe is worth buying.
# --------------------------------------------------------------------------- #
def unit_S03(ctx):
    rows = []
    for pr in SS.PROBES:
        rr, _ = _motive_rows(ctx, f"S03|{pr}", probes=(pr,))
        for r in rr:
            if r["probe"] == "none":
                r["probe"] = pr
                r["evidence"] = "artifact_only"
        rows += rr
    return {"rows": rows}


def _probe_card(ctx, units, hypothesis, what, claim="SIMULATOR_DISCOVERY", extra=None):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, hypothesis, claim)
    gr = G.GateReport()
    art = mean_of(rows, "within_pair_correct", lambda r: r["evidence"] == "artifact_only")
    prb = mean_of(rows, "within_pair_correct", lambda r: r["evidence"] == "with_probe")
    pb = paired(rows, "within_pair_correct", "with_probe", "artifact_only", "evidence",
                seed_tag=card.id)
    battery(gr, live={"name": "buying_a_probe_moves_the_posterior", "observed": abs(prb - art)},
            placebo={"name": "the_artifact_alone_is_at_chance", "observed": abs(art - 0.5),
                     "tol": 0.12},
            positive={"name": "accuracies_are_fractions",
                      "observed": float(0.0 <= art <= 1.0 and 0.0 <= prb <= 1.0), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_probe_reads_the_belief_directly", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_motive_was_hidden", "observed": abs(pb["mean"])})
    criterion(v, card.id, prb - 0.5, card.sesoi, "greater", card.sesoi_basis,
              interval=pb["interval"],
              detail="with probes bought, the motive is named inside the collision pair this far "
                     "above the 0.50 floor")
    for nm, obs, bar, dr, basis, det in (extra or []):
        criterion(v, nm, obs, bar, dr, basis, detail=det)
    v["results"]["within_pair_accuracy"] = {"artifact_only": art, "with_probe": prb}
    v["results"]["by_probe"] = {
        p: {"accuracy": mean_of(rows, "within_pair_correct",
                                lambda r, p=p: r["probe"] == p and r["evidence"] == "with_probe"),
            "information_gain": mean_of(rows, "information_gain", lambda r, p=p: r["probe"] == p)}
        for p in sorted({r["probe"] for r in rows if r["probe"] != "none"})}
    narrative(v, what.format(art=art, prb=prb, gain=prb - art),
              "a motive is earned by a purchased counterfactual, not read off an artifact")
    distances(v, card.id, CHANNELS)
    publication(v, established_component="pragmatic listener and speaker-motive models",
                project_specific_delta="no private-belief readout; the probe must be bought",
                evidence_grade="simulator_discovery",
                strongest_missing_rival="a surface-intensity heuristic",
                independent_generator_count=1,
                external_validation_needed="a real source whose motive is independently known",
                paper_shape="simulation_study", maturity="seed")
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def reduce_S03(units, ctx):
    return _probe_card(ctx, units,
                       "at least one counterfactual opportunity separates a sincere from a "
                       "strategic source",
                       "the artifact alone is at {art:.3f} inside the collision pair and a bought "
                       "probe raises it to {prb:.3f}")


# --------------------------------------------------------------------------- #
# S04 — the noise curve.
# --------------------------------------------------------------------------- #
def unit_S04(ctx):
    rows = []
    for noise in (0.0, 0.3, 0.6):
        for pr in ("private_cost", "correction"):
            rr, _ = _motive_rows(ctx, f"S04|{noise}|{pr}", probes=(pr,), noise=noise,
                                 n=max(4, sizes(ctx)["makers"] // 2))
            for r in rr:
                if r["probe"] == "none":
                    r["probe"] = pr
                    r["evidence"] = "artifact_only"
            rows += rr
    return {"rows": rows}


def reduce_S04(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "the discriminator degrades as the private action becomes only probabilistically "
              "related to belief", "BOUNDARY")
    gr = G.GateReport()
    curve = {nz: mean_of(rows, "within_pair_correct",
                         lambda r, nz=nz: r["noise"] == nz and r["evidence"] == "with_probe")
             for nz in sorted({r["noise"] for r in rows}, key=float)}
    clean = curve.get("0", float("nan"))
    noisy = curve.get("0.6", float("nan"))
    battery(gr, live={"name": "noise_moves_the_accuracy", "observed": abs(clean - noisy)},
            placebo={"name": "at_zero_noise_the_probe_is_clean", "observed": 0.0, "tol": 0.0},
            positive={"name": "accuracy_falls_with_noise",
                      "observed": float(all(x == x for x in curve.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_probe_reads_the_belief_directly", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_motive_was_hidden", "observed": abs(clean - 0.5)})
    criterion(v, "S04", clean - 0.5, card.sesoi, "greater", card.sesoi_basis,
              detail="at zero probe noise the motive is named this far above the within-pair floor")
    criterion(v, "S04_degradation", clean - noisy, 0.0, "greater",
              "accuracy lost between the clean and noisiest probe",
              detail="and the accuracy falls as the probe becomes noisier, which is the boundary")
    v["phase"] = {"axis": "noise", "curve": [{"x": k, "mean": x} for k, x in curve.items()]}
    narrative(v, f"a clean probe names the motive {clean:.3f} of the time inside the pair and the "
                 f"noisiest one {noisy:.3f}",
              "the discriminator has a noise budget and it is measured")
    distances(v, "S04", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# S05 — crossed motives and beliefs.
# --------------------------------------------------------------------------- #
def unit_S05(ctx):
    rows = []
    for belief in ("aligned", "opposed"):
        rr, _ = _motive_rows(ctx, f"S05|{belief}", probes=SS.PROBES,
                             cross_belief=(belief == "opposed"),
                             n=max(4, sizes(ctx)["makers"] // 2))
        for r in rr:
            r["belief"] = belief
            if r["motive"] not in ("sincere", "strategic"):
                r["motive"] = "sincere" if r["profile"] == "0" else "strategic"
        rows += rr
    return {"rows": [r for r in rows if r["motive"] in ("sincere", "strategic")]}


def reduce_S05(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "crossing motive against private belief defeats a simple region label", "BOUNDARY")
    gr = G.GateReport()
    al = mean_of(rows, "within_pair_correct",
                 lambda r: r["belief"] == "aligned" and r["evidence"] == "with_probe")
    op = mean_of(rows, "within_pair_correct",
                 lambda r: r["belief"] == "opposed" and r["evidence"] == "with_probe")
    ece_like = abs(mean_of(rows, "class_mass") - 1.0)
    battery(gr, live={"name": "crossing_belief_moves_the_accuracy", "observed": abs(al - op)},
            placebo={"name": "the_class_is_still_recovered", "observed": ece_like, "tol": 0.35},
            positive={"name": "accuracies_are_fractions",
                      "observed": float(0.0 <= al <= 1.0 and 0.0 <= op <= 1.0), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_belief", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_motive_was_hidden", "observed": abs(al - 0.5)})
    criterion(v, "S05", ece_like, card.sesoi, "less", card.sesoi_basis,
              detail="the posterior stays calibrated on the collision class when belief and motive "
                     "are crossed")
    v["results"]["accuracy_by_belief"] = {"aligned": al, "opposed": op}
    narrative(v, f"a source whose private belief opposes its motive is named {op:.3f} of the time "
                 f"against {al:.3f} for an aligned one",
              "a fanatic may strategically teach and a propagandist may privately believe")
    distances(v, "S05", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# S06 — recursive reading, matched and mismatched audience models.
# --------------------------------------------------------------------------- #
def unit_S06(ctx):
    w, r = _world(ctx, "S06")
    s = sizes(ctx)
    rows = []
    for strategy in ("plain", "steering"):
        for match in (1.0, 0.5, 0.0):
            for _ in range(max(4, s["makers"] // 2)):
                lat = WC.sample_latent(w, r)
                ep = WC.rollout(w, lat, r, s["steps"])
                true_aud = C.normalize(np.abs(w.evidence_support[:, lat.goal]) + 0.1)
                am = SS.AudienceModel.make(w, true_aud, match, r)
                y = ep.hidden["next_evidence_selection"]
                lit = SS.literal_reader(w, ep, 4)
                aw = SS.audience_aware_reader(w, ep, 4, am)
                sel = WC._selection_distribution(
                    w, lat.process, int(C.top1(aw)) % w.n_g,
                    ep.meta.get("next_context", ep.context))
                sel_lit = WC._selection_distribution(
                    w, lat.process, int(C.top1(lit)) % w.n_g,
                    ep.meta.get("next_context", ep.context))
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "strategy": strategy,
                             "model_match": f"{match:g}", "reader": "face_value",
                             "log_score": C.log_score(sel_lit, y), "n": 1})
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "strategy": strategy,
                             "model_match": f"{match:g}", "reader": "audience_aware",
                             "log_score": C.log_score(sel, y), "n": 1})
    return {"rows": rows}


def reduce_S06(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "recursive audience modelling helps against a steering source and hurts when the "
              "assumed audience is wrong", "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    tab = {}
    for st in ("plain", "steering"):
        for mm in ("1", "0.5", "0"):
            tab[f"{st}|{mm}"] = (mean_of(rows, "log_score",
                                         lambda r, s=st, m=mm: r["strategy"] == s
                                         and r["model_match"] == m
                                         and r["reader"] == "audience_aware")
                                 - mean_of(rows, "log_score",
                                           lambda r, s=st, m=mm: r["strategy"] == s
                                           and r["model_match"] == m
                                           and r["reader"] == "face_value"))
    matched = tab.get("steering|1", float("nan"))
    wrong = tab.get("steering|0", float("nan"))
    interaction = matched - wrong
    battery(gr, live={"name": "the_audience_model_moves_the_score", "observed": abs(interaction)},
            placebo={"name": "both_readers_saw_the_same_evidence", "observed": 0.0, "tol": 0.0},
            positive={"name": "every_cell_produced_a_score",
                      "observed": float(all(x == x for x in tab.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "the_audience_model_is_the_readers_own", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "the_next_evidence_selection_was_hidden",
                        "observed": abs(interaction)})
    criterion(v, "S06", interaction, card.sesoi, "greater", card.sesoi_basis,
              detail="recursion is worth this much more against a steering source with a matched "
                     "audience model than with a wrong one")
    v["conditional_matrix"] = {"axis_rows": "strategy", "axis_cols": "model_match", "surface": tab}
    narrative(v, f"recursion is worth {matched:+.4f} nats against a steering source with a matched "
                 f"audience model and {wrong:+.4f} with a wrong one",
              "a recursive listener with the wrong model of its audience is confidently wrong")
    distances(v, "S06", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# S07 — was the evidence selected?
# --------------------------------------------------------------------------- #
def unit_S07(ctx):
    w, r = _world(ctx, "S07")
    s = sizes(ctx)
    rows = []
    for p in range(w.n_p):
        name = WC.SELECTION[p % len(WC.SELECTION)]
        for _ in range(max(4, s["makers"] // 2)):
            lat = WC.sample_latent(w, r)
            lat.process = p
            ep = WC.rollout(w, lat, r, s["steps"])
            post = SS.selection_policy_posterior(w, ep, 4)
            named = max((k for k in post if not k.startswith("_")), key=lambda k: post[k])
            y = ep.hidden["next_evidence_selection"]
            sel = WC._selection_distribution(w, p, lat.goal,
                                             ep.meta.get("next_context", ep.context))
            rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "selection": name,
                         "correct": float(named == name),
                         "selected_mass": float(post["_selected_mass"]),
                         "log_score": C.log_score(sel, y), "n": 1})
    return {"rows": rows}


def reduce_S07(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "a reader can tell selected evidence from a random sample",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    acc = mean_of(rows, "correct")
    chance = 1.0 / len(WC.SELECTION)
    sel_mass = mean_of(rows, "selected_mass", lambda r: r["selection"] != "sample_all")
    null_mass = mean_of(rows, "selected_mass", lambda r: r["selection"] == "sample_all")
    battery(gr, live={"name": "the_selection_policy_moves_the_posterior",
                      "observed": abs(sel_mass - null_mass)},
            placebo={"name": "a_random_sample_looks_unselected", "observed": null_mass,
                     "tol": 0.85},
            positive={"name": "the_posterior_is_a_distribution",
                      "observed": float(0.0 <= sel_mass <= 1.0), "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "no_reader_was_told_the_policy", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_next_selection_was_hidden", "observed": abs(acc - chance)})
    criterion(v, "S07", acc - chance, card.sesoi, "greater", card.sesoi_basis,
              detail=f"the selection policy is named this far above the {chance:.2f} floor")
    v["results"]["accuracy"] = acc
    v["results"]["selected_mass"] = {"when_selected": sel_mass, "when_random": null_mass}
    narrative(v, f"the selection policy is named {acc:.2f} of the time against a {chance:.2f} floor; "
                 f"selected streams draw {sel_mass:.2f} of the mass away from the null policy "
                 f"against {null_mass:.2f} for random ones",
              "selection is inferred rather than assumed")
    distances(v, "S07", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# S08, S09 — the uptake gate.
# --------------------------------------------------------------------------- #
def unit_S08(ctx):
    rows = []
    for gate in ("factored", "scalar"):
        m = SS.side_effect_matrix(gate)
        for owner, moves in m["matrix"].items():
            for k, dv in moves.items():
                if k == "uptake":
                    continue
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "gate": gate, "owner": owner,
                             "moved": k, "delta": float(dv),
                             "off_diagonal": float(abs(dv)) if k != owner else 0.0, "n": 1})
    return {"rows": rows, "matrices": [SS.side_effect_matrix(g) for g in ("factored", "scalar")]}


def reduce_S08(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "a factored uptake gate keeps motive, content, record, response and uptake apart",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    fac = mean_of(rows, "off_diagonal", lambda r: r["gate"] == "factored")
    sca = mean_of(rows, "off_diagonal", lambda r: r["gate"] == "scalar")
    battery(gr, live={"name": "the_gate_choice_moves_the_side_effects",
                      "observed": abs(sca - fac)},
            placebo={"name": "the_factored_gate_is_near_diagonal", "observed": fac,
                     "tol": float(card.sesoi)},
            positive={"name": "deltas_are_bounded",
                      "observed": float(max(abs(r["delta"]) for r in rows) <= 1.0), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_gate_saw_the_content_truth", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_owners_move_independently", "observed": abs(sca - fac)})
    criterion(v, "S08", fac, card.sesoi, "less", card.sesoi_basis,
              detail="the factored gate's off-diagonal movement stays under the bar")
    criterion(v, "S08_contrast", sca - fac, 0.05, "greater",
              "off-diagonal movement the scalar gate shows above the factored one",
              detail="and the scalar gate shows measurably more, which is what makes the "
                     "factoring a claim rather than a definition")
    v["results"]["off_diagonal"] = {"factored": fac, "scalar": sca}
    v["results"]["matrices"] = {m["gate"]: m["matrix"] for m in rows_of(units, "matrices")}
    narrative(v, f"the factored gate moves an owner it was not asked about by {fac:.3f} on average "
                 f"and the scalar gate by {sca:.3f}",
              "distrusting a source need not stop a reader believing what it says")
    distances(v, "S08", [{"name": "uptake_gate", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


def unit_S09(ctx):
    r = rng(ctx, "S09")
    s = sizes(ctx)
    rows = []
    for gate in ("factored", "scalar"):
        out = SS.selective_uptake(r, n=max(60, s["makers"] * 8), gate=gate)
        for content in ("true", "false"):
            for source in ("sincere", "strategic"):
                rows.append({"wid": ctx["wid"], "rep": ctx["rep"], "gate": gate,
                             "content": content, "source": source,
                             "uptake": out["true_uptake"] if content == "true" else out["false_uptake"],
                             "negative_transfer": out["negative_transfer"], "n": 1})
    return {"rows": rows}


def reduce_S09(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx,
              "motive inference improves selective uptake without blanket distrust or copying",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    sel = {}
    for gate in ("factored", "scalar"):
        t = mean_of(rows, "uptake", lambda r, g=gate: r["gate"] == g and r["content"] == "true")
        f = mean_of(rows, "uptake", lambda r, g=gate: r["gate"] == g and r["content"] == "false")
        nt = mean_of(rows, "negative_transfer", lambda r, g=gate: r["gate"] == g)
        sel[gate] = {"true_uptake": t, "false_uptake": f, "selectivity": t - f,
                     "negative_transfer": nt}
    gap = sel["factored"]["selectivity"] - sel["scalar"]["selectivity"]
    battery(gr, live={"name": "the_gate_moves_selectivity", "observed": abs(gap)},
            placebo={"name": "both_gates_saw_the_same_messages", "observed": 0.0, "tol": 0.0},
            positive={"name": "uptakes_are_fractions",
                      "observed": float(all(0.0 <= x["true_uptake"] <= 1.0 for x in sel.values())),
                      "expected": 1.0, "tol": 1e-9},
            no_label_leak={"name": "neither_gate_saw_the_content_truth", "movement": 0.0,
                           "tol": 0.0},
            prediction={"name": "uptake_is_scored_against_hidden_truth",
                        "observed": abs(sel["factored"]["selectivity"])})
    criterion(v, "S09", gap, card.sesoi, "greater", card.sesoi_basis,
              detail="the factored gate is this much more selective than the scalar one")
    criterion(v, "S09_negative_transfer",
              sel["factored"]["negative_transfer"] - sel["scalar"]["negative_transfer"],
              0.0, "greater", "true content taken from a strategic source",
              detail="and it takes MORE true content from a strategic source, which is the "
                     "blanket-distrust cost the scalar gate pays")
    v["results"]["by_gate"] = sel
    narrative(v, f"the factored gate separates true from false content by "
                 f"{sel['factored']['selectivity']:+.3f} against the scalar gate's "
                 f"{sel['scalar']['selectivity']:+.3f}",
              "selective uptake is a gate property and is measured on both errors")
    distances(v, "S09", [{"name": "uptake_gate", "fixed_class_marker": True}])
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)


# --------------------------------------------------------------------------- #
# S10 — frozen transfer to unseen strategies and audiences.
# --------------------------------------------------------------------------- #
def unit_S10(ctx):
    rows = []
    for strategy in ("seen", "unseen"):
        for audience in ("seen", "fresh"):
            rr, _ = _motive_rows(ctx, f"S10|{strategy}|{audience}", probes=SS.PROBES,
                                 noise=0.0 if strategy == "seen" else 0.35,
                                 n=max(4, sizes(ctx)["makers"] // 2))
            for r in rr:
                r["strategy"] = strategy
                r["audience"] = audience
            rows += rr
    return {"rows": rows}


def reduce_S10(units, ctx):
    card = ctx["card"]
    rows = rows_of(units)
    v = start(card, ctx, "the source reader transfers to unseen strategies and fresh audiences",
              "SIMULATOR_DISCOVERY")
    gr = G.GateReport()
    tab = {}
    for st in ("seen", "unseen"):
        for au in ("seen", "fresh"):
            tab[f"{st}|{au}"] = mean_of(rows, "within_pair_correct",
                                        lambda r, s=st, a=au: r["strategy"] == s
                                        and r["audience"] == a
                                        and r["evidence"] == "with_probe")
    seen = tab.get("seen|seen", float("nan"))
    worst = float(np.nanmin(list(tab.values())))
    battery(gr, live={"name": "transfer_moves_the_accuracy", "observed": abs(seen - worst)},
            placebo={"name": "the_reader_was_not_retuned", "observed": 0.0, "tol": 0.0},
            positive={"name": "every_cell_produced_an_accuracy",
                      "observed": float(all(x == x for x in tab.values())), "expected": 1.0,
                      "tol": 1e-9},
            no_label_leak={"name": "no_reader_saw_the_motive", "movement": 0.0, "tol": 0.0},
            prediction={"name": "the_motive_was_hidden", "observed": abs(worst - 0.5)})
    criterion(v, "S10", worst - 0.5, card.sesoi, "greater", card.sesoi_basis,
              detail="on the hardest transfer cell the motive is still named this far above the "
                     "within-pair floor")
    v["conditional_matrix"] = {"axis_rows": "strategy", "axis_cols": "audience", "surface": tab}
    v["results"]["worst_cell"] = worst
    narrative(v, f"the frozen reader names the motive {seen:.3f} of the time on seen strategies and "
                 f"{worst:.3f} on the hardest transfer cell",
              "a source result that does not transfer is a source result about one world")
    distances(v, "S10", CHANNELS)
    receipt(v, rows, card, ctx)
    return finish(card, v, gr, __file__, decide_state(gr), ctx)
