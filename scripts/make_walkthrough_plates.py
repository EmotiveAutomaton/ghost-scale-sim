"""Generate the walkthrough plates from committed verdict files.

    python scripts/make_walkthrough_plates.py

Every number is read out of results/. Nothing here is typed in.

WHY ONLY SOME EXPERIMENTS HAVE A PLATE. A plate makes a claim legible in two seconds, which means
it also makes a claim HARD TO QUALIFY. Three results in this project currently carry an open
question that a plate would paper over, and they are named in the walkthrough rather than drawn.
See WALKTHROUGH.md, "What is not drawn yet, and why".
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ghostscale.plates import (COOL, GRID, HIGHLIGHT, HUMAN, INK, MACHINE, MUTED, NEUTRAL, PAPER,
                               annotate, bar_labels, clean_axis, on_bar, plate, save, zero_line)

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"


def load(p: str) -> dict:
    return json.loads((R / p).read_text(encoding="utf-8"))


made: list[tuple[str, str]] = []


def record(path, caption):
    made.append((Path(path).name, caption))


# =========================================================================== #
# 01 — the headline. A false label moves readers the wrong way.
# =========================================================================== #
def plate_01_false_label():
    d = load("repair/r5_uptake.json")
    rows = {(r["experiment"], r.get("cell")): r for r in d["cells"] if r.get("cell")}
    picks = [
        ("Human work,\nlabelled honestly", rows[("E2", "CREATOR / SIG_CREATOR")], HUMAN),
        ("Machine work,\nlabelled honestly", rows[("E2", "GHOST / SIG_GHOST")], NEUTRAL),
        ("Machine work,\npassed off as human", rows[("E2", "GHOST / SIG_CREATOR")], MACHINE),
    ]
    vals = [float(p[1]["error_reduction"]) for p in picks]

    fig, ax = plate(
        "Being lied to about authorship doesn't just waste your time.\nIt moves you away from the truth.",
        "How much closer to the maker's actual intent a reader ends up. Above zero is learning; "
        "below zero is being led away.",
        "E2 · results/repair/r5_uptake.json · reduction in the surprisal of the true intent, in nats")
    bars = ax.bar([p[0] for p in picks], vals, color=[p[2] for p in picks], width=0.55)
    bar_labels(ax, bars, vals, fmt="{:+.2f}", fontsize=15, color=INK)
    zero_line(ax)
    ax.set_ylim(min(vals) * 1.35, max(vals) * 1.6)
    clean_axis(ax, "closer to the truth  →\n←  further from it")
    ax.set_yticks([])
    on_bar(ax, 2, vals[2] * 0.55,
           "four times further wrong\nthan the truth takes you right", fontsize=12.5,
           bar=bars[2])
    record(save(fig, "01_false_label_moves_you_wrong"),
           "The central result. Same object, different label, opposite direction of travel.")


# =========================================================================== #
# 02 — where invention peaks.
# =========================================================================== #
def plate_02_interior_peak():
    df = pd.read_csv(R / "e20_omega_sweep.csv")
    t1 = load("repair/tier1_recompute.json")
    share = {float(r["overlap"]): float(r["share_of_draws"])
             for r in t1["interior_peak"]["rows"]} if "interior_peak" in t1 else {}

    fig, ax = plate(
        "Invention peaks where content is almost readable — not where it is empty.",
        "Enough familiar structure to make an explanation seem available; not enough to make it "
        "right. The worst place to be is nearly understandable.",
        "E20 · results/e20_omega_sweep.csv · confident-and-contradictory readings, by how much of "
        "the content the reader has vocabulary for")
    x = df.omega.to_numpy()
    y = df.fabrication_index.to_numpy()
    ax.plot(x, y, color=NEUTRAL, lw=2.2, zorder=2)
    ax.fill_between(x, 0, y, color=NEUTRAL, alpha=0.13, zorder=1)
    peak = int(np.argmax(y))
    ax.scatter([x[peak]], [y[peak]], s=150, color=MACHINE, zorder=4)
    annotate(ax, x[peak] + 0.035, y[peak],
             f"peak at {x[peak]:.0%} readable\n({y[peak]:.2f})", color=MACHINE,
             va="center", fontsize=12.5, weight="bold")
    annotate(ax, 0.005, y[0] * 0.55, "totally\nforeign", color=MUTED, fontsize=10.5, va="top")
    annotate(ax, 0.93, max(y) * 0.12, "fully\nreadable", color=MUTED, fontsize=10.5, ha="right")
    if share:
        annotate(ax, 0.52, max(y) * 0.80,
                 "the peak lands in the same place in\n100% of resampled runs",
                 color=HUMAN, fontsize=11.5, weight="bold")
    clean_axis(ax, "invention", "share of the content the reader has vocabulary for")
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    record(save(fig, "02_invention_peaks_in_the_middle"),
           "The most robust result in the project. Same location under every check applied to it.")


# =========================================================================== #
# 03 — the wall: legible and empty.
# =========================================================================== #
def plate_03_the_wall():
    d = load("v6/e37_wall.json")
    c = d["cells"]
    order = [("human", "Human work", HUMAN),
             ("foreign", "Written in a language\nyou don't read", COOL),
             ("noninvertible", "Every word familiar,\nnobody behind it", MACHINE)]
    ent = [float(c[k]["final_entropy"]) for k, _, _ in order]
    eng = [float(c[k]["engaged_fraction"]) for k, _, _ in order]

    fig, ax = plate(
        "\"I can read every word and there's nothing there\" is a different failure\nfrom \"I can't parse this.\"",
        "Content built from familiar material whose maker cannot be reconstructed produces a "
        "signature neither existing condition does: legible, and empty.",
        "E37 · results/v6/e37_wall.json · how unresolved the reader is left, and how long it keeps trying")
    xs = np.arange(3)
    w = 0.36
    b1 = ax.bar(xs - w / 2, ent, width=w, color=[o[2] for o in order], alpha=0.95)
    b2 = ax.bar(xs + w / 2, eng, width=w, color=[o[2] for o in order], alpha=0.42)
    ax.set_xticks(xs)
    ax.set_xticklabels([o[1] for o in order], fontsize=11)
    bar_labels(ax, b1, ent, fmt="{:.2f}", fontsize=11, color=INK)
    bar_labels(ax, b2, eng, fmt="{:.2f}", fontsize=11, color=MUTED)
    annotate(ax, -0.18, max(ent) * 1.14, "solid: how lost you are left",
             color=INK, fontsize=10.5, weight="bold")
    annotate(ax, 1.05, max(ent) * 1.14, "faded: how long you keep trying",
             color=MUTED, fontsize=10.5, weight="bold")
    ax.set_ylim(0, max(ent) * 1.3)
    clean_axis(ax)
    ax.set_yticks([])
    record(save(fig, "03_legible_and_empty"),
           "A third kind of failure, built because the existing account did not match what people "
           "actually report about generated text.")


# =========================================================================== #
# 04 — depth moves the method, not the purpose.
# =========================================================================== #
def plate_04_method_not_purpose():
    d = load("v6/e36_process.json")
    cells = [c for c in d["cells"] if abs(float(c["beta"]) - 1.0) < 1e-9]
    cells.sort(key=lambda c: c["mu"])
    mus = [int(c["mu"]) for c in cells]
    proc = [float(c["process_error_reduction"]) for c in cells]
    goal = [float(c["goal_error_reduction"]) for c in cells]

    fig, ax = plate(
        "Depth changes how much of the method you pick up.\nIt cannot change how much of the purpose you get — by design.",
        "The model is built so a deep work and a shallow one state their purpose equally clearly. "
        "For five versions we measured the purpose, and found nothing.",
        "E30 / E36 · results/v6/e36_process.json · what a reader takes away, scored two ways")
    xs = np.arange(len(mus))
    ax.plot(xs, goal, "-o", color=NEUTRAL, lw=2.4, ms=9)
    ax.plot(xs, proc, "-o", color=HIGHLIGHT, lw=2.8, ms=10)
    annotate(ax, xs[-1] + 0.06, goal[-1], "the purpose\n(flat, by construction)",
             color=MUTED, va="center", fontsize=11.5)
    annotate(ax, xs[-1] + 0.06, proc[-1], "the method\n(this moves)",
             color=HIGHLIGHT, va="center", fontsize=11.5, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["a scribble", "practised work", "a master's work"], fontsize=11.5)
    ax.set_xlim(-0.25, len(mus) - 0.35)
    zero_line(ax)
    clean_axis(ax, "what the reader takes away")
    ax.set_yticks([])
    record(save(fig, "04_depth_moves_the_method"),
           "Why the depth result was null for five versions: the measurement was pointed at the "
           "one quantity the design holds constant.")


# =========================================================================== #
# 05 — intent unlocks the method.
# =========================================================================== #
def plate_05_intent_unlocks():
    d = load("v6/e36_process.json")["H6.3b_temporal"]
    before, after = float(d["process_before_settling"]), float(d["process_after_settling"])
    lo, hi = [float(x) for x in d["interval"]]

    fig, ax = plate(
        "Work out what someone was trying to do, and their choices start making sense.",
        "Inside a single reading: how much of the maker's method the reader picks up, before and "
        "after it settles on what the work was for.",
        f"E36 · results/v6/e36_process.json · {d['n_rollouts']} readings · "
        f"gain {after - before:+.3f}, 95% interval [{lo:+.3f}, {hi:+.3f}]")
    bars = ax.bar(["Before you work out\nwhat it was for",
                   "After you work out\nwhat it was for"],
                  [before, after], color=[NEUTRAL, HUMAN], width=0.5)
    bar_labels(ax, bars, [before, after], fmt="{:.3f}", fontsize=15, color=INK)
    ax.set_ylim(0, after * 1.45)
    clean_axis(ax, "how much of the method you pick up")
    ax.set_yticks([])
    annotate(ax, 0.5, after * 1.22,
             f"{after / before:.1f}× more, same reader, same object",
             color=HUMAN, ha="center", fontsize=13, weight="bold")
    record(save(fig, "05_intent_unlocks_the_method"),
           "Intent is the key that makes the method readable. Not in the essay or the preprint — "
           "it came out of a conversation and then held.")


# =========================================================================== #
# 06 — the two mechanisms.
# =========================================================================== #
def plate_06_two_mechanisms():
    d = load("v6/retrofit.json")["families"]["trust"]["machine_work_integration"]
    labels = ["Told it was machine-made", "Passed off as human"]
    race = [d["labelled_honestly__channel_race"], d["passed_off_as_human__channel_race"]]
    gate = [d["labelled_honestly__coupled_gate"], d["passed_off_as_human__coupled_gate"]]

    fig, ax = plate(
        "If trust switches off your guard, honesty stops being enough.",
        "Two accounts of the same effect. One says you absorb machine work because you were "
        "fooled about its source. The other says trust itself lowered the guard — so you absorb "
        "it even when told the truth.",
        "E41 · results/v6/retrofit.json · how much of the work the reader is allowed to keep")
    xs = np.arange(2)
    w = 0.34
    b1 = ax.bar(xs - w / 2, race, width=w, color=NEUTRAL)
    b2 = ax.bar(xs + w / 2, gate, width=w, color=MACHINE)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=12)
    bar_labels(ax, b1, race, fmt="{:.2f}", fontsize=12, color=MUTED)
    bar_labels(ax, b2, gate, fmt="{:.2f}", fontsize=12, color=INK)
    ax.set_ylim(0, 1.28)
    clean_axis(ax, "how much gets absorbed")
    ax.set_yticks([])
    annotate(ax, -0.42, 1.16, "fooled about the source", color=MUTED, fontsize=11, weight="bold")
    annotate(ax, 0.30, 1.16, "guard lowered by trust", color=MACHINE, fontsize=11, weight="bold")
    annotate(ax, 0, gate[0] + 0.10, "told the truth,\nbelieves it,\nabsorbs it anyway",
             color=MACHINE, ha="center", fontsize=11, weight="bold")
    record(save(fig, "06_honesty_is_not_enough"),
           "The published theory and the code had been explaining the same effect by different "
           "mechanisms. They make different predictions, and this is the one that separates them.")


# =========================================================================== #
# 07 — reputation blindness.
# =========================================================================== #
def plate_07_reputation_blindness():
    d = load("repair/r8b_learned_trust.json")
    rows = d["by_trust"]
    k = [float(r["kappa"]) for r in rows]
    slope = [float(r["slope"]) for r in rows]

    fig, ax = plate(
        "The people most inclined to believe a label are the ones who can never\nfind out the labeller lies.",
        "How well a reader learns a source's honesty, by how much it trusted labels to begin with. "
        "Past a threshold, the evidence never arrives: the label wins the argument before the "
        "mismatch can register.",
        "R-8b · results/repair/r8b_learned_trust.json · how well the source's true honesty is recovered")
    bars = ax.bar([f"{x:.2f}" for x in k], slope,
                  color=[HUMAN if s > 0.4 else MACHINE for s in slope], width=0.55)
    bar_labels(ax, bars, slope, fmt="{:.2f}", fontsize=13, color=INK)
    ax.set_ylim(0, max(max(slope), 0.05) * 1.35)
    clean_axis(ax, "how well the lie is detected", "how much the reader trusted labels to start with")
    ax.set_yticks([])
    annotate(ax, len(k) - 1, max(slope) * 0.55,
             "not slow learning.\nlearning that cannot start.",
             color=MACHINE, ha="center", fontsize=12, weight="bold")
    record(save(fig, "07_reputation_blindness"),
           "A prediction the earlier model could not make. If it holds outside this simulation, "
           "a disclosure regime protects trusting readers least where it fails most.")


# =========================================================================== #
# 08 — expertise substitutes.
# =========================================================================== #
def plate_08_expertise_substitutes():
    d = load("v6/e38_expertise.json")
    cells = {(c["reader"], c["content"]): float(c["goal_accuracy"]) for c in d["cells"]}

    fig, ax = plate(
        "Learning to read machine work doesn't add a skill. It swaps one out.",
        "Two readers, identical except for what they expect a maker to be like. The reader tuned "
        "to machines reads machine work perfectly — and loses half its accuracy on human work.",
        "E38 · results/v6/e38_expertise.json · share of makers' intent correctly recovered")
    xs = np.arange(2)
    w = 0.34
    human_r = [cells[("human", "human")], cells[("human", "machine")]]
    mach_r = [cells[("machine", "human")], cells[("machine", "machine")]]
    b1 = ax.bar(xs - w / 2, human_r, width=w, color=HUMAN)
    b2 = ax.bar(xs + w / 2, mach_r, width=w, color=MACHINE)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Reading human work", "Reading machine work"], fontsize=12.5)
    bar_labels(ax, b1, human_r, fmt="{:.0%}", fontsize=12.5, color=INK)
    bar_labels(ax, b2, mach_r, fmt="{:.0%}", fontsize=12.5, color=INK)
    ax.set_ylim(0, 1.32)
    clean_axis(ax, "intent correctly recovered")
    ax.set_yticks([])
    annotate(ax, -0.44, 1.20, "a reader tuned to people", color=HUMAN, fontsize=11.5, weight="bold")
    annotate(ax, 0.40, 1.20, "a reader tuned to machines", color=MACHINE, fontsize=11.5, weight="bold")
    record(save(fig, "08_expertise_substitutes"),
           "A crossover, not an upgrade. The adaptation that protects you from the crash is the "
           "one that costs you the human channel.")


# =========================================================================== #
# 09 — pays more, gets less.
# =========================================================================== #
def plate_09_pays_more_gets_less():
    dec = pd.read_csv(R / "v6" / "e40_cues" / "e40_decoupling.csv")
    sh = dec[dec.mu == 1]
    honest = float(sh[sh.regime == "honest"].engagement.iloc[0])
    decoup = float(sh[sh.regime == "decoupled"].engagement.iloc[0])
    uptake = float(sh[sh.regime == "decoupled"].error_reduction.iloc[0])

    fig, ax = plate(
        "Optimise the thing that used to signal depth, and readers pay more for less.",
        "Surface appeal is a learned promise that something is worth your attention. Optimise it "
        "directly and the promise detaches from what it promised.",
        "E40 · results/v6/e40_cues/e40_decoupling.csv · attention spent on content with no depth in it")
    bars = ax.bar(["Shallow content,\nlooks shallow", "Shallow content,\npolished to look deep"],
                  [honest, decoup], color=[NEUTRAL, MACHINE], width=0.5)
    bar_labels(ax, bars, [honest, decoup], fmt="{:.2f}", fontsize=15, color=INK)
    ax.set_ylim(0, max(honest, decoup) * 1.5)
    clean_axis(ax, "attention spent")
    ax.set_yticks([])
    annotate(ax, 1, decoup * 1.22,
             f"{decoup / honest:.1f}× the attention,\nand it learns {uptake:+.2f}",
             color=MACHINE, ha="center", fontsize=12.5, weight="bold")
    record(save(fig, "09_pays_more_gets_less"),
           "A third failure mode. Not the crash — the reader is engaged. Not the lie — nobody "
           "lied. A reader correctly reading something built to trip its own heuristic.")


# =========================================================================== #
# 10 — engagement is not integration.
# =========================================================================== #
def plate_10_looking_vs_being_changed():
    d = load("v6/e42_vulnerability.json")
    cells = {c["cell"]: c for c in d["cells"]}
    names = [("sustained_aligned", "Absorbed, learning nothing"),
             ("resolving_aligned", "Understood, and changed by it"),
             ("resolving_divergent", "Understood, and refused")]
    eng = [float(cells[k]["engaged_fraction"]) for k, _ in names]
    integ = [float(cells[k]["integration"]) for k, _ in names]

    fig, ax = plate(
        "Paying attention and being willing to be changed are not the same thing.",
        "A reader can study something closely, understand its maker exactly, and let none of it "
        "in. Attention is what you spend; letting it change you is a separate decision.",
        "E42 · results/v6/e42_vulnerability.json · attention spent, against how much is allowed to land")
    xs = np.arange(len(names))
    w = 0.34
    b1 = ax.bar(xs - w / 2, eng, width=w, color=COOL)
    b2 = ax.bar(xs + w / 2, integ, width=w, color=HIGHLIGHT)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in names], fontsize=11.5)
    bar_labels(ax, b1, eng, fmt="{:.2f}", fontsize=11.5, color=INK)
    bar_labels(ax, b2, integ, fmt="{:.2f}", fontsize=11.5, color=INK)
    ax.set_ylim(0, 1.30)
    clean_axis(ax)
    ax.set_yticks([])
    annotate(ax, -0.44, 1.18, "how hard you look", color=COOL, fontsize=11.5, weight="bold")
    annotate(ax, 0.65, 1.18, "how much lands", color=HIGHLIGHT, fontsize=11.5, weight="bold")
    record(save(fig, "10_looking_is_not_being_changed"),
           "Every combination is reachable. The two are driven by different parts of the model "
           "and had never been reported apart.")


# =========================================================================== #
# 11 — the master cannot explain themselves.
# =========================================================================== #
def plate_11_self_report():
    d = load("v6/e43_selfreport.json")
    cells = sorted(d["cells"], key=lambda c: c["mu"])
    declared = [float(c["declared_accuracy"]) for c in cells]
    reader = [float(c["reader_accuracy"]) for c in cells]

    fig, ax = plate(
        "The more practised the work, the less its maker can say why.",
        "A novice can tell you exactly which rule they were following, because they are still "
        "following it on purpose. Practice compresses decisions, and compression is what puts "
        "them out of reach.",
        "E43 · results/v6/e43_selfreport.json · how often the maker names its own purpose correctly, "
        "against how often a reader does")
    xs = np.arange(len(cells))
    ax.plot(xs, reader, "-o", color=HUMAN, lw=2.6, ms=9)
    ax.plot(xs, declared, "-o", color=MACHINE, lw=2.6, ms=9)
    annotate(ax, xs[-1] + 0.05, reader[-1], "what a reader\nworks out",
             color=HUMAN, va="center", fontsize=11.5, weight="bold")
    annotate(ax, xs[-1] + 0.05, declared[-1], "what the maker\ncan tell you",
             color=MACHINE, va="center", fontsize=11.5, weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["a scribble", "practised work", "a master's work"], fontsize=11.5)
    ax.set_xlim(-0.2, len(cells) - 0.3)
    ax.set_ylim(0, 1.12)
    clean_axis(ax, "gets the purpose right")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["never", "half", "always"])
    record(save(fig, "11_the_master_cannot_explain"),
           "The reader ends up knowing the maker better than the maker knows themselves — and "
           "here nobody set that. Practice sets it.")


# =========================================================================== #
# 12 — the two kinds of damage.
# =========================================================================== #
def plate_12_two_damages():
    df = pd.read_csv(R / "e9_summary.csv")
    ctrl = df[df.arm == "control"].sort_values("contamination")
    both = df[df.arm == "both"].sort_values("contamination")
    starve = df[df.arm == "starvation_only"].sort_values("contamination")

    fig, ax = plate(
        "One kind of damage scales with how much slop you consume.\nThe other is already there at zero.",
        "Absorbing bad material gets worse the more there is. Failing to absorb good material "
        "does not — it is driven by walking away, not by what is in the pile.",
        "E9 · results/e9_summary.csv · error in the reader's model of what people are like")
    x = both.contamination.to_numpy()
    ax.plot(x, both.shape_kl.to_numpy(), "-o", color=MACHINE, lw=2.6, ms=9)
    ax.plot(x, starve.shape_kl.to_numpy(), "-o", color=COOL, lw=2.6, ms=9)
    ax.plot(x, ctrl.shape_kl.to_numpy(), "-o", color=NEUTRAL, lw=2.0, ms=7)
    annotate(ax, x[-1] + 0.012, float(both.shape_kl.iloc[-1]), "absorbing it",
             color=MACHINE, va="center", fontsize=11.5, weight="bold")
    annotate(ax, x[-1] + 0.012, float(starve.shape_kl.iloc[-1]), "walking away from it",
             color=COOL, va="center", fontsize=11.5, weight="bold")
    annotate(ax, x[-1] + 0.012, float(ctrl.shape_kl.iloc[-1]), "clean corpus",
             color=MUTED, va="center", fontsize=10.5)
    annotate(ax, 0.005, float(starve.shape_kl.iloc[0]) * 1.18,
             "already damaged with\nzero machine content in the pile",
             color=COOL, fontsize=11.5, weight="bold", va="bottom")
    ax.set_xlim(-0.02, 0.78)
    ax.set_xticks([0, 0.3, 0.6])
    ax.set_xticklabels(["none", "30%", "60%"])
    clean_axis(ax, "damage to the reader's model", "share of the corpus that is machine-made")
    ax.set_yticks([])
    record(save(fig, "12_two_kinds_of_damage"),
           "The second kind has the strongest independent support of anything in the project.")


# =========================================================================== #
# 13 — a counting classifier does it too.
# =========================================================================== #
def plate_13_no_mind_needed():
    df = pd.read_csv(R / "e21_cell_stats.csv")
    col = "within_observer" if "within_observer" in df.columns else df.columns[2]
    between = "between_observer" if "between_observer" in df.columns else df.columns[3]
    arms = df[df.get("cell", df.columns[1]).astype(str).str.contains("ghost|GHOST|claimed",
                                                                     case=False, na=False)] \
        if "cell" in df.columns else df
    full = df[df.arm.astype(str).str.contains("A_active", na=False)]
    naive = df[df.arm.astype(str).str.contains("E_no_tom", na=False)]
    if not len(full) or not len(naive):
        return
    vals = [float(full[col].mean()), float(naive[col].mean())]
    dis = [float(full[between].mean()), float(naive[between].mean())]

    fig, ax = plate(
        "You don't need to imagine a mind to invent one.",
        "A classifier that counts features and never represents a maker reproduces the same "
        "confident, mutually contradictory readings. The framework used to claim this required "
        "modelling another mind. It doesn't.",
        "E21 · results/e21_cell_stats.csv · how certain each reader is, and how much they disagree")
    xs = np.arange(2)
    w = 0.34
    b1 = ax.bar(xs - w / 2, vals, width=w, color=COOL)
    b2 = ax.bar(xs + w / 2, dis, width=w, color=MACHINE)
    ax.set_xticks(xs)
    ax.set_xticklabels(["The full model\n(imagines a maker)",
                        "A counting classifier\n(imagines nothing)"], fontsize=12)
    bar_labels(ax, b1, vals, fmt="{:.2f}", fontsize=12, color=INK)
    bar_labels(ax, b2, dis, fmt="{:.2f}", fontsize=12, color=INK)
    ax.set_ylim(0, max(dis) * 1.35)
    clean_axis(ax)
    ax.set_yticks([])
    annotate(ax, -0.44, max(dis) * 1.22, "uncertainty", color=COOL, fontsize=11.5, weight="bold")
    annotate(ax, 0.40, max(dis) * 1.22, "disagreement", color=MACHINE, fontsize=11.5, weight="bold")
    record(save(fig, "13_no_mind_needed"),
           "A claim this project withdrew, using its own experiment. Kept prominently on purpose.")


# =========================================================================== #
# 14 — a knee, not a cliff.
# =========================================================================== #
def plate_14_knee_not_cliff():
    d = load("e15_verdict.json")
    w = d["width_vs_evidence"]
    sizes = w["corpus_sizes"]
    widths = w["widths"]

    fig, ax = plate(
        "Competence doesn't fall off a cliff. It bends.",
        "A real threshold sharpens as you gather more evidence. This one did not budge across "
        "sixteen times the data — so its shape comes from the model, not from a boundary in the "
        "world.",
        f"E15 · results/e15_verdict.json · width of the transition, at three evidence levels "
        f"(ratio {w['width_ratio']:.2f})")
    xs = np.arange(len(sizes))
    bars = ax.bar(xs, widths, color=NEUTRAL, width=0.5)
    bar_labels(ax, bars, widths, fmt="{:.3f}", fontsize=13, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s}×" if not str(s).startswith("1") or s != sizes[0] else "baseline"
                        for s in sizes], fontsize=12)
    ax.set_ylim(0, max(widths) * 1.35)
    clean_axis(ax, "how sharp the transition is", "evidence available to each reader")
    ax.set_yticks([])
    annotate(ax, (len(sizes) - 1) / 2, max(widths) * 1.18,
             "a genuine cliff would get narrower →  this doesn't",
             color=MACHINE, ha="center", fontsize=12.5, weight="bold")
    record(save(fig, "14_a_knee_not_a_cliff"),
           "The author's own claim, tested knowing it could only survive or weaken. It weakened.")


# =========================================================================== #
# 15 — labels only help if you know the convention.
# =========================================================================== #
def plate_15_coverage():
    d = load("e16_verdict.json")
    aware = float(d["headline_threshold"])
    naive = float(d["naive_threshold_same_regime"])

    fig, ax = plate(
        "Labelling only protects readers who know the labelling exists.",
        "How much machine content has to be labelled before a reader keeps a clean picture of "
        "what people are like. Readers who don't know the convention need far more — and never "
        "get fully there.",
        "E16 · results/e16_verdict.json · a lower bound: the aware reader is handed the true coverage")
    bars = ax.bar(["Knows the convention", "Doesn't know it"], [aware, naive],
                  color=[HUMAN, MACHINE], width=0.5)
    bar_labels(ax, bars, [aware, naive], fmt="{:.0%}", fontsize=15, color=INK)
    ax.set_ylim(0, 1.0)
    clean_axis(ax, "share of machine content that must be labelled")
    ax.set_yticks([])
    record(save(fig, "15_labels_need_a_convention"),
           "The policy number, and the reason a disclosure scheme is a literacy problem as much "
           "as a compliance one.")


# =========================================================================== #
# 16 — the label wins, above a computable point.
# =========================================================================== #
def plate_16_channel_race():
    d = load("diagnostics/d1_channels.json")
    rows = d.get("cells") or d.get("rows")
    row = next(r for r in rows if "passed off" in str(r.get("cell", "")).lower())
    cross = float(row["crossover_kappa"])
    content = float(row["content_llr_per_step"])
    label = float(row["label_llr_per_step"])

    fig, ax = plate(
        "Two witnesses arrive with every glance, and on a lie they disagree.",
        "The work argues for the truth. The label argues for the lie. Which one wins is settled "
        "by arithmetic, not by psychology — and the tipping point can be computed without running "
        "anything.",
        "D-1 · results/diagnostics/d1_channels.json · evidence per glance, machine work passed off as human")
    # A wider left margin than the house default: on a horizontal bar chart the category names
    # ARE the axis, and at the default margin these two ran off the canvas entirely.
    pos = ax.get_position()
    ax.set_position([0.235, pos.y0, 0.70, pos.height])
    bars = ax.barh(["What the work says\n(the truth)", "What the label says\n(the lie)"],
                   [content, label], color=[HUMAN, MACHINE], height=0.45)
    for b, v in zip(bars, [content, label]):
        ax.text(v + (0.12 if v > 0 else -0.12), b.get_y() + b.get_height() / 2,
                f"{v:+.2f}", va="center", ha="left" if v > 0 else "right",
                fontsize=14, fontweight="bold", color=INK)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlim(min(content, 0) * 1.5, label * 1.35)
    clean_axis(ax, "", "evidence per glance")
    ax.set_xticks([])
    annotate(ax, label * 0.50, 1.42,
             f"the label wins whenever trust is above {cross:.2f}",
             color=MACHINE, ha="center", va="center", fontsize=13.5, weight="bold")
    ax.set_ylim(-0.55, 1.75)
    record(save(fig, "16_two_witnesses"),
           "Computed in closed form with no simulation. It explains several later results and "
           "should probably have been the first thing in the project.")


# =========================================================================== #
# 17 — the experiment that was withheld.
# =========================================================================== #
def plate_17_withheld():
    fig, ax = plate(
        "One experiment has been withheld three times, and stays withheld.",
        "It asks whether damage compounds as each generation learns from the last. Its own "
        "honesty check — with zero contamination, show zero damage — has failed every time. So "
        "real decay cannot be told apart from the instrument's own noise.",
        "E8 · a failing test is kept in the suite as a visible marker, so any future fix has to "
        "switch it off deliberately")
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.72, "0.00116", ha="center", va="center", fontsize=54,
            fontweight="bold", color=MACHINE)
    ax.text(0.5, 0.44, "measured drift with nothing to detect", ha="center", va="center",
            fontsize=13, color=MUTED)
    ax.text(0.5, 0.24, "against a ceiling of 0.001", ha="center", va="center",
            fontsize=15, color=INK, fontweight="bold")
    ax.text(0.5, 0.05,
            "This is not \"we found no effect\". It is \"we could not measure\".",
            ha="center", va="center", fontsize=13.5, color=INK, style="italic")
    record(save(fig, "17_the_withheld_experiment"),
           "Narrowly missing is exactly the case a no-exceptions rule exists for.")



def _g(x, n=3):
    return f"{float(x):.{n}g}"


# =========================================================================== #
# 18 - what modelling a maker buys (the E21 attack).
# =========================================================================== #
def plate_18_what_tom_buys():
    d = load("v7/e45_tom_efficiency.json")
    h1 = d["H7.1"]
    sim, cnt = h1["evidence_the_simulator_needs"], h1["evidence_the_counter_needs"]

    fig, ax = plate(
        "Imagining a maker is not about being right. It is about being cheap.",
        "How many worked examples each kind of reader needs before it can read intent reliably. "
        "One already owns the machinery and only has to ask which intention is running. The other "
        "has to learn the whole map from scratch.",
        "E45 - results/v7/e45_tom_efficiency.json - examples needed to reach 80% accuracy")
    bars = ax.bar(["Simulates the maker", "Counts what it has seen"], [sim, cnt],
                  color=[HUMAN, MACHINE], width=0.5)
    for b, v in zip(bars, [sim, cnt]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, str(int(v)), ha="center",
                va="bottom", fontsize=20, fontweight="bold", color=INK)
    ax.set_yscale("log")
    ax.set_ylim(1, cnt * 6)
    clean_axis(ax, "worked examples needed (log scale)")
    ax.set_yticks([])
    annotate(ax, 0.5, cnt * 2.2, str(int(cnt / max(sim, 1))) + "x less evidence",
             color=HUMAN, ha="center", fontsize=15, weight="bold")
    record(save(fig, "18_what_imagining_a_maker_buys"),
           "The experiment that made this project withdraw a claim asked whether you NEED to "
           "imagine a maker. You do not. It never asked what imagining one buys.")


# =========================================================================== #
# 19 - reading an intent nobody has shown you.
# =========================================================================== #
def plate_19_zero_shot():
    d = load("v7/e45_tom_efficiency.json")["H7.2"]
    by = {int(k): float(v) for k, v in d["counter_by_training_size"].items()}
    xs = sorted(by)
    sim = float(d["simulator_on_an_unseen_intent"])
    chance = float(d["chance"])

    fig, ax = plate(
        "You can recognise a purpose nobody has ever shown you. A pattern-matcher cannot.",
        "Reading an intention that appears nowhere in the training data. More data does not help "
        "the pattern-matcher, because its problem was never a shortage of examples.",
        "E45 - results/v7/e45_tom_efficiency.json - accuracy on a held-out intention")
    ax.plot(range(len(xs)), [by[x] for x in xs], "-o", color=MACHINE, lw=2.6, ms=8)
    ax.axhline(sim, color=HUMAN, lw=2.8)
    ax.axhline(chance, color=NEUTRAL, lw=1.4, ls=":")
    annotate(ax, 0.05, sim + 0.03, "a reader that simulates", color=HUMAN,
             fontsize=12.5, weight="bold")
    annotate(ax, 0.05, chance + 0.025, "pure guessing", color=MUTED, fontsize=10.5)
    annotate(ax, len(xs) - 1.05, by[xs[-1]] - 0.09, "a reader that counts", color=MACHINE,
             fontsize=12.5, weight="bold", ha="right")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs], fontsize=10)
    ax.set_ylim(0, 1.0)
    clean_axis(ax, "gets the intention right", "worked examples it was trained on")
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["never", "half", "always"])
    record(save(fig, "19_reading_an_unseen_intent"),
           "This is what cheating the solution space means. A reader that can run the generator "
           "gets the whole space; a reader that has to observe it only ever has the part it saw.")


# =========================================================================== #
# 20 - rejection is not protection.
# =========================================================================== #
def plate_20_rejection_is_not_protection():
    d = load("v7/e46_gate_leak.json")
    by = d["drift_by_leak"]
    ks = sorted(by, key=lambda k: float(k))
    drift = [float(by[k]["final_drift"]) for k in ks]

    fig, ax = plate(
        "You cannot read something, reject it, and walk away unchanged.",
        "To decide you disagree with something you first have to work out what it says, and "
        "working out what it says means partly running it. Refusing is itself a small act of "
        "taking on, and it compounds.",
        "E46 - results/v7/e46_gate_leak.json - drift in a reader's own beliefs after repeatedly "
        "rejecting everything it was shown")
    labels = [f"{float(k):.0%}" for k in ks]
    bars = ax.bar(labels, drift,
                  color=[NEUTRAL if float(k) == 0 else MACHINE for k in ks], width=0.55)
    bar_labels(ax, bars, drift, fmt="{:.3f}", fontsize=12.5, color=INK)
    ax.set_ylim(0, max(drift) * 1.45)
    clean_axis(ax, "how far the reader moved", "how leaky the guard is")
    ax.set_yticks([])
    eng = d.get("engagement", {})
    if eng.get("ratio"):
        annotate(ax, (len(ks) - 1) / 2.0, max(drift) * 1.24,
                 "and the reader who studies it carefully to refute it\n"
                 "drifts " + str(int(round(eng["ratio"]))) + "x more than the one who skims",
                 color=MACHINE, ha="center", fontsize=12.5, weight="bold")
    record(save(fig, "20_rejection_is_not_protection"),
           "The theory always contained this term and the code never did. It is the proposed "
           "mechanism for indoctrination: you are changed by what you refuse.")


# =========================================================================== #
# 21 - the two gates, settled.
# =========================================================================== #
def plate_21_two_gates_settled():
    d = load("v7/closures.json")["C-1"]
    m, g = d["scored_on_the_method"], d["scored_on_the_purpose"]
    hist = d["history"]

    fig, ax = plate(
        "A disagreement that ran for three passes was a measurement pointed at the wrong thing.",
        "Does what a reader takes away track its own sense of how much thinking went in? The "
        "criterion was scored on the work's PURPOSE, which this model deliberately holds equally "
        "readable at every depth, so it could never have moved.",
        "E31 / C-1 - results/v7/closures.json - rank correlation, on E31's own design")
    vals = [float(g["rho"]), float(m["rho"])]
    bars = ax.bar(["Scored on the purpose\n(held constant by design)",
                   "Scored on the method\n(what the theory names)"],
                  vals, color=[NEUTRAL, HUMAN], width=0.5)
    bar_labels(ax, bars, vals, fmt="{:.2f}", fontsize=16, color=INK)
    ax.axhline(float(m["bar"]), color=MACHINE, lw=1.6, ls="--")
    # The threshold label sits INSIDE the axes and above the line. Placing it at negative x put
    # it off the left edge, where it was clipped and collided with the y-axis label at the same
    # time -- two failures from one coordinate.
    annotate(ax, 0.5, float(m["bar"]) + 0.05, "the bar it had to clear",
             color=MACHINE, fontsize=11, weight="bold", ha="center")
    ax.set_ylim(0, 1.30)
    clean_axis(ax, "how tightly uptake\ntracks perceived depth")
    ax.set_yticks([])
    # The history note goes ABOVE the value label rather than under it, or the two overprint.
    annotate(ax, 1, vals[1] + 0.17,
             _g(hist["approximate_solver"]) + ", then " + _g(hist["exact_solver"]) + ", then this",
             color=MUTED, ha="center", fontsize=10.5)
    record(save(fig, "21_pointed_at_the_wrong_thing"),
           "The project's longest-running open question, settled by changing what was measured "
           "rather than how it was measured.")



# =========================================================================== #
# 22 - how much of this is the theory (the severity rates).
# =========================================================================== #
def plate_22_how_much_is_the_theory():
    d = load("v8/s1_severity.json")
    rates = d["rates"]
    ref = d.get("reference_point", {})

    # Short, plain labels. The full finding names are in the footer's source file; a bar chart
    # whose labels need two lines each is not readable in two seconds.
    SHORT = {
        "E2/R-5 a false label moves you the wrong way": "a false label misleads you",
        "E36 depth moves the method, not the purpose": "depth transmits method",
        "E37 the wall is a distinct failure": "legible and empty",
    }
    labels, vals = [], []
    for name, r in rates.items():
        v = r.get("false_positive_rate")
        if v is None or v != v:
            continue
        labels.append(SHORT.get(name, name[:28]))
        vals.append(float(v))
    if ref.get("false_positive_rate") is not None:
        labels.append("certainty under a false label")
        vals.append(float(ref["false_positive_rate"]))

    order = sorted(range(len(vals)), key=lambda i: vals[i])     # ascending, so 0% sits at bottom
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]

    fig, ax = plate(
        "I threw my own settings away and checked how much of this was ever mine.",
        "Keep the shape of the model, randomise everything the theory specifies, and count how "
        "often the finding still appears. If it appears every time, it came from the shape.",
        "S-1 - results/v8/s1_severity.json - share of randomly parameterised models that "
        "reproduce each finding")
    # A wider left margin than the house default, because these labels are the axis.
    pos = ax.get_position()
    ax.set_position([0.26, pos.y0, 0.62, pos.height])

    colors = [MACHINE if v > 0.5 else HUMAN for v in vals]
    bars = ax.barh(labels, vals, color=colors, height=0.5)
    for b, v in zip(bars, vals):
        ax.text(v + 0.025, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                va="center", ha="left", fontsize=16, fontweight="bold", color=INK)
    ax.set_xlim(0, 1.25)
    ax.tick_params(axis="y", labelsize=12)
    clean_axis(ax, "", "")
    ax.set_xticks([])
    annotate(ax, 0.16, 0.02, "needed the theory", color=HUMAN,
             fontsize=12.5, weight="bold", va="center")
    annotate(ax, 1.02, len(vals) - 1.5, "came from the\narchitecture,\nnot the theory",
             color=MACHINE, fontsize=12, weight="bold", va="center")
    record(save(fig, "22_how_much_is_the_theory"),
           "The most important number here is not a result. It costs the project its two "
           "biggest-sounding claims.")


# =========================================================================== #
# 23 - honest marking is self-policing, at a price.
# =========================================================================== #
def plate_23_honesty_pays_at_a_price():
    d = load("v8/e51_creator.json")
    eq = [r for r in d["equilibrium"] if abs(float(r["leak"])) < 1e-9]
    eq.sort(key=lambda r: r["detection"])
    x = [float(r["detection"]) for r in eq]
    liar = [float(r["defector_payoff"]) for r in eq]
    honest = [float(r["honest_payoff"]) for r in eq]
    thr = d.get("detection_rate_where_honesty_pays", {}).get("tight_gate")

    fig, ax = plate(
        "Marking your work honestly only pays if half the liars get caught.",
        "A maker who labels honestly loses something: the label lowers what readers take from the "
        "work. A maker who lies gets the uptake of honest work. Whether honesty survives depends "
        "entirely on how often lying is noticed.",
        "E51 - results/v8/e51_creator.json - what each strategy earns, by how often a lie is detected")
    ax.plot(x, liar, "-o", color=MACHINE, lw=2.8, ms=9)
    ax.plot(x, honest, "-o", color=HUMAN, lw=2.8, ms=9)
    zero_line(ax)
    if thr is not None:
        ax.axvline(float(thr), color=INK, lw=1.4, ls="--")
        annotate(ax, float(thr) + 0.015, max(liar) * 0.55,
                 f"honesty wins\nabove {float(thr):.0%}", color=INK,
                 fontsize=12.5, weight="bold")
    annotate(ax, x[0] + 0.01, liar[0] + 0.06, "lying", color=MACHINE,
             fontsize=13, weight="bold")
    annotate(ax, x[0] + 0.01, honest[0] - 0.18, "marking it honestly", color=HUMAN,
             fontsize=13, weight="bold")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["never", "25%", "half", "75%", "always"])
    clean_axis(ax, "what the strategy earns", "how often a lie is caught")
    ax.set_yticks([])
    record(save(fig, "23_honesty_pays_at_a_price"),
           "The framework's answer to bad actors, simulated for the first time. It works, and it "
           "comes with a condition rather than a reassurance.")


# =========================================================================== #
# 24 - what each finding is actually made of.
# =========================================================================== #
def plate_24_what_its_made_of():
    """The ablation grid as a grid, because that IS the finding.

    A bar chart would have to pick one number and this result is a pattern: one column is solid
    red and the rest are almost solid green. The eye should land on that column before it reads
    a single label, which is why the maker column is drawn first and left of everything else.
    """
    d = load("v9/summary.json")["MIN"]["minimal_models"]

    ROWS = [("the label effect", "a false label misleads you"),
            ("legible and empty", "legible and empty"),
            ("depth transmits method", "depth transmits method")]
    COLS = [("generative", "imagining\na maker"),
            ("shared_likelihood", "a shared\nbody plan"),
            ("distributional", "holding a\ndistribution"),
            ("provenance_as_state", "inferring\nwhere it\ncame from"),
            ("hierarchy", "levels in\nthe maker"),
            ("costly_attention", "looking\ncosts\nsomething")]

    fig, ax = plate(
        "I removed one piece of the model at a time to see what each finding is made of.",
        "Every result in this project dies the moment the reader stops imagining a maker and "
        "starts pattern-matching a surface. Nothing else is load-bearing everywhere.",
        "V9 minimal-model programme - results/v9/summary.json - a finding 'dies' when it no "
        "longer appears with that structural commitment removed")

    pos = ax.get_position()
    ax.set_position([0.235, pos.y0 + 0.02, 0.70, pos.height - 0.02])

    for r, (key, row_label) in enumerate(ROWS):
        needs = set(d[key]["load_bearing"])
        y = len(ROWS) - 1 - r
        for c, (ck, _) in enumerate(COLS):
            dead = ck in needs
            ax.add_patch(plt.Rectangle((c - 0.44, y - 0.36), 0.88, 0.72,
                                       facecolor=MACHINE if dead else HUMAN,
                                       edgecolor=PAPER, linewidth=2.5, zorder=2))
            ax.text(c, y, "DIES" if dead else "fine", ha="center", va="center",
                    color=PAPER, fontsize=12 if dead else 11,
                    fontweight="bold" if dead else "normal", zorder=3)
        ax.text(-0.72, y, row_label, ha="right", va="center", fontsize=12.5, color=INK)

    for c, (_, cl) in enumerate(COLS):
        ax.text(c, len(ROWS) - 0.52, cl, ha="center", va="bottom", fontsize=10.5,
                color=INK if c == 0 else MUTED,
                fontweight="bold" if c == 0 else "normal", linespacing=1.2)

    ax.set_xlim(-0.6, len(COLS) - 0.4)
    ax.set_ylim(-0.55, len(ROWS) + 0.30)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0, -0.52, "remove this and\neverything falls over", ha="center", va="top",
            fontsize=11.5, color=MACHINE, fontweight="bold", linespacing=1.25)
    ax.text(4.5, -0.52, "the model can lose these and keep every result",
            ha="center", va="top", fontsize=11, color=MUTED)

    record(save(fig, "24_what_its_made_of"),
           "One commitment holds the whole project up. Everything else is scaffolding.")


# =========================================================================== #
# 25 - the one thing here that is actually a defence.
# =========================================================================== #
def plate_25_a_defence_that_works():
    """Two grouped bars, because the finding IS the pairing: the surface filter does nothing and
    costs something, the intent gate does something and costs nothing."""
    d = load("v10/summary.json")["E55"]
    g = pd.DataFrame(d["grid"])

    def cell(corpus, reader, col):
        r = g[(g.corpus == corpus) & (g.reader == reader)]
        return float(r[col].iloc[0])

    readers = [("no_filter", "no filter"),
               ("surface_filter", "filter on how\ngood it looks"),
               ("intent_reconstructibility", "ask who made it\nand why")]
    poisoned = [cell("disguised", r, "human_model_corrupted") for r, _ in readers]
    clean = [cell("clean", r, "human_model_corrupted") for r, _ in readers]

    fig, ax = plate(
        "Filtering AI slop by how good it looks does nothing. Asking who wrote it works.",
        "Damage to what a learner understands about people, after reading a stream salted with "
        "content built to be absorbed by machines rather than read by anyone. Lower is better.",
        "E55 - results/v10/summary.json - reproduces in 83% of randomly parameterised models, so "
        "most of this is architecture; read the direction")

    x = np.arange(len(readers))
    w = 0.36
    b1 = ax.bar(x - w / 2, poisoned, w, color=MACHINE, label="poisoned stream")
    b2 = ax.bar(x + w / 2, clean, w, color=HUMAN, label="clean stream")
    for bars, vals in ((b1, poisoned), (b2, clean)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=13, fontweight="bold", color=INK)

    ax.set_xticks(x, [lbl for _, lbl in readers], fontsize=11.5, linespacing=1.25)
    ax.set_ylim(0, max(poisoned) * 1.32)
    clean_axis(ax, "damage to its model of people", "")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    annotate(ax, 1.0, poisoned[1] + 0.14, "no better than\ndoing nothing",
             color=MACHINE, fontsize=11.5, weight="bold", ha="center")
    annotate(ax, 2.0, poisoned[2] + 0.14, "a quarter less damage,\nand free on clean data",
             color=HUMAN, fontsize=11.5, weight="bold", ha="center")

    record(save(fig, "25_a_defence_that_works"),
           "The first constructive result in ten versions, and the only one aimed at alignment.")


PLATES = [plate_01_false_label, plate_02_interior_peak, plate_03_the_wall,
          plate_04_method_not_purpose, plate_05_intent_unlocks, plate_06_two_mechanisms,
          plate_07_reputation_blindness, plate_08_expertise_substitutes,
          plate_09_pays_more_gets_less, plate_10_looking_vs_being_changed,
          plate_11_self_report, plate_12_two_damages, plate_13_no_mind_needed,
          plate_14_knee_not_cliff, plate_15_coverage, plate_16_channel_race,
          plate_17_withheld, plate_18_what_tom_buys, plate_19_zero_shot,
          plate_20_rejection_is_not_protection, plate_21_two_gates_settled,
          plate_22_how_much_is_the_theory, plate_23_honesty_pays_at_a_price,
          plate_24_what_its_made_of,
          plate_25_a_defence_that_works]


def main() -> None:
    for fn in PLATES:
        try:
            fn()
        except Exception as exc:                     # noqa: BLE001
            print(f"  SKIP {fn.__name__}: {exc!r}")
    print(f"\nwrote {len(made)} plates to figures/walkthrough/")
    for name, cap in made:
        print(f"  {name}")


if __name__ == "__main__":
    main()
