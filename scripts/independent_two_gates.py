#!/usr/bin/env python
"""An independent reimplementation of the two-gates result, from the prose description alone.

    python scripts/independent_two_gates.py

-----------------------------------------------------------------------------------------
WHAT THIS FILE IS, AND THE RULE IT WAS WRITTEN UNDER.

This is validation item V-8. The two-gates result is the strongest and most legible thing in the
project: the same machine-made artifact, one word changed in the label, and a large difference in
how much of the maker's supposed meaning the reader takes on. If it replicates in separate code
written from the description, that is the strongest evidence available anywhere in this project. If
it does not, finding out why is worth more than any new experiment.

**THE RULE: this file imports nothing from `ghostscale` and shares no parameter with it.** It
imports numpy and the standard library. Every number in it was chosen here. It does not read the
committed CSVs, it does not use the project's feature count, goal count, opacity values, trust
parameter, or random seeds, and it does not use pymdp or active inference at all. It is a plain
Bayesian reader, written the way somebody would write one after reading the description below and
nothing else.

That is deliberate and it is the whole point. A reimplementation that shared the parameters would
be testing arithmetic. This tests whether the CLAIM is a claim about a mechanism or a claim about
one particular set of numbers.

-----------------------------------------------------------------------------------------
THE DESCRIPTION IT WAS BUILT FROM, quoted so a reader can check that nothing else was used:

  "A dishonest label and genuine depth move the reader by the same mechanism, in opposite
   directions. How far a reader shifts its beliefs tracks how much thought it thinks went in,
   regardless of whether content or a label put that idea there. Machine content labelled honestly
   reads as shallow and moves nobody; the same content passed off as human reads as deeper and
   moves them a lot."

  "The generative crash is a correctly low estimate of how much thinking went in. The trust
   exploit is a falsely high one. A dishonest provenance signal inflates the estimate, the reader
   over-invests, and fabricates to justify the investment."

  "The reader in this build cannot doubt the label it conditioned on, so the multiple is an upper
   bound."

-----------------------------------------------------------------------------------------
HOW IT WAS BUILT FROM THAT, in four sentences.

A maker has a purpose, one of several. It produces a sequence of observations; how much of its
purpose is legible in that sequence depends on how much thinking went in — deep work carries a
signature of its purpose, shallow work carries the same marginal statistics with the purpose
scrambled out. A reader holds beliefs over the purpose and over the depth, updates both by Bayes as
observations arrive, and then integrates what it concluded into its standing view of the world at a
rate set by how much thinking it believes went in. The label acts by shifting the reader's PRIOR
over depth before it sees anything, and cannot subsequently be doubted.

The measured quantity is how far the reader's standing view moves across a run of encounters, and
the reported number is the ratio between the false-label condition and the honest-label condition
on identical machine-made content.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------------------- #
# A world. Every number below is chosen here and shares nothing with the original.
# ------------------------------------------------------------------------------------------- #
N_PURPOSES = 3          # the original uses four; three is enough to have a wrong answer available
N_SIGNS = 6             # observable marks a work can carry
N_DEPTHS = 3            # how many levels of the maker's thinking reach the surface
N_MARKS = 9             # marks in one work
N_ENCOUNTERS = 5        # the reader sees several works from the same source
LEGIBILITY = 0.72       # how sharply a purpose shows up in a deeply made work
FOREIGN_LEAK = 0.12     # how much human-shaped structure machine work carries


def purpose_signatures(rng: np.random.Generator) -> np.ndarray:
    """p(sign | purpose) for a work whose purpose is fully legible. One block each, plus a floor."""
    sig = np.full((N_PURPOSES, N_SIGNS), (1.0 - LEGIBILITY) / N_SIGNS)
    block = N_SIGNS // N_PURPOSES
    for k in range(N_PURPOSES):
        sig[k, k * block:(k + 1) * block] += LEGIBILITY / block
    return sig / sig.sum(axis=1, keepdims=True)


def depth_mixture(sig: np.ndarray, depth: int) -> np.ndarray:
    """p(sign | purpose, depth).

    Depth is how much of the purpose survives into the surface. At the shallowest level the marks
    are the AVERAGE over purposes — the same overall statistics, with the purpose scrambled out —
    which is the property the original insists on: a reader who counts marks and ignores which
    purpose they belong to cannot tell a deep work from a shallow one by its histogram alone.
    """
    w = depth / (N_DEPTHS - 1)
    flat = sig.mean(axis=0, keepdims=True)
    out = w * sig + (1.0 - w) * flat
    return out / out.sum(axis=1, keepdims=True)


def machine_signatures(sig: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Machine-made work: a real purpose pursued in a vocabulary the reader has no entry for.

    Built as mostly-its-own structure with a little human-shaped leakage, which is the "content
    the reader cannot express" the description turns on. Note this is drawn once and frozen, so
    every condition reads the SAME content and only the label differs.
    """
    own = rng.dirichlet(np.full(N_SIGNS, 0.3), size=N_PURPOSES)
    mixed = (1.0 - FOREIGN_LEAK) * own + FOREIGN_LEAK * sig
    return mixed / mixed.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------------------------------- #
# A reader.
# ------------------------------------------------------------------------------------------- #
LABEL_DEPTH_PRIOR = {
    # A label saying a person made this leads the reader to expect more thinking behind it.
    "passed_off_as_human": np.array([1.0, 2.0, 5.0]),
    "labelled_honestly":   np.array([5.0, 2.0, 1.0]),
    "no_label":            np.array([1.0, 1.0, 1.0]),
}


def read_one_work(marks: np.ndarray, sig: np.ndarray, purpose_prior: np.ndarray,
                  depth_prior: np.ndarray) -> tuple:
    """Exact joint Bayes over (purpose, depth) from a sequence of marks.

    No approximation and no variational anything: the joint is 3 x 3 and is written out in full.
    """
    log_joint = np.log(np.outer(purpose_prior, depth_prior))
    for depth in range(N_DEPTHS):
        like = depth_mixture(sig, depth)          # (purpose, sign)
        log_joint[:, depth] += np.sum(np.log(like[:, marks]), axis=1)
    log_joint -= log_joint.max()
    joint = np.exp(log_joint)
    joint /= joint.sum()
    return joint.sum(axis=1), joint.sum(axis=0)   # (purpose posterior, depth posterior)


def run_condition(label: str, content: str, seed: int) -> dict:
    """One reader, several encounters with one source, and how far its view moved."""
    rng = np.random.default_rng(seed)
    sig = purpose_signatures(rng)
    machine = machine_signatures(sig, rng)
    source_purpose = int(rng.integers(N_PURPOSES))

    emit = sig if content == "human" else machine
    true_depth = N_DEPTHS - 1 if content == "human" else 0

    # The reader's standing view of what people are for. Slightly non-uniform per reader, which is
    # what makes disagreement between readers possible at all.
    standing = rng.dirichlet(np.full(N_PURPOSES, 40.0))
    standing_start = standing.copy()
    depth_prior = LABEL_DEPTH_PRIOR[label] / LABEL_DEPTH_PRIOR[label].sum()

    believed_depths = []
    for _ in range(N_ENCOUNTERS):
        if content == "human":
            emit_row = depth_mixture(sig, true_depth)[source_purpose]
        else:
            emit_row = machine[source_purpose]
        marks = rng.choice(N_SIGNS, size=N_MARKS, p=emit_row)
        purpose_post, depth_post = read_one_work(marks, sig, standing, depth_prior)

        # THE MECHANISM UNDER TEST, in one line: how much of what you concluded you keep is set by
        # how much thinking you believe went into it.
        believed_depth = float(np.dot(depth_post, np.arange(N_DEPTHS)) / (N_DEPTHS - 1))
        believed_depths.append(believed_depth)
        standing = (1.0 - believed_depth) * standing + believed_depth * purpose_post
        standing /= standing.sum()

    shift = float(np.sum(standing * np.log(standing / np.clip(standing_start, 1e-12, None))))
    return {
        "label": label, "content": content, "seed": seed,
        "believed_depth": float(np.mean(believed_depths)),
        "view_shift_nats": shift,
        "ended_on_a_purpose_the_source_did_not_have": int(int(np.argmax(standing))
                                                          != source_purpose),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--readers", type=int, default=400)
    ap.add_argument("--out", default=None, help="write the verdict JSON here")
    args = ap.parse_args()

    rows = []
    for content in ("machine", "human"):
        for label in ("passed_off_as_human", "labelled_honestly", "no_label"):
            for r in range(args.readers):
                rows.append(run_condition(label, content, seed=90_000 + 977 * r))

    def cell(content, label, field):
        vals = [r[field] for r in rows if r["content"] == content and r["label"] == label]
        return float(np.mean(vals))

    lie_shift = cell("machine", "passed_off_as_human", "view_shift_nats")
    honest_shift = cell("machine", "labelled_honestly", "view_shift_nats")
    lie_depth = cell("machine", "passed_off_as_human", "believed_depth")
    honest_depth = cell("machine", "labelled_honestly", "believed_depth")
    multiple = lie_shift / honest_shift if honest_shift > 0 else float("inf")

    # Does the update track believed depth whichever channel moved it? Across all six cells.
    xs, ys = [], []
    for content in ("machine", "human"):
        for label in ("passed_off_as_human", "labelled_honestly", "no_label"):
            xs.append(cell(content, label, "believed_depth"))
            ys.append(cell(content, label, "view_shift_nats"))
    order_x = np.argsort(np.argsort(xs)).astype(float)
    order_y = np.argsort(np.argsort(ys)).astype(float)
    order_x -= order_x.mean()
    order_y -= order_y.mean()
    den = float(np.sqrt((order_x ** 2).sum() * (order_y ** 2).sum()))
    rho = float((order_x * order_y).sum() / den) if den > 0 else float("nan")

    verdict = {
        "what_this_is": ("an independent reimplementation of the two-gates result, from the prose "
                         "description alone, sharing no code and no parameter with the original"),
        "readers_per_cell": args.readers,
        "own_parameters": {"purposes": N_PURPOSES, "signs": N_SIGNS, "depths": N_DEPTHS,
                           "marks_per_work": N_MARKS, "encounters": N_ENCOUNTERS,
                           "legibility": LEGIBILITY, "machine_leak": FOREIGN_LEAK},
        "believed_depth_under_false_label": lie_depth,
        "believed_depth_under_honest_label": honest_depth,
        "view_shift_under_false_label": lie_shift,
        "view_shift_under_honest_label": honest_shift,
        "uptake_multiple": multiple,
        "update_tracks_believed_depth_rho": rho,
        "direction_matches": bool(lie_shift > honest_shift and lie_depth > honest_depth),
        "limitation_reproduced": ("the reader here also cannot doubt the label it conditioned on, "
                                 "so this multiple is an upper bound for the same reason the "
                                 "original's is. That was not a design choice. It is what the "
                                 "description says the reader does, and reimplementing the "
                                 "description reproduces the limitation along with the result"),
        "cells": rows[:0] or [
            {"content": c, "label": l,
             "believed_depth": cell(c, l, "believed_depth"),
             "view_shift_nats": cell(c, l, "view_shift_nats"),
             "wrong_purpose_rate": cell(c, l, "ended_on_a_purpose_the_source_did_not_have")}
            for c in ("machine", "human")
            for l in ("passed_off_as_human", "labelled_honestly", "no_label")],
    }
    text = json.dumps(verdict, indent=2, default=float)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
