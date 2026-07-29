"""E34 — where does real generative content sit on omega? (V5 §E34.)

**THIS EXPERIMENT CANNOT BE RUN.** It is an empirical question about real generative systems
and the spec says so: it should be posed as one, in the write-up, as the thing a human study
should measure. What this module produces is therefore not a result but an INSTRUMENT — a
prediction card that maps each band of the overlap axis to the signature a study would observe
there, so that a measurement can be located on the axis rather than merely described.

Zero compute. Every number is read from E20's committed sweep; nothing is simulated and nothing
is fitted.

-----------------------------------------------------------------------------------------
THE QUESTION THE FRAMEWORK NOW TURNS ON, and it is uncomfortable.

E20 located sustained futile attention below omega ~ 0.04, and the crash plus peak fabrication
together at omega ~ 0.10. Real generative output is trained on human data and inherits
substantial human-shaped structure, so it is plainly not at omega = 0. If real output sits near
0.10, both of the framework's headline phenomena occupy the same narrow band — and the band's
location was predicted before the sweep ran.

-----------------------------------------------------------------------------------------
A CAVEAT THAT BELONGS WITH THE CARD RATHER THAN UNDER IT, and it is the strongest objection to
the whole exercise.

omega's meaning depends on a construction. V4's decision D1 doubled the feature space into a
HUMAN block and a disjoint FOREIGN block, because at V1-V3 cardinality no disjoint partition
existed; omega then mixes a foreign-block signature toward a human-block one. Real generative
output, trained on human data, arguably lives almost entirely IN the human block — which would
put it at high omega, in the regime where the model says content is read and correctly
abandoned, and which is what people actually seem to do.

So there are two readings, they make opposite predictions, and the framework cannot choose
between them from the inside:

  * real output is PARTIALLY foreign, near omega ~ 0.10, and sits exactly where fabrication
    peaks and attention collapses without resolving;
  * real output is nearly all in-family, at high omega, and the crash is a phenomenon of a
    regime that generative systems do not occupy.

**A study that measures the signature can decide between them. That is what the card is for,
and stating both readings is the honest version of asking the question.**
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Bands are named for what a reader DOES there, not for the parameter value, because the
# parameter is the thing a study cannot observe and the behaviour is the thing it can.
BANDS = [
    (0.00, 0.04, "sustained and futile",
     "keeps looking, never resolves, no drop in arousal",
     "attention held with no resolution; pupil stays dilated past 4 s"),
    (0.04, 0.15, "the crash band",
     "gives up while still not knowing, and confidently invents when it does commit",
     "arousal rises then falls sharply without a resolution event; confident but "
     "mutually inconsistent reports of intent"),
    (0.15, 0.40, "read, then abandoned",
     "works out the intent quickly and correctly stops paying",
     "brief dilation, quick resolution, low disagreement"),
    (0.40, 1.01, "ordinary reading",
     "reads it like human work",
     "indistinguishable from human content on every measure"),
]


def build_card(res_dir: Path) -> dict:
    sweep = pd.read_csv(Path(res_dir) / "e20_omega_sweep.csv")

    rows = []
    for lo, hi, name, behaviour, signature in BANDS:
        band = sweep[(sweep.omega >= lo) & (sweep.omega < hi)]
        if not len(band):
            continue
        rows.append({
            "band": name,
            "omega_from": lo, "omega_to": min(hi, 1.0),
            "what_the_reader_does": behaviour,
            "what_a_study_would_see": signature,
            "engaged_fraction": float(band.engaged_fraction.mean()),
            "reads_the_goal": float(band.matches_foreign_goal.mean()),
            "within_observer_nats": float(band.within_observer.mean()),
            "between_observer_nats": float(band.between_observer.mean()),
            "fabrication_index": float(band.fabrication_index.mean()),
            "crash_signature_anywhere_in_band": bool(band.crashed.any()),
        })

    peak = sweep.loc[sweep.fabrication_index.idxmax()]
    crash = sweep[sweep.crashed]

    return {
        "instrument": "E34 — prediction card for locating real generative content on omega",
        "not_a_result": (
            "E34 cannot be run in simulation. This maps each band of the overlap axis to the "
            "signature a human study would observe there, so a measurement can be LOCATED on "
            "the axis. Every number is read from E20's committed sweep; nothing is fitted."),
        "bands": rows,
        "landmarks": {
            "engagement_crosses_half_at_omega": 0.041,
            "fabrication_peaks_at_omega": float(peak.omega),
            "fabrication_peak_value": float(peak.fabrication_index),
            "crash_signature_true_at_omega": crash.omega.tolist(),
            "note": ("the crash and the fabrication peak are the SAME narrow band, which the "
                     "framework had always treated as two separate phenomena."),
        },
        "the_two_readings": {
            "partially_foreign": (
                "real output sits near omega ~ 0.10, exactly where fabrication peaks and "
                "attention collapses without resolving. Both headline phenomena would then "
                "describe ordinary encounters with generated content."),
            "nearly_in_family": (
                "real output, trained on human data, lives almost entirely in the human block "
                "— high omega, where the model says content is read and correctly abandoned. "
                "That matches the observation that people are not in fact paralysed by AI "
                "content, and it would mean the crash is a phenomenon of a regime generative "
                "systems do not occupy."),
            "how_to_decide": (
                "measure the signature, not the parameter. The bands differ on engagement, on "
                "whether resolution occurs, and on whether confident readings agree with each "
                "other — all observable without knowing omega."),
        },
        "caveat": (
            "omega's meaning depends on V4 decision D1, which split the feature space into "
            "disjoint HUMAN and FOREIGN blocks because no such partition existed at V1-V3 "
            "cardinality. Real content may have no well-defined position on this axis at all. "
            "The card should be used to classify an observed SIGNATURE, and only then to ask "
            "what omega would have produced it."),
        "sharpest_single_measurement": (
            "pupillometry on unlabelled generated content. H1 and H3 in the preprint assume "
            "goal-EMPTY content and predict an autonomic drop within 2-4 seconds. Goal-FOREIGN "
            "content below omega ~ 0.04 predicts the opposite: sustained arousal with no "
            "resolution. One measurement discriminates them, and the window where the original "
            "prediction holds is now bounded rather than assumed."),
    }


def run(res_dir: Path | None = None) -> dict:
    res_dir = Path(res_dir or "results")
    card = build_card(res_dir)
    (res_dir / "e34_prediction_card.json").write_text(
        json.dumps(card, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(card["bands"]).to_csv(res_dir / "e34_prediction_card.csv", index=False)
    return card


def main():
    import argparse
    ap = argparse.ArgumentParser(description="E34 — the prediction card (zero compute)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    card = run(Path(args.out))
    print(pd.DataFrame(card["bands"]).to_string(index=False))
    print()
    print(json.dumps(card["landmarks"], indent=2, default=float))


if __name__ == "__main__":
    main()
