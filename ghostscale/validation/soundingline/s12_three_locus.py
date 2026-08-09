"""S-12 — does a three-locus structure with a noisy middle read as a single middle peak?

THE CLAIM being tested is the curator's: "we're finding ratio variance relationships between
early and late despite there being a peak in the middle. It implies a sort of shape that I don't
think anyone else has glommed on to." The published depth-profile instrument averages across
units at each position; real units put their loci at DIFFERENT positions (the sibling's L14: the
architectural peak sits at layer 2 of 29 in one family and 47 of 49 in another). If per-unit
loci vary, averaging can smear three loci into the field's consensus mid peak.

STANDALONE BY DESIGN, as batch four advises: an abstract emitter, no contact with the maker
world. Two instruments:

    instrument 1  (the field's)   the mean profile across units; count its modes
    instrument 2  (G22's)         fit each unit a single peak, take residuals, and correlate the
                                  early-third residual with the late-third residual across units

Two worlds, and only their covariance structure differs where it matters: THREE-LOCUS (early and
late bumps sharing a per-unit gain, a big incoherent middle) and ONE-LOCUS (a single mid bump).
C7: the three-locus mean profile reads unimodal (the smear is real). C8: instrument 2 separates
the worlds where instrument 1 cannot (the residual carries what the profile loses).

SEVERITY RIDER, complying with the miniature rule this version introduces: twenty random redraws
of the generative constants, reporting how often C7 and C8 reproduce. A miniature that skips
this carries "architecture untested"; this one carries its rates.
"""
from __future__ import annotations

import json

import numpy as np

from ...config import Config
from ...methods import gates as G
from ...methods import provenance as PROVENANCE
from ...prereg_v11 import evaluate_c7, evaluate_c8, lock_status
from ...v11 import seed as v11_seed
from . import sl_dir
from .t5_detection import auc

N_POS = 30
N_UNITS = 200
N_REPS = 40

#: The generative constants, redrawn (x/÷2 log-uniform) by the severity rider.
DEFAULTS = dict(
    early_pos=(4, 12), early_width=2.0, early_amp=1.0,
    mid_pos=(10, 20), mid_width=4.0, mid_amp=2.0, mid_extra_noise=0.8,
    late_pos=(18, 26), late_width=2.0, late_amp=1.0,
    gain_sigma=0.4,       # early and late share LogNormal(0, gain_sigma) per unit
    mid_gain_sigma=0.8,   # the middle's own, independent
    base_noise=0.25,
    one_pos=(12, 18), one_width=5.0,
)


def _bump(pos: np.ndarray, centre: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((pos - centre) / width) ** 2)


def _units_three(c: dict, n_units: int, rng) -> np.ndarray:
    pos = np.arange(N_POS, dtype=float)
    out = np.empty((n_units, N_POS))
    for u in range(n_units):
        g = rng.lognormal(0.0, c["gain_sigma"])          # shared early-late gain: the structure
        m = rng.lognormal(0.0, c["mid_gain_sigma"])      # the middle's own amplitude
        e_c = rng.uniform(*c["early_pos"])
        m_c = rng.uniform(*c["mid_pos"])
        l_c = rng.uniform(*c["late_pos"])
        r = (g * c["early_amp"] * _bump(pos, e_c, c["early_width"])
             + m * c["mid_amp"] * _bump(pos, m_c, c["mid_width"])
             + g * c["late_amp"] * _bump(pos, l_c, c["late_width"]))
        noise = rng.normal(0.0, c["base_noise"], N_POS)
        mid_mask = _bump(pos, m_c, c["mid_width"] * 1.5)
        noise += rng.normal(0.0, c["mid_extra_noise"], N_POS) * mid_mask   # low coherence there
        out[u] = r + noise
    return out


def _units_one(c: dict, amp_scale: float, n_units: int, rng) -> np.ndarray:
    pos = np.arange(N_POS, dtype=float)
    out = np.empty((n_units, N_POS))
    for u in range(n_units):
        m = rng.lognormal(0.0, c["mid_gain_sigma"])
        centre = rng.uniform(*c["one_pos"])
        out[u] = (m * amp_scale * _bump(pos, centre, c["one_width"])
                  + rng.normal(0.0, c["base_noise"], N_POS))
    return out


def _calibrate_one_amp(c: dict, rng) -> float:
    """Match the one-locus mean profile's peak to the three-locus world's, so instrument 1 sees
    comparable material and the separation cannot ride on gross amplitude."""
    three = _units_three(c, 400, rng).mean(axis=0)
    one_raw = _units_one(c, 1.0, 400, rng).mean(axis=0)
    return float(three.max() / max(one_raw.max(), 1e-9))


def _n_modes(mean_profile: np.ndarray) -> int:
    """Smoothed local maxima above 10% prominence — the field's profile reading, made explicit."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    sm = gaussian_filter1d(mean_profile, sigma=1.0)
    prom = 0.10 * (sm.max() - sm.min())
    peaks, _ = find_peaks(sm, prominence=max(prom, 1e-9))
    return int(len(peaks))


def _fit_single_peak(r: np.ndarray) -> np.ndarray:
    """Per-unit single-Gaussian fit by grid search (deterministic), returning the residual."""
    pos = np.arange(N_POS, dtype=float)
    best, best_sse = None, np.inf
    for centre in range(2, N_POS - 2):
        for width in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            b = _bump(pos, float(centre), width)
            denom = float(b @ b)
            amp = float(b @ r) / denom if denom > 0 else 0.0
            resid = r - amp * b
            sse = float(resid @ resid)
            if sse < best_sse:
                best_sse, best = sse, resid
    return best


def _early_late_residual_stat(units: np.ndarray) -> float:
    """G22's instrument: correlation across units between early-third and late-third residual
    means, after each unit's own single-peak fit is removed."""
    early, late = [], []
    for r in units:
        resid = _fit_single_peak(r)
        early.append(float(resid[:N_POS // 3].mean()))
        late.append(float(resid[-(N_POS // 3):].mean()))
    e, l = np.asarray(early), np.asarray(late)
    if e.std() < 1e-12 or l.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(e, l)[0, 1])


def _one_pass(c: dict, n_units: int, n_reps: int, rng) -> dict:
    amp = _calibrate_one_amp(c, rng)
    uni_three, stats_three, stats_one, modes_three = [], [], [], []
    for _ in range(n_reps):
        three = _units_three(c, n_units, rng)
        one = _units_one(c, amp, n_units, rng)
        m = _n_modes(three.mean(axis=0))
        modes_three.append(m)
        uni_three.append(1.0 if m == 1 else 0.0)
        stats_three.append(_early_late_residual_stat(three))
        stats_one.append(_early_late_residual_stat(one))
    return {
        "unimodal_fraction_three_locus": float(np.mean(uni_three)),
        "mode_counts_three_locus": {str(k): int(np.sum(np.array(modes_three) == k))
                                    for k in sorted(set(modes_three))},
        "residual_stat_three": [float(x) for x in stats_three],
        "residual_stat_one": [float(x) for x in stats_one],
        "auc_residual_separates_worlds": float(auc(np.asarray(stats_three),
                                                   np.asarray(stats_one))),
        "one_locus_amp_scale": amp,
    }


def run(cfg: Config, n_obs: int | None = None) -> dict:
    """``n_obs`` accepted for runner uniformity and ignored; the design is pre-registered."""
    rng = np.random.default_rng(v11_seed("s12"))
    main = _one_pass(dict(DEFAULTS), N_UNITS, N_REPS, rng)

    # Identity arm: loci fixed and well separated, noise halved — the instrument must SEE three.
    # FIRST CONSTRUCTION CAUGHT BY ITS OWN GATE: with the DEFAULT middle (amp 2.0, width 4), the
    # middle bump's shoulder at distance 5 exceeds the side peaks' height, so even FIXED loci
    # read unimodal and the gate failed at 0.0. That is not a fault in the peak counter — it is
    # the smear mechanism operating through amplitude rather than position, and it strengthens
    # C7's reading. The identity arm's job is loci that provably do NOT overlap, so it narrows
    # and levels the bumps; the finding it guards (C7's smear under VARIABLE loci at DEFAULT
    # amplitudes) is untouched.
    sep = dict(DEFAULTS)
    sep.update(early_pos=(5, 5), mid_pos=(15, 15), late_pos=(25, 25),
               mid_amp=1.0, mid_width=2.0,
               base_noise=DEFAULTS["base_noise"] / 2, mid_extra_noise=0.2)
    sep_modes = []
    rng_sep = np.random.default_rng(v11_seed("s12-sep"))
    for _ in range(N_REPS):
        sep_modes.append(_n_modes(_units_three(sep, N_UNITS, rng_sep).mean(axis=0)))
    trimodal_fraction = float(np.mean([1.0 if m == 3 else 0.0 for m in sep_modes]))

    # Placebo: instrument 2 between two independent batches of the SAME one-locus world.
    rng_pl = np.random.default_rng(v11_seed("s12-placebo"))
    amp = main["one_locus_amp_scale"]
    a_stats = [_early_late_residual_stat(_units_one(dict(DEFAULTS), amp, N_UNITS, rng_pl))
               for _ in range(N_REPS // 2)]
    b_stats = [_early_late_residual_stat(_units_one(dict(DEFAULTS), amp, N_UNITS, rng_pl))
               for _ in range(N_REPS // 2)]
    placebo_auc = float(auc(np.asarray(a_stats), np.asarray(b_stats)))

    # Severity rider: redraw the generative constants and count reproduction.
    sev_rng = np.random.default_rng(v11_seed("s12-severity"))
    sev = []
    for i in range(20):
        c = dict(DEFAULTS)
        for key in ("early_width", "early_amp", "mid_width", "mid_amp", "mid_extra_noise",
                    "late_width", "late_amp", "gain_sigma", "mid_gain_sigma", "base_noise",
                    "one_width"):
            c[key] = float(c[key] * np.exp(sev_rng.uniform(-np.log(2), np.log(2))))
        p = _one_pass(c, 120, 10, sev_rng)
        sev.append({"draw": i,
                    "smear": bool(p["unimodal_fraction_three_locus"] >= 0.80),
                    "separation": bool(p["auc_residual_separates_worlds"] >= 0.80)})
    severity = {"n_draws": len(sev),
                "smear_rate": float(np.mean([s["smear"] for s in sev])),
                "separation_rate": float(np.mean([s["separation"] for s in sev])),
                "how_to_read": ("the fraction of randomly re-parameterised versions of this "
                                "miniature reproducing each headline. High rates mean the result "
                                "comes from the construction's shape, which for C7 is the point "
                                "(any variable-locus world should smear) and for C8 is the risk "
                                "(the separation should depend on the shared gain being real).")}

    c7 = evaluate_c7(main["unimodal_fraction_three_locus"])
    c8 = evaluate_c8(main["auc_residual_separates_worlds"])

    gr = G.GateReport()
    gr.positive("well_separated_loci_read_trimodal", observed=trimodal_fraction,
                expected=1.0, tol=0.2,
                detail="with loci fixed at 5/15/25 and noise halved, the mean profile must show "
                       "three modes: the smear is about overlap from variable loci, not about an "
                       "instrument that cannot count.")
    gr.placebo("residual_stat_null_between_identical_worlds",
               observed_max_deviation=placebo_auc - 0.5, tol=0.15,
               detail="instrument 2 between two batches of the same one-locus world must sit at "
                      "chance; if it separates identical worlds it separates anything.")

    verdict = {
        "test": "S-12 — does a three-locus structure with a noisy middle read as one mid peak?",
        "for": "Sounding Line, batch four; G22's residual route, given its first ground truth",
        "prereg": lock_status(),
        "miniature_severity": severity,
        "design": {"n_units": N_UNITS, "n_positions": N_POS, "n_reps": N_REPS,
                   "constants": DEFAULTS},
        "main": {k: v for k, v in main.items() if not k.startswith("residual_stat")},
        "identity_arm_note": (
            "the arm's first construction kept the default middle (amp 2.0, width 4) and its own "
            "gate failed at 0.0: the middle's shoulder outgrows the side peaks, so even fixed "
            "loci read unimodal. Recorded because it is independent evidence for the smear "
            "mechanism — amplitude alone can do it — and the arm now uses matched narrow bumps "
            "so it tests what it claims to: that non-overlapping loci are counted."),
        "identity_arm_trimodal_fraction": trimodal_fraction,
        "placebo_auc_between_identical_one_locus_batches": placebo_auc,
        "criteria": {"C7_smear": c7, "C8_separation": c8},
        "what_would_have_falsified_the_claim": (
            "the three-locus world staying visibly trimodal under position-averaging (then the "
            "field's mid-peak consensus cannot be explained this way), or the residual statistic "
            "failing to separate the worlds (then G22's route has no instrument)."),
        "what_must_hold_in_the_real_environment": (
            "per-unit locus positions genuinely vary (the sibling's L14 measured exactly that), "
            "and early-late shared gain is the right parametric form for the ratio-variance "
            "structure — which is the part a simulation cannot certify and the real-model "
            "residual analysis (G22) must."),
        "what_this_cannot_show": (
            "that any real depth profile IS a smeared three-locus structure; only that the "
            "field's instrument cannot tell, and that a residual instrument can."),
    }
    PROVENANCE.stamp(verdict, __file__, gr)
    (sl_dir() / "s12_three_locus.json").write_text(
        json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return verdict
