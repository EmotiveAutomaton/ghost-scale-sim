"""Card implementations for V14, one module per trunk (spec §5).

THE CARD CONTRACT (unchanged from V13). Every card ``<ID>`` is two functions in its trunk module:

    unit_<ID>(ctx)   -> dict     one independent unit: a (world, repeat) pair (or the single
                                 unit of a "single" card). Returns JSON-able rows and numbers.
    reduce_<ID>(units, ctx) -> verdict dict, written by ``finish``.

Rows a card emits carry the card's declared factor keys (manifest ``factors``) plus ``wid`` and
``rep``; the expected-cell receipt counts the realized crossings per independent unit, and the
sparsest cell must appear in EVERY unit (V13's C06 lesson: bins over a continuous quantity must
be per-unit quantiles or guaranteed levels). The smoke pass enforces the receipt.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import verdict_dir
from ..schemas import RESOLVED, TIERS, new_verdict
from ..world import World, WorldParams, make_world

SMOKE = bool(os.environ.get("GS_V14_SMOKE"))
_WORLD_CACHE = {}
ACCESS_REGIMES = {"full": ("action", "semantic", "context", "forensic"), "no_forensic": ("action", "semantic", "context"),
                  "artifact_only": ("action", "semantic"), "context_only": ("context",)}


def smoke_mode() -> bool:
    return bool(os.environ.get("GS_V14_SMOKE"))


# --------------------------------------------------------------------------- #
# Worlds and sizes.
# --------------------------------------------------------------------------- #
def world_params_for(lane: str, wid: int, cfg: dict | None = None) -> WorldParams:
    cfg = cfg or {}
    p = WorldParams()
    if lane in ("transfer", "attack"):
        p.vocabulary = 1 + (wid % 3)                     # fresh action vocabularies lead the transfer lineage
    p.n_families = 2 + int(cfg.get("extra_vocabularies", 0))
    p.equifinal_x = int(cfg.get("equifinal_x", 1))
    return p


def world_for(ctx: dict) -> World:
    lane, wid = ctx["lane"], int(ctx["wid"])
    key = (lane, wid, C.obj_sha(ctx.get("cfg", {})))
    if key not in _WORLD_CACHE:
        if len(_WORLD_CACHE) > 8:
            _WORLD_CACHE.clear()
        cfg = ctx.get("cfg", {}) or {}
        p = world_params_for(lane, wid, cfg)
        rng = np.random.default_rng(C.world_seed(lane, wid) + 1)
        _WORLD_CACHE[key] = make_world(wid, lane, params=p, rng=rng)
    return _WORLD_CACHE[key]


def sizes(ctx: dict) -> dict:
    """Per-world sizes from the tier; the smoke flag shrinks them so every card runs in seconds."""
    t = ctx.get("tier") or TIERS["T0"]
    makers = int(round(int(t["makers"]) * float((ctx.get("cfg") or {}).get("makers_x", 1.0))))
    out = {"makers": makers, "readers": max(6, makers // 8), "episodes": 4, "training": max(6, makers // 8),
           "items": 12, "doses": [1, 2, 4, 8]}
    if smoke_mode() or ctx.get("smoke"):
        out.update({"makers": 12, "readers": 4, "training": 4, "items": 8})
    return out


def rng(ctx: dict, tag: str = "") -> np.random.Generator:
    return C.rng_for(ctx["lane"], ctx["card"].id if hasattr(ctx["card"], "id") else ctx["card"], int(ctx["wid"]), int(ctx["rep"]), tag)


def elapsed_hours(ctx: dict) -> float | None:
    """Hours since the window opened, when the runner supplies it (the hour-20 freeze)."""
    return ctx.get("elapsed_hours")


# --------------------------------------------------------------------------- #
# Rows, units, bins.
# --------------------------------------------------------------------------- #
def rows_of(units: list, key: str = "rows") -> list:
    out = []
    for u in units:
        if u and key in u:
            out.extend(u[key])
    return out


def boot(rows: list, value: str, where=None, unit: str = "wid", seed_tag: str = "", draws: int = 500) -> dict:
    return C.hboot(C.by_unit(rows, value, unit, where), np.random.default_rng(C.seed("boot|" + seed_tag)), draws)


def paired(rows: list, value: str, key_a: str, key_b: str, factor: str, where=None, unit: str = "wid",
           seed_tag: str = "", draws: int = 500) -> dict:
    a = C.by_unit(rows, value, unit, lambda r: r.get(factor) == key_a and (where is None or where(r)))
    b = C.by_unit(rows, value, unit, lambda r: r.get(factor) == key_b and (where is None or where(r)))
    return C.paired_hboot(a, b, np.random.default_rng(C.seed("paired|" + seed_tag)), draws)


def ci_abs(rows: list, value: str, where=None, seed_tag: str = "") -> float:
    """|mean| for a null-style gate, but 0.0 when the unit-level bootstrap interval covers zero."""
    b = boot(rows, value, where, seed_tag="ci|" + seed_tag)
    if b["n_units"] == 0 or b["mean"] != b["mean"]:
        return 0.0
    lo, hi = b["interval"]
    if lo <= 0.0 <= hi:
        return 0.0
    return abs(b["mean"])


def ci_pos(rows: list, value: str, where=None, seed_tag: str = "") -> float:
    """Demonstrated POSITIVE mean for a one-sided gate: the lower bootstrap bound, clipped at zero."""
    b = boot(rows, value, where, seed_tag="cp|" + seed_tag)
    if b["n_units"] == 0 or b["mean"] != b["mean"]:
        return 0.0
    lo, hi = b["interval"]
    return float(max(0.0, lo))


def mean_of(rows: list, value: str, where=None) -> float:
    v = [r[value] for r in rows if (where is None or where(r)) and r.get(value) is not None]
    return float(np.mean(v)) if v else float("nan")


def held_out_classifier(X, labels, rng, metric="js") -> float:
    """Held-out-half centroid classifier accuracy (never leave-one-out)."""
    X = np.asarray(X, float)
    L = np.asarray(labels)
    idx = np.arange(len(X))
    perm = rng.permutation(len(X))
    train = np.zeros(len(X), dtype=bool)
    train[perm[: len(X) // 2]] = True
    classes = sorted(set(L.tolist()))
    if len(classes) == 2 and sorted(map(tuple, X[L == classes[0]])) == sorted(map(tuple, X[L == classes[1]])):
        return 0.5
    cents = {c: X[train & (L == c)].mean(axis=0) for c in classes if (train & (L == c)).any()}
    if not cents:
        return float("nan")

    def dist(a, b):
        return C.js(a, b) if metric == "js" else float(np.linalg.norm(a - b))
    correct = [min(cents, key=lambda c: dist(X[i], cents[c])) == L[i] for i in idx[~train]]
    return float(np.mean(correct)) if correct else float("nan")


# --------------------------------------------------------------------------- #
# The seven-gate battery (spec §4.3) from numbers a card supplies.
# --------------------------------------------------------------------------- #
def battery(gr: G.GateReport, live=None, placebo=None, positive=None, surface=None, oracle=None,
            prediction=None, calibration=None) -> G.GateReport:
    """Each argument is a dict of the numbers the gate judges on, or None to skip visibly.
    live: {observed, min}; placebo: {observed, tol}; positive: {observed, expected, tol};
    surface: {accuracy, chance, tol}; oracle: {observed, min}; prediction: {gain, min};
    calibration: {observed, direction ('down'|'up'), reference, tol}."""
    if live is not None:
        gr.live("live:" + live.get("name", "manipulation_moves_the_target"), observed_change=float(live["observed"]),
                min_change=float(live["min"]), detail=live.get("detail", "the intervention changes the intended latent or policy"))
    else:
        gr.skip("live", "live", "not applicable to this card")
    if placebo is not None:
        gr.placebo("placebo:" + placebo.get("name", "irrelevant_intervention_leaves_target"), observed_max_deviation=float(placebo["observed"]),
                   tol=float(placebo["tol"]), detail=placebo.get("detail", "an irrelevant matched intervention does not change the target"))
    else:
        gr.skip("placebo", "placebo", "not applicable to this card")
    if positive is not None:
        gr.positive("positive:" + positive.get("name", "known_manipulation_recovered"), observed=float(positive["observed"]),
                    expected=float(positive["expected"]), tol=float(positive["tol"]), detail=positive.get("detail", "a known readable manipulation is recovered"))
    else:
        gr.skip("positive", "positive", "not applicable to this card")
    if surface is not None:
        gr.positive("surface:" + surface.get("name", "cheap_cues_at_chance"), observed=float(surface["accuracy"]),
                    expected=float(surface["chance"]), tol=float(surface["tol"]),
                    detail=surface.get("detail", "cheap length, count, label or entropy cues cannot solve the card"))
    else:
        gr.skip("surface", "positive", "not applicable to this card")
    if oracle is not None:
        gr.live("oracle:" + oracle.get("name", "target_identifiable_with_correct_variables"), observed_change=float(oracle["observed"]),
                min_change=float(oracle["min"]), detail=oracle.get("detail", "the target is identifiable when the correct variables are supplied"))
    else:
        gr.skip("oracle", "live", "not applicable to this card")
    if prediction is not None:
        gr.live("prediction:" + prediction.get("name", "inferred_object_constrains_held_out"), observed_change=float(prediction["gain"]),
                min_change=float(prediction["min"]), detail=prediction.get("detail", "the inferred object constrains a held-out choice, action or response"))
    else:
        gr.skip("prediction", "live", "not applicable to this card")
    if calibration is not None:
        obs, ref = float(calibration["observed"]), float(calibration["reference"])
        ok = (obs <= ref + float(calibration.get("tol", 0.0))) if calibration.get("direction", "down") == "down" else (obs >= ref - float(calibration.get("tol", 0.0)))
        gr.positive("calibration:" + calibration.get("name", "uncertainty_responds_correctly"), observed=obs, expected=ref,
                    tol=abs(obs - ref) + 1e-9 if ok else 0.0,
                    detail=calibration.get("detail", "uncertainty or abstention responds in the correct direction"))
    else:
        gr.skip("calibration", "positive", "not applicable to this card")
    return gr


def extra_gate(gr: G.GateReport, kind: str, name: str, observed: float, bar: float, direction: str = "min", detail: str = "") -> None:
    """Trunk-specific gates the spec adds (route divergence, equal-ease control, surface collision,
    reward equivalence, unlearnable noise): recorded like any other gate."""
    if direction == "min":
        gr.live(f"{kind}:{name}", observed_change=float(observed), min_change=float(bar), detail=detail or kind)
    else:
        gr.placebo(f"{kind}:{name}", observed_max_deviation=float(observed), tol=float(bar), detail=detail or kind)


# --------------------------------------------------------------------------- #
# Verdict assembly (unchanged from V13, with V14 lanes).
# --------------------------------------------------------------------------- #
def start(card, ctx: dict, hypothesis: str, ceiling: str) -> dict:
    v = new_verdict(card, ctx["lane"], hypothesis, ceiling)
    v["tier"] = ctx.get("tier_name")
    v["units"] = {"n": ctx.get("n_units"), "kind": card.unit_kind}
    return v


def narrative(v: dict, what_happened: str, what_changed: str, rival: str | None = None) -> None:
    v["record"]["what_happened"] = what_happened
    v["record"]["what_changed_in_the_project_world"] = what_changed
    if rival is not None:
        v["record"]["what_remains_a_rival"] = rival


def receipt(v: dict, rows: list, card, ctx: dict) -> dict:
    realized = C.realized_cells(rows, card.factors)
    expected = (ctx.get("expected_cells") or {}).get(card.id, {}).get(ctx["lane"]) if ctx.get("expected_cells") else None
    if expected is None:
        from ..schemas import expected_cells
        expected = expected_cells(card, ctx.get("tier") or TIERS["T0"], ctx["lane"])
        expected["units_required"] = 1 if card.unit_kind == "list" else (ctx.get("n_units") or expected["units"])
    if card.unit_kind != "world":
        # the instantiated template counts worlds; a list card's units are its items and a single
        # card's unit is one, whatever the tier (I01 resolved RESOURCE_BLOCKED at 6 of 2048 without this)
        # (sparsest-cell rule: a list card's cells each live in exactly one unit, so its requirement is one)
        expected = dict(expected, units_required=1 if card.unit_kind == "list" else int(ctx.get("n_units") or 1))
    if ctx.get("smoke"):
        # smoke keeps the receipt honest at its own scale: every cell must appear in every smoke unit
        # (a list card's cells each live in exactly one unit, so its requirement stays one)
        expected = dict(expected, units_required=1 if card.unit_kind == "list" else int(ctx.get("n_units") or 1), min_rows_per_unit=0)
    rec = C.cell_receipt(realized, expected)
    v["cells"] = realized
    v["expected_cell_receipt"] = rec
    v["effective_n"] = {"units": len({(str(r.get("wid")), str(r.get("rep", 0))) for r in rows}), "rows": len(rows)}
    return rec


def decide_state(gr: G.GateReport, resource_blocked: bool = False, void: bool = False) -> str:
    if resource_blocked:
        return "RESOURCE_BLOCKED"
    if void:
        return "VOID"
    return "LANDED" if gr.to_dict()["all_passed"] else "INSTRUMENT_FAILED"


def finish(card, v: dict, gr: G.GateReport, module_file: str, state: str, ctx: dict, closure_reason: str = "",
           pursuit: str | None = None, warrant: str | None = None) -> dict:
    assert state in RESOLVED, state
    v["state"] = state
    v["gates_summary"] = gr.to_dict()
    if not closure_reason and state != "LANDED":
        failed = v["gates_summary"].get("failed_names") or []
        closure_reason = f"{state.lower().replace('_', ' ')}: " + (", ".join(failed) if failed else "no gate named")
    v["closure_reason"] = closure_reason
    if pursuit:
        v["pursuit"] = pursuit
    if warrant:
        v["warrant"] = warrant
    if state == "INSTRUMENT_FAILED":
        v["claim_ceiling"] = "INSTRUMENT_FAILURE"
        v["record"]["claim_ceiling"] = "INSTRUMENT_FAILURE"
        v["warrant"] = "INSTRUMENT_FAILED"
    elif state == "VOID":
        v["claim_ceiling"] = "VOID"
        v["record"]["claim_ceiling"] = "VOID"
        v["warrant"] = "VOID"
    elif state == "RESOURCE_BLOCKED":
        v["claim_ceiling"] = "RESOURCE_BLOCKED"
        v["record"]["claim_ceiling"] = "RESOURCE_BLOCKED"
        v["warrant"] = "RESOURCE_BLOCKED"
    if v.get("expected_cell_receipt") and not v["expected_cell_receipt"].get("ok", True) and state == "LANDED":
        v["state"] = state = "RESOURCE_BLOCKED"
        v["closure_reason"] = "expected-cell receipt not met: " + C.dumps(v["expected_cell_receipt"])
    v["runtime"]["units_wall_s"] = ctx.get("units_wall_s")
    if v.get("runtime_seconds") is None:
        ws = ctx.get("units_wall_s") or 0.0
        v["runtime_seconds"] = round(float(ws if isinstance(ws, (int, float)) else sum(ws)), 3)
    v["runtime"]["units_cpu_s"] = ctx.get("units_cpu_s")
    v["runtime"]["workers"] = ctx.get("workers")
    v["evaluations"] = ctx.get("evaluations")
    out_dir = ctx.get("out_dir")
    lane = ctx["lane"]
    C.write_verdict(card.id, v, gr, module_file, "smoke" if ctx.get("smoke") else lane, out_dir=out_dir,
                    ledger=not ctx.get("smoke") and lane != "pilot")
    return v


def criterion(v: dict, name: str, passed: bool, **numbers) -> None:
    v["results"][f"criterion_{name}"] = {"passed": bool(passed), **{k: (float(x) if isinstance(x, (int, float, np.floating)) else x) for k, x in numbers.items()}}


def pursuit_of(passed: bool | None) -> str:
    if passed is None:
        return "OPENED"
    return "PROMISING" if passed else "STALLED"


# --------------------------------------------------------------------------- #
# Cells: per-unit accumulation by factor key (per-cell means). Per-item statistics that must not
# be averaged travel in side lists, never through Cells (V13's P14/Q03/A01 lesson).
# --------------------------------------------------------------------------- #
class Cells:
    def __init__(self, wid, rep):
        self.wid, self.rep, self._acc = int(wid), int(rep), {}

    def add(self, key: dict, **vals) -> None:
        k = tuple(sorted((str(a), b) for a, b in key.items()))
        a = self._acc.setdefault(k, {"n": 0, "sums": {}})
        a["n"] += 1
        for f, v in vals.items():
            if v is None or (isinstance(v, float) and v != v):
                continue
            a["sums"][f] = a["sums"].get(f, 0.0) + float(v)

    def rows(self) -> list:
        out = []
        for k, a in self._acc.items():
            row = {"wid": self.wid, "rep": self.rep, "n": a["n"], **dict(k)}
            for f, s in a["sums"].items():
                row[f] = s / a["n"]
            out.append(row)
        return out
