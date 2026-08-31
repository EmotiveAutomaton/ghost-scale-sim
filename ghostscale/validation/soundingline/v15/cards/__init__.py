"""Card implementations for V15, one module per trunk (spec §6).

THE CARD CONTRACT. Every card ``<ID>`` is two functions in its trunk module::

    unit_<ID>(ctx)          -> dict     one independent unit: a (world, repeat) pair, or the single
                                        unit of a "single" card, or one item of a "list" card.
    reduce_<ID>(units, ctx) -> dict     the verdict, written by ``finish``.

Rows carry the card's declared factor keys plus ``wid`` and ``rep``; the expected-cell receipt
counts realized crossings per independent unit.

The one rule this module enforces rather than documents
--------------------------------------------------------
**A gate bar is never a criterion bar.** ``battery`` takes only an *observed* number for its
live, positive, prediction and oracle gates and always judges it against zero: the gate asks
whether the effect exists at all. The pre-registered magnitude lives in ``criterion`` and nowhere
else. V14 conflated the two on twenty-six gate lines, and the consequence was not cosmetic -- three
small *real* effects (a +0.011-nat joint advantage, a +0.009-nat routing advantage) were recorded
as INSTRUMENT_FAILED at scale, because a gate demanding the criterion's magnitude fails exactly
when the finding is a modest true positive. The repair had to be made mid-window and entered as an
amendment. Here the separation is structural: ``battery`` has no parameter that could carry a
magnitude, and ``tests/test_v15_gates.py`` fails the suite if any committed gate's expected value
is nonzero for those kinds.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .....methods import gates as G
from .. import common as C
from .. import verdict_dir
from ..causal_distance import distance_receipt
from ..ontology import Knobs
from ..schemas import RESOLVED, TIERS, new_verdict

SMOKE = bool(os.environ.get("GS_V15_SMOKE"))
_WORLD_CACHE: dict = {}

#: Which endpoint each family scores by default.
ENDPOINT = {"chain": "next_action", "composition": "next_edit",
            "communication": "next_evidence_selection"}
#: Access regimes used across the atlas trunk.
ACCESS = {"full": None, "no_forensic": ("action", "semantic", "context"),
          "artifact_only": ("action", "semantic"), "context_only": ("context",)}


def smoke_mode() -> bool:
    return SMOKE or bool(os.environ.get("GS_V15_SMOKE"))


def family_module(name: str):
    from .. import world_chain, world_communication, world_composition
    return {"chain": world_chain, "composition": world_composition,
            "communication": world_communication}[name]


# --------------------------------------------------------------------------- #
# Worlds and sizes.
# --------------------------------------------------------------------------- #
def knobs_for(ctx: dict, **over) -> Knobs:
    cfg = dict(ctx.get("cfg") or {})
    base = {"kappa": 0.5, "overlap": 0.33, "dose": 4, "temperature": 0.6, "competence": 0.85}
    base.update({k: v for k, v in cfg.items() if k in Knobs.__dataclass_fields__})
    base.update(over)
    return Knobs(**base)


def world_for(ctx: dict, family: str = "chain", **over):
    """Cached per (lane, wid, family, knobs). Worlds are the expensive object in most cards."""
    F = family_module(family)
    k = knobs_for(ctx, **over)
    key = (ctx["lane"], int(ctx["wid"]), family, k.key())
    if key not in _WORLD_CACHE:
        if len(_WORLD_CACHE) > 24:
            _WORLD_CACHE.clear()
        lane = ctx["lane"]
        # world_seed asserts that the world id sits inside its lane's range, which is the
        # lineage guarantee for the scientific lanes. A smoke or pilot pass has no lineage to
        # protect and its ids are arbitrary, so it seeds from the same named string directly.
        base = (C.world_seed(lane, int(ctx["wid"])) if lane in ("discovery", "transfer", "attack",
                                                                "confirmation", "coverage")
                else C.seed(f"world|{lane}|{int(ctx['wid'])}"))
        rng = np.random.default_rng(base + 1)
        _WORLD_CACHE[key] = F.sample_world(k, rng)
    return _WORLD_CACHE[key]


def sizes(ctx: dict) -> dict:
    t = ctx.get("tier") or TIERS["T0"]
    out = {"makers": int(t["makers"]), "episodes": int(t.get("episodes", 5)),
           "steps": int(t.get("steps", 12)), "readers": max(4, int(t["makers"]) // 8),
           "training": max(8, int(t["makers"]) // 4), "sims": 16}
    if smoke_mode() or ctx.get("smoke"):
        out.update({"makers": 6, "episodes": 2, "steps": 8, "readers": 3, "training": 4,
                    "sims": 4})
    return out


def rng(ctx: dict, tag: str = "") -> np.random.Generator:
    cid = ctx["card"].id if hasattr(ctx["card"], "id") else ctx["card"]
    return C.rng_for(ctx["lane"], cid, int(ctx["wid"]), int(ctx["rep"]), tag)


def families_of(ctx: dict) -> list:
    """Every family the card declares, in smoke as well as at scale.

    Truncating the family list under smoke was tried and is wrong: a card that declares ``family``
    as a factor then realizes fewer cells than it declared and the receipt blocks it, so the smoke
    pass reports RESOURCE_BLOCKED for a card that is perfectly healthy.
    """
    c = ctx["card"]
    return list(getattr(c, "families", ["chain"]) or ["chain"])


# --------------------------------------------------------------------------- #
# Rows and inference.
# --------------------------------------------------------------------------- #
def rows_of(units: list, key: str = "rows") -> list:
    out = []
    for u in units:
        if u and key in u:
            out.extend(u[key])
    return out


def side_of(units: list, key: str) -> list:
    out = []
    for u in units:
        if u and key in u:
            v = u[key]
            out.extend(v if isinstance(v, list) else [v])
    return out


def boot(rows: list, value: str, where=None, unit: str = "wid", seed_tag: str = "",
         draws: int = 400) -> dict:
    return C.hboot(C.by_unit(rows, value, unit, where),
                   np.random.default_rng(C.seed("boot|" + seed_tag)), draws)


def paired(rows: list, value: str, key_a: str, key_b: str, factor: str, where=None,
           unit: str = "wid", seed_tag: str = "", draws: int = 400) -> dict:
    a = C.by_unit(rows, value, unit,
                  lambda r: r.get(factor) == key_a and (where is None or where(r)))
    b = C.by_unit(rows, value, unit,
                  lambda r: r.get(factor) == key_b and (where is None or where(r)))
    return C.paired_hboot(a, b, np.random.default_rng(C.seed("paired|" + seed_tag)), draws)


def equivalence(rows: list, value: str, key_a: str, key_b: str, factor: str, margin: float,
                where=None, unit: str = "wid", seed_tag: str = "") -> dict:
    a = C.by_unit(rows, value, unit,
                  lambda r: r.get(factor) == key_a and (where is None or where(r)))
    b = C.by_unit(rows, value, unit,
                  lambda r: r.get(factor) == key_b and (where is None or where(r)))
    return C.equivalence(a, b, np.random.default_rng(C.seed("equiv|" + seed_tag)), margin)


def mean_of(rows: list, value: str, where=None) -> float:
    v = [r[value] for r in rows if (where is None or where(r)) and r.get(value) is not None]
    return float(np.mean(v)) if v else float("nan")


def abs_mean(rows: list, value: str, where=None) -> float:
    m = mean_of(rows, value, where)
    return abs(m) if m == m else 0.0


def onset(rows: list, x: str, y: str, bar: float, where=None) -> dict:
    """Where a conditional effect first clears a bar along an ordered axis: the atlas's estimand.

    Returned with the whole curve, because spec §5 forbids a pooled headline over an axis whose
    sign or magnitude changes along it.
    """
    xs = sorted({r[x] for r in rows if (where is None or where(r)) and r.get(x) is not None})
    curve = []
    for xv in xs:
        m = mean_of(rows, y, lambda r: r.get(x) == xv and (where is None or where(r)))
        curve.append({"x": xv, "mean": m})
    first = next((c["x"] for c in curve if c["mean"] == c["mean"] and c["mean"] >= bar), None)
    return {"axis": x, "value": y, "bar": float(bar), "curve": curve, "onset": first,
            "max": max((c["mean"] for c in curve if c["mean"] == c["mean"]), default=float("nan")),
            "monotone": bool(all(curve[i]["mean"] <= curve[i + 1]["mean"] + 1e-9
                                 for i in range(len(curve) - 1)
                                 if curve[i]["mean"] == curve[i]["mean"]
                                 and curve[i + 1]["mean"] == curve[i + 1]["mean"]))}


# --------------------------------------------------------------------------- #
# The gate battery. Every bar here is ZERO by construction (see the module docstring).
# --------------------------------------------------------------------------- #
def battery(gr: G.GateReport, live=None, placebo=None, positive=None, no_label_leak=None,
            surface=None, prediction=None, calibration=None) -> G.GateReport:
    """Build the causal battery from numbers a card supplies.

    ``live``/``prediction``   {"observed": x, "name": ..} -- does the effect exist at all?
    ``placebo``               {"observed": x, "tol": t}   -- a zero-strength manipulation is inert.
    ``positive``              {"observed": x, "expected": e, "tol": t} -- a known answer returns.
    ``no_label_leak``         {"movement": m, "tol": t}   -- a hidden label's permutation is inert.
    ``surface``               {"accuracy": a, "chance": c, "tol": t} -- cheap cues cannot solve it.
    ``calibration``           {"observed": o, "reference": r, "direction": "down"|"up"}.

    There is deliberately no ``min`` parameter on ``live`` or ``prediction``. If you want to say
    "and the effect must be at least this big", that is a criterion, and it goes in ``criterion``.
    """
    if live is not None:
        gr.live("live:" + live.get("name", "manipulation_moves_the_target"),
                observed_change=float(live["observed"]), min_change=0.0,
                detail=live.get("detail", "the intervention moves the intended quantity at all; "
                                          "how far it must move is the card's criterion, not this gate"))
    else:
        gr.skip("live", "live", "not applicable to this card")

    if placebo is not None:
        gr.placebo("placebo:" + placebo.get("name", "zero_strength_is_inert"),
                   observed_max_deviation=float(placebo["observed"]),
                   tol=float(placebo.get("tol", 1e-9)),
                   detail=placebo.get("detail", "a manipulation at zero strength reproduces the control"))
    else:
        gr.skip("placebo", "placebo", "not applicable to this card")

    if positive is not None:
        gr.positive("positive:" + positive.get("name", "known_answer_recovered"),
                    observed=float(positive["observed"]), expected=float(positive["expected"]),
                    tol=float(positive["tol"]),
                    detail=positive.get("detail", "a task whose answer is known by construction returns it"))
    else:
        gr.skip("positive", "positive", "not applicable to this card")

    if no_label_leak is not None:
        gr.no_oracle("no_label_leak:" + no_label_leak.get("name", "hidden_label_permutation_inert"),
                     observed_change=float(no_label_leak["movement"]),
                     tol=float(no_label_leak.get("tol", 0.02)),
                     detail=no_label_leak.get("detail", "permuting a label the reader must not see "
                                                        "does not move the score"))
    else:
        gr.skip("no_label_leak", "no_oracle", "not applicable to this card")

    if surface is not None:
        gr.positive("surface:" + surface.get("name", "cheap_cues_at_chance"),
                    observed=float(surface["accuracy"]), expected=float(surface["chance"]),
                    tol=float(surface["tol"]),
                    detail=surface.get("detail", "cheap length, count or frequency cues cannot solve the card"))
    else:
        gr.skip("surface", "positive", "not applicable to this card")

    if prediction is not None:
        gr.live("prediction:" + prediction.get("name", "inferred_object_constrains_hidden_event"),
                observed_change=float(prediction["observed"]), min_change=0.0,
                detail=prediction.get("detail", "the inferred object moves a genuinely hidden event "
                                                "at all; the required magnitude is the criterion"))
    else:
        gr.skip("prediction", "live", "not applicable to this card")

    if calibration is not None:
        obs, ref = float(calibration["observed"]), float(calibration["reference"])
        tol = float(calibration.get("tol", 0.0))
        ok = (obs <= ref + tol) if calibration.get("direction", "down") == "down" else (obs >= ref - tol)
        gr.positive("calibration:" + calibration.get("name", "uncertainty_responds_correctly"),
                    observed=obs, expected=ref, tol=abs(obs - ref) + 1e-9 if ok else 0.0,
                    detail=calibration.get("detail", "uncertainty or abstention moves in the correct direction"))
    else:
        gr.skip("calibration", "positive", "not applicable to this card")
    return gr


def extra_gate(gr: G.GateReport, kind: str, name: str, observed: float, tol: float = 0.0,
               inert: bool = False, detail: str = "") -> None:
    """A trunk-specific gate. ``inert=True`` asserts the quantity stays near zero; otherwise it
    asserts the quantity is nonzero, and again the bar is zero."""
    if inert:
        gr.placebo(f"{kind}:{name}", observed_max_deviation=float(observed), tol=float(tol),
                   detail=detail or kind)
    else:
        gr.live(f"{kind}:{name}", observed_change=float(observed), min_change=0.0,
                detail=detail or kind)


# --------------------------------------------------------------------------- #
# Verdict assembly.
# --------------------------------------------------------------------------- #
def start(card, ctx: dict, hypothesis: str, claim_class: str) -> dict:
    v = new_verdict(card, ctx["lane"], hypothesis, claim_class)
    v["tier"] = ctx.get("tier_name")
    v["units"] = {"n": ctx.get("n_units"), "kind": card.unit_kind}
    v["criteria"] = []
    return v


def narrative(v: dict, what_happened: str, what_changed: str, rival: str | None = None) -> None:
    v["record"]["what_happened"] = what_happened
    v["record"]["what_changed_in_the_project_world"] = what_changed
    if rival is not None:
        v["record"]["what_remains_a_rival"] = rival


def criterion(v: dict, name: str, observed, bar: float, direction: str = "greater",
              basis: str = "", interval=None, detail: str = "") -> dict:
    """Record one pre-registered criterion. This is the ONLY place a magnitude belongs."""
    c = C.criterion(name, observed, bar, direction, basis, interval, detail)
    v.setdefault("criteria", []).append(c)
    v["results"][f"criterion_{name}"] = c
    return c


def distances(v: dict, card_id: str, channels: list, empirical: dict | None = None) -> dict:
    r = distance_receipt(card_id, channels, empirical)
    v["causal_distance"] = r
    return r


def budgets(v: dict, per_architecture: dict, tolerance: float = 0.25) -> dict:
    r = C.budget_receipt(per_architecture, tolerance)
    v["budgets"] = r
    return r


def publication(v: dict, **kw) -> dict:
    from ..schemas import publication_row
    v["publication"] = publication_row(**kw)
    return v["publication"]


def receipt(v: dict, rows: list, card, ctx: dict) -> dict:
    realized = C.realized_cells(rows, card.factors)
    from ..schemas import expected_cells
    expected = expected_cells(card, ctx.get("tier") or TIERS["T0"], ctx["lane"])
    # Sparsest-cell rule: a list card's cells each live in exactly one unit, so requiring every
    # cell in every unit is arithmetically impossible (V14's I01 blocked twice on this).
    if card.unit_kind != "world":
        expected = dict(expected,
                        units_required=1 if card.unit_kind == "list" else int(ctx.get("n_units") or 1))
    else:
        expected = dict(expected, units_required=int(ctx.get("n_units") or expected["units"]))
    if ctx.get("smoke") or smoke_mode():
        expected = dict(expected,
                        units_required=1 if card.unit_kind == "list" else int(ctx.get("n_units") or 1),
                        min_rows_per_unit=0)
    rec = C.cell_receipt(realized, expected)
    v["cells"] = realized
    v["expected_cell_receipt"] = rec
    v["effective_n"] = {"units": len({(str(r.get("wid")), str(r.get("rep", 0))) for r in rows}),
                        "rows": len(rows)}
    return rec


def decide_state(gr: G.GateReport, resource_blocked: bool = False, void: bool = False) -> str:
    if resource_blocked:
        return "RESOURCE_BLOCKED"
    if void:
        return "VOID"
    return "LANDED" if gr.to_dict()["all_passed"] else "INSTRUMENT_FAILED"


def pursuit_of(criteria: list) -> str:
    st = C.criterion_status(criteria)
    return {"HELD": "PROMISING", "FAILED": "STALLED"}.get(st, "OPENED")


def finish(card, v: dict, gr: G.GateReport, module_file: str, state: str, ctx: dict,
           closure_reason: str = "", pursuit: str | None = None,
           warrant: str | None = None) -> dict:
    assert state in RESOLVED, state
    v["state"] = state
    v["gates_summary"] = gr.to_dict()
    v["criterion_status"] = C.criterion_status(v.get("criteria") or [])
    if not closure_reason and state != "LANDED":
        failed = v["gates_summary"].get("failed_names") or []
        closure_reason = f"{state.lower().replace('_', ' ')}: " + (", ".join(failed) if failed
                                                                   else "no gate named")
    v["closure_reason"] = closure_reason
    v["pursuit"] = pursuit or pursuit_of(v.get("criteria") or [])
    if warrant:
        v["warrant"] = warrant
    if state == "INSTRUMENT_FAILED":
        v["claim_class"] = v["record"]["claim_class"] = "INSTRUMENT_FAILURE"
        v["warrant"] = "INSTRUMENT_FAILED"
    elif state == "VOID":
        v["claim_class"] = v["record"]["claim_class"] = "VOID"
        v["warrant"] = "VOID"
    elif state == "RESOURCE_BLOCKED":
        v["warrant"] = "RESOURCE_BLOCKED"
    # a card whose limiting causal distance is not inference-through-behaviour cannot be a
    # simulator discovery, whatever its numbers say (spec §8.3 clause 6)
    cd = v.get("causal_distance") or {}
    if cd and not cd.get("promotable_as_discovery", True) and v.get("claim_class") == "SIMULATOR_DISCOVERY":
        v["claim_class"] = v["record"]["claim_class"] = "CONSTRUCTION_IDENTITY"
        v.setdefault("deviations", []).append(
            "claim class lowered to CONSTRUCTION_IDENTITY by the causal-distance audit")
    if v.get("expected_cell_receipt") and not v["expected_cell_receipt"].get("ok", True) \
            and state == "LANDED":
        v["state"] = state = "RESOURCE_BLOCKED"
        v["closure_reason"] = "expected-cell receipt not met: " + C.dumps(v["expected_cell_receipt"])
    v["runtime"]["units_wall_s"] = ctx.get("units_wall_s")
    if v.get("runtime_seconds") is None:
        ws = ctx.get("units_wall_s") or 0.0
        v["runtime_seconds"] = round(float(ws if isinstance(ws, (int, float)) else sum(ws)), 3)
    v["runtime"]["units_cpu_s"] = ctx.get("units_cpu_s")
    v["runtime"]["workers"] = ctx.get("workers")
    C.write_verdict(card.id, v, gr, module_file, "smoke" if ctx.get("smoke") else ctx["lane"],
                    out_dir=ctx.get("out_dir"),
                    ledger=not ctx.get("smoke") and ctx["lane"] != "pilot")
    return v


# --------------------------------------------------------------------------- #
# Cells: per-unit accumulation by factor key.
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


# --------------------------------------------------------------------------- #
# The tournament helper every architecture card uses.
# --------------------------------------------------------------------------- #
def run_tournament(ctx, family: str, names, *, knobs_over=None, endpoint=None, n_makers=None,
                   cfg=None, cells: Cells | None = None, extra_key=None, budgets_out=None):
    """Sample a world, roll out makers, read every architecture, and emit rows.

    Returns ``(rows, world, budget_totals)``. The budget totals are what ``budgets`` turns into a
    compute-matching receipt; a card that reports an architecture comparison without one is
    refused at reduce time.
    """
    from .. import architectures as A
    F = family_module(family)
    s = sizes(ctx)
    world = world_for(ctx, family, **(knobs_over or {}))
    ep_name = endpoint or ENDPOINT[family]
    r = rng(ctx, f"tourney|{family}|{ep_name}")
    n = int(n_makers or s["makers"])
    rows = []
    totals = {nm: {"likelihood_evaluations": 0.0, "proposals": 0.0, "observations": 0.0,
                   "cpu_s": 0.0} for nm in names}
    training = [F.rollout(world, F.sample_latent(world, r), r, s["steps"])
                for _ in range(s["training"])] if "direct_predictor" in names else None
    for mi in range(n):
        lat = F.sample_latent(world, r)
        ep = F.rollout(world, lat, r, s["steps"])
        y = ep.hidden.get(ep_name)
        if y is None:
            continue
        upto = min(int(world.knobs.dose), s["steps"])
        reads = A.tournament(F, world, ep, upto, ep_name, names,
                             rng=np.random.default_rng(r.integers(0, 2 ** 62)),
                             cfg=cfg, training=training)
        for nm, rd in reads.items():
            for k in totals[nm]:
                totals[nm][k] += float(rd.budget.get(k, 0))
            row = {"wid": ctx["wid"], "rep": ctx["rep"], "architecture": nm, "family": family,
                   "log_score": C.log_score(rd.dist, y), "brier": C.brier(rd.dist, y),
                   "correct": float(C.top1(rd.dist) == y), "confidence": C.confidence(rd.dist),
                   "n": 1}
            if extra_key:
                row.update(extra_key)
            rows.append(row)
            if cells is not None:
                cells.add({"architecture": nm, **(extra_key or {})},
                          log_score=row["log_score"], correct=row["correct"])
    for nm in totals:
        for k in totals[nm]:
            totals[nm][k] = totals[nm][k] / max(n, 1)
    if budgets_out is not None:
        budgets_out.update(totals)
    return rows, world, totals


def arch_gap(rows: list, a: str, b: str, where=None, value: str = "log_score") -> float:
    """Mean paired advantage of architecture ``a`` over ``b`` on ``value``."""
    return (mean_of(rows, value, lambda r: r.get("architecture") == a and (where is None or where(r)))
            - mean_of(rows, value, lambda r: r.get("architecture") == b and (where is None or where(r))))
