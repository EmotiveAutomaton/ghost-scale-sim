"""Resumable, tier-calibrated runner for V13 (spec §7, §18, §20).

    python runners/run_v13.py --stage prepare        # manifest, expected-cell template, structural lock
    python runners/run_v13.py --stage smoke          # every card once on one world at smoke sizes; verdicts to a scratch dir
    python runners/run_v13.py --stage wave0          # I01-I12, I14 on the discovery lane
    python runners/run_v13.py --stage pilot          # the discarded runtime pilot; selects the tier; writes the locks
    python runners/run_v13.py --stage discovery      # waves 1-3, discovery lane (resumes)
    python runners/run_v13.py --stage transfer       # transfer-lane cards
    python runners/run_v13.py --stage attacks        # the X matrix on the transfer lineage
    python runners/run_v13.py --stage confirmation   # promoted cards on the untouched lineage
    python runners/run_v13.py --stage bridge         # L01-L12
    python runners/run_v13.py --stage report         # packet, coverage, runtime audit
    python runners/run_v13.py --stage all            # everything above in order, resuming what is done

Workers are processes with one BLAS thread each, below normal priority, no accelerator visible;
the parent records parent and child CPU, peak RSS, wall time and throughput per card. Units
checkpoint by (lane, card, world, repeat) and resume is hash-aware: a checkpoint written by a
different source hash is recomputed.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import importlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ghostscale.validation.soundingline.v13 as V                          # noqa: E402
from ghostscale.validation.soundingline.v13 import common as C              # noqa: E402
from ghostscale.validation.soundingline.v13 import manifest as M            # noqa: E402
from ghostscale.validation.soundingline.v13.runtime import init_worker, run_unit  # noqa: E402
from ghostscale.validation.soundingline.v13.schemas import ENVELOPE_HOURS, EXPANSIONS, RESOLVED, TIERS, TIER_ORDER, card_from_dict  # noqa: E402

STATUS = V.V13_RESULTS / "RUNNER_STATUS.json"
PILOT = V.V13_RESULTS / "PILOT.json"
FORECAST = V.V13_RESULTS / "FORECAST.json"
WAVE_ORDER = [0, 1, 2, 3, 4, 5]
TRUNK_PREREQ = {"C": ["I03", "I04"], "A": ["I05"], "O": ["I06"], "P": ["I03"], "G": ["I07", "I08"], "H": ["I09"], "Q": ["I10", "Q08"], "L": []}
OVERHEAD = 1.15


# --------------------------------------------------------------------------- #
# Status and accounting.
# --------------------------------------------------------------------------- #
def status(**kw) -> None:
    doc = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    doc.update(kw)
    doc["heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["pid"] = os.getpid()
    STATUS.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")


def default_workers() -> int:
    return max(1, min(12, (os.cpu_count() or 2) // 2))


def resolve(module: str, cid: str):
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        return None, None
    return getattr(mod, f"unit_{cid}", None), getattr(mod, f"reduce_{cid}", None)


def source_sha(module: str) -> str:
    mod = importlib.import_module(module)
    return C.file_sha(Path(mod.__file__))


def program_cfg(doc: dict) -> dict:
    wl = doc.get("workload") or {}
    cfg = {}
    for e in wl.get("expansions_instantiated", []):
        cfg.update(next((x["change"] for x in EXPANSIONS if x["id"] == e), {}))
    return cfg


def tier_for(doc: dict, override: str | None = None) -> tuple:
    name = override or doc.get("selected_tier") or "T0"
    t = dict(TIERS[name])
    cfg = program_cfg(doc)
    if cfg.get("discovery_worlds_x"):
        t["discovery_worlds"] *= int(cfg["discovery_worlds_x"])
    if cfg.get("transfer_worlds_x"):
        t["transfer_worlds"] *= int(cfg["transfer_worlds_x"])
    if cfg.get("confirmation_worlds_x"):
        t["confirmation_worlds"] *= int(cfg["confirmation_worlds_x"])
    if cfg.get("teams_x"):
        t["teams"] *= int(cfg["teams_x"])
    t["pilot_worlds"] = 4
    return name, t


# --------------------------------------------------------------------------- #
# Running one card.
# --------------------------------------------------------------------------- #
def unit_list(card, lane: str, tier: dict, smoke: bool = False) -> list:
    if card.unit_kind == "single":
        return [(C.LANE_BASE.get(lane, 0), 0, None)]
    if card.unit_kind == "list":
        key = list(card.factors)[0]
        items = card.factors[key]
        if smoke:
            items = items[:1]
        return [(i, 0, it) for i, it in enumerate(items)]
    ids = C.lane_ids(lane, tier)
    reps = int(tier["repeats"])
    if smoke:
        ids, reps = ids[:6], 1                     # six worlds: single-world smoke flips borderline gates on noise
    return [(wid, rep, None) for wid in ids for rep in range(reps)]


def run_card(doc: dict, card, lane: str, tier_name: str, tier: dict, workers: int, pool, smoke: bool = False,
             out_dir: Path | None = None, cfg: dict | None = None, upstream_oracle: bool = False, quarantine: bool = False) -> dict:
    mod_name = card.module.split(":")[0]
    unit_fn, reduce_fn = resolve(mod_name, card.id)
    if unit_fn is None or reduce_fn is None:
        raise RuntimeError(f"{card.id}: card functions not found in {mod_name}")
    sha = source_sha(mod_name)
    units = unit_list(card, lane, tier, smoke)
    ck_lane = "pilot" if quarantine else ("smoke" if smoke else lane)
    results = [None] * len(units)
    tasks = {}
    t0 = time.perf_counter()
    child_cpu, peak_rss, wall_units = 0.0, 0, []
    for i, (wid, rep, item) in enumerate(units):
        cached = None if (smoke or quarantine) else C.load_ckpt(ck_lane, card.id, wid, rep, sha)
        if cached is not None:
            results[i] = cached
            continue
        task = {"card": card.to_dict(), "module": mod_name, "lane": lane, "wid": wid, "rep": rep, "tier": tier, "tier_name": tier_name,
                "cfg": cfg or {}, "smoke": smoke, "upstream_oracle": upstream_oracle, "attack": None, "n_units": len(units), "item": item}
        tasks[pool.submit(run_unit, task)] = i
    n_done = len(units) - len(tasks)
    for fut in as_completed(tasks):
        i = tasks[fut]
        out = fut.result()
        wid, rep, _ = units[i]
        if not out["ok"]:
            raise RuntimeError(f"{card.id} unit w{wid} r{rep} failed: {out['error']}")
        results[i] = out["unit"]
        child_cpu += out["runtime"]["cpu_s"]
        peak_rss = max(peak_rss, out["runtime"].get("peak_rss") or 0)
        wall_units.append(out["runtime"]["wall_s"])
        if not (smoke or quarantine):
            C.save_ckpt(ck_lane, card.id, wid, rep, sha, out["unit"], out["runtime"])
        n_done += 1
        if n_done % max(1, len(units) // 10) == 0:
            status(card=card.id, lane=lane, units_done=n_done, units_total=len(units))
    ctx = {"lane": lane, "wid": units[0][0], "rep": 0, "tier": tier, "tier_name": tier_name, "card": card, "cfg": cfg or {}, "smoke": smoke,
           "upstream_oracle": upstream_oracle, "attack": None, "n_units": len(units), "out_dir": out_dir, "workers": workers,
           "units_wall_s": round(time.perf_counter() - t0, 2), "units_cpu_s": round(child_cpu, 2), "expected_cells": (M.load_cells() or {}).get("cards")}
    c0 = time.process_time()
    verdict = reduce_fn(card, results, ctx)
    wall = time.perf_counter() - t0
    acct = {"wall_s": round(wall, 2), "parent_cpu_s": round(time.process_time() - c0, 2), "children_cpu_s": round(child_cpu, 2),
            "peak_child_rss": int(peak_rss), "workers": workers, "units": len(units), "units_computed": len(tasks),
            "mean_unit_wall_s": float(np.mean(wall_units)) if wall_units else None, "throughput_units_per_s": (len(tasks) / wall) if wall > 0 else None}
    return {"verdict": verdict, "accounting": acct}


def record_runtime(cid: str, lane: str, acct: dict, state: str) -> None:
    rt = json.loads(M.RUNTIME.read_text(encoding="utf-8")) if M.RUNTIME.exists() else {"cards": {}}
    rt["cards"][f"{lane}:{cid}"] = {**acct, "state": state, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    M.RUNTIME.write_text(json.dumps(rt, indent=2), encoding="utf-8")


def prerequisites_failed(doc: dict, card) -> bool:
    states = {d["id"]: d["status"] for d in doc["cards"]}
    return any(states.get(p) in ("INSTRUMENT_FAILED", "VOID") for p in TRUNK_PREREQ.get(card.trunk, []))


def run_lane(doc: dict, lane: str, cards: list, workers: int, pool, tier_name: str, tier: dict, only=None, waves=None) -> None:
    cfg = program_cfg(doc)
    for d in cards:
        cid = d["id"]
        if only and cid not in only:
            continue
        if waves is not None and d["wave"] not in waves:
            continue
        key = f"{lane}:{cid}"
        ledger = json.loads(C.COMPLETION.read_text(encoding="utf-8")).get("entries", {}) if C.COMPLETION.exists() else {}
        if key in ledger and ledger[key].get("state") in RESOLVED:
            continue
        if lane == "discovery" and d["status"] in RESOLVED:
            continue
        card = card_from_dict(d)
        oracle = prerequisites_failed(doc, card)
        card.upstream_oracle = oracle
        print(f"\n=== {cid} [{lane}] (wave {d['wave']}, trunk {d['trunk']}): {d['question']}", flush=True)
        status(card=cid, lane=lane, stage=lane)
        if lane == "discovery":
            M.update_card(doc, cid, status="RUNNING")
            M.save_manifest(doc)
        try:
            out = run_card(doc, card, lane, tier_name, tier, workers, pool, cfg=cfg, upstream_oracle=oracle)
            v, acct = out["verdict"], out["accounting"]
            state = v["state"]
            record_runtime(cid, lane, acct, state)
            if lane == "discovery":
                M.update_card(doc, cid, status=state, closure_reason=v.get("closure_reason", ""), pursuit=v.get("pursuit", "OPENED"),
                              warrant=v.get("warrant", "DESCRIPTIVE_ONLY"), actual=acct, upstream_oracle=oracle, repairs_used=int(len(v.get("repairs", []))))
            print(f"    -> {state} in {acct['wall_s']:.1f}s wall, {acct['children_cpu_s']:.1f}s child CPU", flush=True)
        except Exception as exc:                                             # noqa: BLE001
            record_runtime(cid, lane, {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}, "ERROR")
            if lane == "discovery":
                M.update_card(doc, cid, status="BUILT")
            print(f"    !! ERROR {exc!r}", flush=True)
        if lane == "discovery":
            M.save_manifest(doc)
            M.write_coverage(doc)


# --------------------------------------------------------------------------- #
# Stages.
# --------------------------------------------------------------------------- #
def stage_prepare(doc: dict) -> dict:
    from ghostscale.prereg_v13 import write_structural_lock
    M.write_cells_template()
    lock = write_structural_lock()
    for d in doc["cards"]:
        if d["status"] == "PLANNED":
            d["status"] = "BUILT"
    M.save_manifest(doc)
    print(f"structural lock: cards {lock['cards_sha256'][:12]} criteria {lock['criteria_sha256'][:12]}")
    return doc


def stage_smoke(doc: dict, workers: int, pool, only=None) -> None:
    os.environ["GS_V13_SMOKE"] = "1"
    out = V.SMOKE_DIR
    out.mkdir(parents=True, exist_ok=True)
    tier = dict(TIERS["T0"], pilot_worlds=4)
    summary = {}
    for d in doc["cards"]:
        if only and d["id"] not in only:
            continue
        card = card_from_dict(d)
        lane = "attack" if card.trunk == "X" else ("transfer" if "discovery" not in card.lanes else "discovery")
        t0 = time.perf_counter()
        try:
            res = run_card(doc, card, lane, "T0", tier, workers, pool, smoke=True, out_dir=out)
            v = res["verdict"]
            summary[card.id] = {"state": v["state"], "failed": v["gates"]["failed_names"], "wall": round(time.perf_counter() - t0, 1)}
        except Exception as exc:                                             # noqa: BLE001
            summary[card.id] = {"state": "ERROR", "error": repr(exc), "wall": round(time.perf_counter() - t0, 1)}
        print(f"{card.id}: {summary[card.id]}", flush=True)
    (V.V13_RESULTS / "SMOKE.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def stage_pilot(doc: dict, workers: int, pool) -> dict:
    """The discarded runtime pilot: twelve heavy cards, four pilot worlds, two repeats, at the T0
    and T3 per-world sizes; a power law in makers-per-world interpolates the other tiers."""
    from ghostscale.prereg_v13 import write_scientific_lock, write_workload_lock
    pilot_cards = [card_from_dict(d) for d in doc["cards"] if d["pilot"]]
    out_dir = V.PILOT_QUARANTINE
    out_dir.mkdir(parents=True, exist_ok=True)
    measured = {}
    for tname in ("T0", "T3"):
        tier = dict(TIERS[tname], pilot_worlds=4, repeats=2)
        for card in pilot_cards:
            lane = "transfer" if "discovery" not in card.lanes else "discovery"
            plane = "pilot"
            t0 = time.perf_counter()
            status(stage="pilot", card=card.id, tier=tname)
            ptier = dict(tier, discovery_worlds=4, transfer_worlds=4)
            # pilot units live on the pilot lineage: override the lane ids through a pilot tier
            units = [(wid, rep, None) for wid in C.lane_ids("pilot", {"pilot_worlds": 4}) for rep in range(2)]
            mod_name = card.module.split(":")[0]
            unit_fn, reduce_fn = resolve(mod_name, card.id)
            tasks = {}
            for wid, rep, _ in units:
                task = {"card": card.to_dict(), "module": mod_name, "lane": "pilot", "wid": wid, "rep": rep, "tier": ptier, "tier_name": tname,
                        "cfg": {}, "smoke": False, "upstream_oracle": False, "attack": None, "n_units": len(units), "item": None}
                tasks[pool.submit(run_unit, task)] = (wid, rep)
            walls, cpus, rss = [], [], 0
            errors = []
            for fut in as_completed(tasks):
                o = fut.result()
                if not o["ok"]:
                    errors.append(o["error"][-400:])
                    continue
                walls.append(o["runtime"]["wall_s"])
                cpus.append(o["runtime"]["cpu_s"])
                rss = max(rss, o["runtime"].get("peak_rss") or 0)
            measured.setdefault(card.id, {})[tname] = {"unit_wall_s": float(np.mean(walls)) if walls else None, "unit_cpu_s": float(np.mean(cpus)) if cpus else None,
                                                       "peak_rss": rss, "n_units": len(units), "errors": errors, "elapsed_s": round(time.perf_counter() - t0, 1)}
            print(f"pilot {card.id} @ {tname}: {measured[card.id][tname]['unit_wall_s']} s/unit ({len(errors)} errors)", flush=True)
    acct = C.process_accounting()
    pilot = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "cards": [c.id for c in pilot_cards], "measured": measured,
             "accounting": {"parent_cpu_s": acct.get("user", 0) + acct.get("system", 0), "children_cpu_s": float(sum(m[t]["unit_cpu_s"] or 0 for m in measured.values() for t in m)),
                            "peak_rss": max((m[t]["peak_rss"] for m in measured.values() for t in m), default=0), "workers": workers},
             "quarantine": str(out_dir), "non_scientific": True}
    PILOT.write_text(json.dumps(pilot, indent=2, default=str), encoding="utf-8")
    fc = forecast(doc, measured, workers)
    FORECAST.write_text(json.dumps(fc, indent=2, default=str), encoding="utf-8")
    sel = fc["selected"]
    doc["selected_tier"] = sel["tier"]
    doc["workload"] = {"tier": sel["tier"], "expansions_instantiated": sel["expansions"], "forecast_hours": sel["hours"], "rule": sel["rule"]}
    doc["lineages"] = M.lineages(tier_for(doc)[1])
    M.save_manifest(doc)
    tname, tier = tier_for(doc)
    M.instantiate_cells(tname, tier)
    write_workload_lock(tname, tier, sel["expansions"], {"hours": sel["hours"], "rule": sel["rule"], "by_tier": fc["by_tier"]}, pilot)
    write_scientific_lock()
    print(f"tier {sel['tier']} selected: forecast {sel['hours']:.1f} h under rule {sel['rule']}; expansions {sel['expansions']}", flush=True)
    return doc


def forecast(doc: dict, measured: dict, workers: int) -> dict:
    """Seconds per unit by trunk and tier from the pilot; a power law in makers-per-world between the
    measured T0 and T3 points; per-card scaling by declared work weight."""
    cards = [card_from_dict(d) for d in doc["cards"]]
    by_trunk = {}
    for c in cards:
        if c.pilot and c.id in measured:
            by_trunk.setdefault(c.trunk, []).append(c)
    m0, m3 = TIERS["T0"]["makers"], TIERS["T3"]["makers"]

    def unit_seconds(card, tname):
        refs = by_trunk.get(card.trunk) or by_trunk.get("C") or list(by_trunk.values())[0]
        est = []
        for ref in refs:
            a = measured[ref.id].get("T0", {}).get("unit_wall_s") or 1.0
            b = measured[ref.id].get("T3", {}).get("unit_wall_s") or a
            k = np.log(max(b, 1e-6) / max(a, 1e-6)) / np.log(m3 / m0) if b > 0 and a > 0 else 1.0
            mk = TIERS[tname]["makers"]
            s = a * (mk / m0) ** k
            est.append(s * card.work_weight / max(ref.work_weight, 1e-6))
        return float(np.mean(est))

    def hours_for(tname, expansions=()):
        t = dict(TIERS[tname])
        for e in expansions:
            ch = next(x["change"] for x in EXPANSIONS if x["id"] == e)
            for k, v in ch.items():
                if k.endswith("_x") and k[:-2] in t:
                    t[k[:-2]] = int(t[k[:-2]] * v)
        total = 0.0
        for c in cards:
            s = unit_seconds(c, tname)
            if c.trunk == "X":
                total += s * t["transfer_worlds"] * t["repeats"]
                continue
            for lane in c.lanes:
                n = {"discovery": t["discovery_worlds"], "transfer": t["transfer_worlds"]}.get(lane, 0) * t["repeats"] if c.unit_kind == "world" else 1
                total += s * n
            if c.claim_ceiling in ("METHOD", "CONSTRUCTED_MECHANISM") and c.trunk not in ("I", "L", "X"):
                total += 0.5 * s * t["confirmation_worlds"] * t["repeats"]      # about half the candidates reach confirmation
        return float(total * OVERHEAD / max(workers, 1) / 3600.0)

    by_tier = {tname: hours_for(tname) for tname in TIER_ORDER}
    lo, hi = ENVELOPE_HOURS
    sel = None
    for tname in TIER_ORDER:
        if lo <= by_tier[tname] <= hi:
            sel = {"tier": tname, "hours": by_tier[tname], "expansions": [], "rule": "in_envelope"}
            break
    if sel is None:
        if by_tier["T0"] > hi:
            sel = {"tier": "T0", "hours": by_tier["T0"], "expansions": [], "rule": "T0_above_envelope_kept"}
        else:
            # the largest tier under the envelope, then expansion packets in the frozen order until the forecast enters it
            base = max((t for t in TIER_ORDER if by_tier[t] <= hi), key=lambda t: by_tier[t])
            exps, hours = [], by_tier[base]
            for e in EXPANSIONS:
                if hours >= lo:
                    break
                trial = hours_for(base, exps + [e["id"]])
                if trial > hi and hours >= lo * 0.8:
                    break
                exps.append(e["id"])
                hours = trial
            sel = {"tier": base, "hours": hours, "expansions": exps, "rule": "T3_below_envelope_expanded" if base == "T3" else "largest_tier_under_envelope_expanded"}
    return {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "workers": workers, "overhead": OVERHEAD, "by_tier": by_tier, "selected": sel, "envelope_hours": list(ENVELOPE_HOURS)}


def stage_bridge(doc: dict, workers: int, pool) -> None:
    tname, tier = tier_for(doc)
    run_lane(doc, "discovery", [d for d in doc["cards"] if d["trunk"] == "L"], workers, pool, tname, tier)


def stage_report() -> None:
    import runners.report_v13 as R
    R.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["prepare", "smoke", "wave0", "pilot", "discovery", "transfer", "attacks", "confirmation", "bridge", "report", "all"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tier", default=None, help="override the selected tier (smoke and pilot only)")
    args = ap.parse_args()
    if os.environ.get("GS_V13_SMOKE") and args.stage not in ("smoke",):
        sys.exit("refusing to run a scientific stage under GS_V13_SMOKE")
    workers = args.workers or default_workers()
    prio = C.lower_priority()
    C.hide_accelerators()
    doc = M.load_manifest()
    status(stage=args.stage, workers=workers, priority=prio, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    stages = [args.stage] if args.stage != "all" else ["prepare", "wave0", "pilot", "discovery", "transfer", "attacks", "confirmation", "bridge", "report"]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        for st in stages:
            status(stage=st)
            if st == "prepare":
                doc = stage_prepare(doc)
            elif st == "smoke":
                stage_smoke(doc, workers, pool, only=args.only)
            elif st == "wave0":
                if not V.V13_RESULTS.joinpath("prereg_v13_structural_lock.json").exists():
                    doc = stage_prepare(doc)
                tname, tier = tier_for(doc, args.tier)
                run_lane(doc, "discovery", [d for d in doc["cards"] if d["trunk"] == "I" and d["id"] != "I13"], workers, pool, tname, tier, only=args.only)
            elif st == "pilot":
                if not PILOT.exists() or args.only:
                    doc = stage_pilot(doc, workers, pool)
                tname, tier = tier_for(doc)
                run_lane(doc, "discovery", [d for d in doc["cards"] if d["id"] == "I13"], workers, pool, tname, tier)
            elif st == "discovery":
                from ghostscale.prereg_v13 import lock_status
                ls = lock_status()
                if not ls.get("locked"):
                    sys.exit(f"discovery refused: scientific lock not in place: {ls}")
                tname, tier = tier_for(doc)
                cards = sorted([d for d in doc["cards"] if d["trunk"] not in ("I", "L", "X") and "discovery" in d["lanes"]], key=lambda d: (d["wave"], d["id"]))
                run_lane(doc, "discovery", cards, workers, pool, tname, tier, only=args.only)
            elif st == "transfer":
                tname, tier = tier_for(doc)
                cards = sorted([d for d in doc["cards"] if "transfer" in d["lanes"]], key=lambda d: (d["wave"], d["id"]))
                run_lane(doc, "transfer", cards, workers, pool, tname, tier, only=args.only)
            elif st == "attacks":
                tname, tier = tier_for(doc)
                run_lane(doc, "attack", [d for d in doc["cards"] if d["trunk"] == "X"], workers, pool, tname, tier, only=args.only)
            elif st == "confirmation":
                import runners.run_v13_confirmation as RC
                RC.run(doc, workers, pool, only=args.only)
            elif st == "bridge":
                stage_bridge(doc, workers, pool)
            elif st == "report":
                stage_report()
    cov = M.write_coverage(doc)
    status(stage="idle", coverage=cov)
    print(f"\ncoverage: {cov['mandatory_resolved']}/{cov['mandatory_total']} mandatory resolved; states {cov['by_state']}")


if __name__ == "__main__":
    main()
