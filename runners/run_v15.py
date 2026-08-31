"""V15 — The Boundary Map. The resumable, work-conserving 168-hour scheduler.

Usage::

    python -m runners.run_v15 --stage prepare      # manifest, lock inputs, structural lock
    python -m runners.run_v15 --stage smoke        # every card and attack at scratch sizes
    python -m runners.run_v15 --stage pilot        # discarded timing pilot on disjoint seeds
    python -m runners.run_v15 --stage guard        # forecast and the opening guard
    python -m runners.run_v15 --stage all          # everything, in order, and then the window

**Always launch as a module, never as a script path.** The sibling project's orphan sweeper kills
any python whose command line matches ``runners[\\/]run_``; it killed V14's runner seven times and
cost that program its first 24-hour window, and it retro-explains four of V13's "unexplained"
silent deaths. ``python -m runners.run_v15`` does not match. Attack X24 checks that the launcher on
disk still uses the module form, so the fix cannot quietly rot.

What makes this scheduler different from V14's
----------------------------------------------
V14's queue emptied after 6.8 wall hours and the runner then waited fourteen hours for a freeze.
This one dispatches the *balanced coverage stream* whenever the card queue is empty, and records
occupancy against the integral of safe worker capacity. If the queue empties anyway, or workers
wait for the deadline, ``RUNTIME_FAILED`` is set and stays set: spec §9.4 allows the results to be
reported after that, and does not allow the seven-day contract to be claimed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ghostscale.validation.soundingline.v15 import common as C          # noqa: E402
from ghostscale.validation.soundingline.v15 import coverage as CV       # noqa: E402
from ghostscale.validation.soundingline.v15 import manifest as M        # noqa: E402
from ghostscale.validation.soundingline.v15 import runtime_contract as RC  # noqa: E402
from ghostscale.validation.soundingline.v15 import v15_dir, verdict_dir  # noqa: E402
from ghostscale.validation.soundingline.v15.atomicio import write_json_atomic  # noqa: E402
from ghostscale.validation.soundingline.v15.runtime import init_worker, run_coverage, run_unit  # noqa: E402
from ghostscale.validation.soundingline.v15.schemas import RESOLVED, TIERS  # noqa: E402

STATUS = v15_dir() / "RUNNER_STATUS.json"
CHECKPOINTS = v15_dir() / "CHECKPOINTS.jsonl"
PILOT = v15_dir() / "PILOT.json"
COMPLETION = v15_dir() / "COMPLETION.json"
COVERAGE_DIR = v15_dir("coverage")
SMOKE_DIR = v15_dir("smoke")
PILOT_DIR = v15_dir("pilot_quarantine")

#: Trunk order for the mandatory core. I first (it audits everything else), B last (it reads the
#: record). Otherwise by wave, which is the dependency order the manifest declares.
TRUNK_ORDER = ["I", "C", "M", "E", "G", "V", "S", "R", "F", "H", "P", "B"]


# --------------------------------------------------------------------------- #
# Status and workers.
# --------------------------------------------------------------------------- #
def safe_workers(requested: int | None = None) -> int:
    """House rule: at most min(12, cores/2), one BLAS thread each, below-normal priority."""
    cores = os.cpu_count() or 4
    n = min(12, max(1, cores // 2))
    if requested:
        n = max(1, min(int(requested), n))
    return n


def status(**kw) -> None:
    doc = {"program": "v15", "pid": os.getpid(),
           "heartbeat": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "elapsed_hours": round(RC.elapsed_hours(), 4), "phase": RC.phase(), **kw}
    write_json_atomic(STATUS, doc)


def checkpoint(kind: str, **kw) -> None:
    """A machine-readable checkpoint. Never a narrative conclusion (spec §9.1)."""
    line = json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "elapsed_hours": round(RC.elapsed_hours(), 4), "kind": kind, **kw},
                      default=str)
    with CHECKPOINTS.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


class Pool:
    """A spawn pool that can be resized by the coexistence governor."""

    def __init__(self, workers: int):
        self.n = int(workers)
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.n, initializer=init_worker, maxtasksperchild=24)

    def resize(self, n: int) -> None:
        n = max(1, int(n))
        if n == self.n:
            return
        self.close()
        self.n = n
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(self.n, initializer=init_worker, maxtasksperchild=24)

    def map(self, fn, tasks):
        if not tasks:
            return []
        return self.pool.map(fn, tasks, chunksize=1)

    def close(self) -> None:
        try:
            self.pool.close()
            self.pool.join()
        except Exception:                                          # noqa: BLE001
            self.pool.terminate()


# --------------------------------------------------------------------------- #
# Running one card.
# --------------------------------------------------------------------------- #
def card_tasks(card, lane: str, tier: dict, tier_name: str, cfg: dict, smoke: bool,
               attack: str | None = None) -> list:
    from ghostscale.validation.soundingline.v15.schemas import expected_cells
    if card.unit_kind == "list":
        items = list(next(iter(card.factors.values())))
        wids = list(range(len(items)))
        reps = [0] * len(items)
    elif card.unit_kind == "single":
        items, wids, reps = [None], [C.LANE_BASE.get(lane, 0)], [0]
    else:
        ids = (C.lane_ids("discovery", tier) if lane == "smoke" else C.lane_ids(lane, tier))
        if smoke:
            ids = ids[:2]
        reps_n = 1 if smoke else int(tier["repeats"])
        items, wids, reps = [], [], []
        for w in ids:
            for r in range(reps_n):
                items.append(None)
                wids.append(w)
                reps.append(r)
    expected = expected_cells(card, tier, lane)
    return [{"card": card.to_dict(), "module": card.module, "lane": lane, "wid": int(w),
             "rep": int(r), "tier": tier, "tier_name": tier_name, "cfg": cfg,
             "smoke": bool(smoke), "attack": attack, "n_units": len(wids), "item": it,
             "expected": expected}
            for it, w, r in zip(items, wids, reps)]


def run_card(pool: Pool, card, lane: str, tier: dict, tier_name: str, cfg: dict, smoke: bool,
             out_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    tasks = card_tasks(card, lane, tier, tier_name, cfg, smoke,
                       attack=card.id if card.trunk == "X" else None)
    results = pool.map(run_unit, tasks) if pool else [run_unit(t) for t in tasks]
    units, errors, cpu = [], [], 0.0
    for res in results:
        if res["ok"]:
            units.append(res["unit"])
            cpu += float(res["runtime"].get("cpu_s", 0.0))
        else:
            errors.append(res["error"])
    if not units:
        return {"card": card.id, "state": "VOID", "errors": errors[:2],
                "wall_s": time.perf_counter() - t0}
    mod = importlib.import_module(card.module)
    ctx = {"lane": lane, "wid": tasks[0]["wid"], "rep": 0, "tier": tier, "tier_name": tier_name,
           "card": card, "cfg": cfg, "smoke": smoke, "n_units": len(units), "item": None,
           "expected": tasks[0]["expected"], "out_dir": out_dir,
           "units_wall_s": time.perf_counter() - t0, "units_cpu_s": cpu,
           "workers": pool.n if pool else 1}
    try:
        v = getattr(mod, f"reduce_{card.id}")(units, ctx)
    except Exception as exc:                                       # noqa: BLE001
        return {"card": card.id, "state": "VOID",
                "errors": [f"{exc!r}\n{traceback.format_exc()[-2000:]}"],
                "wall_s": time.perf_counter() - t0}
    return {"card": card.id, "state": v.get("state"),
            "criterion_status": v.get("criterion_status"),
            "claim_class": v.get("claim_class"), "errors": errors[:2],
            "wall_s": time.perf_counter() - t0, "cpu_s": cpu, "n_units": len(units)}


# --------------------------------------------------------------------------- #
# Stages.
# --------------------------------------------------------------------------- #
def stage_prepare(args) -> dict:
    from ghostscale.prereg_v15 import lock_status, write_structural_lock
    cards = M.build_cards()
    doc = M.write_manifest(cards, note="V15 — The Boundary Map")
    M.write_cells_template(cards)
    M.write_construction_graph(cards)
    M.write_generator_families()
    M.write_architecture_budgets()
    M.write_source_lineages()
    M.write_publication_template(cards)
    lock = write_structural_lock()
    st = lock_status()
    print(f"manifest: {doc['n_cards']} mandatory cards, {doc['n_attacks']} attacks")
    print(f"structural lock: cards {lock['cards_sha256'][:12]} sesoi {lock['sesoi_sha256'][:12]} "
          f"generators {len(lock['generators'])}")
    print(f"lock status: {st}")
    checkpoint("prepare", n_cards=doc["n_cards"], n_attacks=doc["n_attacks"],
               cards_sha256=lock["cards_sha256"])
    return {"manifest": doc["n_cards"], "attacks": doc["n_attacks"], "lock": st}


def stage_smoke(args) -> dict:
    """Every card and attack at scratch sizes, into a scratch directory. Never the record."""
    os.environ["GS_V15_SMOKE"] = "1"
    cards = M.build_cards()
    only = set(args.only.split(",")) if args.only else None
    out = SMOKE_DIR / "verdicts"
    out.mkdir(parents=True, exist_ok=True)
    pool = Pool(safe_workers(args.workers))
    rows, bad = [], []
    try:
        for card in cards:
            if only and card.id not in only:
                continue
            r = run_card(pool, card, "smoke", TIERS["T0"], "T0", {}, True, out_dir=out)
            rows.append(r)
            flag = "" if r["state"] == "LANDED" else "   <-- "
            print(f"  {card.id}: {r['state']:18} criterion={r.get('criterion_status','')} "
                  f"{r['wall_s']:6.1f}s{flag}{(r['errors'] or [''])[0][:120]}")
            if r["state"] != "LANDED":
                bad.append(r)
    finally:
        pool.close()
    total = sum(r["wall_s"] for r in rows)
    print(f"\nsmoke: {sum(1 for r in rows if r['state'] == 'LANDED')} landed, {len(bad)} not, "
          f"{total:.0f}s wall")
    checkpoint("smoke", landed=sum(1 for r in rows if r["state"] == "LANDED"), failed=len(bad),
               wall_s=round(total, 1))
    return {"rows": rows, "bad": bad, "wall_s": total}


def stage_pilot(args) -> dict:
    """A discarded timing pilot on disjoint seeds. Its outputs never enter the record."""
    os.environ.pop("GS_V15_SMOKE", None)
    cards = M.build_cards()
    tier_name = args.tier or "T0"
    tier = TIERS[tier_name]
    out = PILOT_DIR / "verdicts"
    out.mkdir(parents=True, exist_ok=True)
    pool = Pool(safe_workers(args.workers))
    per_card = {}
    try:
        # one representative card per trunk, timed at the pilot tier
        seen = set()
        for card in cards:
            if card.trunk in seen:
                continue
            seen.add(card.trunk)
            r = run_card(pool, card, "pilot", tier, tier_name, {}, False, out_dir=out)
            per_card[card.id] = {"wall_s": r["wall_s"], "trunk": card.trunk,
                                 "work_weight": card.work_weight}
            print(f"  pilot {card.id} ({card.trunk}): {r['wall_s']:.1f}s  state={r['state']}")
    finally:
        pool.close()
    # a per-work-weight rate, and a coverage-cell rate
    rates = [v["wall_s"] / max(v["work_weight"], 0.1) for v in per_card.values() if v["wall_s"] > 0]
    median_rate = float(sorted(rates)[len(rates) // 2]) if rates else 60.0
    t0 = time.perf_counter()
    cell = CV.block(0)["cells"][0]
    CV.execute_cell(cell, tier)
    cell_s = time.perf_counter() - t0
    doc = {"program": "v15", "tier": tier_name, "workers": safe_workers(args.workers),
           "per_card": per_card, "median_seconds_per_work_unit": median_rate,
           "coverage_cell_seconds": cell_s,
           "seeds_excluded_from_science": True,
           "lane": "pilot",
           "note": "discarded: pilot verdicts live in results/v15/pilot_quarantine and are "
                   "never read by any scientific stage"}
    write_json_atomic(PILOT, doc)
    print(f"pilot: {median_rate:.1f}s per work unit, {cell_s:.2f}s per coverage cell")
    checkpoint("pilot", median_seconds_per_work_unit=round(median_rate, 2),
               coverage_cell_seconds=round(cell_s, 3))
    return doc


def forecast(tier_name: str, workers: int, pilot: dict) -> dict:
    """Conservative upper and lower forecasts for the core and for the locked coverage."""
    cards = M.build_cards()
    rate = float(pilot["median_seconds_per_work_unit"])
    cell_s = float(pilot["coverage_cell_seconds"])
    tier = TIERS[tier_name]
    pilot_tier = TIERS[pilot["tier"]]
    # the pilot ran at its own tier; scale by the unit count the chosen tier implies
    scale = (int(tier["discovery_worlds"]) * int(tier["repeats"])) / \
            max(int(pilot_tier["discovery_worlds"]) * int(pilot_tier["repeats"]), 1)
    core_units = sum(c.work_weight * (len(c.lanes)) for c in cards)
    # cards run their units across the pool, so wall time is the serialized work divided by the
    # worker count -- with a floor, because the single-unit cards cannot use the pool at all
    core_seconds = core_units * rate * scale
    core_h = core_seconds / 3600.0 / max(workers, 1) + core_seconds / 3600.0 * 0.15
    upper = core_h * 1.6                                   # conservative: 60% headroom
    lower = core_h / RC.FAST_MACHINE_FACTOR
    # the coverage stream: enough blocks that a machine three times faster cannot empty it
    need_lower_h = RC.CORE_PLUS_COVERAGE_LOWER_FORECAST_MIN_H - lower + 24.0
    per_block_h = (CV.BLOCK_CELLS * cell_s * scale) / 3600.0 / max(workers, 1)
    n_blocks = int(max(64, (need_lower_h * RC.FAST_MACHINE_FACTOR) / max(per_block_h, 1e-6)))
    coverage_lower_h = n_blocks * per_block_h / RC.FAST_MACHINE_FACTOR
    return {"tier": tier_name, "workers": workers, "seconds_per_work_unit": rate,
            "coverage_cell_seconds": cell_s, "tier_scale": scale,
            "core_work_units": core_units, "core_median_h": core_h,
            "core_upper_h": upper, "core_lower_h": lower,
            "coverage_blocks": n_blocks, "coverage_hours_per_block": per_block_h,
            "coverage_lower_h": coverage_lower_h,
            "confirmation_worker_h": 30.0,
            "note": ("upper adds 60 per cent headroom to the median; lower divides by the "
                     "fast-machine factor, so condition 2 asks whether the queue survives a "
                     "machine three times faster than the pilot")}


def stage_guard(args) -> dict:
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    workers = safe_workers(args.workers)
    tier_name = args.tier or pick_tier(pilot, workers)
    fc = forecast(tier_name, workers, pilot)
    guard = RC.opening_guard(core_upper_h=fc["core_upper_h"], core_lower_h=fc["core_lower_h"],
                             coverage_lower_h=fc["coverage_lower_h"],
                             confirmation_worker_h=fc["confirmation_worker_h"],
                             hashed=True, recovery_tests=True)
    RC.write_opening_receipt(guard, {"forecast": fc, "tier": tier_name, "workers": workers})
    print(f"tier {tier_name}, {workers} workers")
    print(f"  core upper {fc['core_upper_h']:.1f} h (max {RC.CORE_UPPER_FORECAST_MAX_H})")
    print(f"  core+coverage lower {fc['core_lower_h'] + fc['coverage_lower_h']:.1f} h "
          f"(min {RC.CORE_PLUS_COVERAGE_LOWER_FORECAST_MIN_H})")
    print(f"  coverage blocks {fc['coverage_blocks']}")
    print(f"  may_open={guard['may_open']} failed={guard['failed']}")
    checkpoint("guard", may_open=guard["may_open"], failed=guard["failed"], tier=tier_name,
               coverage_blocks=fc["coverage_blocks"])
    return {"guard": guard, "forecast": fc, "tier": tier_name, "workers": workers}


def pick_tier(pilot: dict, workers: int) -> str:
    """The largest tier whose conservative upper forecast still fits under the core ceiling."""
    for name in ("T3", "T2", "T1", "T0"):
        fc = forecast(name, workers, pilot)
        if fc["core_upper_h"] <= RC.CORE_UPPER_FORECAST_MAX_H:
            return name
    return "T0"


def stage_open(args) -> dict:
    g = stage_guard(args)
    if not g["guard"]["may_open"]:
        print("RUN_REFUSED: " + g["guard"]["reason"])
        checkpoint("run_refused", failed=g["guard"]["failed"])
        return {"opened": False, **g}
    from ghostscale.prereg_v15 import write_scientific_lock, write_workload_lock
    fc = g["forecast"]
    cov = CV.sequence_definition(fc["coverage_blocks"])
    gov = {"safe_workers": g["workers"], "reserve_for_sibling": 2,
           "priority": "below_normal", "blas_threads": 1, "accelerators": "hidden"}
    M.instantiate_cells(g["tier"], TIERS[g["tier"]])
    write_json_atomic(v15_dir() / "BALANCED_COVERAGE_SEQUENCE.json", cov)
    write_json_atomic(RC.RESOURCE_GOVERNOR, gov)
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    write_workload_lock(g["tier"], TIERS[g["tier"]], fc, pilot, cov, gov)
    write_scientific_lock()
    w = RC.open_window(note="V15 168-hour window")
    print(f"window opened {w['opened']} -> deadline {w['deadline']}")
    checkpoint("window_opened", opened=w["opened"], deadline=w["deadline"], tier=g["tier"])
    return {"opened": True, "window": w, **g}


def _queue(lane: str) -> list:
    """Cards for a lane, in trunk then wave order, skipping ones already resolved."""
    doc = M.load_manifest()
    cards = [M.get_card(doc, cid) for cid in doc["cards"]]
    q = [c for c in cards if lane in c.lanes]
    q.sort(key=lambda c: (TRUNK_ORDER.index(c.trunk) if c.trunk in TRUNK_ORDER else 99,
                          c.wave, c.id))
    out = []
    for c in q:
        v = C.load_verdict(c.id, lane)
        if v and v.get("state") in RESOLVED:
            continue
        out.append(c)
    return out


def _coverage_progress() -> int:
    return len(list(COVERAGE_DIR.glob("block_*.json")))


def run_coverage_block(pool: Pool, index: int, tier: dict, occ: RC.Occupancy) -> dict:
    b = CV.block(index)
    tasks = [{"cell": c, "tier": tier, "smoke": False} for c in b["cells"]]
    t0 = time.perf_counter()
    res = pool.map(run_coverage, tasks)
    ok = [r["unit"] for r in res if r["ok"]]
    doc = {"block": index, "secondary": b["secondary"], "n_cells": len(b["cells"]),
           "n_ok": len(ok), "digest": CV.block_digest(index),
           "wall_s": round(time.perf_counter() - t0, 2), "cells": ok}
    write_json_atomic(COVERAGE_DIR / f"block_{index:06d}.json", doc)
    occ.coverage_blocks_executed += 1
    occ.coverage_cells_executed += len(ok)
    checkpoint("coverage_block", block=index, n_ok=len(ok), wall_s=doc["wall_s"])
    return doc


def stage_science(args) -> dict:
    """Hours 0-150: the mandatory core, transfer, attacks, then the balanced coverage stream."""
    from ghostscale.prereg_v15 import lock_status
    ls = lock_status()
    if not ls.get("locked"):
        print(f"REFUSED: scientific lock is not intact: {ls}")
        checkpoint("refused_unlocked", lock=ls)
        return {"refused": True, "lock": ls}
    wl = json.loads((v15_dir() / "WORKLOAD_LOCK.json").read_text(encoding="utf-8"))
    tier_name, tier = wl["tier"], wl["tier_config"]
    base_workers = int(wl["resource_governor"]["safe_workers"])
    occ = RC.Occupancy()
    pool = Pool(base_workers)
    doc = M.load_manifest()
    done = []
    try:
        for lane in ("discovery", "transfer", "attack"):
            for card in _queue(lane):
                if RC.frozen():
                    break
                g = RC.govern(base_workers)
                pool.resize(g["workers"])
                occ.tick(pool.n, base_workers, base_workers - pool.n)
                status(stage="science", lane=lane, card=card.id, workers=pool.n,
                       coverage_blocks=_coverage_progress())
                r = run_card(pool, card, lane, tier, tier_name, {}, False)
                occ.cards_executed += 1
                occ.tick(pool.n, base_workers, base_workers - pool.n)
                M.update_card(doc, card.id, status=r["state"],
                              criterion_status=r.get("criterion_status", "UNEVALUATED"),
                              actual={"wall_s": r["wall_s"], "lane": lane})
                done.append(r)
                checkpoint("card", card=card.id, lane=lane, state=r["state"],
                           criterion_status=r.get("criterion_status"),
                           wall_s=round(r["wall_s"], 2))
                print(f"  [{RC.elapsed_hours():6.2f}h] {lane[:4]} {card.id}: {r['state']} "
                      f"{r.get('criterion_status','')} {r['wall_s']:.0f}s")
                M.write_coverage(doc)
        # the card queue is exhausted: the coverage stream keeps the machine on science
        idx = _coverage_progress()
        n_blocks = int(wl["coverage_definition"]["n_blocks"])
        while not RC.frozen():
            if idx >= n_blocks:
                occ.note_queue_empty()
                checkpoint("coverage_exhausted", blocks=idx)
                break
            g = RC.govern(base_workers)
            pool.resize(g["workers"])
            occ.tick(pool.n, base_workers, base_workers - pool.n)
            status(stage="coverage", block=idx, workers=pool.n, coverage_blocks=idx)
            run_coverage_block(pool, idx, tier, occ)
            occ.tick(pool.n, base_workers, base_workers - pool.n)
            idx += 1
            occ.write()
    finally:
        pool.close()
        occ.write()
        write_json_atomic(RC.RUNTIME, {"program": "v15", "stage": "science",
                                       "cards_executed": occ.cards_executed,
                                       "coverage_blocks": occ.coverage_blocks_executed,
                                       "elapsed_hours": RC.elapsed_hours(),
                                       "tree_cpu": C.tree_cpu_seconds()})
    return {"cards": done, "occupancy": occ.to_dict()}


def stage_confirmation(args) -> dict:
    import runners.run_v15_confirmation as RCONF
    return RCONF.run(workers=safe_workers(args.workers))


def stage_integrity(args) -> dict:
    """Hours 166-168: clean clone, aggregate regeneration, ledger reconciliation."""
    out = {}
    try:
        import runners.fresh_clone_v15 as FC
        out["fresh_clone"] = FC.run()
    except Exception as exc:                                       # noqa: BLE001
        out["fresh_clone"] = {"error": repr(exc)}
    doc = M.load_manifest()
    out["coverage"] = M.write_coverage(doc)
    try:
        import runners.validate_v15_program as VAL
        out["validator"] = VAL.run(quiet=True)
    except Exception as exc:                                       # noqa: BLE001
        out["validator"] = {"error": repr(exc)}
    write_json_atomic(v15_dir() / "COMPLETION_SUMMARY.json", C.to_jsonable(out))
    checkpoint("integrity", **{k: (v if isinstance(v, (int, float, str)) else "written")
                               for k, v in out.items()})
    return out


def stage_report(args) -> dict:
    import runners.report_v15 as REP
    return REP.run(force=bool(args.force))


def stage_all(args) -> dict:
    out = {"prepare": stage_prepare(args)}
    s = stage_smoke(args)
    out["smoke"] = {"landed": sum(1 for r in s["rows"] if r["state"] == "LANDED"),
                    "failed": len(s["bad"])}
    if s["bad"] and not args.force:
        print(f"REFUSED to open the window: {len(s['bad'])} cards did not land in smoke")
        return out
    os.environ.pop("GS_V15_SMOKE", None)
    out["pilot"] = {"median_seconds_per_work_unit":
                    stage_pilot(args)["median_seconds_per_work_unit"]}
    o = stage_open(args)
    out["open"] = {"opened": o["opened"]}
    if not o["opened"]:
        return out
    out["science"] = stage_science(args)
    # wait for the freeze if the science finished early -- but only after the coverage stream is
    # exhausted, which is itself recorded as a runtime failure
    out["confirmation"] = stage_confirmation(args)
    while RC.elapsed_hours() < RC.INTEGRITY_END_HOUR - 2.0:
        status(stage="waiting_for_integrity_window")
        time.sleep(60)
    out["integrity"] = stage_integrity(args)
    while not RC.window_closed():
        status(stage="waiting_for_deadline")
        time.sleep(120)
    out["report"] = stage_report(args)
    return out


STAGES = {"prepare": stage_prepare, "smoke": stage_smoke, "pilot": stage_pilot,
          "guard": stage_guard, "open": stage_open, "science": stage_science,
          "confirmation": stage_confirmation, "integrity": stage_integrity,
          "report": stage_report, "all": stage_all}


def main() -> int:
    ap = argparse.ArgumentParser(description="V15 — The Boundary Map")
    ap.add_argument("--stage", default="prepare", choices=sorted(STAGES))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tier", default=None, choices=[None, "T0", "T1", "T2", "T3"])
    ap.add_argument("--only", default=None, help="comma-separated card ids (smoke only)")
    ap.add_argument("--force", action="store_true",
                    help="open the window despite smoke failures; report despite the deadline")
    args = ap.parse_args()
    C.set_numeric_threads(1)
    C.hide_accelerators()
    C.lower_priority()
    status(stage=args.stage, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    t0 = time.perf_counter()
    try:
        STAGES[args.stage](args)
    finally:
        status(stage=args.stage, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
               wall_s=round(time.perf_counter() - t0, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
