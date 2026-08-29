"""Resumable, tier-calibrated runner for V14 under the one-window contract (spec §7-§9, §18).

    python runners/run_v14.py --stage prepare        # manifest, expected-cell template, structural lock
    python runners/run_v14.py --stage smoke          # every card once on six worlds at smoke sizes; verdicts to a scratch dir
    python runners/run_v14.py --stage pilot          # opens the 24-hour window; the discarded runtime pilot; selects the tier; writes the locks
    python runners/run_v14.py --stage discovery      # discovery lane in trunk order (resumes)
    python runners/run_v14.py --stage transfer       # transfer-lane cards
    python runners/run_v14.py --stage attacks        # X01-X12 on the transfer lineage
    python runners/run_v14.py --stage confirmation   # the frozen candidates (<= 4, one per flight) on the untouched lineage
    python runners/run_v14.py --stage closure        # B01-B02; the SHORT_RUN marker
    python runners/run_v14.py --stage report         # the final packet, only after the deadline
    python runners/run_v14.py --stage all            # everything above in order, resuming what is done

The clock starts when the pilot starts and the deadline is written once (DEADLINE.json); a restart
reads it back and never extends it. At hour 20 the confirmation candidates freeze; at hour 24 no
new unit is submitted. There is no early packet: the report stage refuses before the deadline and
waits for it when the queue empties early.

Workers are processes with one BLAS thread each, below normal priority, no accelerator visible.
Units checkpoint by (lane, card, world, repeat) and resume is hash-aware. The governor samples
Sounding Line's scheduler status at card boundaries and sheds a quarter of the workers when a
stage of theirs overruns its estimate by more than five percent while this window is open.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import glob
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

import ghostscale.validation.soundingline.v14 as V                          # noqa: E402
from ghostscale.validation.soundingline.v14 import common as C              # noqa: E402
from ghostscale.validation.soundingline.v14 import manifest as M            # noqa: E402
from ghostscale.validation.soundingline.v14.atomicio import write_json_atomic  # noqa: E402
from ghostscale.validation.soundingline.v14.runtime import init_worker, run_unit  # noqa: E402
from ghostscale.validation.soundingline.v14.schemas import ENVELOPE_HOURS, EXPANSIONS, FREEZE_HOUR, RESOLVED, TIERS, TIER_ORDER, WINDOW_HOURS, card_from_dict  # noqa: E402

STATUS = V.V14_RESULTS / "RUNNER_STATUS.json"
PILOT = V.V14_RESULTS / "PILOT.json"
FORECAST = V.V14_RESULTS / "FORECAST.json"
COEXISTENCE = V.V14_RESULTS / "COEXISTENCE.json"
SHORT_RUN = V.V14_RESULTS / "SHORT_RUN.json"
SOUNDING_LINE = REPO.parents[2] / "SoundingLine" / "sounding-line" / "results"
TRUNK_ORDER = ["I", "J", "R", "E", "A", "H", "F"]
TRUNK_PREREQ = {"J": ["I03", "I04"], "R": ["I04"], "E": ["I05"], "A": ["I05"], "H": ["I03"], "F": []}
OVERHEAD = 1.15
GOVERNOR_TOLERANCE = 0.05


class WindowClosed(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Status, the window, the governor.
# --------------------------------------------------------------------------- #
def status(**kw) -> None:
    doc = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    doc.update(kw)
    doc["heartbeat"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    doc["pid"] = os.getpid()
    if M.DEADLINE.exists():
        doc["elapsed_hours"] = round(elapsed_hours(), 3)
    write_json_atomic(STATUS, doc, newline=None)


def default_workers() -> int:
    return max(1, min(12, (os.cpu_count() or 2) // 2))


def open_window() -> dict:
    """Write the deadline once. A restart reads it back; nothing ever moves it."""
    if M.DEADLINE.exists():
        return json.loads(M.DEADLINE.read_text(encoding="utf-8"))
    now = time.time()
    doc = {"opened": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), "opened_epoch": now,
           "window_hours": WINDOW_HOURS, "freeze_hour": FREEZE_HOUR,
           "freeze_epoch": now + FREEZE_HOUR * 3600.0, "deadline_epoch": now + WINDOW_HOURS * 3600.0,
           "deadline": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now + WINDOW_HOURS * 3600.0)),
           "immutable": True, "window_closed_cards": []}
    write_json_atomic(M.DEADLINE, doc)
    return doc


def window() -> dict | None:
    return json.loads(M.DEADLINE.read_text(encoding="utf-8")) if M.DEADLINE.exists() else None


def elapsed_hours() -> float:
    w = window()
    return 0.0 if w is None else (time.time() - w["opened_epoch"]) / 3600.0


def window_closed() -> bool:
    w = window()
    return w is not None and time.time() >= w["deadline_epoch"]


def frozen() -> bool:
    w = window()
    return w is not None and time.time() >= w["freeze_epoch"]


def note_window_closed(cid: str, lane: str) -> None:
    w = window()
    if w is None:
        return
    w.setdefault("window_closed_cards", []).append({"card": cid, "lane": lane, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    write_json_atomic(M.DEADLINE, w)


def coexistence_sample() -> dict:
    """Sounding Line's scheduler status, as it is on disk; the only coexistence signal that
    exists (plan judgment call 3). Absent files are recorded as absent, never invented."""
    out = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "elapsed_hours": round(elapsed_hours(), 3), "files": {}}
    for f in sorted(glob.glob(str(SOUNDING_LINE / "phase_*" / "SCHEDULER_STATUS.json"))):
        try:
            out["files"][f] = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception as exc:                                             # noqa: BLE001
            out["files"][f] = {"unreadable": repr(exc)}
    return out


def _overruns(sample: dict) -> list:
    """Stages in the sibling's status whose realized hours exceed their estimate by more than
    the tolerance. The schema is theirs; keys are matched loosely and anything unparseable is
    skipped, so a format change disables the governor rather than firing it."""
    found = []
    for path, doc in sample.get("files", {}).items():
        if not isinstance(doc, dict):
            continue
        stages = doc.get("stages") or doc.get("per_stage") or {}
        items = stages.items() if isinstance(stages, dict) else enumerate(stages) if isinstance(stages, list) else []
        for name, st in items:
            if not isinstance(st, dict):
                continue
            est = next((st[k] for k in st if "est" in k.lower() and isinstance(st[k], (int, float))), None)
            real = next((st[k] for k in st if "realized" in k.lower() and isinstance(st[k], (int, float))), None)
            if est and real and est > 0 and real / est > 1.0 + GOVERNOR_TOLERANCE:
                found.append({"file": path, "stage": str(name), "est": est, "realized": real})
    return found


def record_coexistence(sample: dict, decision: dict | None = None) -> None:
    doc = json.loads(COEXISTENCE.read_text(encoding="utf-8")) if COEXISTENCE.exists() else {"samples": [], "decisions": []}
    doc["samples"].append({k: v for k, v in sample.items() if k != "files"} | {"n_files": len(sample.get("files", {}))})
    doc["samples"] = doc["samples"][-500:]
    if decision:
        doc["decisions"].append(decision)
    write_json_atomic(COEXISTENCE, doc, newline=None)


class Pool:
    """A process pool that can be shrunk at a card boundary."""

    def __init__(self, n: int):
        self.n = int(n)
        self.ex = ProcessPoolExecutor(max_workers=self.n, initializer=init_worker)

    def submit(self, fn, task):
        return self.ex.submit(fn, task)

    def resize(self, n: int) -> None:
        n = max(1, int(n))
        if n == self.n:
            return
        self.ex.shutdown(wait=True)
        self.n = n
        self.ex = ProcessPoolExecutor(max_workers=self.n, initializer=init_worker)

    def close(self) -> None:
        self.ex.shutdown(wait=True)


def govern(pool: Pool, baseline_workers: int) -> None:
    """At a card boundary while the window is open: sample, and shed a quarter of the workers
    on an overrun of the sibling's schedule; restore when no overrun is seen."""
    if window() is None:
        return
    sample = coexistence_sample()
    over = _overruns(sample)
    decision = None
    if over and pool.n > max(1, baseline_workers // 4):
        new = max(1, int(round(pool.n * 0.75)))
        decision = {"at": sample["at"], "action": "shed", "from": pool.n, "to": new, "overruns": over}
        pool.resize(new)
    elif not over and pool.n < baseline_workers:
        decision = {"at": sample["at"], "action": "restore", "from": pool.n, "to": baseline_workers}
        pool.resize(baseline_workers)
    record_coexistence(sample, decision)
    if decision:
        print(f"    governor: {decision['action']} workers {decision['from']} -> {decision['to']}", flush=True)


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
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
    for key in ("discovery_worlds", "transfer_worlds", "confirmation_worlds"):
        if cfg.get(key + "_x"):
            t[key] = int(t[key] * cfg[key + "_x"])
    if cfg.get("repeats_x"):
        t["repeats"] = int(t["repeats"] * cfg["repeats_x"])
    t["pilot_worlds"] = 4
    return name, t


def unit_list(card, lane: str, tier: dict, smoke: bool = False) -> list:
    if card.unit_kind == "single":
        return [(C.LANE_BASE.get(lane, 0), 0, None)]
    if card.unit_kind == "list":
        key = list(card.factors)[0]
        return [(i, 0, it) for i, it in enumerate(card.factors[key])]
    ids = C.lane_ids(lane, tier)
    reps = int(tier["repeats"])
    if smoke:
        ids, reps = ids[:6], 1                     # six worlds: single-world smoke flips borderline gates on noise
    return [(wid, rep, None) for wid in ids for rep in range(reps)]


# --------------------------------------------------------------------------- #
# Running one card.
# --------------------------------------------------------------------------- #
def run_card(doc: dict, card, lane: str, tier_name: str, tier: dict, pool: Pool, smoke: bool = False,
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
        if not (smoke or quarantine) and window_closed():
            for f in tasks:
                f.cancel()
            raise WindowClosed(f"{card.id}: hour {WINDOW_HOURS:.0f} passed before every unit was submitted")
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
           "upstream_oracle": upstream_oracle, "attack": None, "n_units": len(units), "out_dir": out_dir, "workers": pool.n,
           "units_wall_s": round(time.perf_counter() - t0, 2), "units_cpu_s": round(child_cpu, 2), "expected_cells": (M.load_cells() or {}).get("cards")}
    c0 = time.process_time()
    r0 = time.perf_counter()
    verdict = reduce_fn(card, results, ctx)
    reduce_s = time.perf_counter() - r0
    wall = time.perf_counter() - t0
    acct = {"wall_s": round(wall, 2), "reduce_s": round(reduce_s, 2), "parent_cpu_s": round(time.process_time() - c0, 2), "children_cpu_s": round(child_cpu, 2),
            "peak_child_rss": int(peak_rss), "workers": pool.n, "units": len(units), "units_computed": len(tasks),
            "mean_unit_wall_s": float(np.mean(wall_units)) if wall_units else None, "throughput_units_per_s": (len(tasks) / wall) if wall > 0 else None,
            "elapsed_hours_at_finish": round(elapsed_hours(), 3)}
    return {"verdict": verdict, "accounting": acct}


def record_runtime(cid: str, lane: str, acct: dict, state: str) -> None:
    rt = json.loads(M.RUNTIME.read_text(encoding="utf-8")) if M.RUNTIME.exists() else {"cards": {}}
    rt["cards"][f"{lane}:{cid}"] = {**acct, "state": state, "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json_atomic(M.RUNTIME, rt, newline=None)


def prerequisites_failed(doc: dict, card) -> bool:
    states = {d["id"]: d["status"] for d in doc["cards"]}
    return any(states.get(p) in ("INSTRUMENT_FAILED", "VOID") for p in TRUNK_PREREQ.get(card.trunk, []))


def _claim_sentence(v: dict) -> str:
    nar = v.get("narrative")
    if isinstance(nar, dict):
        return str(nar.get("finding") or nar.get("sentence") or v.get("hypothesis", ""))
    if isinstance(nar, str):
        return nar
    return str(v.get("hypothesis", ""))


def run_lane(doc: dict, lane: str, cards: list, pool: Pool, baseline_workers: int, tier_name: str, tier: dict, only=None) -> None:
    cfg = program_cfg(doc)
    for d in cards:
        cid = d["id"]
        if only and cid not in only:
            continue
        key = f"{lane}:{cid}"
        ledger = json.loads(C.COMPLETION.read_text(encoding="utf-8")).get("entries", {}) if C.COMPLETION.exists() else {}
        if key in ledger and ledger[key].get("state") in RESOLVED:
            continue
        if lane == "discovery" and d["status"] in RESOLVED:
            continue
        if window_closed():
            print(f"\n=== {cid} [{lane}]: not opened, the window is closed", flush=True)
            note_window_closed(cid, lane)
            continue
        govern(pool, baseline_workers)
        card = card_from_dict(d)
        oracle = prerequisites_failed(doc, card)
        card.upstream_oracle = oracle
        print(f"\n=== {cid} [{lane}] (wave {d['wave']}, trunk {d['trunk']}, h{elapsed_hours():.1f}): {d['question']}", flush=True)
        status(card=cid, lane=lane, stage=lane)
        owns_status = lane == "discovery" or (lane == "transfer" and "discovery" not in d["lanes"]) or lane == "attack"
        if owns_status:
            M.update_card(doc, cid, status="RUNNING")
            M.save_manifest(doc)
        try:
            out = run_card(doc, card, lane, tier_name, tier, pool, cfg=cfg, upstream_oracle=oracle)
            v, acct = out["verdict"], out["accounting"]
            state = v["state"]
            record_runtime(cid, lane, acct, state)
            if owns_status:
                M.update_card(doc, cid, status=state, closure_reason=v.get("closure_reason", ""), pursuit=v.get("pursuit", "OPENED"),
                              warrant=v.get("warrant", "DESCRIPTIVE_ONLY"), actual=acct, upstream_oracle=oracle, repairs_used=int(len(v.get("repairs", []))))
                try:
                    M.claim(cid, v.get("claim_ceiling", ""), _claim_sentence(v), state)
                except Exception as exc:                                     # noqa: BLE001
                    print(f"    (claim ledger not updated: {exc!r})", flush=True)
            print(f"    -> {state} in {acct['wall_s']:.1f}s wall, {acct['children_cpu_s']:.1f}s child CPU", flush=True)
        except WindowClosed as exc:
            print(f"    !! {exc}", flush=True)
            note_window_closed(cid, lane)
            if owns_status:
                M.update_card(doc, cid, status="BUILT")
        except Exception as exc:                                             # noqa: BLE001
            record_runtime(cid, lane, {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}, "ERROR")
            if owns_status:
                M.update_card(doc, cid, status="BUILT")
            print(f"    !! ERROR {exc!r}", flush=True)
            from concurrent.futures.process import BrokenProcessPool
            if isinstance(exc, BrokenProcessPool):
                raise SystemExit("worker pool broken; relaunch --stage all to resume") from exc
        if owns_status:
            M.save_manifest(doc)
            M.write_coverage(doc)


# --------------------------------------------------------------------------- #
# Stages.
# --------------------------------------------------------------------------- #
def stage_prepare(doc: dict) -> dict:
    from ghostscale.prereg_v14 import lock_status, write_structural_lock
    if lock_status().get("locked"):
        print("prepare skipped: the scientific lock is in place; the program is frozen")
        return doc
    M.write_cells_template()
    M.write_source_lineages()
    M.write_construction_identities()
    lock = write_structural_lock()
    fresh = {c.id: c.to_dict() for c in M.build_cards()}
    for i, d in enumerate(doc["cards"]):
        nd = dict(fresh[d["id"]])
        for k in ("status", "state", "resolved", "verdict_path"):
            if k in d:
                nd[k] = d[k]
        doc["cards"][i] = nd
    for d in doc["cards"]:
        if d["status"] == "PLANNED":
            d["status"] = "BUILT"
    M.save_manifest(doc)
    print(f"structural lock: cards {lock['cards_sha256'][:12]} criteria {lock['criteria_sha256'][:12]}")
    return doc


def stage_smoke(doc: dict, pool: Pool, only=None) -> None:
    os.environ["GS_V14_SMOKE"] = "1"
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
            res = run_card(doc, card, lane, "T0", tier, pool, smoke=True, out_dir=out)
            v = res["verdict"]
            summary[card.id] = {"state": v["state"], "failed": v["gates"]["failed_names"], "wall": round(time.perf_counter() - t0, 1)}
        except Exception as exc:                                             # noqa: BLE001
            summary[card.id] = {"state": "ERROR", "error": repr(exc), "wall": round(time.perf_counter() - t0, 1)}
        print(f"{card.id}: {summary[card.id]}", flush=True)
    prev = json.loads((V.V14_RESULTS / "SMOKE.json").read_text(encoding="utf-8")) if (V.V14_RESULTS / "SMOKE.json").exists() and only else {}
    prev.update(summary)
    write_json_atomic(V.V14_RESULTS / "SMOKE.json", prev)


def stage_pilot(doc: dict, pool: Pool) -> dict:
    """Opens the window. The discarded runtime pilot: every pilot card end to end (units and
    reduce) on four pilot worlds at the T0 and T3 per-world sizes, in quarantine; a power law in
    makers-per-world interpolates the other tiers; the forecast is per card, end to end."""
    from ghostscale.prereg_v14 import ATTACK_RELEVANCE, write_scientific_lock, write_workload_lock
    from ghostscale.validation.soundingline.v14 import routes as R
    from ghostscale.validation.soundingline.v14.routes import ROUTE_COST
    from ghostscale.validation.soundingline.v14.cards import world_for
    w = open_window()
    print(f"window opened {w['opened']}; deadline {w['deadline']}; freeze at hour {FREEZE_HOUR:.0f}", flush=True)
    status(stage="pilot", window=w)
    pilot_cards = [card_from_dict(d) for d in doc["cards"] if d["pilot"]]
    out_dir = V.PILOT_QUARANTINE
    out_dir.mkdir(parents=True, exist_ok=True)
    measured = {}
    for tname in ("T0", "T3"):
        tier = dict(TIERS[tname], pilot_worlds=4, repeats=2, discovery_worlds=4, transfer_worlds=4, confirmation_worlds=4)
        for card in pilot_cards:
            lane = "transfer" if "discovery" not in card.lanes else "discovery"
            t0 = time.perf_counter()
            status(stage="pilot", card=card.id, tier=tname)
            try:
                res = run_card(doc, card, "pilot", tname, tier, pool, out_dir=out_dir, quarantine=True)
                a = res["accounting"]
                measured.setdefault(card.id, {})[tname] = {"unit_wall_s": a["mean_unit_wall_s"], "reduce_s": a["reduce_s"], "peak_rss": a["peak_child_rss"],
                                                           "n_units": a["units"], "state": res["verdict"]["state"], "errors": [], "elapsed_s": round(time.perf_counter() - t0, 1), "lane": lane}
            except Exception as exc:                                         # noqa: BLE001
                measured.setdefault(card.id, {})[tname] = {"unit_wall_s": None, "reduce_s": None, "peak_rss": 0, "n_units": 0, "state": "ERROR",
                                                           "errors": [repr(exc)[-400:]], "elapsed_s": round(time.perf_counter() - t0, 1), "lane": lane}
            print(f"pilot {card.id} @ {tname}: {measured[card.id][tname]['unit_wall_s']} s/unit, reduce {measured[card.id][tname]['reduce_s']} s ({measured[card.id][tname]['state']})", flush=True)
    acct = C.process_accounting()
    pilot = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "cards": [c.id for c in pilot_cards], "measured": measured,
             "accounting": {"parent_cpu_s": acct.get("user", 0) + acct.get("system", 0), "peak_rss": max((m[t]["peak_rss"] for m in measured.values() for t in m), default=0), "workers": pool.n},
             "quarantine": str(out_dir), "non_scientific": True, "window": w}
    write_json_atomic(PILOT, pilot)
    # the planted route information and the attack matrix, both part of the workload lock
    world = world_for({"wid": 9000, "lane": "pilot", "cfg": {}, "smoke": False})
    rd = importlib.import_module("ghostscale.validation.soundingline.v14.joint").Reader(world, 0, 0.75, 0.8)
    rng = np.random.default_rng(1_400_000 + 9000)
    route_info = {}
    for rt in R.ROUTES:
        try:
            route_info[rt] = {"cost": ROUTE_COST[rt], "doses_to_entropy": R.doses_to_entropy(world, rd, rt, rng), "ease": R.ease(rt)}
        except Exception as exc:                                             # noqa: BLE001
            route_info[rt] = {"cost": ROUTE_COST[rt], "error": repr(exc)}
    from ghostscale.prereg_v14 import ROUTE_INFORMATION, ATTACK_MATRIX
    write_json_atomic(ROUTE_INFORMATION, {"written": pilot["written"], "routes": route_info, "planted": True})
    write_json_atomic(ATTACK_MATRIX, {"written": pilot["written"], "relevance": ATTACK_RELEVANCE})
    fc = forecast(doc, measured, pool.n)
    write_json_atomic(FORECAST, fc)
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
    """Hours for the whole program per tier: for every card, unit seconds (power law in makers
    between the measured T0 and T3 points, scaled by declared work weight) times units over the
    workers, plus the reduce; attacks on the transfer lineage; confirmation for the capped set."""
    from ghostscale.prereg_v14 import CONFIRMATION_CAP
    cards = [card_from_dict(d) for d in doc["cards"]]
    by_trunk = {}
    for c in cards:
        if c.pilot and c.id in measured:
            by_trunk.setdefault(c.trunk, []).append(c)
    m0, m3 = TIERS["T0"]["makers"], TIERS["T3"]["makers"]

    def _interp(ref, key, tname):
        a = measured[ref.id].get("T0", {}).get(key) or 1.0
        b = measured[ref.id].get("T3", {}).get(key) or a
        k = np.log(max(b, 1e-6) / max(a, 1e-6)) / np.log(m3 / m0) if b > 0 and a > 0 else 1.0
        return a * (TIERS[tname]["makers"] / m0) ** k

    def seconds(card, tname):
        refs = by_trunk.get(card.trunk) or by_trunk.get("J") or list(by_trunk.values())[0]
        unit = float(np.mean([_interp(ref, "unit_wall_s", tname) * card.work_weight / max(ref.work_weight, 1e-6) for ref in refs]))
        red = float(np.mean([_interp(ref, "reduce_s", tname) for ref in refs]))
        return unit, red

    def hours_for(tname, expansions=()):
        t = dict(TIERS[tname])
        for e in expansions:
            ch = next(x["change"] for x in EXPANSIONS if x["id"] == e)
            for k, v in ch.items():
                if k.endswith("_x") and k[:-2] in t:
                    t[k[:-2]] = int(t[k[:-2]] * v)
        total = 0.0
        candidates = []
        for c in cards:
            unit, red = seconds(c, tname)
            if c.trunk == "X":
                n = t["transfer_worlds"] * t["repeats"]
                total += unit * n / max(workers, 1) + red
                continue
            for lane in c.lanes:
                if c.unit_kind == "world":
                    n = {"discovery": t["discovery_worlds"], "transfer": t["transfer_worlds"]}.get(lane, 0) * t["repeats"]
                elif c.unit_kind == "list":
                    n = len(list(c.factors.values())[0]) if c.factors else 1
                else:
                    n = 1
                total += unit * n / max(workers, 1) + red
            if c.claim_ceiling in ("METHOD", "CONSTRUCTED_MECHANISM") and c.trunk not in ("I", "B", "X"):
                candidates.append(unit * t["confirmation_worlds"] * t["repeats"] / max(workers, 1) + red)
        total += sum(sorted(candidates, reverse=True)[:CONFIRMATION_CAP])
        return float(total * OVERHEAD / 3600.0)

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
            under = [t for t in TIER_ORDER if by_tier[t] <= hi]
            base = max(under, key=lambda t: by_tier[t])
            exps, hours = [], by_tier[base]
            for e in EXPANSIONS:
                if hours >= lo:
                    break
                trial = hours_for(base, exps + [e["id"]])
                if trial > hi:
                    continue
                exps.append(e["id"])
                hours = trial
            sel = {"tier": base, "hours": hours, "expansions": exps, "rule": "largest_tier_under_envelope_expanded"}
    return {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "workers": workers, "overhead": OVERHEAD, "by_tier": by_tier, "selected": sel,
            "envelope_hours": list(ENVELOPE_HOURS), "window_hours": WINDOW_HOURS, "per_card_end_to_end": True}


def discovery_cards(doc: dict) -> list:
    return sorted([d for d in doc["cards"] if d["trunk"] in TRUNK_ORDER and "discovery" in d["lanes"]],
                  key=lambda d: (TRUNK_ORDER.index(d["trunk"]), d["wave"], d["id"]))


def stage_closure(doc: dict, pool: Pool, baseline_workers: int) -> None:
    tname, tier = tier_for(doc)
    run_lane(doc, "discovery", [d for d in doc["cards"] if d["trunk"] == "B"], pool, baseline_workers, tname, tier)
    rt = json.loads(M.RUNTIME.read_text(encoding="utf-8")) if M.RUNTIME.exists() else {"cards": {}}
    last = max((x.get("elapsed_hours_at_finish") or 0.0 for x in rt["cards"].values()), default=0.0)
    w = window() or {}
    short = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "last_unit_hour": last, "window_closed_cards": w.get("window_closed_cards", []),
             "short_run": bool(last < WINDOW_HOURS - 0.5 and not w.get("window_closed_cards"))}
    if short["short_run"]:
        short["note"] = f"the queue emptied at hour {last:.1f}; the deadline stands and the packet waits for it"
    write_json_atomic(SHORT_RUN, short)


def stage_report() -> None:
    if window() is None:
        sys.exit("report refused: no window was opened")
    while not window_closed():
        remaining = (window()["deadline_epoch"] - time.time()) / 3600.0
        status(stage="report", waiting_for_deadline_hours=round(remaining, 3))
        print(f"report: waiting for the deadline ({remaining:.2f} h remaining); no early packet", flush=True)
        time.sleep(min(600.0, max(30.0, remaining * 3600.0)))
    import runners.report_v14 as R
    R.main()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["prepare", "smoke", "pilot", "discovery", "transfer", "attacks", "confirmation", "closure", "report", "all"])
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tier", default=None, help="override the selected tier (smoke only)")
    args = ap.parse_args()
    if os.environ.get("GS_V14_SMOKE") and args.stage not in ("smoke",):
        sys.exit("refusing to run a scientific stage under GS_V14_SMOKE")
    workers = args.workers or default_workers()
    prio = C.lower_priority()
    C.hide_accelerators()
    doc = M.load_manifest()
    status(stage=args.stage, workers=workers, priority=prio, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    stages = [args.stage] if args.stage != "all" else ["prepare", "pilot", "discovery", "transfer", "attacks", "confirmation", "closure", "report"]
    pool = Pool(workers)
    try:
        for st in stages:
            status(stage=st)
            if st == "prepare":
                doc = stage_prepare(doc)
            elif st == "smoke":
                stage_smoke(doc, pool, only=args.only)
            elif st == "pilot":
                if PILOT.exists():
                    print(f"pilot skipped: {PILOT.name} exists; the window opened {window()['opened']}")
                else:
                    doc = stage_pilot(doc, pool)
            elif st in ("discovery", "transfer", "attacks"):
                from ghostscale.prereg_v14 import lock_status
                ls = lock_status()
                if not ls.get("locked"):
                    sys.exit(f"{st} refused: scientific lock not in place: {ls}")
                tname, tier = tier_for(doc)
                if st == "discovery":
                    run_lane(doc, "discovery", discovery_cards(doc), pool, workers, tname, tier, only=args.only)
                elif st == "transfer":
                    cards = sorted([d for d in doc["cards"] if "transfer" in d["lanes"] and d["trunk"] != "X"], key=lambda d: (TRUNK_ORDER.index(d["trunk"]), d["wave"], d["id"]))
                    run_lane(doc, "transfer", cards, pool, workers, tname, tier, only=args.only)
                else:
                    run_lane(doc, "attack", [d for d in doc["cards"] if d["trunk"] == "X"], pool, workers, tname, tier, only=args.only)
            elif st == "confirmation":
                import runners.run_v14_confirmation as RC
                try:
                    RC.run(doc, pool, only=args.only, elapsed=elapsed_hours(), window_closed=window_closed)
                except RC.FreezeViolation as exc:
                    print(f"\n!! confirmation REFUSED and skipped: {exc}\n", flush=True)
                    status(confirmation_refused=str(exc))
            elif st == "closure":
                stage_closure(doc, pool, workers)
            elif st == "report":
                stage_report()
    finally:
        pool.close()
    cov = M.write_coverage(doc)
    status(stage="idle", coverage=cov)
    print(f"\ncoverage: {cov['mandatory_resolved']}/{cov['mandatory_total']} mandatory resolved; states {cov['by_state']}")


if __name__ == "__main__":
    main()
