"""Resumable card runner for V12 (spec section 17.4, section 18).

    python runners/run_v12.py                      # every unresolved card, wave order
    python runners/run_v12.py --wave 0 1           # only these waves
    python runners/run_v12.py --only I01 S04       # named cards
    python runners/run_v12.py --workers 8          # cap per-card parallelism

Runs at below-normal process priority with BLAS threads capped, so it coexists with the sibling's
GPU program. Skips resolved cards. Writes the manifest after every card; a clean exit without a
card's produce marker is a failure and the card stays unresolved.
"""
from __future__ import annotations

import os
# Cap numeric threads BEFORE numpy loads: per-card parallelism comes from processes, not BLAS.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ghostscale.config import load_config                       # noqa: E402
from ghostscale.validation.soundingline.v12 import manifest as M  # noqa: E402
from ghostscale.validation.soundingline.v12 import verdict_dir   # noqa: E402
from ghostscale.validation.soundingline.v12.schemas import RESOLVED  # noqa: E402

WAVE_ORDER = [0, 1, 2, 3, 4, 5]


def lower_priority() -> str:
    try:
        import psutil
        p = psutil.Process()
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
        return "below_normal"
    except Exception as exc:                                    # noqa: BLE001
        return f"unchanged ({exc!r})"


def default_workers() -> int:
    return max(1, min(12, (os.cpu_count() or 2) // 2))


def resolve(card_module: str):
    mod_name, fn_name = card_module.split(":")
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        return None
    return getattr(mod, fn_name, None)


def main() -> None:
    if os.environ.get("GS_V12_WORLD_LIMIT"):
        sys.exit("refusing to run the program under GS_V12_WORLD_LIMIT; that cap is for smoke tests only")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wave", nargs="*", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--lane", default="both", choices=["discovery", "confirmation", "both"])
    args = ap.parse_args()

    prio = lower_priority()
    workers = args.workers or default_workers()
    cfg = load_config()
    doc = M.load_manifest()
    runtime_path = M.RUNTIME
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {
        "cards": {}, "workers": workers, "priority": prio}
    runtime["workers"], runtime["priority"] = workers, prio

    from ghostscale.prereg_v12 import lock_status
    ls = lock_status()
    print(f"prereg lock: {ls}")

    order = sorted(doc["cards"], key=lambda d: (WAVE_ORDER.index(d["wave"]) if d["wave"] in WAVE_ORDER
                                                 else 99, d["id"]))
    for d in order:
        cid = d["id"]
        if args.only and cid not in args.only:
            continue
        if args.wave is not None and d["wave"] not in args.wave:
            continue
        if d["status"] in RESOLVED:
            continue
        fn = resolve(d["module"])
        if fn is None:
            print(f"  [{cid}] not built yet ({d['module']}); stays {d['status']}")
            continue
        deps_ok = all(any(x["id"] == dep and x["status"] in RESOLVED for x in doc["cards"])
                      for dep in d.get("depends_on", []))
        if not deps_ok:
            print(f"  [{cid}] waiting on dependencies {d['depends_on']}")
            continue
        print(f"\n=== {cid} (wave {d['wave']}, trunk {d['trunk']}): {d['question']}")
        M.update_card(doc, cid, status="RUNNING")
        M.save_manifest(doc)
        t0 = time.perf_counter()
        cpu0 = time.process_time()
        try:
            card = M.get_card(doc, cid)
            verdict = fn(card, cfg, workers=workers, lane=args.lane)
            state = verdict.get("state", "LANDED")
            assert state in RESOLVED, f"card returned non-resolved state {state!r}"
            marker = verdict_dir() / f"{cid}.produced"
            assert marker.exists(), "card finished without its produce marker"
            wall = time.perf_counter() - t0
            M.update_card(doc, cid, status=state, actual_cpu_minutes=round(
                (time.process_time() - cpu0) / 60.0, 3), closure_reason=verdict.get("closure_reason", ""),
                repairs_used=int(verdict.get("repairs_used", 0)))
            runtime["cards"][cid] = {"wall_seconds": round(wall, 2), "state": state,
                                     "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
            print(f"    -> {state} in {wall:.1f}s")
        except Exception as exc:                                # noqa: BLE001
            wall = time.perf_counter() - t0
            M.update_card(doc, cid, status="BUILT")
            runtime["cards"][cid] = {"wall_seconds": round(wall, 2), "state": "ERROR",
                                     "error": repr(exc), "traceback": traceback.format_exc()[-2000:]}
            print(f"    !! ERROR {exc!r}")
        M.save_manifest(doc)
        M.write_coverage(doc)
        runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    cov = M.write_coverage(doc)
    print(f"\ncoverage: {cov['mandatory_resolved']}/{cov['mandatory_total']} mandatory resolved; "
          f"states {cov['by_state']}")


if __name__ == "__main__":
    main()
