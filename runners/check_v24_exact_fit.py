"""Retained-input equivalence and cost check, excluded from scientific results."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from ghostscale.validation.soundingline.v23.common import digest, read, write
from ghostscale.validation.soundingline.v23 import runtime
from ghostscale.validation.soundingline.v24.exact_fit import load_certified, grouped_objective
from runners.check_v23_backbone_stationarity import decimal_objective


def compare_models(reference, candidate, cases, current):
    if reference.train_ids != candidate.train_ids or reference.penalties != candidate.penalties:
        raise ValueError("training membership or selected ridge differs")
    errors = []
    for row in cases:
        a = reference.predict(row.maker, row.features, at=current)
        b = candidate.predict(row.maker, row.features, at=current)
        errors.append(float(np.max(np.abs(a - b))))
    maximum = max(errors, default=0.)
    if maximum > 1e-9:
        raise ValueError(f"prediction equivalence failed: {maximum}")
    covariance_error = max((float(np.max(np.abs(reference.conditional_covariances[m]
        - candidate.conditional_covariances[m]))) for m in reference.offsets), default=0.)
    if covariance_error > 1e-7:
        raise ValueError("conditional covariance equivalence failed")
    return dict(prediction_max_absolute_error=maximum, covariance_max_absolute_error=covariance_error,
                identical_training_ids=True, identical_selected_penalties=True,
                candidate_max_gradient=max(r["gradient_max"] for r in candidate.fit_receipts.values()),
                candidate_max_gap=max(r["certified_gap"] for r in candidate.fit_receipts.values()))


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.process_time()
    runtime.below_normal()
    last_pulse = [0.]
    def pulse(**values):
        elapsed = time.process_time() - start
        if time.monotonic()-last_pulse[0] > 5 or values.get("phase") != "grouped-fit":
            write(root / "PROGRESS.json", dict(cpu_seconds=elapsed, at=datetime.now(timezone.utc).isoformat(),
                                             **values), immutable=False)
            last_pulse[0] = time.monotonic()
        if elapsed > args.cpu_cap:
            raise TimeoutError("retained engineering CPU allowance")
    old = read(args.predecessor_plan)
    backbone = next(Path(p) for p in old["external_source_files"] if p.replace("\\", "/").endswith("/models/backbone.py"))
    source_hash = old["external_source_files"][str(backbone)]
    scalar = load_certified(backbone, source_hash, accelerated=False)
    fast = load_certified(backbone, source_hash, accelerated=True, guard=lambda: pulse(phase="grouped-fit"))
    results = []
    try:
        # The exact retained T1 failure input predates the certified implementation.
        if digest(args.t1_input) != "b9d79cd4184249e0e9764f06a1e8ed8a687486614bf9d483d88487568c83ec5e":
            raise ValueError("retained T1 inputs changed")
        with np.load(args.t1_input, allow_pickle=False) as data:
            taken = data["taken"].copy()
            features = [data[f"features_{i}"].copy() for i in range(len(taken))]
            weights, center, initial = (data[k].copy() for k in ("weights", "center", "failed"))
            penalty = float(data["penalty"])
        clock = datetime(2000, 1, 1, tzinfo=timezone.utc)
        rows = [scalar.Choice(str(i), "retained", clock, clock, x, int(y), weight=float(w))
                for i, (x, y, w) in enumerate(zip(features, taken, weights, strict=True))]
        fitted = []
        for name, module in (("scalar", scalar), ("grouped", fast)):
            before = time.process_time()
            offset, receipt = module._fit(rows, center, penalty)
            fitted.append(offset)
            write(root / f"T1_{name.upper()}.json", dict(offset=offset.tolist(), receipt=receipt,
                                                       cpu_seconds=time.process_time()-before))
        objective, hessian = grouped_objective(rows, center, penalty)
        exact = decimal_objective(features, taken, weights, center, initial, penalty)
        value, gradient = objective(initial)
        errors = dict(objective=abs(value-float(exact[0])),
                      gradient=float(np.max(np.abs(gradient-np.asarray(exact[1], dtype=float)))),
                      hessian=float(np.max(np.abs(hessian(initial)-np.asarray(exact[2], dtype=float)))))
        if max(errors.values()) > 1e-9:
            raise ValueError("independent decimal arithmetic equivalence failed")
        exact_fitted = decimal_objective(features, taken, weights, center, fitted[1], penalty)
        if max(abs(float(x)) for x in exact_fitted[1]) > 1e-10:
            raise ValueError("independent gradient certificate failed")
        if np.max(np.abs(fitted[0]-fitted[1])) > 1e-9:
            raise ValueError("certified T1 coefficient mismatch")
        write(root / "T1_EQUIVALENCE.json", dict(passed=True, input_sha256=digest(args.t1_input),
              decimal_precision=50, errors=errors,
              independently_checked_gradient_max=max(abs(float(x)) for x in exact_fitted[1]),
              original_failure_preserved=True, scientific_profile_completed=False))
        # All three retained D5 conditions, all three arms at the largest prefix.
        # This benchmarks fitting only, not an entire world or a scientific effect.
        retained = args.predecessor_plan.parent / "original"
        for condition in ("planted", "uninformative", "constant"):
            folder = retained / ("w1000-" + condition)
            records, labels = read(folder / "READER.json"), read(folder / "LABELS.json")
            label = {r["key"]: r["action"] for r in labels}
            target = records[-1]
            ts = lambda x: datetime.fromtimestamp(x/1000, tz=timezone.utc)
            current = ts(target["open_at"])
            for arm in ("baseline", "own", "donor"):
                cases = [scalar.Choice(r["key"], r["writer"], ts(r["open_at"]), ts(r["resolved_at"]),
                                       np.asarray(r["features"][arm]), label[r["key"]], r["prompt"])
                         for r in records]
                models = []
                costs = {}
                for name, module in (("scalar", scalar), ("grouped", fast)):
                    pulse(phase=name, condition=condition, arm=arm)
                    before = time.process_time()
                    models.append(module.Backbone.fit(cases, until=current, dimensions=cases[0].features.shape[1]))
                    costs[name] = time.process_time()-before
                checked = compare_models(*models, cases, current)
                item = dict(condition=condition, arm=arm, cpu_seconds=costs, **checked,
                    reader_sha256=digest(folder/"READER.json"), labels_sha256=digest(folder/"LABELS.json"))
                results.append(item)
                write(root/f"D5_{condition}_{arm}.json", item)
        receipt = dict(state="RETAINED_ENGINEERING_VERIFIED", scientific_worlds_drawn=0,
            complete_scientific_profile=False, source=fast.ghost_provenance, cases=results,
            original_v23_failures_preserved=True,
            scope="T1 retained failed fit and D5 largest chronological prefix in all three conditions and arms.",
            limitations=["No full-world cost projection or scientific acceptance follows from these selected fits.",
                         "Adapter changes floating-point reduction order; exact equality is mathematical, checked numerically.",
                         "Fresh pilot, full replay and independent review remain required."])
        write(root/"COMPLETE.json", receipt)
        return receipt
    except BaseException as exc:
        write(root/"FAILED.json", dict(error=f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        write(root/"COST.json", dict(cpu_seconds=time.process_time()-start, scientific_worlds_drawn=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor-plan", type=Path, required=True)
    parser.add_argument("--t1-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-cap", type=float, default=1200)
    run(parser.parse_args())
