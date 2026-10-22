"""V24 full-twin successors for exact export and certified choice fitting.

The V23 scientific modules remain unchanged.  A V24 packet supplies immutable
engine/source pins, constructs one exporter for the complete world, and loads
the certified backbone privately.  Evaluator labels are opened only after the
model and public forecasts exist, exactly as in the retained twin pipeline.
"""
from pathlib import Path
import time

import numpy as np

from ..v23 import checker, ledger, registry, twin_contract, twin_engineering, twin_graph, twin_ledger
from ..v23.common import canonical, digest, read, write
from .exact_export import ExactExporter
from .exact_fit import load_certified


def exporter(plan, schema):
    required = ("engine_files", "implementation_hash")
    if any(name not in plan for name in required):
        raise ValueError("full engine fingerprint is required")
    return ExactExporter(schema, plan["engine_files"], plan["implementation_hash"])


def certified_backbone(plan, guard=lambda: None):
    source = Path(plan["certified_backbone_source"])
    expected = plan["certified_backbone_sha256"]
    if digest(source) != expected:
        raise ValueError("certified backbone source changed")
    module = load_certified(source, expected, accelerated=True, guard=guard)
    if not module.ghost_provenance["canonical_engine_unchanged"]:
        raise ValueError("private certified adaptation lost provenance")
    return module.Backbone


def engine_baseline(root, variant, identity, schema, pulse, plan, *, guard=lambda: None):
    """Fit the full public world with the certified arithmetic successor."""
    from tompathy_engine.ghost.bridge import load_reader, reader_choices

    root = Path(root)
    exported, cutoff = twin_graph.build(root, variant, identity, schema)
    if variant["family"] == "T3":
        result = dict(
            state="TARGET_UNAVAILABLE",
            reason=("The committed choice backbone does not infer unobserved "
                    "revision purposes; V24 does not fabricate that target."),
            engine_module_admission=False,
            reader_hash=exported["reader_hash"],
        )
        write(root / "ENGINE_BASELINE.json", result)
        return result

    rows, _ = load_reader(root / "WHOLE_WORLD_READER.json")
    choices = reader_choices(rows)
    pulse(phase="v24-certified-twin-fitting", choices=len(choices))
    if not choices:
        raise ValueError("supported twin target has no observed history")
    model = certified_backbone(plan, guard=guard).fit(choices, until=cutoff)
    names = sorted(next(row for row in rows if type(row).__name__ == "Decision").options[0].features)
    predicted = {}
    for row in rows:
        if type(row).__name__ != "Decision" or row.taken is not None:
            continue
        x = np.array([[option.features[name] for name in names] for option in row.options])
        probability = model.predict(row.maker_id, x)
        if (not np.isfinite(probability).all() or np.any(probability <= 0)
                or not np.isclose(probability.sum(), 1)):
            raise ValueError("invalid certified twin forecast")
        predicted[row.maker_id] = probability

    # Truth remains isolated until fitting and all public forecasts finish.
    scores = []
    for folder in sorted(root.glob("maker-*")):
        public = read(folder / "PUBLIC.json")
        truth = read(folder / "TRUTH.json")
        key = ledger._identity(identity, "maker", public["maker"])
        rebuilt = checker.scores(predicted[key][None, :], truth["y"])
        scores.append(dict(
            maker=public["maker"],
            probabilities=predicted[key],
            scores={name: float(value[0]) for name, value in rebuilt.items()},
        ))
    write(root / "ENGINE_BASELINE.json", dict(
        state="V24_CERTIFIED_ENGINE_BASELINE_EXECUTED",
        implementation_hash=exported["implementation_hash"],
        rows=scores,
        train_ids=list(model.train_ids),
        reader_hash=exported["reader_hash"],
        engine_module_admission=False,
        certified_backbone_sha256=plan["certified_backbone_sha256"],
        arithmetic_adapter=model.__class__.__module__,
        target=("Constructed handling choice" if variant["family"] == "T1"
                else "Constructed nominal rating; no ordered preference interpretation"),
        information=("Public numeric option features only; native latent support "
                     "and evaluator labels excluded from fitting."),
    ))
    return dict(state="V24_CERTIFIED_ENGINE_BASELINE_EXECUTED",
                engine_module_admission=False, makers=len(scores))


def check_full_profile(summary, expected_variants):
    """Fail closed on reduced grids or partial full-profile records."""
    rows = summary.get("variants", [])
    identities = [row.get("variant", {}).get("id") for row in rows]
    if identities != list(expected_variants) or len(identities) != len(set(identities)):
        raise ValueError("complete fixed twin grid required")
    if any(len(row.get("worlds", [])) != 1 for row in rows):
        raise ValueError("every profile setting needs one complete world")
    if any(world.get("makers", 0) <= 0 for row in rows for world in row["worlds"]):
        raise ValueError("partial or empty twin world")
    return dict(state="FULL_PROFILE_COMPLETE", settings=len(rows), reduced_grid=False,
                confirmation=False)


def validate_profile_plan(plan):
    """Validate an excluded one-world-per-setting cost/control profile."""
    family = plan.get("family")
    expected = [variant["id"] for variant in registry.variants(family)] if family in {"T1", "T3", "T4"} else []
    if (plan.get("kind") != "pilot" or plan.get("task") != "v24-twin-profile"
            or plan.get("confirmation") is not False or plan.get("production") is not False):
        raise ValueError("excluded V24 twin profile required")
    if plan.get("variant_ids") != expected:
        raise ValueError("complete fixed twin grid required")
    if plan.get("world_indices") != [0]:
        raise ValueError("one cost world required")
    if not 0 < plan.get("cpu_cap_seconds", 0) <= 10800:
        raise ValueError("bounded twin profile CPU cap required")
    return expected


def preflight(plan):
    expected = validate_profile_plan(plan)
    schema, _ = ledger.canonical_schema(plan["schema_directory"], plan["schema_pin"])
    exact = exporter(plan, schema)
    backbone = certified_backbone(plan)
    return dict(passed=True, full_grid=len(expected), world_indices=[0],
                scientific_worlds=0, cost_profile_only=True,
                confirmation=False, production=False,
                engine_sources_verified=len(exact.engine_files),
                engine_implementation_hash=exact.implementation,
                certified_backbone_loaded=backbone.__name__)


def evaluate_world(root, variant, phase, world, reference, schema, key, pulse,
                   exact_exporter, plan, *, guard=lambda: None):
    """Execute one complete world with cached exact exports and T1 certified fit."""
    from ..v23.pipeline import Stream
    from ..v23.pipeline_new import score
    import hashlib
    import hmac

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    # Shape lives in twin_pipeline; import here to keep V23 sources untouched.
    from ..v23.twin_pipeline import Shape, population
    shape = Shape(variant["family"])
    gains, exports = [], []
    stream = Stream(root / "maker_scores_points.jsonl.gz")
    try:
        for public, truth in population(variant, phase, world, reference["aggregates"], root, pulse):
            maker = public["maker"]
            unit = root / f"maker-{maker:05d}"
            unit.mkdir()
            prediction, detail, validity = twin_engineering.evaluate(variant, public)
            identity = [variant["family"], variant["id"], phase, world, maker]
            canary = hmac.new(bytes.fromhex(key), canonical(identity), hashlib.sha256).hexdigest()
            bundle = twin_ledger.emit(variant, public, schema, identity, world_identity=identity[:-1])
            ledger.validate(bundle, schema, canaries=[canary])
            if canonical(twin_ledger.decode(variant, bundle, schema)) != canonical(public):
                raise ValueError("whole-twin ledger loses public evidence")
            measured = {name: score(probability, truth) for name, probability in prediction.items()}
            mechanism = "joint" if variant["family"] == "T3" else "individual"
            baseline = ("summary-only" if variant["family"] == "T3" else
                        "group-population" if variant["family"] == "T4" else "population")
            a, b = measured[baseline]["log_loss"], measured[mechanism]["log_loss"]
            gains.append(a - b if a is not None and b is not None else None)
            write(unit / "PUBLIC.json", public)
            write(unit / "TRUTH.json", dict(truth, canary=canary))
            write(unit / "PREDICTION.json", dict(probabilities=prediction, detail=detail, scores=measured))
            write(unit / "VALIDITY.json", validity)
            receipt = exact_exporter.verify_export(unit / "engine", variant, bundle, canaries=[canary])
            exports.append(dict(maker=maker, reader_hash=receipt["reader_hash"]))
            stream.write(dict(maker=maker, scores=measured, baseline=baseline, mechanism=mechanism))
            shape.add(public, truth)
            pulse(phase="v24-whole-twin", variant=variant["id"], world=world, maker=maker + 1)
    finally:
        stream.close()
    realism = twin_contract.realism(variant["family"], reference, shape.result())
    twin_contract.require_realism(realism)
    write(root / "REALISM.json", realism)
    engine = engine_baseline(root, variant, [variant["family"], variant["id"], phase, world],
                             schema, pulse, plan, guard=guard)
    write(root / "SPIKE_INS.json", dict(
        world=world, namespace=phase, synthetic=True, reader_exports=exports,
        evaluator_separate=True, public_records="maker-*/engine/reader.json",
        evaluator_records="maker-*/TRUTH.json", whole_world_reader="WHOLE_WORLD_READER.json",
        engine_module_admission=False,
        prohibition="Do not train a reader on evaluator sidecars.",
    ))
    result = dict(
        variant=variant, world=world, makers=len(gains),
        world_gain=float(np.mean(gains)) if all(value is not None for value in gains) else None,
        realism_sha256=digest(root / "REALISM.json"),
        spike_ins_sha256=digest(root / "SPIKE_INS.json"), engine_baseline=engine,
        cost_profile_only=True, scientific_primary_decided=False,
    )
    write(root / "SUMMARY.json", result)
    return result


def run_profile(root, plan, pulse, *, guard=lambda: None):
    """Run the complete excluded cost grid; never make an effect decision."""
    from ..v23 import twin_pipeline

    expected = validate_profile_plan(plan)
    reference = twin_pipeline.reference(plan)
    schema, _ = ledger.canonical_schema(plan["schema_directory"], plan["schema_pin"])
    exact = exporter(plan, schema)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    variants = registry.variants(plan["family"])
    # Measure the largest generated populations first, while retaining the
    # declared registry order in the final record.
    order = sorted(variants, key=lambda value: (
        value["settings"].get("makers", value["settings"].get("players", 1))
        * value["settings"].get("max_records", value["settings"].get("items", 1))
    ), reverse=True)
    by_id, costs = {}, []
    for variant in order:
        guard()
        started = time.process_time()
        world = evaluate_world(
            root / variant["id"] / "world-0000", variant,
            "v24-twin-cost-profile", 0, reference, schema, plan["canary_key"], pulse,
            exact, plan, guard=guard,
        )
        by_id[variant["id"]] = dict(variant=variant, worlds=[world])
        elapsed = time.process_time() - started
        costs.append(dict(variant=variant["id"], makers=world["makers"], complete_worlds=1))
        pulse(phase="v24-twin-setting-complete", variant=variant["id"],
              setting_cpu_seconds=elapsed, makers=world["makers"])
        write(root / "profile-cost.json", dict(profile=costs), immutable=False)
    summary = dict(
        state="COST_PROFILE_EXECUTED_PENDING_INDEPENDENT_REVIEW",
        family=plan["family"], variants=[by_id[name] for name in expected],
        costs=costs, cost_profile_only=True, confirmation=False, production=False,
        scientific_primary_decided=False, reduced_grid=False,
        limitation=("Complete excluded cost/control profile. Native mechanism "
                    "scores are retained but no scientific effect decision is made."),
    )
    check_full_profile(summary, expected)
    write(root / "SUMMARY.json", summary)
    return summary


def reconstruct_profile(root, plan, pulse=lambda **_: None):
    """Independently rebuild native forecasts, scores, and complete coverage."""
    expected = validate_profile_plan(plan)
    root = Path(root)
    summary = read(root / "SUMMARY.json")
    coverage = check_full_profile(summary, expected)
    checked = {}
    for item in summary["variants"]:
        variant = item["variant"]
        world = item["worlds"][0]
        folder = root / variant["id"] / "world-0000"
        gains = []
        for unit in sorted(folder.glob("maker-*")):
            public = read(unit / "PUBLIC.json")
            truth = read(unit / "TRUTH.json")
            saved = read(unit / "PREDICTION.json")
            native = twin_engineering.evaluate(variant, public)[0]
            rebuilt = {name: checker.scores(np.asarray(probability).reshape(
                -1, len(probability[0])), truth["y"])
                for name, probability in saved["probabilities"].items()}
            for name, probability in native.items():
                if not np.allclose(probability, saved["probabilities"][name], atol=1e-12, rtol=0):
                    raise ValueError("retained twin forecast differs")
            for name, metrics in rebuilt.items():
                for metric, values in metrics.items():
                    value = float(np.mean(values))
                    actual = saved["scores"][name][metric]
                    if not (actual is None and not np.isfinite(value)) and not np.isclose(
                            value, actual, atol=1e-11, rtol=0):
                        raise ValueError("retained twin proper score differs")
            baseline = ("summary-only" if variant["family"] == "T3" else
                        "group-population" if variant["family"] == "T4" else "population")
            mechanism = "joint" if variant["family"] == "T3" else "individual"
            gains.append(float(np.mean(rebuilt[baseline]["log_loss"]
                                       - rebuilt[mechanism]["log_loss"])))
            conformance = read(unit / "engine" / "CONFORMANCE.json")
            exported = read(unit / "engine" / "reader.json")
            if conformance["reader_hash"] != exported["reader_hash"]:
                raise ValueError("cached export receipt differs")
        if len(gains) != world["makers"]:
            raise ValueError("incomplete twin maker coverage")
        observed = float(np.mean(gains)) if np.isfinite(gains).all() else None
        if ((observed is None) != (world["world_gain"] is None)
                or observed is not None and not np.isclose(observed, world["world_gain"], atol=1e-11, rtol=0)):
            raise ValueError("whole-world twin aggregate differs")
        checked[variant["id"]] = dict(makers=len(gains), world_gain=observed,
                                      native_scores_reconstructed=True,
                                      cached_exports_bound=True)
        pulse(phase="v24-twin-independent-review", variant=variant["id"], makers=len(gains))
    return dict(coverage, state="FULL_PROFILE_INDEPENDENTLY_RECONSTRUCTED",
                variants=checked, scientific_primary_decided=False,
                confirmation=False, production=False)


# The common source-frozen V24 parent calls these names for original/replay and
# independent review.  Timing stays in parent/progress accounting, outside the
# byte-replayed scientific tree.
validate_design = validate_profile_plan
execute = run_profile
reconstruct = reconstruct_profile
