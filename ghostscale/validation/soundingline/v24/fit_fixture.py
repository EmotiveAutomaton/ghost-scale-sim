"""Complete retained D5 fixture; no generator or fresh scientific draws.

The scalar checker reconstructs chronology, ridge selection, every fitted
objective/gradient certificate and every prediction from recorded coefficients.
It does not rerun optimization or treat retained cost inputs as new evidence.
"""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, softmax

from ..v23.common import canonical, digest, read, write
from ..v23.pipeline import Stream
from ..v23.family import rows as stream_rows
from .exact_fit import load_certified

ARMS = ("baseline", "own", "donor")
CONDITIONS = ("planted", "uninformative", "constant", "future", "option")
PERMUTATION = [2, 0, 3, 1]


def verify_inputs(plan):
    for name, expected in plan["input_files"].items():
        if digest(Path(plan["input_root"]) / name) != expected:
            raise ValueError("retained fixture input changed")
    if plan["identity"] != "v24-retained-fit-preparation-1" or plan["new_scientific_worlds"] != 0:
        raise ValueError("retained engineering only")


def inputs(plan, condition):
    base = "planted" if condition in ("future", "option") else condition
    folder = Path(plan["input_root"]) / base
    records, labels = read(folder/"READER.json"), read(folder/"LABELS.json")
    if condition == "future":
        cutoff = records[plan["future_prefix"]-1]["open_at"]
        for row, label in zip(records, labels, strict=True):
            if row["resolved_at"] >= cutoff:
                label["action"] = (label["action"]+1) % 4
    if condition == "option":
        inverse = np.argsort(PERMUTATION)
        for row, label in zip(records, labels, strict=True):
            for arm in ARMS:
                row["features"][arm] = np.asarray(row["features"][arm])[PERMUTATION].tolist()
            label["action"] = int(inverse[label["action"]])
    return records, labels


def execute(root, plan, pulse, *, input_validator=verify_inputs):
    input_validator(plan)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    model = load_certified(plan["backbone_source"], plan["backbone_sha256"],
                           accelerated=True, guard=lambda: pulse(phase="fit"))
    # The unchanged sibling function supplies the chronological outer loop.
    source = Path(plan["profile_source"])
    if digest(source) != plan["profile_sha256"]:
        raise ValueError("profile source changed")
    tree = ast.parse(source.read_bytes())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict_all"]
    if len(nodes) != 1:
        raise ValueError("missing frozen predictor")
    namespace = dict(np=np, datetime=datetime, timezone=timezone, ARMS=ARMS,
                     engine=lambda: (model.Backbone, model.Choice, None, None))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    original_fit = model._fit
    for condition in CONDITIONS:
        records, labels = inputs(plan, condition)
        stream = Stream(root/(condition+"_fits_points.jsonl.gz"))
        call_index = [0]
        def capture(rows, center, penalty, initial=None, *, maxiter=500):
            offset, receipt = original_fit(rows, center, penalty, initial, maxiter=maxiter)
            stream.write(dict(call=call_index[0], row_ids=[r.id for r in rows],
                              center=np.asarray(center).tolist(), penalty=float(penalty),
                              offset=offset.tolist(), receipt=receipt))
            call_index[0] += 1
            return offset, receipt
        model._fit = capture
        try:
            predictions = namespace["predict_all"](records, labels,
                lambda value: pulse(condition=condition, **value))
        finally:
            stream.close()
            model._fit = original_fit
        write(root/(condition+"_predictions_points.json"), predictions)
        write(root/(condition+"_SUMMARY.json"), dict(targets=len(records), fit_calls=call_index[0]))
    write(root/"RAW_INDEX.json", {p.name:dict(sha256=digest(p), bytes=p.stat().st_size)
          for p in sorted(root.iterdir()) if p.is_file()})


def objective_gradient(rows, labels, arm, center, offset, penalty):
    """Independent scalar summation; no use of the grouped implementation."""
    beta = np.asarray(center) + offset
    loss = penalty * float(np.dot(offset, offset))
    gradient = [2 * penalty * float(x) for x in offset]
    for row in rows:
        x = np.asarray(row["features"][arm])
        logits = [math.fsum(float(a)*float(b) for a,b in zip(option, beta, strict=True)) for option in x]
        top = max(logits)
        masses = [math.exp(z-top) for z in logits]
        total = math.fsum(masses)
        probability = [m/total for m in masses]
        chosen = labels[row["key"]]
        loss += top + math.log(total) - logits[chosen]
        # V23 D5's Choice constructor uses exactly unit weights.
        for j in range(len(beta)):
            gradient[j] += math.fsum(p*float(option[j]) for p,option in zip(probability,x,strict=True))-float(x[chosen,j])
    return loss, np.asarray(gradient)


def reconstruct(root, plan, pulse, *, input_validator=verify_inputs):
    input_validator(plan)
    root = Path(root)
    raw = read(root/"RAW_INDEX.json")
    expected = {f"{c}_{suffix}" for c in CONDITIONS for suffix in
                ("fits_points.jsonl.gz", "predictions_points.json", "SUMMARY.json")}
    if set(raw) != expected:
        raise ValueError("incomplete retained fixture roster")
    for name, item in raw.items():
        if digest(root/name) != item["sha256"] or (root/name).stat().st_size != item["bytes"]:
            raise ValueError("retained output hash mismatch")
    all_predictions = {}
    counts = {}
    maxima = dict(objective=0., gradient=0., prediction=0., retained_reference=0.)
    for condition in CONDITIONS:
        records, label_rows = inputs(plan, condition)
        labels = {r["key"]:r["action"] for r in label_rows}
        trace = iter(stream_rows(root/(condition+"_fits_points.jsonl.gz")))
        predictions = read(root/(condition+"_predictions_points.json"))
        if len(records) != len(predictions):
            raise ValueError("target roster mismatch")
        counter = [0]
        def fit(expected_rows, arm, center, penalty):
            try:
                item = next(trace)
            except StopIteration as exc:
                raise ValueError("incomplete fit trace") from exc
            if (item["call"] != counter[0] or item["row_ids"] != [r["key"] for r in expected_rows]
                    or item["penalty"] != penalty or item["center"] != np.asarray(center).tolist()):
                raise ValueError("fit membership, center or ridge differs")
            counter[0] += 1
            offset = np.asarray(item["offset"])
            if offset.shape != np.asarray(center).shape or not np.isfinite(offset).all():
                raise ValueError("invalid fitted coefficients")
            if not expected_rows:
                if np.any(offset != 0):
                    raise ValueError("empty fit must be zero")
                return offset
            loss, gradient = objective_gradient(expected_rows, labels, arm, center, offset, penalty)
            bound = float(gradient @ gradient)/(4*penalty)
            if not math.isfinite(bound) or bound > 1e-9*max(1.,abs(loss)):
                raise ValueError("independent strong-convexity certificate failed")
            objective_error = abs(loss-item["receipt"]["objective"])
            gradient_error = abs(float(np.max(np.abs(gradient)))-item["receipt"]["gradient_max"])
            if objective_error > 1e-9 or gradient_error > 1e-9:
                raise ValueError("independent objective or gradient mismatch")
            maxima["objective"] = max(maxima["objective"],objective_error)
            maxima["gradient"] = max(maxima["gradient"],gradient_error)
            return offset
        for index,(target,prediction) in enumerate(zip(records,predictions,strict=True)):
            training = sorted((r for r in records if r["resolved_at"] < target["open_at"]),
                              key=lambda r:(r["open_at"],r["key"]))
            if len({r["key"] for r in training}) != len(training):
                raise ValueError("duplicate training records")
            if any(prediction[k] != target[k] for k in ("key","writer","prompt")):
                raise ValueError("target identity mismatch")
            for arm in ARMS:
                dimension = len(target["features"][arm][0])
                beta = fit(training, arm, np.zeros(dimension), .005)
                offsets = {}
                for maker in sorted({r["writer"] for r in training}):
                    own = [r for r in training if r["writer"] == maker]
                    penalty = 1.
                    calibration = own[max(4,int(.8*len(own))):]
                    if calibration:
                        cutoff = min(r["open_at"] for r in calibration)
                        historical = [r for r in training if r["resolved_at"] < cutoff]
                        ownpast = [r for r in historical if r["writer"] == maker]
                        if ownpast:
                            prior = fit(historical,arm,np.zeros(dimension),.005)
                            candidates = []
                            for value in (.01,.1,1.,10.):
                                offset = fit(ownpast,arm,prior,value)
                                losses = []
                                for row in calibration:
                                    logits = np.asarray(row["features"][arm]) @ (prior+offset)
                                    losses.append(float(logsumexp(logits)-logits[labels[row["key"]]]))
                                candidates.append((math.fsum(losses)/len(losses),value))
                            penalty = min(candidates)[1]
                    offsets[maker] = fit(own,arm,beta,penalty)
                predicted = softmax(np.asarray(target["features"][arm]) @
                                    (beta+offsets.get(target["writer"],np.zeros(dimension))))
                actual = prediction["arms"][arm]
                if (actual["training_keys"] != [r["key"] for r in training]
                        or actual["cold_start"] != (not training)):
                    raise ValueError("chronological output membership differs")
                error = float(np.max(np.abs(predicted-np.asarray(actual["probabilities"]))))
                if error > 1e-12:
                    raise ValueError("independent prediction differs")
                maxima["prediction"] = max(maxima["prediction"],error)
            pulse(condition=condition,target=index,fit_calls=counter[0])
        if next(trace,None) is not None:
            raise ValueError("extra fit trace")
        if read(root/(condition+"_SUMMARY.json")) != dict(targets=len(records),fit_calls=counter[0]):
            raise ValueError("fit counts differ")
        counts[condition] = dict(targets=len(records),fit_calls=counter[0])
        all_predictions[condition] = predictions
        reference_path = Path(plan["input_root"])/condition/"PREDICTIONS.json"
        if reference_path.exists():
            for a,b in zip(predictions,read(reference_path),strict=True):
                for arm in ARMS:
                    error = float(np.max(np.abs(np.asarray(a["arms"][arm]["probabilities"])
                                               - b["arms"][arm]["probabilities"])))
                    maxima["retained_reference"] = max(maxima["retained_reference"],error)
                    if error > 1e-9:
                        raise ValueError("retained scalar prediction differs")
    original = all_predictions["planted"]
    future = all_predictions["future"]
    option = all_predictions["option"]
    prefix = plan["future_prefix"]
    inverse = np.argsort(PERMUTATION)
    controls = dict(
        future=max(float(np.max(np.abs(np.asarray(a["arms"][arm]["probabilities"])
            - b["arms"][arm]["probabilities"]))) for a,b in zip(original[:prefix],future[:prefix]) for arm in ARMS),
        option=max(float(np.max(np.abs(np.asarray(a["arms"][arm]["probabilities"])
            - np.asarray(b["arms"][arm]["probabilities"])[inverse]))) for a,b in zip(original,option) for arm in ARMS),
        constant=max(float(np.max(np.abs(np.asarray(r["arms"]["own"]["probabilities"])
            - r["arms"]["donor"]["probabilities"]))) for r in all_predictions["constant"]))
    if controls["future"] > 1e-7 or controls["option"] > 1e-9 or controls["constant"] > 1e-12:
        raise ValueError("retained structural controls failed")
    return dict(state="RETAINED_FULL_FIXTURE_VERIFIED", new_scientific_worlds=0,
                conditions=counts, maximum_errors=maxima, controls=controls,
                scope="All retained targets, chronological fits, ridge selections, certificates and predictions; no fresh pilot or scientific family acceptance.",
                complete_fresh_pilot=False, original_v23_failures_preserved=True)
