"""Finite artifact recipients and maker/editor selection, using actual graphic physics.

The recipient is a Bayesian decoder of visible cells. It is not a human model.
B supplies the shared brief; its practical unknown is the recipient convention.
E separately removes the brief when asking an observer to infer a future choice.
"""
from itertools import product
import math
import random
from ..v16.records import digest, seed_for
from .contracts import Costs, forecast_scores
from .programs import execute

REGIMES = ("correct", "incomplete", "wrong", "changed_recipient")
METHODS = ("surface_task", "retrieval", "inferred_recipient", "supplied_recipient_ceiling")


def normalize(weights):
    total = math.fsum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("nonpositive or nonfinite normalization")
    return [w / total for w in weights]


def softmax(values, precision=1.0):
    peak = max(values)
    return normalize([math.exp(precision*(x-peak)) for x in values])


def draw(probabilities, rng):
    return rng.choices(range(len(probabilities)), weights=probabilities, k=1)[0]


def recipient(artifact, world, model, costs=None):
    """Likelihood comes from executable distances to visible target configurations."""
    if type(artifact) is not int or not 0 <= artifact < 65536:
        raise ValueError("invalid graphic artifact")
    order, precision, prior = model["order"], model["precision"], model["prior"]
    if sorted(order) != list(range(len(world["templates"]))):
        raise ValueError("invalid recipient permutation")
    if len(prior) != len(order) or any(p <= 0 for p in prior):
        raise ValueError("invalid recipient prior")
    logits = []
    for interpretation, template_index in enumerate(order):
        distance = ((artifact ^ world["templates"][template_index]) & world["visible"]).bit_count()
        if costs:
            costs.charge("hypothetical_execution", world["visible"].bit_count())
        logits.append(math.log(prior[interpretation]) - precision*distance)
    return softmax(logits)


def model_space(world):
    n = len(world["templates"])
    # A declared finite prior over cyclic conventions, reliability and prior bias.
    for offset, precision, bias in product(range(n), (0.4, 1.2, 3.0), range(n+1)):
        prior = [1.0 if bias == n else (3.0 if k == bias else 1.0) for k in range(n)]
        yield dict(order=[(k+offset) % n for k in range(n)],
                   precision=precision, prior=normalize(prior))


def infer_models(world, demonstrations, *, effort=None, costs=None):
    models = list(model_space(world))
    # Deterministic balanced subset; no hidden target or evaluation label enters here.
    if effort is not None and effort < len(models):
        indices = [i*len(models)//effort for i in range(effort)]
        models = [models[i] for i in indices]
    logs = []
    for model in models:
        ll = 0.0
        for demonstration in demonstrations:
            p = recipient(demonstration["artifact"], world, model, costs)
            ll += math.log(p[demonstration["response"]])
            if costs:
                costs.charge("selection")
        logs.append(ll)
    return list(zip(models, softmax(logs)))


def mixture(artifact, world, models, costs=None):
    result = [0.0]*len(world["templates"])
    for model, weight in models:
        p = recipient(artifact, world, model, costs)
        for k, value in enumerate(p):
            result[k] += weight*value
    return normalize(result)


def opportunities(artifact, world):
    # All methods see precisely the same publicly executable alternatives.
    active = sorted({c for template in world["templates"] for c in range(16) if template & (1 << c)})
    candidates = [{"id": "STOP", "program": [], "artifact": artifact}]
    for cell in active:
        action = cell+16 if artifact & (1 << cell) else cell
        result = execute([action], initial=artifact)
        candidates.append({"id": str(action), "program": [action], "artifact": result["artifact"]})
    return candidates


def outcome_value(probabilities, desired):
    # Match a declared distribution, including intentionally uncertain targets.
    return -sum((a-b)**2 for a, b in zip(probabilities, desired))


def utilities(candidates, world, desired, models, role, costs):
    target = world["templates"][max(range(len(desired)), key=desired.__getitem__)]
    values = []
    for candidate in candidates:
        costs.charge("proposal_generation")
        costs.charge("hypothetical_execution", len(candidate["program"]))
        if role == "physical":
            value = -((candidate["artifact"] ^ target) & world["visible"]).bit_count()/world["visible"].bit_count()
        else:
            value = outcome_value(mixture(candidate["artifact"], world, models, costs), desired)
        values.append(value - .025*len(candidate["program"]))
        costs.charge("selection")
    return values


def accepted_distribution(candidates, world, desired, models, role, knowledge, costs):
    """Enumerate actual proposal and original-maker ratification probabilities."""
    if knowledge == "incomplete":
        belief = [(dict(m, precision=0.0), weight) for m, weight in models]
    elif knowledge == "wrong":
        belief = [(dict(m, order=m["order"][1:]+m["order"][:1]), weight) for m, weight in models]
    else:
        belief = models
    editor_role = "communicator" if role == "producer_editor" else role
    proposer = softmax(utilities(candidates, world, desired, belief, editor_role, costs), 5)
    if role != "producer_editor":
        return proposer
    # Producer's direct physical subgoal and editor's shared communication brief
    # are separate. Original maker ratifies using a joint physical/effect value.
    physical = utilities(candidates, world, desired, models, "physical", costs)
    effect = utilities(candidates, world, desired, models, "communicator", costs)
    retained = [0.0]*len(candidates)
    for i, mass in enumerate(proposer):
        costs.charge("selection")
        keep = i if .25*physical[i]+effect[i] >= .25*physical[0]+effect[0] else 0
        retained[keep] += mass
    return normalize(retained)


def make_case(namespace, constructor_index, history_index, regime):
    if regime not in REGIMES:
        raise ValueError("unknown recipient regime")
    rng = random.Random(seed_for(namespace, "recipient-constructor", constructor_index))
    cells = rng.sample(range(16), 8)
    n = 2 + constructor_index % 2
    # Both overlapping and disjoint cue architectures, with random cell identities.
    shared = 1 << cells[-1] if constructor_index % 3 else 0
    templates = [shared | (1 << cells[2*k]) | (1 << cells[2*k+1]) for k in range(n)]
    world = {"templates": templates, "visible": sum(1 << c for c in cells),
             "structure": "overlap" if shared else "disjoint"}
    rng = random.Random(seed_for(namespace, constructor_index, history_index, "recipient-history"))
    models = list(model_space(world))
    model = rng.choice(models)
    changed = dict(model, order=model["order"][1:]+model["order"][:1])
    goal = rng.randrange(n)
    desired = [float(k == goal) for k in range(n)]
    if history_index % 4 == 3:
        desired = [1/n]*n
    role = ("physical", "communicator", "producer_editor")[history_index % 3]
    # Actual preceding maker execution; artifact-only requests receive no trace.
    initial_program = [c for c in cells if templates[goal] & (1 << c)]
    if history_index % 2:
        initial_program = initial_program[:-1]
    initial_run = execute(initial_program)
    artifact = initial_run["artifact"]
    candidates = opportunities(artifact, world)
    demonstrations = []
    for i in range(12):
        display = templates[i % n] ^ (1 << rng.choice(cells))
        response = draw(recipient(display, world, model), rng)
        demonstrations.append({"artifact": display, "response": response})
    # Recipient intervention is applied to an unseen continuation. Calibration
    # examples from the changed recipient are explicitly supplied only in that arm.
    deployed_model = changed if regime == "changed_recipient" else model
    if regime == "changed_recipient":
        for example in demonstrations:
            example["response"] = draw(recipient(example["artifact"], world, deployed_model), rng)
    knowledge = regime if regime != "changed_recipient" else "correct"
    target_costs = Costs()
    true_edit = accepted_distribution(candidates, world, desired, [(deployed_model, 1.0)], role, knowledge, target_costs)
    accepted = draw(true_edit, rng)
    execution = execute(candidates[accepted]["program"], initial=artifact)
    response_distribution = recipient(execution["artifact"], world, deployed_model)
    response = draw(response_distribution, rng)
    public = {"schema": "v17.recipient.1", "world": world, "artifact": artifact,
              "candidates": candidates, "shared_brief": desired, "role": role,
              "editor_knowledge_condition": knowledge, "calibration": demonstrations,
              "query_cost": 1, "answer_support": [c["id"] for c in candidates]}
    private = {"maker_training": demonstrations, "current_goal": desired, "earlier_goals": [goal],
               "actual_constraints": [], "believed_constraints": [], "considered_options": candidates,
               "realized_history": initial_run["trace"], "contributor_role": role,
               "recipient_state": deployed_model, "previous_recipient_state": model,
               "proposal_role": "editor" if role == "producer_editor" else "maker",
               "execution_role": "editor" if role == "producer_editor" else "maker",
               "ratification_role": "original maker" if role == "producer_editor" else None,
               "accepted_edit": candidates[accepted]["id"], "edit_distribution": true_edit,
               "execution": execution, "recipient_distribution": response_distribution,
               "recipient_response": str(response), "target_generation_costs": target_costs.receipt()}
    return {"case_id": digest([namespace, constructor_index, history_index, regime]),
            "constructor_id": digest([namespace, constructor_index]), "history_id": digest([namespace, constructor_index, history_index]),
            "namespace": namespace, "family": "B", "regime": regime, "public": public, "private": private,
            "public_problem_sha256": digest(public), "private_construction_sha256": digest([world, public, deployed_model]),
            "structural_family": world["structure"]}


def predict(public, method, *, supplied_model=None, effort=None):
    expected = {"schema", "world", "artifact", "candidates", "shared_brief", "role",
                "editor_knowledge_condition", "calibration", "query_cost", "answer_support"}
    if set(public) != expected or public["schema"] != "v17.recipient.1":
        raise ValueError("recipient public schema violation")
    world, costs = public["world"], Costs()
    if public["candidates"] != opportunities(public["artifact"], world) or public["answer_support"] != [c["id"] for c in public["candidates"]]:
        raise ValueError("invalid public candidate execution/support")
    costs.charge("training_acquisition", len(public["calibration"]))
    if method == "supplied_recipient_ceiling":
        if supplied_model is None:
            raise ValueError("ceiling requires an explicitly supplied recipient")
        models = [(supplied_model, 1.0)]
        costs.charge("definition_storage", len(supplied_model["order"])*2+1)
    elif method == "inferred_recipient":
        models = infer_models(world, public["calibration"], effort=effort, costs=costs)
        costs.charge("definition_storage", sum(2*len(m["order"])+2 for m, _ in models))
    elif method == "surface_task":
        n = len(world["templates"])
        models = [(dict(order=list(range(n)), precision=1.2, prior=[1/n]*n), 1.0)]
        costs.charge("definition_storage", 2*n+1)
    elif method == "retrieval":
        models = []
        costs.charge("definition_storage", len(public["calibration"])*2)
    else:
        raise ValueError("unknown recipient method")
    candidates = public["candidates"]
    if method == "retrieval":
        values = []
        predicted = []
        for candidate in candidates:
            distances = []
            for example in public["calibration"]:
                costs.charge("retrieval", 2)
                distances.append((candidate["artifact"] ^ example["artifact"]).bit_count())
            nearest = min(distances)
            n = len(world["templates"])
            counts = [1.0]*n
            for example, distance in zip(public["calibration"], distances):
                costs.charge("selection")
                if distance == nearest:
                    counts[example["response"]] += 1
            predicted.append(normalize(counts))
            values.append(outcome_value(predicted[-1], public["shared_brief"]) - .025*len(candidate["program"]))
            costs.charge("proposal_generation")
        edit = softmax(values, 5)
    else:
        edit = accepted_distribution(candidates, world, public["shared_brief"], models,
             "physical" if method == "surface_task" else public["role"], public["editor_knowledge_condition"], costs)
        predicted = [mixture(c["artifact"], world, models, costs) for c in candidates]
    effect = normalize([sum(w*p[k] for w,p in zip(edit,predicted)) for k in range(len(world["templates"]))])
    return {"edit": dict(zip(public["answer_support"], edit)),
            "recipient": {str(i): p for i,p in enumerate(effect)}, "costs": costs.receipt(16),
            "represented_variables": ["artifact", "recipient_convention", "recipient_precision", "recipient_prior"]
                if method in ("inferred_recipient","supplied_recipient_ceiling") else ["artifact", "empirical_response"],
            "hypotheses": len(models)}


def evaluate_case(case, design=None):
    public, private = case["public"], case["private"]
    rows = []
    for method in METHODS:
        result = predict(public, method, supplied_model=private["recipient_state"] if method == METHODS[-1] else None)
        for target, truth, support in (
            ("accepted_edit", private["accepted_edit"], public["answer_support"]),
            ("recipient_outcome", private["recipient_response"], list(result["recipient"]))):
            probabilities = result["edit" if target == "accepted_edit" else "recipient"]
            rows.append({"method": method, "target": target, "probabilities": probabilities,
                         **forecast_scores(probabilities, support, truth), "costs": result["costs"],
                         "represented_variables": result["represented_variables"], "hypotheses": result["hypotheses"],
                         "evidence_tier": "supplied_shared_brief_and_recipient_calibration",
                         "assistance": "true recipient supplied" if method == METHODS[-1] else "recipient inferred or approximated",
                         "missing_output": False, "invalid_program": False,
                         "task_success": max(probabilities, key=probabilities.get) == truth})
    return rows
