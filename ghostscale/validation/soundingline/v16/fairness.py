"""X04 source-bound observation comparisons and independently checked costs."""
from copy import deepcopy
from itertools import combinations
import importlib
from .records import canonical, digest, read
from .reference import interpret

PREFIX = "ghostscale.validation.soundingline.v16."
READING = {"K02", "P01", "P02", "P03"}
BATCH = {"K01", "K03", "K04", "K05"}
AUDITORS = {"K03": "purpose", "K04": "options", "K05": "attention", "P04": "mechanism",
    "S03": "dependency", "M01": "recognition", "M02": "selection", "M03": "audience",
    "M04": "multi_actor", "V01": "tradeoffs", "V02": "tradeoffs", "V03": "preference_probe"}
ACCESS = {
    "K01": "Personal and primitive use identical own training. Pooled receives equally many generic training attempts, not the same examples.",
    "K03": "All arms receive old/new training, target and allowance. Inhibition pays its candidate-checking cost separately.",
    "K04": "All arms receive the same observed transitions and budget. Generic options are a declared fixed procedure rival.",
    "K05": "Same offered topics and total attention allocation; processed instructions differ by the attention intervention. Learned outputs determine later competence.",
    "O01": "Probe access ablation is explicitly different evidence with its query cost; model alternatives use the same augmented evidence.",
    "O02": "Supplied-state upper bounds: own resources, actual constraints and known repertoire differ. No equal-access learned-reader claim.",
    "O03": "Same observations and finite allowance; restricted cause models are declared rivals.",
    "O04": "Same accessible evidence for different belief models; without-probe has less paid evidence.",
    "S01": "Memory-only evidence ablation differs in artifacts. Self-model and Bayesian error reader share full evidence.",
    "S02": "Same observations, monitoring charge and repair opportunities; different decisions are scored on paired reset/outcome draws.",
    "S04": "Same full evidence for self-model and ordinary comparators; memory-only is an explicit artifact ablation.",
    "S05": "Same initial memory and artifacts, paired outcome draws. Later histories differ through each policy's actual interventions.",
    "S03": "Same initial assembly and inspection offers. Purchased inspections change later access with explicit costs.",
    "P04": "Same initial artifact/world model; prior-work ablation buys no work. Separate W1/W2 interfaces, no fitted-parameter transfer.",
    "M01": "Same references and anonymous works; methods may ignore components. No additional private maker label is supplied.",
    "M02": "Same selected works and publicly declared selection information; methods may ignore it. Full rejected candidates remain evaluator-side unless purchased.",
    "M03": "Same initial public history/audience and query offers. Later rehearsals depend on paid query policy; audience cues alone are already public.",
    "M04": "Same initial released artifacts and probe offers. Later producer/revision/unselected/brief views differ through paid queries.",
    "V01": "Identical observed episodes and candidate context; readers make different assumptions about persistent profiles.",
    "V02": "Identical dated episodes and context; chronological and pooled assumptions differ, not reader access.",
    "V03": "Same initial tradeoff record and probe offers. Actual paid interventions supply different later evidence.",
    **{card: "Same artifact, public law and permitted works for maker/direct/generic/primitive readers; without-query explicitly removes purchased evidence." for card in READING},
    **{f"R{i:02d}": "Same initial beliefs, offers and permitted feedback law; policies earn different learning histories from actual choices. Paired world/offer/outcome streams remain evaluator-side." for i in range(1, 5)},
    "R05": "Enact, observe and unrelated arms receive declared different practice/example access. Later map belief is acquired, not an oracle label."}


def require_same(values, label):
    if not values or len({digest(value) for value in values}) != 1:
        raise ValueError("unequal supposedly matched observations: " + label)


def information(card, frames):
    if not frames or card not in ACCESS:
        raise ValueError("missing explicit consumer information contract")
    checked = []
    if card == "K01":
        training = frames[0]["public"]["arm_training"]
        require_same([training["personal"], training["primitive"]], "own training")
        require_same([len(record["attempts"]) for record in training.values()], "training opportunity count")
        checked += ["identical own training", "equal generic opportunity count"]
    elif card in READING:
        require_same([frame["public"] for frame in frames[:4]], "four full-access reading rivals")
        checked.append("four full-access readers share exact observation bytes")
    elif card in {"M01", "M02", "V01", "V02", "O03", "S02"}:
        require_same([frame["public"] for frame in frames], card)
        checked.append("all compared readers share exact observation bytes")
    elif card in {"S01", "S04"}:
        # Alphabetical saved arm order is explicit in the original transport.
        n = 2 if card == "S01" else 3
        require_same([frames[i]["public"] for i in range(len(frames)) if i != n-1], card+" full-access")
        checked.append("full-access competitors match; memory-only is retained separately")
    elif card in {"R01", "R02", "R03", "R04"}:
        first = {}
        for frame in frames:
            if frame["kind"].endswith(":choose"):
                first.setdefault(frame["options"]["policy"], frame["public"])
        require_same(list(first.values()), "inquiry initial state")
        checked.append("every policy begins from identical query input")
    elif card in {"S03", "M03", "M04", "V03", "P04"}:
        initial = [frame["public"] for frame in frames if frame["public"].get("phase") in {1, "query", "choose", "before"}]
        # Actual stage spelling is retained; pair-equality inventory below never
        # asserts that post-intervention observations are matched.
        if initial:
            require_same(initial, card+" initial phase")
            checked.append("initial query inputs are identical")
    groups = {}
    for index, frame in enumerate(frames):
        groups.setdefault(digest(frame["public"]), []).append(index)
    differences = []
    for left, right in combinations(range(len(frames)), 2):
        a, b = frames[left], frames[right]
        if a["kind"] == b["kind"] and a["public"] != b["public"]:
            differences.append({"left": left, "right": right,
                "different_top_level_fields": sorted(k for k in set(a["public"]) | set(b["public"]) if a["public"].get(k) != b["public"].get(k))})
    return {"interpretation": ACCESS[card], "validated_matches": checked,
            "identical_observation_groups": list(groups.values()), "different_observation_pairs": differences,
            "measurement_scope": "whole all-arm request" if card in BATCH else "individual declared reader request"}


def check_w1_search(search, library, budget):
    spent = 0
    for attempt in search["attempted_programs"]:
        primitives = []
        for token in attempt["tokens"]:
            if type(token) is int:
                primitives.append(token)
            else:
                primitives.extend(library[int(token[1:])])
        result = interpret(primitives)
        if (result["legal"], result["artifact"], result["primitive_cost"]) != (attempt["legal"], attempt["artifact"], attempt["cost"]):
            raise ValueError("macro expansion/primitive execution charge mismatch")
        spent += result["primitive_cost"]
    if spent != search["search_primitives"] or spent > budget:
        raise ValueError("reported search cost or finite budget mismatch")
    return spent


def check_craft(public, result):
    for arm, prediction in result.items():
        library = prediction["learned_library"]
        record = public["arm_training"][arm]
        if prediction["costs"]["training_primitives"] != sum(map(len, record["attempts"])):
            raise ValueError("training cost erased")
        if prediction["costs"]["library_definition"] != sum(map(len, library)):
            raise ValueError("library definition cost erased")
        for search in prediction["submissions"]:
            check_w1_search(search, library, public["search_primitive_budget"])


def old_frame_costs(frame):
    public, result, kind = frame["public"], frame["result"], frame["kind"]
    if result.get("model_mismatch"):
        return {"state": "mismatch retained"}
    if kind == "craft":
        check_craft(public, result)
    elif kind == "reading":
        strategy = frame["options"]["strategy"]
        libraries = [[], [[0, 1]], [[2, 3]], [[0, 1], [2, 3]]]
        library = libraries[-1] if strategy == "direct-table" else libraries[max(range(4), key=lambda i: result["history_probabilities"][i])]
        check_w1_search(result["reconstruction"], library, public["reader_action_budget"])
        evidence = 1+len(public["permitted_prior_artifacts"])
        if result["costs"] != {"likelihood_evaluations": 4*evidence+64,
            "permitted_evidence_items": evidence, "query_cost": sum(public["query_costs"].values()),
            "finite_model_training_support": 2340}:
            raise ValueError("original reading cost receipt differs")
        return {"original_likelihood_counter": 4*evidence+64,
            "actual_likelihood_call_invocations": (8 if strategy == "direct-table" else 4)*evidence+64,
            "qualification": "direct table repeats evidence likelihood calls; old counter is incomplete for invocation-cost comparison. Calls can hit caches; 2340 is model support, not measured training work."}
    elif kind == "opportunity":
        causes = list(public["allowed_causes"])
        model = frame["options"]["model"]
        if model == "fixed-menu": causes = [x for x in causes if x != "consideration"]
        if model == "budget-only": causes = [x for x in causes if x in {"search", "purpose"}]
        if model == "accurate-belief": causes = [x for x in causes if x != "false-affordance"]
        expected = len(causes)*9*(1+len(public["observations"]))
        if result["costs"] != {"state_evaluations": expected, "query_cost": public["query_cost"]} or expected > public["reader_search_budget"]:
            raise ValueError("opportunity evidence/finite-model charge mismatch")
    elif kind == "self":
        if result["costs"] != {"state_evaluations": 36*(1+len(public["artifacts"])),
            "monitoring": public["monitoring_cost"], "repair": public["repair_cost"] if result["repair"] else 0}:
            raise ValueError("self monitoring or repair charge mismatch")
    elif kind == "critic":
        if result["costs"]["search_evaluations"] != min(2, public["search_budget"]):
            raise ValueError("critic successor charge mismatch")
    elif "inquiry_stable:" in kind:
        from .inquiry_stable_reference import readout
        from .audit_inquiry import close
        close(result, readout(kind, public, frame["options"]))
    return {"state": "cost checks passed", "scope": "declared primitive/model work; not CPU instructions"}


def audit_unit(root, row, frames):
    card = row["card_id"]
    if card in AUDITORS:
        importlib.import_module(PREFIX+"audit_"+AUDITORS[card]).audit_unit(root, row)
    if card.startswith("R"):
        from .audit_inquiry import audit_unit as audit_inquiry
        audit_inquiry(root, row)
    return [old_frame_costs(frame) for frame in frames]


def cost_fields(value, prefix=""):
    """Preserve named resource fields, never add unlike units or nested totals."""
    output = {}
    if isinstance(value, dict):
        for key, item in value.items():
            path = prefix+"/"+key
            if isinstance(item, (float, int)) and not isinstance(item, bool) and any(
                    word in key for word in ("cost", "primitives", "evaluations", "terms", "queries", "training", "definition")):
                output[path] = item
            elif isinstance(item, (dict, list)):
                output.update(cost_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.update(cost_fields(item, prefix+"/"+str(index)))
    return output


def amortization(public, result, workloads=(1, 8, 32)):
    check_craft(public, result)
    rows = []
    for name, arm in result.items():
        searches = [item["search_primitives"] for item in arm["submissions"]]
        initial = arm["costs"]["training_primitives"]
        definition = arm["costs"]["library_definition"]
        for workload in workloads:
            rows.append({"arm": name, "future_task_packets": workload,
                "training_primitives_once": initial, "definition_tokens_once": definition,
                "search_primitives": workload*sum(searches),
                "training_plus_search_primitives_per_packet": initial/workload+sum(searches),
                "definition_tokens_per_packet": definition/workload,
                "qualification": "declared repetition of this paired target packet, not newly sampled tasks; definition tokens separate from executed primitives"})
    return rows
