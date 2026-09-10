"""X02 representation controls with explicit semantic targets.

Macro spelling is compiled to expanded primitive traces before scientific use.
World isomorphisms are tested on predictive quantities; bounded lexical search
order is a separate algorithmic parameter, not silently treated as equivariant.
"""
import copy
from itertools import permutations
from .records import digest

TOLERANCE = 1e-10
SWAP_MOTIFS = (2, 3, 0, 1)
MAPPINGS = tuple(permutations(range(4)))


def macro_record(program, cuts, spelling="first", reverse_definitions=False):
    if any(type(action) is not int for action in program):
        raise ValueError("macro bodies contain only primitive integers")
    boundaries = [0, *cuts, len(program)]
    if boundaries != sorted(set(boundaries)) and program:
        raise ValueError("macro boundaries must partition the expanded trace")
    if not program:
        return {"procedures": {}, "calls": []}
    definitions, calls = {}, []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        name = "routine_"+digest([spelling, index])[:12]
        definitions[name] = list(program[start:end])
        calls.append(name)
    if reverse_definitions:
        definitions = dict(reversed(list(definitions.items())))
    return {"procedures": definitions, "calls": calls}


def compile_macros(record):
    if set(record) != {"procedures", "calls"} or len(record["calls"]) > 1024:
        raise ValueError("invalid bounded macro interface")
    result = []
    for name in record["calls"]:
        if name not in record["procedures"]:
            raise ValueError("unknown procedure")
        body = record["procedures"][name]
        if not body or any(type(action) is not int for action in body):
            raise ValueError("only nonempty primitive bodies are supported")
        result.extend(body)
        if len(result) > 4096:
            raise ValueError("expanded primitive budget exceeded")
    return result


def board(value, permutation):
    if type(value) is not int or not 0 <= value < 1 << len(permutation):
        raise ValueError("unsupported board for relabeling")
    return sum(1 << permutation[cell] for cell in range(len(permutation)) if value & (1 << cell))


def action(value, permutation):
    cells = len(permutation)
    if type(value) is not int or not 0 <= value < 2*cells:
        raise ValueError("unsupported primitive for relabeling")
    return permutation[value % cells] + cells*(value//cells)


def pushed_probabilities(values, permutation):
    result = [0.0]*len(values)
    if len(values) == 1 << len(permutation):
        for artifact, value in enumerate(values):
            result[board(artifact, permutation)] = value
    elif len(values) == len(permutation):
        for cell, value in enumerate(values):
            result[permutation[cell]] = value
    else:
        raise ValueError("probability support is not declared by this recoding")
    return result


def reading_observation(public, permutation=SWAP_MOTIFS):
    # This particular bijection preserves the two finite acquired motif families.
    if tuple(permutation) != SWAP_MOTIFS:
        raise ValueError("reading law permits only the declared motif-family exchange")
    result = copy.deepcopy(public)
    result["final_artifact"] = board(result["final_artifact"], permutation)
    observations = [result["declared_context"]["current_observation"], *result["permitted_prior_artifacts"]]
    for item in observations:
        item["artifact"] = board(item["artifact"], permutation)
        item["target"] = board(item["target"], permutation)
        item["prefix"] = [action(command, permutation) for command in item["prefix"]]
        item["feasible"] = [action(command, permutation) for command in item["feasible"]]
    result["target_request"]["future_target"] = board(result["target_request"]["future_target"], permutation)
    if "feasible" in result["target_request"]:
        result["target_request"]["feasible"] = [action(command, permutation) for command in result["target_request"]["feasible"]]
    return result


def recognition_observation(public, permutation):
    if sorted(permutation) != list(range(16)):
        raise ValueError("sixteen-cell bijection required")
    result = copy.deepcopy(public)
    result["world"]["permutation"] = [permutation[cell] for cell in result["world"]["permutation"]]
    for observed in [*result["anonymous"], *(item for history in result["references"] for item in history)]:
        observed["artifact"] = board(observed["artifact"], permutation)
        if observed["first_action"] is not None:
            observed["first_action"] = action(observed["first_action"], permutation)
    return result


def inquiry_belief(belief, permutation):
    if sorted(permutation) != list(range(4)) or len(belief) != 25:
        raise ValueError("invalid command-map state relabeling")
    result = [0.0]*25
    result[24] = belief[24]  # genuinely random outputs stay random
    for weight, mapping in zip(belief[:24], MAPPINGS):
        result[MAPPINGS.index(tuple(permutation[cell] for cell in mapping))] = weight
    return result


def inquiry_observation(public, permutation):
    result = copy.deepcopy(public)
    if "beliefs" in result:
        result["beliefs"] = [inquiry_belief(belief, permutation) for belief in result["beliefs"]]
    if "belief" in result:
        result["belief"] = inquiry_belief(result["belief"], permutation)
    for example in result.get("examples", []):
        example["cell"] = permutation[example["cell"]]
    if "offers" in result:
        result["offers"] = [permutation[cell] for cell in result["offers"]]
    if "offer_probabilities" in result:
        result["offer_probabilities"] = [pushed_probabilities(values, permutation)
                                         for values in result["offer_probabilities"]]
    if "artifact" in result:
        result["artifact"] = board(result["artifact"], permutation)
    return result


def close(left, right, tolerance=TOLERANCE):
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(close(left[key], right[key], tolerance) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(close(a, b, tolerance) for a, b in zip(left, right))
    if type(left) in (float, int) and type(right) in (float, int):
        return abs(left-right) <= tolerance
    return left == right


def command_pair_value(belief, commands, goal):
    # Independent scalar value under the full represented map/noise distribution.
    value = 0.0
    for probability, mapping in zip(belief[:24], MAPPINGS):
        artifact = 0
        for command in commands:
            artifact |= 1 << mapping[command]
        value += probability*(artifact == goal)
    successes = sum(((1 << a) | (1 << b)) == goal for a in range(4) for b in range(4))
    return value + belief[24]*successes/16


def inquiry_result_equivalent(kind, original, transformed, before, after, permutation):
    if kind == "inquiry-learning":
        expected = copy.deepcopy(original)
        expected["beliefs"] = [inquiry_belief(belief, permutation) for belief in original["beliefs"]]
        return close(expected, transformed)
    if kind == "inquiry-reading":
        expected = copy.deepcopy(original)
        expected["future_probabilities"] = pushed_probabilities(original["future_probabilities"], permutation)
        return close(expected, transformed)
    if kind == "inquiry":
        # Domain labels and commands keep their meanings. Output-cell names change.
        # A floating-point tie can select another command with the same task value;
        # preserve the score/cost and declared target, not an invented unique string.
        left, right = dict(original), dict(transformed)
        old_command, new_command = left.pop("commands"), right.pop("commands")
        if not close(left, right):
            return False
        domain = original["domain"]
        if domain is None:
            return old_command is None and new_command is None
        def target_probability(public, command):
            goal = public["offers"][domain]
            belief = public["beliefs"][domain]
            return sum(probability*(mapping[command] == goal) for probability, mapping in zip(belief[:24], MAPPINGS)) + belief[24]/4
        return close(target_probability(before, old_command), target_probability(after, new_command))
    if kind != "inquiry-construction":
        raise ValueError("undeclared inquiry output")
    if original["logical_model_evaluations"] != transformed["logical_model_evaluations"]:
        return False
    actual = {(item["domain"], item["goal"]): item for item in transformed["programs"]}
    for program in original["programs"]:
        domain, goal = program["domain"], program["goal"]
        recoded_goal = board(goal, permutation)
        candidate = actual.get((domain, recoded_goal))
        if candidate is None:
            return False
        old_value = command_pair_value(before["beliefs"][domain], program["commands"], goal)
        new_value = command_pair_value(after["beliefs"][domain], candidate["commands"], recoded_goal)
        # Numerical ties among equally good executable programs do not establish
        # different competence or a unique historical string.
        if not close(old_value, new_value):
            return False
    return True
