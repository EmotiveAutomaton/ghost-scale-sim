"""Independent scalar specification of the amended inquiry reader.

No NumPy or primary inquiry imports. The explicit numerical tie resolution is
duplicated from the repair contract, while all probabilities use scalar sums.
"""
from functools import lru_cache
from itertools import permutations, combinations
from math import fsum, log

MAPS = tuple(permutations(range(4)))
PAIRS = tuple(combinations(range(4), 2))
GOALS = tuple((1 << a) | (1 << b) for a, b in PAIRS)
PRIOR = tuple([0.9/24]*24+[0.1])
TIE = 2.0**-46


def normalized(weights):
    total = fsum(weights)
    if total <= 0:
        raise ValueError("independent model has zero support")
    return tuple(value/total for value in weights)


def posterior(belief, command, cell):
    return normalized([p*(mapping[command] == cell) for p, mapping in zip(belief[:24], MAPS)]
                      + [belief[24]/4])


def outcome_probabilities(belief, command):
    return [fsum(p for p, mapping in zip(belief[:24], MAPS) if mapping[command] == cell)+belief[24]/4
            for cell in range(4)]


def first_best(values):
    maximum = max(values)
    return min(index for index, value in enumerate(values) if maximum-value <= TIE)


def command(belief, target):
    return first_best([outcome_probabilities(belief, candidate)[target] for candidate in range(4)])


@lru_cache(maxsize=32768)
def program_values(belief):
    rows = []
    for goal in GOALS:
        values = []
        for a, b in PAIRS:
            mass = fsum(p for p, mapping in zip(belief[:24], MAPS)
                        if ((1 << mapping[a]) | (1 << mapping[b])) == goal)
            values.append(mass + belief[24]/8)
        rows.append(tuple(values))
    return tuple(rows)


def competence(belief):
    return fsum(max(values) for values in program_values(tuple(belief)))/6


def uncertainty(belief):
    return fsum(-value*log(value) for value in belief if value > 0)


@lru_cache(maxsize=65536)
def after_queries(belief, commands):
    if not commands:
        return competence(belief), uncertainty(belief), 925
    outcomes = outcome_probabilities(belief, commands[0])
    skills, entropies, operations = [], [], 100
    for cell, weight in enumerate(outcomes):
        if weight == 0:
            continue
        skill, entropy, cost = after_queries(posterior(belief, commands[0], cell), commands[1:])
        skills.append(weight*skill)
        entropies.append(weight*entropy)
        operations += 25+cost
    return fsum(skills), fsum(entropies), operations


def learned(public):
    beliefs = [tuple(belief) for belief in public["beliefs"]]
    for domain in public["forget_domains"]:
        beliefs[domain] = PRIOR
    for example in public["examples"]:
        domain = example["domain"]
        beliefs[domain] = posterior(beliefs[domain], example["command"], example["cell"])
    return {"beliefs": [list(belief) for belief in beliefs],
            "logical_model_evaluations": 25*len(public["examples"])}


def constructed(public):
    programs = []
    for domain, belief in enumerate(public["beliefs"]):
        for goal, values in zip(GOALS, program_values(tuple(belief))):
            programs.append({"domain": domain, "goal": goal, "commands": list(PAIRS[first_best(values)])})
    return {"programs": programs, "logical_model_evaluations": 1800}


def reading(public):
    belief, artifact = public["belief"], public["artifact"]
    outcomes = [0.0]*4
    for cell in range(4):
        weights = []
        for probability, mapping in zip(belief[:24], MAPS):
            for a, b in ((0, 1), (2, 3)):
                if mapping[a] == cell and ((1 << mapping[a]) | (1 << mapping[b])) == artifact:
                    weights.append(probability/2)
        occupied = artifact.bit_count()
        noise = (1/16 if occupied == 1 else 2/16 if occupied == 2 else 0)*belief[24]/4
        outcomes[cell] = fsum(weights)+noise
    return {"future_probabilities": list(normalized(outcomes)), "logical_model_evaluations": 50}


def progress(history):
    values = [float(item["success"]) for item in history]
    if not values:
        return 0.25
    recent = fsum(values[-2:])/len(values[-2:])
    previous = fsum(values[-4:-2])/len(values[-4:-2]) if len(values) > 2 else 0.25
    return recent-previous


def chosen(public, policy):
    if policy == "decline":
        return {"domain": None, "commands": None, "scores": [0.0, 0.0],
                "logical_model_evaluations": 0, "commit_next": False}
    if public["committed_domain"] is not None:
        domain = public["committed_domain"]
        return {"domain": domain, "commands": command(public["beliefs"][domain], public["offers"][domain]),
                "scores": [0.0, 0.0], "logical_model_evaluations": 100, "commit_next": False}
    scores, commands, work = [], [], 200
    for domain in range(2):
        belief = tuple(public["beliefs"][domain])
        selected = command(belief, public["offers"][domain])
        commands.append(selected)
        batch = tuple(public["pending_commands"][domain])+(selected,)
        released = len(batch) >= public["feedback_batches"][domain]
        uses_model = policy in ("eig", "recognition", "value-learning")
        if uses_model:
            base_skill, base_entropy = competence(belief), uncertainty(belief)
            work += 925
        else:
            base_skill, base_entropy = 0.0, 0.0
        if uses_model and released:
            skill, entropy, extra = after_queries(belief, batch)
            work += extra
        else:
            skill, entropy = base_skill, base_entropy
        gain, information = skill-base_skill, base_entropy-entropy
        if policy == "surprise":
            history = public["histories"][domain]
            value = history[-1]["surprise"] if history else log(4)
        elif policy in ("signed-progress", "absolute-progress"):
            value = progress(public["histories"][domain])
            if policy == "absolute-progress":
                value = abs(value)
        elif policy in ("eig", "recognition"):
            value = max(0.0, information)
            if policy == "recognition":
                value *= (1+public["familiarity"][domain])/5
        elif policy == "uniform":
            value = 1.0
        elif policy == "value-learning":
            if not released and public["remaining_interactions"] >= 2:
                weighted = []
                for future_goal, probability in enumerate(public["offer_probabilities"][domain]):
                    second = command(belief, future_goal)
                    later, _, extra = after_queries(belief, batch+(second,))
                    work += 100+extra
                    weighted.append(probability*(later-base_skill)/2)
                gain = fsum(weighted)
            value = public["future_weights"][domain]*max(0.0, gain)-public["opportunity_cost"]
        else:
            raise ValueError("unknown independent inquiry policy")
        scores.append(value)
    if policy == "value-learning" and max(scores) <= TIE:
        domain = None
    else:
        best = max(scores)
        tied = [index for index, value in enumerate(scores) if abs(value-best) < 1e-12]
        domain = tied[min(int(public["tie_draw"]*len(tied)), len(tied)-1)]
    return {"domain": domain, "commands": commands[domain] if domain is not None else None,
            "scores": scores, "logical_model_evaluations": work,
            "commit_next": domain is not None and policy == "value-learning"
                and len(public["pending_commands"][domain])+1 < public["feedback_batches"][domain]
                and public["remaining_interactions"] >= 2}


def readout(kind, public, options):
    operation = kind.split(":")[-1]
    if operation == "choose":
        return chosen(public, options["policy"])
    if options:
        raise ValueError("undeclared independent reader options")
    return {"learn_public": learned, "construct_public": constructed, "read_maker_public": reading}[operation](public)
