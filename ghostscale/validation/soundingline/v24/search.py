"""Predeclared finite search primitives; selection never equals confirmation."""
import math


def halving_schedule(cells,initial_worlds=1,eta=3):
    if not cells or len(set(cells))!=len(cells) or type(initial_worlds) is not int or initial_worlds<1 or eta!=3:raise ValueError('unique roster, positive integer worlds and eta=3 required')
    result=[];n=len(cells);worlds=initial_worlds
    while True:
        result.append(dict(cells=n,cumulative_worlds_per_cell=worlds))
        if n==1:return result
        n=math.ceil(n/eta);worlds*=eta


def survivors(scores,roster):
    if not roster or len(set(roster))!=len(roster) or set(scores)!=set(roster):raise ValueError('complete fixed roster required')
    if not all(math.isfinite(scores[k]) for k in roster):raise ValueError('unavailable scores cannot silently disappear')
    rank={name:i for i,name in enumerate(roster)}
    return sorted(roster,key=lambda k:(-scores[k],rank[k]))[:math.ceil(len(roster)/3)]


def implausibility(mean,target,emulator_variance,discrepancy_variance,observation_variance):
    values=(mean,target,emulator_variance,discrepancy_variance,observation_variance)
    if not all(math.isfinite(x) for x in values) or min(values[2:])<0:raise ValueError('finite nonnegative variances required')
    variance=math.fsum(values[2:])
    if variance==0:return 0. if mean==target else math.inf
    return abs(mean-target)/math.sqrt(variance)


def history_match(*args):
    value=implausibility(*args)
    return dict(implausibility=value,retain=value<3,threshold=3,squared_threshold=9)


def seed_extension(planned,already_used,promising):
    if not planned or len(set(planned))!=len(planned) or any(type(x) is not int or x<0 for x in planned):raise ValueError('distinct planned nonnegative seeds required')
    if len(set(already_used))!=len(already_used) or any(type(x) is not int for x in already_used):raise ValueError('distinct observed seeds required')
    if type(promising) is not bool:raise ValueError('prospective rule must return a boolean')
    if not set(already_used)<=set(planned):raise ValueError('only the frozen initial look may trigger one extension')
    if set(already_used)!=set(planned):raise ValueError('complete initial look required')
    if not promising:return []
    return list(range(max(planned)+1,max(planned)+1+len(planned)))


def successive_halving(roster, evaluate, *, initial_worlds=1, eta=3):
    """Execute the frozen N1 allocation without shedding unavailable cells.

    ``evaluate`` receives the complete surviving roster, the new disjoint world
    indices for that round, and the round number.  It must return one finite
    score for every requested cell.  All scores and eliminations are retained;
    selection remains a design result and never a confirmation verdict.
    """
    if not callable(evaluate):
        raise ValueError("an explicit N1 evaluator is required")
    schedule = halving_schedule(roster, initial_worlds, eta)
    active = list(roster)
    previous = 0
    rounds = []
    for number, allocation in enumerate(schedule):
        cumulative = allocation["cumulative_worlds_per_cell"]
        worlds = list(range(previous, cumulative))
        scores = evaluate(tuple(active), tuple(worlds), number)
        if not isinstance(scores, dict):
            raise ValueError("N1 evaluator must return a complete score mapping")
        # survivors performs the complete-roster, finite-value and tie checks.
        ranked = sorted(active, key=lambda name: (-scores[name], active.index(name))) \
            if set(scores) == set(active) and all(math.isfinite(scores[x]) for x in active) else None
        if ranked is None:
            survivors(scores, active)  # raise the precise contract violation
        keep = active if len(active) == 1 else survivors(scores, active)
        rounds.append(dict(
            round=number,
            cells=list(active),
            new_world_indices=worlds,
            cumulative_worlds_per_cell=cumulative,
            scores={name: float(scores[name]) for name in active},
            ranking=ranked,
            survivors=list(keep),
            eliminated=[name for name in active if name not in set(keep)],
        ))
        active = list(keep)
        previous = cumulative
    return dict(
        state="SEARCH_COMPLETE_NOT_CONFIRMATION",
        objective_frozen=True,
        initial_roster=list(roster),
        schedule=schedule,
        rounds=rounds,
        selected=active[0],
        total_worlds=sum(len(row["cells"]) * len(row["new_world_indices"])
                         for row in rounds),
        confirmation=False,
    )
