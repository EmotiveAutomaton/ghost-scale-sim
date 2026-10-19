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
