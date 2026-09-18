"""Explicit physical model and bounded counted search, with depth-aware dominance.

The planner follows V17's shared state-search/representation pattern. It fixes
remaining-depth dominance here without changing the closed V17 implementation.
"""
from collections import Counter, deque
from dataclasses import dataclass, field
import heapq
from itertools import product

from ..v16 import world as w1, graphic_world

ONLINE = ('checking', 'hypothetical_execution', 'proposal_generation', 'retrieval',
          'selection', 'ordering', 'actual_execution')


class Exhausted(Exception):
    pass


@dataclass
class Work:
    cap: int
    counts: dict = field(default_factory=lambda: dict.fromkeys(ONLINE, 0))

    @property
    def spent(self):
        return sum(self.counts.values())

    def charge(self, kind, amount=1):
        if kind not in ONLINE or type(amount) is not int or amount < 0:
            raise ValueError('invalid work charge')
        if self.spent + amount > self.cap:
            raise Exhausted
        self.counts[kind] += amount

    def receipt(self):
        return dict(self.counts, total_online=self.spent, envelope=self.cap,
                    unit='declared operations; not CPU seconds')


def validate_world(world):
    if world['kind'] == 'graphic':
        if world['cells'] not in (4, 16):
            raise ValueError('unsupported graphic family')
    elif world['kind'] == 'assembly':
        parents, defaults = world['parents'], world['defaults']
        if len(parents) not in (3, 5, 7) or len(parents) != len(defaults):
            raise ValueError('invalid assembly size')
        n=len(parents)
        if any(type(p) is not int or p not in range(-1,n) or p==i for i,p in enumerate(parents)):
            raise ValueError('invalid dependency label')
        for start in range(n):
            seen=set();part=start
            while part>=0:
                if part in seen:raise ValueError('cyclic dependency graph')
                seen.add(part);part=parents[part]
        if any(type(v) is not int or v not in (0, 1) for v in defaults):
            raise ValueError('invalid defaults')
    else:
        raise ValueError('unknown world')


def actions(world):
    return list(range(2 * world['cells'] if world['kind'] == 'graphic' else 3 * len(world['parents']) + 1))


def identity(state):
    return tuple(state) if isinstance(state, (tuple, list)) else state


def distance(state, target):
    return ((state ^ target).bit_count() if type(state) is int
            else sum(a != b for a, b in zip(state, target)))


def step(world, state, stopped, action):
    if stopped or type(action) is not int or action not in actions(world) or action in world.get('forbidden', []):
        return identity(state), stopped, False
    if world['kind'] == 'graphic':
        n = world['cells']
        if n == 4:
            after = w1.step(state, action)
        else:
            after = graphic_world.execute([action], initial=state, max_steps=1)['artifact']
        return after, False, True
    n = len(world['parents'])
    if action == 3*n:
        return tuple(state), True, True
    part = action % n
    attached_child = any(parent == part and state[child] >= 0 for child, parent in enumerate(world['parents']))
    after = list(state)
    if action < n:
        parent = world['parents'][part]
        if state[part] >= 0 or (parent >= 0 and state[parent] < 0):
            return tuple(state), False, False
        after[part] = world['defaults'][part]
    elif action < 2*n:
        if state[part] < 0 or attached_child:
            return tuple(state), False, False
        after[part] = -1
    else:
        if state[part] < 0 or attached_child:
            return tuple(state), False, False
        after[part] = 1-state[part]
    return tuple(after), False, True


def execute(world, initial, program, max_steps):
    validate_world(world)
    if type(max_steps) is not int or max_steps<0:
        raise ValueError('invalid execution limit')
    if world['kind']=='graphic':
        if type(initial) is not int or not 0<=initial<2**world['cells']:
            raise ValueError('invalid initial graphic state')
    elif (len(initial)!=len(world['parents']) or any(type(v) is not int or v not in (-1,0,1) for v in initial)
          or any(initial[i]>=0 and p>=0 and initial[p]<0 for i,p in enumerate(world['parents']))):
        raise ValueError('invalid initial assembly state')
    state, stopped, trace = identity(initial), False, []
    for index, action in enumerate(program):
        if index >= max_steps:
            return dict(state=state, stopped=stopped, legal=False, primitive_cost=index, trace=trace, error='length')
        after, stop, legal = step(world, state, stopped, action)
        trace.append(dict(before=state, action=action, after=after, legal=legal, stopped=stop))
        if not legal:
            return dict(state=state, stopped=stopped, legal=False, primitive_cost=len(trace), trace=trace, error='illegal')
        state, stopped = after, stop
    return dict(state=state, stopped=stopped or world['kind'] == 'graphic', legal=True,
                primitive_cost=len(trace), trace=trace, error=None)


def exhaustive(world, initial, target, max_steps, *, monotone=False):
    """Evaluator-only exact reachability. Never passed to a reader implicitly."""
    queue = deque([(identity(initial), [], False)])
    seen = {(identity(initial), False)}
    evaluated = 0
    while queue:
        state, program, stopped = queue.popleft()
        if state == identity(target) and (stopped or world['kind'] == 'graphic'):
            return dict(reachable=True, program=program, evaluations=evaluated, states=len(seen))
        if stopped or len(program) >= max_steps:
            continue
        for action in actions(world):
            after, stop, legal = step(world, state, stopped, action)
            evaluated += 1
            if not legal or (monotone and distance(after, target) > distance(state, target)):
                continue
            key = (after, stop)
            if key not in seen:
                seen.add(key)
                queue.append((after, program + [action], stop))
    return dict(reachable=False, program=None, evaluations=evaluated, states=len(seen))


def learn_fragments(training, *, capacity=4, threshold=2, storage_cap=64):
    counts = Counter()
    for trial in training:
        if trial['feedback'] is not True:
            continue
        program = trial['program']
        for i in range(len(program)-1):
            counts[tuple(program[i:i+2])] += 1
    candidates = sorted((p for p, count in counts.items() if count >= threshold), key=lambda p: (-counts[p], p))
    kept, storage = [], 0
    for fragment in candidates[:capacity]:
        cost = 2*len(fragment)+1
        if storage+cost <= storage_cap:
            kept.append(list(fragment)); storage += cost
    return dict(fragments=kept, episodes=[], storage=storage, scanned=sum(len(t['program']) for t in training))


def learn_episodes(training, *, storage_cap=64, exceptions=()):
    seen = set(map(tuple, exceptions))
    kept, storage = [], 0
    for trial in sorted(training, key=lambda t: (len(t['program']), t['program'])):
        program = tuple(trial['program'])
        # Retain failed episodes as observations, but do not propose them as successful routines.
        if trial['feedback'] is not True or program in seen:
            continue
        seen.add(program)
        cost = 2*len(program)+3
        if storage+cost <= storage_cap:
            kept.append(list(program)); storage += cost
    return dict(fragments=[], episodes=kept, storage=storage, scanned=sum(len(t['program']) for t in training))


def solve(world, initial, target, representation, cap, max_steps, action_order, *, ordering='fixed', gate=None):
    """Known/supplied-model planner; evaluator truth is not an argument.

    All online work, including final execution reservation, fits cap. A shallower
    arrival may reopen a state. Actual execution is performed separately under truth.
    """
    work = Work(cap)
    initial = identity(initial)
    target = identity(target)
    queue = [(distance(initial, target), 0, initial, [], False)]
    best_depth = {(initial, False): 0}
    serial = 0
    selected = None
    rejection = accepted = 0
    try:
        while queue:
            work.charge('selection')
            _, _, state, program, stopped = heapq.heappop(queue)
            if len(program) > best_depth.get((state, stopped), max_steps+1):
                continue
            if state == target and (stopped or world['kind'] == 'graphic'):
                if work.spent + len(program) <= cap:
                    selected = program
                break
            if stopped:
                continue
            candidates = []
            for fragment in representation.get('fragments', []) + representation.get('episodes', []):
                work.charge('retrieval', len(fragment))
                work.charge('proposal_generation')
                candidates.append(list(fragment))
            for action in action_order:
                work.charge('proposal_generation')
                candidates.append([action])
            if ordering == 'goal':
                ranked = []
                for ordinal, fragment in enumerate(candidates):
                    work.charge('ordering')
                    # Goal relevance is computed from public state and action semantics, paid per candidate.
                    action = fragment[0]
                    if world['kind'] == 'graphic':
                        n = world['cells']; bit = 1 << (action % n)
                        useful = bool(target & bit) == (action < n) and bool(state & bit) != bool(target & bit)
                    else:
                        n = len(world['parents']); part = action % n
                        useful = action == 3*n if state == target else (
                            action < n and state[part] < 0 and target[part] >= 0 or
                            n <= action < 2*n and state[part] >= 0 and target[part] < 0 or
                            2*n <= action < 3*n and state[part] >= 0 and target[part] >= 0 and state[part] != target[part])
                    ranked.append((not useful, ordinal, fragment))
                candidates = [x[-1] for x in sorted(ranked)]
            for fragment in candidates:
                work.charge('selection')
                if not fragment or len(program)+len(fragment) > max_steps:
                    continue
                if gate is not None:
                    admitted = gate(world, state, target, fragment, max_steps-len(program), work)
                    rejection += not admitted; accepted += bool(admitted)
                    if not admitted:
                        continue
                current, stop, legal = state, False, True
                for action in fragment:
                    work.charge('hypothetical_execution')
                    current, stop, legal = step(world, current, stop, action)
                    if not legal:
                        break
                if not legal:
                    continue
                proposal = program + fragment
                key = (current, stop)
                # State alone is insufficient with macro edges and a finite action-depth budget.
                if len(proposal) < best_depth.get(key, max_steps+1):
                    best_depth[key] = len(proposal)
                    serial += 1
                    heapq.heappush(queue, (distance(current, target), serial, current, proposal, stop))
    except Exhausted:
        pass
    # Reservation is explicit and charged at submission. The evaluator refunds only
    # unexecuted suffixes on illegal true-law execution, preserving attempted cost.
    if selected is not None:
        work.charge('actual_execution', len(selected))
    return dict(program=selected, costs=work.receipt(), states=len(best_depth),
                rejected_candidates=rejection, accepted_candidates=accepted)


def score_submission(true_world, initial, target, result, max_steps):
    program = result['program']
    actual = None if program is None else execute(true_world, initial, program, max_steps)
    costs = dict(result['costs'])
    if actual is not None:
        refund = len(program)-actual['primitive_cost']
        costs['actual_execution'] -= refund
        costs['total_online'] -= refund
    return dict(result, costs=costs, execution=actual,
                success=bool(actual and actual['legal'] and actual['stopped'] and identity(actual['state']) == identity(target)),
                missing_output=actual is None, invalid=bool(actual and not actual['legal']))
