"""U2: nested evidence prefixes with fixed change times and common random draws.

Known-answer live check: changing skill changes the planted state at the switch.
Placebo: changing a future switch cannot change observations or forecasts before it.
Neither an adaptive advantage nor a horizon reversal is a validity condition.
"""
import math
from . import adaptation as A
from ..v18_3 import world as W

CONDITIONS = ('stationary', 'goal', 'skill', 'rule')
DIMENSIONS = ('family', 'cell', 'condition', 'length', 'copy_span', 'change_at')
METRICS = ('expected_loss', 'pre_change_loss', 'post_change_loss',
           'expected_match', 'abstention_loss', 'prefix32_loss', 'prefix64_loss',
           'prefix96_loss', 'age0_16_loss', 'age16_32_loss', 'age48_64_loss',
           'prefix32_match', 'prefix64_match', 'prefix96_match')


def stream(index, cell, condition, length, copy_span, change_at):
    if condition not in CONDITIONS or length < 1 or not 0 < change_at < length:
        raise ValueError('invalid matched-prefix design')
    if cell not in range(0, 16, 2) or copy_span not in (1, 3, 6):
        raise ValueError('use active non-sharing cells and declared copy spans')
    w = W.make_world(cell, 18542000 + index)
    # Seed excludes horizon, condition, copy span and switch time. Every step
    # consumes one evidence draw and one target draw even when evidence is copied.
    r = W.rng('v18.4-matched-prefix', index, cell)
    initial = list(W.STATES[int(r.integers(len(W.STATES)))])
    history, states, worlds, queries, targets = [], [], [], [], []
    for t in range(length):
        state, wt = list(initial), dict(w)
        if t >= change_at:
            if condition == 'goal': state[1] ^= 1
            if condition == 'skill': state[0] = (state[0] + 1) % 3
            if condition == 'rule': wt['rule'] = 'lexicographic'
        s = W.STATES.index(tuple(state))
        obs = W.observe(wt, s, W.QUERIES[t % 5], r, f'root-{t}')
        copied = t > 0 and t % copy_span != 0
        history.append(dict(history[-1]) if copied else obs)
        query = W.FUTURES[t % len(W.FUTURES)]
        queries.append(query)
        targets.append(W.observe(wt, s, query, r, f'target-{t}'))
        states.append(s); worlds.append(wt)
    return w, history, dict(states=states, worlds=worlds, change=change_at,
                           queries=queries, targets=targets)


def window_metrics(trace, change_at):
    windows = {'prefix32': (0, 32), 'prefix64': (0, 64), 'prefix96': (0, 96),
               'age0_16': (change_at, change_at + 16),
               'age16_32': (change_at + 16, change_at + 32),
               'age48_64': (change_at + 48, change_at + 64)}
    result = {}
    for name, (start, end) in windows.items():
        if end > len(trace): raise ValueError('incomplete scoring window')
        result[name + '_loss'] = math.fsum(t['expected_loss'] for t in trace[start:end]) / (end-start)
        if name.startswith('prefix'):
            result[name + '_match'] = math.fsum(t['expected_match'] for t in trace[start:end]) / (end-start)
    return result


def unit(index, cell=0, condition='skill', length=96, copy_span=1, change_at=12, split='test'):
    if length != 96 or change_at not in (12, 28):
        raise ValueError('U2 uses the reviewed 96-step, 12/28-switch roster')
    w, history, truth = stream(index, cell, condition, length, copy_span, change_at)
    result = A.evaluate_stream(index, cell, condition, length, copy_span, w, history, truth)
    result.update(family='U2', change_at=change_at,
                  scope='nested prefixes of common fresh streams; fixed absolute switches; descriptive constructed method; miniature — architecture untested')
    for row in result['rows']:
        row.update(window_metrics(row['trace'], change_at))
    result['gates'] = [
        dict(kind='identity', name='fixed_switch_time', passed=truth['change']==change_at),
        dict(kind='live' if condition in ('goal','skill') else 'placebo',
             name='planted_state_switch' if condition in ('goal','skill') else 'no_planted_state_switch',
             passed=(truth['states'][change_at] != truth['states'][0]) == (condition in ('goal','skill'))),
        dict(kind='live' if condition=='rule' else 'placebo', name='planted_rule_switch',
             passed=(truth['worlds'][change_at]['rule'] != w['rule']) == (condition=='rule'))]
    return result


def verify(result):
    A.verify(result)
    if not all(g['passed'] for g in result['gates']):
        raise ValueError('matched-prefix instrument gate failed')
    for row in result['rows']:
        for name, expected in window_metrics(row['trace'], result['change_at']).items():
            if abs(row[name] - expected) > 1e-12:
                raise ValueError('window score differs')
    # Evaluator truth must not drift from the admitted construction.
    w, history, truth = stream(result['index'], result['cell'], result['condition'],
                               result['length'], result['copy_span'], result['change_at'])
    if w != result['world'] or history != result['history'] or truth != result['evaluator']:
        raise ValueError('matched trajectory identity differs')
    return True
