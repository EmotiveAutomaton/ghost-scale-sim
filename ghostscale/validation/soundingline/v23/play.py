"""Optional N8: public play signals, threat transitions and concealed preference."""
import numpy as np
from .common import rng, normalize


def sample(v, phase, world, case, law=None):
    s = v['settings']; r = rng(v['id'], phase, world, case)
    law = world % 16 if law is None else law
    preference = int(r.integers(2)); precision = np.linspace(.6, .95, 16)[law]
    history, modes = [], []
    for _ in range(s['records']):
        threat = bool(r.random() < s['threat_rate'])
        playing = bool(not threat and r.random() < s['play_share']); modes.append(playing)
        signal = int(playing ^ (r.random() > s['signal_reliability']))
        target = preference ^ playing
        choice = int(target ^ (r.random() > precision))
        history.append(dict(threat=threat, signal=signal, choice=choice))
    return dict(history=history), dict(y=[preference], modes=modes, law=int(law))


def evaluate(v, p, truth=None):
    s = v['settings']; precision = np.linspace(.6, .95, 16)
    exact, blind = np.zeros((16, 2)), np.zeros((16, 2))
    for row in p['history']:
        playing_prior = 0 if row['threat'] else s['play_share']
        for preference in (0, 1):
            mixture = np.zeros(16)
            for playing in (0, 1):
                mode_prior = playing_prior if playing else 1-playing_prior
                signal_prob = s['signal_reliability'] if row['signal'] == playing else 1-s['signal_reliability']
                choice = precision if row['choice'] == preference ^ playing else 1-precision
                mixture += mode_prior*signal_prob*choice
            exact[:, preference] += np.log(mixture)
            blind[:, preference] += np.log(precision if row['choice'] == preference else 1-precision)
    def marginal(ll):
        return normalize(ll.ravel()).reshape(16, 2).sum(0)[None, :]
    readers = {'law-marginal': marginal(exact), 'mode-blind': marginal(blind),
               'always-closed': np.array([[.5, .5]])}
    if truth is not None:
        readers['oracle-law'] = normalize(exact[truth['law']])[None, :]
    return readers, dict(modes_inferred_without_sidecar=True)


def features(v, p):
    x = np.zeros((32, 4))
    for i, row in enumerate(p['history']):
        x[i] = [row['threat'], row['signal'], row['choice'], 1]
    return x.ravel()
