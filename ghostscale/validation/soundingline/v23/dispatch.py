"""Explicit dispatch; population worlds have their own contract."""
from . import repairs, diagnosticity, corrections, values, projection, uptake, statements, play

MODULES = {'R1': repairs, 'R2': repairs, 'R3': repairs, 'R4': repairs,
           'N2': diagnosticity, 'N3': corrections, 'N4': values, 'N5': projection,
           'N6': uptake, 'N7': statements, 'N8': play}


def sample(v, *args, **kwargs): return MODULES[v['family']].sample(v, *args, **kwargs)
def evaluate(v, *args, **kwargs): return MODULES[v['family']].evaluate(v, *args, **kwargs)
def features(v, *args, **kwargs): return MODULES[v['family']].features(v, *args, **kwargs)
