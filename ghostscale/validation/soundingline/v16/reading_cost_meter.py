"""Observe actual call invocations without editing the frozen reading function.

The new function object shares bytecode, defaults and immutable model objects with
the original, but owns a copied globals dictionary. Only three call bindings are
wrapped. Cached calls count as invocations, not as fresh model evaluations or CPU
instructions. No scientific result or original module global is changed.
"""
from collections import Counter
from types import FunctionType
from . import reconstruction


def public_reader(payload, strategy="maker"):
    original = reconstruction.reader
    calls = Counter()
    bindings = dict(original.__globals__)
    for name in ("observation_mass", "library_prior", "construct"):
        function = bindings[name]
        def counted(*args, _name=name, _function=function, **kwargs):
            calls[_name] += 1
            return _function(*args, **kwargs)
        bindings[name] = counted
    copied = FunctionType(original.__code__, bindings, original.__name__, original.__defaults__, original.__closure__)
    result = copied(payload, strategy)
    return {"result": result, "invocations": dict(calls),
            "scope": "actual call invocations, including cache hits; not CPU instructions or cache misses"}
