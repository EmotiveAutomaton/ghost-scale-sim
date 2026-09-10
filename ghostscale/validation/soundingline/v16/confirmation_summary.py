"""Summarize only frozen confirmation regimes with unchanged native estimands.

This adapter changes the reporting inventory, never scientific unit code, reader
access, generators, scoring, comparators or practical bars. Confirmation's formal
primary analysis is separate from these descriptive native comparisons.
"""
from copy import deepcopy
from types import FunctionType
from .expansion_adapter import module, design, READING, BEHAVIOR, INQUIRY


def summarize_subset(card, rows, condition_ids):
    if not rows or not condition_ids or len(condition_ids) != len(set(condition_ids)):
        raise ValueError("confirmation summary needs a nonempty fixed condition inventory")
    original = design(card)
    available = {condition["id"] for condition in original["conditions"]}
    if not set(condition_ids) <= available or {row["condition"] for row in rows} != set(condition_ids):
        raise ValueError("confirmation data do not match the frozen original regimes")
    owner = module(card)
    function = owner.summarize
    bindings = dict(function.__globals__)
    selected = deepcopy(original)
    selected["conditions"] = [condition for condition in selected["conditions"] if condition["id"] in condition_ids]
    if card in READING|BEHAVIOR|INQUIRY:
        definitions = dict(bindings["DESIGNS"])
        definitions[card] = selected
        bindings["DESIGNS"] = definitions
    elif card in {"V01", "V02"}:
        raise ValueError("tradeoff summary requires its separately justified confirmation target")
    else:
        bindings["DESIGN"] = selected
    adapted = FunctionType(function.__code__, bindings, function.__name__, function.__defaults__, function.__closure__)
    adapted.__kwdefaults__ = function.__kwdefaults__
    summary = adapted(card, rows) if card in READING|BEHAVIOR|INQUIRY else adapted(rows)
    if set(summary["conditions"]) != set(condition_ids) or summary["n_maker_packets"] != len(rows):
        raise ValueError("native subset reporting changed its declared denominators")
    return summary
