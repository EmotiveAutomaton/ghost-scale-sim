"""Typed paired outcomes and honest constructor/maker uncertainty."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import random


@dataclass(frozen=True)
class Estimand:
    id: str
    target: str
    arm: str
    rival: str
    units: str
    practical_bar: float
    numerator: str
    denominator: str = "all independent maker task packets, including failures"
    aggregation: str = "equal maker mean of paired task-packet differences"
    interval_target: str = "same mean; constructors then makers within constructor"

    def __post_init__(self):
        if self.arm == self.rival:
            raise ValueError("same-arm subtraction cannot identify an effect")
        if self.units not in {"success_fraction", "nats_per_event", "primitive_evaluations",
                              "net_repair_fraction", "utility_per_packet"}:
            raise ValueError("unknown estimand units")
        if not all([self.id, self.target, self.numerator, self.interval_target]):
            raise ValueError("incomplete estimand")

    def record(self):
        return asdict(self)


def paired_summary(rows, estimand: Estimand, *, seed=771, replicates=1999):
    if not rows:
        raise ValueError("empty evidence is not an effect")
    groups = {}
    seen = set()
    for row in rows:
        if row["unit_id"] in seen:
            raise ValueError("duplicate maker history")
        seen.add(row["unit_id"])
        if estimand.arm not in row["arms"] or estimand.rival not in row["arms"]:
            raise ValueError("missing paired arm")
        difference = row["arms"][estimand.arm]["outcomes"][estimand.target] - row["arms"][estimand.rival]["outcomes"][estimand.target]
        if not math.isfinite(difference):
            raise ValueError("nonfinite outcome requires a separate mismatch receipt")
        groups.setdefault(row["constructor_id"], []).append(difference)
    values = [value for group in groups.values() for value in group]
    mean = sum(values) / len(values)
    rng = random.Random(seed)
    group_values = list(groups.values())
    draws = []
    for _ in range(replicates):
        resampled = []
        for _ in group_values:
            group = rng.choice(group_values)
            resampled.extend(rng.choice(group) for _ in group)
        draws.append(sum(resampled) / len(resampled))
    draws.sort()
    low = draws[int(0.025 * (replicates - 1))]
    high = draws[int(0.975 * (replicates - 1))]
    standard_deviation = math.sqrt(sum((x - mean)**2 for x in values) / max(1, len(values)-1))
    return {"estimand": estimand.record(), "n_makers": len(values),
            "n_constructors": len(groups), "mean": mean, "interval_95": [low, high],
            "bootstrap": {"seed": seed, "replicates": replicates,
                          "method": "constructors then makers, empirical percentile",
                          "quantile": "floor(p*(B-1))"},
            "paired_standard_deviation": standard_deviation,
            "criterion_state": "held" if low >= estimand.practical_bar else
                               ("failed" if high < estimand.practical_bar else "inconclusive"),
            "evidence_scope": rows[0]["evidence_scope"]}
