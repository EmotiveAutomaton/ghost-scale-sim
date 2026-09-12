"""Common finite forecasts, explicit evidence and counted computational costs."""
from dataclasses import dataclass, field
import math

TIERS = ("artifact", "artifact_collection", "recorded_process")
COST_KINDS = ("training_acquisition", "learning", "definition_storage", "retrieval",
              "proposal_generation", "argument_binding", "hypothetical_execution",
              "actual_execution", "selection", "feedback_query")
SEARCH_KINDS = ("retrieval", "proposal_generation", "argument_binding",
                "hypothetical_execution", "selection")


class BudgetExhausted(Exception):
    pass


@dataclass
class Costs:
    values: dict = field(default_factory=lambda: dict.fromkeys(COST_KINDS, 0))
    search_budget: int | None = None

    def charge(self, kind, count=1):
        if kind not in COST_KINDS or type(count) is not int or count < 0:
            raise ValueError("invalid cost charge")
        if self.search_budget is not None and kind in SEARCH_KINDS:
            if sum(self.values[k] for k in SEARCH_KINDS) + count > self.search_budget:
                raise BudgetExhausted
        self.values[kind] += count

    def receipt(self, deployments=1):
        if type(deployments) is not int or deployments < 1:
            raise ValueError("deployments must be a positive integer")
        setup = sum(self.values[k] for k in ("training_acquisition", "learning", "definition_storage"))
        online = sum(self.values.values()) - setup
        return {**self.values, "search_total": sum(self.values[k] for k in SEARCH_KINDS),
                "cold_total": setup + online, "repeat_online": online,
                "amortized_total": online + setup / deployments, "deployments": deployments,
                "unit": "declared discrete operations/storage tokens; not CPU seconds"}


def forecast_scores(probabilities, support, truth):
    """Unclipped proper scores. Zero mass on truth is explicitly infinite log loss."""
    if not isinstance(support, list) or not support or len(set(support)) != len(support):
        raise ValueError("answer support must be nonempty and unique")
    if truth not in support or not isinstance(probabilities, dict) or set(probabilities) != set(support):
        raise ValueError("missing or extra prediction/answer support")
    values = list(probabilities.values())
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 or x > 1 for x in values):
        raise ValueError("invalid probability")
    if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("probabilities do not normalize")
    p = probabilities[truth]
    return {"log_loss_nats": -math.log(p) if p else None,
            "log_loss_infinite": p == 0,
            "brier_score": sum((probabilities[k] - int(k == truth)) ** 2 for k in support)}


def public_evidence(case, tier):
    """Explicit allowlist, never exclusion of a few known private field names."""
    if tier not in TIERS:
        raise ValueError("unknown evidence tier")
    evidence = {"artifact": case["artifact"], "context": case["public_context"]}
    if tier != "artifact":
        evidence["earlier_artifacts"] = case["earlier_artifacts"]
    if tier == "recorded_process":
        evidence["recorded_process"] = case["recorded_process"]
    # JSON boundary also removes references to evaluator-owned mutable objects.
    from ..v16.records import canonical
    import json
    return json.loads(canonical(evidence))
