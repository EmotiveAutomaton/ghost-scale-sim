"""One prospective seed expansion with a durable, immutable decision receipt.

The trigger is only a pilot-design convenience. Optional expansion does not
license fixed-sample p-values or confidence intervals on the pooled sample.
"""
import math
import os
from pathlib import Path

from ..v23.common import canonical, digest, read, write
from .search import seed_extension


def freeze(root, seeds, *, threshold, protocol_sha256):
    root=Path(root)
    # Validate the complete initial roster without observing outcomes.
    seed_extension(seeds,seeds,False)
    if not math.isfinite(threshold):
        raise ValueError("finite prospective threshold required")
    if (len(protocol_sha256)!=64 or any(c not in "0123456789abcdef" for c in protocol_sha256)):
        raise ValueError("frozen protocol hash required")
    root.mkdir(parents=True,exist_ok=False)
    plan=dict(version="v24-seed-extension-1",initial_seeds=list(seeds),
              trigger=dict(kind="complete_initial_mean_at_least",threshold=threshold),
              protocol_sha256=protocol_sha256,maximum_total_seeds=2*len(seeds),
              inference="descriptive pilot only; powered confirmation needs a separate frozen design")
    write(root/"PLAN.json",plan)
    write(root/"PIN.json",dict(plan_sha256=digest(root/"PLAN.json")))
    return plan


def decide(root, observations, *, source_sha256, protocol_sha256):
    root=Path(root)
    plan=read(root/"PLAN.json")
    if digest(root/"PLAN.json")!=read(root/"PIN.json")["plan_sha256"]:
        raise ValueError("prospective seed plan changed")
    if plan["protocol_sha256"]!=protocol_sha256:
        raise ValueError("protocol changed after freezing")
    if (len(source_sha256)!=64 or any(c not in "0123456789abcdef" for c in source_sha256)):
        raise ValueError("complete initial evidence hash required")
    if set(observations)!=set(plan["initial_seeds"]) or any(type(s) is not int for s in observations):
        raise ValueError("complete initial look required")
    if not all(math.isfinite(x) for x in observations.values()):
        raise ValueError("unavailable values cannot be dropped")
    mean=math.fsum(observations[s] for s in plan["initial_seeds"])/len(observations)
    extension=seed_extension(plan["initial_seeds"],list(observations),
                             mean>=plan["trigger"]["threshold"])
    receipt=dict(plan_sha256=digest(root/"PLAN.json"),source_sha256=source_sha256,
                 protocol_sha256=protocol_sha256,observations=[[s,observations[s]] for s in plan["initial_seeds"]],
                 mean=mean,extension=extension,consumed=True,confirmation=False,
                 fixed_sample_inference_on_pooled_outcomes_authorized=False)
    # Exclusive write consumes both positive and negative initial looks. An
    # interruption after creation leaves a visible hold, never another look.
    path=root/"DECISION.json"
    with path.open("xb") as stream:
        stream.write(canonical(receipt)+b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return receipt
