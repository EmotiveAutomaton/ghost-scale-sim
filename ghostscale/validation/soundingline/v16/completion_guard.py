"""Finite campaign accounting; empty dispatch is not scientific closeout.

The caller supplies the complete frozen manifest and file-bound evidence. These
checks establish accounting, never the truth of scientific criteria. Final B04
must independently execute and bind each of the three closeout proof producers.
"""
from .records import read, file_digest

PROOFS = ("raw_archive", "independent_aggregates", "scientific_replay", "documentary_write_through")
TERMINAL = {"completed", "blocked", "failed"}


def assess(required, cards, expansions, proofs):
    reasons, qualifications = [], []
    required = set(required)
    identities = [row["card_id"] for row in cards]
    if not required:
        reasons.append("empty commission cannot establish completion")
    if len(identities) != len(set(identities)):
        reasons.append("duplicate card identities")
    if set(identities) != required:
        reasons.append("missing or unexpected commissioned cards")
    for row in cards:
        name, state = row["card_id"], row.get("execution_state")
        if state not in TERMINAL:
            reasons.append(name+": execution unresolved")
        if state in {"blocked", "failed"}:
            if not row.get("reason"):
                reasons.append(name+": terminal failure lacks reason")
            qualifications.append(name+": "+state)
        elif state == "completed" and row.get("instrument_state") not in {"valid", "not_applicable"}:
            reasons.append(name+": completion lacks instrument disposition")
        if not row.get("evidence_verified"):
            reasons.append(name+": evidence unverified")
        if row.get("unresolved_dependencies"):
            if state == "completed":
                reasons.append(name+": completed with missing dependency")
            qualifications.append(name+": dependencies unavailable")
        if row.get("scientific_criterion") not in {"supported", "null", "boundary", "not_tested", "not_applicable"}:
            reasons.append(name+": scientific criterion not accounted")
    if not expansions or {row["card_id"] for row in expansions} != required:
        reasons.append("expansion ladder not accounted for every card")
    if len(expansions) != len({row["card_id"] for row in expansions}):
        reasons.append("duplicate expansion dispositions")
    for row in expansions:
        if row.get("state") not in {"exhausted", "ineligible", "budget_closed", "not_applicable"} or not row.get("reason"):
            reasons.append(row["card_id"]+": expansion unresolved")
        if row.get("state") == "budget_closed":
            qualifications.append(row["card_id"]+": bounded allocation closed")
    for name in PROOFS:
        if not proofs.get(name, {}).get("verified"):
            reasons.append(name+": missing independently checked receipt")
    return {"campaign_closed": not reasons, "campaign_clean": not reasons and not qualifications,
            "blocking_reasons": reasons, "qualifications": sorted(set(qualifications)),
            "job_completion_is_campaign_completion": False}


def bound_receipt(root, relative, expected):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file() or file_digest(path) != expected:
        raise ValueError("closeout evidence missing, outside archive, or changed")
    return read(path)


def from_progress(progress):
    """Current operational evidence cannot accidentally become final evidence."""
    cards = [{"card_id": row["card_id"], "execution_state": row["implementation_state"],
              "instrument_state": "untested", "scientific_criterion": "not_tested",
              "evidence_verified": False} for row in progress["cards"]]
    return assess([row["card_id"] for row in cards], cards, [], {})
