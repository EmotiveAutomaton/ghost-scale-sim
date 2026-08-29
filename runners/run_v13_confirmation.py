"""V13 confirmation pass (spec wave 5): promoted cards re-run on the untouched confirmation
lineage. Promotion is mechanical: a card is promoted when its discovery verdict LANDED, every
pre-specified criterion it carries passed, and its claim ceiling is METHOD or
CONSTRUCTED_MECHANISM. Confirmation verdicts go to results/validation/soundingline/v13/confirmation/
and the ledger to results/v13/CONFIRMATION.json.

The promoted set is frozen in the ledger before the first confirmation world is touched, and
the freeze is binding on every later entry path:

  * an automatic resume runs the FROZEN membership, never today's recomputed promotion;
  * --only is a filter over the frozen packet, never a way to add a card to it;
  * the recorded discovery hashes, the structural/scientific lock hashes and the card and
    criterion identities are verified before any worker is created;
  * a resume whose inputs no longer match the freeze is refused without writing anything.

Growing the packet is an amendment to a frozen scientific program, not a resume -- but it is
permitted, because HEALING_PLAN.md's confirmation step adds cards promoted by the healing pass
and the curator kept that plan (2026-08-28). The rule is therefore *recorded, never silent*:

  * an amendment preserves the original packet beside the replacement, with the reason, the
    time, and the cards added, in the ledger and in results/v13/AMENDMENTS.json;
  * a card added by amendment gets an untouched confirmation lineage for free -- ``rng_for``
    seeds on the card id, so no added card reuses or perturbs another card's worlds;
  * everything already in the packet is still verified before any of it runs.

``--no-amend`` refuses to widen instead, leaving the packet untouched.

    python runners/run_v13_confirmation.py --auto
    python runners/run_v13_confirmation.py --only C04 O02
    python runners/run_v13_confirmation.py --auto --no-amend
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import ghostscale.validation.soundingline.v13 as V                          # noqa: E402
from ghostscale.validation.soundingline.v13 import common as C              # noqa: E402
from ghostscale.validation.soundingline.v13 import manifest as M            # noqa: E402
from ghostscale.validation.soundingline.v13.atomicio import write_json_atomic  # noqa: E402
from ghostscale.validation.soundingline.v13.runtime import init_worker      # noqa: E402
from ghostscale.validation.soundingline.v13.schemas import RESOLVED, card_from_dict  # noqa: E402

LEDGER = V.V13_RESULTS / "CONFIRMATION.json"

#: Bumped when the freeze record gains fields. A packet written by an older schema is honoured
#: for the fields it actually carries; a missing field is reported as unverifiable, never
#: silently treated as a verified match.
PACKET_SCHEMA = 1


class FreezeViolation(RuntimeError):
    """The requested confirmation work is not the frozen program. Nothing has been written."""


def criteria_passed(verdict: dict):
    crit = [x for k, x in (verdict or {}).get("results", {}).items() if k.startswith("criterion_") and isinstance(x, dict) and "passed" in x]
    if not crit:
        return None
    return all(bool(x["passed"]) for x in crit)


def promoted(doc: dict) -> list:
    out = []
    for d in doc["cards"]:
        if d["trunk"] in ("I", "L", "X") or d["status"] != "LANDED":
            continue
        v = C.load_verdict(d["id"], "discovery")
        if v is None:
            continue
        if criteria_passed(v) and v.get("claim_ceiling") in ("METHOD", "CONSTRUCTED_MECHANISM"):
            out.append(d["id"])
    return out


# --------------------------------------------------------------------------- #
# The frozen packet: identity, creation, verification.
# --------------------------------------------------------------------------- #
def _discovery_path(cid: str) -> Path:
    return V.verdict_dir("discovery") / f"{cid}.json"


def _lock_identity() -> dict:
    """Which frozen program this packet belongs to.

    Recorded at freeze time so a later resume can tell that the criteria, the card set or the
    selected workload have not been re-specified underneath a running confirmation pass.
    """
    from ghostscale.prereg_v13 import PREREG_PATH, STRUCTURAL_PATH
    return {"structural_lock_sha256": C.file_sha(STRUCTURAL_PATH) if STRUCTURAL_PATH.exists() else None,
            "scientific_lock_sha256": C.file_sha(PREREG_PATH) if PREREG_PATH.exists() else None}


def _card_identity(doc: dict, ids: list) -> dict:
    """The definition of each packet card, independent of its working status.

    Status and verdict pointers move as the program runs; the question, criteria, lanes and
    claim ceiling are the scientific content and must not.
    """
    by_id = {d["id"]: d for d in doc["cards"]}
    out = {}
    for cid in ids:
        d = by_id.get(cid)
        if d is None:
            continue
        out[cid] = C.obj_sha({k: v for k, v in d.items() if k not in ("status", "state", "resolved", "verdict_path", "actual")})
    return out


def make_packet(doc: dict, ids: list) -> dict:
    return {"schema": PACKET_SCHEMA,
            "promoted": list(ids),
            "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "discovery_hashes": {i: C.file_sha(_discovery_path(i)) for i in ids if _discovery_path(i).exists()},
            "card_identity": _card_identity(doc, ids),
            **_lock_identity()}


def verify_packet(doc: dict, packet: dict) -> list:
    """Return the reasons this packet no longer matches the checkout. Empty means it does.

    Every card is checked, including ones already marked resolved: a resumed pass must not
    inherit a result whose discovery input has since changed.
    """
    problems = []
    ids = list(packet.get("promoted") or [])
    if not ids:
        problems.append("the frozen packet records no promoted cards")

    hashes = packet.get("discovery_hashes") or {}
    for cid in ids:
        p = _discovery_path(cid)
        recorded = hashes.get(cid)
        if recorded is None:
            problems.append(f"{cid}: no discovery hash was recorded at freeze time (unverifiable, not a match)")
            continue
        if not p.exists():
            problems.append(f"{cid}: frozen discovery verdict is missing from disk")
            continue
        if C.file_sha(p) != recorded:
            problems.append(f"{cid}: discovery verdict bytes changed since the freeze")

    recorded_cards = packet.get("card_identity")
    if recorded_cards is None:
        problems.append("no card identity was recorded at freeze time (unverifiable, not a match)")
    else:
        now = _card_identity(doc, ids)
        for cid in ids:
            if cid not in recorded_cards:
                problems.append(f"{cid}: no card identity was recorded at freeze time")
            elif cid not in now:
                problems.append(f"{cid}: card is absent from the current manifest")
            elif now[cid] != recorded_cards[cid]:
                problems.append(f"{cid}: card definition or criteria changed since the freeze")

    live = _lock_identity()
    for key, label in (("structural_lock_sha256", "structural lock"), ("scientific_lock_sha256", "scientific lock")):
        if packet.get(key) is None:
            problems.append(f"no {label} hash was recorded at freeze time (unverifiable, not a match)")
        elif live.get(key) != packet[key]:
            problems.append(f"the {label} changed since the freeze")
    return problems


def amend_packet(doc: dict, packet: dict, add: list, reason: str) -> tuple:
    """Widen a frozen packet explicitly, keeping the original beside the replacement.

    Returns ``(new_packet, amendment_record)``. The original packet is copied verbatim into the
    record before anything is changed, so the program a result was produced under stays
    recoverable even after several amendments.

    An added card needs no new lineage allocation: ``C.rng_for`` seeds on ``lane|card|world|rep``,
    so its confirmation worlds are disjoint from every other card's by construction. The cards
    already in the packet keep the exact worlds they were frozen with.
    """
    original = json.loads(json.dumps(packet))                # deep copy before mutation
    merged = list(packet["promoted"]) + [c for c in add if c not in packet["promoted"]]
    new_packet = make_packet(doc, merged)
    new_packet["amended_from"] = C.obj_sha(original)
    new_packet["amendment_count"] = int(packet.get("amendment_count", 0)) + 1
    # the frozen cards keep their ORIGINAL recorded hashes and identities: re-deriving them here
    # would let a card that drifted since the freeze be silently re-blessed by the amendment
    for field in ("discovery_hashes", "card_identity"):
        new_packet[field] = {**new_packet.get(field, {}), **(original.get(field) or {})}
    record = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason,
              "added": [c for c in merged if c not in original["promoted"]],
              "original_packet": original, "replacement_identity": C.obj_sha(new_packet)}
    return new_packet, record


def supersede(ledger: dict, ids: list, reason: str) -> dict:
    """Remove cards from the frozen packet because their discovery verdicts are being replaced
    by an instrument correction (HEALING_PLAN.md). The inverse of ``amend_packet`` and equally
    recorded: the original packet is preserved verbatim, the removed cards' confirmation entries
    move to ``ledger["superseded"]`` and into the record, and the packet's hash for each removed
    card is dropped so a later ordinary amendment re-adds it with the NEW discovery hash. A card
    that is not in the packet is refused: there is nothing to supersede.
    """
    packet = ledger.get("frozen")
    if not packet:
        raise FreezeViolation("no frozen packet to supersede from")
    missing = [c for c in ids if c not in packet["promoted"]]
    if missing:
        raise FreezeViolation(f"cards {missing} are not in the frozen confirmation packet {packet['promoted']}")
    original = json.loads(json.dumps(packet))
    new_packet = json.loads(json.dumps(packet))
    new_packet["promoted"] = [c for c in packet["promoted"] if c not in ids]
    for field in ("discovery_hashes", "card_identity"):
        new_packet[field] = {k: v for k, v in (packet.get(field) or {}).items() if k not in ids}
    new_packet["amended_from"] = C.obj_sha(original)
    new_packet["amendment_count"] = int(packet.get("amendment_count", 0)) + 1
    removed_conf = {c: ledger.get("cards", {}).pop(c) for c in ids if c in ledger.get("cards", {})}
    record = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason, "removed": list(ids),
              "superseded_confirmations": removed_conf, "original_packet": original,
              "replacement_identity": C.obj_sha(new_packet)}
    ledger["frozen"] = new_packet
    ledger.setdefault("amendments", []).append(record)
    ledger.setdefault("superseded", {}).update({c: removed_conf.get(c) for c in ids})
    return record


def resolve_ids(packet: dict, only) -> list:
    """The cards this invocation may run: the frozen packet, or an explicit subset of it.

    ``only is None`` means "no subset requested" and runs the whole packet. An explicitly empty
    subset is a caller error, not an instruction to expand back to today's promotion result --
    that silent widening is exactly what the freeze exists to prevent.
    """
    ids = list(packet["promoted"])
    if only is None:
        return ids
    requested = list(only)
    if not requested:
        raise FreezeViolation("an explicitly empty --only selects no cards; pass --auto to run the whole frozen packet")
    extra = [c for c in requested if c not in ids]
    if extra:
        raise FreezeViolation(
            f"cards {extra} are not in the frozen confirmation packet {ids}. "
            "Confirmation runs the frozen program; adding a card is an amendment with its own "
            "untouched lineage, not a resume.")
    seen, out = set(), []
    for c in requested:                                      # preserve the caller's order, drop repeats
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# The pass.
# --------------------------------------------------------------------------- #
def run(doc: dict, workers: int, pool, only=None, amend: bool = True,
        amend_reason: str = "HEALING_PLAN.md confirmation step: cards promoted by the healing pass") -> None:
    from runners.run_v13 import record_runtime, run_card, tier_for
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"cards": {}, "frozen": None}

    # ---- freeze, or honour the existing freeze ---------------------------- #
    packet = ledger.get("frozen")
    if packet is None:
        first = list(only) if only is not None else promoted(doc)
        if not first:
            raise FreezeViolation("nothing to freeze: no card is currently promoted")
        packet = make_packet(doc, first)
        ledger["frozen"] = packet
        write_json_atomic(LEDGER, ledger)                    # frozen before the first confirmation world is touched
        print(f"confirmation packet frozen: {len(first)} cards {first}", flush=True)
    else:
        problems = verify_packet(doc, packet)
        if problems:
            raise FreezeViolation(
                "the frozen confirmation packet no longer matches this checkout:\n  - "
                + "\n  - ".join(problems)
                + "\nNothing was written. Repairing a technical fault and amending a frozen "
                  "scientific program are different decisions; an amendment needs an explicit "
                  "new or amended packet on an unused confirmation lineage.")

        # ---- explicit, recorded widening (the healing plan's step) -------- #
        if amend:
            candidates = list(only) if only else promoted(doc)
            added = [c for c in candidates if c not in packet["promoted"]]
            if added:
                packet, record = amend_packet(doc, packet, added, amend_reason)
                ledger["frozen"] = packet
                ledger.setdefault("amendments", []).append(record)
                write_json_atomic(LEDGER, ledger)            # the amendment lands before the work does
                M.add_amendment("confirmation_packet", record["original_packet"], packet, amend_reason)
                print(f"!! confirmation packet AMENDED: added {added} "
                      f"(amendment {packet['amendment_count']}; original preserved)", flush=True)

    ids = resolve_ids(packet, only)
    packet_id = C.obj_sha(packet)
    tname, tier = tier_for(doc)
    print(f"confirmation pass on {len(ids)} of {len(packet['promoted'])} frozen cards at {tname}: {ids}", flush=True)
    print(f"packet frozen {packet.get('when')} (identity {packet_id[:12]})", flush=True)

    for cid in ids:
        if cid in ledger["cards"] and ledger["cards"][cid].get("state") in RESOLVED:
            print(f"  [{cid}] already confirmed: {ledger['cards'][cid]['state']}")
            continue
        card = card_from_dict(next(d for d in doc["cards"] if d["id"] == cid))
        card.lanes = ["confirmation"]
        disc = C.load_verdict(cid, "discovery") or {}
        t0 = time.perf_counter()
        try:
            out = run_card(doc, card, "confirmation", tname, tier, workers, pool)
            v, acct = out["verdict"], out["accounting"]
            ledger["cards"][cid] = {"state": v["state"], "wall_s": acct["wall_s"], "children_cpu_s": acct["children_cpu_s"],
                                    "criteria_passed_discovery": criteria_passed(disc), "criteria_passed_confirmation": criteria_passed(v),
                                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "closure_reason": v.get("closure_reason", ""),
                                    "packet_identity": packet_id}
            record_runtime(cid, "confirmation", acct, v["state"])
            print(f"  [{cid}] -> {v['state']}; criteria {ledger['cards'][cid]['criteria_passed_confirmation']}", flush=True)
        except Exception as exc:                                             # noqa: BLE001
            ledger["cards"][cid] = {"state": "ERROR", "error": repr(exc), "wall_s": round(time.perf_counter() - t0, 1),
                                    "packet_identity": packet_id}
            print(f"  [{cid}] !! ERROR {exc!r}", flush=True)
            from concurrent.futures.process import BrokenProcessPool
            if isinstance(exc, BrokenProcessPool):
                write_json_atomic(LEDGER, ledger)
                raise SystemExit("worker pool broken; relaunch to resume the confirmation pass") from exc
        write_json_atomic(LEDGER, ledger)
    held = [c for c, x in ledger["cards"].items() if x.get("criteria_passed_confirmation") is True]
    print(f"\nconfirmed (criteria held on the untouched lineage): {held}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-amend", action="store_true",
                    help="refuse to widen the frozen packet instead of recording an amendment")
    ap.add_argument("--amend-reason", default="HEALING_PLAN.md confirmation step: cards promoted by the healing pass")
    ap.add_argument("--supersede", nargs="*", default=None,
                    help="remove these cards from the frozen packet (recorded amendment) because their discovery "
                         "verdicts are being replaced; runs no worker, writes only the ledger and the amendment record")
    ap.add_argument("--supersede-reason", default="HEALING_PLAN.md: discovery verdict superseded by an instrument correction")
    args = ap.parse_args()
    if os.environ.get("GS_V13_SMOKE"):
        sys.exit("refusing to run the confirmation pass under GS_V13_SMOKE")
    if args.supersede is not None:
        if not args.supersede:
            sys.exit("--supersede needs card ids")
        ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else None
        if not ledger or not ledger.get("frozen"):
            sys.exit("no frozen packet to supersede from")
        try:
            record = supersede(ledger, args.supersede, args.supersede_reason)
        except FreezeViolation as exc:
            sys.exit(f"supersede refused: {exc}")
        write_json_atomic(LEDGER, ledger)                    # the record lands before anything moves
        M.add_amendment("confirmation_packet", record["original_packet"], ledger["frozen"], args.supersede_reason)
        stamp = record["when"].replace(":", "")
        for cid in args.supersede:                           # the superseded verdict stays on disk beside the record
            p = V.verdict_dir("confirmation") / f"{cid}.json"
            if p.exists():
                dest = V.verdict_dir("confirmation") / "superseded" / f"{cid}.{stamp}.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                p.rename(dest)
        print(f"!! confirmation packet AMENDED: superseded {args.supersede} (amendment {ledger['frozen']['amendment_count']}; "
              f"original preserved); packet now {ledger['frozen']['promoted']}", flush=True)
        return
    workers = args.workers or max(1, min(12, (os.cpu_count() or 2) // 2))
    C.lower_priority()
    C.hide_accelerators()
    doc = M.load_manifest()
    if not args.auto and args.only is None:
        sys.exit("nothing to run: pass --auto or --only")
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        try:
            run(doc, workers, pool, only=args.only, amend=not args.no_amend, amend_reason=args.amend_reason)
        except FreezeViolation as exc:
            sys.exit(f"confirmation refused: {exc}")


if __name__ == "__main__":
    main()
