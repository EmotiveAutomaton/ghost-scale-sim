"""Operational coverage from retained receipts; never infer scientific completion."""
from collections import Counter
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read,write,now,file_digest


def main():
    root=REPO/"results/v16"
    inventory=read(root/"COMMISSION_MANIFEST.json")
    receipts={}
    for path in sorted(list(root.glob("*/COMPLETION.json"))+list(root.glob("*/*/COMPLETION.json"))):
        if "setup" in str(path) or "amended" in str(path):
            continue
        if path.parent.name.startswith("transfer-fixture-"):
            path=path.parent/"PUBLIC_REPORT.json"
        record=read(path)
        card=record.get("card_id","K01" if path.parent.name=="k01-scout-1" else path.parent.name)
        receipts.setdefault(card,[]).append((path,record))
    cards=[]
    for row in inventory["cards"]:
        card=row["card_id"]
        fixture=card.startswith(("X","B"))
        evidence=[]
        for path,record in receipts.get(card,[]):
            evidence.append({"receipt":str(path.relative_to(root)).replace("\\","/"),"sha256":file_digest(path),
                "execution_state":record["execution_state"],"instrument_state":record.get("instrument_state","untested"),
                "evidence_scope":record.get("evidence_scope","fixture" if fixture else "discovery"),"n_maker_packets":record.get("n_maker_packets",record.get("n_condition_records",0)),
                "makers_per_condition":record.get("n_makers_per_condition"),"constructors_per_condition":record.get("constructors_per_condition"),
                "confirmation_state":record.get("confirmation_state","untested"),
                "reader_process":record.get("reader_process","same-process public-byte API"),
                "pending_consumers":record.get("pending_consumers",[]),
                "scope_limit":record.get("scope_limit",record.get("reading_cost_qualification"))})
        for item in evidence:
            if item["receipt"].startswith("inquiry-scout-1/") and (root/"repairs/inquiry-ties-1/PLAN.json").exists():
                item.update(source_receipt_instrument_state=item["instrument_state"], instrument_state="failed",
                    qualification="later recoding and zero-value controls exposed strict-comparison rounding defects",
                    superseded_by="inquiry-scout-2; original raw records and gate receipts preserved")
            if item["receipt"].startswith("inquiry-scout-2/"):
                item["reader_process"]="separate public-only worker"
                item["repair_id"]="inquiry-ties-1"
        completed=[item for item in evidence if item["execution_state"]=="completed" and item["instrument_state"]=="valid"]
        cards.append({"card_id":card,"question":row["question_and_comparison"],"evidence":evidence,
            "implementation_state":("partial" if any(item["pending_consumers"] for item in completed) else "completed") if completed else "pending",
            "scout_state":"not applicable" if fixture else "completed" if completed else "pending",
            "fixture_state":(("native consumers completed; remaining consumer pending" if any(item["pending_consumers"] for item in completed) else "completed") if completed else "pending") if fixture else "not applicable",
            "expansion_state":"not applicable" if fixture else "pending decision" if completed else "not eligible yet",
            "confirmation_state":"untested","closure_state":"open","warrant":"DESCRIPTIVE ONLY","pursuit":"OPENED",
            "process_boundary":("separate from original discovery" if completed and any(
                item["reader_process"]=="separate public-only worker" for item in completed) else
                "retrospective guarded reader replay pending" if completed else "untested")})
    replay=root/"reader-access-replay-1/REPLAY.json"
    if replay.exists() and read(replay)["instrument_state"]=="valid":
        for card in cards:
            if card["scout_state"]=="completed" and any(
                    item["receipt"].split("/")[0] in read(replay)["packet_units"] for item in card["evidence"]):
                card["process_boundary"]="guarded reader re-execution verified; original layout retained in replay receipt"
    access=root/"access-attack-fixture-1/COMPLETION.json"
    if access.exists() and read(access)["instrument_state"]=="valid":
        for card in cards:
            if card["card_id"] in read(access)["consumer_cards"]:
                card["access_attack"]="X01 valid for this actual consumer; no inherited X02-X08 credit"
    amended_access=root/"access-attack-fixture-2/COMPLETION.json"
    recoding=root/"recoding-attack-fixture-1/COMPLETION.json"
    for card in cards:
        for attack,path,key in [("X01",amended_access,"consumer_cards"), ("X02",recoding,"consumer_requests_checked")]:
            if path.exists() and read(path)["instrument_state"]=="valid" and card["card_id"] in read(path)[key]:
                card.setdefault("attack_receipts",{})[attack]={"state":"valid",
                    "receipt":str(path.relative_to(root)).replace("\\","/"),"sha256":file_digest(path)}
                if attack=="X01":
                    card["access_attack"]="X01 valid for amended inquiry or corrected standalone consumer; source-specific receipt retained"
    for attack,packet,key in [("X03","collision-attack-fixture-1","source_condition_calibrations"),
                              ("X04","fairness-attack-fixture-1","native_consumer_condition_profiles"),
                              ("X05","context-attack-fixture-1","source_condition_calibrations"),
                              ("X06","misspecification-attack-fixture-1","source_condition_calibrations"),
                              ("X07","dependence-attack-fixture-1","native_consumer_condition_calibrations"),
                              ("X08","noise-runtime-attack-fixture-1","native_consumer_condition_calibrations")]:
        path=root/packet/"COMPLETION.json"
        if path.exists():
            receipt=read(path)
            if receipt["execution_state"]=="completed" and receipt["instrument_state"]=="valid":
                for card in cards:
                    if card["card_id"] in receipt[key]:
                        card.setdefault("attack_receipts",{})[attack]={"state":"valid",
                            "receipt":str(path.relative_to(root)).replace("\\","/"),"sha256":file_digest(path),
                            "scope_limit":receipt.get("scope_limit",receipt.get("reading_cost_qualification")),
                            "sampling_scope":"known controls and bounded profiles, not new independent discovery makers"}
    archive=root/"archive-search-1/CONSUMER_ATTACKS.json"
    if archive.exists() and read(archive)["instrument_state"]=="valid":
        joined=read(archive)
        for card in cards:
            if card["card_id"]=="B02":
                card["archive_comparison_state"]="completed"
                card["explanatory_catalogue_state"]="completed" if (root/"explanatory-catalogue/COMPLETION.json").exists() else "pending"
                card["implementation_state"]="completed" if card["explanatory_catalogue_state"]=="completed" else "partial"
                for attack,passed in joined["attack_checks"].items():
                    if passed:
                        card.setdefault("attack_receipts",{})[attack]={"state":"valid", "receipt":"archive-search-1/CONSUMER_ATTACKS.json", "sha256":file_digest(archive)}
                card.setdefault("attack_receipts",{})["X08"]={"state":"valid", "receipt":"archive-search-1/private/runtime-control/RECEIPT.json",
                    "sha256":file_digest(root/"archive-search-1/private/runtime-control/RECEIPT.json")}
            if card["card_id"] in {"X04","X07","X08"}:
                for evidence in card["evidence"]:
                    evidence["pending_consumers"]=[name for name in evidence["pending_consumers"] if name!="B02"]
                card["archive_consumer_join"]={"receipt":"archive-search-1/CONSUMER_ATTACKS.json", "sha256":file_digest(archive)}
                remaining=any(item["pending_consumers"] for item in card["evidence"])
                card["implementation_state"]="partial" if remaining else "completed"
                card["fixture_state"]="native and archive consumers completed; confirmation consumers pending" if remaining else "completed"
    expansion=root/"packets/constructor-expansion-1.json"
    if expansion.exists():
        decision=read(expansion)["identity"]["design"]
        planned={item["card_id"] for item in decision["cards"]}
        exhausted={item["card_id"]:item["reason"] for item in decision["closed_at_scout"]}
        for card in cards:
            cid=card["card_id"]
            if cid in planned:
                card["expansion_state"]="256 completed; final ladder decision pending" if (root/"constructor-expansion-1"/cid/"COMPLETION.json").exists() else "256 frozen and eligible"
            elif cid in exhausted:
                card["expansion_state"]="exhausted at scout"
                card["expansion_disposition_reason"]=exhausted[cid]
    counts=dict(Counter(row["implementation_state"] for row in cards))
    report={"written_at":now(),"commission_sha256":read(root/"CAMPAIGN.json")["commission_sha256"],
        "cards":cards,"implementation_counts":counts,
        "scout_counts":dict(Counter(row["scout_state"] for row in cards)),
        "fixture_counts":dict(Counter(row["fixture_state"] for row in cards)),"campaign_complete":False,
        "unit_count_rule":"do not pool different cards or conditions into a scientific sample size",
        "runtime_snapshot":read(root/"RUNNER_STATUS.json") if (root/"RUNNER_STATUS.json").exists() else None}
    write(root/"PROGRESS.json",report,immutable=False)
    print(counts)


if __name__=="__main__":
    main()
