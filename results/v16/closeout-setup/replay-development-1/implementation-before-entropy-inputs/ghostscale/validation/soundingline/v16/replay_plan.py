"""Mechanical, outcome-independent selection of a bounded diverse replay set."""
from pathlib import Path
from .records import read, write, digest, file_digest, now
from .aggregate_science import registered_designs
from .record_integrity import source_locks

MAX_UNITS=128


def lookup(base, condition, index):
    probe=next((base/"units").glob("*_points.json"),None)
    if probe is None:
        raise ValueError("replay source has no units")
    example=read(probe)
    namespace=example["lineage"]
    identities=[[namespace,condition,index],[namespace,example["card_id"],condition,index]]
    matches=[base/"units"/(digest(parts)[:24]+"_points.json") for parts in identities]
    matches=[path for path in matches if path.exists()]
    if len(matches)!=1:
        raise ValueError("replay seed lookup is missing or ambiguous")
    row=read(matches[0])
    if row["condition"]!=condition or row["seed_components"]["index"]!=index:
        raise ValueError("replay immutable seed identity differs")
    return matches[0],row


def item(root,path,packet_id,kind,condition,constructors):
    return {"source_unit":path.relative_to(root).as_posix(),"source_unit_sha256":file_digest(path),
            "packet_id":packet_id,"kind":kind,"condition":condition,"constructors":constructors}


def select_native(root, *, include_expansion=False):
    items=[]
    for packet_path in sorted((root/"packets").glob("*.json")):
        name=packet_path.stem
        if name=="inquiry-scout-1":
            continue
        is_scout="-scout-" in name
        is_expansion=name.startswith("constructor-expansion-") or name.startswith("boundary-expansion-")
        if not is_scout and not (is_expansion and include_expansion):
            continue
        packet=read(packet_path)
        specification=packet["identity"]["design"]
        designs=registered_designs(specification)
        if is_expansion and not (root/name/"COMPLETION.json").exists():
            raise ValueError("cannot select an unfinished expansion for completed scientific replay")
        for card,design in sorted(designs.items()):
            base=root/name/card if (root/name/card/"units").exists() else root/name
            n,constructors=specification.get("n_per_condition",64),specification.get("constructors",8)
            if isinstance(specification.get("cards"),list):
                entry=next(entry for entry in specification["cards"] if entry["card_id"]==card)
                n,constructors=entry["n_per_condition"],entry["constructors"]
            for condition,index in [(design["conditions"][0],0),(design["conditions"][-1],n-1)]:
                path,row=lookup(base,condition["id"],index)
                actual_constructors=row["seed_components"].get("constructors",constructors)
                items.append(item(root,path,name,"native-card",condition,actual_constructors))
    return items


def select_archive(root):
    from .archive_design import CONDITIONS
    definitions={condition["id"]:condition for condition in CONDITIONS}
    base=root/"archive-search-1"
    if not (base/"COMPLETION.json").exists():
        raise ValueError("archive replay requires a completed frozen follow-up")
    items=[]
    journals=[read(path) for path in (base/"units").glob("*_points.json")]
    for method in ["fixed","adaptive"]:
        selected=sorted((row for row in journals if row["method"]==method),key=lambda row:(row["replicate"],row["step"]))
        for journal in [selected[0],selected[-1]]:
            receipt_path=base/journal["candidate_receipt"]
            receipt=read(receipt_path)
            path=receipt_path.parent.parent/"units"/(receipt["case"]["unit_id"]+"_points.json")
            row=read(path)
            items.append(item(root,path,"archive-search-1","archive",definitions[row["condition"]],12))
    follow=base/"private/followup"
    for condition,index in [(CONDITIONS[0],0),(CONDITIONS[-1],23)]:
        path,row=lookup(follow,condition["id"],index)
        items.append(item(root,path,"archive-search-1","archive",condition,24))
    return items


def plan(root, repo, *, include_expansion=False):
    checks=source_locks(root,repo)
    selected=select_native(root,include_expansion=include_expansion)+select_archive(root)
    for path in sorted((root/"native-fixture-1/units").glob("*_points.json")):
        selected.append(item(root,path,"native-fixture-1","native",None,1))
    if not selected or len(selected)>MAX_UNITS or len({row["source_unit"] for row in selected})!=len(selected):
        raise ValueError("bounded replay allocation empty, duplicated or exceeds cap")
    return {"schema_version":"v16.whole-replay-plan.1","source_checks":checks,
            "selected":selected,"n_units":len(selected),"hard_cap":MAX_UNITS,
            "selection":"First registered condition/index zero and last registered condition/final index of each valid current native card; fixed/adaptive archive endpoints and frozen follow-up endpoints; two original native fixtures.",
            "outcomes_used_for_selection":False,"includes_completed_expansion":include_expansion,
            "invalid_original_inquiry":"Preserved and independently reaggregated; not admitted as a valid deterministic replay source.",
            "pending_coverage":["future confirmation packets","any later finite boundary expansion"],
            "timestamp_rule":"Event timestamps differ; training dates and scientific seed lineages must match.",
            "platform_rule":"OS CPU/wall/RSS are measured observations; primitive, training, search and likelihood costs must match.",
            "opaque_alias_rule":"task_id and lineage_id are transport aliases; scientific lineage, maker and constructor identities must match.",
            "comparison":"Regenerate full worlds, acquisition, guarded reader predictions, continuations, execution and scores; compare every retained unit file via referenced contents.",
            "floating_absolute_tolerance":1e-10,"planned_at":now()}

