"""Read-only raw/source integrity and independent sampling identity checks."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def independent_units(rows):
    if not rows:
        raise ValueError("empty evidence is not an independent sample")
    ids=set();seed_identities=set();history_identities=set();groups=Counter()
    for row in rows:
        required={"unit_id","lineage","card_id","condition","constructor_id","maker_history_id","seed_components"}
        if not required<=set(row) or "index" not in row["seed_components"]:
            raise ValueError("missing independent sampling identity")
        if row["unit_id"] in ids:
            raise ValueError("duplicate unit identity")
        ids.add(row["unit_id"])
        context=(row["lineage"],row["card_id"],row["condition"])
        seed=context+(row["seed_components"]["index"],)
        history=context+(row["maker_history_id"],)
        if seed in seed_identities or history in history_identities:
            raise ValueError("duplicate maker history despite renamed row")
        seed_identities.add(seed);history_identities.add(history)
        groups[context+(row["constructor_id"],)]+=1
    return {"n_retained_unit_records":len(rows),
        "constructor_groups":[{"lineage":lineage,"card":card,"condition":condition,"constructor":constructor,"makers":count}
            for (lineage,card,condition,constructor),count in sorted(groups.items())],
        "scope":"separate condition-specific sampling groups; do not pool cards or nested artifacts",
        "coincident_outcomes":"identical artifacts from independently generated histories are allowed"}

def source_locks(root,repo,git_ref=None):
    if git_ref is not None and git_ref not in {"HEAD","origin/main"}:
        raise ValueError("unsupported local Git source reference")
    sources={};packets={}
    for path in sorted((root/"packets").glob("*.json")):
        packet=json.loads(path.read_bytes())
        identity=packet["identity"];packets[identity["packet_id"]]=packet["packet_hash"]
        for relative,expected in identity["files"].items():
            if relative in sources and sources[relative]!=expected:
                raise ValueError("different frozen bytes require separate source checkouts")
            sources[relative]=expected
    if not packets:
        raise ValueError("no scientific packet identities")
    for relative,expected in sources.items():
        if sha(repo/relative)!=expected:
            raise ValueError(f"working scientific source differs: {relative}")
        if git_ref is not None:
            # Trust only this explicitly supplied source checkout for this read.
            # No global Git setting or hook policy is changed.
            result=subprocess.run(["git","-c","safe.directory="+repo.resolve().as_posix(),
                "cat-file","blob",f"{git_ref}:{relative}"],cwd=repo,capture_output=True)
            if result.returncode:
                raise ValueError("Git source read failed: "+result.stderr.decode("utf-8",errors="replace"))
            value=result.stdout
            if hashlib.sha256(value).hexdigest()!=expected:
                raise ValueError(f"committed scientific bytes differ: {relative}")
    return {"packets":packets,"unique_scientific_files":len(sources),"working_source":"valid",
        "committed_source":"valid" if git_ref else "not checked","git_reference":git_ref}

def raw_integrity(root,packets):
    all_rows=[];manifests=[];files_checked=0;retained_bytes=0
    for packet_id,packet_hash in packets.items():
        packet_root=root/packet_id
        paths=sorted(packet_root.glob("**/RAW_MANIFEST.json"))
        if not paths:
            raise ValueError(f"missing raw retention manifest: {packet_id}")
        for path in paths:
            manifest=json.loads(path.read_bytes())
            mapping=manifest.get("files")
            if not isinstance(mapping,dict) or not mapping:
                raise ValueError("raw manifest lacks its complete file map")
            base=path.parent.resolve()
            actual={str(file.relative_to(base)).replace("\\","/") for name in ["public","private","predictions","units"]
                for file in (base/name).glob("*.json")}
            if set(mapping)!=actual:
                raise ValueError(f"missing or unmanifested raw files: {path}")
            for relative,expected in mapping.items():
                target=(base/relative).resolve()
                if not target.is_relative_to(base):
                    raise ValueError("raw manifest path escapes its packet")
                if sha(target)!=expected:
                    raise ValueError(f"retained raw bytes differ: {relative}")
                files_checked+=1;retained_bytes+=target.stat().st_size
            sample=[]
            for file in sorted((base/"units").glob("*_points.json")):
                row=json.loads(file.read_bytes())
                if row["packet_hash"]!=packet_hash or file.name!=row["unit_id"]+"_points.json":
                    raise ValueError("unit file or scientific packet identity differs")
                sample.append(row)
            audit=independent_units(sample)
            manifests.append({"manifest":str(path.relative_to(root)).replace("\\","/"),"sha256":sha(path),
                "files":len(mapping),"sampling":audit})
            all_rows.extend(sample)
    # Different cards/conditions remain separate; global unit IDs cannot collide.
    independent_units(all_rows)
    return {"instrument_state":"valid","manifests":manifests,"raw_files_checked":files_checked,
        "raw_bytes_checked":retained_bytes,"retained_unit_records":len(all_rows),
        "aggregate_regeneration":False,"scientific_rollout_replay":False,
        "scope":"integrity and sampling identities only; physical/scoring audits are separate"}
