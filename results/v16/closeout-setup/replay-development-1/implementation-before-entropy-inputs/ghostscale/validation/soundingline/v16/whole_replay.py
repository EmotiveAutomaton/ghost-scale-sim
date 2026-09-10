"""Bounded whole-world, reader, scoring and retained-program replay.

Opaque transport aliases, event timestamps and OS resource samples are the only
non-scientific exclusions. Internal raw hashes are compared through the actual
referenced contents, so fresh timestamps cannot hide a changed scientific input.
"""
import math
from pathlib import Path
from .records import read, write, digest, file_digest

PLATFORM_FIELDS = {"parent_cpu_seconds", "reader_cpu_seconds", "wall_seconds",
    "cpu_seconds", "resident_bytes", "peak_resident_bytes", "startup_wall_seconds"}
TIMESTAMP_FIELDS = {"submitted_at", "prediction_submitted_at", "scored_at", "completed_at",
    "started_at", "recorded_at", "written_at", "finished_at"}
OPAQUE_FIELDS = {"task_id", "lineage_id"}
EXCLUDED_FIELDS = PLATFORM_FIELDS | TIMESTAMP_FIELDS | OPAQUE_FIELDS


def unit_files(base, uid):
    result = {}
    for folder in ["public", "private", "predictions", "units"]:
        for path in sorted((base/folder).glob(uid+"*.json")):
            if "resources" in path.name:
                continue
            result[folder+"/"+path.name] = path
    if "units/"+uid+"_points.json" not in result:
        raise ValueError("whole replay has no retained unit")
    return result


class Contents:
    def __init__(self, files):
        self.payloads = {name:read(path) for name,path in files.items()}
        self.references = {}
        for name,path in files.items():
            for key in [file_digest(path),digest(self.payloads[name])]:
                self.references[key] = self.payloads[name]
        self.cache = {}

    def clean(self, value, chain=frozenset()):
        if isinstance(value, dict):
            return {key:self.clean(item,chain) for key,item in value.items() if key not in EXCLUDED_FIELDS}
        if isinstance(value,list):
            return [self.clean(item,chain) for item in value]
        if isinstance(value,str) and value in self.references:
            if value in chain:
                raise ValueError("cyclic raw hash reference")
            if value not in self.cache:
                self.cache[value] = {"referenced_contents":self.clean(self.references[value],chain|{value})}
            return self.cache[value]
        return value


def compare(actual, expected, path=""):
    if isinstance(actual,dict) and isinstance(expected,dict):
        if set(actual)!=set(expected):
            raise ValueError("whole replay field set differs: "+path)
        for key in actual:
            compare(actual[key],expected[key],path+"/"+key)
    elif isinstance(actual,list) and isinstance(expected,list):
        if len(actual)!=len(expected):
            raise ValueError("whole replay list length differs: "+path)
        for index,(left,right) in enumerate(zip(actual,expected)):
            compare(left,right,path+"/"+str(index))
    elif isinstance(actual,float) or isinstance(expected,float):
        if isinstance(actual,bool) or isinstance(expected,bool) or not isinstance(actual,(int,float)) or not isinstance(expected,(int,float)) or not math.isfinite(actual) or not math.isfinite(expected) or not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-10):
            raise ValueError("whole replay numerical result differs: "+path)
    elif actual!=expected or type(actual)!=type(expected):
        raise ValueError("whole replay deterministic value differs: "+path)


def compare_unit(source, replay, uid):
    source_files,replay_files=unit_files(source,uid),unit_files(replay,uid)
    if set(source_files)!=set(replay_files):
        raise ValueError("whole replay raw file inventory differs: "+repr(set(source_files)^set(replay_files)))
    left,right=Contents(source_files),Contents(replay_files)
    for name in source_files:
        compare(right.clean(right.payloads[name]),left.clean(left.payloads[name]),name)
    return {"unit_id":uid,"deterministic_files":len(source_files),
            "source_files":{name:file_digest(path) for name,path in source_files.items()},
            "replay_files":{name:file_digest(path) for name,path in replay_files.items()},
            "world_reader_and_score_match":True,
            "platform_rule":"Original OS costs retained; fresh clock/RSS values need not match.",
            "opaque_alias_rule":"Transport aliases may differ; source-specific X01 checks validate non-use as scientific evidence.",
            "floating_tolerance":1e-10}


def replay_item(root, destination, item, reader):
    from .expansion_adapter import execute_unit
    original_path=root/item["source_unit"]
    if file_digest(original_path)!=item["source_unit_sha256"]:
        raise ValueError("selected replay source changed")
    original=read(original_path)
    source=original_path.parent.parent
    packet=read(root/"packets"/(item["packet_id"]+".json"))
    if packet["packet_hash"]!=original["packet_hash"]:
        raise ValueError("replay source packet differs")
    if item["kind"]=="native":
        from .vertical import run_case
        row=run_case(destination,packet_hash=packet["packet_hash"],index=original["seed_components"]["index"])
    elif item["kind"]=="archive":
        from .archive_study import candidate
        record=candidate(destination,item["condition"],original["seed_components"]["index"],
                         original["lineage"],packet,reader,item["constructors"])
        row=read(destination/"units"/(record["case"]["unit_id"]+"_points.json"))
    else:
        row=execute_unit(original["card_id"],destination,item["condition"],original["seed_components"]["index"],
                         namespace=original["lineage"],packet=packet,reader=reader,constructors=item["constructors"],
                         scope=original["evidence_scope"])
    if row["unit_id"]!=original["unit_id"]:
        raise ValueError("whole world replay changed its immutable unit identity")
    return compare_unit(source,destination,original["unit_id"])

