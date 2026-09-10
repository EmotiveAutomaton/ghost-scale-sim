"""Fresh-source condition controls, including the actual metered reading endpoint."""
from collections import Counter
from .records import read, write, digest, file_digest, now
from .consumer_frames import requests
from .resource_accounting import MeasuredReader
from .attack_access import alias
from .recoded_interface import encode, decode
from .attack_recoding import compare_physical
from .fairness import information, audit_unit
from .dependence import duplicate_attacks
from .collision_cases import Recorder, CASES as COLLISIONS
from .context_cases import CASES as CONTEXTS
from .misspecification_cases import CASES as MISPECIFICATIONS
from .attack_collisions import FAMILIES as COLLISION_FAMILIES
from .attack_context import FAMILIES as CONTEXT_FAMILIES
from .attack_misspecification import FAMILIES as MISSPECIFICATION_FAMILIES
from .noise_cases import noise_case

METER="ghostscale.validation.soundingline.v16.reading_cost_meter:public_reader"


class ActualReader:
    """Route reading controls through the endpoint used by the expansion."""
    def __init__(self, reader):
        self.reader=MeasuredReader(reader)
        self.calls=[]

    def request(self, kind, public, **options):
        endpoint=METER if kind=="reading" else kind
        record={"requested_kind":kind,"actual_endpoint":endpoint,"public_sha256":digest(public),"options":options}
        try:
            actual=self.reader.request(endpoint,public,**options)
        except Exception as error:
            record["error"]=str(error)
            self.calls.append(record)
            raise
        if kind=="reading":
            record["invocations"]=actual["invocations"]
            record["invocation_scope"]=actual["scope"]
            result=actual["result"]
        else:
            result=actual
        record["result_sha256"]=digest(result)
        self.calls.append(record)
        return result


def known(reader, cases, family):
    recorded=Recorder(reader)
    checks,witness=cases[family](recorded)
    if not checks or not all(checks.values()):
        raise ValueError("new source failed known "+family+" control")
    return {"family":family,"checks":checks,"witness":witness,"requests":recorded.frames}


def condition_control(source, row, output, raw_reader, required):
    """The whole-packet driver separately verifies cold replay, grouping and runtime."""
    output.mkdir(parents=True,exist_ok=True)
    reader=ActualReader(raw_reader)
    frames=list(requests(source,row))
    if not frames:
        raise ValueError("new consumer has no actual recorded requests")
    baseline=[]
    alias_records=[]
    shadow=output/"private-shadow.json"
    variants=[{"hidden_answer":0,"private_seed":100},{"hidden_answer":1,"private_seed":999}]
    write(shadow,variants[0])
    for frame in frames:
        value=reader.request(frame["kind"],frame["public"],**frame["options"])
        if value!=frame["result"]:
            raise ValueError("new-source actual endpoint differs from its committed prediction")
        baseline.append(reader.calls[-1].copy())
        changed=reader.request(frame["kind"],alias(frame["public"]),**frame["options"])
        if changed!=frame["result"]:
            raise ValueError("new-source result depends on its opaque alias")
        if frame["kind"]=="reading" and reader.calls[-1]["invocations"]!=baseline[-1]["invocations"]:
            raise ValueError("reading-meter invocation count depends on opaque alias")
        alias_records.append(reader.calls[-1].copy())
    # Preserve both fixture values; replacing private state must not change output.
    write(shadow,variants[1],immutable=False)
    renamed_shadow=output/"renamed-private-seed.json"
    shadow.rename(renamed_shadow)
    denied=[]
    for candidate in [shadow,renamed_shadow]:
        try:
            reader.request("_probe_forbidden_read",{"path":str(candidate.resolve())})
        except RuntimeError as error:
            denied.append("PermissionError" in str(error))
        else:
            denied.append(False)
    for frame in reversed(frames):
        if reader.request(frame["kind"],frame["public"],**frame["options"])!=frame["result"]:
            raise ValueError("warm cache/private replacement changed new source")
    if not all(denied):
        raise ValueError("new consumer read private data")
    checks={"X01":True}
    evidence={"private_variants":variants,"private_reads_denied":denied,
              "actual_baselines":baseline,"alias_requests":alias_records}
    if "X02" in required:
        syntax=[]
        for frame in frames:
            representations=[]
            for spelling,split in [("original",False),("renamed",True)]:
                decoded,representation=decode(encode(frame["public"],spelling,split))
                if decoded!=frame["public"] or reader.request(frame["kind"],decoded,**frame["options"])!=frame["result"]:
                    raise ValueError("new source changed under compiled macro representation")
                representations.append(representation)
            physical=compare_physical(frame["kind"],frame["public"],frame["result"],frame["options"],
                                      lambda kind,public,options:reader.request(kind,public,**options))
            syntax.append({"representations":representations,"physical":physical})
        evidence["recoding"]=syntax
        checks["X02"]=True
    for attack,cases,families in [("X03",COLLISIONS,COLLISION_FAMILIES),
                                  ("X05",CONTEXTS,CONTEXT_FAMILIES),
                                  ("X06",MISPECIFICATIONS,MISSPECIFICATION_FAMILIES)]:
        if attack in required:
            family=families[row["card_id"]]
            if attack=="X03" and row["card_id"]=="P04" and row["condition"].startswith("W2-"):
                family="assembly-mechanism"
            evidence[attack]=known(reader,cases,family)
            checks[attack]=True
    # These validate actual executed programs, source-specific costs and paired access.
    cost=audit_unit(source,row,frames)
    evidence["costs"]=cost
    evidence["information"]=information(row["card_id"],frames)
    if row["card_id"] in {"K02","P01","P02","P03"}:
        invocations=[]
        for frame,entry,reference in zip(frames,baseline,cost):
            if entry["actual_endpoint"]!=METER or entry["invocations"]["observation_mass"]!=reference["actual_likelihood_call_invocations"]:
                raise ValueError("actual metered reader work differs from independent call accounting")
            invocations.append({"strategy":frame["options"].get("strategy","maker"),
                                "invocations":entry["invocations"],"scope":entry["invocation_scope"]})
        saved=read(source/"private"/(row["unit_id"]+"-expansion-resources.json"))
        if Counter(digest(item) for item in invocations)!=Counter(digest(item) for item in saved["reading_invocation_corrections"]):
            raise ValueError("new-source metered calls differ from original expansion supplements")
        evidence["reading_meter_join"]=invocations
    checks["X04"]=True
    if "X07" in required:
        evidence["duplicates"]=duplicate_attacks(row)
        checks["X07"]=evidence["duplicates"]["instrument_state"]=="valid"
    if "X08" in required:
        records=[]
        if row["card_id"].startswith("R"):
            for direction in ["rise","decline"]:
                recorded=Recorder(reader)
                noise_checks,witness=noise_case(recorded,direction)
                if not all(noise_checks.values()):
                    raise ValueError("new source failed known physical noise control")
                records.append({"checks":noise_checks,"witness":witness,"requests":recorded.frames})
        evidence["noise"]=records
        # The driver must supply the actual source-bound interruption receipt.
        evidence["runtime"]="pending whole-packet driver join"
    if not all(checks.values()):
        raise ValueError("new-source local control failed")
    return {"execution_state":"completed","instrument_state":"valid","card_id":row["card_id"],
            "condition":row["condition"],"source_unit_id":row["unit_id"],
            "source_unit_sha256":file_digest(source/"units"/(row["unit_id"]+"_points.json")),
            "source_packet_hash":row["packet_hash"],"source_frames_sha256":digest(frames),
            "required_adversaries":sorted(required),"completed_local_checks":checks,
            "evidence":evidence,"requests":reader.calls,"resources":reader.reader.samples,
            "scope":"Condition-level actual source controls; cold-process, full grouping and runtime joins remain driver requirements.",
            "completed_at":now()}
