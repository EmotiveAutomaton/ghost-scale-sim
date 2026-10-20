"""One fresh excluded D5 cost pilot using the verified exact-fit adapter.

The original forty-world null/positive study is not admitted here. Every
chronological fit and full control path is retained for independent checking.
"""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

import numpy as np

from ..v23 import ledger, profile_validation as base, profile_successor
from ..v23.common import canonical, digest, read, write
from . import fit_fixture, pilot as transport


def configuration():
    config=profile_successor.configuration()
    config.update(study="V24-D5-EXACT-COST-PILOT",pilot_seed_namespace="ghost-v24-d5-pilot-1",
                  seed_namespace="ghost-v24-d5-production-not-admitted")
    return config


def validate_design(plan):
    if (plan["kind"]!="pilot" or plan["family"]!="D5" or plan["identity"]!="v24-d5-pilot-1"
            or plan["configuration"]!=configuration() or plan["world_indices"]!=[0]
            or plan.get("confirmation") is not False):
        raise ValueError("fresh bounded D5 cost pilot required")
    if plan["cpu_cap_seconds"]!=5400 or plan["wall_cap_seconds"]!=14400:
        raise ValueError("frozen complete-pass D5 limits changed")
    if plan["backbone_sha256"]!="5563cc4b7e46638640d2455bf4aaae69c7ffb32f1c434f381b1b7bedc251e244":
        raise ValueError("wrong certified fitting source")
    if plan["profile_sha256"]!=base.PROFILE_HASH or plan["task_sha256"]!=base.TASK_HASH:
        raise ValueError("upstream feature functions changed")
    proof=Path(plan["retained_equivalence"])
    if digest(proof)!=plan["retained_equivalence_sha256"]:
        raise ValueError("retained equivalence binding changed")
    if read(proof)["state"]!="RETAINED_FULL_FIXTURE_VERIFIED":
        raise ValueError("complete retained equivalence is required before admission")


def functions(plan):
    namespace=dict(np=np,re=re,hashlib=hashlib,datetime=datetime,timezone=timezone,
                   ARMS=base.ARMS,ACTIONS=base.ACTIONS)
    for source_key,hash_key,names in (
        ("profile_source","profile_sha256",("profile_vector","choice_features")),
        ("task_source","task_sha256",("features",))):
        path=Path(plan[source_key])
        if digest(path)!=plan[hash_key]:
            raise ValueError("pinned feature source changed")
        tree=ast.parse(path.read_bytes())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=set(names):
            raise ValueError("missing exact feature functions")
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),namespace)
    return namespace


def generated(plan):
    fn=functions(plan)
    return {condition:base.construct(0,condition,fn,pilot=True,cfg=plan["configuration"])
            for condition in base.CONDITIONS}


def fitting_plan(root,plan):
    inputs=Path(root)/"inputs"
    return dict(plan,input_root=str(inputs),future_prefix=plan["configuration"]["future_prefix"],
                input_files={p.relative_to(inputs).as_posix():digest(p)
                             for p in inputs.rglob("*") if p.is_file()})


def validate_generated(plan):
    # Called only inside the already source-verified fresh pilot. Data hashes
    # authenticate streams; independent reconstruction also regenerates the inputs.
    if plan["identity"]!="v24-d5-pilot-1":
        raise ValueError("wrong fresh pilot identity")
    for name,expected in plan["input_files"].items():
        if digest(Path(plan["input_root"])/name)!=expected:
            raise ValueError("generated input changed")


def preflight(plan):
    validate_design(plan)
    functions(plan)  # Parse and pin exact functions; do not call the generator.
    return dict(passed=True,new_scientific_worlds=0,
                retained_full_equivalence_sha256=plan["retained_equivalence_sha256"])


def summary(plan):
    return dict(state="COST_PILOT_EXECUTED_PENDING_REVIEW",family="D5",
        identity=plan["identity"],new_independent_worlds=1,conditions=list(fit_fixture.CONDITIONS),
        targets_per_condition=128,confirmation=False,scientific_primary_decided=False,
        limitation="Excluded complete cost/control pilot; original forty-world criteria and failures unchanged.")


def execute(root,plan,pulse):
    validate_design(plan);root=Path(root);root.mkdir(parents=True,exist_ok=False)
    schema,_=ledger.canonical_schema(plan["schema_directory"],plan["schema_pin"])
    for condition,(records,labels,truth) in generated(plan).items():
        write(root/"inputs"/condition/"READER.json",records)
        write(root/"inputs"/condition/"LABELS.json",labels)
        write(root/"truth"/condition/"EVALUATOR.json",truth)
        bundle=transport.emit(schema,[plan["identity"],condition],
                              dict(records=records,resolved_labels=labels))
        write(root/"ledger"/(condition+"_points.json"),bundle)
    effective=fitting_plan(root,plan)
    fit_fixture.execute(root/"fits",effective,pulse,input_validator=validate_generated)
    write(root/"SUMMARY.json",summary(plan))
    write(root/"RAW_INDEX.json",{p.relative_to(root).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size)
                                for p in sorted(root.rglob("*")) if p.is_file()})


def reconstruct(root,plan,pulse):
    validate_design(plan);root=Path(root)
    index=read(root/"RAW_INDEX.json")
    expected={p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p!=root/"RAW_INDEX.json"}
    if expected!=set(index):
        raise ValueError("complete D5 file roster differs")
    for name,item in index.items():
        if digest(root/name)!=item["sha256"] or (root/name).stat().st_size!=item["bytes"]:
            raise ValueError("D5 raw hash mismatch")
    if read(root/"SUMMARY.json")!=summary(plan):
        raise ValueError("D5 summary differs from the declared pilot")
    schema,_=ledger.canonical_schema(plan["schema_directory"],plan["schema_pin"])
    for condition,(records,labels,truth) in generated(plan).items():
        if (canonical(read(root/"inputs"/condition/"READER.json"))!=canonical(records)
                or canonical(read(root/"inputs"/condition/"LABELS.json"))!=canonical(labels)
                or canonical(read(root/"truth"/condition/"EVALUATOR.json"))!=canonical(truth)):
            raise ValueError("fresh world or evaluator regeneration differs")
        bundle=read(root/"ledger"/(condition+"_points.json"))
        ledger.validate(bundle,schema)
        if canonical(bundle)!=canonical(transport.emit(schema,[plan["identity"],condition],
                                                       dict(records=records,resolved_labels=labels))):
            raise ValueError("independent canonical round trip differs")
    result=fit_fixture.reconstruct(root/"fits",fitting_plan(root,plan),pulse,
                                   input_validator=validate_generated)
    result.update(state="D5_COST_PILOT_VERIFIED",new_scientific_worlds=1,
                  complete_fresh_pilot=True,confirmation=False,scientific_primary_decided=False,
                  scope="Fresh excluded cost/control pilot; full chronological fit, source, ledger, score-independent certificate and prediction review.",
                  model_limit="Finite constructed profile mechanism; no human or engine-admission conclusion.")
    return result
