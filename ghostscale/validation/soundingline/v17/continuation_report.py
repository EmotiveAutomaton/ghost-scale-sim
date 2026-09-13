"""Deterministic final tables, illustrative selection and portable reader execution."""
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from ..v16.records import canonical,digest,file_digest,write
from . import observer as e, recipient as b

TYPES=("advantage","reversal","failure","plausible_history_wrong","memory_catches_exception",
       "recipient_model_wrong","extra_computation_wasted","surprising_downstream_edit")

def tags(item):
    case=item["case"];rows=item["rows"];family=case["family"];out=set()
    grouped=defaultdict(list)
    for row in rows: grouped[row["target"]].append(row)
    for group in grouped.values():
        losses=[r.get("brier_score",float(not r["task_success"])) for r in group]
        if max(losses)-min(losses)>.05: out.add("advantage")
        if any(not r["task_success"] for r in group): out.add("failure")
    if family=="E":
        artifact={r["method"]:r for r in rows if r["target"]=="unseen_choice_artifact"}
        if artifact["inverse_maker"]["brier_score"]>artifact["surface_continuation"]["brier_score"]+.01:out.add("reversal")
        if artifact["inverse_maker"]["compatible_histories"]>1 and not artifact["inverse_maker"]["task_success"]:out.add("plausible_history_wrong")
        if case["regime"]=="A":
            actual=case["private"]["realized_history"][3]
            hyp=dict(case["private"]["true_hypothesis"],method="primitive_search")
            primitive=e.forward(case["public"]["context"],hyp,3)
            if case["private"]["true_hypothesis"]["method"]=="episodic_adaptation" and actual["artifact"]!=primitive["artifact"]:
                cells=case["public"]["context"]["cells"];goal=case["private"]["true_hypothesis"]["goal"]
                target=sum(1<<x for x in (cells[:3] if goal==0 else cells[1:4]))
                if actual["artifact"]==target and primitive["artifact"]!=target:out.add("memory_catches_exception")
    elif family in ("A2","AP"):
        for group in grouped.values():
            method={r["method"]:r for r in group}
            for name,row in method.items():
                if "episodic_adaptation" in name and row["task_success"] and case["regime"].endswith("changed_constraint"):
                    rival=method.get(name.replace("episodic_adaptation","stitch_abstractions"))
                    if rival and not rival["task_success"]:out.add("memory_catches_exception")
    elif family=="B":
        posterior=b.infer_models(case["public"]["world"],case["public"]["calibration"])
        inferred=max(posterior,key=lambda x:x[1])[0]
        if inferred!=case["private"]["recipient_state"]:out.add("recipient_model_wrong")
    elif family=="C":
        for group in grouped.values():
            lookup={r["method"]:r for r in group}
            if "fixed_inverse" in lookup and "empirical" in lookup:
                x,y=lookup["fixed_inverse"],lookup["empirical"]
                if x["brier_score"]>=y["brier_score"] and x["costs"]["repeat_online"]>y["costs"]["repeat_online"]:
                    out.add("extra_computation_wasted")
    elif family=="D":
        lookup={r["method"]:r for r in rows}
        x,y=lookup["local_edit"],lookup["dependency_aware"]
        if x["decision_regret"]>y["decision_regret"]+1e-12 and x["program"]!=y["program"] and x["collateral_parts_removed"]>0:out.add("surprising_downstream_edit")
    return out

def choose_examples(store):
    # Complete retained expansion data only: finite selection, no ranking by most
    # persuasive magnitude. Lowest scientific case hash within each observed type.
    chosen={};readers={}
    for item in store.iter_items(stage="expansion"):
        case=item["case"];identity=case["case_id"]
        kinds=tags(item)
        for kind in kinds:
            if kind not in chosen or identity<chosen[kind]["case"]["case_id"]: chosen[kind]=item
        if case["family"]=="E":
            for kind in kinds|{"family_"+case["regime"]}:
                if kind not in readers or identity<readers[kind]["case"]["case_id"]: readers[kind]=item
    status={kind:dict(state="present" if kind in chosen else "absent",
        case_id=chosen[kind]["case"]["case_id"] if kind in chosen else None,
        blind_reader_state="present" if kind in readers else "absent; evaluator example only" if kind in chosen else "absent")
        for kind in TYPES}
    return dict(rule="lowest case hash within each observed type and each E native family, on the completed expansion cohort only",
        status=status,evaluator_examples=chosen,reader_examples=readers)

def reader_bundle(root,selection,source_files):
    from runners.package_v17_observers import package
    unique={v["case"]["case_id"]:v for v in selection["reader_examples"].values()}
    if not unique:
        return dict(state="unavailable",reason="no completed observer case")
    items=[unique[k] for k in sorted(unique)]
    source=root/"reader-selection"
    source.mkdir(parents=True,exist_ok=True)
    write(source/"LOCK.json",dict(schema="v17.export-selection.1",design=dict(family="E"),
        source_files=source_files,selection_rule=selection["rule"],new_observations=0))
    payload=b"".join(canonical(i)+b"\n" for i in items)
    raw=source/"selected_points.jsonl"
    if raw.exists() and raw.read_bytes()!=payload:raise ValueError("retained reader selection differs")
    if not raw.exists():raw.write_bytes(payload)
    write(source/"INDEX.json",dict(chunks=[dict(path=raw.name,cases=len(items),sha256=file_digest(raw))]))
    write(source/"COMPLETION.json",dict(schema="v17.export-projection.1",index_sha256=file_digest(source/"INDEX.json"),
        cases=len(items),source_case_ids=sorted(unique),new_observations=0))
    output=root/"transfer"
    exported=package(source,output,limit=len(items))
    extracted=output/"extracted"
    with zipfile.ZipFile(output/"reader.zip") as z:
        for info in z.infolist():
            target=(extracted/info.filename).resolve()
            if extracted.resolve() not in target.parents:raise ValueError("unsafe archive member")
        z.extractall(extracted)
    requests=(extracted/"requests.jsonl").read_text()
    run=subprocess.run([sys.executable,"-B","consumer.py"],cwd=extracted,input=requests,
        text=True,capture_output=True,timeout=180,env=dict(os.environ,PYTHONPATH=""))
    if run.returncode:raise ValueError("portable reader failed: "+run.stderr[:400])
    expected={}
    with zipfile.ZipFile(output/"evaluator.zip") as z:
        for line in z.read("answers.jsonl").splitlines():
            answer=json.loads(line)
            expected[answer["task_id"]]=next(r["probabilities"] for r in answer["reference_rows"] if r["method"]=="inverse_maker")
    predictions=[json.loads(line) for line in run.stdout.splitlines()]
    if len(predictions)!=len(expected):raise ValueError("portable prediction count mismatch")
    for prediction in predictions:
        wanted=expected[prediction["task_id"]]
        if set(prediction["probabilities"])!=set(wanted) or any(not math.isclose(v,wanted[k],rel_tol=1e-11,abs_tol=1e-11) for k,v in prediction["probabilities"].items()):
            raise ValueError("portable predictions differ from source")
    (extracted/"PRIVATE-canary.txt").write_text("must not be read")
    probe=subprocess.run([sys.executable,"-B","consumer.py","--guard-probe"],cwd=extracted,capture_output=True,text=True,timeout=30)
    if probe.returncode or not json.loads(probe.stdout).get("private_read_denied"):raise ValueError("private reader boundary failed")
    receipt=dict(schema="v17.final-reader-portability.1",passed=True,requests=len(predictions),
        reader_sha256=file_digest(output/"reader.zip"),private_read_denied=True,selection_rule=selection["rule"],
        scope="fixed trusted Python consumer, not a hostile-code sandbox")
    write(output/"PORTABILITY.json",receipt)
    return dict(state="executed_and_verified",**receipt)

def finish(root,store,plan,window):
    root=Path(root);out=root/"closeout";out.mkdir(parents=True,exist_ok=True)
    verification=store.verify_all()
    write(out/"VERIFICATION.json",verification)
    surfaces=[]
    for stage in ("expansion","robustness"):
        for design in plan["initial_expansions"]:
            surface=store.surface(stage,design)
            if surface["comparisons"]:surfaces.append(surface)
    for contrast in plan["confirmation"]:
        surface=store.surface("confirmation",contrast)
        if surface["comparisons"]:surfaces.append(surface)
    write(out/"COMPARISONS.json",dict(schema="v17.final-comparisons.1",surfaces=surfaces,
        confirmation=store.get_snapshot("CONFIRMATION"),counts=store.counts()))
    selection=choose_examples(store)
    write(out/"ILLUSTRATIONS.json",selection)
    reader=reader_bundle(out,selection,plan["source_files"])
    confirmation=store.get_snapshot("CONFIRMATION")
    counts=store.counts()
    core_complete=counts["failed_units"]==0 and counts["apparatus_issue_rows"]==0 and confirmation is not None and not confirmation["incomplete"] and reader["state"]=="executed_and_verified"
    lines=["# V17 continuation results","",
      "Can inferred procedures, concrete memories, recipient models and selective computation improve useful prediction?","",
      "This is a constructed-world mechanism study. The continuation adds targeted expansion, fixed fresh confirmation, pooled acquisition, assembly recipients and partial sharing. No human-theory conclusion follows.","",
      "The table lists each selected fresh comparison, its average paired gain and Holm decision. Gain is rival minus method Brier score for the first two comparisons, and method minus rival task-success proportion for revision. Constructor averages are the independent units.","",
      "| Comparison | Constructor clusters | Mean gain | Adjusted probability bound | Decision |","|---|---:|---:|---:|---|"]
    for row in (confirmation or {}).get("entries",[]):
        lines.append("| "+row["id"]+" | "+str(row.get("n_constructor_clusters","incomplete"))+" | "+str(row.get("mean_gain","unavailable"))+" | "+str(row["holm_adjusted_p"])+" | "+row["claim_status"]+" |")
    lines+=["","All other surfaces are descriptive. They include costs, unique constructions, missing outputs and structural limits. A failed superiority test is not equivalence. The fixed Hoeffding bound requires independent bounded constructor averages; its conservatism is retained.","",
        "Illustrations use the lowest case hash within each observed outcome type on the completed expansion cohort. Unavailable blind-reader types are explicitly absent, even when an evaluator example exists.","",
        "The initial V17 screens remain archived separately. This report covers the authorized continuation window and does not overwrite them.","",
        "Commissioned executable scope complete: "+str(core_complete)+". Final agent review, public filing and push must be recorded separately.",
        "", "Window: "+window["started_at"]+" through "+window["ends_at"]+".",
        "Retained counts: "+json.dumps(counts,sort_keys=True)+".",""]
    content="\n".join(lines)
    path=out/"RESULTS.md"
    if path.exists() and path.read_text()!=content:raise ValueError("retained results text differs")
    if not path.exists():path.write_text(content,encoding="utf-8",newline="\n")
    write(out/"TIME_ACCOUNT.json",dict(window=window,counts=counts,gpu_used=False,workers=1,
        scope="worker process CPU excludes child Stitch CPU; per-unit dependency receipts retain child measurements",
        setup="source-bound admission and launch receipts separate",recovery="failure/event ledgers retain every interruption",
        scientific_sampling_completed=True,public_filing_complete=False))
    return dict(schema="v17.continuation-completion.1",execution_complete=True,commissioned_executable_scope_complete=core_complete,
        public_filing_complete=False,window=window,counts=counts,verification=verification,reader=reader,
        outputs=["closeout/RESULTS.md","closeout/COMPARISONS.json","closeout/ILLUSTRATIONS.json",
                 "closeout/transfer/reader.zip","closeout/transfer/evaluator.zip","closeout/TIME_ACCOUNT.json","records.sqlite"],
        next_action="one transition-triggered agent review: verify results, complete write-through, archive, file and push; no polling of completed work")
