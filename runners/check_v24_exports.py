"""Retained twin export equivalence; no generated maker or outcome."""
import argparse
from pathlib import Path
import sys
import time

from ghostscale.validation.soundingline.v23.common import read,write,digest,canonical
from ghostscale.validation.soundingline.v23 import runtime,ledger,engine_bridge
from ghostscale.validation.soundingline.v24.exact_export import ExactExporter


def run(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.process_time();runtime.below_normal();records=[]
    try:
        for plan_path,index_path in zip(args.plans,args.retained_indexes,strict=True):
            plan=read(plan_path)
            retained_index=read(index_path)
            for path in plan["extra_pythonpath"]:
                if path not in sys.path:sys.path.insert(0,path)
            # The separately pinned canonical runtime is not installed in .venv.
            runtime_dependencies=plan_path.resolve().parents[2]/"dependencies"
            sys.path.insert(0,str(runtime_dependencies))
            for name,sha in plan["external_source_files"].items():
                if digest(Path(name))!=sha:raise ValueError("retained engine source changed")
            schema,_=ledger.canonical_schema(plan["schema_directory"],plan["schema_pin"])
            engine_root=Path(plan["extra_pythonpath"][-1])/"tompathy_engine"
            engine_files={}
            for name,sha in plan["external_source_files"].items():
                path=Path(name)
                if path.is_relative_to(engine_root) and path.suffix in (".py",".yaml"):
                    engine_files[path.relative_to(engine_root).as_posix()]=sha
            folders=sorted(p.parent.parent for p in (plan_path.parent/"original").glob("*/*/maker-*/engine/reader.json"))
            if len(folders)<6:raise ValueError("incomplete retained export fixtures")
            indices=sorted({0,1,len(folders)//3,2*len(folders)//3,len(folders)-2,len(folders)-1})
            first=read(folders[0]/"engine/reader.json")
            # The old file roster omitted the question catalogue. Its bytes
            # were nevertheless covered by the engine implementation hash in
            # every retained export. Bind them explicitly in this new adapter.
            from tompathy_engine.admission import implementation_hash
            if implementation_hash()!=first["implementation_hash"]:
                raise ValueError("retained full engine fingerprint differs")
            supplementary={}
            for path in engine_root.rglob("*.yaml"):
                name=path.relative_to(engine_root).as_posix()
                if name not in engine_files:
                    supplementary[name]=digest(path);engine_files[name]=digest(path)
            before=time.process_time()
            exporter=ExactExporter(schema,engine_files,first["implementation_hash"])
            initialization=time.process_time()-before
            for index in indices:
                folder=folders[index]
                relative=(folder/"engine/reader.json").relative_to(plan_path.parent/"original").as_posix()
                if digest(folder/"engine/reader.json")!=retained_index[relative]:
                    raise ValueError("retained public export binding changed")
                old=read(folder/"engine/reader.json")
                native=engine_bridge.native_view(old,schema)
                outputs=[];cost={}
                for name,call in (("scalar",lambda:engine_bridge.export(native,schema)),
                                  ("cached",lambda:exporter.export(native))):
                    before=time.process_time();outputs.append(call());cost[name]=time.process_time()-before
                expected=canonical(old)
                if any(canonical(output)!=expected for output in outputs):
                    raise ValueError("export bytes or identity changed")
                before_hash=digest(folder/"engine/reader.json")
                if expected!=canonical(read(folder/"engine/reader.json")):
                    raise ValueError("retained input changed")
                records.append(dict(family=plan["family"],retained_maker_index=index,
                    complete_retained_exports=len(folders),records=len(old["records"]),
                    reader_sha256=before_hash,reader_hash=old["reader_hash"],
                    cpu_seconds=cost,exact_canonical_bytes=True))
                if time.process_time()-started>args.cpu_cap:raise TimeoutError("export engineering cap")
            exporter.verify_sources()
            # Canary rejection and copy isolation are acceptance controls.
            probe=engine_bridge.native_view(first,schema)
            original=canonical(probe)
            out=exporter.export(probe)
            out["records"][0]["record"]["unexpected"]="mutation"
            if canonical(probe)!=original:raise ValueError("export aliases input")
            try:exporter.export(probe,canaries=["Synthetic"])
            except ValueError:pass
            else:raise ValueError("truth canary not rejected")
            write(root/(plan["family"]+"_CHECK.json"),dict(
                retained_plan_sha256=digest(plan_path),engine_initialization_cpu_seconds=initialization,
                retained_file_index_sha256=digest(index_path),
                explicitly_added_catalogue_pins=supplementary,
                source_files_verified=len(engine_files),copy_isolation=True,canary_rejected=True))
        result=dict(state="RETAINED_EXPORT_EQUIVALENCE_VERIFIED",new_scientific_worlds=0,fixtures=records,
            full_grid_measured=False,canonical_bytes_identical=True,
            limits="Six fixed positions per retained partial twin world. Full 24-setting profiles, fresh identities and replay still required.")
        write(root/"COMPLETE.json",result)
    except BaseException as exc:
        write(root/"FAILED.json",dict(error=f"{type(exc).__name__}: {exc}"));raise
    finally:write(root/"COST.json",dict(cpu_seconds=time.process_time()-started))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--plans",type=Path,nargs="+",required=True)
    parser.add_argument("--retained-indexes",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cpu-cap",type=float,default=120)
    run(parser.parse_args())
