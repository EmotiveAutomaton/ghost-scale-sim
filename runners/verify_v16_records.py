"""Verify retained V16 bytes and sampling identities without scoring new worlds."""
import argparse
from pathlib import Path
import time
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import write,file_digest,now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks,raw_integrity

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=REPO/"results/v16")
    parser.add_argument("--source-repo",type=Path,default=REPO)
    parser.add_argument("--git-ref",choices=["HEAD","origin/main"])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter();cpu=time.process_time()
    sources=source_locks(args.root,args.source_repo,args.git_ref)
    integrity=raw_integrity(args.root,sources["packets"])
    record={"execution_state":"completed","evidence_scope":"operational integrity","recorded_at":now(),
        "source_checks":sources,"retention_checks":integrity,"elapsed_seconds":time.perf_counter()-started,
        "parent_cpu_seconds":time.process_time()-cpu,"campaign_complete":False,
        "verifier_sources":{str(path.relative_to(REPO)).replace("\\","/"):file_digest(path) for path in [
            Path(__file__).resolve(),REPO/"ghostscale/validation/soundingline/v16/record_integrity.py"]}}
    write(args.output,record)
    print({key:integrity[key] for key in ["instrument_state","raw_files_checked","raw_bytes_checked","retained_unit_records"]})

if __name__=="__main__":
    main()
