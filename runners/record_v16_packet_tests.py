"""Bind actual passed pytest results to a named packet's declared source files."""
import argparse
import importlib
from pathlib import Path
import xml.etree.ElementTree as ET
from ghostscale.validation.soundingline.v16.runtime import REPO,PACKAGE
from ghostscale.validation.soundingline.v16.records import write,file_digest,now


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--junit",type=Path,required=True)
    parser.add_argument("--runner",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not args.runner.isidentifier():
        raise ValueError("expected one local runner module name")
    module=importlib.import_module("ghostscale.validation.soundingline.v16."+args.runner)
    tests={}
    for case in ET.parse(args.junit).findall(".//testcase"):
        name=case.attrib["name"]
        if name in tests:
            raise ValueError("duplicate test identity")
        tests[name]="failed" if case.find("failure") is not None or case.find("error") is not None else (
                     "skipped" if case.find("skipped") is not None else "passed")
    if not tests or any(state!="passed" for state in tests.values()):
        raise ValueError("actual admission suite is empty or not passed")
    files=[PACKAGE/name for name in module.MODULES]+[REPO/"runners/v16_reader_worker.py"]
    write(args.output,{"recorded_at":now(),"evidence_scope":"fixture","tests":tests,
                       "junit_sha256":file_digest(args.junit),
                       "source_hashes":{str(path.relative_to(REPO)).replace("\\","/"):file_digest(path) for path in files}})
    print(f"Recorded {len(tests)} actual passed tests.")


if __name__=="__main__":
    main()
