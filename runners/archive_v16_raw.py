"""Create verified portable chunks from explicitly named completed V16 packets."""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO
from ghostscale.validation.soundingline.v16.records import read,write,file_digest,now
from ghostscale.validation.soundingline.v16.record_integrity import source_locks
from ghostscale.validation.soundingline.v16.aggregate_science import registered_designs
from ghostscale.validation.soundingline.v16.raw_archive import manifest,create


def completed_source(root,name,packet):
    base=root/name
    if (base/"COMPLETION.json").exists():
        if read(base/"COMPLETION.json").get("execution_state")!="completed":
            raise ValueError("cannot archive an unfinished named packet as complete")
        return
    if name=="native-fixture-1":
        proof=read(root/"closeout-setup/aggregate-development-2/RECEIPT.json")
        if proof.get("instrument_state")!="valid" or len(list((base/"units").glob("*_points.json")))!=2:
            raise ValueError("native fixture archival scope lacks its known completed calculation")
        return
    if "-scout-" not in name:
        raise ValueError("unfinished allocation cannot enter the completed archive")
    designs=registered_designs(packet["identity"]["design"])
    if not designs:
        raise ValueError("missing scientific card inventory for archival scope")
    for card in designs:
        path=base/card/"COMPLETION.json"
        if not path.exists() or read(path).get("execution_state")!="completed":
            raise ValueError("scout card lacks a completed execution receipt: "+card)


def run(root,output,names):
    checks=source_locks(root,REPO)
    reports=[]
    for name in names:
        if "/" in name or "\\" in name or name in {".",".."}:
            raise ValueError("archive packet must be an exact local packet ID")
        packet_path=root/"packets"/(name+".json")
        packet=read(packet_path)
        completed_source(root,name,packet)
        base=root/name
        code=[REPO/path for path in packet["identity"]["files"]]
        common=code+[packet_path,root/"CAMPAIGN.json",REPO/"pyproject.toml",REPO/"uv.lock",
                     REPO/"docs/versions/v16-acquired-craft/CODING_PACKAGE.md",
                     Path(__file__).resolve(),REPO/"ghostscale/validation/soundingline/v16/raw_archive.py"]
        children=[path for path in sorted(base.iterdir()) if path.is_dir() and (path/"COMPLETION.json").exists()
                  and "card_id" in read(path/"COMPLETION.json")]
        portions=[(path.name,[*path.rglob("*"),*base.glob("*")]) for path in children] if children else [("packet",list(base.rglob("*")))]
        if children:
            residual=[path for path in base.rglob("*") if path.is_file() and path.parent!=base and
                      not any(path.is_relative_to(child) for child in children)]
            if residual:
                portions.append(("packet-extra",residual))
        for label,files in portions:
            selected=sorted(set(common+[path for path in files if path.is_file()]))
            plan=manifest(REPO,selected,scope={"packet_id":name,"portion":label,"packet_hash":packet["packet_hash"],
                "evidence_state":"Preserved original invalid inquiry remains invalid; archival byte verification is not scientific admission" if name=="inquiry-scout-1" else "Original scientific/control validity labels retained"})
            receipt=create(REPO,output/name/label/"attempt-1",plan)
            report={"packet_id":name,"portion":label,"member_count":receipt["verified_members"],
                "uncompressed_bytes":receipt["uncompressed_bytes"],"archive_identity":receipt["archive_identity"],
                "relative_archive":name+"/"+label+"/attempt-1/raw.zip","plan_sha256":receipt["plan_sha256"]}
            reports.append(report)
            print(report,flush=True)
    receipt={"execution_state":"completed","instrument_state":"valid","recorded_at":now(),"source_checks":checks,
        "chunks":reports,"complete_campaign_archive":False,
        "scope":"Only the explicitly named completed source packets; setup, later science and final documentary coverage require the final archive inventory"}
    write(output/"RECEIPTS"/("batch-"+file_digest(root/"CAMPAIGN.json")[:12]+"-"+str(len(list((output/"RECEIPTS").glob("*.json"))))+".json"),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=REPO/"results/v16")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--packet",action="append",required=True)
    args=parser.parse_args()
    result=run(args.root,args.output,args.packet)
    print({"completed_chunks":len(result["chunks"]),"complete_campaign_archive":False})


if __name__=="__main__":
    main()
