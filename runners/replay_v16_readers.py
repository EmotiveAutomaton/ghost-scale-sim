"""Re-execute retained public reader requests in the guarded separate process.

This checks reader access and deterministic prediction identity. It does not claim
to regenerate world rollouts or to retroactively change the original process layout.
"""
import argparse
from pathlib import Path
from ghostscale.validation.soundingline.v16.runtime import REPO,supervisor,remaining_seconds
from ghostscale.validation.soundingline.v16.records import read,write,digest,file_digest,now
from ghostscale.validation.soundingline.v16.reader_process import ReaderProcess


def recursive_requests(value):
    if isinstance(value,dict):
        if set(value)=={"kind","public","options","result"}:
            yield value
        else:
            for nested in value.values():
                yield from recursive_requests(nested)
    elif isinstance(value,list):
        for nested in value:
            yield from recursive_requests(nested)


def requests(root,row):
    uid=row["unit_id"]
    public=read(root/"public"/f"{uid}.json")
    prediction=read(root/"predictions"/f"{uid}.json")
    def frame(kind,public,result,**options):
        return {"kind":kind,"public":public,"options":options,"result":result}
    if row.get("unit_kind")=="inquiry":
        for path in sorted((root/"predictions").glob(f"{uid}*.json")):
            yield from recursive_requests(read(path))
    elif "prediction" in prediction:
        expected=prediction["prediction"]
        yield frame("native",public,expected,reader="maker",reader_seed=expected["reader_seed"])
    elif row["card_id"]=="K01":
        yield frame("craft",public,prediction["arms"])
    elif "ablation_observation" in prediction:
        for name,result in prediction["arms"].items():
            yield frame("reading",prediction["ablation_observation"] if name=="without-query" else public,
                        result,strategy="maker" if name=="without-query" else name)
    elif row["unit_kind"]=="self-trajectory":
        for index in range(len(row["private"]["rounds"])):
            round_record=read(root/"predictions"/f"{uid}-round-{index}.json")
            for name,result in round_record["arms"].items():
                yield frame("trajectory",round_record["public_inputs"][name],result,strategy=name)
        for name,result in prediction["final_predictions"].items():
            yield frame("trajectory",prediction["final_public_inputs"][name],result,strategy=name)
    else:
        for name,result in prediction["arms"].items():
            family=row["unit_kind"]
            options={}
            if family=="opportunity":
                options["model"]="latent-menu" if name=="without-probe" else name
            elif family=="self":
                options["strategy"]="self-model" if name=="memory-only" else name
            yield frame(family,public["arms"][name],result,**options)


def execute(root,output,heartbeat,packets):
    # Each source lock is verified before any replay. The process-boundary extension
    # has separate tested provenance and leaves all original scientific source intact.
    locks={}
    for packet in packets:
        record=read(root/"packets"/f"{packet}.json")
        for path,expected in record["identity"]["files"].items():
            if file_digest(REPO/path)!=expected:
                raise ValueError(f"scientific source changed: {path}")
        locks[packet]=record["packet_hash"]
    source_identity={str(path.relative_to(REPO)).replace("\\","/"):file_digest(path)
                     for path in [Path(__file__).resolve(),REPO/"runners/v16_reader_worker.py",
                                  REPO/"ghostscale/validation/soundingline/v16/reader_process.py"]}
    identity={"packets":locks,"source_hashes":source_identity,
              "admission_receipt_sha256":file_digest(root/"inquiry-setup/ADMISSION.json")}
    write(output/"REPLAY_IDENTITY.json",identity)
    completed,total_requests=0,0
    counts={}
    unit_hashes={}
    with ReaderProcess(output/"public/reader-workspace") as reader:
        for packet in packets:
            count=0
            for path in sorted((root/packet).glob("**/units/*_points.json")):
                if remaining_seconds(root)<=0:
                    return {"execution_state":"checkpointed","completed_units":completed,"campaign_complete":False}
                row=read(path)
                if row["packet_hash"]!=locks[packet]:
                    raise ValueError("raw unit packet mismatch")
                unit_root=path.parent.parent
                receipt_path=output/"units"/f"{row['unit_id']}_points.json"
                request_list=list(requests(unit_root,row))
                expected_identity={"source_identity_hash":digest(identity),"raw_sha256":file_digest(path),
                                   "requests_sha256":digest(request_list),"n_requests":len(request_list)}
                if receipt_path.exists():
                    previous=read(receipt_path)
                    if previous["identity"]!=expected_identity or previous["instrument_state"]!="valid":
                        raise ValueError("saved replay receipt differs")
                else:
                    for request in request_list:
                        actual=reader.request(request["kind"],request["public"],**request["options"])
                        if actual!=request["result"]:
                            write(output/"failures"/f"{row['unit_id']}.json",
                                  {"unit_id":row["unit_id"],"request":request,"actual":actual,"observed_at":now()})
                            raise ValueError("separate-process reader prediction mismatch; failure retained")
                    write(receipt_path,{"identity":expected_identity,"instrument_state":"valid","completed_at":now()})
                unit_hashes[row["unit_id"]]=file_digest(receipt_path)
                completed+=1
                count+=1
                total_requests+=len(request_list)
                heartbeat(completed_units=completed,reader_requests=total_requests,packet_id=packet)
            counts[packet]=count
    report={"execution_state":"completed","instrument_state":"valid","packet_units":counts,
            "n_units":completed,"n_requests":total_requests,"exact_prediction_agreement":True,
            "original_process_layout":"native/acquisition/reading/behavior: pure serialized API in evaluator process; inquiry: separate worker",
            "reader_access_replay":"fixed trusted scientific code; post-import file/process/network operations denied",
            "general_hostile_native_sandbox":False,"full_rollout_replay":False,"campaign_complete":False,
            "receipt_manifest_sha256":digest(unit_hashes)}
    write(output/"RAW_MANIFEST.json",{"unit_receipts":unit_hashes})
    write(output/"REPLAY.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",type=Path,default=REPO/"results/v16")
    parser.add_argument("--packet",action="append")
    args=parser.parse_args()
    packets=args.packet or ["native-fixture-1","k01-scout-1","reading-scout-1","behavior-scout-1","inquiry-scout-1"]
    with supervisor(args.root,"reader-access-replay") as heartbeat:
        result=execute(args.root,args.root/"reader-access-replay-1",heartbeat,packets)
        heartbeat(execution_state=result["execution_state"],result=result)
    print(result)


if __name__=="__main__":
    main()
