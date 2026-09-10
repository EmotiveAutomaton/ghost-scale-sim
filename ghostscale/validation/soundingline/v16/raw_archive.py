"""Streaming, independently verified raw archive chunks with retained failures."""
import hashlib
import os
from pathlib import Path
import zipfile
from .records import read,write,canonical,digest,now


def stream_digest(stream):
    result=hashlib.sha256()
    size=0
    while block:=stream.read(1024*1024):
        result.update(block)
        size+=len(block)
    return result.hexdigest(),size


def identity(path):
    with path.open("rb") as stream:
        sha,size=stream_digest(stream)
    return {"sha256":sha,"bytes":size}


def manifest(repo, paths, *, scope):
    root=repo.resolve()
    files={}
    for path in paths:
        resolved=path.resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file() or path.is_symlink():
            raise ValueError("raw archive member escapes the declared repository or is not a regular file")
        name=resolved.relative_to(root).as_posix()
        if name in files:
            raise ValueError("duplicate raw archive member")
        files[name]=identity(resolved)
    if not files or not scope:
        raise ValueError("raw archive needs a finite nonempty inventory and scope")
    return {"schema_version":"v16.raw-chunk-plan.1","scope":scope,"files":dict(sorted(files.items())),
            "member_count":len(files),"uncompressed_bytes":sum(row["bytes"] for row in files.values())}


def verify(path, plan):
    checked=0
    with zipfile.ZipFile(path,"r") as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(plan["files"])|{"ARCHIVE_MEMBER_MANIFEST.json"}:
            raise ValueError("raw archive omitted, duplicated or added members")
        if archive.read("ARCHIVE_MEMBER_MANIFEST.json")!=canonical(plan)+b"\n":
            raise ValueError("raw archive embedded inventory changed")
        for name,expected in plan["files"].items():
            with archive.open(name,"r") as stream:
                sha,size=stream_digest(stream)
            if {"sha256":sha,"bytes":size}!=expected:
                raise ValueError("raw archive member content differs: "+name)
            checked+=1
    return {"instrument_state":"valid","verified_members":checked,"all_expected_member_bytes_verified":True,
            "archive_identity":identity(path),"plan_sha256":digest(plan)}


def create(repo, directory, plan):
    """An interrupted/failed directory is never overwritten; use a new attempt."""
    if directory.exists():
        receipt_path=directory/"RECEIPT.json"
        if not receipt_path.exists():
            raise ValueError("incomplete raw archive attempt retained; choose a new attempt directory")
        receipt=read(receipt_path)
        if receipt["plan_sha256"]!=digest(plan):
            raise ValueError("completed raw archive belongs to a different inventory")
        proof=verify(directory/"raw.zip",plan)
        if proof["archive_identity"]!=receipt["archive_identity"]:
            raise ValueError("completed raw archive bytes changed")
        return receipt
    directory.mkdir(parents=True)
    write(directory/"member_manifest_points.json",plan)
    try:
        with zipfile.ZipFile(directory/"raw.zip","x",compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
            archive.writestr("ARCHIVE_MEMBER_MANIFEST.json",canonical(plan)+b"\n")
            for name,expected in plan["files"].items():
                source=(repo/name).resolve()
                if not source.is_relative_to(repo.resolve()) or identity(source)!=expected:
                    raise ValueError("raw source changed after archive inventory: "+name)
                archive.write(source,arcname=name)
        proof=verify(directory/"raw.zip",plan)
        result={"execution_state":"completed",**proof,"created_at":now(),"scope":plan["scope"],
                "uncompressed_bytes":plan["uncompressed_bytes"],
                "retention":"Preserve this chunk, its inventory and receipt through final handoff; no automatic deletion",
                "complete_campaign_archive":False}
        write(directory/"RECEIPT.json",result)
        return result
    except BaseException as error:
        write(directory/"FAILURE.json",{"execution_state":"failed","recorded_at":now(),
              "plan_sha256":digest(plan),"error":type(error).__name__+": "+str(error),"partial_archive_preserved":True})
        raise
