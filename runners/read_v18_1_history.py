"""Blind consumer: this CLI opens only the supplied reader packet, never evaluator truth."""
import argparse
import gzip
import json
from pathlib import Path
from ghostscale.validation.soundingline.v16.records import canonical,file_digest,write
from ghostscale.validation.soundingline.v18_1.g4 import predict


def consume(reader,output,method):
    packet=json.loads(reader.read_bytes())
    if set(packet)!={'schema','cases'} or packet['schema']!='v18.1.blind-reader-packet.1':raise ValueError('reader packet required')
    rows=[]
    for case in packet['cases']:
        if set(case)!={'case_id','tiers'}:raise ValueError('private fields in reader case')
        for tier in case['tiers']:
            if tier['case_id']!=case['case_id']:raise ValueError('case identity mismatch')
            rows.append(dict(case_id=case['case_id'],tier=tier['tier'],**predict(canonical(tier),method)))
    output.mkdir(parents=True,exist_ok=True)
    raw=output/'predictions_points.json.gz'
    with raw.open('xb') as stream:stream.write(gzip.compress(canonical(rows),mtime=0))
    receipt=dict(reader_sha256=file_digest(reader),predictions_sha256=file_digest(raw),requests=len(rows),method=method,
        evaluator_access=False,scope='no accuracy claim; scoring requires the separate evaluator')
    write(output/'CONSUMER.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reader',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--method',default='inverse-history')
    args=parser.parse_args();print(json.dumps(consume(args.reader,args.output,args.method)))


if __name__=='__main__':main()
