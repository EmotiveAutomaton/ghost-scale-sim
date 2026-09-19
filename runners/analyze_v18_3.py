"""Descriptive architecture contrasts with coefficient draws kept as clusters."""
import argparse
import gzip
from itertools import combinations
from pathlib import Path
import numpy as np
from ghostscale.validation.soundingline.v18_3 import world as W
from ghostscale.validation.soundingline.v18_3.io import read,write,canonical,file_digest
from ghostscale.validation.soundingline.v18_3.runtime import load,stats
from ghostscale.validation.soundingline.v18_3.verify import metrics

FACTORS=('decision rule','acquisition interference','endogenous opportunities','source dependence')


def contrasts(values):
    values=np.asarray(values,float);factors=np.asarray(W.FACTORS);result={}
    if values.shape[-1]!=16:raise ValueError('complete sixteen-cell factorial required')
    for a,name in enumerate(FACTORS):
        result[name]=values[...,factors[:,a]==1].mean(-1)-values[...,factors[:,a]==0].mean(-1)
    for a,b in combinations(range(4),2):
        groups={}
        for x,y in ((0,0),(0,1),(1,0),(1,1)):
            groups[x,y]=values[...,(factors[:,a]==x)&(factors[:,b]==y)].mean(-1)
        result[FACTORS[a]+' x '+FACTORS[b]]=groups[1,1]-groups[1,0]-groups[0,1]+groups[0,0]
    return result


def selected(unit):
    rows=metrics(unit);by={};family=unit['family']
    if family=='A' and unit['control']!='ordinary':return []
    output=[]
    if family=='A':
        for purpose,metric,method,baseline in (('enactment','enactment','task-three-attempts','information-three-attempts'),('prediction','expected_loss','task','fixed')):
            arms={tags['method']:values for tags,values in rows if tags['purpose']==purpose}
            output.append((dict(family=family,mode=unit['mode'],purpose=purpose,metric=metric,comparison=method+' minus '+baseline),arms[method][metric]-arms[baseline][metric]))
    elif family=='B':
        arms={tags['method']:values for tags,values in rows}
        output.append((dict(family=family,condition=unit['condition'],metric='post_change_loss',comparison='selective-transition minus static'),arms['selective-transition']['post_change_loss']-arms['static']['post_change_loss']))
    elif family=='D':
        arms={tags['method']:values for tags,values in rows}
        for method in ('revision','mixture'):
            output.append((dict(family=family,kind=unit['kind'],order=unit['order'],metric='expected_loss',comparison=method+' minus fixed'),arms[method]['expected_loss']-arms['fixed']['expected_loss']))
    elif family=='H':
        for cardinality in (1,2,4,8):
            arms={tags['method']:values for tags,values in rows if tags['cardinality']==cardinality}
            for metric in ('old_loss','new_loss'):
                output.append((dict(family=family,cardinality=cardinality,metric=metric,comparison='exhaustive-flat minus observation-product'),arms['exhaustive-flat'][metric]-arms['observation-product'][metric]))
    return output


def factorial(roots):
    grouped={};points=[]
    for root in roots:
        complete=read(root/'COMPLETE.json')
        for block in complete['blocks']:
            for unit in load(root,block):
                for tags,value in selected(unit):
                    key=canonical(tags).decode();cell=unit['cell'];draw=unit['index']
                    if cell in grouped.setdefault(key,{}).setdefault(draw,{}):raise ValueError('duplicate factorial observation')
                    grouped[key][draw][cell]=float(value);points.append(dict(**tags,cell=cell,lineage=draw,value=float(value)))
    summary={}
    for key,draws in grouped.items():
        if any(set(cells)!=set(range(16)) for cells in draws.values()):raise ValueError('incomplete paired architecture support')
        values=np.array([[draws[draw][cell] for cell in range(16)] for draw in sorted(draws)])
        summary[key]=dict(pooled=stats(values.mean(1).tolist(),('factorial-pooled',key)),
            cells={str(cell):stats(values[:,cell].tolist(),('factorial-cell',key,cell)) for cell in range(16)},
            contrasts={name:stats(effect.tolist(),('factorial-effect',key,name)) for name,effect in contrasts(values).items()},
            cells_with_positive_mean=int(np.sum(values.mean(0)>1e-10)),cells_with_negative_mean=int(np.sum(values.mean(0)<-1e-10)))
    return dict(schema='v18.3.factorial.1',effects=summary,independent_unit='coefficient-draw lineage, repeated architecture conditions paired',
        factor_order=FACTORS,factor_cells=[list(x) for x in W.FACTORS],scope='descriptive fixed-factor discovery; D source factor is inactive'),points


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--roots',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary,points=factorial(a.roots);a.output.mkdir(exist_ok=True,parents=True)
    path=a.output/'FACTORIAL_points.json.gz';path.write_bytes(gzip.compress(canonical(points),mtime=0))
    summary['raw_sha256']=file_digest(path);summary['analyzer_sha256']=file_digest(Path(__file__));write(a.output/'FACTORIAL.json',summary)
