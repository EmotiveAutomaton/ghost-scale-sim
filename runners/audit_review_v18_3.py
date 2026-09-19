"""Independent raw-to-mean checks for the descriptive V18.3 review."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import zipfile
from ghostscale.validation.soundingline.v18_3.io import read,write,file_digest,canonical


def audit(root):
    plan=read(root/'PLAN.json');checks=0
    assert file_digest(root/'SOURCE.zip')==plan['source_archive_sha256']
    with zipfile.ZipFile(root/'SOURCE.zip') as archive:
        for name,sha in plan['sources'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==sha
    def raw(name):
        summary=read(root/(name+'.json'));path=root/(name+'_points.json.gz')
        assert file_digest(path)==summary['raw_sha256']
        return summary,json.loads(gzip.decompress(path.read_bytes()))
    def check(values,entry):
        nonlocal checks
        assert len(values)==entry['n']
        assert abs(math.fsum(values)/len(values)-entry['mean'])<1e-10
        checks+=1
    summary,rows=raw('CALIBRATION')
    for condition,metrics in summary['fixed_instrument_coverage'].items():
        selected=[r for r in rows['coverage'] if r['condition']==condition]
        for metric,entry in metrics.items():
            values=[[r[metric] for r in selected if r['lineage']==i] for i in sorted({r['lineage'] for r in selected})]
            check([math.fsum(v)/len(v) for v in values],entry)
    for key,metrics in summary['neural_reliability'].items():
        condition,method=key.split('|');selected=[r for r in rows['reliability'] if r['condition']==condition and r['method']==method]
        for metric,entry in metrics.items():
            values=[[r[metric] for r in selected if r['lineage']==i] for i in sorted({r['lineage'] for r in selected})]
            check([math.fsum(v)/len(v) for v in values],entry)
    for condition in ('in-support','new-combinations','new-queries','both-new'):
        for metric in ('expected_absolute_calibration_gap','overconfidence'):
            assert abs(summary['neural_reliability'][condition+'|exact'][metric]['mean']-summary['neural_reliability'][condition+'|intervention-summary'][metric]['mean'])<1e-10
    summary,rows=raw('NEURAL_FACTORIAL')
    for key,effect in summary['effects'].items():
        selected=[r for r in rows if '|'.join(str(r[k]) for k in ('family','condition','method','baseline'))==key]
        values=[[r['value'] for r in selected if r['lineage']==i] for i in sorted({r['lineage'] for r in selected})]
        check([math.fsum(v)/len(v) for v in values],effect['pooled'])
        for cell,entry in effect['cells'].items():check([r['value'] for r in selected if r['cell']==int(cell)],entry)
    summary,rows=raw('SOURCE_UPTAKE');grouped={}
    for case in rows:
        for row in case['rows']:
            key=canonical({k:case[k] for k in ('mode','roots','copies')}|{'method':row['method']}).decode()
            group=grouped.setdefault(key,dict(valid=[],initial_success=[],corrected_success=[],practice_actions=[],execution_actions=[],counterfactual_check_work=[]))
            group['valid'].append(float(row['valid']))
            if row['valid']:
                for metric,phase,field in (('initial_success','initial','success'),('corrected_success','corrected','success'),('practice_actions','initial','practice_actions'),('execution_actions','initial','execution_actions')):group[metric].append(float(row[phase][field]))
                group['counterfactual_check_work'].append(float(row.get('counterfactual_check_work',0)))
    assert len(rows)==summary['assigned_cases']
    for key,metrics in grouped.items():
        for metric,values in metrics.items():
            if values:check(values,summary['cells'][key][metric])
            else:assert summary['cells'][key][metric]['unavailable']
    ties=read(root/'COMPRESSION_TIES.json')['rows']
    assert len(ties)==1920
    for r in ties:
        assert 1<=r['numerically_optimal_codes']<=r['candidates']
        assert r['best_new_loss_among_ties']-1e-10<=r['selected_new_loss']<=r['worst_new_loss_among_ties']+1e-10
    write(root/'INDEPENDENT_REAGGREGATION.json',dict(passed=True,means_checked=checks,compression_rows_checked=len(ties),source_manifest_checked=True,equivalent_calibration_checked=True,
        files={p.name:file_digest(p) for p in sorted(root.iterdir()) if p.is_file()},auditor_sha256=file_digest(__file__),
        scope='all review coverage/calibration/uptake means and neural pooled/cell means; source/raw hashes; compression range consistency. Does not rederive factorial bootstrap intervals, native coverage probabilities, or exhaustively replay compression ranges.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();audit(a.root)
