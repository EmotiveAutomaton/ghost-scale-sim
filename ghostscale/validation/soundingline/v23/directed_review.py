"""One bounded retained-data verification repair; never draws a scientific case."""
import ast
from pathlib import Path
from .common import read,write,digest
from . import directed_common

def producer_signature(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.name!='reconstruct']
    return ast.dump(tree,include_attributes=False)

def validate(extras,family,repo):
    from .family import verify_packet
    from .extension_contract import CONDITIONAL_MODULES
    source=Path(extras['retained_packet']);plan,summary=verify_packet(source)
    if plan['family']!=family or plan['task']!='conditional-batch':raise ValueError('wrong retained directed batch')
    if digest(source/'COMPLETE.json')!=extras['retained_complete_sha256'] or digest(source/'REVIEW_FAILURE.json')!=extras['retained_review_failure_sha256']:
        raise ValueError('retained review binding differs')
    if read(source/'REVIEW_FAILURE.json')['error']!="TimeoutError('reserved independent review cap reached')":raise ValueError('only resource-only independent-review repair is admitted')
    if not read(source/'original/CONTROLS.json')['passed']:raise ValueError('original controls failed')
    for module in CONDITIONAL_MODULES[family]:
        relative='ghostscale/validation/soundingline/v23/'+module+'.py'
        if module=='directed_common':
            original=Path(plan['source_archive']).with_suffix('')/relative
            if digest(original)!=plan['sources'][relative] or producer_signature(original)!=producer_signature(Path(repo)/relative):raise ValueError('retained producer changed')
        elif digest(Path(repo)/relative)!=plan['sources'][relative]:raise ValueError('retained scientific model changed')
    return source,plan

def run(root,plan,pulse):
    repo=Path(__file__).resolve().parents[4]
    source,original=validate(plan,plan['family'],repo)
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    result=directed_common.reconstruct(source/'original',original,lambda:pulse(phase='retained-independent-review',family=plan['family']))
    write(root/'SUMMARY.json',dict(state='DIRECTED_RETAINED_REVIEW_VERIFIED',family=plan['family'],
        retained_complete_sha256=digest(source/'COMPLETE.json'),retained_plan_sha256=digest(source/'PLAN.json'),
        original_failed_review_preserved=True,new_scientific_draws=0,review=result))
