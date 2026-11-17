"""One finite resolution-healing gate per family; preserve original fixture seeds."""
from pathlib import Path
from .common import digest


def validate(setting,dest,pulse=lambda **kw:None):
    from . import w_hidden_validation_r1,q_hidden_validation_r1
    module=w_hidden_validation_r1 if setting['family']=='W' else q_hidden_validation_r1
    c=Path(setting['campaign']);rows=[]
    for config in setting['configurations']:
        out=dest/config['parent'];out.mkdir(parents=True,exist_ok=True)
        result=module.validate(config['setting'],out,pulse)
        retained=c/'jobs'/config['parent']/'output'
        checked=[]
        for path in out.glob('*_points.npz'):
            old=retained/path.name
            if old.exists():
                assert digest(path)==digest(old),'original fixture changed'
                checked.append(dict(file=path.name,sha256=digest(path)))
        rows.append(dict(parent=config['parent'],setting=config['setting'],result=result,
                         retained_path_checks=checked))
        if not result['passed']:
            return dict(passed=False,configurations=rows,production_admitted=False,
                        further_automatic_repair=False)
    return dict(passed=True,configurations=rows,production_admitted=False,
                next_gate='Measured full production profile before scientific admission')
