"""Serial regression barrier for frozen V19 tests and already specified successors."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
import os,shutil,subprocess,time,zipfile
from ..v18_3.io import read,write,file_digest,now
from ..v18_3.native import ProcessClock


def validate_design(cfg,repo):
    names=cfg['tests']
    if not names or len(names)!=len(set(names)):raise ValueError('missing or repeated tests')
    for n in names:
        if not n.startswith('tests/test_v19') or not n.endswith('.py') or '..' in Path(n).parts or not (repo/n).is_file():raise ValueError('invalid test inventory')
    if 'tests/test_v19_tiny_torch.py' in names:raise ValueError('Torch regression has its own installed environment')
    for successor in cfg['successors']:
        if successor['design']['handler'] not in ('crossed-rules','forward-support'):raise ValueError('unadmitted successor')
        if successor['name'] not in ('G19-D-crossed-rules-1','G19-F-support-1'):raise ValueError('unadmitted namespace')
        if set(successor['design']['lineages']) & set(cfg['protected_lineages']):raise ValueError('protected lineage consumption')
    return True


def run(root,plan,pulse):
    import pytest
    from . import runtime as R
    cfg=plan['design'];campaign=Path(plan['campaign']);validate_design(cfg,R.REPO)
    reports=[]
    class Progress:
        def pytest_runtest_setup(self,item):pulse(phase='regression',test=item.nodeid)
        def pytest_runtest_logreport(self,report):
            if report.when=='call' or report.failed or report.skipped:reports.append(dict(test=report.nodeid,phase=report.when,outcome=report.outcome))
    with (root/'pytest.log').open('w',encoding='utf-8') as log,redirect_stdout(log),redirect_stderr(log):
        code=int(pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(root/'pytest-temp'),*cfg['tests']],plugins=[Progress()]))
    write(root/'GHOST_TESTS.json',dict(exit_code=code,reports=reports,log_sha256=file_digest(root/'pytest.log')))
    if code:raise ValueError('V19 regression failed; no successor admitted')
    pulse(phase='before-installed-torch-controls')
    owner=read(campaign/'TINY_TRAINING_CONFIG.json');python=Path(owner['python']);receipt=read(campaign/'TINY_SETTINGS.json')
    if file_digest(python)!=owner['python_sha256'] or file_digest(campaign/'TINY_SETTINGS.json')!=cfg['tiny_settings_sha256']:raise ValueError('installed Torch ownership changed')
    output=root/'torch-controls';output.mkdir();env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    clock=None;charged=0.;began=time.monotonic()
    with (root/'torch-tests.log').open('wb') as log:
        child=subprocess.Popen([str(python),'-B','-m','ghostscale.validation.soundingline.v19.validation_torch_worker','--output',str(output)],cwd=R.REPO,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        try:
            while child.poll() is None:
                status=read(output/'STATUS.json') if (output/'STATUS.json').exists() else {}
                if clock is None and (status.get('pid')==child.pid or status.get('parent_pid')==child.pid):
                    try:clock=ProcessClock(status['pid'])
                    except OSError:pass
                charged=max(charged,status.get('cpu_seconds',0),clock.seconds() if clock else 0.)
                pulse(phase='waiting-for-Torch-regression',child_pid=child.pid,child_cpu_seconds=charged)
                time.sleep(1)
        except BaseException:
            # Keep ownership while the bounded test finishes; never orphan a venv child.
            (output/'STOP').touch()
            child.wait()
            raise
        finally:
            status=read(output/'STATUS.json') if (output/'STATUS.json').exists() else {}
            charged=max(charged,status.get('cpu_seconds',0),clock.seconds() if clock else time.monotonic()-began)
            if clock:clock.close()
            write(root/'CHILD_ACCOUNTING.json',dict(cpu_seconds=charged,pid=child.pid,exit_code=child.returncode,parent_waited=True))
            pulse(phase='Torch-regression-exited',child_cpu_seconds=charged)
    if child.returncode or status.get('state')!='complete':raise ValueError('installed Torch regression failed; no successor admitted')
    if file_digest(campaign/'TINY_SETTINGS.json')!=cfg['tiny_settings_sha256']:raise ValueError('tests changed setting allowance')
    write(root/'VALIDATION.json',dict(at=now(),ghost_tests=sum(r['outcome']=='passed' and r['phase']=='call' for r in reports),
        ghost_skips=[r for r in reports if r['outcome']=='skipped'],torch_exit_code=child.returncode,torch_status=status,
        torch_log_sha256=file_digest(root/'torch-tests.log'),source_plan_sha256=file_digest(root/'PLAN.json'),passed=True))
    admission=dict(passed=True,sources=plan['sources'],tests='complete frozen V19 suite plus installed-Torch controls',validation_plan_sha256=file_digest(root/'PLAN.json'),validation_sha256=file_digest(root/'VALIDATION.json'))
    pending=[]
    for successor in cfg['successors']:
        name=successor['name'];design=successor['design'];source=campaign/('source-'+name);packet=campaign/name
        pulse(phase='admit-predeclared-successor',successor=name)
        R.freeze(source,packet,design,admission)
        inputs=root/'inputs'/name
        if design.get('input_files'):
            for n,h in design['input_files'].items():
                if file_digest(inputs/n)!=h:raise ValueError('successor input changed')
            shutil.copytree(inputs,packet/'inputs')
        for suffix in ('','-replay','-portable'):
            dest=campaign/(name+suffix);use=source
            if suffix:
                dest.mkdir()
                for n in ('PLAN.json','SOURCE.zip'):shutil.copyfile(packet/n,dest/n)
                if design.get('input_files'):shutil.copytree(inputs,dest/'inputs')
            if suffix=='-portable':
                use=campaign/('portable-'+name)/'source';use.mkdir(parents=True)
                with zipfile.ZipFile(packet/'SOURCE.zip') as z:
                    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise ValueError('unsafe archive path')
                    z.extractall(use)
            pending.append(dict(root=str(dest),source=str(use),plan_sha256=file_digest(dest/'PLAN.json')))
    queue=read(campaign/'QUEUE.json')
    if any(j['root']==k['root'] for j in pending for k in queue['jobs']):raise ValueError('duplicate successor job')
    queue['jobs'].extend(pending);write(campaign/'QUEUE.json',queue,immutable=False)
    write(root/'ADMITTED.json',dict(at=now(),jobs=[Path(j['root']).name for j in pending],queue_sha256=file_digest(campaign/'QUEUE.json')))
    return dict(controls={'all_frozen_regressions_passed':True,'installed_torch_passed':True,'source_bound_successors':True},
        tests=sum(r['outcome']=='passed' and r['phase']=='call' for r in reports),successors=[Path(j['root']).name for j in pending],
        scope='engineering validation and predeclared admission only; no new scientific finding')
