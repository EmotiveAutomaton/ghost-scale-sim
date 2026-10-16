from pathlib import Path
import shutil
import pytest
from runners import queue_v23_families as queue, admit_v23_repair as admission
from runners.watch_v23_events import owner, OwnerBusyError
from ghostscale.validation.soundingline.v23.common import digest


def test_real_native_lock_contention_retries_without_releasing_owner(tmp_path):
    state=tmp_path/'admission'
    def attempt():
        with owner(state):return {'state':'ENTERED'}
    with owner(state):
        result=queue.admission_step(tmp_path,attempt)
        assert result==dict(state='RUNNING',phase='WAITING_FOR_ADMISSION_OWNER')
        with pytest.raises(OwnerBusyError):attempt()
    assert queue.admission_step(tmp_path,attempt)=={'state':'ENTERED'}


@pytest.mark.parametrize('kind',['file_permission','wrong_owner','science'])
def test_retry_does_not_hide_other_failures(tmp_path,kind):
    error={'file_permission':PermissionError(13,'file denied'),
           'wrong_owner':OwnerBusyError(tmp_path/'another-service'),
           'science':ValueError('invalid evidence')}[kind]
    def fail():raise error
    with pytest.raises(type(error)):queue.admission_step(tmp_path,fail)


def source_fixture(tmp_path):
    repository=Path(__file__).resolve().parents[1]
    relative='ghostscale/validation/soundingline/v23/'
    names=['common','registry','dispatch','ledger','checker','statements','makers','checker_recipes','checker_new']
    frozen=tmp_path/'frozen';current=tmp_path/'current'
    for folder in (frozen,current):
        (folder/relative).mkdir(parents=True)
        for name in names:shutil.copyfile(repository/(relative+name+'.py'),folder/(relative+name+'.py'))
    plan=dict(source_archive=str(frozen.with_suffix('.zip')),
              sources={relative+n+'.py':digest(frozen/(relative+n+'.py')) for n in names})
    return plan,current,current/(relative+'makers.py')


def test_n7_profile_allows_only_unrelated_population_function_changes(tmp_path):
    plan,current,maker=source_fixture(tmp_path)
    maker.write_text(maker.read_text().replace('logp = np.zeros(len(weights))','logp = np.zeros(len(weights)) # population-only'),encoding='utf-8')
    admission.verify_profile_sources(plan,'N7',current)
    maker.write_text(maker.read_text().replace('logp = np.zeros(len(weights))','logp = np.ones(len(weights))'),encoding='utf-8')
    admission.verify_profile_sources(plan,'N7',current)


@pytest.mark.parametrize('change',['recipe','satisfaction','import','top_level','caller','frozen'])
def test_n7_profile_rejects_changed_dependencies_or_frozen_evidence(tmp_path,change):
    plan,current,maker=source_fixture(tmp_path)
    text=maker.read_text()
    if change=='recipe':text=text.replace('maximum=2','maximum=3')
    elif change=='satisfaction':text=text.replace('need_scale=1.','need_scale=2.')
    elif change=='import':text=text.replace('from itertools import combinations, product','from itertools import combinations, permutations as product')
    elif change=='top_level':text+='\nrecipe_library.cache_clear()\n'
    elif change=='caller':
        caller=maker.with_name('statements.py');caller.write_text(caller.read_text()+'\nchanged=True\n')
    else:
        frozen=Path(plan['source_archive']).with_suffix('')/'ghostscale/validation/soundingline/v23/makers.py'
        frozen.write_text(frozen.read_text()+'\nchanged=True\n');text+='\n# changed bytes\n'
    maker.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError,match='source changed'):admission.verify_profile_sources(plan,'N7',current)
