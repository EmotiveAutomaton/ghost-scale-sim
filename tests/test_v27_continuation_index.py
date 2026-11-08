from ghostscale.validation.soundingline.v27 import continuation as C
from ghostscale.validation.soundingline.v27.common import write


def packet(c, name, card, *, probe=False, replaces=None, landed=True):
    path=c/'packets'/name
    write(path/'PLAN.json',dict(card=card,scientific_kind='probe' if probe else 'study',replaces_card=replaces))
    if landed:write(path/'LOCAL_LANDING.json',{'fixture':True})
    return path


def test_index_keeps_healing_ambiguity_and_withdrawals(tmp_path):
    original=packet(tmp_path,'original','K5')
    healing=packet(tmp_path,'healing','K5-H2',replaces='K5')
    packet(tmp_path,'probe','K5',probe=True)
    index,present=C.packet_index(tmp_path)
    assert C.terminal_card(tmp_path,'K5',index) is None
    assert C.terminal_card(tmp_path,'K5-H2',index)==healing
    assert present=={'K5','K5-H2'}
    write(tmp_path/'withdrawals/original.json',{'state':'WITHDRAWN'})
    fresh,_=C.packet_index(tmp_path)
    assert C.terminal_card(tmp_path,'K5',fresh)==healing
    assert original not in fresh['K5']


def test_index_keeps_live_landing_and_refreshes_new_packets(tmp_path):
    first=packet(tmp_path,'first','K1',landed=False)
    index,present=C.packet_index(tmp_path)
    assert C.terminal_card(tmp_path,'K1',index) is None
    write(first/'LOCAL_LANDING.json',{'fixture':True})
    assert C.terminal_card(tmp_path,'K1',index)==first
    second=packet(tmp_path,'second','K2')
    assert C.terminal_card(tmp_path,'K2',index) is None
    assert C.terminal_card(tmp_path,'K2')==second
    assert 'K2' not in present


def test_index_reads_each_plan_once(tmp_path,monkeypatch):
    for i in range(20):packet(tmp_path,str(i),'K'+str(i))
    calls=[];old=C.read
    def counted(path):
        calls.append(path)
        return old(path)
    monkeypatch.setattr(C,'read',counted)
    index,_=C.packet_index(tmp_path)
    for i in range(20):assert C.terminal_card(tmp_path,'K'+str(i),index)
    assert len(calls)==20


def test_probe_recovery_preserves_active_failed_and_existing_science(tmp_path):
    from runners.queue_v27 import probes_without_successor
    def done(name,card):
        path=packet(tmp_path,name,card,probe=True)
        write(path/'COMPLETE.json',{'fixture':True})
        return path
    done('active','K1')
    failed=done('continuation-failed','K2')
    write(failed/'CONTINUATION_FAILURE.json',{'error':'retained failure'})
    done('has-science','K3')
    science=packet(tmp_path,'failed-study','K3',landed=False)
    write(science/'STATUS.json',{'state':'FAILED'})
    first=done('recover','K4')
    done('duplicate-probe','K4')
    ready=list(probes_without_successor(tmp_path,{'active':None}))
    assert len(ready)==1 and ready[0][1]['card']=='K4'
    assert ready[0][0].name in {'recover','duplicate-probe'}


def test_continuation_revision_tracks_admission_inputs_not_heartbeats(tmp_path):
    from runners.queue_v27 import continuation_revision
    previous=continuation_revision(tmp_path)
    for relative in ['cards/K1.json','protocols/K1.json','dispositions/K1.json',
                     'packets/primary/LOCAL_LANDING.json','withdrawals/primary.json']:
        write(tmp_path/relative,{'fixture':True})
        current=continuation_revision(tmp_path)
        assert current!=previous
        previous=current
    write(tmp_path/'packets/primary/STATUS.json',{'heartbeat':'changed'})
    assert continuation_revision(tmp_path)==previous
    write(tmp_path/'dispositions/K1.json',{'state':'WITHDRAWN'},immutable=False)
    assert continuation_revision(tmp_path)!=previous
