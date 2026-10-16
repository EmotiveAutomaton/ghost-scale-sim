from copy import deepcopy
import pytest
from ghostscale.validation.soundingline.v23 import registry
from ghostscale.validation.soundingline.v23.common import write,digest
from runners.freeze_v23_discovery import build_order


def fixture(tmp_path):
    evidence=tmp_path/'held.json';write(evidence,dict(reason='measured cost exceeds finite ceiling'))
    rows={name:dict(state='RESOURCE_DEFERRED',reason='measured cost exceeds finite ceiling',
        evidence=str(evidence),evidence_sha256=digest(evidence)) for name in registry.ORDER}
    for name, strength in [('R1',.2),('R2',.3)]:
        identity=registry.variants(name)[0]['id'];card=tmp_path/(name+'-card.json');review=tmp_path/(name+'-review.json')
        write(card,dict(family=name,state='NOMINATED',best_id=identity,
            curve=[dict(variant=dict(id=identity),shrunk_mean=strength)]))
        write(review,dict(state='verified_discovery',family=name,decision='NOMINATED',nominated=identity,
            full_fresh_process_replay=True,receipts=[dict(machine_card_sha256=digest(card))]))
        rows[name]=dict(state='NOMINATED',reason='complete reviewed discovery',evidence=str(card),evidence_sha256=digest(card),
            review=str(review),review_sha256=digest(review))
    return dict(families=rows)


def test_freeze_orders_only_fully_reviewed_nominations(tmp_path):
    manifest=fixture(tmp_path);order=build_order(manifest)
    assert [r['family'] for r in order['nominations']]==['R2','R1']
    assert len(order['dispositions'])==16
    bad=deepcopy(manifest);del bad['families']['T1']
    with pytest.raises(ValueError,match='every family'):build_order(bad)
    bad=deepcopy(manifest);bad['families']['N2']['state']='RUNNING'
    with pytest.raises(ValueError,match='explicit valid'):build_order(bad)


def test_freeze_rejects_stale_evidence_and_unreviewed_nomination(tmp_path):
    manifest=fixture(tmp_path);bad=deepcopy(manifest)
    bad['families']['R1']['evidence_sha256']='0'*64
    with pytest.raises(ValueError,match='evidence changed'):build_order(bad)
    bad=deepcopy(manifest);row=bad['families']['R1']
    write(row['review'],dict(state='verified_discovery',family='R1',decision='NOMINATED',
        full_fresh_process_replay=False),immutable=False)
    row['review_sha256']=digest(row['review'])
    with pytest.raises(ValueError,match='complete bound review'):build_order(bad)
