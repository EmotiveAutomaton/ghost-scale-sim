from pathlib import Path
import gzip
import json
import pytest
from ghostscale.validation.soundingline.v23.common import write, digest
from runners import review_v23_family as review


def test_partitioned_review_requires_every_replay_owner_and_bound_part(tmp_path, monkeypatch):
    root = tmp_path/'logical'; plan = dict(scheduler_only=True, worlds=4,
        partition_layout=dict(chunks=[[0, 1], [2, 3]]))
    write(root/'PLAN.json', plan)
    plans = {}
    for i, worlds in enumerate(plan['partition_layout']['chunks']):
        part = root.with_name(f'logical-part-{i+1}')
        plans[part] = dict(world_indices=worlds, logical_parent=str(root),
            logical_plan_sha256=digest(root/'PLAN.json'))
        write(part/'COMPLETE.json', {'state':'verified'})
        for label, pid, at in [('original',10+i,'2026-01-01'), ('replay',20+i,'2026-01-02')]:
            write(part/(label+'-owner.json'), dict(pid=pid, at=at))
    monkeypatch.setattr(review.family, 'verify_packet', lambda p: (plans[p], {}))
    assert len(review.replay_owners(root, plan)) == 2
    bad = root.with_name('logical-part-2')
    plans[bad]['world_indices'] = [1, 2]
    with pytest.raises(ValueError, match='provenance'):
        review.replay_owners(root, plan)
    plans[bad]['world_indices'] = [2, 3]
    write(bad/'replay-owner.json', dict(pid=11, at='2026-01-02'), immutable=False)
    with pytest.raises(ValueError, match='distinct sequential'):
        review.replay_owners(root, plan)


def test_review_uses_and_checks_separately_frozen_fits(tmp_path):
    fits = tmp_path/'prefit'; write(fits/'FITS.json', {'rival':{'converged':True}})
    plan = dict(fit_directory=str(fits), fit_hashes={'FITS.json':digest(fits/'FITS.json')})
    assert review.fit_evidence(tmp_path/'batch', plan)['summary']['rival']['converged']
    write(fits/'FITS.json', {'rival':{'converged':False}}, immutable=False)
    with pytest.raises(ValueError, match='frozen fit changed'):
        review.fit_evidence(tmp_path/'batch', plan)


def test_raw_review_rejects_duplicates_and_unbalanced_laws(tmp_path):
    plan = dict(family='R4', batch=1, worlds=2, cases=1)
    summary = dict(variants=[dict(variant=dict(id='v'))]); folder = tmp_path/'original/v'
    folder.mkdir(parents=True)
    def save(preds, truths):
        for name, rows in [('prediction',preds), ('truth',truths)]:
            with gzip.open(folder/(name+'_points.jsonl.gz'), 'wt') as stream:
                for row in rows: stream.write(json.dumps(row)+'\n')
    preds = [dict(world=w, case=0) for w in range(2)]
    truths = [dict(identity=['R4','v',1,w,0], truth=dict(law=w)) for w in range(2)]
    save(preds, truths)
    assert review.validate_raw_identities(tmp_path, plan, summary) == 2
    save([preds[0]]*2, [truths[0]]*2)
    with pytest.raises(ValueError, match='duplicate'):
        review.validate_raw_identities(tmp_path, plan, summary)
    truths[1]['truth']['law'] = 0; save(preds, truths)
    with pytest.raises(ValueError, match='unbalanced'):
        review.validate_raw_identities(tmp_path, plan, summary)
