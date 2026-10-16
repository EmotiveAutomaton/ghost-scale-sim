import pytest
from ghostscale.validation.soundingline.v23.twin_contract import realism,require_realism,empirical_quantiles
from ghostscale.validation.soundingline.v23.common import rng


def test_nominal_dices_answers_cannot_be_relabelled_ordinal():
    source=dict(revision='a'*40,sha256='b'*64,citation='https://example.invalid/engineering-fixture')
    aggregates=dict(raters=123,items=350,ratings_per_rater=350,raters_per_item=123,
        rating_semantics='nominal yes/no/unsure',rating_marginals=[.3,.5,.2])
    result=realism('T4',dict(source=source,aggregates=aggregates),dict(aggregates,rating_semantics='ordinal low/mid/high'))
    assert not result['semantics_match']
    with pytest.raises(ValueError,match='semantics'):require_realism(result)


def test_census_counts_alone_never_admit_a_realistic_twin():
    result=realism('T1',dict(aggregates=dict(makers=61,sessions=1445,decisions=16959)),dict(makers=61))
    assert 'latency_quantiles_s' in result['missing'] and not result['real_shape_citation_allowed']
    with pytest.raises(ValueError):require_realism(result)
    with pytest.raises(ValueError,match='complete'):empirical_quantiles([.1,.9],[1,2],rng('quantile-fixture'),4)
