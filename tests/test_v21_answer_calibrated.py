import pytest
from ghostscale.validation.soundingline.v21.answer_policy import forecast_entries


def test_calibrated_selection_excludes_copied_cases_and_fixed_powers():
    rows = [dict(fit=f, lineage=71000860, reader=r, duplication_rate=d)
            for f in (0, 1) for r in ('episode-calibrated', 'episode-power-0.25', 'latent-full')
            for d in (0., .5, .9)]
    selected = forecast_entries(dict(population=rows), 'dependence-calibrated')
    assert len(selected) == 2
    assert [r['fit'] for r, _ in selected] == [0, 1]
    assert all('-0.0-episode-calibrated_points.npz' in name for _, name in selected)
    assert all(r['duplication_rate'] == 0. for r, _ in selected)


def test_missing_duplicate_and_unknown_source_are_rejected():
    row = dict(fit=0, lineage=71000860, reader='episode-calibrated', duplication_rate=0.)
    for rows in ([], [row, row]):
        with pytest.raises(ValueError):
            forecast_entries(dict(population=rows), 'dependence-calibrated')
    with pytest.raises(ValueError):
        forecast_entries(dict(population=[row]), 'unknown')


def test_original_forecast_format_keeps_its_identity():
    row = dict(fit=1, lineage=71000860, reader='converged-direct')
    assert forecast_entries(dict(population=[row]), 'learned') == [
        (row, 'raw/1-71000860-converged-direct_points.npz')]
