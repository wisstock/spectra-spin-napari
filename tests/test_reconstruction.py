""" Regression of `PhaseModelRecon` against the golden run of the bundled sample.

`data/golden_sample.npz` holds the fitted phase and geometry and the digests
of the stacks; the match is exact. ``python tests/reference.py`` remakes it,
which is done only when a change of the numbers has been approved.

"""
import numpy as np
import pytest

from reference import DATA, run


@pytest.fixture(scope='module')
def golden():
    return np.load(DATA / 'golden_sample.npz')


@pytest.mark.slow
@pytest.mark.parametrize('key, passes', [('single', None), ('one', 1)])
def test_matches_golden_run(golden, key, passes):
    result = run(passes)
    np.testing.assert_array_equal(result['phase'], golden[f'{key}_phase'])
    np.testing.assert_array_equal(result['line_coef'], golden[f'{key}_line_coef'])
    for name in ('mean_sha256', 'max_sha256', 'hits_sha256'):
        assert str(result[name]) == str(golden[f'{key}_{name}']), name
