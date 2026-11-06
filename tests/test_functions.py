""" Fast checks of the functions behind the widgets, on synthetic data. """
import numpy as np
import pytest
import yaml
from scipy import ndimage as ndi

from spectra_spin_napari import _widgets as w
from spectra_spin_napari.phase_model_recon import _sample
from spectra_spin_napari import utils
from spectra_spin_napari.utils import spectral_calibration


def test_setup_file_holds_every_key():
    params = w.setup_params()
    assert all(key in params[section] for section, keys in w.SETUP_KEYS.items() for key in keys)


def test_setup_missing_key_is_named(tmp_path, monkeypatch):
    params = yaml.safe_load(open(w.SETUP_PATH))
    del params['optics']['spectral_fwhm']
    path = tmp_path / 'setup_params.yaml'
    path.write_text(yaml.safe_dump(params))
    monkeypatch.setattr(w, 'SETUP_PATH', path)
    w.setup_params.cache_clear()
    try:
        with pytest.raises(ValueError, match='optics.spectral_fwhm'):
            w.setup_params()
    finally:
        w.setup_params.cache_clear()


def test_line_fwhm():
    width = 0.0533 * 89
    values = [utils.line_fwhm(k, 11.0, width) for k in range(1, 7)]
    assert values[0] == pytest.approx(11.0, abs=0.01)
    assert np.all(np.diff(values) > 0)
    # no aperture left: a pure box of the sweep
    assert utils.line_fwhm(3, 4.0, 4.0) == pytest.approx(12.0, abs=0.05)


def test_spectral_calibration_recovers_a_line():
    channels = np.array([20.3, 38.1, 61.7, 22.0, 40.2])
    nm = 440.0 + 3.6 * channels
    calibration = spectral_calibration(channels, nm, n_lambda=89, degree=1)
    assert calibration['dispersion'] == pytest.approx(3.6, abs=1e-9)
    assert calibration['range'][0] == pytest.approx(440.0, abs=1e-6)


def _synthetic_stack(seed=0):
    """ Three Gaussian dyes at channels 20, 40, 60 in sparse mixtures, plus noise. """
    rng = np.random.default_rng(seed)
    k = np.arange(89)
    spectra = np.stack([np.exp(-0.5 * ((k - c) / 5.0) ** 2) for c in (20, 40, 60)], 1)
    abundance = rng.gamma(0.3, 200.0, size=(3, 120, 120)) * (rng.random((3, 120, 120)) < 0.5)
    stack = np.tensordot(spectra, abundance, axes=(1, 0)) + rng.normal(0, 2.0, (89, 120, 120))
    return np.clip(stack + 20, 0, None).astype(np.float32)


@pytest.mark.parametrize('components', ['auto', 6, None])
def test_ads_finds_the_dyes(components):
    result = utils.adaptive_dye_separation(_synthetic_stack(), 3, pca_components=components)
    peaks = np.sort(result['references'].argmax(axis=0))
    np.testing.assert_allclose(peaks, [20, 40, 60], atol=1)
    if components == 'auto':
        assert result['components'] >= 3
    else:
        assert result['components'] == components


def test_numba_sampling_matches_scipy():
    rng = np.random.default_rng(1)
    image = rng.random((60, 80))
    rows, cols = rng.uniform(-2, 62, (30, 40)), rng.uniform(-2, 82, (30, 40))
    expected = ndi.map_coordinates(image, (rows, cols), order=1, mode='nearest')
    np.testing.assert_allclose(_sample(image, rows, cols), expected, atol=1e-5)


def test_roi_mean_spectra():
    rng = np.random.default_rng(2)
    stack = rng.integers(0, 1000, (5, 20, 30)).astype(np.uint16)
    mask = np.zeros((20, 30), int)
    mask[2:6, 3:9], mask[10:18, 12:25] = 1, 4
    labels, spectra = utils.roi_mean_spectra(stack, mask)
    assert labels.tolist() == [1, 4]
    for label, spectrum in zip(labels, spectra):
        np.testing.assert_allclose(spectrum, stack[:, mask == label].mean(axis=1))


def test_max_pooling2d():
    rng = np.random.default_rng(3)
    image = rng.integers(0, 1000, (11, 13)).astype(np.uint16)
    out = utils.max_pooling2d(image, (4, 2), (3, 2))
    expected = [[image[i * 3:i * 3 + 4, j * 2:j * 2 + 2].max() for j in range(6)] for i in range(3)]
    assert out.dtype == image.dtype
    np.testing.assert_array_equal(out, expected)
