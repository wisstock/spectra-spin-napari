""" Computations of the plugin outside the reconstruction itself.

Preprocessing of frames, post-processing, colour and dye separation of lambda
stacks, spectral calibration and console diagnostics. No Qt, napari or
`phase_model_recon` import: the functions work on any array. scipy,
tifffile and matplotlib are imported inside the functions that use them, so
importing the module is cheap. The measurements behind the defaults are in
``docs/design_notes.md``.

"""

import json
import logging
import math
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CMF_PATH = Path(__file__).parent / 'cie1931_2deg.csv'
# XYZ to linear sRGB, IEC 61966-2-1, defined on the CIE 1931 2 degree observer
XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])
# Bradford cone response, adapts the equal-energy white of a spectrum to D65
BRADFORD = np.array([[0.8951, 0.2664, -0.1614],
                     [-0.7502, 1.7135, 0.0367],
                     [0.0389, -0.0685, 1.0296]])

# warning limits of `adaptive_dye_separation`, set on synthetic spectra
ADS_UNSTABLE_DEG = 5.0
ADS_NEGATIVE_PERCENT = 10.0
ADS_COINCIDE_DEG = 5.0
# automatic PCA rank: components above this multiple of the noise floor
ADS_AUTO_FACTOR = 20.0

# --- Frame preprocessing -----------------------------------------------------

def despeckle(frame:np.ndarray, threshold:float=6.0, size:int=3) -> tuple:
    """ Replace isolated hot and dead pixels, and nothing else.

    A pixel is replaced by the mean of the ring of its neighbours (the window
    without its centre) when it lies outside the whole ring by more than
    `threshold` robust standard deviations (MAD of the difference from the ring
    mean). A feature that continues into its neighbourhood, one-pixel pattern
    lines included, is never outside its ring; a median filter would blur those.

    Parameters
    ----------
    frame : numpy.ndarray
        Frame to clean.
    threshold : float, optional
        Robust standard deviations outside the ring, by default 6.0.
    size : int, optional
        Window side, by default 3. The left-out centre is ``size - 2`` wide: 3
        catches single pixels, 5 also pairs and clusters up to 3 x 3.

    Returns
    -------
    numpy.ndarray
        Cleaned frame, same shape and dtype.
    int
        Number of replaced pixels.

    """
    from scipy import ndimage as ndi

    footprint = np.ones((size, size), bool)
    half, inner = size // 2, size - 2
    footprint[half - inner // 2:half + inner // 2 + 1,
              half - inner // 2:half + inner // 2 + 1] = False
    high = ndi.maximum_filter(frame, footprint=footprint)
    low = ndi.minimum_filter(frame, footprint=footprint)

    # the ring mean, as the difference of two separable box filters
    total = ndi.uniform_filter(frame, size=size) * size ** 2
    centre = ndi.uniform_filter(frame, size=inner) * inner ** 2 if inner > 1 else frame
    neighbours = (total - centre) / (size ** 2 - max(inner, 1) ** 2)

    deviation = frame - neighbours
    spread = 1.4826 * np.median(np.abs(deviation - np.median(deviation)))
    spread = max(float(spread), 1e-6)
    outliers = (frame > high + threshold * spread) | (frame < low - threshold * spread)
    return np.where(outliers, neighbours, frame).astype(frame.dtype), int(outliers.sum())


def subtract_dark_level(frame:np.ndarray, percentile:float=1.0) -> tuple:
    """ Subtract the camera pedestal, taken as a low percentile of the frame.

    The pedestal inflates the mean of the fold contrast without adding signal.
    Negative values are clipped, as the band sampling stores unsigned integers.

    Parameters
    ----------
    frame : numpy.ndarray
        Frame to correct.
    percentile : float, optional
        Percentile taken as the dark level, by default 1.0; it must stay below
        any signal, which the dark gaps of a pattern frame make easy.

    Returns
    -------
    numpy.ndarray
        Corrected frame.
    float
        Subtracted level.

    """
    level = float(np.percentile(frame, percentile))
    return np.clip(frame - level, 0, None), level


# --- Gap filling and pooling -------------------------------------------------

def interpolate_zero_gaps(img:np.ndarray, axis:int=0, mask:np.ndarray=None,
                          extrapolate:bool=False) -> np.ndarray:
    """ Fill zero pixels by linear interpolation along one axis.

    The gaps of a reconstruction are horizontal stripes of unwritten rows, so
    the default axis 0 fills them from the measured rows above and below.
    Measured pixels are returned untouched; pixels with data on one side only
    stay zero unless `extrapolate` is set.

    Parameters
    ----------
    img : numpy.ndarray
        Image to fill, normally one channel. Float64 work arrays of the input
        size are allocated, so fill a stack channel by channel.
    axis : int, optional
        Axis to interpolate along, by default 0.
    mask : numpy.ndarray, optional
        True where a pixel holds data, by default ``img != 0``;
        ``PhaseModelRecon.lambda_hit_count > 0`` is the exact mask.
    extrapolate : bool, optional
        Extend the end values into one-sided gaps, by default False.

    Returns
    -------
    numpy.ndarray
        Filled array, shape and dtype of `img`; integers are rounded.

    Examples
    --------
    >>> interpolate_zero_gaps(np.array([[10, 0, 5], [0, 0, 0], [0, 0, 0], [22, 0, 9]]))
    array([[10,  0,  5],
           [14,  0,  6],
           [18,  0,  8],
           [22,  0,  9]])

    """
    arr = np.asarray(img)
    work = np.moveaxis(arr, axis, 0).astype(np.float64)
    valid = (np.moveaxis(np.asarray(mask, dtype=bool), axis, 0) if mask is not None
             else work != 0)
    n_samples = work.shape[0]

    # index of every sample along the working axis, broadcast over the rest
    position = np.arange(n_samples).reshape((n_samples,) + (1,) * (work.ndim - 1))

    # nearest valid sample at or before each position, -1 where there is none
    lower = np.maximum.accumulate(np.where(valid, position, -1), axis=0)
    # nearest valid sample at or after each position, n_samples where there is none
    upper = np.minimum.accumulate(np.where(valid, position, n_samples)[::-1], axis=0)[::-1]

    has_lower, has_upper = lower >= 0, upper < n_samples
    low_idx = np.clip(lower, 0, n_samples - 1)
    high_idx = np.clip(upper, 0, n_samples - 1)
    low_value = np.take_along_axis(work, low_idx, axis=0)
    high_value = np.take_along_axis(work, high_idx, axis=0)

    # linear weight between the two bracketing samples, 0 where they coincide
    span = (high_idx - low_idx).astype(np.float64)
    weight = np.divide(position - low_idx, span,
                       out=np.zeros_like(work), where=span > 0)
    interpolated = low_value + (high_value - low_value) * weight

    bracketed = has_lower & has_upper
    filled = np.where(bracketed, interpolated, 0.0)
    if extrapolate:
        # one-sided gaps take the value of the single end point that exists
        one_sided = np.where(has_lower, low_value, np.where(has_upper, high_value, 0.0))
        filled = np.where(bracketed, interpolated, one_sided)

    filled = np.where(valid, work, filled)
    if np.issubdtype(arr.dtype, np.integer):
        filled = np.rint(filled)

    gaps = int((~valid).sum())
    written = int((~valid & (filled != 0)).sum())
    logger.info(f'Interpolated {written} of {gaps} empty pixels along axis {axis} '
                 f'({100 * written / max(gaps, 1):.1f}% of the gaps, '
                 f'{100 * (valid.sum() + written) / valid.size:.1f}% of the frame now non-zero)')

    return np.moveaxis(filled, 0, axis).astype(arr.dtype)


def max_pooling2d(image:np.ndarray, pool_size:tuple=(2, 2), stride:tuple=(2, 2)) -> np.ndarray:
    """ Maximum over a sliding window.

    Keeps measured values only and closes the row gaps of a sparse stack at
    the cost of resolution; biased upwards, so a display and screening tool.
    A tall narrow window, ``(4, 1)``, suits the horizontal gaps; choose its
    height from the longest run of empty rows, see ``docs/design_notes.md``.

    Parameters
    ----------
    image : numpy.ndarray
        ``(height, width)`` image; pool a stack channel by channel.
    pool_size : tuple of int, optional
        Window height and width, by default (2, 2).
    stride : tuple of int, optional
        Row and column step, by default (2, 2).

    Returns
    -------
    numpy.ndarray
        ``((height - pool_h) // stride_h + 1, (width - pool_w) // stride_w + 1)``,
        dtype of `image`; rows and columns that do not fill a window are dropped.

    """
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(np.asarray(image), tuple(pool_size))
    return windows[::stride[0], ::stride[1]].max(axis=(2, 3))


# --- Regions of interest -----------------------------------------------------

def flatten_roi_mask(mask:np.ndarray) -> np.ndarray:
    """ Collapse a labels volume onto one plane.

    A labels layer drawn over a stack has one plane per channel, but a region
    is a place in the frame; the largest label of every pixel column is kept,
    so overlapping regions give their shared pixels to the higher label.

    Parameters
    ----------
    mask : numpy.ndarray
        ``(height, width)`` or ``(planes, height, width)`` labels.

    Returns
    -------
    numpy.ndarray
        ``(height, width)`` labels; a 2D input is returned unchanged.

    """
    if mask.ndim == 2:
        return mask
    flat = mask.max(axis=0)
    logger.info(f'Flattened a {mask.shape} labels volume onto one plane, '
                f'{len(np.unique(flat[flat > 0]))} regions, '
                f'{int((flat > 0).sum())} labelled pixels')
    return flat


def roi_mean_spectra(stack:np.ndarray, mask:np.ndarray, report=None) -> tuple:
    """ Mean spectrum of every labelled region.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_lambda, height, width)`` stack.
    mask : numpy.ndarray
        ``(height, width)`` labels, 0 is background.
    report : callable, optional
        ``report(percent)`` progress callback.

    Returns
    -------
    numpy.ndarray
        Sorted label values.
    numpy.ndarray
        ``(n_labels, n_lambda)`` float64 mean spectra.

    """
    labels, index = np.unique(mask, return_inverse=True)
    index = index.ravel()
    counts = np.bincount(index, minlength=len(labels))
    spectra = np.empty((len(labels), len(stack)))
    for channel in range(len(stack)):
        spectra[:, channel] = np.bincount(index, weights=np.asarray(stack[channel], np.float64).ravel(),
                                          minlength=len(labels)) / np.maximum(counts, 1)
        if report is not None:
            report(int(100 * (channel + 1) / len(stack)))
    keep = labels > 0
    return labels[keep], spectra[keep]


# --- Colour and virtual channels ---------------------------------------------

def lambda_stack_to_rgb(stack:np.ndarray, start_nm:float, step_nm:float,
                        report=None) -> tuple:
    """ Linear sRGB colour of every pixel spectrum.

    1. Channel ``k`` sits at ``start_nm + k * step_nm``; the CIE 1931 2 degree
       colour matching functions are interpolated onto it, zero outside
       360..830 nm.
    2. ``XYZ = sum_k S_k * cmf(lambda_k) * step_nm``, in blocks of rows.
    3. Bradford adaptation from the equal-energy white to D65, so a flat
       spectrum renders neutral.
    4. XYZ to linear sRGB; out-of-gamut pixels are desaturated towards white
       by their negative deficit, which keeps the hue.
    5. Division by the summed luminance weight: a flat spectrum of value ``v``
       has luminance ``v``.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_lambda, height, width)`` stack, best with its gaps filled.
    start_nm : float
        Wavelength of channel 0, nm.
    step_nm : float
        Positive step per channel, nm.
    report : callable, optional
        ``report(percent)`` progress callback.

    Returns
    -------
    numpy.ndarray
        ``(3, height, width)`` float32 non-negative linear sRGB in stack units.
    dict
        'visible' - channels inside 360..830 nm, 'range' - wavelengths of the
        first and last channel, 'out_of_gamut' - fraction of lit pixels
        desaturated.

    Raises
    ------
    ValueError
        If `step_nm` is not positive or no channel is visible.

    """
    if step_nm <= 0:
        raise ValueError(f'The wavelength step must be positive, got {step_nm}')
    table = np.loadtxt(CMF_PATH, delimiter=',')
    n_lambda, height, width = stack.shape
    wavelengths = start_nm + step_nm * np.arange(n_lambda)
    cmf = np.stack([np.interp(wavelengths, table[:, 0], table[:, i], left=0, right=0)
                    for i in (1, 2, 3)], axis=1) * step_nm
    visible = int((cmf[:, 1] > 0).sum())
    if not visible:
        raise ValueError(f'The spectral axis {wavelengths[0]:.0f}..{wavelengths[-1]:.0f} nm '
                         f'lies outside the visible range, 360..830 nm')

    # equal-energy white (X = Y = Z) adapted to D65, then to linear sRGB
    d65 = np.array([0.95047, 1.0, 1.08883])
    gain = (BRADFORD @ d65) / (BRADFORD @ np.ones(3))
    to_rgb = XYZ_TO_SRGB @ np.linalg.inv(BRADFORD) @ np.diag(gain) @ BRADFORD
    weights = to_rgb @ cmf.T / cmf[:, 1].sum()   # (3, n_lambda), in stack units

    rgb = np.empty((3, height, width), dtype=np.float32)
    block = max(1, 2 ** 22 // max(n_lambda * width, 1))  # ~16 MB of float32 per block
    for row in range(0, height, block):
        chunk = np.asarray(stack[:, row:row + block], dtype=np.float32).reshape(n_lambda, -1)
        rows = min(block, height - row)
        rgb[:, row:row + rows] = (weights @ chunk).reshape(3, rows, width)
        if report is not None:
            report(int(100 * (row + rows) / height))

    # out of gamut: lift all three channels by the deficit, which desaturates
    # towards white at constant hue instead of clipping the hue away
    deficit = np.minimum(rgb.min(axis=0), 0)
    lit = rgb.max(axis=0) > 0
    out_of_gamut = float((deficit < 0)[lit].mean()) if lit.any() else 0.0
    rgb -= deficit

    info = {'visible': visible, 'range': (float(wavelengths[0]), float(wavelengths[-1])),
            'out_of_gamut': out_of_gamut}
    logger.info(f'Converted {stack.shape} to linear sRGB over {info["range"][0]:.1f}..'
                f'{info["range"][1]:.1f} nm, {visible} of {n_lambda} channels visible, '
                f'{100 * out_of_gamut:.1f}% of lit pixels outside the sRGB gamut')
    return rgb, info


def rgb_auto_levels(rgb:np.ndarray, black_percentile:float=50.0,
                    white_percentile:float=99.5) -> tuple:
    """ Black and white levels from percentiles of the brightness.

    Brightness is the brightest channel of a pixel. Every fourth row and
    column of the lit pixels is sampled.

    Parameters
    ----------
    rgb : numpy.ndarray
        ``(3, height, width)`` linear sRGB from `lambda_stack_to_rgb`.
    black_percentile, white_percentile : float, optional
        By default 50 (the background goes black) and 99.5 (hot pixels do not
        set the white).

    Returns
    -------
    tuple of float
        ``(black, white)`` in stack units.

    """
    level = rgb[:, ::4, ::4].max(axis=0)
    level = level[level > 0]
    if not level.size:
        return 0.0, 1.0
    black, white = np.percentile(level, [black_percentile, white_percentile])
    return float(black), float(max(white, black + 1e-6 * max(abs(black), 1)))


def rgb_levels(rgb:np.ndarray, black:float, white:float, gamma:float=1.0,
               intensity:str='absolute') -> np.ndarray:
    """ Map linear sRGB to a uint8 image through levels of the brightness.

    The three channels are scaled by one factor per pixel, so the levels change
    brightness without shifting the hue, unlike per-channel contrast limits.

    Parameters
    ----------
    rgb : numpy.ndarray
        ``(3, height, width)`` linear sRGB from `lambda_stack_to_rgb`.
    black, white : float
        Brightness shown black and full white, in stack units.
    gamma : float, optional
        Exponent between the levels, by default 1.0 (linear), applied before
        the sRGB transfer curve.
    intensity : {'absolute', 'per-pixel'}, optional
        'absolute' (default) maps brightness through the levels; 'per-pixel'
        shows every pixel above `black` at full brightness, colour only.

    Returns
    -------
    numpy.ndarray
        ``(height, width, 3)`` uint8 sRGB.

    Raises
    ------
    ValueError
        If `intensity` is unknown.

    """
    if intensity not in ('absolute', 'per-pixel'):
        raise ValueError(f'Unknown intensity mode {intensity!r}')
    level = rgb.max(axis=0)
    if intensity == 'absolute':
        target = np.clip((level - black) / max(white - black, 1e-12), 0, 1) ** gamma
    else:
        target = (level > black).astype(np.float32)
    out = rgb * (target / np.where(level > 0, level, 1))
    # sRGB transfer curve
    out = np.where(out <= 0.0031308, 12.92 * out, 1.055 * np.power(out, 1 / 2.4) - 0.055)
    return np.round(np.moveaxis(out, 0, -1) * 255).astype(np.uint8)


def virtual_channels(stack:np.ndarray, start_nm:float, step_nm:float, ranges:list,
                     report=None) -> tuple:
    """ Sum a stack over wavelength ranges, like filter cubes.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_lambda, height, width)`` stack.
    start_nm, step_nm : float
        Wavelength of channel 0 and positive step per channel, nm.
    ranges : list of tuple
        ``(low_nm, high_nm)`` per output channel, bounds included, any order.
    report : callable, optional
        ``report(percent)`` progress callback.

    Returns
    -------
    numpy.ndarray
        ``(len(ranges), height, width)`` float32 sums.
    list of int
        Input channels summed into each output channel.

    Raises
    ------
    ValueError
        If `step_nm` is not positive or a range holds no channel.

    """
    if step_nm <= 0:
        raise ValueError(f'The wavelength step must be positive, got {step_nm}')
    wavelengths = start_nm + step_nm * np.arange(len(stack))
    members = [np.flatnonzero((wavelengths >= min(r)) & (wavelengths <= max(r)))
               for r in ranges]
    for (low, high), index in zip(ranges, members):
        if not index.size:
            raise ValueError(f'The range {min(low, high):g}..{max(low, high):g} nm holds no '
                             f'channel of the axis {wavelengths.min():.1f}..'
                             f'{wavelengths.max():.1f} nm')
    out = np.zeros((len(ranges), *stack.shape[1:]), dtype=np.float32)
    total, done = sum(len(index) for index in members), 0
    for k, index in enumerate(members):
        for i in index:                # channel by channel: no float copy of the cube
            out[k] += stack[i]
            done += 1
            if report is not None:
                report(int(100 * done / total))
    logger.info('Virtual channels: ' + ', '.join(
        f'{min(r):g}..{max(r):g} nm ({len(index)} channels)' for r, index in zip(ranges, members)))
    return out, [len(index) for index in members]


# --- Adaptive dye separation -------------------------------------------------

def _spherical_kmeans(points:np.ndarray, k:int, rng, restarts:int=5,
                      iterations:int=60) -> np.ndarray:
    """ Cluster unit vectors by angle, best of several k-means++ starts.

    Parameters
    ----------
    points : numpy.ndarray
        ``(n_points, n_features)`` unit vectors.
    k : int
        Number of clusters.
    rng : numpy.random.Generator
        Source of the starting points.
    restarts : int, optional
        Independent starts, by default 5.
    iterations : int, optional
        Largest number of rounds per start, by default 60.

    Returns
    -------
    numpy.ndarray
        ``(n_points,)`` cluster index.

    """
    best, best_score = None, -np.inf
    for _ in range(restarts):
        centres = [points[rng.integers(len(points))]]
        for _ in range(k - 1):
            distance = np.maximum(1 - (points @ np.array(centres).T).max(axis=1), 0) ** 2
            centres.append(points[rng.choice(len(points), p=distance / distance.sum())
                                  if distance.sum() > 0 else rng.integers(len(points))])
        centres = np.array(centres)
        for _ in range(iterations):
            labels = (points @ centres.T).argmax(axis=1)
            updated = centres.copy()
            for j in np.unique(labels):
                updated[j] = points[labels == j].mean(axis=0)
                updated[j] /= np.linalg.norm(updated[j])
            converged = np.allclose(updated, centres)
            centres = updated
            if converged:
                break
        labels = (points @ centres.T).argmax(axis=1)
        score = (points * centres[labels]).sum()
        if score > best_score:
            best, best_score = labels, score
    return best


def _nmf_refine(pixels:np.ndarray, references:np.ndarray, iterations:int=300,
                max_pixels:int=20000) -> np.ndarray:
    """ Move reference spectra towards pure ones with multiplicative NMF updates.

    Parameters
    ----------
    pixels : numpy.ndarray
        ``(n_pixels, n_lambda)`` non-negative spectra.
    references : numpy.ndarray
        ``(n_lambda, n_dyes)`` starting spectra.
    iterations : int, optional
        Number of updates, by default 300.
    max_pixels : int, optional
        Largest number of evenly spaced pixels used, by default 20000.

    Returns
    -------
    numpy.ndarray
        ``(n_lambda, n_dyes)`` spectra of unit length.

    """
    data = pixels[::max(1, len(pixels) // max_pixels)].T
    spectra = np.maximum(references, 1e-6)
    contents = np.maximum(np.linalg.pinv(spectra) @ data, 1e-6)
    for _ in range(iterations):
        contents *= (spectra.T @ data) / np.maximum(spectra.T @ spectra @ contents, 1e-9)
        spectra *= (data @ contents.T) / np.maximum(spectra @ contents @ contents.T, 1e-9)
        length = np.linalg.norm(spectra, axis=0, keepdims=True)
        spectra /= length
        contents *= length.T
    return spectra


def _ads_references(samples:np.ndarray, n_dyes:int, mode:str, noise_gate:float,
                    signal_percentile:float, pca_components:int, refine:bool, rng) -> dict:
    """ Reference spectra of `adaptive_dye_separation` from a pixel sample.

    Parameters
    ----------
    samples : numpy.ndarray
        ``(n_pixels, n_lambda)`` spectra without background.
    n_dyes : int
        Number of dyes.
    mode : str
        'weak' or 'strong'.
    noise_gate : float
        Noise multiples a pixel must clear.
    signal_percentile : float
        Percentile of the pixels above the gate taken as the signal threshold.
    pca_components : int or None
        Components to cluster in; None clusters in the channels.
    refine : bool
        Polish the spectra with `_nmf_refine`.
    rng : numpy.random.Generator
        Source of the clustering starts.

    Returns
    -------
    dict
        'references' - ``(n_lambda, n_dyes)`` spectra of unit sum by peak
        channel, 'basis' - ``(n_lambda, rank)`` subspace of the noise estimate,
        'gate' - noise gate in it, 'explained' - energy fraction per component,
        'noise', 'used' - fraction clustered, 'view' - see
        `adaptive_dye_separation`.

    Raises
    ------
    ValueError
        If too few pixels clear the thresholds or a cluster is empty.

    """
    n_lambda = samples.shape[1]
    # uncentred on purpose: mixtures of dyes are rays through the origin, and
    # moving the origin to the centre of mass would no longer leave them rays
    energy, vectors = np.linalg.eigh(samples.T @ samples / len(samples))
    energy = np.maximum(energy[::-1], 0)
    rank = pca_components or n_dyes
    basis = vectors[:, ::-1][:, :rank]
    scores = samples @ basis

    # what the subspace leaves out of a pixel is taken as its noise
    noise = np.sqrt(np.median(((samples - scores @ basis.T) ** 2).sum(axis=1) / (n_lambda - rank)))
    length = np.linalg.norm(scores, axis=1)
    gate = noise_gate * noise * np.sqrt(rank)
    above = length > gate
    cut = max(gate, np.percentile(length[above], signal_percentile)) if above.any() else np.inf
    keep = above & (length >= cut)
    if keep.sum() < 100 * n_dyes:
        raise ValueError(f'Only {int(keep.sum())} pixels clear the thresholds, at least '
                         f'{100 * n_dyes} are needed: lower them, or the signal is too weak')

    pixels = scores[keep] @ basis.T if pca_components else samples[keep]
    features = scores[keep] if pca_components else pixels
    labels = _spherical_kmeans(features / np.linalg.norm(features, axis=1, keepdims=True),
                               n_dyes, rng)
    if len(np.unique(labels)) < n_dyes:
        raise ValueError(f'Only {len(np.unique(labels))} of {n_dyes} clusters hold pixels: '
                         f'lower the number of dyes')

    # the reference point of each cluster: its centre, or its side facing away
    # from the other clusters; a centre is itself a mixture and over-corrects
    unit = pixels / np.linalg.norm(pixels, axis=1, keepdims=True)
    centres = np.array([unit[labels == j].mean(axis=0) for j in range(n_dyes)])
    references = []
    for j in range(n_dyes):
        members = unit[labels == j]
        if mode == 'weak':
            away = -(members @ np.delete(centres, j, axis=0).T).max(axis=1)
            members = members[away >= np.percentile(away, 95)]
        reference = members.mean(axis=0)
        references.append(reference / np.linalg.norm(reference))
    references = np.array(references).T
    if refine:
        references = _nmf_refine(np.maximum(pixels, 0), references)

    references = np.maximum(references, 0)
    order = np.argsort(references.argmax(axis=0), kind='stable')
    references = references[:, order]

    # the feature space as the clustering saw it, for display: an evenly spaced
    # sample of the pixels on the three leading components, the dye each one
    # was clustered into (in the order of the references, -1 below the
    # thresholds) and the direction of every reference; no random draw, so
    # the results stay what they are without the display
    display = vectors[:, ::-1][:, :3].copy()
    # an eigenvector has no sign of its own: point PC1 along the spectra,
    # which are non-negative, so that the pixels lie at positive PC1
    display[:, 0] *= np.sign(display[:, 0].sum()) or 1.0
    dye = np.full(len(samples), -1, dtype=np.int8)
    rank_of = np.empty(n_dyes, dtype=np.int8)
    rank_of[order] = np.arange(n_dyes)
    dye[keep] = rank_of[labels]
    shown = np.arange(0, len(samples), max(1, len(samples) // 4000))
    unit_refs = references / np.maximum(np.linalg.norm(references, axis=0), 1e-12)
    view = {'scores': (samples[shown] @ display).astype(np.float32), 'dye': dye[shown],
            'references': (unit_refs.T @ display).astype(np.float32),
            'explained': (energy / max(energy.sum(), 1e-12))[:3]}
    return {'references': references / references.sum(axis=0), 'basis': basis, 'gate': gate,
            'explained': energy / max(energy.sum(), 1e-12), 'noise': noise,
            'used': float(keep.mean()), 'view': view}


def _ads_auto_components(samples:np.ndarray, n_dyes:int) -> int:
    """ Number of uncentred PCA components above the noise.

    The floor is the median eigenvalue of the lower half; a component counts
    above `ADS_AUTO_FACTOR` floors. Besides the dyes, a spectrum shifting across
    the field adds about one direction per dye and the background one more,
    see ``docs/design_notes.md``.

    Parameters
    ----------
    samples : numpy.ndarray
        ``(n_pixels, n_lambda)`` spectra without background.
    n_dyes : int
        Number of dyes, the smallest result.

    Returns
    -------
    int
        Between `n_dyes` and ``n_lambda - 1``.

    """
    n_lambda = samples.shape[1]
    energy = np.sort(np.linalg.eigvalsh(samples.T @ samples / len(samples)))[::-1]
    floor = max(float(np.median(energy[n_lambda // 2:])), 1e-12)
    return int(np.clip((energy > ADS_AUTO_FACTOR * floor).sum(), n_dyes, n_lambda - 1))


def adaptive_dye_separation(stack:np.ndarray, n_dyes:int, mode:str='weak',
                            noise_gate:float=3.0, signal_percentile:float=50.0,
                            subtract_background:bool=True, pca_components:int=None,
                            refine:bool=True, max_samples:int=200000, seed:int=0,
                            report=None) -> dict:
    """ Separate the dyes of a stack without known spectra.

    The adaptive dye separation of the Leica TCS SP2 manual (pp. 221-224):
    pixels are points of a feature space, noise near the origin is dropped, the
    rest is projected onto the unit sphere and clustered, and the cluster
    directions give the unmixing matrix. Additions to the manual:

    * reference point: 'weak' takes the 5% of a cluster farthest from the
      others, 'strong' the cluster centre;
    * the noise gate is measured as the part of a pixel outside the leading
      uncentred principal components;
    * optional clustering in those components (the manual's dimensionality
      reduction, p. 226) and NMF polishing of the references.

    The mixture is assumed linear: reconstruct with 'mean' accumulation. The
    darkest 5% of the pixels are taken as background. The warnings are
    necessary conditions only, see ``docs/design_notes.md``.

    Parameters
    ----------
    stack : numpy.ndarray
        ``(n_lambda, height, width)``; pixels zero in every channel are not
        measured.
    n_dyes : int
        Number of dyes.
    mode : str, optional
        'weak' (default) or 'strong'.
    noise_gate : float, optional
        Noise multiples a pixel must exceed, by default 3.0; 0 keeps any signal.
    signal_percentile : float, optional
        Percentile of the pixels above the gate dropped as dim, by default 50.0.
    subtract_background : bool, optional
        Subtract the median spectrum of the darkest 5%, by default True.
    pca_components : int or str, optional
        Components to cluster in, at least `n_dyes`; 'auto' uses
        `_ads_auto_components`; None (default) clusters in the channels.
    refine : bool, optional
        Polish the references with `_nmf_refine`, by default True.
    max_samples : int, optional
        Largest pixel sample for the references, by default 200000.
    seed : int, optional
        Seed of the sample and the clustering, by default 0.
    report : callable, optional
        ``report(percent)`` progress callback.

    Returns
    -------
    dict
        'abundance' - ``(n_dyes, height, width)`` float32 dye signal summed over
        channels, clipped at 0; 'clusters' - ``(height, width)`` uint8 strongest
        dye from 1, 0 below the gate; 'references' - ``(n_lambda, n_dyes)``
        spectra of unit sum by peak channel; 'background'; 'explained' - energy
        fraction per component; 'noise'; 'used' - fraction clustered;
        'negative' - percent of the energy clipped; 'stability' - largest angle,
        degrees, between references of two sample halves, NaN if one failed;
        'warnings' - list of str; 'components' - PCA rank or None; 'view' -
        'scores' ``(n, 3)`` float32 projections of about 4000 pixels on the
        three leading uncentred components, 'dye' ``(n,)`` int8 cluster or -1,
        'references' ``(n_dyes, 3)`` directions, 'explained' of the three.

    Raises
    ------
    ValueError
        If the arguments do not fit the stack, nothing is measured or too few
        pixels clear the thresholds.

    """
    if stack.ndim != 3:
        raise ValueError(f'A (n_lambda, height, width) stack is needed, got {stack.shape}')
    n_lambda, height, width = stack.shape
    if mode not in ('weak', 'strong'):
        raise ValueError(f'Unknown mode "{mode}"')
    if not 2 <= n_dyes < n_lambda:
        raise ValueError(f'{n_dyes} dyes do not fit a stack of {n_lambda} channels')
    if pca_components is not None and pca_components != 'auto':
        pca_components = int(np.clip(pca_components, n_dyes, n_lambda - 1))
    report = report if report is not None else (lambda percent: None)
    rng = np.random.default_rng(seed)

    # channel by channel throughout: no float copy of the cube
    measured = np.zeros((height, width), dtype=bool)
    for channel in range(n_lambda):
        measured |= np.asarray(stack[channel]) != 0
        report(int(15 * (channel + 1) / n_lambda))
    index = np.flatnonzero(measured)
    if not index.size:
        raise ValueError('Every pixel of the stack is zero')
    if index.size > max_samples:
        index = np.sort(rng.choice(index, max_samples, replace=False))

    samples = np.empty((index.size, n_lambda), dtype=np.float64)
    for channel in range(n_lambda):
        samples[:, channel] = np.asarray(stack[channel]).ravel()[index]
        report(15 + int(15 * (channel + 1) / n_lambda))
    if not np.isfinite(samples).all():
        raise ValueError('The stack holds non-finite values')
    background = np.zeros(n_lambda)
    if subtract_background:
        total = samples.sum(axis=1)
        background = np.median(samples[total <= np.percentile(total, 5)], axis=0)
        samples -= background
    if pca_components == 'auto':
        # chosen once on the whole sample, so that the two halves below use it too
        pca_components = _ads_auto_components(samples, n_dyes)
    logger.info(f'ADS: {index.size} sample pixels of {measured.sum()} measured, '
                f'{n_dyes} dyes, {mode} mode, {pca_components or "no"} PCA components')

    arguments = (n_dyes, mode, noise_gate, signal_percentile, pca_components, refine)
    fit = _ads_references(samples, *arguments, rng)
    references = fit['references']
    report(50)

    # the same search on two halves: what differs between them is not the sample
    order = rng.permutation(len(samples))
    stability = np.nan
    try:
        first, second = (_ads_references(samples[part], *arguments, rng)['references']
                         for part in (order[:len(order) // 2], order[len(order) // 2:]))
        first = first / np.linalg.norm(first, axis=0)
        second = second / np.linalg.norm(second, axis=0)
        stability = float(np.degrees(np.arccos(np.clip((first.T @ second).max(axis=1), 0, 1))).max())
    except ValueError as exc:
        logger.info(f'ADS: a half of the sample failed: {exc}')
    report(70)

    # abundance and PCA scores from one pass over the stack
    basis = fit['basis']
    weights = np.vstack((np.linalg.pinv(references), basis.T)).astype(np.float32)
    out = np.zeros((len(weights), height, width), dtype=np.float32)
    for channel in range(n_lambda):
        plane = np.asarray(stack[channel], dtype=np.float32) - np.float32(background[channel])
        for k, weight in enumerate(weights[:, channel]):
            out[k] += weight * plane
        report(70 + int(29 * (channel + 1) / n_lambda))
    length = np.sqrt(np.einsum('ijk,ijk->jk', out[n_dyes:], out[n_dyes:]))
    abundance = out[:n_dyes]
    abundance[:, ~measured] = 0
    clipped = np.minimum(abundance, 0)
    energy = np.square(abundance, dtype=np.float64).sum()
    negative = float(100 * np.square(clipped, dtype=np.float64).sum() / max(energy, 1e-12))
    np.maximum(abundance, 0, out=abundance)
    clusters = abundance.argmax(axis=0).astype(np.uint8) + 1
    clusters[~measured | (length <= fit['gate'])] = 0

    unit = references / np.linalg.norm(references, axis=0)
    cosine = unit.T @ unit - 2 * np.eye(n_dyes)
    closest = float(np.degrees(np.arccos(np.clip(cosine.max(), 0, 1))))
    warnings = []
    if not stability <= ADS_UNSTABLE_DEG:      # also when NaN
        warnings.append(f'Unstable: two halves of the sample differ by {stability:.1f} deg')
    if negative > ADS_NEGATIVE_PERCENT:
        warnings.append(f'{negative:.1f}% of the energy is negative: check the background '
                        f'and the number of dyes')
    if closest < ADS_COINCIDE_DEG:
        warnings.append(f'Two spectra differ by {closest:.1f} deg only: too many dyes?')
    report(100)
    logger.info(f'ADS: peaks at channels {references.argmax(axis=0).tolist()}, '
                f'{100 * fit["used"]:.1f}% of the sample clustered, negative {negative:.2f}%, '
                f'stability {stability:.2f} deg')
    return {'abundance': abundance, 'clusters': clusters, 'references': references,
            'background': background, 'explained': fit['explained'], 'noise': fit['noise'],
            'used': fit['used'], 'negative': negative, 'stability': stability,
            'warnings': warnings, 'view': fit['view'], 'components': pca_components}


# --- Spectral resolution and calibration -------------------------------------

def line_fwhm(passes:int, fwhm_one:float, pass_width:float) -> float:
    """ FWHM of a narrow emission line at `passes` passes.

    The aperture is a Gaussian whose sweep over one pass gives `fwhm_one`; the
    result is the FWHM of that Gaussian swept over a box of
    ``passes * pass_width``. Validation: ``docs/wide_lines_validation.md``,
    section 7.1.

    Parameters
    ----------
    passes : int
        Passes of the pattern per exposure, 1 or more.
    fwhm_one : float
        FWHM at one pass, `setup_params.yaml` ``optics.spectral_fwhm``.
    pass_width : float
        Travel of one pass, in the unit of `fwhm_one`.

    Returns
    -------
    float
        FWHM in the unit of `fwhm_one`.

    """
    def swept(sigma, width):
        # Gaussian convolved with a box, solved for its half maximum by bisection
        scale, half_box = math.sqrt(2) * max(sigma, 1e-9), width / 2
        profile = lambda x: math.erf((x + half_box) / scale) - math.erf((x - half_box) / scale)
        half, low, high = profile(0.0) / 2, 0.0, half_box + 5 * scale
        for _ in range(60):
            mid = (low + high) / 2
            low, high = (mid, high) if profile(mid) > half else (low, mid)
        return 2 * low

    low, high = 0.0, fwhm_one
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if swept(mid, pass_width) < fwhm_one else (low, mid)
    return swept(low, passes * pass_width)


def detect_spectral_peaks(y, x=None, window:int=15, poly:int=3,
                          noise_thresh:float=0.02, max_peaks:int=None) -> dict:
    """ Peaks of a spectrum as minima of its Savitzky-Golay second derivative.

    A shoulder on the flank of a stronger peak has no local maximum but has a
    curvature minimum.

    Parameters
    ----------
    y : array_like
        Spectrum.
    x : array_like, optional
        Channel positions, by default the indices.
    window : int, optional
        Filter window in channels, by default 15: shorter finds noise, longer
        merges close emitters.
    poly : int, optional
        Filter polynomial order, by default 3.
    noise_thresh : float, optional
        Smallest spectrum value at a peak, as a fraction of the maximum, by
        default 0.02.
    max_peaks : int, optional
        Keep at most this many, the tallest; None (default) keeps all.

    Returns
    -------
    dict
        'center', 'amplitude' - by position, in the units of `x`; 'n_peaks'.

    """
    from scipy import signal

    y = np.asarray(y, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)

    # Savitzky-Golay differentiates a locally fitted polynomial, so the second
    # derivative survives the noise that a plain difference would amplify
    window = min(int(window) | 1, len(y) - (1 - len(y) % 2))
    curvature = signal.savgol_filter(y, window_length=window, polyorder=poly, deriv=2)

    minima = np.where((curvature[:-2] > curvature[1:-1]) &
                      (curvature[1:-1] < curvature[2:]))[0] + 1
    minima = minima[y[minima] > np.max(y) * noise_thresh]

    if max_peaks is not None and len(minima) > max_peaks:
        minima = np.sort(minima[np.argsort(y[minima])[::-1][:max_peaks]])

    return {'center': x[minima], 'amplitude': y[minima], 'n_peaks': len(minima)}


def spectral_calibration(peak_index, peak_wavelength, n_lambda:int=None,
                         degree:int=1) -> dict:
    """ Polynomial from channel index to wavelength.

    The residuals mean something only with more points than parameters: two
    peaks on a line, or three on a parabola, always give rms 0. Wavelengths
    outside the reference peaks are extrapolated.

    Parameters
    ----------
    peak_index : array_like
        Channel of every identified peak; repeats from several regions add
        degrees of freedom.
    peak_wavelength : array_like
        Known wavelength of each, nm.
    n_lambda : int, optional
        Channels of the stack; gives the 'wavelength' axis.
    degree : int, optional
        Polynomial degree, by default 1; needs ``degree + 1`` distinct indices.

    Returns
    -------
    dict
        'to_nm' - `numpy.poly1d`, 'coef', 'wavelength' (or None), 'dispersion' -
        nm per channel at the middle, 'residual' per peak, 'rms', 'max_error',
        'range' - wavelengths of the first and last channel (or None), 'degree'.

    """
    index = np.asarray(peak_index, dtype=float).ravel()
    wavelength_nm = np.asarray(peak_wavelength, dtype=float).ravel()

    coef = np.polyfit(index, wavelength_nm, degree)
    to_nm = np.poly1d(coef)
    residual = wavelength_nm - to_nm(index)

    axis = np.arange(n_lambda, dtype=float) if n_lambda else None
    wavelength = to_nm(axis) if axis is not None else None
    # local slope in the middle of the calibrated range
    mid = float(np.mean(axis)) if axis is not None else float(np.mean(index))
    dispersion = float(np.polyder(to_nm)(mid))

    logger.info(f'Spectral calibration from {len(index)} peaks, degree {degree}: '
                 f'{dispersion:.3f}nm per channel, fit residual rms {residual.std():.2f}nm, '
                 f'max {np.abs(residual).max():.2f}nm')
    if wavelength is not None:
        logger.info(f'Calibrated axis: {wavelength[0]:.1f}..{wavelength[-1]:.1f}nm '
                     f'over {n_lambda} channels')

    return {'to_nm': to_nm,
            'coef': coef,
            'wavelength': wavelength,
            'dispersion': dispersion,
            'residual': residual,
            'rms': float(residual.std()),
            'max_error': float(np.abs(residual).max()),
            'range': (float(wavelength[0]), float(wavelength[-1])) if wavelength is not None else None,
            'degree': int(degree)}


def fit_spectral_peaks(y, x=None, max_peaks:int=None, sigma_bounds:tuple=None,
                       **detect_kwargs) -> dict:
    """ Bounded fit of a sum of Gaussians, seeded by `detect_spectral_peaks`.

    Amplitudes are non-negative, centres inside the axis, widths inside
    `sigma_bounds`; the width seed is a third of the peak spacing, so the seeds
    do not overlap. A failed fit returns the seeds with 'success' False instead
    of raising.

    Parameters
    ----------
    y : array_like
        Spectrum.
    x : array_like, optional
        Channel positions, by default the indices.
    max_peaks : int, optional
        Largest number of components, the tallest; set it to the number of
        emitters.
    sigma_bounds : tuple of float, optional
        ``(min, max)`` width, by default one step to a quarter of the axis.
    **detect_kwargs
        `detect_spectral_peaks` arguments: `window`, `poly`, `noise_thresh`.

    Returns
    -------
    dict
        'center', 'amplitude', 'sigma' - by position; 'params' - flat, for
        `gaussian_sum`; 'fit' - curve at `x`; 'residual_rms'; 'n_peaks';
        'success'.

    """
    from scipy import optimize

    y = np.asarray(y, dtype=np.float64)
    x = np.arange(len(y), dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)

    seed = detect_spectral_peaks(y, x, max_peaks=max_peaks, **detect_kwargs)
    centre, amplitude = seed['center'], seed['amplitude']
    if seed['n_peaks'] == 0:
        return {'center': centre, 'amplitude': amplitude, 'sigma': np.array([]),
                'params': np.array([]), 'fit': np.zeros_like(y), 'residual_rms': float(y.std()),
                'n_peaks': 0, 'success': False}

    span = float(x.max() - x.min())
    step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    sigma_lo, sigma_hi = sigma_bounds if sigma_bounds else (step, span / 4)

    # a width of a third of the gap to the nearest neighbour keeps the starting
    # components apart; a lone peak gets a sixth of the axis
    gaps = np.diff(centre)
    typical = np.median(gaps) if len(gaps) else span / 2
    sigma_0 = float(np.clip(typical / 3, sigma_lo, sigma_hi))

    p0 = np.ravel([[a, c, sigma_0] for a, c in zip(amplitude, centre)])
    lower = np.ravel([[0.0, x.min(), sigma_lo] for _ in centre])
    upper = np.ravel([[2 * np.max(y), x.max(), sigma_hi] for _ in centre])
    p0 = np.clip(p0, lower + 1e-9, upper - 1e-9)

    try:
        popt, _ = optimize.curve_fit(gaussian_sum, x, y, p0=p0,
                                     bounds=(lower, upper), maxfev=20000)
        success = True
    except (RuntimeError, ValueError) as err:
        logger.warning(f'Gaussian fit did not converge ({err}), returning detection seeds')
        popt, success = p0, False

    order = np.argsort(popt[1::3])
    params = np.ravel(popt.reshape(-1, 3)[order])
    fit = gaussian_sum(x, *params)
    residual_rms = float(np.sqrt(np.mean((y - fit) ** 2)))

    logger.info(f'Fitted {len(order)} Gaussians, centres '
                 f'{np.round(params[1::3], 1).tolist()}, '
                 f'residual rms {residual_rms:.1f} ({100*residual_rms/max(np.max(y), 1):.1f}% of peak), '
                 f'converged {success}')

    return {'center': params[1::3], 'amplitude': params[0::3], 'sigma': params[2::3],
            'params': params, 'fit': fit, 'residual_rms': residual_rms,
            'n_peaks': len(order), 'success': success}


def gaussian_sum(x, *params) -> np.ndarray:
    """ Sum of Gaussians, the model of `fit_spectral_peaks`.

    Parameters
    ----------
    x : array_like
        Positions.
    *params : float
        ``amplitude, centre, sigma`` per component; three values give one.

    Returns
    -------
    numpy.ndarray
        Curve, shape of `x`.

    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude, centre, sigma = params[i:i+3]
        y += amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    return y


# --- Saving ------------------------------------------------------------------

def save_lambda_stack(cube:np.ndarray, path:str, metadata:dict=None,
                      wavelength=None, compression:str='zlib') -> tuple:
    """ Write a stack as an ImageJ TIFF with a JSON sidecar.

    Parameters
    ----------
    cube : numpy.ndarray
        ``(n_lambda, height, width)`` stack, written in ``CYX`` order; float64
        is written as float32.
    path : str
        TIFF path; the sidecar takes the suffix `.json`.
    metadata : dict, optional
        Stored in the sidecar; numpy values become plain lists and numbers.
    wavelength : array_like, optional
        Wavelength per channel, stored in the sidecar and as slice labels.
    compression : str, optional
        `tifffile.imwrite` compression, by default 'zlib'; None writes faster.

    Returns
    -------
    tuple of str
        ``(tiff_path, json_path)``.

    """
    import tifffile

    data = np.asarray(cube)          # already channel-first, ImageJ order
    if data.dtype == np.float64:
        data = data.astype(np.float32)

    tiff_path = Path(path)
    json_path = tiff_path.with_suffix('.json')
    tiff_path.parent.mkdir(parents=True, exist_ok=True)

    tiff_meta = {'axes': 'CYX'}
    if wavelength is not None:
        tiff_meta['Labels'] = [f'{nm:.1f}nm' for nm in np.asarray(wavelength, dtype=float)]

    tifffile.imwrite(tiff_path, data, imagej=True, metadata=tiff_meta,
                     compression=compression)

    def _plain(value):
        """ Make a value JSON-serialisable without losing what it says. """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): _plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_plain(v) for v in value]
        return value

    sidecar = {'shape': list(np.asarray(cube).shape),
               'dtype': str(data.dtype),
               'axes': 'CYX',
               'wavelength': _plain(np.asarray(wavelength, dtype=float)) if wavelength is not None else None,
               'metadata': _plain(metadata or {})}
    json_path.write_text(json.dumps(sidecar, indent=2))

    logger.info(f'Wrote {tiff_path} ({tiff_path.stat().st_size / (1024 * 1024):.1f}MB, '
                 f'{data.shape} {data.dtype}, compression {compression}) '
                 f'and {json_path.name}')
    return str(tiff_path), str(json_path)


# --- Console diagnostics -----------------------------------------------------

def estimate_band_periodicity(img_stack:np.ndarray,
                              strip_center:int=None,
                              strip_width:int=64,
                              smooth_sigma:float=3.0,
                              min_period:float=10.0,
                              min_bands:int=3,
                              harmonic_ratio:float=0.25) -> dict:
    """ Band period and inter-frame drift from raw frames, without the pattern.

    An independent check of a reconstruction:

    1. a narrow column strip of every frame is averaged into a row profile
       (the arcs sag, a wide strip blurs the period);
    2. the profile is mean-subtracted and Hann-windowed;
    3. the strongest FFT peak is replaced by a sub-harmonic that carries more
       than `harmonic_ratio` of it, as bands are not sinusoidal;
    4. the frequency is refined on a continuous grid;
    5. the phase at it, unwrapped across frames, gives the drift.

    Parameters
    ----------
    img_stack : numpy.ndarray
        Frame or ``(n_frames, height, width)`` stack, cropped.
    strip_center : int, optional
        Centre column of the strip, by default the middle.
    strip_width : int, optional
        Strip width, by default 64.
    smooth_sigma : float, optional
        Gaussian sigma on the profile, by default 3.0.
    min_period : float, optional
        Shortest period considered, px, by default 10.0.
    min_bands : int, optional
        Fewest whole bands in the strip, by default 3.
    harmonic_ratio : float, optional
        Sub-harmonic acceptance, by default 0.25.

    Returns
    -------
    dict
        'period' px, 'frequency' - cycles over the strip, 'n_bands', 'snr',
        'shift' - cumulative drift per frame from the first, px, 'step' -
        drift between frames, 'monotone', 'strip' - columns used.

    """
    from scipy import ndimage as ndi

    stack = np.asarray(img_stack, dtype=np.float64)
    if stack.ndim == 2:
        stack = stack[np.newaxis]
    n_frames, n_rows, n_cols = stack.shape

    # analysis strip, clipped to the frame
    center = n_cols // 2 if strip_center is None else int(strip_center)
    col_min = int(np.clip(center - strip_width // 2, 0, max(n_cols - 1, 0)))
    col_max = int(np.clip(col_min + strip_width, 1, n_cols))

    # one mean-subtracted row profile per frame
    profiles = np.empty((n_frames, n_rows))
    for frame_idx in range(n_frames):
        profile = stack[frame_idx, :, col_min:col_max].mean(axis=1)
        profile = ndi.gaussian_filter(profile, sigma=smooth_sigma)
        profiles[frame_idx] = profile - profile.mean()

    # Hann window suppresses the leakage that would otherwise bias a
    # frequency estimate taken between the integer FFT bins
    windowed = profiles * np.hanning(n_rows)

    def _amplitude(freq):
        """ Mean amplitude over frames at arbitrary (non-integer) frequencies. """
        freq = np.atleast_1d(np.asarray(freq, dtype=float))
        kernel = np.exp(-2j * np.pi * np.outer(freq, np.arange(n_rows)) / n_rows)
        return np.abs(kernel @ windowed.T).mean(axis=1)

    # coarse search on the integer grid
    freq_min = max(int(min_bands), 1)
    freq_max = int(n_rows / min_period)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)).mean(axis=0)
    band = np.arange(freq_min, min(freq_max, len(spectrum) - 1) + 1)
    peak_freq = float(band[np.argmax(spectrum[band])])
    background = float(np.median(spectrum[band]))

    # a spectral peak may be a harmonic of the band frequency: walk down to
    # the lowest sub-harmonic that still carries a substantial amplitude
    for divisor in (4, 3, 2):
        candidate = peak_freq / divisor
        if candidate < freq_min:
            continue
        near = band[(band > candidate * 0.94) & (band < candidate * 1.06)]
        if len(near) and spectrum[near].max() > harmonic_ratio * spectrum[band].max():
            logger.info(f'Spectral peak at {peak_freq:.0f} cycles looks like harmonic '
                         f'{divisor} of {candidate:.1f} cycles, using the sub-harmonic')
            peak_freq = float(near[np.argmax(spectrum[near])])
            break

    # continuous refinement around the chosen bin, wide enough to reach the
    # true frequency even when the sub-harmonic sits between two bins
    fine_freq = np.linspace(peak_freq - 1.5, peak_freq + 1.5, 1201)
    fine_freq = fine_freq[fine_freq >= 1.0]
    fine_amp = _amplitude(fine_freq)
    frequency = float(fine_freq[np.argmax(fine_amp)])
    period = n_rows / frequency

    # sub-pixel pattern position from the phase at that frequency
    kernel = np.exp(-2j * np.pi * frequency * np.arange(n_rows) / n_rows)
    phase = np.unwrap(np.angle(kernel @ windowed.T))
    shift = -phase * period / (2 * np.pi)
    shift -= shift[0]
    step = np.diff(shift)

    snr = float(fine_amp.max() / background) if background > 0 else float('inf')
    monotone = bool(np.all(np.sign(step) == np.sign(step.mean()))) if len(step) else None

    logger.info(f'Band periodicity from columns {col_min}:{col_max} of {n_frames} frame(s): '
                 f'period {period:.3f}px, {frequency:.2f} bands across the strip, '
                 f'SNR (band peak / median spectral background) {snr:.1f}')
    if len(step):
        logger.info(f'Pattern drift: {step.mean():+.3f}px/frame (std {step.std():.3f}), '
                     f'range {step.min():+.3f}..{step.max():+.3f}px, one-directional {monotone}, '
                     f'total {shift[-1] / period:+.3f} periods')

    return {'period': float(period),
            'frequency': frequency,
            'n_bands': float(n_rows / period),
            'snr': snr,
            'shift': shift,
            'step': step,
            'monotone': monotone,
            'strip': (col_min, col_max)}


def band_profile_snr(cube:np.ndarray, mask:np.ndarray=None,
                     block_rows:int=256) -> np.ndarray:
    """ Per-pixel SNR of the spectra: highest peak over the median, divided by
    the noise ``1.4826 * median(|diff|) / sqrt(2)``.

    Pure noise reads about 1, not 0; below about 3 is no detection.

    Parameters
    ----------
    cube : numpy.ndarray
        ``(n_lambda, height, width)`` stack.
    mask : numpy.ndarray, optional
        True where a pixel holds data, by default a non-zero spectrum.
    block_rows : int, optional
        Rows per block, by default 256; memory only.

    Returns
    -------
    numpy.ndarray
        ``(height, width)`` float32, NaN where unmeasured or noiseless.

    """
    data = np.asarray(cube)
    height, width = data.shape[1], data.shape[2]
    valid = (np.asarray(mask, dtype=bool) if mask is not None
             else np.any(data != 0, axis=0))

    snr = np.full((height, width), np.nan, dtype=np.float32)
    for row in range(0, height, block_rows):
        block = data[:, row:row + block_rows].astype(np.float32)
        # median of the successive differences ignores the smooth spectral shape
        noise = 1.4826 * np.median(np.abs(np.diff(block, axis=0)), axis=0) / np.sqrt(2)
        signal = block.max(axis=0) - np.median(block, axis=0)
        usable = valid[row:row + block_rows] & (noise > 0)
        snr[row:row + block_rows] = np.where(usable, signal / np.where(noise > 0, noise, 1),
                                             np.nan)

    finite = np.isfinite(snr)
    logger.info(f'Per-pixel spectral SNR (peak over baseline / noise) '
                 f'over {finite.sum()} measured pixels: '
                 f'median {np.median(snr[finite]):.1f}, '
                 f'5-95% {np.percentile(snr[finite], 5):.1f}..'
                 f'{np.percentile(snr[finite], 95):.1f}')
    return snr


def residual_pattern_power(img:np.ndarray, period:float, strip_center:int=None,
                           strip_width:int=64, n_harmonics:int=2) -> dict:
    """ Band pattern left in an image, at the band frequency and its harmonics.

    The row profile of a narrow strip is Hann-windowed and evaluated at the
    band frequency. Compare a reconstruction with a raw frame rather than with
    a fixed threshold. 'power_fraction' cannot exceed about 0.67, the share the
    Hann window leaves at the exact frequency of a pure sinusoid.

    Parameters
    ----------
    img : numpy.ndarray
        2D image, a channel or a projection.
    period : float
        Band period along the rows, px.
    strip_center : int, optional
        Centre column, by default the middle.
    strip_width : int, optional
        Strip width, by default 64.
    n_harmonics : int, optional
        Harmonics included, by default 2.

    Returns
    -------
    dict
        'relative' - modulation over the mean, 'amplitude', 'power_fraction' -
        share of the profile variance at the band frequencies,
        'harmonic_amplitude', 'period', 'strip'.

    """
    data = np.asarray(img, dtype=np.float64)
    n_rows, n_cols = data.shape

    center = n_cols // 2 if strip_center is None else int(strip_center)
    col_min = int(np.clip(center - strip_width // 2, 0, max(n_cols - 1, 0)))
    col_max = int(np.clip(col_min + strip_width, 1, n_cols))

    profile = data[:, col_min:col_max].mean(axis=1)
    mean_level = profile.mean()
    window = np.hanning(n_rows)
    windowed = (profile - mean_level) * window

    # the band frequency in cycles over the strip height, plus its harmonics
    freqs = (n_rows / period) * np.arange(1, n_harmonics + 1)
    kernel = np.exp(-2j * np.pi * np.outer(freqs, np.arange(n_rows)) / n_rows)
    component = np.abs(kernel @ windowed)

    # a windowed sinusoid of amplitude A transforms to A * sum(window) / 2
    harmonic_amplitude = 2 * component / window.sum()
    amplitude = float(harmonic_amplitude[0])

    # share of the profile variance carried by those frequencies
    total_power = float(np.sum(np.abs(np.fft.fft(windowed)) ** 2))
    band_power = float(2 * np.sum(component ** 2))
    power_fraction = band_power / total_power if total_power > 0 else 0.0
    relative = amplitude / mean_level if mean_level > 0 else 0.0

    logger.info(f'Residual band pattern in columns {col_min}:{col_max}: '
                 f'modulation (Fourier amplitude / mean) {relative:.4f}, '
                 f'{100 * power_fraction:.1f}% of the profile variance '
                 f'at the band frequency and {n_harmonics - 1} harmonic(s)')

    return {'relative': relative,
            'amplitude': amplitude,
            'power_fraction': power_fraction,
            'harmonic_amplitude': harmonic_amplitude,
            'period': float(period),
            'strip': (col_min, col_max)}


def peak_drift_map(lambda_stack, mask=None, reference=None, smooth_sigma:float=1.5,
                   min_snr:float=4.0, max_shift:float=None, mode_prominence:float=0.03,
                   block_rows:int=256, plot:bool=True) -> dict:
    """ Sub-channel position of the dominant peak of every pixel and its drift
    from reference peaks.

    The spectrum is smoothed, baseline-subtracted and its maximum refined by a
    parabola through three channels. References are the prominent modes of the
    position histogram unless given; every pixel is assigned to the nearest.
    Where emitters overlap, the maximum follows the locally dominant one, so an
    uneven mixture reads as drift: check with `peak_drift_unmixed`.

    Parameters
    ----------
    lambda_stack : numpy.ndarray
        ``(n_lambda, height, width)`` stack, best gap-filled.
    mask : numpy.ndarray, optional
        True where a pixel holds data.
    reference : array_like or float, optional
        Reference positions, channels; None (default) finds them.
    smooth_sigma : float, optional
        Spectral Gaussian sigma, by default 1.5 channels.
    min_snr : float, optional
        Peak height over the noise of `band_profile_snr`, by default 4.0.
    max_shift : float, optional
        Largest distance to a reference, by default half the smallest gap.
    mode_prominence : float, optional
        Smallest mode prominence as a fraction of the tallest, by default 0.03.
    block_rows : int, optional
        Rows per block, by default 256.
    plot : bool, optional
        Draw `plot_peak_drift`, by default True.

    Returns
    -------
    dict
        'peak', 'shift' - ``(height, width)`` float32, NaN where untested;
        'group' - int8 reference index or -1; 'reference'; 'column', 'row' -
        'median', 'q25', 'q75' and quadratic 'coef' of the shift along each
        axis; 'rms'; 'span' - 5-95 percentile range; 'n_valid'; 'coverage'.

    Raises
    ------
    ValueError
        If no pixel passes the tests.

    """
    from scipy import ndimage as ndi, signal

    data = np.asarray(lambda_stack)
    n_lambda, height, width = data.shape
    peak = np.full((height, width), np.nan, dtype=np.float32)

    for row in range(0, height, block_rows):
        block = data[:, row:row + block_rows].astype(np.float32)
        if smooth_sigma > 0:
            block = ndi.gaussian_filter1d(block, smooth_sigma, axis=0)
        block = block - np.median(block, axis=0)

        top = np.argmax(block, axis=0)
        index = top[np.newaxis]
        left = np.take_along_axis(block, np.clip(index - 1, 0, n_lambda - 1), axis=0)[0]
        centre = np.take_along_axis(block, index, axis=0)[0]
        right = np.take_along_axis(block, np.clip(index + 1, 0, n_lambda - 1), axis=0)[0]

        # vertex of the parabola through the three points, in channel units
        curvature = left - 2 * centre + right
        offset = np.divide(0.5 * (left - right), curvature,
                           out=np.zeros_like(centre), where=curvature < 0)
        noise = 1.4826 * np.median(np.abs(np.diff(block, axis=0)), axis=0) / np.sqrt(2)

        usable = ((top > 0) & (top < n_lambda - 1) & (curvature < 0) &
                  (noise > 0) & (centre > min_snr * noise) & (np.abs(offset) <= 1))
        if mask is not None:
            usable &= np.asarray(mask, dtype=bool)[row:row + block_rows]
        peak[row:row + block_rows] = np.where(usable, top + offset, np.nan)

    finite = np.isfinite(peak)
    if not finite.any():
        raise ValueError('No pixel passed the peak tests - lower min_snr or check the mask!')

    # reference peaks as the modes of the measured distribution
    if reference is None:
        counts, edges = np.histogram(peak[finite], bins=max(n_lambda * 2, 16),
                                     range=(0, n_lambda))
        centres = 0.5 * (edges[:-1] + edges[1:])
        smoothed = ndi.gaussian_filter1d(counts.astype(float), 2.0)
        # prominence rather than curvature: a histogram mode is a genuine hump,
        # and second-derivative detection would also return every shoulder
        modes, properties = signal.find_peaks(
            smoothed, prominence=mode_prominence * smoothed.max(),
            distance=max(len(centres) // n_lambda, 1) * 3)
        if not len(modes):
            modes, properties = np.array([int(np.argmax(smoothed))]), {'prominences': [np.nan]}
        reference = centres[modes]
        logger.info(f'Reference peaks found automatically at '
                     f'{np.round(reference, 2).tolist()} channels, prominence '
                     f'{np.round(np.asarray(properties["prominences"]) / smoothed.max(), 3).tolist()} '
                     f'of the tallest mode - a value near {mode_prominence:.2f} is marginal, '
                     f'pass `reference` explicitly if it matters')
    reference = np.atleast_1d(np.asarray(reference, dtype=float))
    if max_shift is None:
        gaps = np.diff(np.sort(reference))
        max_shift = float(gaps.min() / 2) if len(gaps) else float(n_lambda)

    # assign every measured pixel to its nearest reference
    distance = peak[np.newaxis] - reference[:, np.newaxis, np.newaxis]
    nearest = np.argmin(np.abs(np.where(np.isfinite(distance), distance, np.inf)), axis=0)
    shift = np.take_along_axis(distance, nearest[np.newaxis], axis=0)[0]
    assigned = finite & (np.abs(shift) <= max_shift)
    shift = np.where(assigned, shift, np.nan).astype(np.float32)
    group = np.where(assigned, nearest, -1).astype(np.int8)

    def _profile(axis):
        """ Median and quartiles of the shift collapsed onto one image axis. """
        with warnings.catch_warnings():      # whole rows or columns may be empty
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(shift, axis=axis)
            q25 = np.nanpercentile(shift, 25, axis=axis)
            q75 = np.nanpercentile(shift, 75, axis=axis)
        position = np.arange(len(med), dtype=float)
        good = np.isfinite(med)
        coef = (np.polyfit(position[good], med[good], 2) if good.sum() > 3
                else np.array([0.0, 0.0, 0.0]))
        return {'median': med, 'q25': q25, 'q75': q75, 'coef': coef}

    column, row_profile = _profile(0), _profile(1)
    values = shift[assigned]
    result = {'peak': peak, 'shift': shift, 'group': group, 'reference': reference,
              'column': column, 'row': row_profile,
              'rms': float(np.sqrt(np.mean(values ** 2))),
              'span': float(np.percentile(values, 95) - np.percentile(values, 5)),
              'n_valid': int(assigned.sum()),
              'coverage': float(assigned.mean())}

    logger.info(f'Peak drift over {result["n_valid"]} pixels '
                 f'({100 * result["coverage"]:.1f}% of the frame), '
                 f'{len(reference)} reference peak(s) at '
                 f'{np.round(reference, 2).tolist()} channels')
    logger.info(f'Drift (peak position minus its reference): rms {result["rms"]:.3f}ch, '
                 f'5-95% span {result["span"]:.3f}ch, across columns '
                 f'{np.nanmax(column["median"]) - np.nanmin(column["median"]):.3f}ch, '
                 f'across rows {np.nanmax(row_profile["median"]) - np.nanmin(row_profile["median"]):.3f}ch')

    if plot:
        plot_peak_drift(result)
    return result


def plot_peak_drift(drift:dict, max_int_m:float=None):
    """ Control figure of `peak_drift_map`: drift map, peak histogram with the
    references, and the drift along columns and rows.

    Parameters
    ----------
    drift : dict
        Result of `peak_drift_map`.
    max_int_m : float, optional
        Symmetric colour limit, channels, by default the 98th percentile.

    """
    import matplotlib.pyplot as plt

    shift = drift['shift']
    limit = (max_int_m if max_int_m is not None
             else float(np.nanpercentile(np.abs(shift), 98)) or 1.0)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    im = ax[0, 0].imshow(shift, cmap='coolwarm', vmin=-limit, vmax=limit)
    ax[0, 0].set_title(f'Peak drift, rms {drift["rms"]:.2f}ch')
    ax[0, 0].set_xlabel('Column'); ax[0, 0].set_ylabel('Row')
    plt.colorbar(im, ax=ax[0, 0], label='channels from reference')

    finite = np.isfinite(drift['peak'])
    ax[0, 1].hist(drift['peak'][finite], bins=120, color='0.55')
    for position in drift['reference']:
        ax[0, 1].axvline(position, color='tab:red', lw=1.2, ls='--')
    ax[0, 1].set_xlabel('Peak position, channels')
    ax[0, 1].set_ylabel('Pixels')
    ax[0, 1].set_title(f'{len(drift["reference"])} reference peak(s), dashed')

    for panel, key, label in ((ax[1, 0], 'column', 'Column'), (ax[1, 1], 'row', 'Row')):
        profile = drift[key]
        position = np.arange(len(profile['median']))
        panel.fill_between(position, profile['q25'], profile['q75'],
                           color='tab:blue', alpha=.20, label='interquartile')
        panel.plot(position, profile['median'], color='tab:blue', lw=1.0, label='median')
        panel.plot(position, np.polyval(profile['coef'], position),
                   color='tab:red', lw=1.4, label='quadratic fit')
        panel.axhline(0, color='0.5', lw=0.8, ls='--')
        panel.set_xlabel(label); panel.set_ylabel('Drift, channels')
        panel.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()


def fit_column_drift(drift:dict, reject_sigma:float=3.0) -> dict:
    """ Weighted straight line through the column medians of the peak drift.

    Columns are weighted by their inverse interquartile width squared and, after
    a first pass, outliers beyond `reject_sigma` robust deviations are dropped.
    'curvature' is what a parabola would add: compare
    ``curvature * width**2`` with 'span'. Qualify the drift with
    `peak_drift_unmixed` first, see ``docs/design_notes.md``.

    Parameters
    ----------
    drift : dict
        Result of `peak_drift_map`.
    reject_sigma : float, optional
        Outlier limit, by default 3.0; None fits once.

    Returns
    -------
    dict
        'slope' - channels per column, 'intercept', 'span' - across the width,
        'residual_rms', 'curvature', 'n_columns', 'n_rejected'.

    """
    median = np.asarray(drift['column']['median'], dtype=float)
    position = np.arange(len(median), dtype=float)
    # a column with few measured pixels has a noisy median and must not steer
    # the fit; the interquartile width is the cheapest proxy for that
    spread = np.asarray(drift['column']['q75'], dtype=float) - np.asarray(drift['column']['q25'], dtype=float)
    good = np.isfinite(median) & np.isfinite(spread) & (spread > 0)
    weight = np.where(good, 1.0 / np.maximum(spread, 1e-6) ** 2, 0.0)

    def _weighted_line(keep):
        design = np.vstack([position[keep], np.ones(keep.sum())]).T
        root = np.sqrt(weight[keep])[:, None]
        return np.linalg.lstsq(design * root, median[keep] * root[:, 0], rcond=None)[0]

    keep = good.copy()
    slope, intercept = _weighted_line(keep)
    n_rejected = 0
    if reject_sigma is not None:
        residual = median - (slope * position + intercept)
        scale = 1.4826 * np.nanmedian(np.abs(residual[keep] - np.nanmedian(residual[keep])))
        if scale > 0:
            tight = keep & (np.abs(residual) < reject_sigma * scale)
            if tight.sum() > 3:
                n_rejected = int(keep.sum() - tight.sum())
                keep = tight
                slope, intercept = _weighted_line(keep)

    residual = median[keep] - (slope * position[keep] + intercept)
    quadratic = np.polyfit(position[good], median[good], 2)[0]

    result = {'slope': float(slope), 'intercept': float(intercept),
              'span': float(slope * (len(median) - 1)),
              'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
              'curvature': float(quadratic),
              'n_columns': int(keep.sum()), 'n_rejected': n_rejected}

    logger.info(f'Column drift line: {result["slope"] * 1000:+.4f} channels per 1000 columns, '
                 f'span {result["span"]:+.3f}ch across the frame, residual rms '
                 f'{result["residual_rms"]:.3f}ch over {result["n_columns"]} columns '
                 f'({n_rejected} rejected)')
    logger.info(f'Curvature a parabola would have used: {quadratic:+.3e} ch/column^2, '
                 f'i.e. {quadratic * (len(median) - 1) ** 2:+.3f}ch across the frame '
                 f'against a linear span of {result["span"]:+.3f}ch')
    return result


def correct_column_drift(lambda_stack, fit:dict, block_rows:int=256) -> np.ndarray:
    """ Shift every column along the spectral axis by the fitted line.

    Linear interpolation between channels; channels whose source falls outside
    the axis become zero. Check with `peak_drift_map`: the column span should
    vanish while the per-pixel scatter stays.

    Parameters
    ----------
    lambda_stack : numpy.ndarray
        ``(n_lambda, height, width)`` stack.
    fit : dict
        'slope' and 'intercept', as from `fit_column_drift`.
    block_rows : int, optional
        Rows per block, by default 256.

    Returns
    -------
    numpy.ndarray
        Corrected stack, shape and dtype of the input.

    """
    data = np.asarray(lambda_stack)
    n_lambda, height, width = data.shape
    shift = fit['slope'] * np.arange(width, dtype=np.float32) + fit['intercept']

    out = np.zeros_like(data)
    channel = np.arange(n_lambda, dtype=np.float32)[:, np.newaxis, np.newaxis]
    # source position of every output channel, one offset per column
    source = channel + shift[np.newaxis, np.newaxis, :]
    lower = np.floor(source).astype(np.int32)
    frac = (source - lower).astype(np.float32)
    inside = (lower >= 0) & (lower + 1 < n_lambda)
    low_idx = np.clip(lower, 0, n_lambda - 2)

    for row in range(0, height, block_rows):
        block = data[:, row:row + block_rows].astype(np.float32)
        take_low = np.take_along_axis(block, np.broadcast_to(low_idx, block.shape), axis=0)
        take_high = np.take_along_axis(block, np.broadcast_to(low_idx + 1, block.shape), axis=0)
        value = take_low * (1.0 - frac) + take_high * frac
        value = np.where(inside, value, 0.0)
        if np.issubdtype(data.dtype, np.integer):
            value = np.rint(value)
        out[:, row:row + block_rows] = value.astype(data.dtype)

    lost = float(1.0 - inside.mean())
    logger.info(f'Column drift corrected: shift {shift.min():+.3f}..{shift.max():+.3f} channels '
                 f'across the width, {100 * lost:.2f}% of the stack fell outside the '
                 f'spectral axis and was zeroed')
    return out


def peak_drift_unmixed(lambda_stack, n_peaks:int, tile:int=64, mask=None,
                       min_snr:float=4.0, plot:bool=True, **fit_kwargs) -> dict:
    """ Drift of Gaussian components fitted on square tiles.

    A fitted component centre does not move when its neighbour grows, unlike
    the maximum of `peak_drift_map`. An instrumental shift moves all components
    the same way; slopes that differ in sign or ratio are the sample. Valid only
    for Gaussian-like emission, such as quantum dots.

    Parameters
    ----------
    lambda_stack : numpy.ndarray
        ``(n_lambda, height, width)`` stack.
    n_peaks : int
        Components per tile; tiles that resolve another number are dropped.
    tile : int, optional
        Tile side, px, by default 64.
    mask : numpy.ndarray, optional
        True where a pixel holds data; tiles average masked pixels only.
    min_snr : float, optional
        Tile spectrum over its noise, by default 4.0.
    plot : bool, optional
        Draw `plot_peak_drift_unmixed`, by default True.
    **fit_kwargs
        `fit_spectral_peaks` arguments.

    Returns
    -------
    dict
        'center' - ``(n_peaks, tiles_y, tiles_x)``, NaN where dropped; 'drift' -
        minus the component median; 'reference'; 'column_median' -
        ``(n_peaks, tiles_x)``; 'slope' - channels per image column; 'span';
        'success'; 'tile'; 'coverage'.

    Raises
    ------
    ValueError
        If fewer than 2 x 2 tiles fit or no tile is fitted.

    """
    data = np.asarray(lambda_stack)
    n_lambda, height, width = data.shape
    tiles_y, tiles_x = height // tile, width // tile
    if tiles_y < 2 or tiles_x < 2:
        raise ValueError('Tile is too large for this frame - at least 2x2 tiles are needed!')

    keep = (np.asarray(mask, dtype=bool)[:tiles_y * tile, :tiles_x * tile] if mask is not None
            else np.ones((tiles_y * tile, tiles_x * tile), dtype=bool))
    counts = keep.reshape(tiles_y, tile, tiles_x, tile).sum(axis=(1, 3))

    # bin one channel at a time: the whole stack in float would be gigabytes
    binned = np.zeros((n_lambda, tiles_y, tiles_x), dtype=np.float64)
    for channel in range(n_lambda):
        plane = data[channel, :tiles_y * tile, :tiles_x * tile].astype(np.float64)
        binned[channel] = np.where(keep, plane, 0.0).reshape(
            tiles_y, tile, tiles_x, tile).sum(axis=(1, 3))
    binned /= np.maximum(counts, 1)[np.newaxis]

    centre = np.full((n_peaks, tiles_y, tiles_x), np.nan, dtype=np.float32)
    success = np.zeros((tiles_y, tiles_x), dtype=bool)
    axis = np.arange(n_lambda, dtype=float)
    for row in range(tiles_y):
        for col in range(tiles_x):
            spectrum = binned[:, row, col]
            if counts[row, col] == 0:
                continue
            noise = 1.4826 * np.median(np.abs(np.diff(spectrum))) / np.sqrt(2)
            if noise <= 0 or spectrum.max() - np.median(spectrum) < min_snr * noise:
                continue
            fit = fit_spectral_peaks(spectrum, axis, max_peaks=n_peaks, **fit_kwargs)
            # a tile that resolved a different number of components would put a
            # different emitter into slot k and quietly corrupt the comparison
            if fit['success'] and fit['n_peaks'] == n_peaks:
                centre[:, row, col] = fit['center']
                success[row, col] = True

    if not success.any():
        raise ValueError('No tile produced a complete fit - lower min_snr, raise tile, '
                         'or check that n_peaks matches the sample!')

    reference = np.nanmedian(centre.reshape(n_peaks, -1), axis=1)
    drift = centre - reference[:, np.newaxis, np.newaxis]

    with warnings.catch_warnings():          # whole tile columns may be empty
        warnings.simplefilter('ignore', RuntimeWarning)
        column_median = np.nanmedian(drift, axis=1)

    position = (np.arange(tiles_x) + 0.5) * tile          # image columns
    slope = np.zeros(n_peaks)
    for k in range(n_peaks):
        good = np.isfinite(column_median[k])
        if good.sum() > 2:
            slope[k] = np.polyfit(position[good], column_median[k][good], 1)[0]
    span = slope * (width - 1)

    result = {'center': centre, 'drift': drift, 'reference': reference,
              'column_median': column_median, 'slope': slope, 'span': span,
              'success': success, 'tile': int(tile),
              'coverage': float(success.mean())}

    logger.info(f'Unmixed drift on {tiles_y}x{tiles_x} tiles of {tile}px, '
                 f'{success.sum()} fitted ({100 * result["coverage"]:.1f}%), '
                 f'{n_peaks} components at {np.round(reference, 2).tolist()} channels')
    for k in range(n_peaks):
        logger.info(f'  component {k} (ch {reference[k]:.2f}): column slope '
                     f'{slope[k] * 1000:+.3f} channels per 1000 columns, '
                     f'span {span[k]:+.3f}ch')
    ratio = slope / slope[0] if slope[0] != 0 else np.full(n_peaks, np.nan)
    logger.info(f'  slope ratios {np.round(ratio, 2).tolist()} against the geometric '
                 f'prediction {np.round(reference / reference[0], 2).tolist()} - '
                 f'disagreement in sign or ratio means the sample, not the instrument')

    if plot:
        plot_peak_drift_unmixed(result)
    return result


def plot_peak_drift_unmixed(unmixed:dict, max_int_m:float=None):
    """ Control figure of `peak_drift_unmixed`: one drift map per component and
    their column profiles; parallel profiles mean the instrument.

    Parameters
    ----------
    unmixed : dict
        Result of `peak_drift_unmixed`.
    max_int_m : float, optional
        Symmetric colour limit, channels, by default the 98th percentile.

    """
    import matplotlib.pyplot as plt

    drift, reference = unmixed['drift'], unmixed['reference']
    n_peaks = len(reference)
    limit = (max_int_m if max_int_m is not None
             else float(np.nanpercentile(np.abs(drift), 98)) or 1.0)
    tile = unmixed['tile']
    position = (np.arange(drift.shape[2]) + 0.5) * tile

    fig = plt.figure(figsize=(4.2 * n_peaks, 8))
    grid = fig.add_gridspec(2, n_peaks, height_ratios=[1.15, 1])
    for k in range(n_peaks):
        ax = fig.add_subplot(grid[0, k])
        im = ax.imshow(drift[k], cmap='coolwarm', vmin=-limit, vmax=limit)
        ax.set_title(f'component {k}, ch {reference[k]:.1f}')
        ax.set_xlabel(f'tile column ({tile}px)')
        if k == 0:
            ax.set_ylabel('tile row')
        plt.colorbar(im, ax=ax, label='channels' if k == n_peaks - 1 else None)

    ax = fig.add_subplot(grid[1, :])
    for k in range(n_peaks):
        line, = ax.plot(position, unmixed['column_median'][k], lw=1.2,
                        label=f'component {k} (ch {reference[k]:.1f}), '
                              f'{unmixed["slope"][k] * 1000:+.2f}/1000col')
        ax.plot(position, unmixed['slope'][k] * position +
                np.nanmean(unmixed['column_median'][k] - unmixed['slope'][k] * position),
                color=line.get_color(), lw=1.0, ls='--', alpha=.7)
    ax.axhline(0, color='0.5', lw=0.8, ls='--')
    ax.set_xlabel('Image column')
    ax.set_ylabel('Component drift, channels')
    ax.set_title('Parallel lines mean the instrument, crossing lines mean the sample')
    ax.legend(loc='best', fontsize=9)

    plt.tight_layout()
    plt.show()
