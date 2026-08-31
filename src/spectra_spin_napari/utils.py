""" Shared helpers for the spectra-spin reconstruction modules.

Small, self-contained utilities that are useful to more than one part of the
pipeline, or after it. Nothing here imports `simple_recon` or
`phase_model_recon`, so this module can be used on its own - on frames that
have not been reconstructed yet, or on a cube reconstructed elsewhere.

"""

import json
import logging
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi, optimize, signal
from skimage import io

logging.basicConfig(level=logging.INFO,
                    format='%(name)s | %(levelname)s | %(message)s')


def interpolate_zero_gaps(img:np.ndarray, axis:int=0, mask:np.ndarray=None,
                          extrapolate:bool=False) -> np.ndarray:
    """ Fill the gaps of a sparsely populated image by linear interpolation
    along one axis, treating zero-intensity pixels as missing data.

    A reconstructed spectral frame is not a dense image. Each phase image
    contributes one row per spectral band - the row the pinhole that produced
    that spectrum actually sat on - so a single frame fills roughly one row in
    every band period, and even a whole phase series leaves a few percent of
    the frame empty. Those untouched pixels are exactly zero, which makes them
    easy to recognise and easy to fill.

    The gaps left by that geometry are **horizontal stripes**: whole rows, or
    parts of rows where a band left the field of view. Interpolating down the
    columns therefore closes them with data from the nearest measured rows
    above and below, which is why `axis` defaults to 0.

    Every output pixel that was already non-zero is returned untouched; only
    the gaps are written. Pixels that have measured data on one side only -
    before the first or after the last valid sample of their column - are left
    at zero unless `extrapolate` is set, so no value is invented outside the
    range the instrument actually sampled.

    Parameters
    ----------
    img : numpy.ndarray
        Image to fill, normally one spectral channel of a reconstructed cube.
        Arrays of more than two dimensions are accepted and are filled along
        `axis` as well, but see the memory note below before passing a whole
        stack at once.
    axis : int, optional
        Axis to interpolate along, by default 0 (down the columns, which is
        the direction that closes the row gaps of a reconstruction).
    mask : numpy.ndarray, optional
        Boolean array of the same shape as `img`, True where a pixel holds
        real data. If None (default) validity is taken as ``img != 0``.
        Passing an explicit mask is the way to keep genuinely dark
        measurements from being mistaken for gaps - for `PhaseModelRecon`,
        ``lambda_hit_count > 0`` is exactly that mask.
    extrapolate : bool, optional
        Whether to extend the first and last measured value of each line into
        the gaps beyond them, by default False. The default leaves those
        pixels at zero, which keeps the reconstruction honest about where it
        has no data at all.

    Returns
    -------
    numpy.ndarray
        Filled array, same shape and dtype as `img`. Integer inputs are
        rounded rather than truncated, so a fill sitting between two
        measured levels lands on the nearer of them.

    Notes
    -----
    The implementation is fully vectorised: the nearest valid sample on each
    side is found with a running maximum and a running minimum of the sample
    indices, so cost is a handful of passes over the array rather than one
    interpolation call per line.

    That speed is paid for in memory. Several working arrays the size of the
    input are allocated in float64, so a full ``(88, 2000, 1500)`` stack would
    need tens of gigabytes. Fill a stack channel by channel instead - each
    channel is a few tens of megabytes:

    ``np.stack([interpolate_zero_gaps(stack[k]) for k in
    range(len(stack))])``

    Lines that hold no valid sample at all are returned as they were.

    Examples
    --------
    >>> import numpy as np
    >>> import utils
    >>> img = np.array([[10, 0, 5],
    ...                 [ 0, 0, 0],
    ...                 [ 0, 0, 0],
    ...                 [22, 0, 9]])
    >>> utils.interpolate_zero_gaps(img)
    array([[10,  0,  5],
           [14,  0,  6],
           [18,  0,  8],
           [22,  0,  9]])

    The middle column has no measured sample anywhere and stays empty; the
    outer two are filled linearly between their end points.

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
    logging.info(f'Interpolated {written} of {gaps} empty pixels along axis {axis} '
                 f'({100 * written / max(gaps, 1):.1f}% of the gaps, '
                 f'{100 * (valid.sum() + written) / valid.size:.1f}% of the frame now non-zero)')

    return np.moveaxis(filled, 0, axis).astype(arr.dtype)

def max_pooling2d(image: np.ndarray, pool_size: tuple = (2, 2), stride: tuple = (2, 2)) -> np.ndarray:
    """ Reduce an image by taking the maximum over a sliding window.

    The blunt counterpart to `interpolate_zero_gaps`. Where interpolation
    invents values to close the gaps a sparse reconstruction leaves, pooling
    keeps only measured ones and pays for it in resolution: every output pixel
    is the brightest input pixel of its window, so a window that spans at least
    one filled row always lands on real data.

    That makes the pool height the interesting parameter, and a tall narrow
    window the right shape: the gaps in a reconstruction are horizontal
    stripes, so height buys coverage while width only throws away column
    resolution that was never missing.

    How tall depends on the data, not on a rule. On a full 40-frame reference
    reconstruction 41.7% of pixels carry a band write and the longest run of
    empty rows within a column is 4, so ``(4, 1)`` lifts coverage from 41.7% to
    95.6% at a quarter of the vertical resolution. Going taller barely helps -
    ``(12, 1)`` reaches only 96.4% - because the remaining holes are the top and
    bottom margins the phase series never reached, and no window closes those.
    On a short series the picture is worse: six frames give 6.4% coverage, and
    ``(4, 1)`` only reaches 25.4%. Measure the gap runs of your own data before
    choosing.

    Because the reduction is a maximum, it is biased upwards: it keeps peaks
    and discards everything else in the window, so it is a display and
    screening tool rather than a step to run before quantitative spectroscopy.

    Parameters
    ----------
    image : numpy.ndarray
        Input array of shape ``(height, width)``. Two-dimensional only - pool a
        spectral stack one channel at a time.
    pool_size : tuple of int, optional
        Height and width of the pooling window, by default (2, 2).
    stride : tuple of int, optional
        Row and column step between windows, by default (2, 2). Equal to
        `pool_size` gives non-overlapping windows; smaller values overlap them
        and return a larger output.

    Returns
    -------
    numpy.ndarray
        Pooled array of shape
        ``((height - pool_h) // stride_h + 1, (width - pool_w) // stride_w + 1)``,
        always **float64** whatever the input dtype was. Cast it back if the
        result feeds something that expects the camera's uint16.

    Notes
    -----
    Rows and columns that do not fit a whole window at the end of the image are
    **dropped**, so a 5 x 5 input with a 2 x 2 window and stride 2 returns
    2 x 2, not 3 x 3.

    The implementation is an explicit Python loop over output pixels, which is
    clear but not fast: about 0.83 s for a 2000 x 1500 frame with a 2 x 2
    window, so roughly a minute for an 88-channel cube. That is fine for
    inspecting a few channels and worth replacing with a strided view if a
    whole stack ever has to go through it.

    Examples
    --------
    >>> import utils
    >>> channel = lambda_stack[40]              # 41.7% of pixels carry data
    >>> dense = utils.max_pooling2d(channel, (4, 1), (4, 1))
    >>> dense.shape, (dense > 0).mean()
    ((500, 1500), 0.956)

    The pool height to use is the longest run of empty rows in a column, which
    ``lambda_hit_count`` gives directly:

    >>> hit = pm.lambda_hit_count > 0
    >>> max(np.diff(np.flatnonzero(hit[:, c])).max() for c in range(0, 1500, 37))
    4

    See Also
    --------
    interpolate_zero_gaps : closes the same gaps without losing resolution.

    """
    h, w = image.shape
    ph, pw = pool_size
    sh, sw = stride

    # Calculate output dimensions
    out_h = (h - ph) // sh + 1
    out_w = (w - pw) // sw + 1
    
    # Initialize the output array
    pooled = np.zeros((out_h, out_w))

    for i in range(0, out_h):
        for j in range(0, out_w):
            # Calculate start and end indices for the current window
            y_start = i * sh
            y_end = y_start + ph
            x_start = j * sw
            x_end = x_start + pw
            
            # Extract window and find max
            window = image[y_start:y_end, x_start:x_end]
            pooled[i, j] = np.max(window)

    return pooled

def estimate_band_periodicity(img_stack:np.ndarray,
                              strip_center:int=None,
                              strip_width:int=64,
                              smooth_sigma:float=3.0,
                              min_period:float=10.0,
                              min_bands:int=3,
                              harmonic_ratio:float=0.25) -> dict:
    """ Estimate the spectral band period, and the inter-frame drift of the
    pattern, straight from the raw frames.

    A standalone exploratory tool: it uses no simulated pattern, no
    alignment and nothing from the reconstruction modules, so its answer
    is an independent check on a reconstruction rather than a product of
    it. Useful on a new dataset to confirm the band spacing, to measure
    how far the pattern moves between phase images, and to verify that
    the motion is one-directional before running a batch reconstruction.

    How it works
    ------------
    1. A narrow vertical strip of every frame is averaged along the
       columns into a single row profile. The strip must be narrow
       because the pattern arcs sag across the frame: averaging over the
       full width mixes columns whose lines sit at different rows and
       destroys the periodic signal.
    2. Each profile is mean-subtracted and Hann-windowed, then
       transformed. The windowing is what allows the band frequency to be
       read off at a non-integer number of cycles: on the raw integer FFT
       grid the period of a 2000 px strip jumps in steps of about 4%.
    3. The strongest spectral peak is found on the integer grid, then
       tested against its own sub-harmonics. Spectral bands are not
       sinusoidal - the pinhole spots inside a band can make the second
       harmonic stronger than the fundamental - so if a sub-harmonic
       carries more than `harmonic_ratio` of the peak amplitude, the
       lower frequency is taken as the true band frequency.
    4. The chosen frequency is refined on a fine continuous grid, giving
       the band period with sub-pixel accuracy.
    5. The phase of the transform at that frequency locates the pattern
       to a fraction of a pixel. Unwrapped across frames it becomes the
       cumulative drift, and its difference is the per-frame step.

    Parameters
    ----------
    img_stack : numpy.ndarray
        A single frame (2D) or an ordered stack of phase images
        (3D, frames along axis 0). Frames must already be cropped to the
        region of interest.
    strip_center : int, optional
        Centre column of the analysis strip. If None (default) the middle
        column of the frame is used.
    strip_width : int, optional
        Width of the analysis strip in columns, by default 64. Wider
        strips raise the signal-to-noise ratio but bias the period, since
        the line spacing itself changes across the frame.
    smooth_sigma : float, optional
        Sigma of the Gaussian filter applied to the row profile, by
        default 3.0. Suppresses pixel noise and higher harmonics while
        leaving the band frequency essentially untouched.
    min_period : float, optional
        Shortest band period to consider in pixels, by default 10.0.
    min_bands : int, optional
        Smallest number of whole bands that must fit into the strip
        height, by default 3. Together with `min_period` this bounds the
        frequency search.
    harmonic_ratio : float, optional
        A sub-harmonic replaces the spectral peak if its amplitude
        exceeds this fraction of the peak amplitude, by default 0.25.

    Returns
    -------
    dict
        'period' - band period in pixels,
        'frequency' - band frequency in cycles over the strip height,
        'n_bands' - number of whole bands across the strip,
        'snr' - peak amplitude over the median spectral background,
        'shift' - cumulative pattern drift per frame in pixels,
          relative to the first frame,
        'step' - drift between consecutive frames in pixels,
        'monotone' - whether every step keeps the same sign,
        'strip' - the column range actually used.

    Notes
    -----
    Only the sign and relative values of 'shift' are meaningful across
    frames; the absolute phase origin is arbitrary. A low 'snr', or a
    'monotone' of False on a batch that should be one continuous disk
    sweep, usually means the strip is too wide or sits where the arcs are
    steep - move `strip_center` or reduce `strip_width` and repeat.

    On the reference datasets the period comes out within 0.1% of the
    line spacing that `PhaseModelRecon` fits from the simulated pattern,
    and the drift within 1.5% of the phase step it settles on - two
    routes that share no code agreeing on the same numbers.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage import io
    >>> import utils
    >>> stack = np.stack([io.imread(p)[500:2500, 1000:2500] for p in paths])
    >>> info = utils.estimate_band_periodicity(stack)
    >>> info['period'], info['step'].mean(), info['monotone']
    (88.34, -4.72, True)

    """
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
            logging.info(f'Spectral peak at {peak_freq:.0f} cycles looks like harmonic '
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

    logging.info(f'Band periodicity from columns {col_min}:{col_max} of {n_frames} frame(s): '
                 f'period {period:.3f}px, {frequency:.2f} bands across the strip, '
                 f'SNR (band peak / median spectral background) {snr:.1f}')
    if len(step):
        logging.info(f'Pattern drift: {step.mean():+.3f}px/frame (std {step.std():.3f}), '
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
    """ Per-pixel signal-to-noise ratio of the reconstructed spectra.

    Tells you where in the frame the spectrum is real and where it is noise -
    the question `bands_valid` cannot answer, because a band can be perfectly
    inside the frame and still carry nothing but background.

    For every pixel the spectrum along the leading axis is reduced to two
    numbers. The **signal** is the height of its highest peak above the
    baseline, taken as the median of the spectrum, which is robust to a peak
    occupying a good part of the axis. The **noise** is estimated from the
    successive differences rather than from the spread of the values, so the
    smooth shape of a real spectrum does not inflate it:

    ``sigma = 1.4826 * median(|diff(spectrum)|) / sqrt(2)``

    The 1.4826 converts a median absolute deviation to a standard deviation
    for normal noise, and the square root of two undoes the variance doubling
    that differencing introduces.

    Parameters
    ----------
    cube : numpy.ndarray
        Lambda stack of shape ``(n_lambda, height, width)``, as returned
        by `lambda_stack_recon`.
    mask : numpy.ndarray, optional
        Boolean array of shape ``(height, width)``, True where a pixel holds
        real data. If None (default) a pixel counts as measured when its
        spectrum is not identically zero. `PhaseModelRecon.lambda_hit_count`
        greater than zero is the exact mask.
    block_rows : int, optional
        Number of image rows processed at a time, by default 256. Only
        affects peak memory, not the result.

    Returns
    -------
    numpy.ndarray
        Map of shape ``(height, width)``, float32. Pixels without data, and
        pixels whose noise estimate is zero - a spectrum with no variation at
        all - are **NaN** rather than 0, so that a genuine signal-to-noise of
        zero stays distinguishable from an absent measurement. Use
        `numpy.nanmedian` and friends, and note that `matplotlib.imshow`
        leaves NaN blank.

    Notes
    -----
    Memory is bounded by `block_rows`: the cube is converted to float32 one
    block at a time, so a full ``(88, 2000, 1500)`` stack needs a few hundred
    megabytes rather than the several gigabytes a whole-array cast would take.

    A spectrum sampled below the noise floor gives a ratio near 1, not 0: at
    that point the largest of `n_lambda` noise samples is what the "peak"
    measures. Treat anything under about 3 as no detection.

    Examples
    --------
    >>> snr = utils.band_profile_snr(cube, mask=pm.lambda_hit_count > 0)
    >>> np.nanmedian(snr)
    >>> plt.imshow(snr, cmap='viridis', vmin=0, vmax=np.nanpercentile(snr, 99))

    See Also
    --------
    residual_pattern_power : whether the band pattern survived into the result.

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
    logging.info(f'Per-pixel spectral SNR (peak over baseline / noise) '
                 f'over {finite.sum()} measured pixels: '
                 f'median {np.median(snr[finite]):.1f}, '
                 f'5-95% {np.percentile(snr[finite], 5):.1f}..'
                 f'{np.percentile(snr[finite], 95):.1f}')
    return snr


def residual_pattern_power(img:np.ndarray, period:float, strip_center:int=None,
                           strip_width:int=64, n_harmonics:int=2) -> dict:
    """ Measure how much of the band pattern survived into a reconstructed
    image.

    A reconstruction that worked looks like the sample. A reconstruction that
    did not still carries horizontal striping at the band period, because the
    bands were written to the wrong rows or the phase series never filled the
    gaps evenly. This puts a number on that striping instead of leaving it to
    the eye.

    The column-averaged row profile of a narrow strip is mean-subtracted,
    Hann-windowed and evaluated at the band frequency and its first few
    harmonics. Harmonics are included because a spectral band is not a
    sinusoid: residual pattern shows up at the period **and** at half of it.

    Parameters
    ----------
    img : 2D numpy.ndarray
        Image to test, normally one spectral channel of a reconstructed cube
        or a maximum projection over the channels.
    period : float
        Band period in pixels along the rows. `estimate_band_periodicity`
        measures it from the raw frames; `PhaseModelRecon.median_period`
        carries it from the fitted line family.
    strip_center : int, optional
        Centre column of the analysis strip. If None (default) the middle
        column is used.
    strip_width : int, optional
        Width of the strip in columns, by default 64. Narrow for the same
        reason as in `estimate_band_periodicity`: the pattern arcs sag across
        the frame, so a wide strip averages the striping away and reports a
        residual that is too good.
    n_harmonics : int, optional
        Number of harmonics of the band frequency to include, by default 2.

    Returns
    -------
    dict
        'relative' - modulation depth at the band frequency divided by the
          mean intensity, the headline number,
        'amplitude' - the same modulation in intensity units,
        'power_fraction' - share of the profile variance sitting at the band
          frequency and its harmonics,
        'harmonic_amplitude' - amplitude of each harmonic separately,
        'period', 'strip' - what was actually measured.

    Notes
    -----
    There is no universal threshold, because the number depends on how much
    real horizontal structure the sample has. Use it as a **comparison**: run
    it on a raw frame and on the reconstruction built from it. On the
    reference data a raw frame gives a relative modulation near 1.9 - the
    bands are the dominant structure - so a reconstruction that lands one to
    two orders of magnitude below that has removed the pattern. A value that
    stays close to the raw one means the bands were not removed at all.

    'power_fraction' is the stricter of the two numbers: it asks what share of
    everything varying along the column sits at the band frequency, so real
    sample structure dilutes it while striping drives it up. Its practical
    ceiling is **0.67, not 1**: the Hann window spreads even a perfect
    sinusoid over the neighbouring frequency bins, and evaluating at the exact
    band frequency recovers two thirds of its power. A profile that is nothing
    but band pattern therefore reads about 0.67, and pure noise reads a few
    thousandths.

    Examples
    --------
    >>> raw = utils.residual_pattern_power(frame, period=88.3)
    >>> out = utils.residual_pattern_power(lambda_stack[40], period=88.3)
    >>> out['relative'] / raw['relative']      # how much the pattern shrank

    See Also
    --------
    estimate_band_periodicity : measures the period this function needs.

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

    logging.info(f'Residual band pattern in columns {col_min}:{col_max}: '
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
    """ Position of the dominant emission peak in every pixel, and how far it
    drifts across the field of view.

    A calibrated spectral axis is only worth the assumption behind it: that
    channel `k` means the same wavelength everywhere in the frame. It does not.
    The prism disperses slightly differently across the field, the local band
    period changes from column to column, and the fitted line family carries
    its own sub-pixel residual - all of which move the zero point of the
    normalised spectral coordinate. The result is that one emitter, imaged in
    two corners of the same frame, reports two different channels.

    This measures that directly. For every pixel the spectrum is smoothed,
    baseline-subtracted and its highest peak located to a fraction of a channel
    by fitting a parabola through the maximum and its two neighbours. The map
    of those positions is then compared against **reference peaks** rather than
    against a single number, because a real sample carries several emitters and
    their positions differ for reasons that have nothing to do with the
    instrument.

    Reference peaks are found as the modes of the histogram of all measured
    positions, each pixel is assigned to its nearest mode, and the drift is the
    distance from a pixel to the mode it belongs to. A field that disperses
    identically everywhere gives a drift map of pure noise around zero; a
    smooth gradient across it is instrumental.

    Parameters
    ----------
    lambda_stack : numpy.ndarray
        Lambda stack of shape ``(n_lambda, height, width)``, as returned by
        `lambda_stack_recon`. Gaps are tolerated - empty pixels simply fail the
        signal-to-noise test - but filling them first with
        `interpolate_zero_gaps` gives a denser map.
    mask : numpy.ndarray, optional
        Boolean array of shape ``(height, width)``, True where a pixel holds
        real data. `PhaseModelRecon.lambda_hit_count` greater than zero is the
        exact mask for an unfilled stack.
    reference : array_like or float, optional
        Reference peak positions in channels. If None (default) they are found
        automatically as the prominent modes of the measured distribution.
        Pass them explicitly when the emitters are known and the automatic
        search splits or merges a mode - `fit_spectral_peaks` on a bright ROI
        spectrum is the usual source, and doing so is the single most effective
        way to sharpen the result.
    smooth_sigma : float, optional
        Gaussian sigma applied along the spectral axis before the peak search,
        by default 1.5 channels. Suppresses the pixel noise that would
        otherwise make the argmax jump between neighbouring channels.
    min_snr : float, optional
        A pixel is measured only if its peak stands this many noise sigmas
        above the baseline, by default 4.0. Noise is estimated from successive
        differences exactly as in `band_profile_snr`.
    mode_prominence : float, optional
        Smallest prominence, as a fraction of the tallest mode, at which a
        histogram hump counts as a reference peak, by default 0.03. Raising it
        merges weak emitters into their neighbour, lowering it splits noise off
        as a separate reference; on the bundled data the secondary emitter sits
        at 0.043, so the margin either way is not large. This is the parameter
        to reach for before anything else when the histogram panel looks wrong.
    max_shift : float, optional
        Largest distance from a reference peak, in channels, at which a pixel
        is still attributed to it. If None (default) half the smallest gap
        between references is used, which is the point where the assignment
        would become ambiguous anyway.
    block_rows : int, optional
        Image rows processed at a time, by default 256. Affects peak memory
        only.
    plot : bool, optional
        Draw the four-panel control figure, by default True.

    Returns
    -------
    dict
        'peak' - ``(height, width)`` float32 map of peak position in channels,
          NaN where no peak passed the tests,
        'shift' - the same map minus the reference each pixel was assigned to,
          the headline result,
        'group' - ``(height, width)`` int8 index of that reference, -1 where
          unassigned,
        'reference' - the reference positions actually used,
        'column', 'row' - dicts with 'median', 'q25', 'q75' of the shift along
          each axis, and 'coef' of a quadratic fitted to the median,
        'rms', 'span' - scatter of the shift and its 5-95 percentile range,
        'n_valid', 'coverage' - measured pixels and their fraction.

    Notes
    -----
    The parabolic refinement is what makes this worth doing at all: a plain
    argmax quantises to whole channels, and the drift being looked for is a few
    channels at most. Its cost is that it is only meaningful near a genuine
    maximum, so pixels whose peak sits on the first or last channel, or whose
    curvature is not negative, are rejected rather than reported.

    **What it reads on the reference data.** On the gap-filled 88-channel stack
    of `demo_data/QD_mix` two reference peaks are found automatically, at
    channels 29.25 and 38.25, covering 77.5% of the frame. The drift has an rms
    of 2.36 channels and a 5-95 percentile span of 7.7. Its systematic part is
    strongly asymmetric between the two image axes: the column median runs
    almost linearly from -1.5 to +2.0 channels across the width, while the row
    median is a shallow parabola of about 1.2 channels about the centre. A
    monotone tilt along one axis with mild curvature along the other is the
    signature of a dispersing element imaging the field slightly differently
    across it, not of anything the reconstruction does. At the dispersion of
    the reference calibration those 3.5 channels are on the order of 13 nm.

    **What the map cannot tell you on its own** is whether the drift is
    instrumental or a property of the sample. It follows the highest point of
    each spectrum, and where emitters overlap that point is pulled towards
    whichever of them locally dominates - so an unevenly deposited mixture
    reads as a drift. On the bundled data this artefact is larger than the real
    effect and even reverses its sign. Use `peak_drift_unmixed` to settle it
    before acting on anything here.

    Examples
    --------
    >>> stack = pm.lambda_stack_recon()
    >>> drift = utils.peak_drift_map(stack, mask=pm.lambda_hit_count > 0)
    >>> drift['rms'], drift['span']
    >>> drift['column']['coef']            # quadratic in column index

    See Also
    --------
    fit_spectral_peaks : per-spectrum peak fitting, for reference positions.
    spectral_calibration : turns channel positions into wavelengths.
    band_profile_snr : the same noise estimator, as a per-pixel quality map.

    """
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
        logging.info(f'Reference peaks found automatically at '
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

    logging.info(f'Peak drift over {result["n_valid"]} pixels '
                 f'({100 * result["coverage"]:.1f}% of the frame), '
                 f'{len(reference)} reference peak(s) at '
                 f'{np.round(reference, 2).tolist()} channels')
    logging.info(f'Drift (peak position minus its reference): rms {result["rms"]:.3f}ch, '
                 f'5-95% span {result["span"]:.3f}ch, across columns '
                 f'{np.nanmax(column["median"]) - np.nanmin(column["median"]):.3f}ch, '
                 f'across rows {np.nanmax(row_profile["median"]) - np.nanmin(row_profile["median"]):.3f}ch')

    if plot:
        plot_peak_drift(result)
    return result


def plot_peak_drift(drift:dict, max_int_m:float=None):
    """ Four-panel control figure for `peak_drift_map`.

    Parameters
    ----------
    drift : dict
        Whatever `peak_drift_map` returned.
    max_int_m : float, optional
        Symmetric colour limit of the drift map in channels. If None (default)
        the 98th percentile of the absolute shift is used, which keeps a few
        outliers from flattening the whole image.

    Notes
    -----
    How to read the four panels:

    * **Drift map** - the answer. Pure speckle around zero means the spectral
      axis is the same everywhere. A smooth left-to-right or top-to-bottom
      gradient is instrumental and is what a correction would remove. Sharp
      patches that follow the sample rather than the geometry are emitters,
      not drift.
    * **Peak histogram** - the sanity check that has to pass first. The
      reference peaks must sit on well separated modes. Two modes merging into
      one hump, or one emitter split across two, makes every number in the
      other panels meaningless.
    * **Along columns** and **along rows** - the systematic part, with the
      interquartile band showing how much of the scatter is per-pixel noise.
      A trend that clears the band is real; one buried inside it is not.

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
    """ Fit a straight line to the column dependence of the peak drift.

    Deliberately linear and deliberately one-dimensional. The drift measured by
    `peak_drift_map` is dominated by a monotone tilt across the columns, and a
    tilt is what a dispersing element mounted at a slightly wrong angle to the
    sensor produces: every column sees the beam at a marginally different
    incidence, so the wavelength that lands on a given fraction of a band walks
    steadily from one side of the frame to the other. Fitting anything richer
    than a line - a quadratic, or a two-dimensional surface - buys residual at
    the cost of absorbing sample structure, which is the failure mode that
    matters here.

    The fit is weighted by how many pixels each column actually measured, and
    columns whose median sits more than `reject_sigma` robust deviations from
    a first pass are dropped and the line refitted. Both guards exist for the
    same reason: the left and right margins of a reconstruction hold the fewest
    pixels and the largest leverage.

    Parameters
    ----------
    drift : dict
        Whatever `peak_drift_map` returned.
    reject_sigma : float, optional
        Columns further than this many robust deviations from the first-pass
        line are excluded from the second, by default 3.0. Set to None to fit
        in one pass.

    Returns
    -------
    dict
        'slope' - channels per column,
        'intercept' - channels at column 0,
        'span' - modelled drift across the whole width, the headline number,
        'residual_rms' - scatter of the column medians about the line,
        'curvature' - the quadratic coefficient a parabola would have taken,
          reported only so that ignoring it stays an informed choice,
        'n_columns', 'n_rejected' - columns used and dropped.

    Notes
    -----
    'curvature' is the honesty check on the linear choice. Multiply it by the
    square of the frame width: if the result is small next to 'span', the line
    is the whole story. If it is comparable, the tilt hypothesis is incomplete
    and the residual should be looked at before correcting anything.

    **Qualify the drift before treating this as a calibration.** The fit itself
    is sound - on a synthetic stack carrying a known tilt it recovers the slope
    to 0.14% and removes 99.7% of the span - but whether the tilt it finds
    belongs to the instrument is a separate question, and `peak_drift_map`
    alone cannot answer it. Run `peak_drift_unmixed` first.

    The reason is spectral overlap. Following the highest point of a spectrum
    measures where the *mixture* peaks, not where any emitter sits, so a sample
    whose species are laid down unevenly produces an apparent tilt of its own.
    On the bundled `QD_mix` data that artefact is larger than the real effect:
    from raw peak positions the two apparent modes drift in opposite directions
    at a ratio of -4.31, while unmixing the same reconstruction into its three
    Gaussian components puts all of them on the **same** slope, +7.05, +7.93
    and +8.52 channels per 1000 columns. The opposite-sign reading was the
    overlap, not the optics.

    Once unmixed, the drift behaves like an instrument: the slope holds to three
    decimals across tile sizes of 48, 64 and 96 px, agrees to 0.3% between the
    18-frame and 40-frame series of the same field, and moves only about 7% when
    the crop shifts by 83 columns - against a factor of three for the raw
    estimate. Its span is near 11.7 channels across a 1500 px frame.

    The component ratios say what kind of error it is. A mis-scaled spectral
    coordinate - a wrong band period, a normalisation error - would move the
    components in the ratio of their channel indices, here 1.00 : 1.62 : 2.38.
    The measured ratio is 1.00 : 1.12 : 1.21, close to the 1 : 1 : 1 of a
    **rigid** channel offset that grows with column. That is a tilt between the
    dispersion axis and the pattern, with a smaller wavelength-proportional term
    on top. Fitting a straight line is therefore the right model, and the
    residual proportional part is what the line leaves behind.

    Worth noting that the fitted `theta` on these runs is about 0.001 degrees
    while the rigid part of the drift implies 0.34 degrees. The line family is
    fitted to the illumination pattern; the direction the prism disperses along
    is a separate axis the reconstruction never measures, so a mismatch of that
    size is invisible to the geometry fit and is not evidence of a bad one.

    Still open: everything above rests on one specimen. Confirming it needs the
    same unmixed test on a second slide carrying the **same** emitters - a
    sample with different fluorophores cannot be compared, and one whose
    emission is not a sum of Gaussians cannot be unmixed at all.

    See Also
    --------
    peak_drift_map : produces the input.
    correct_column_drift : applies the fitted line to a lambda stack.

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

    logging.info(f'Column drift line: {result["slope"] * 1000:+.4f} channels per 1000 columns, '
                 f'span {result["span"]:+.3f}ch across the frame, residual rms '
                 f'{result["residual_rms"]:.3f}ch over {result["n_columns"]} columns '
                 f'({n_rejected} rejected)')
    logging.info(f'Curvature a parabola would have used: {quadratic:+.3e} ch/column^2, '
                 f'i.e. {quadratic * (len(median) - 1) ** 2:+.3f}ch across the frame '
                 f'against a linear span of {result["span"]:+.3f}ch')
    return result


def correct_column_drift(lambda_stack, fit:dict, block_rows:int=256) -> np.ndarray:
    """ Resample every spectrum so that the fitted column tilt is removed.

    Each column is shifted along the spectral axis by the amount
    `fit_column_drift` modelled for it, by linear interpolation between
    channels. The shift depends on the column only, so a whole column of the
    frame moves by one number and no spatial structure is touched.

    Parameters
    ----------
    lambda_stack : numpy.ndarray
        Lambda stack of shape ``(n_lambda, height, width)``.
    fit : dict
        Whatever `fit_column_drift` returned; only 'slope' and 'intercept' are
        used, so a hand-written ``{'slope': ..., 'intercept': ...}`` works too.
    block_rows : int, optional
        Image rows processed at a time, by default 256.

    Returns
    -------
    numpy.ndarray
        Corrected stack, same shape and dtype as the input. Channels whose
        source position falls outside the original axis are set to **zero**,
        which loses up to `span` channels at one end of the spectrum - the
        price of not inventing data beyond what was measured.

    Notes
    -----
    Interpolating a spectrum that is already the product of one interpolation
    inside the reconstruction blurs it slightly a second time. For a shift of a
    few channels on an axis of ninety that is a small price, but it is the
    reason this belongs at the end of the pipeline rather than in the middle:
    applied to `t_axis` during band sampling it would cost nothing extra, at
    the price of coupling the correction to the reconstruction.

    Verify by running `peak_drift_map` again on the result. The column span
    should collapse towards zero while the row profile and the per-pixel
    scatter stay where they were - if the scatter drops too, the correction is
    absorbing something other than a tilt.

    Examples
    --------
    >>> drift = utils.peak_drift_map(stack, plot=False)
    >>> line = utils.fit_column_drift(drift)
    >>> fixed = utils.correct_column_drift(stack, line)
    >>> utils.peak_drift_map(fixed, reference=drift['reference'])

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
    logging.info(f'Column drift corrected: shift {shift.min():+.3f}..{shift.max():+.3f} channels '
                 f'across the width, {100 * lost:.2f}% of the stack fell outside the '
                 f'spectral axis and was zeroed')
    return out

def peak_drift_unmixed(lambda_stack, n_peaks:int, tile:int=64, mask=None,
                       min_snr:float=4.0, plot:bool=True, **fit_kwargs) -> dict:
    """ Drift of **unmixed** emission components across the field of view.

    The control that `peak_drift_map` needs. That function follows the highest
    point of each pixel's spectrum, and where emitters overlap spectrally the
    highest point is not any emitter's centre - it is pulled towards whichever
    of them locally dominates. A sample whose species are laid down unevenly
    then produces an apparent drift that has nothing to do with the instrument.
    Fitting a fixed number of Gaussians and following the **fitted centre of a
    named component** removes that mechanism: a component's centre does not
    move when its neighbour grows.

    The price is that a bounded multi-Gaussian fit cannot run three million
    times, so the frame is binned into square tiles and one spectrum is fitted
    per tile. That trades spatial resolution, which the drift does not need,
    for the signal-to-noise ratio the fit does.

    Only use this where the emission really is a sum of Gaussians. Quantum dots
    are close enough; most fluorescent proteins and dyes are not, and on those
    the fitted centres would be a smooth function of the fit's own failure
    rather than of the sample.

    Parameters
    ----------
    lambda_stack : numpy.ndarray
        Lambda stack of shape ``(n_lambda, height, width)``.
    n_peaks : int
        Number of emitters to unmix. A tile that does not yield exactly this
        many components is dropped rather than reported, which keeps component
        `k` the same emitter in every tile.
    tile : int, optional
        Side of the square binning tile in pixels, by default 64. Larger tiles
        fit more reliably and resolve the drift more coarsely; the drift being
        looked for is a smooth gradient, so coarse is cheap.
    mask : numpy.ndarray, optional
        Boolean ``(height, width)``, True where a pixel holds real data. Tiles
        average only over masked pixels.
    min_snr : float, optional
        A tile is fitted only if its mean spectrum clears this many noise
        sigmas, by default 4.0. Noise is estimated as in `band_profile_snr`.
    plot : bool, optional
        Draw the control figure, by default True.
    **fit_kwargs
        Passed to `fit_spectral_peaks` - `sigma_bounds`, `window`, `poly`,
        `noise_thresh`.

    Returns
    -------
    dict
        'center' - ``(n_peaks, tiles_y, tiles_x)`` fitted centres in channels,
          NaN where the tile was dropped,
        'drift' - the same minus each component's own median, the result,
        'reference' - ``(n_peaks,)`` median centre of every component,
        'column_median' - ``(n_peaks, tiles_x)`` drift collapsed onto columns,
        'slope' - ``(n_peaks,)`` channels per **image** column, from a straight
          fit to that profile,
        'span' - ``(n_peaks,)`` the same across the full frame width,
        'success' - ``(tiles_y, tiles_x)`` which tiles were used,
        'tile', 'coverage'.

    Notes
    -----
    **How to read 'slope'.** A genuine instrument effect - a dispersing element
    at a slightly wrong angle, a varying incidence across the field - shifts the
    wavelength that lands on a given fraction of a band. Every component
    therefore moves the **same way**, and a geometric error in the normalised
    spectral coordinate moves them in the ratio of their channel indices. Slopes
    that disagree in sign, or in ratio, are the sample.

    This is the test that settles what `peak_drift_map` alone cannot, and it
    should be run before `fit_column_drift` is ever used as a calibration.

    Examples
    --------
    >>> stack = pm.lambda_stack_recon()
    >>> unmixed = utils.peak_drift_unmixed(stack, n_peaks=3, tile=64)
    >>> unmixed['slope'] * 1000            # channels per 1000 image columns
    >>> unmixed['reference']               # where each component sits

    See Also
    --------
    peak_drift_map : the per-pixel version, faster and overlap-sensitive.
    fit_spectral_peaks : the bounded fit used on every tile.
    fit_column_drift : the linear model this test qualifies.

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

    logging.info(f'Unmixed drift on {tiles_y}x{tiles_x} tiles of {tile}px, '
                 f'{success.sum()} fitted ({100 * result["coverage"]:.1f}%), '
                 f'{n_peaks} components at {np.round(reference, 2).tolist()} channels')
    for k in range(n_peaks):
        logging.info(f'  component {k} (ch {reference[k]:.2f}): column slope '
                     f'{slope[k] * 1000:+.3f} channels per 1000 columns, '
                     f'span {span[k]:+.3f}ch')
    ratio = slope / slope[0] if slope[0] != 0 else np.full(n_peaks, np.nan)
    logging.info(f'  slope ratios {np.round(ratio, 2).tolist()} against the geometric '
                 f'prediction {np.round(reference / reference[0], 2).tolist()} - '
                 f'disagreement in sign or ratio means the sample, not the instrument')

    if plot:
        plot_peak_drift_unmixed(result)
    return result


def plot_peak_drift_unmixed(unmixed:dict, max_int_m:float=None):
    """ Control figure for `peak_drift_unmixed`.

    One drift map per unmixed component along the top, and their column
    profiles together underneath so the slopes can be compared directly.

    Parameters
    ----------
    unmixed : dict
        Whatever `peak_drift_unmixed` returned.
    max_int_m : float, optional
        Symmetric colour limit in channels. If None (default) the 98th
        percentile of the absolute drift is used.

    Notes
    -----
    The lower panel is the whole point. Lines that run parallel, in the ratio
    of the components' channel indices, mean a spectral axis that really does
    tilt across the field. Lines that cross, or run the other way, mean the
    apparent drift belongs to the specimen.

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

def save_lambda_stack(cube:np.ndarray, path:str, metadata:dict=None,
                      wavelength=None, compression:str='zlib') -> tuple:
    """ Write a hyperspectral cube to disk as a TIFF, with its metadata
    alongside.

    A reconstruction takes minutes and produces hundreds of megabytes, so it
    should not have to be repeated every time the data is looked at. The cube
    is written in ``(channel, row, column)`` order, which is what ImageJ and
    Fiji expect from a stack, and everything needed to interpret it goes into
    a JSON file next to it.

    Parameters
    ----------
    cube : numpy.ndarray
        Lambda stack of shape ``(n_lambda, height, width)``, as returned by
        `lambda_stack_recon`.
    path : str
        Destination of the TIFF. The metadata sidecar takes the same name with
        a `.json` suffix.
    metadata : dict, optional
        Anything worth keeping with the data - the shared geometry, the fitted
        phase, the crop, the input paths. NumPy arrays and scalars are
        converted to plain lists and numbers so the file stays readable.
    wavelength : array_like, optional
        Calibrated wavelength of each channel, from
        ``spectral_calibration(...)['wavelength']``. Stored in the sidecar and
        written into the TIFF so ImageJ labels the slices.
    compression : str, optional
        Passed to `tifffile.imwrite`, by default 'zlib'. A cube straight out
        of `lambda_stack_recon` is mostly zeros and compresses several-fold;
        pass None to trade the file size for write speed.

    Returns
    -------
    tuple of str
        Paths actually written, ``(tiff_path, json_path)``.

    Notes
    -----
    The channel axis is moved to the front on write, so the file is a stack of
    `n_lambda` images rather than an image of `n_lambda`-element vectors.
    Reading it back gives that order too:

    ``lambda_stack = tifffile.imread(path)`` - the axis order round-trips

    ImageJ TIFF supports uint8, uint16 and float32; a float64 cube is written
    as float32, which costs nothing since the reconstruction is uint16 to
    begin with.

    Examples
    --------
    >>> cal = utils.spectral_calibration([12.4, 38.1, 71.9], [525, 585, 659], n_lambda=88)
    >>> utils.save_lambda_stack(cube, 'output/qd_mix.tiff',
    ...                         metadata={'geometry': pm.geometry,
    ...                                   'phase': pm.phase,
    ...                                   'median_period': pm.median_period},
    ...                         wavelength=cal['wavelength'])
    ('output/qd_mix.tiff', 'output/qd_mix.json')

    """
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

    logging.info(f'Wrote {tiff_path} ({tiff_path.stat().st_size / (1024 * 1024):.1f}MB, '
                 f'{data.shape} {data.dtype}, compression {compression}) '
                 f'and {json_path.name}')
    return str(tiff_path), str(json_path)

def detect_spectral_peaks(y, x=None, window:int=15, poly:int=3,
                          noise_thresh:float=0.02, max_peaks:int=None) -> dict:
    """ Locate emission peaks in a spectrum through second-derivative minima.

    Overlapping emitters do not each produce a local maximum - a weak
    shoulder on the flank of a strong peak never turns the slope around - but
    they do each produce a minimum of the second derivative. Differentiating a
    noisy spectrum directly would be hopeless, so the derivative is taken with
    a Savitzky-Golay filter, which fits a local polynomial and differentiates
    that instead.

    Parameters
    ----------
    y : array_like
        Spectrum, one value per spectral channel.
    x : array_like, optional
        Positions of the channels. If None (default) the channel indices are
        used; pass a calibrated axis to work directly in nanometres.
    window : int, optional
        Length of the Savitzky-Golay window in channels, by default 15. It
        sets the smallest structure that survives: too short and noise
        produces spurious peaks, too long and close emitters merge.
    poly : int, optional
        Polynomial order of the filter, by default 3.
    noise_thresh : float, optional
        A candidate is kept only if the spectrum there exceeds this fraction
        of its maximum, by default 0.02. Rejects minima that sit in the
        baseline.
    max_peaks : int, optional
        Keep at most this many candidates, the tallest first. If None
        (default) every candidate is kept.

    Returns
    -------
    dict
        'center' - positions of the detected peaks, in the units of `x`,
        'amplitude' - spectrum value at each of them,
        'n_peaks' - how many were kept.
        Peaks are ordered by position, not by height.

    Notes
    -----
    This step is the reliable half of peak analysis. On the reference ROI
    spectra it returns three peaks with spacings of 18-19 and 19-21 channels
    for every region of interest, which is the consistency the emitters
    actually have. What follows it - the least-squares fit - is where the
    result can still be lost, which is what `fit_spectral_peaks` guards
    against.

    See Also
    --------
    fit_spectral_peaks : refines these positions with a bounded Gaussian fit.

    """
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
    """ Turn spectral channel indices into wavelengths.

    A reconstructed cube has an uncalibrated spectral axis: channel `k` is a
    fraction of a band, not a wavelength. Given a few reference emitters whose
    peaks are known - quantum dots at 525, 585 and 659 nm on the bundled data -
    fitting channel index against known wavelength calibrates the whole axis.

    Parameters
    ----------
    peak_index : array_like
        Channel index of every identified peak. Repeated measurements of the
        same emitter from different regions of interest are welcome and are
        simply extra points in the fit.
    peak_wavelength : array_like
        Known wavelength in nanometres of each entry of `peak_index`, same
        length.
    n_lambda : int, optional
        Number of spectral channels of the cube. If given, the returned dict
        carries a ready `wavelength` axis for ``numpy.arange(n_lambda)``.
    degree : int, optional
        Degree of the polynomial fitted from index to wavelength, by default
        1. Prism dispersion is not exactly linear, so degree 2 is worth trying
        once there are enough reference peaks - it needs at least `degree + 1`
        distinct indices.

    Returns
    -------
    dict
        'to_nm' - `numpy.poly1d` mapping channel index to nanometres,
        'coef' - its coefficients, highest power first,
        'wavelength' - the calibrated axis if `n_lambda` was given, else None,
        'dispersion' - nanometres per channel at the middle of the axis,
        'residual' - fit residual at each reference peak, in nanometres,
        'rms', 'max_error' - summary of those residuals,
        'range' - wavelength of the first and last channel if `n_lambda` was
          given,
        'degree'.

    Notes
    -----
    The residuals are the honest measure of the calibration, but **only when
    there are more reference points than the fit has parameters**. Two peaks
    and a straight line, or three peaks and a parabola, leave no degrees of
    freedom and return an rms of exactly zero - which says nothing at all. With
    three emitters and a straight line one degree of freedom is left, so an rms
    of a nanometre or two means the peaks and the linear model agree, while an
    rms of tens of nanometres means either a peak was misidentified or the
    dispersion is not linear over that range. Repeating the measurement over
    several regions of interest is the cheap way to get real degrees of
    freedom.

    The fit is only as good as its span: wavelengths outside the range of
    `peak_wavelength` are extrapolated, and a degree of 2 or more extrapolates
    badly. Check `range` against the emitters actually used.

    Examples
    --------
    >>> cal = utils.spectral_calibration(peak_index=[12.4, 38.1, 71.9],
    ...                                  peak_wavelength=[525, 585, 659],
    ...                                  n_lambda=88)
    >>> cal['dispersion']          # nm per channel
    >>> plt.plot(cal['wavelength'], spectrum)
    >>> cal['to_nm'](40.0)         # wavelength of channel 40

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

    logging.info(f'Spectral calibration from {len(index)} peaks, degree {degree}: '
                 f'{dispersion:.3f}nm per channel, fit residual rms {residual.std():.2f}nm, '
                 f'max {np.abs(residual).max():.2f}nm')
    if wavelength is not None:
        logging.info(f'Calibrated axis: {wavelength[0]:.1f}..{wavelength[-1]:.1f}nm '
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
    """ Fit a sum of Gaussians to a spectrum, seeded by second-derivative
    peak detection.

    Unmixing overlapping emitters means fitting three free parameters per
    peak, and an unconstrained least-squares fit of nine or more coupled
    parameters is easy to lose: two components slide onto each other, one
    absorbs the other's amplitude, and the result is a perfect fit to the data
    that says nothing about the emitters. This function keeps the same model
    but removes the freedom that makes the failure possible.

    Three guards do the work:

    * **every parameter is bounded** - amplitudes stay non-negative, centres
      stay inside the spectral axis, and widths stay between `sigma_bounds`,
    * **the width seed comes from the peak spacing** rather than a fixed
      number, so the initial components are narrow enough not to overlap
      before the fit starts,
    * **failure is reported, not raised** - if the optimiser does not
      converge the detected positions come back with ``success`` False,
      which is far more useful than an exception in the middle of a loop over
      regions of interest.

    Parameters
    ----------
    y : array_like
        Spectrum, one value per spectral channel.
    x : array_like, optional
        Positions of the channels, channel indices if None (default).
    max_peaks : int, optional
        Fit at most this many components, the tallest first. Setting it to the
        number of emitters actually in the sample is the single most effective
        thing you can do for stability.
    sigma_bounds : tuple of float, optional
        Allowed ``(min, max)`` width. If None (default) the range is one
        sample to a quarter of the axis, which excludes both a spike on a
        single channel and a component broad enough to act as a baseline.
    **detect_kwargs
        Passed to `detect_spectral_peaks` - `window`, `poly`, `noise_thresh`.

    Returns
    -------
    dict
        'center', 'amplitude', 'sigma' - fitted parameters, one entry per
        component, ordered by position,
        'params' - the same values flattened for `gaussian_sum`,
        'fit' - the fitted curve sampled at `x`,
        'residual_rms' - root mean square of data minus fit,
        'n_peaks' - number of components,
        'success' - whether the optimiser converged; when False the other
        entries hold the detection seeds rather than a fit.

    Notes
    -----
    Bounded fitting uses a trust-region method rather than
    Levenberg-Marquardt, which is slower per call and immune to the runaway
    that loses a component. On the reference ROI spectra the unbounded version
    failed on three regions of five - two peaks collapsing to within one
    channel - while the bounded one recovers consistent positions on all five.

    Examples
    --------
    >>> fit = utils.fit_spectral_peaks(roi_spectrum, max_peaks=3)
    >>> fit['center']
    array([28.1, 46.9, 65.2])
    >>> cal = utils.spectral_calibration(fit['center'], [525, 585, 659], n_lambda=88)

    See Also
    --------
    detect_spectral_peaks : the seeding step, useful on its own.
    gaussian_sum : evaluates the fitted model.
    spectral_calibration : turns fitted centres into a wavelength axis.

    """
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
        logging.warning(f'Gaussian fit did not converge ({err}), returning detection seeds')
        popt, success = p0, False

    order = np.argsort(popt[1::3])
    params = np.ravel(popt.reshape(-1, 3)[order])
    fit = gaussian_sum(x, *params)
    residual_rms = float(np.sqrt(np.mean((y - fit) ** 2)))

    logging.info(f'Fitted {len(order)} Gaussians, centres '
                 f'{np.round(params[1::3], 1).tolist()}, '
                 f'residual rms {residual_rms:.1f} ({100*residual_rms/max(np.max(y), 1):.1f}% of peak), '
                 f'converged {success}')

    return {'center': params[1::3], 'amplitude': params[0::3], 'sigma': params[2::3],
            'params': params, 'fit': fit, 'residual_rms': residual_rms,
            'n_peaks': len(order), 'success': success}

def gaussian_sum(x, *params) -> np.ndarray:
    """ Evaluate a sum of Gaussians.

    The model behind `fit_spectral_peaks`, exposed separately so that a fit
    can be drawn, or a single component of it isolated, without re-deriving
    the formula.

    Parameters
    ----------
    x : array_like
        Positions to evaluate at, normally spectral channel indices.
    *params : float
        Flat parameter list, three per component in the order
        ``amplitude, centre, sigma``. Passing a slice of three reproduces one
        component on its own.

    Returns
    -------
    numpy.ndarray
        The summed curve, same shape as `x`.

    Examples
    --------
    >>> fit = utils.fit_spectral_peaks(spectrum)
    >>> plt.plot(utils.gaussian_sum(np.arange(len(spectrum)), *fit['params']))
    >>> for p in fit['params'].reshape(-1, 3):          # one peak at a time
    ...     plt.plot(utils.gaussian_sum(np.arange(len(spectrum)), *p), '--')

    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude, centre, sigma = params[i:i+3]
        y += amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2)
    return y