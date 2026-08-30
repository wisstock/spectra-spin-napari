""" Shared helpers for the spectra-spin reconstruction modules.

Small, self-contained utilities that are useful to more than one part of the
pipeline, or after it. Nothing here imports `simple_recon` or
`phase_model_recon`, so this module can be used on its own - on frames that
have not been reconstructed yet, or on a cube reconstructed elsewhere.

"""

import json
import logging
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