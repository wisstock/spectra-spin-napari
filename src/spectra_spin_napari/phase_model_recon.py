""" Phase-model reconstruction of spinning-disk spectral images.

`PhaseModelRecon` turns a batch of phase images into a lambda stack. The
heavy arithmetic - bilinear sampling of a frame at many floating-point
positions and the scatter of the bands into the output - runs on the numba
kernels of this module, multithreaded with `numba.prange`.

"""
import logging
import os
import time

import numba
import numpy as np
from numba import njit, prange
from numpy import ma
from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage import io, measure, morphology

logger = logging.getLogger(__name__)

ACCUMULATION_METHODS = ('max', 'mean', 'overwrite')


# numba kernels: compiled on first use, the machine code is cached on disk

@njit(inline='always')
def _bilinear(image, r, c):
    """ Order-1 sample of `image` at ``(r, c)``, edges extended, as float32. """
    h, w = image.shape
    r = min(max(r, 0.0), h - 1.0)
    c = min(max(c, 0.0), w - 1.0)
    i0, j0 = int(r), int(c)
    i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
    fr, fc = r - i0, c - j0
    return np.float32((1 - fr) * ((1 - fc) * image[i0, j0] + fc * image[i0, j1])
                      + fr * ((1 - fc) * image[i1, j0] + fc * image[i1, j1]))


@njit(inline='always')
def _transformed(r, c, dr, dc, cos_s, sin_s, center_r, center_c):
    """ `PhaseModelRecon._forward_matrix` applied to one point. """
    y, x = r - center_r, c - center_c
    return (cos_s * y - sin_s * x + center_r + dr,
            sin_s * y + cos_s * x + center_c + dc)


@njit(parallel=True, cache=True)
def _template_grid(image, pts_r, pts_c, params, center_r, center_c):
    """ Template cost of every ``[dr, dc, theta, scale]`` row of `params`, one per thread. """
    h, w = image.shape
    out = np.empty(params.shape[0])
    for k in prange(params.shape[0]):
        cos_s = params[k, 3] * np.cos(params[k, 2])
        sin_s = params[k, 3] * np.sin(params[k, 2])
        total, n = 0.0, 0
        for p in range(pts_r.shape[0]):
            r, c = _transformed(pts_r[p], pts_c[p], params[k, 0], params[k, 1],
                                cos_s, sin_s, center_r, center_c)
            if 0 <= r < h - 1 and 0 <= c < w - 1:
                total += _bilinear(image, r, c)
                n += 1
        out[k] = 1e9 if n < 0.2 * pts_r.shape[0] else total / n
    return out


@njit(parallel=True, cache=True)
def _template_cost(image, pts_r, pts_c, dr, dc, theta, scale, center_r, center_c):
    """ Template cost of one candidate, the pattern points split over threads. """
    h, w = image.shape
    cos_s, sin_s = scale * np.cos(theta), scale * np.sin(theta)
    total, n = 0.0, 0
    for p in prange(pts_r.shape[0]):
        r, c = _transformed(pts_r[p], pts_c[p], dr, dc, cos_s, sin_s, center_r, center_c)
        if 0 <= r < h - 1 and 0 <= c < w - 1:
            total += _bilinear(image, r, c)
            n += 1
    return 1e9 if n < 0.2 * pts_r.shape[0] else total / n


@njit(parallel=True, cache=True)
def _fold(image, rows, cols, valid, phase_bins):
    """ Mean intensity per phase bin; bin `b` gathers grid rows ``b, b + phase_bins, ...``. """
    n_cycles, width = rows.shape[0] // phase_bins, rows.shape[1]
    profile, counts = np.zeros(phase_bins), np.zeros(phase_bins)
    for b in prange(phase_bins):
        total, n = 0.0, 0
        for cycle in range(n_cycles):
            i = cycle * phase_bins + b
            for j in range(width):
                if valid[i, j]:
                    total += _bilinear(image, rows[i, j], cols[i, j])
                    n += 1
        profile[b], counts[b] = total / max(n, 1), n
    return profile, counts


@njit(parallel=True, cache=True)
def _sample(image, rows, cols):
    """ `scipy.ndimage.map_coordinates(image, (rows, cols), order=1, mode='nearest')`. """
    out = np.empty(rows.shape, dtype=np.float32)
    for i in prange(rows.shape[0]):
        for j in range(rows.shape[1]):
            out[i, j] = _bilinear(image, rows[i, j], cols[i, j])
    return out


@njit(parallel=True, cache=True)
def _accumulate(out, hit, bands, valid, rows, method, channel):
    """ Write the bands of one frame into `out`, one image column per thread.

    Every column is written by its own bands only, so the columns share
    nothing and each pixel sees its writes in the order of the serial loop:
    band, then row of the line. `method` is the index into
    `ACCUMULATION_METHODS`; `channel` selects one spectral channel into a
    one-plane `out`, -1 writes them all.

    """
    for j in prange(bands.shape[1]):
        for b in range(bands.shape[0]):
            if not valid[b, j]:
                continue
            for o in range(rows.shape[0]):
                r = rows[o, b, j]
                for k in range(out.shape[0]):
                    value = np.float32(bands[b, j, k if channel < 0 else channel])
                    if method == 0:
                        if value > out[k, r, j]:
                            out[k, r, j] = value
                    elif method == 1:
                        out[k, r, j] += value
                    else:
                        out[k, r, j] = value
                hit[r, j] += 1


class PhaseModelRecon():
    """ Batch reconstruction with a shared line-family geometry and a
    monotonically constrained phase trajectory.

    The imaging geometry (scale, rotation, column shift) belongs to the
    optical setup and is estimated once for the batch; the only quantity that
    changes between phase images is the phase of the disk pattern, which the
    disk forces to advance monotonically. The pattern lines are an analytic
    family ``row = L(x, u)``: ``x`` is the image column, ``u`` a continuous
    line index whose integer steps are consecutive lines, so a phase shift is
    a fractional offset of ``u`` and every line keeps one global index.

    Pipeline of `run`:

    1. `_extract_arc_profiles` - rows of every full-width arc of the simulated
       pattern.
    2. `_estimate_initial_geometry` - coarse grid and Nelder-Mead template fit
       on a few reference frames, a starting point only.
    3. Refinement of the shared geometry by the fold contrast
       (`_geometry_contrast`), which has no periodic ambiguity.
    4. `_fold_profile` - every frame rectified into ``(u, x)`` and folded
       modulo one period into its phase cost profile.
    5. `_fit_phase_trajectory` and `_viterbi_phase` - one global trajectory
       under a hard one-directional constraint.
    6. `_sample_bands` - band intensities on a normalised spectral coordinate.

    `lambda_stack_recon` then writes every band into the rows its pattern line
    swept during the exposure, see `_accumulate_bands`.

    Parameters
    ----------
    frames : numpy.ndarray or list of str
        Phase images in acquisition order, the order the monotonicity
        constraint applies to: an array ``(n_image, height, width)`` or the
        paths of image files (any format `skimage.io.imread` reads).
    pattern_sim : numpy.ndarray
        Simulated disk pattern, a binary skeleton of one-pixel-wide lines.
    input_crop : list of int, optional
        ``[row_min, row_max, col_min, col_max]`` slice applied to every frame.
    input_pattern_dilation : int, optional
        Radius of the disk dilating `pattern_sim` for the initial template fit
        only, by default 1.
    init_sigma : float, optional
        Gaussian sigma before the initial template fit, by default 6.0; a
        large sigma widens the basin of the coarse search.
    fold_sigma : float, optional
        Gaussian sigma before folding, by default 4.0.
    arc_edge_tolerance : int, optional
        An arc of `pattern_sim` enters the family model only if it reaches
        within this many pixels of both edges, by default 6.
    family_degree : int, optional
        Polynomial degree of the line family in ``u``, by default 2.
    phase_bins : int, optional
        Bins of the folded profile per line period, by default 128.
    n_lambda : int, optional
        Spectral channels per band; None (default) takes the rounded median
        line period in pixels, one channel per row.
    geometry_frames : int, optional
        Evenly spaced frames for the shared geometry, by default 4.
    refine_geometry : bool, optional
        Polish the shared geometry by the fold contrast, by default True.
    grid_search_params : dict, optional
        Coarse grid of the template fit: 'scale' ``[min, max]``, 'shift' as a
        fraction of the frame, 'rotation' as the largest angle in degrees.
    max_phase_step : float, optional
        Largest phase change between consecutive frames in periods, by default
        0.25; 0.5 or more makes the trajectory ambiguous.
    min_phase_step : float, optional
        Smallest phase change along the drift in periods, by default 0.0 (a
        frame may repeat a phase but never go back).
    phase_slack : float, optional
        Largest per-frame deviation from the constant-step trajectory in
        periods, by default 0.06.
    step_penalty : float, optional
        Weight of the quadratic penalty on the relative deviation of a step
        from the constant step, by default 0.02.
    drift_direction : int, optional
        +1 or -1 forces the sign of the drift; None (default) takes it from
        the trajectory fit.
    line_passes : int or None, optional
        Passes of the pattern during one exposure, 1 to 6, by default 1. The
        line is ``line_passes * pass_width`` periods wide and every band is
        written into all the rows it covers; None writes it into one row.
    pass_width : float, optional
        Distance one pass moves a line, as a fraction of the period, by
        default 0.0533; ignored when `line_passes` is None.
    preprocess : callable, optional
        ``preprocess(frame) -> frame`` applied to every frame after the crop.
    progress : callable, optional
        ``progress(percent)``, told how far `run` has got: 0..95, and -1 while
        the geometry refinement, whose length is unknown, runs.
    threads : int, optional
        Threads of the numba kernels; None (default) uses all numba offers.
    frame_cache_gb : float, optional
        Frames are kept in memory after their first read when the batch fits
        in this many gigabytes, by default 4.0; otherwise every pass reads
        them again.

    Raises
    ------
    ValueError
        If `line_passes` is outside 1..6 or its line is a period or wider.

    """

    # percent reached by `progress` at the end of each stage of `run`
    PROGRESS = {'geometry': 30, 'fold': 60, 'sampling': 95}

    def __init__(self, frames, pattern_sim:np.ndarray,
                 input_crop:list=None,
                 input_pattern_dilation:int=1,
                 init_sigma:float=6.0,
                 fold_sigma:float=4.0,
                 arc_edge_tolerance:int=6,
                 family_degree:int=2,
                 phase_bins:int=128,
                 n_lambda:int=None,
                 geometry_frames:int=4,
                 refine_geometry:bool=True,
                 grid_search_params:dict=None,
                 max_phase_step:float=0.25,
                 min_phase_step:float=0.0,
                 phase_slack:float=0.06,
                 step_penalty:float=0.02,
                 drift_direction:int=None,
                 line_passes:int=1,
                 pass_width:float=0.0533,
                 preprocess=None,
                 progress=None,
                 threads:int=None,
                 frame_cache_gb:float=4.0):
        if line_passes is not None and not 1 <= line_passes <= 6:
            raise ValueError(f'line_passes must be None or 1..6, got {line_passes}')
        if line_passes is not None and not 0 < line_passes * pass_width < 1:
            raise ValueError(f'{line_passes} passes of {pass_width} period are not '
                             f'narrower than one period')

        self.frames = frames
        self.input_crop = input_crop
        self.pattern_sim = pattern_sim
        self.input_pattern_dilation = input_pattern_dilation
        self.init_sigma = init_sigma
        self.fold_sigma = fold_sigma
        self.arc_edge_tolerance = arc_edge_tolerance
        self.family_degree = family_degree
        self.phase_bins = phase_bins
        self.n_lambda = n_lambda
        self.geometry_frames = geometry_frames
        self.refine_geometry = refine_geometry
        self.grid_search_params = (grid_search_params if grid_search_params is not None
                                   else {'scale': [1.0, 2.0], 'shift': 0.2, 'rotation': 30})
        self.max_phase_step = max_phase_step
        self.min_phase_step = min_phase_step
        self.phase_slack = phase_slack
        self.step_penalty = step_penalty
        self.drift_direction = drift_direction
        self.line_passes = line_passes
        self.pass_width = pass_width
        self.preprocess = preprocess
        self.progress = progress if progress is not None else (lambda percent: None)
        self.n_threads = min(int(threads or numba.config.NUMBA_NUM_THREADS),
                             numba.config.NUMBA_NUM_THREADS)
        self.frame_cache_gb = frame_cache_gb

        self.n_image = len(frames)
        self._cache = {}
        self._cache_on = None

    @staticmethod
    def _forward_matrix(params:list, center:tuple) -> np.ndarray:
        """ Affine matrix mapping simulated pattern coordinates onto the frame.

        The pattern is centred on `center`, scaled, rotated and moved back to
        `center` plus the shift.

        Parameters
        ----------
        params : list of float
            ``[dr, dc, theta, scale]``, `theta` in radians.
        center : tuple of float
            Rotation and scaling centre ``(row, col)``.

        Returns
        -------
        numpy.ndarray
            Homogeneous 3x3 matrix.

        """
        dr, dc, theta, scale = params
        center_r, center_c = center

        T_to_origin = np.array([[1, 0, -center_r],
                                [0, 1, -center_c],
                                [0, 0, 1]])
        S = np.array([[scale, 0, 0],
                      [0, scale, 0],
                      [0, 0, 1]])
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [0, 0, 1]])
        T_to_exp = np.array([[1, 0, center_r + dr],
                             [0, 1, center_c + dc],
                             [0, 0, 1]])

        return T_to_exp @ R @ S @ T_to_origin

    @staticmethod
    def _extract_arc_profiles(pattern:np.ndarray, edge_tolerance:int=6) -> np.ndarray:
        """ Per-column row of every full-width arc of the simulated pattern.

        Parameters
        ----------
        pattern : numpy.ndarray
            Binary skeleton of the simulated pattern.
        edge_tolerance : int, optional
            An arc is kept if it starts at most this many pixels from the left
            edge and ends as close to the right one, by default 6.

        Returns
        -------
        numpy.ndarray
            ``(n_arcs, pattern_width)`` rows, sorted top to bottom.

        Raises
        ------
        ValueError
            If fewer than 3 arcs span the pattern.

        """
        labeled_arcs, n_total = measure.label(pattern, connectivity=2, return_num=True)
        pattern_width = pattern.shape[1]

        arc_list = []
        for label_id in range(1, n_total + 1):
            rows, cols = np.where(labeled_arcs == label_id)

            # keep only arcs that reach both image edges
            if cols.min() > edge_tolerance or cols.max() < pattern_width - 1 - edge_tolerance:
                continue

            # mean row per column, gaps filled by interpolation
            row_sum = np.bincount(cols, weights=rows, minlength=pattern_width)
            row_count = np.bincount(cols, minlength=pattern_width)
            has_value = row_count > 0
            arc_list.append(np.interp(np.arange(pattern_width),
                                      np.flatnonzero(has_value),
                                      row_sum[has_value] / row_count[has_value]))

        arc_profiles = np.array(arc_list)
        if arc_profiles.shape[0] < 3:
            raise ValueError('Less than 3 full-width arcs found in the simulated pattern, '
                             'check the pattern mask and arc_edge_tolerance!')

        # sort top-to-bottom so that the arc index increases downwards
        arc_profiles = arc_profiles[np.argsort(arc_profiles[:, pattern_width // 2])]

        logger.info(f'Simulated pattern {pattern.shape}px, '
                    f'{arc_profiles.shape[0]} full-width arcs of {n_total} components')
        return arc_profiles

    @staticmethod
    def _estimate_initial_geometry(image:np.ndarray, pattern:np.ndarray,
                                   smooth_sigma:float=6.0,
                                   grid_search_params:dict=None) -> tuple:
        """ Template fit of the shared geometry, a starting point only.

        The cost is the mean smoothed intensity under the transformed pattern
        lines: a 7 x 5 x 5 x 5 grid over rotation, scale, row and column
        shift, the first of equal minima kept, then Nelder-Mead. The row shift
        it returns is discarded by `run`: the phase absorbs it.

        Parameters
        ----------
        image : numpy.ndarray
            One phase image.
        pattern : numpy.ndarray
            Dilated binary pattern used as the template.
        smooth_sigma : float, optional
            Gaussian sigma of the pre-filter, by default 6.0.
        grid_search_params : dict, optional
            Coarse grid bounds, see the class.

        Returns
        -------
        tuple
            ``(params, cost)``, `params` as ``[dr, dc, theta, scale]``.

        """
        if grid_search_params is None:
            grid_search_params = {'scale': [1.0, 2.0], 'shift': 0.2, 'rotation': 30}
        pts_r, pts_c = (np.asarray(a, dtype=np.float64) for a in np.where(pattern))
        center = (image.shape[0] / 2.0, image.shape[1] / 2.0)
        smooth = np.ascontiguousarray(ndi.gaussian_filter(image, sigma=smooth_sigma))

        params = grid_search_params
        scales = np.linspace(params['scale'][0], params['scale'][-1], 5)
        shifts_r = np.linspace(-image.shape[0] * params['shift'],
                               image.shape[0] * params['shift'], 5)
        shifts_c = np.linspace(-image.shape[1] * params['shift'],
                               image.shape[1] * params['shift'], 5)
        angles = np.linspace(np.radians(-params['rotation']),
                             np.radians(params['rotation']), 7)
        # rows in the nesting order angle, scale, dr, dc
        angle, scale, dr, dc = np.meshgrid(angles, scales, shifts_r, shifts_c, indexing='ij')
        grid = np.column_stack([dr.ravel(), dc.ravel(), angle.ravel(), scale.ravel()])
        costs = _template_grid(smooth, pts_r, pts_c, grid, *center)
        best = int(np.argmin(costs))

        res = minimize(lambda p: _template_cost(smooth, pts_r, pts_c, *p, *center),
                       grid[best], method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        return list(res.x), float(res.fun)

    @staticmethod
    def _line_family(arc_profiles:np.ndarray, geometry:list, img_shape:tuple,
                     degree:int=2) -> tuple:
        """ Fit the line family ``row = sum_d coef[d](x) * u**d`` on the frame.

        Every arc is transformed into the frame, resampled onto the integer
        columns, and the rows of all arcs are fitted per column as a
        polynomial in the arc index.

        Parameters
        ----------
        arc_profiles : numpy.ndarray
            ``(n_arcs, pattern_width)`` arc rows of the simulated pattern.
        geometry : list of float
            ``[dc, theta, scale]``; the row shift is absorbed by the phase.
        img_shape : tuple of int
            Shape of the frame.
        degree : int, optional
            Polynomial degree in the line index, by default 2.

        Returns
        -------
        tuple
            ``(coef, residual, n_used)``: ``(degree + 1, width)`` coefficients,
            the residuals of the arcs used and their number; ``(None, None,
            n_used)`` if too few arcs span the frame or lines overlap.

        """
        dc, theta, scale = geometry
        n_arcs, pattern_width = arc_profiles.shape
        img_height, img_width = img_shape

        center = (img_height / 2.0, img_width / 2.0)
        matrix = PhaseModelRecon._forward_matrix([0.0, dc, theta, scale], center)

        pattern_cols = np.arange(pattern_width, dtype=float)
        out_cols = np.arange(img_width, dtype=float)

        # transform each arc polyline and resample onto the image columns
        arc_rows_exp = np.empty((n_arcs, img_width))
        for arc_idx in range(n_arcs):
            transformed = matrix @ np.vstack((arc_profiles[arc_idx],
                                              pattern_cols,
                                              np.ones(pattern_width)))
            t_r, t_c = transformed[0], transformed[1]
            order = np.argsort(t_c)  # np.interp needs increasing x
            arc_rows_exp[arc_idx] = np.interp(out_cols, t_c[order], t_r[order],
                                              left=np.nan, right=np.nan)

        # keep only the arcs that cover every column of the experimental frame
        covers_all = np.isfinite(arc_rows_exp).all(axis=1)
        if covers_all.sum() < degree + 2:
            return None, None, int(covers_all.sum())

        arc_index = np.flatnonzero(covers_all).astype(float)
        vander = np.vander(arc_index, degree + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(vander, arc_rows_exp[covers_all], rcond=None)

        # a non-positive local period means the arcs were mapped on top of each
        # other, which makes the line index meaningless
        if np.any(coef[1] <= 0):
            return None, None, int(covers_all.sum())

        return coef, arc_rows_exp[covers_all] - vander @ coef, int(covers_all.sum())

    @staticmethod
    def _line_rows(coef:np.ndarray, line_index) -> np.ndarray:
        """ Rows of the line family at any line indices.

        Parameters
        ----------
        coef : numpy.ndarray
            ``(degree + 1, width)`` line family coefficients.
        line_index : array_like
            Real line indices `u`.

        Returns
        -------
        numpy.ndarray
            ``(len(line_index), width)`` rows.

        """
        u = np.atleast_1d(np.asarray(line_index, dtype=float))[:, None]
        rows = np.zeros((u.shape[0], coef.shape[1]))
        for d in range(coef.shape[0] - 1, -1, -1):  # Horner's scheme
            rows = rows * u + coef[d]
        return rows

    @staticmethod
    def _sample_grid(coef:np.ndarray, img_shape:tuple, phase_bins:int) -> tuple:
        """ Rectification grid for folding: whole periods from an integer `u`.

        Starting at an integer `u` keeps bin `b` of a folded profile at phase
        ``b / phase_bins`` exactly.

        Parameters
        ----------
        coef : numpy.ndarray
            Line family coefficients.
        img_shape : tuple of int
            Shape of the frame.
        phase_bins : int
            Phase bins per period.

        Returns
        -------
        tuple
            ``(rows, cols, valid, u_start, n_cycles)``, rows and columns of
            shape ``(n_cycles * phase_bins, width)`` and the mask of samples
            inside the frame; all None if no line reaches the frame.

        """
        img_height, img_width = img_shape

        # bracket the line indices that can reach the frame, using the linear
        # part of the model only: extrapolating the full polynomial far outside
        # the fitted range is not reliable
        u_low = float(np.min((0 - coef[0]) / coef[1])) - 2.0
        u_high = float(np.max((img_height - 1 - coef[0]) / coef[1])) + 2.0
        probe = np.linspace(u_low, u_high, max(int(16 * (u_high - u_low)), 64))
        probe_rows = PhaseModelRecon._line_rows(coef, probe)
        inside = ((probe_rows >= 0) & (probe_rows <= img_height - 1)).any(axis=1)
        if not inside.any():
            return None, None, None, None, None

        u_start = np.floor(probe[inside].min())
        n_cycles = int(np.ceil(probe[inside].max()) - u_start)

        u_grid = u_start + np.arange(n_cycles * phase_bins) / phase_bins
        rows = PhaseModelRecon._line_rows(coef, u_grid)
        cols = np.broadcast_to(np.arange(img_width, dtype=float), rows.shape)
        valid = (rows >= 0) & (rows <= img_height - 1)

        return rows, cols, valid, u_start, n_cycles

    @staticmethod
    def _fold_profile(image:np.ndarray, rows:np.ndarray, cols:np.ndarray,
                      valid:np.ndarray, phase_bins:int) -> tuple:
        """ Rectify a frame into line-index coordinates and fold it modulo one period.

        Every bin is normalised by its own sample count, so the profile does
        not depend on how much pattern falls inside the frame.

        Parameters
        ----------
        image : numpy.ndarray
            Smoothed frame.
        rows, cols, valid : numpy.ndarray
            Grid and mask from `_sample_grid`.
        phase_bins : int
            Phase bins per period.

        Returns
        -------
        tuple
            ``(profile, counts)``: mean intensity and sample count per bin.

        """
        return _fold(image, rows, cols, valid, phase_bins)

    @staticmethod
    def _normalise_profiles(profiles:np.ndarray) -> np.ndarray:
        """ Scale every phase cost profile to [0, 1], 0 at the pattern line.

        Parameters
        ----------
        profiles : numpy.ndarray
            ``(n_frames, phase_bins)`` folded profiles.

        Returns
        -------
        numpy.ndarray
            Normalised profiles.

        """
        low = profiles.min(axis=1, keepdims=True)
        span = profiles.max(axis=1, keepdims=True) - low
        return (profiles - low) / np.maximum(span, 1e-12)

    @staticmethod
    def _profile_at(profiles:np.ndarray, frame_idx:int, phase) -> np.ndarray:
        """ Circular linear interpolation of phase profiles.

        Parameters
        ----------
        profiles : numpy.ndarray
            ``(n_frames, phase_bins)`` normalised profiles.
        frame_idx : int or array_like
            Frame index, broadcastable against `phase`.
        phase : array_like
            Phases in periods, wrapped internally.

        Returns
        -------
        numpy.ndarray
            Interpolated profile values.

        """
        phase_bins = profiles.shape[1]
        x = (np.asarray(phase, dtype=float) % 1.0) * phase_bins
        low = np.floor(x).astype(int) % phase_bins
        high = (low + 1) % phase_bins
        frac = x - np.floor(x)
        return profiles[frame_idx, low] * (1 - frac) + profiles[frame_idx, high] * frac

    @staticmethod
    def _fit_phase_trajectory(profiles:np.ndarray, max_phase_step:float,
                              drift_direction:int=None) -> tuple:
        """ Fit a constant-step trajectory ``Phi_0 + i * delta`` to all frames.

        Minimises the summed profile cost over ``|delta| <= max_phase_step``:
        brute force on a grid, then Nelder-Mead. One straight line through the
        whole batch removes the periodic ambiguity no single frame can.

        Parameters
        ----------
        profiles : numpy.ndarray
            ``(n_frames, phase_bins)`` normalised profiles.
        max_phase_step : float
            Largest |delta| in periods.
        drift_direction : int, optional
            +1 or -1 restricts the sign of delta.

        Returns
        -------
        tuple
            ``(phase_origin, phase_delta, grid_cost, origin_grid, step_grid)``,
            the cost surface kept for `plot_trajectory_cost`.

        """
        n_frames, phase_bins = profiles.shape
        frames = np.arange(n_frames)

        # brute force on a half-bin origin grid and a fine step grid
        origin_grid = np.arange(2 * phase_bins) / (2.0 * phase_bins)
        step_grid = np.arange(-max_phase_step, max_phase_step + 1e-12, 0.0005)
        if drift_direction is not None:
            step_grid = step_grid[np.sign(step_grid) == np.sign(drift_direction)]

        frame_axis = np.arange(n_frames)[:, None, None]
        phase_cube = origin_grid[None, :, None] + frame_axis * step_grid[None, None, :]
        grid_cost = PhaseModelRecon._profile_at(profiles, frame_axis, phase_cube).sum(axis=0)

        best = np.unravel_index(np.argmin(grid_cost), grid_cost.shape)
        start = [origin_grid[best[0]], step_grid[best[1]]]

        # sub-grid refinement of the two global parameters
        def _traj_cost(params):
            if abs(params[1]) > max_phase_step:
                return 1e9
            if drift_direction is not None and np.sign(params[1]) != np.sign(drift_direction):
                return 1e9
            phase = params[0] + np.arange(n_frames) * params[1]
            return float(PhaseModelRecon._profile_at(profiles, frames, phase).sum())

        res = minimize(_traj_cost, start, method='Nelder-Mead',
                       options={'xatol': 1e-7, 'fatol': 1e-9, 'maxiter': 4000})

        return float(res.x[0]), float(res.x[1]), grid_cost, origin_grid, step_grid

    @staticmethod
    def _phase_filling(phase:np.ndarray, median_period:float, grid:int=4096,
                       line_width:float=1.0) -> tuple:
        """ Fraction of one line period the batch sampled.

        Every frame covers ``line_width / median_period`` of the wrapped
        period around its phase; the coefficient is the measure of the union.

        Parameters
        ----------
        phase : numpy.ndarray
            Unwrapped phase per frame, in periods.
        median_period : float
            Median line period in pixels.
        grid : int, optional
            Resolution of the wrapped phase axis, by default 4096.
        line_width : float, optional
            Line width in rows, by default 1.0, a single row.

        Returns
        -------
        tuple
            ``(coefficient, occupancy)``, the covered fraction and the boolean
            occupancy of the phase axis.

        """
        occupancy = np.zeros(grid, dtype=bool)
        half = max(int(round(0.5 * grid * line_width / median_period)), 1)
        offset = np.arange(-half, half + 1)
        for centre in np.rint((np.asarray(phase) % 1.0) * grid).astype(int):
            occupancy[(centre + offset) % grid] = True
        return float(occupancy.mean()), occupancy

    @staticmethod
    def _viterbi_phase(profiles:np.ndarray, phase_linear:np.ndarray,
                       phase_delta:float, drift_direction:int,
                       phase_slack:float, step_penalty:float,
                       max_phase_step:float, min_phase_step:float) -> np.ndarray:
        """ Refine the trajectory by dynamic programming under a one-directional constraint.

        Each frame may deviate from the constant-step trajectory by up to
        `phase_slack`; the emission cost is its own profile, the transition
        forbids steps against the drift or beyond the bounds and penalises the
        relative deviation from the constant step. Viterbi returns the global
        optimum.

        Parameters
        ----------
        profiles : numpy.ndarray
            ``(n_frames, phase_bins)`` normalised profiles.
        phase_linear : numpy.ndarray
            Constant-step trajectory.
        phase_delta : float
            Fitted constant step in periods.
        drift_direction : int
            Sign the steps must keep.
        phase_slack : float
            Half-width of the deviation grid in periods.
        step_penalty : float
            Weight of the quadratic step penalty.
        max_phase_step, min_phase_step : float
            Bounds on the signed step.

        Returns
        -------
        numpy.ndarray
            Refined unwrapped phase per frame; the constant-step trajectory if
            no admissible path exists.

        """
        n_frames, phase_bins = profiles.shape

        # deviation grid, four times finer than the phase profile bins
        slack_step = 1.0 / (4.0 * phase_bins)
        deviation = np.arange(-phase_slack, phase_slack + 1e-12, slack_step)
        n_state = len(deviation)

        emission = PhaseModelRecon._profile_at(profiles, np.arange(n_frames)[:, None],
                                               phase_linear[:, None] + deviation)

        # transition: the realised step when going from state j to state k
        step_matrix = phase_delta + (deviation[None, :] - deviation[:, None])
        signed_step = step_matrix * drift_direction
        # scale the penalty by the fitted step, falling back to the slack width when
        # the fitted step is degenerate
        penalty_scale = abs(phase_delta) if abs(phase_delta) > 1e-6 else max(phase_slack, 1e-6)
        transition = step_penalty * ((step_matrix - phase_delta) / penalty_scale) ** 2
        transition = np.where((signed_step < min_phase_step) |
                              (np.abs(step_matrix) > max_phase_step),
                              np.inf, transition)

        # forward pass
        accumulated = emission[0].copy()
        backpointer = np.zeros((n_frames, n_state), dtype=int)
        for i in range(1, n_frames):
            total = accumulated[:, None] + transition
            backpointer[i] = np.argmin(total, axis=0)
            accumulated = total[backpointer[i], np.arange(n_state)] + emission[i]

        if not np.isfinite(accumulated).any():
            logger.warning('Viterbi found no admissible trajectory, '
                           'falling back to the constant-step model')
            return phase_linear.copy()

        # backward pass
        path = np.zeros(n_frames, dtype=int)
        path[-1] = int(np.argmin(accumulated))
        for i in range(n_frames - 1, 0, -1):
            path[i - 1] = backpointer[i, path[i]]

        return phase_linear + deviation[path]

    @staticmethod
    def _sample_bands(image:np.ndarray, coef:np.ndarray, line_index:np.ndarray,
                      phase:float, n_lambda:int) -> tuple:
        """ Sample every band on a normalised spectral coordinate.

        Band `b` is the strip between lines ``line_index[b] + phase`` and
        ``line_index[b + 1] + phase``, sampled at the centres of `n_lambda`
        equal bins, so a channel is the same fraction of a band everywhere.

        Parameters
        ----------
        image : numpy.ndarray
            Frame to sample.
        coef : numpy.ndarray
            Line family coefficients.
        line_index : numpy.ndarray
            Global integer line indices.
        phase : float
            Phase of the frame in periods.
        n_lambda : int
            Channels per band.

        Returns
        -------
        tuple
            ``(bands, band_valid)``: ``(n_bands, width, n_lambda)`` uint16 and
            the ``(n_bands, width)`` mask of band columns entirely inside the
            frame.

        """
        img_height, img_width = image.shape
        n_bands = len(line_index) - 1

        # sample bin centres, never exactly on a pattern line
        t_axis = (np.arange(n_lambda) + 0.5) / n_lambda
        u_axis = (line_index[:-1, None] + phase + t_axis[None, :]).ravel()

        rows = PhaseModelRecon._line_rows(coef, u_axis)
        cols = np.broadcast_to(np.arange(img_width, dtype=float), rows.shape)
        inside = (rows >= 0) & (rows <= img_height - 1)
        sampled = np.where(inside, _sample(image, rows, cols), 0.0)

        # (n_bands * n_lambda, width) -> (n_bands, width, n_lambda)
        bands = sampled.reshape(n_bands, n_lambda, img_width).transpose(0, 2, 1)
        band_valid = inside.reshape(n_bands, n_lambda, img_width).all(axis=1)

        return np.rint(np.clip(bands, 0, 65535)).astype(np.uint16), band_valid

    def _frame(self, index:int) -> np.ndarray:
        """ Frame `index`: read, cropped, preprocessed, as float32.

        Frames are kept after their first read when the batch fits in
        `frame_cache_gb`, so `run`, `input_stack` and `plot_pattern` read and
        preprocess each of them once.

        Parameters
        ----------
        index : int
            Position of the frame in the batch.

        Returns
        -------
        numpy.ndarray
            The frame; shared with the cache, not to be modified.

        """
        if index in self._cache:
            return self._cache[index]
        source = self.frames[index]
        frame = np.asarray(io.imread(source) if isinstance(source, (str, os.PathLike))
                           else source, dtype=np.float32)
        if self.input_crop is not None:
            frame = frame[self.input_crop[0]:self.input_crop[1],
                          self.input_crop[2]:self.input_crop[3]]
        if self.preprocess is not None:
            frame = self.preprocess(frame)
        if self._cache_on is None:
            self._cache_on = self.n_image * frame.nbytes <= self.frame_cache_gb * 2 ** 30
            logger.info(f'Frames {"kept in memory" if self._cache_on else "read on every pass"}: '
                        f'{self.n_image * frame.nbytes / 2 ** 30:.2f}GB for the batch, '
                        f'limit {self.frame_cache_gb:g}GB')
        if self._cache_on:
            self._cache[index] = frame
        return frame

    def input_stack(self) -> np.ndarray:
        """ The batch as the reconstruction saw it: cropped and preprocessed.

        Returns
        -------
        numpy.ndarray
            ``(n_image, height, width)`` float32.

        """
        return np.stack([self._frame(i) for i in range(self.n_image)])

    def _geometry_contrast(self, geometry:list, frames:list,
                           phase_bins:int=64, col_step:int=4) -> float:
        """ Fold contrast of a candidate shared geometry.

        The right geometry folds every frame onto a deep profile, a wrong one
        smears the lines over the phase bins; the phase never enters, so the
        objective has no periodic ambiguity.

        Parameters
        ----------
        geometry : list of float
            ``[dc, theta, scale]``.
        frames : list of numpy.ndarray
            Smoothed reference frames.
        phase_bins : int, optional
            Phase bins, by default 64.
        col_step : int, optional
            Column subsampling, by default 4.

        Returns
        -------
        float
            Mean relative modulation depth, 0.0 for an unusable geometry.

        """
        coef, _, _ = self._line_family(self.arc_profiles, geometry, self.img_shape,
                                       degree=self.family_degree)
        if coef is None:
            return 0.0

        coef = coef[:, ::col_step]
        shape = (self.img_shape[0], coef.shape[1])
        rows, cols, valid, _, _ = self._sample_grid(coef, shape, phase_bins)
        if rows is None:
            return 0.0

        total = 0.0
        for frame in frames:
            profile, counts = self._fold_profile(frame[:, ::col_step], rows, cols,
                                                 valid, phase_bins)
            if counts.min() == 0 or profile.mean() <= 0:
                return 0.0
            total += (profile.max() - profile.min()) / profile.mean()
        return total / len(frames)

    def run(self):
        """ Run the full batch reconstruction.

        Returns
        -------
        None
            Results are stored on the instance: `geometry` (shared transform
            and its quality), `line_coef` (line family), `phase_cost`,
            `phase_counts`, `phase_modulation` (folded profiles), `phase`,
            `phase_linear`, `phase_step`, `phase_direction` (trajectory),
            `line_width` (rows a line covers), `phase_filling` and
            `phase_occupancy` (sampled share of the period), `line_index`
            (global line numbering), `patterns_stack` (line rows per frame),
            `bands_stack` and `bands_valid` (spectra and their mask).

        """
        run_start_time = time.perf_counter()
        # the thread count of numba belongs to the calling thread
        numba.set_num_threads(self.n_threads)
        logger.info(f'Reconstruction on {self.n_threads} numba threads')

        # pattern preprocessing
        self.arc_profiles = self._extract_arc_profiles(self.pattern_sim,
                                                       edge_tolerance=self.arc_edge_tolerance)
        pattern_fat = morphology.dilation(self.pattern_sim,
                                          morphology.disk(self.input_pattern_dilation))

        # stage 1: shared geometry
        stage_start_time = time.perf_counter()
        ref_idx = np.unique(np.linspace(0, self.n_image - 1,
                                        min(self.geometry_frames, self.n_image)).astype(int))
        logger.info(f'==> Shared geometry from reference frames {list(ref_idx)}')

        ref_frames = [self._frame(i) for i in ref_idx]
        self.img_shape = ref_frames[0].shape
        ref_folded = [ndi.gaussian_filter(f, sigma=self.fold_sigma) for f in ref_frames]

        candidates = []
        for done, (frame, idx) in enumerate(zip(ref_frames, ref_idx), start=1):
            params, cost = self._estimate_initial_geometry(frame, pattern_fat,
                                                           smooth_sigma=self.init_sigma,
                                                           grid_search_params=self.grid_search_params)
            contrast = self._geometry_contrast(params[1:], ref_folded)
            logger.info(f'Frame {idx}: template cost {cost:.1f}, '
                        f'scale {params[3]:.5f}, theta {np.degrees(params[2]):+.3f} deg, '
                        f'dc {params[1]:+.1f}, '
                        f'fold contrast (peak-to-trough / mean) {contrast:.4f}')
            candidates.append((contrast, params))
            self.progress(int(self.PROGRESS['geometry'] * done / len(ref_idx)))

        # the template cost is biased by how much pattern falls inside the frame,
        # so pick the candidate by fold contrast, which is not
        best_contrast, best_params = max(candidates, key=lambda c: c[0])
        best_geom = list(best_params[1:])

        if self.refine_geometry:
            self.progress(-1)  # the optimiser runs for as long as it needs
            logger.info('Refining the shared geometry, this takes a while')
            res = minimize(lambda g: -self._geometry_contrast(g, ref_folded),
                           best_geom, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            if -float(res.fun) > best_contrast:
                best_geom, best_contrast = list(res.x), -float(res.fun)

        self.line_coef, family_resid, n_used = self._line_family(self.arc_profiles, best_geom,
                                                                 self.img_shape,
                                                                 degree=self.family_degree)
        if self.line_coef is None:
            raise ValueError('Line family fit failed: the transformed pattern does not span '
                             'the image width, check the geometry search ranges!')

        self.geometry = {'dc': float(best_geom[0]),
                         'theta': float(best_geom[1]),
                         'scale': float(best_geom[2]),
                         'contrast': float(best_contrast),
                         'family_arcs': int(n_used),
                         'family_resid_rms': float(family_resid.std()),
                         'family_resid_max': float(np.abs(family_resid).max())}
        self.family_resid = family_resid

        logger.info(f'Shared geometry: scale {self.geometry["scale"]:.5f}, '
                    f'theta {np.degrees(self.geometry["theta"]):+.3f} deg, '
                    f'dc {self.geometry["dc"]:+.1f}, '
                    f'fold contrast (peak-to-trough / mean) {best_contrast:.4f}')
        logger.info(f'Line family: degree {self.family_degree}, {n_used} arcs, '
                    f'arc fit residual rms {family_resid.std():.2f}px, '
                    f'max {np.abs(family_resid).max():.2f}px, '
                    f'period {self.line_coef[1].min():.2f}..{self.line_coef[1].max():.2f}px '
                    f'({time.perf_counter() - stage_start_time:.2f}s)')

        # stage 2: per-frame phase cost profiles
        stage_start_time = time.perf_counter()
        rows, cols, valid, u_start, n_cycles = self._sample_grid(self.line_coef, self.img_shape,
                                                                 self.phase_bins)
        if rows is None:
            raise ValueError('The line family model does not intersect the frame, '
                             'check the geometry search ranges and the input crop!')
        self.u_start, self.n_cycles = float(u_start), int(n_cycles)

        self.phase_cost = np.zeros((self.n_image, self.phase_bins))
        self.phase_counts = np.zeros((self.n_image, self.phase_bins))
        span = self.PROGRESS['fold'] - self.PROGRESS['geometry']
        for i in range(self.n_image):
            frame = ndi.gaussian_filter(self._frame(i), sigma=self.fold_sigma)
            self.phase_cost[i], self.phase_counts[i] = self._fold_profile(frame, rows, cols,
                                                                          valid, self.phase_bins)
            self.progress(int(self.PROGRESS['geometry'] + span * (i + 1) / self.n_image))
        self.phase_modulation = ((self.phase_cost.max(axis=1) - self.phase_cost.min(axis=1))
                                 / self.phase_cost.mean(axis=1))
        logger.info(f'Folded {self.n_image} frames over {n_cycles} periods, '
                    f'fold modulation (peak-to-trough / mean) '
                    f'{self.phase_modulation.min():.3f}..{self.phase_modulation.max():.3f} '
                    f'({time.perf_counter() - stage_start_time:.2f}s)')

        # stage 3: global phase trajectory
        stage_start_time = time.perf_counter()
        profiles = self._normalise_profiles(self.phase_cost)
        self.phase_profiles = profiles

        origin, delta, grid_cost, origin_grid, step_grid = self._fit_phase_trajectory(
            profiles, self.max_phase_step, self.drift_direction)
        self.phase_origin, self.phase_delta = origin, delta
        self.trajectory_cost = grid_cost
        self.trajectory_axes = (origin_grid, step_grid)
        self.phase_linear = origin + np.arange(self.n_image) * delta

        direction = int(np.sign(delta)) if self.drift_direction is None else int(np.sign(self.drift_direction))
        direction = direction if direction != 0 else 1  # keep the constraint well defined
        self.phase = self._viterbi_phase(profiles, self.phase_linear, delta, direction,
                                         self.phase_slack, self.step_penalty,
                                         self.max_phase_step, self.min_phase_step)
        self.phase_step = np.diff(self.phase)
        self.phase_direction = direction

        self.median_period = float(np.median(self.line_coef[1]))
        if abs(delta) < 1e-4:
            logger.warning('Fitted phase step is degenerate (close to zero): the batch shows no '
                           'systematic drift, or drift_direction was forced against the data')
            frames_per_period = float('inf')
        else:
            frames_per_period = abs(1 / delta)
        logger.info(f'Phase trajectory: delta {delta:+.6f} period/frame '
                    f'({delta * self.median_period:+.3f}px), '
                    f'{frames_per_period:.2f} frames per period, '
                    f'total drift {delta * (self.n_image - 1):+.3f} periods')
        monotone = bool(np.all(self.phase_step * direction >= 0))
        logger.info(f'Constrained steps: min {self.phase_step.min() * self.median_period:+.3f}px, '
                    f'max {self.phase_step.max() * self.median_period:+.3f}px, '
                    f'one-directional {monotone} '
                    f'({time.perf_counter() - stage_start_time:.2f}s)')

        # rows a pattern line swept during the exposure, one row for the original algorithm
        self.line_width = (1.0 if self.line_passes is None
                           else self.line_passes * self.pass_width * self.median_period)
        logger.info(f'Line width {self.line_width:.2f}px'
                    + (' (single row)' if self.line_passes is None else
                       f' ({self.line_passes} passes of {self.pass_width:.4f} period)'))
        self.phase_filling, self.phase_occupancy = self._phase_filling(
            self.phase, self.median_period, line_width=self.line_width)
        logger.info(f'Phase filling coefficient (period sampled by the batch) '
                    f'{self.phase_filling:.3f} from {self.n_image} phase images')

        # stage 4: global line numbering
        # one fixed index range covers every frame, line index always refers to the same physical arc anywhere in the batch
        u_end = self.u_start + self.n_cycles
        k_min = int(np.floor(self.u_start - self.phase.max())) - 1
        k_max = int(np.ceil(u_end - self.phase.min())) + 1
        self.line_index = np.arange(k_min, k_max + 1)
        n_lines = len(self.line_index)

        if self.n_lambda is None:
            self.n_lambda = int(round(self.median_period))

        # stage 5: band sampling
        stage_start_time = time.perf_counter()
        img_width = self.img_shape[1]
        self.patterns_stack = np.zeros((self.n_image, n_lines, img_width))
        self.bands_stack = np.zeros((self.n_image, n_lines - 1, img_width, self.n_lambda),
                                    dtype=np.uint16)
        self.bands_valid = np.zeros((self.n_image, n_lines - 1, img_width), dtype=bool)

        span = self.PROGRESS['sampling'] - self.PROGRESS['fold']
        for i in range(self.n_image):
            self.patterns_stack[i] = self._line_rows(self.line_coef,
                                                     self.line_index + self.phase[i])
            self.bands_stack[i], self.bands_valid[i] = self._sample_bands(
                self._frame(i), self.line_coef, self.line_index, self.phase[i], self.n_lambda)
            self.progress(int(self.PROGRESS['fold'] + span * (i + 1) / self.n_image))

        stack_size = self.bands_stack.nbytes / (1024 * 1024)
        # the global index range must cover every frame, so it is wider than the number of bands that fit inside any single frame
        bands_per_frame = self.bands_valid.any(axis=2).sum(axis=1)
        logger.info(f'Sampled {n_lines - 1} bands x {self.n_lambda} lambda channels per frame, '
                    f'bands stack {self.bands_stack.shape} ({stack_size:.1f}MB)')
        logger.info(f'Usable bands per frame {bands_per_frame.min()}..{bands_per_frame.max()} '
                    f'of {n_lines - 1} global indices, '
                    f'{100 * self.bands_valid.mean():.1f}% of band columns valid '
                    f'({time.perf_counter() - stage_start_time:.2f}s)')
        logger.info(f'Batch reconstruction of {self.n_image} images finished in '
                    f'{time.perf_counter() - run_start_time:.2f}s')

    def _accumulate_bands(self, lambda_idx:int=None,
                          accumulation_method:str='mean') -> np.ndarray:
        """ Write the sampled bands of every frame into an output image.

        A band goes into the rows half a band below its upper line, where its
        pinholes sat: ``round(line_width)`` rows spread evenly across the
        ``line_passes * pass_width`` periods the line swept, centred on the
        fitted phase, or the single centre row with ``line_passes=None``.
        Every row of a line gets the same spectrum. Only band columns flagged
        in `bands_valid` are written; the number of writes per pixel is kept
        in `lambda_hit_count`.

        Parameters
        ----------
        lambda_idx : int, optional
            One spectral channel to assemble; None (default) assembles all.
        accumulation_method : str, optional
            Merge of writes to the same pixel: 'mean' (default) averages them,
            'max' keeps the brightest, 'overwrite' the last.

        Returns
        -------
        numpy.ndarray
            ``img_shape`` for one channel, ``(n_lambda, *img_shape)``
            otherwise, uint16.

        Raises
        ------
        ValueError
            If `accumulation_method` is unknown.

        """
        if accumulation_method not in ACCUMULATION_METHODS:
            raise ValueError(f'Unknown accumulation method "{accumulation_method}", '
                             f'expected one of {ACCUMULATION_METHODS}')

        numba.set_num_threads(self.n_threads)
        n_out = 1 if lambda_idx is not None else self.n_lambda
        # channel-first, the axis order ImageJ and napari expect of a stack;
        # float32 so that 'mean' divides without rounding twice
        out_img = np.zeros((n_out, *self.img_shape), dtype=np.float32)
        hit_count = np.zeros(self.img_shape, dtype=np.int32)

        # line-index offsets of the rows a line covers, evenly across its width;
        # a single offset of zero is the centre row of the original algorithm
        if self.line_passes is None:
            offsets = np.zeros(1)
        else:
            width_u = self.line_passes * self.pass_width
            n_rows = max(1, int(round(self.line_width)))
            offsets = width_u * ((np.arange(n_rows) + 0.5) / n_rows - 0.5)

        method = ACCUMULATION_METHODS.index(accumulation_method)
        channel = -1 if lambda_idx is None else int(lambda_idx)
        for frame_idx in range(self.n_image):
            rows = np.stack([np.rint(self._line_rows(self.line_coef, self.line_index[:-1]
                                                     + self.phase[frame_idx] + 0.5 + offset))
                             for offset in offsets]).astype(np.int64)
            _accumulate(out_img, hit_count, self.bands_stack[frame_idx],
                        self.bands_valid[frame_idx], rows, method, channel)

        if accumulation_method == 'mean':
            out_img /= np.maximum(hit_count, 1)[None, ...]

        self.lambda_hit_count = hit_count
        filled = 100 * (hit_count > 0).mean()
        logger.info(f'Band writes reached {filled:.1f}% of the frame pixels '
                    f'(mean {hit_count[hit_count > 0].mean():.2f} contributions per reached pixel)')

        out_img = out_img.astype(np.uint16)
        return out_img[0] if lambda_idx is not None else out_img

    def lambda_stack_recon(self, accumulation_method:str='mean') -> np.ndarray:
        """ Assemble the lambda stack from all phase images, see `_accumulate_bands`.

        Parameters
        ----------
        accumulation_method : str, optional
            'mean' (default), 'max' or 'overwrite'.

        Returns
        -------
        numpy.ndarray
            ``(n_lambda, *img_shape)`` uint16.

        """
        logger.info(f'Assembling lambda stack {(self.n_lambda, *self.img_shape)}, '
                    f'{np.prod(self.img_shape) * self.n_lambda * 4 / 1048576:.0f}MB accumulator')
        return self._accumulate_bands(None, accumulation_method)

    def lambda_frame_recon(self, lambda_idx:int=10,
                           accumulation_method:str='mean') -> np.ndarray:
        """ One spectral channel, ``lambda_stack_recon()[lambda_idx]`` without the cube.

        Parameters
        ----------
        lambda_idx : int, optional
            Channel, 0..n_lambda - 1, by default 10.
        accumulation_method : str, optional
            'mean' (default), 'max' or 'overwrite'.

        Returns
        -------
        numpy.ndarray
            ``img_shape`` uint16.

        """
        return self._accumulate_bands(lambda_idx, accumulation_method)

    def arc_labels(self) -> np.ndarray:
        """ The modelled arcs of every frame as a labels volume.

        One plane per frame, the frame shape; every arc keeps one label in all
        frames, the lowest global line index being 1, so stepping through the
        frames follows one arc. Arcs are drawn only inside the frame.

        Returns
        -------
        numpy.ndarray
            ``(n_image, *img_shape)`` uint16.

        """
        height, width = self.img_shape
        cols = np.arange(width)
        labels = np.zeros((self.n_image, height, width), dtype=np.uint16)
        for i in range(self.n_image):
            rows = np.rint(self.patterns_stack[i]).astype(int)
            for value, row in enumerate(rows, start=1):
                inside = (row >= 0) & (row < height)
                labels[i, row[inside], cols[inside]] = value
        logger.info(f'Rasterised {len(self.line_index)} arcs over {self.n_image} frames, '
                    f'labels {labels.shape} ({labels.nbytes / 1048576:.0f}MB), '
                    f'label 1 is line index {int(self.line_index[0])}')
        return labels

    def plot_phase_cost_map(self, figure=None):
        """ Phase cost profile of every frame with the fitted phase, and the phase filling.

        One row per frame, dark where the pattern line sits; a healthy batch
        shows one ridge drifting across the period with every marker on it. A
        broken ridge means a wrong geometry, markers off the ridge too tight a
        trajectory (raise `phase_slack`, lower `step_penalty`). The strip
        below has one stripe per frame as wide as the line; its inked share is
        `phase_filling`, gaps are parts of every band no frame sampled.

        Parameters
        ----------
        figure : matplotlib.figure.Figure, optional
            Figure to draw into; None (default) opens a pyplot window.

        Returns
        -------
        matplotlib.figure.Figure
            The figure drawn.

        """
        import matplotlib.pyplot as plt

        fig = figure if figure is not None else plt.figure(figsize=(10, 6.5))
        ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [8, 1]})

        im = ax[0].imshow(self.phase_profiles, cmap='inferno',
                          aspect='auto', origin='upper',
                          extent=[0, 1, self.n_image - 0.5, -0.5])
        ax[0].plot(self.phase % 1.0, np.arange(self.n_image), '.', color='cyan',
                   ms=7, label='fitted phase')
        ax[0].set_ylabel('Phase img num')
        ax[0].set_title(f'Phase filling coefficient {self.phase_filling:.3f}')
        ax[0].legend(loc='upper right')

        # one stripe per phase image, the inked fraction is the coefficient;
        # a wide line is drawn at its width, wrapped around the period ends
        if self.line_passes is None:
            ax[1].vlines(self.phase % 1.0, 0, 1, color='#1f4e79', lw=1.5)
        else:
            half = 0.5 * self.line_width / self.median_period
            for centre in self.phase % 1.0:
                for shift in (-1.0, 0.0, 1.0):
                    ax[1].axvspan(centre + shift - half, centre + shift + half,
                                  color='#1f4e79', lw=0)
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].set_yticks([])
        ax[1].set_xlabel('Phase within one line period')
        ax[1].set_ylabel('phase\nfilling', rotation=0, ha='right',
                         va='center', fontsize=9)

        fig.colorbar(im, ax=ax.tolist(), label='normalised fold cost')
        if figure is None:
            plt.show()
        return fig

    def plot_phase_trajectory(self):
        """ Unwrapped phase per frame against the constant-step model, and the residual in pixels.

        The upper panel should be a straight line; the residual should stay
        well inside `phase_slack`, and one pressed against it means the slack
        clips real motion.

        """
        import matplotlib.pyplot as plt

        residual = (self.phase - self.phase_linear) * self.median_period

        fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})
        ax[0].plot(np.arange(self.n_image), self.phase_linear, '-', color='grey',
                   label=f'constant step {self.phase_delta:+.5f} period/frame')
        ax[0].plot(np.arange(self.n_image), self.phase, 'o', ms=5, color='#1f4e79',
                   label='constrained phase')
        ax[0].set_ylabel('Unwrapped phase, line periods')
        ax[0].legend()
        ax[1].axhline(0, color='grey', lw=0.8)
        ax[1].plot(np.arange(self.n_image), residual, 'o-', ms=4, color='#a8721a')
        ax[1].set_xlabel('Phase img num')
        ax[1].set_ylabel('Residual, px')
        plt.tight_layout()
        plt.show()

    def plot_phase_steps(self):
        """ Phase step per frame in pixels, with the admissible band and the forbidden direction.

        Every step must stay on one side of zero; steps pressed against the
        band edge mean `phase_slack` or `max_phase_step` clips the data, steps
        pinned to the constant step mean `step_penalty` is too high.

        """
        import matplotlib.pyplot as plt

        steps = self.phase_step * self.median_period

        plt.figure(figsize=(10, 4))
        ax = plt.axes()
        ax.axhline(0, color='crimson', lw=1.2, label='forbidden direction')
        band = sorted([self.min_phase_step * self.phase_direction * self.median_period,
                       self.max_phase_step * self.phase_direction * self.median_period])
        ax.axhspan(band[0], band[1], color='#1d6d67', alpha=0.12,
                   label='admissible band')
        ax.axhline(self.phase_delta * self.median_period, color='grey', ls='--',
                   label='fitted constant step')
        ax.plot(np.arange(len(steps)), steps, 'o-', ms=4, color='#1f4e79')
        ax.set_xlabel('Phase img num')
        ax.set_ylabel('Phase step, px')
        ax.legend(fontsize=8)
        plt.show()

    def plot_trajectory_cost(self):
        """ Cost surface of the global trajectory search with the fitted solution.

        One isolated minimum with the marker in it means the batch fixes the
        trajectory; several equal minima mean it is too short to.

        """
        import matplotlib.pyplot as plt

        origin_grid, step_grid = self.trajectory_axes

        plt.figure(figsize=(10, 5))
        ax = plt.axes()
        im = ax.imshow(self.trajectory_cost / self.n_image, cmap='viridis', aspect='auto',
                       origin='lower',
                       extent=[step_grid[0], step_grid[-1], origin_grid[0], origin_grid[-1]])
        ax.plot(self.phase_delta, self.phase_origin % 1.0, 'o', color='crimson', ms=8,
                label='fitted (delta, phase origin)')
        ax.set_xlabel('Phase step delta, line periods per frame')
        ax.set_ylabel('Phase origin, line periods')
        ax.legend()
        plt.colorbar(im, ax=ax, label='mean normalised fold cost')
        plt.show()

    def plot_line_model(self):
        """ Local line period across the frame and the residual of the line family fit.

        The period should vary smoothly; a smooth large-scale residual is
        model error that a higher `family_degree` reduces, a fine striping is
        the pixel quantisation of the simulated pattern.

        """
        import matplotlib.pyplot as plt

        cols_idx = np.arange(self.img_shape[1])

        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(cols_idx, self.line_coef[1], color='#1f4e79')
        ax[0].set_xlabel('Image column')
        ax[0].set_ylabel('Local line period, px')
        ax[0].set_title(f'Line spacing, degree {self.family_degree} family model')

        im = ax[1].imshow(self.family_resid, cmap='coolwarm', aspect='auto',
                          vmin=-np.abs(self.family_resid).max(),
                          vmax=np.abs(self.family_resid).max())
        ax[1].set_xlabel('Image column')
        ax[1].set_ylabel('Arc num')
        ax[1].set_title(f'Fit residual, rms {self.family_resid.std():.2f}px')
        plt.colorbar(im, ax=ax[1], label='px')
        plt.tight_layout()
        plt.show()

    def plot_pattern(self, phase_img_num:int=0, band_mid:bool=True):
        """ Modelled band boundaries, and optionally the mid-band rows, over one frame.

        The boundaries must sit in the dark gaps along the whole width.

        Parameters
        ----------
        phase_img_num : int, optional
            Frame to show, by default 0.
        band_mid : bool, optional
            Also draw the mid-band rows, by default True.

        """
        import matplotlib.pyplot as plt

        frame = self._frame(phase_img_num)
        img_height, img_width = self.img_shape
        cols_idx = np.arange(img_width)

        # label 1 for the band boundaries, label 2 for the mid-band rows
        overlay = np.zeros(self.img_shape, dtype=int)
        levels = [(self.patterns_stack[phase_img_num], 1)]
        if band_mid:
            levels.append((self._line_rows(self.line_coef, self.line_index[:-1]
                                           + self.phase[phase_img_num] + 0.5), 2))
        for line_rows, label in levels:
            for row in np.rint(line_rows).astype(int):
                inside = (row >= 0) & (row < img_height)
                overlay[row[inside], cols_idx[inside]] = label

        plt.figure(figsize=(10, 10))
        ax = plt.axes()
        ax.imshow(frame, cmap='inferno', vmin=0, vmax=np.max(frame) * 0.5)
        ax.imshow(ma.masked_equal(overlay, 0), cmap='Greys')
        ax.set_title(f'Phase img {phase_img_num}, phase {self.phase[phase_img_num]:+.4f} periods')
        plt.show()

    def plot_band_coverage(self):
        """ Share of columns each global band covers completely, per frame.

        A solid block of valid bands with soft edges drifting across the
        frames is expected; an abrupt change of its height between frames is
        not.

        """
        import matplotlib.pyplot as plt

        coverage = 100 * self.bands_valid.mean(axis=2)

        plt.figure(figsize=(10, 5))
        ax = plt.axes()
        im = ax.imshow(coverage.T, cmap='viridis', aspect='auto', vmin=0, vmax=100,
                       extent=[-0.5, self.n_image - 0.5,
                               self.line_index[-2] + 0.5, self.line_index[0] - 0.5])
        ax.set_xlabel('Phase img num')
        ax.set_ylabel('Global line num')
        plt.colorbar(im, ax=ax, label='valid columns, %')
        plt.show()

    def plot_lambda_frame(self, lambda_idx:int=10, max_int_m:float=0.5,
                          accumulation_method:str='mean'):
        """ One spectral channel of the reconstruction, see `lambda_frame_recon`.

        Parameters
        ----------
        lambda_idx : int, optional
            Channel, by default 10.
        max_int_m : float, optional
            Display ceiling as a fraction of the channel maximum, by default 0.5.
        accumulation_method : str, optional
            By default 'mean'.

        """
        import matplotlib.pyplot as plt

        frame = self.lambda_frame_recon(lambda_idx, accumulation_method)

        plt.figure(figsize=(10, 10))
        plt.imshow(frame, cmap='magma', vmax=np.max(frame) * max_int_m)
        plt.title(f'Lambda channel {lambda_idx} of {self.n_lambda}')
        plt.show()
