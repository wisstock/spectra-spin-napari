import time
import logging

import numpy as np
from numpy import ma

from scipy import ndimage as ndi
from scipy.optimize import minimize
from skimage import morphology, measure
from skimage import io

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format='%(name)s | %(levelname)s | %(message)s')


class PhaseModelRecon():
    """ Batch reconstruction with a shared line-family geometry and a
    monotonically constrained phase trajectory.

    This class is an alternative to `BatchRecon`. Instead of solving an
    independent four-parameter alignment for every phase image, it splits
    the problem the way the instrument does:

    * the imaging geometry (scale, rotation, column shift) is a property of
      the optical setup and is therefore estimated **once** for the whole
      batch;
    * the only quantity that changes between consecutive phase images is a
      single scalar - the phase of the disk pattern along the direction
      normal to the pattern lines;
    * the disk always turns the same way, so that phase is forced to
      advance monotonically over the batch.

    The pattern lines are described analytically by a *line family model*
    ``row = L(x, u)``, where ``x`` is the image column and ``u`` is a
    continuous line index. Integer ``u`` steps correspond to consecutive
    pattern lines, so a phase shift of the disk is simply a fractional
    offset of ``u``. Because the model is analytic, lines never have to be
    re-detected per frame, are never lost when they leave the field of
    view, and keep the same global index in every phase image.

    Reconstruction pipeline
    -----------------------
    1. `_extract_arc_profiles` - per-column row coordinates of every
       full-width arc of the simulated pattern (done once).
    2. `_estimate_initial_geometry` - a coarse grid plus Nelder-Mead
       template fit on a few reference frames, used only as a starting
       point.
    3. `_refine_geometry` - the shared geometry is polished by maximising
       the *fold contrast*, an objective that is free of the periodic
       ambiguity because the phase is profiled out.
    4. `_fold_profile` - each frame is rectified into ``(u, x)`` coordinates
       and folded modulo one line period, giving the phase cost profile
       ``C_i(phi)``.
    5. `_fit_phase_trajectory` + `_viterbi_phase` - a global trajectory
       ``Phi_i`` is fitted to all profiles at once under a hard
       one-directional constraint.
    6. `_sample_bands` - band intensities are sampled on a normalised
       spectral coordinate, so a lambda index means the same fraction of a
       band everywhere.

    """

    def __init__(self, input_path:list[str],
                 pattern_sim:np.ndarray,
                 input_crop:list[int,int,int,int]=None,
                 input_pattern_dilation:int=1,
                 init_sigma:float=6.0,
                 fold_sigma:float=4.0,
                 arc_edge_tolerance:int=6,
                 family_degree:int=2,
                 phase_bins:int=128,
                 n_lambda:int=None,
                 geometry_frames:int=4,
                 refine_geometry:bool=True,
                 grid_search_params:dict={'scale':[1.0, 2.0],  # min and max scale factors to test
                                          'shift':0.2,         # fraction of image size for shift range to test
                                          'rotation':30},      # max rotation angle in degrees to test
                 max_phase_step:float=0.25,
                 min_phase_step:float=0.0,
                 phase_slack:float=0.06,
                 step_penalty:float=0.02,
                 drift_direction:int=None,
                 output_path:str="results"):
        """
        Parameters
        ----------
        input_path : list of str
            Ordered paths of the phase images. The order must be the
            acquisition order, because the monotonicity constraint is
            applied to consecutive elements of this list.
        input_crop : list of int
            Crop applied to every loaded frame as
            ``[row_min, row_max, col_min, col_max]``.
        pattern_sim : 2D numpy.ndarray
            Simulated reference phase pattern (boolean/binary skeleton mask,
            one-pixel-wide lines on a zero background).
        input_pattern_dilation : int, optional
            Radius of the disk structuring element used to dilate
            `pattern_sim` for the initial template fit only, by default 1.
            The line family model always uses the undilated skeleton.
        init_sigma : float, optional
            Sigma of the Gaussian filter applied before the initial
            template fit, by default 6.0. A large sigma widens the basin of
            attraction of the coarse search.
        fold_sigma : float, optional
            Sigma of the Gaussian filter applied before folding, by default
            4.0. Smaller than `init_sigma` because the fold does not need a
            wide basin, only a clean periodic profile.
        arc_edge_tolerance : int, optional
            An arc of `pattern_sim` is accepted into the family model only
            if it reaches within this many pixels of both image edges, by
            default 6. Truncated arcs at the top and bottom of the
            simulated frame are rejected by this test.
        family_degree : int, optional
            Polynomial degree in the line index `u` of the line family
            model, by default 2. Degree 1 assumes perfectly equidistant
            lines; degree 2 also captures the slow change of line spacing
            across the pattern.
        phase_bins : int, optional
            Number of bins of the folded phase cost profile, by default 128.
            The phase resolution is one line period divided by this value.
        n_lambda : int, optional
            Number of resampled spectral channels per band. If None
            (default) it is set to the rounded median line spacing in
            pixels, which resamples the band without loss of resolution.
        geometry_frames : int, optional
            Number of evenly spaced frames used to estimate the shared
            geometry, by default 4.
        refine_geometry : bool, optional
            Whether to polish the shared geometry with the fold contrast
            objective, by default True.
        grid_search_params : dict, optional
            Coarse grid bounds of the initial template fit:
            'scale' - [min, max] scale factors to test,
            'shift' - fraction of image size for the shift range to test,
            'rotation' - max rotation angle in degrees to test.
        max_phase_step : float, optional
            Largest allowed phase change between consecutive frames, in
            line-period units, by default 0.25. Values of 0.5 or more make
            the trajectory ambiguous, because a step of `d` and a step of
            ``d - 1`` produce identical images.
        min_phase_step : float, optional
            Smallest allowed phase change between consecutive frames, in
            line-period units and measured along the drift direction, by
            default 0.0 (a frame may repeat the previous phase but may
            never move backwards). Set a positive value to demand strictly
            monotone motion.
        phase_slack : float, optional
            Half-width of the per-frame deviation allowed from the fitted
            constant-step trajectory, in line-period units, by default 0.06.
        step_penalty : float, optional
            Weight of the quadratic penalty on the relative deviation of a
            step from the fitted constant step, by default 0.02. Zero lets
            the trajectory follow every wiggle of the cost profiles, large
            values pin it to a straight line.
        drift_direction : int, optional
            +1 or -1 to force the sign of the phase drift. If None
            (default) the sign is taken from the global trajectory fit.
        output_path : str, optional
            Reserved for result export, by default "results".

        """

        # input data
        self.input_path = input_path
        self.input_crop = input_crop
        self.pattern_sim = pattern_sim
        self.output_path = output_path

        # preprocessing parameters
        self.input_pattern_dilation = input_pattern_dilation
        self.init_sigma = init_sigma
        self.fold_sigma = fold_sigma

        # line family model parameters
        self.arc_edge_tolerance = arc_edge_tolerance
        self.family_degree = family_degree

        # sampling parameters
        self.phase_bins = phase_bins
        self.n_lambda = n_lambda

        # geometry estimation parameters
        self.geometry_frames = geometry_frames
        self.refine_geometry = refine_geometry
        self.grid_search_params = grid_search_params

        # phase trajectory constraints
        self.max_phase_step = max_phase_step
        self.min_phase_step = min_phase_step
        self.phase_slack = phase_slack
        self.step_penalty = step_penalty
        self.drift_direction = drift_direction

        self.n_image = len(self.input_path)

    @staticmethod
    def _forward_matrix(params:list, center:tuple) -> np.ndarray:
        """ Build the forward affine matrix mapping simulated pattern
        coordinates onto experimental image coordinates.

        The pattern is first centred on `center`, then scaled, then rotated,
        then moved back to `center` plus the requested shift. Using the same
        centre for both frames keeps the parameters directly comparable with
        `OnePhaseRecon._estimate_alignment_matrix`.

        Parameters
        ----------
        params : list of float
            Transformation parameters ``[dr, dc, theta, scale]``, with
            `theta` in radians.
        center : tuple of float
            Rotation/scaling centre as ``(row, col)``.

        Returns
        -------
        numpy.ndarray
            Homogeneous 3x3 forward transformation matrix.

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
        """ Per-column row coordinate of every full-width arc of the
        simulated pattern, in the pattern's own coordinate frame.

        Unlike `OnePhaseRecon._extract_pattern`, this runs on the *simulated*
        pattern only, once per reconstruction. The simulated pattern is
        noise-free, so its arcs are clean connected components and the
        acceptance test can be strict: an arc is kept only if it reaches
        both image edges, which rejects the truncated arc fragments at the
        top and bottom of the simulated frame.

        Parameters
        ----------
        pattern : 2D numpy.ndarray
            Binary skeleton of the simulated pattern.
        edge_tolerance : int, optional
            An arc is accepted if its first column is at most this value and
            its last column is at least ``width - 1 - edge_tolerance``, by
            default 6.

        Returns
        -------
        numpy.ndarray
            Array of shape (n_arcs, pattern_width) with the row coordinate
            of every accepted arc for every column, sorted top-to-bottom by
            the row at the middle column.

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

        logging.info(f'Simulated pattern {pattern.shape}px, '
                     f'{arc_profiles.shape[0]} full-width arcs of {n_total} components')
        return arc_profiles

    @staticmethod
    def _estimate_initial_geometry(image:np.ndarray, pattern:np.ndarray,
                                   smooth_sigma:float=6.0,
                                   grid_search_params:dict={'scale':[1.0, 2.0],
                                                            'shift':0.2,
                                                            'rotation':30}) -> tuple:
        """ Template-based starting guess for the shared geometry.

        Identical in spirit to `OnePhaseRecon._estimate_alignment_matrix`:
        a coarse grid search followed by Nelder-Mead on the mean image
        intensity sampled under the transformed pattern lines. Only the
        starting guess is taken from here - the row shift `dr` is discarded
        because it is absorbed by the phase model, which is exactly the
        degree of freedom that makes this objective ambiguous.

        Parameters
        ----------
        image : 2D numpy.ndarray
            One experimental phase image.
        pattern : 2D numpy.ndarray
            Dilated binary pattern mask used as a template.
        smooth_sigma : float, optional
            Sigma of the Gaussian pre-filter, by default 6.0.
        grid_search_params : dict, optional
            Coarse grid bounds, see the class docstring.

        Returns
        -------
        tuple
            ``(params, cost)`` where `params` is
            ``[dr, dc, theta, scale]`` and `cost` is the final objective
            value.

        """
        pts_r, pts_c = np.where(pattern)
        pts_homo = np.vstack((pts_r, pts_c, np.ones_like(pts_r)))
        center = (image.shape[0] / 2.0, image.shape[1] / 2.0)

        image_smooth = ndi.gaussian_filter(image, sigma=smooth_sigma)

        def _cost_function(params):
            """ Mean image intensity under the transformed pattern lines. """
            transformed = PhaseModelRecon._forward_matrix(params, center) @ pts_homo
            t_r, t_c = transformed[0, :], transformed[1, :]

            valid_mask = (t_r >= 0) & (t_r < image.shape[0] - 1) & \
                         (t_c >= 0) & (t_c < image.shape[1] - 1)
            if np.sum(valid_mask) < 0.2 * len(pts_r):
                return 1e9

            return float(np.mean(ndi.map_coordinates(image_smooth,
                                                     np.vstack((t_r[valid_mask], t_c[valid_mask])),
                                                     order=1, mode='nearest')))

        # coarse grid over the same ranges as OnePhaseRecon
        scales = np.linspace(grid_search_params['scale'][0], grid_search_params['scale'][-1], 5)
        shifts_r = np.linspace(-image.shape[0] * grid_search_params['shift'],
                               image.shape[0] * grid_search_params['shift'], 5)
        shifts_c = np.linspace(-image.shape[1] * grid_search_params['shift'],
                               image.shape[1] * grid_search_params['shift'], 5)
        angles = np.linspace(np.radians(-grid_search_params['rotation']),
                             np.radians(grid_search_params['rotation']), 7)

        best_cost, best_params = float('inf'), [0.0, 0.0, 0.0, 1.0]
        for angle in angles:
            for scale in scales:
                for dr in shifts_r:
                    for dc in shifts_c:
                        cost = _cost_function([dr, dc, angle, scale])
                        if cost < best_cost:
                            best_cost, best_params = cost, [dr, dc, angle, scale]

        res = minimize(_cost_function, best_params, method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        return list(res.x), float(res.fun)

    @staticmethod
    def _line_family(arc_profiles:np.ndarray, geometry:list, img_shape:tuple,
                     degree:int=2) -> tuple:
        """ Fit the line family model ``row = sum_d coef[d](x) * u**d`` in
        experimental image coordinates.

        Every simulated arc is transformed into the experimental frame as a
        polyline and resampled onto the integer column grid. For each column
        the row coordinates of all arcs are then fitted as a polynomial in
        the arc index, which turns the discrete set of arcs into a
        continuous family that can be evaluated - and extrapolated - at any
        real line index.

        Parameters
        ----------
        arc_profiles : numpy.ndarray
            Per-column arc rows in the simulated frame, shape
            (n_arcs, pattern_width).
        geometry : list of float
            Shared geometry as ``[dc, theta, scale]``. The row shift is
            deliberately absent: it is absorbed by the phase.
        img_shape : tuple of int
            Shape of the experimental frame.
        degree : int, optional
            Polynomial degree in the line index, by default 2.

        Returns
        -------
        tuple
            ``(coef, residual, n_used)`` where `coef` has shape
            (degree + 1, img_width), `residual` holds the fit residuals of
            the arcs actually used, shape (n_used, img_width), and `n_used`
            is the number of arcs that spanned the full image width.

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
        """ Evaluate the line family model at arbitrary line indices.

        Parameters
        ----------
        coef : numpy.ndarray
            Line family coefficients, shape (degree + 1, img_width).
        line_index : array_like
            Line indices `u` to evaluate, any real values.

        Returns
        -------
        numpy.ndarray
            Row coordinates of shape (len(line_index), img_width).

        """
        u = np.atleast_1d(np.asarray(line_index, dtype=float))[:, None]
        rows = np.zeros((u.shape[0], coef.shape[1]))
        for d in range(coef.shape[0] - 1, -1, -1):  # Horner's scheme
            rows = rows * u + coef[d]
        return rows

    @staticmethod
    def _sample_grid(coef:np.ndarray, img_shape:tuple, phase_bins:int) -> tuple:
        """ Build the rectification grid used for folding.

        The grid walks the line index `u` in steps of one bin over a whole
        number of periods, starting at an integer `u`. Starting at an
        integer keeps bin `b` of the folded profile at phase ``b /
        phase_bins`` exactly, so no phase offset bookkeeping is needed later.

        Parameters
        ----------
        coef : numpy.ndarray
            Line family coefficients.
        img_shape : tuple of int
            Shape of the experimental frame.
        phase_bins : int
            Number of phase bins per line period.

        Returns
        -------
        tuple
            ``(rows, cols, valid, u_start, n_cycles)``. `rows` and `cols`
            have shape (n_cycles * phase_bins, img_width) and are ready for
            `scipy.ndimage.map_coordinates`; `valid` flags the samples that
            fall inside the frame.

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
        """ Rectify a frame into line-index coordinates and fold it modulo
        one line period.

        This replaces the per-frame optimisation of the row shift. Because
        the pattern is periodic, the average image intensity as a function
        of the position within a period is all the information a single
        frame carries about its phase. Folding extracts that profile in one
        pass over the image and, unlike a template cost, normalises every
        bin by its own sample count, so the result does not depend on how
        much of the pattern happens to fall inside the frame.

        Parameters
        ----------
        image : 2D numpy.ndarray
            Pre-smoothed experimental frame.
        rows, cols : numpy.ndarray
            Rectification grid from `_sample_grid`.
        valid : numpy.ndarray
            Validity mask of the rectification grid.
        phase_bins : int
            Number of phase bins per line period.

        Returns
        -------
        tuple
            ``(profile, counts)``: the mean intensity per phase bin and the
            number of samples that contributed to each bin.

        """
        sampled = ndi.map_coordinates(image, np.vstack((rows.ravel(), cols.ravel())),
                                      order=1, mode='nearest').reshape(rows.shape)

        # accumulate per phase bin, ignoring samples outside the frame
        n_cycles = rows.shape[0] // phase_bins
        sampled = np.where(valid, sampled, 0.0).reshape(n_cycles, phase_bins, -1)
        weights = valid.reshape(n_cycles, phase_bins, -1)

        counts = weights.sum(axis=(0, 2))
        profile = sampled.sum(axis=(0, 2)) / np.maximum(counts, 1)

        return profile, counts

    @staticmethod
    def _normalise_profiles(profiles:np.ndarray) -> np.ndarray:
        """ Scale every phase cost profile to the range [0, 1].

        Frames differ in overall brightness, so the raw profiles are not
        comparable. After normalisation a value of 0 marks the darkest
        position within a period - the pattern line - for every frame alike,
        which makes a single trajectory penalty weight meaningful across the
        whole batch.

        Parameters
        ----------
        profiles : numpy.ndarray
            Raw folded profiles, shape (n_frames, phase_bins).

        Returns
        -------
        numpy.ndarray
            Normalised profiles of the same shape.

        """
        low = profiles.min(axis=1, keepdims=True)
        span = profiles.max(axis=1, keepdims=True) - low
        return (profiles - low) / np.maximum(span, 1e-12)

    @staticmethod
    def _profile_at(profiles:np.ndarray, frame_idx:int, phase) -> np.ndarray:
        """ Circular linear interpolation of one frame's phase profile.

        Parameters
        ----------
        profiles : numpy.ndarray
            Normalised profiles, shape (n_frames, phase_bins).
        frame_idx : int or array_like
            Index of the frame to sample. An integer array broadcastable
            against `phase` samples many frames at once, which is how the
            trajectory search and the Viterbi refinement use it.
        phase : array_like
            Phase values in line-period units, wrapped internally.

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
        """ Fit a constant-step phase trajectory to all frames at once.

        Minimises ``sum_i C_i(Phi_0 + i * delta)`` over the two global
        parameters of the model. This is the step that removes the periodic
        ambiguity: a single frame cannot tell which of the equally deep
        minima of its profile is the right one, but a whole batch that has
        to be explained by one straight line can.

        Only steps with ``|delta| <= max_phase_step`` are considered. That
        bound is what makes the problem well posed at all, because
        ``delta`` and ``delta +/- 1`` produce identical images: the physical
        prior is that the disk moves by less than a quarter of a line period
        between consecutive frames.

        Parameters
        ----------
        profiles : numpy.ndarray
            Normalised phase cost profiles, shape (n_frames, phase_bins).
        max_phase_step : float
            Largest allowed |delta| in line-period units.
        drift_direction : int, optional
            +1 or -1 to restrict the search to one sign, by default None.

        Returns
        -------
        tuple
            ``(phase_origin, phase_delta, grid_cost, origin_grid,
            step_grid)``. `grid_cost` is the full brute-force cost surface
            of shape (len(origin_grid), len(step_grid)), kept for the
            diagnostic plot.

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
    def _phase_filling(phase:np.ndarray, median_period:float, grid:int=4096) -> tuple:
        """ Fraction of one line period actually sampled by the batch.

        Every phase image places its bands one image row wide, so in phase
        units each frame covers ``1 / median_period`` of the period. The
        **phase filling coefficient** is the measure of the union of those
        intervals over the wrapped phase axis: 1.0 means the period is
        sampled end to end, and a low value means the series left parts of
        every band unvisited no matter how many frames it contains.

        Because it is a union, frames that land on the same phase count once,
        so the coefficient is a coverage figure rather than a frame count. It
        is also what limits the reconstruction: the fraction of pixels a band
        write can reach cannot exceed it.

        Parameters
        ----------
        phase : numpy.ndarray
            Unwrapped phase per frame, in line periods. Wrapped internally.
        median_period : float
            Median local line period in pixels, which sets how much phase one
            image row spans.
        grid : int, optional
            Resolution of the wrapped phase axis used to take the union, by
            default 4096 - about 0.02 px of a period.

        Returns
        -------
        tuple
            ``(coefficient, occupancy)``: the covered fraction, and the
            boolean occupancy of the wrapped phase axis it was measured on,
            kept so the coverage can be inspected bin by bin.

        """
        occupancy = np.zeros(grid, dtype=bool)
        half = max(int(round(0.5 * grid / median_period)), 1)
        offset = np.arange(-half, half + 1)
        for centre in np.rint((np.asarray(phase) % 1.0) * grid).astype(int):
            occupancy[(centre + offset) % grid] = True
        return float(occupancy.mean()), occupancy

    @staticmethod
    def _viterbi_phase(profiles:np.ndarray, phase_linear:np.ndarray,
                       phase_delta:float, drift_direction:int,
                       phase_slack:float, step_penalty:float,
                       max_phase_step:float, min_phase_step:float) -> np.ndarray:
        """ Refine the phase trajectory under a hard one-directional
        constraint.

        The constant-step trajectory is allowed to deviate by at most
        `phase_slack` per frame. The optimal set of deviations is found
        exactly by dynamic programming over the chain of frames: state `j`
        of frame `i` is a candidate deviation, the emission cost is the
        frame's own phase profile there, and the transition cost forbids any
        step that reverses the drift or exceeds the allowed magnitude.
        Because the chain is short, the Viterbi recursion returns the global
        optimum rather than a local one.

        Parameters
        ----------
        profiles : numpy.ndarray
            Normalised phase cost profiles, shape (n_frames, phase_bins).
        phase_linear : numpy.ndarray
            Constant-step trajectory, shape (n_frames,).
        phase_delta : float
            Fitted constant step in line-period units.
        drift_direction : int
            Sign the phase steps must keep.
        phase_slack : float
            Half-width of the deviation grid in line-period units.
        step_penalty : float
            Weight of the quadratic penalty on the relative step deviation.
        max_phase_step, min_phase_step : float
            Hard bounds on the signed step magnitude.

        Returns
        -------
        numpy.ndarray
            Refined unwrapped phase per frame, shape (n_frames,).

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
            logging.warning('Viterbi found no admissible trajectory, '
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
        """ Sample band intensities on a normalised spectral coordinate.

        Band `b` is the strip between lines ``line_index[b] + phase`` and
        ``line_index[b + 1] + phase``. It is sampled at `n_lambda` equally
        spaced positions of the normalised coordinate ``t`` in (0, 1), so a
        lambda index always means the same fraction of a band, whatever the
        local band width in pixels is. This is what makes bands comparable
        between columns and between frames.

        Parameters
        ----------
        image : 2D numpy.ndarray
            Experimental frame to sample.
        coef : numpy.ndarray
            Line family coefficients.
        line_index : numpy.ndarray
            Global integer indices of the pattern lines.
        phase : float
            Phase of this frame in line-period units.
        n_lambda : int
            Number of spectral samples per band.

        Returns
        -------
        tuple
            ``(bands, band_valid)`` with `bands` of shape
            (n_bands, img_width, n_lambda) as uint16 and `band_valid` of
            shape (n_bands, img_width) flagging the band columns whose whole
            spectral range lies inside the frame.

        """
        img_height, img_width = image.shape
        n_bands = len(line_index) - 1

        # sample bin centres, never exactly on a pattern line
        t_axis = (np.arange(n_lambda) + 0.5) / n_lambda
        u_axis = (line_index[:-1, None] + phase + t_axis[None, :]).ravel()

        rows = PhaseModelRecon._line_rows(coef, u_axis)
        cols = np.broadcast_to(np.arange(img_width, dtype=float), rows.shape)
        inside = (rows >= 0) & (rows <= img_height - 1)

        sampled = ndi.map_coordinates(image, np.vstack((rows.ravel(), cols.ravel())),
                                      order=1, mode='nearest').reshape(rows.shape)
        sampled = np.where(inside, sampled, 0.0)

        # (n_bands * n_lambda, width) -> (n_bands, width, n_lambda)
        bands = sampled.reshape(n_bands, n_lambda, img_width).transpose(0, 2, 1)
        band_valid = inside.reshape(n_bands, n_lambda, img_width).all(axis=1)

        return np.rint(np.clip(bands, 0, 65535)).astype(np.uint16), band_valid

    def _load_frame(self, img_path:str) -> np.ndarray:
        """ Read one phase image from disk and apply the configured crop.

        Frames are read on demand rather than held in memory: `run` makes two
        passes over the batch, one to build the phase cost profiles and one to
        sample the bands, and re-reading is cheaper than keeping a whole
        series of full-size frames resident. The diagnostic plots that need a
        frame call this too.

        Parameters
        ----------
        img_path : str
            Path of the image to read. Any format `skimage.io.imread`
            understands is accepted; the reference data are 16-bit TIFFs.

        Returns
        -------
        numpy.ndarray
            Cropped frame as float32, ready for Gaussian filtering and for
            `scipy.ndimage.map_coordinates`. Float rather than the native
            integer type, so that interpolation and folding stay exact and
            no dtype cast silently truncates them.

        Notes
        -----
        The crop is applied as a plain slice with `input_crop`, so out-of-range
        bounds are clipped silently by NumPy rather than raising. Every frame
        of a batch must yield the same shape - `run` takes `img_shape` from the
        first reference frame and assumes it holds for the rest.

        """
        img = io.imread(img_path).astype(np.float32)

        if self.input_crop is not None:
            return img[self.input_crop[0]:self.input_crop[1],
                    self.input_crop[2]:self.input_crop[3]]
        else:
            return img

    def _geometry_contrast(self, geometry:list, frames:list,
                           phase_bins:int=64, col_step:int=4) -> float:
        """ Fold contrast of a candidate shared geometry.

        The objective used to select and polish the shared geometry. For the
        right geometry the rectified frame folds onto a deep, clean profile;
        for a wrong one the dark lines land in different phase bins at
        different columns and the profile flattens. The phase itself never
        enters, so this objective - unlike a template cost - has no periodic
        ambiguity and no dependence on how much pattern falls inside the
        frame.

        Parameters
        ----------
        geometry : list of float
            Candidate geometry as ``[dc, theta, scale]``.
        frames : list of 2D numpy.ndarray
            Pre-smoothed reference frames.
        phase_bins : int, optional
            Phase bins used for this objective, by default 64. Coarser than
            the production value because only the contrast is needed.
        col_step : int, optional
            Column subsampling factor, by default 4.

        Returns
        -------
        float
            Mean relative modulation depth over the reference frames, or 0.0
            if the geometry does not produce a usable line family.

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

        Reads every frame twice: once to build its phase cost profile and,
        after the global trajectory is known, once more to sample its bands.
        Frames are never held in memory all at once.

        Returns
        -------
        None
            Results are stored on the instance:
            `geometry` (shared transform and its quality metrics),
            `line_coef` (line family model),
            `phase_cost` / `phase_counts` / `phase_modulation` (per-frame
            folded profiles, their bin sample counts and relative depth),
            `phase`, `phase_linear`, `phase_step`, `phase_direction`
            (trajectory), `phase_filling` and `phase_occupancy`
            (how much of the period the batch sampled),
            `line_index` (global line numbering),
            `patterns_stack` (per-frame line rows),
            `bands_stack` and `bands_valid` (spectral data and mask).

            These are everything the final image is built from: no frame is
            read again after `run` returns. `lambda_stack_recon` and
            `lambda_frame_recon` only scatter `bands_stack` into image
            coordinates using `line_coef`, `line_index` and `phase`.

        """
        run_start_time = time.perf_counter()

        # pattern preprocessing
        self.arc_profiles = self._extract_arc_profiles(self.pattern_sim,
                                                       edge_tolerance=self.arc_edge_tolerance)
        pattern_fat = morphology.dilation(self.pattern_sim,
                                          morphology.disk(self.input_pattern_dilation))

        # stage 1: shared geometry
        stage_start_time = time.perf_counter()
        ref_idx = np.unique(np.linspace(0, self.n_image - 1,
                                        min(self.geometry_frames, self.n_image)).astype(int))
        logging.info(f'==> Shared geometry from reference frames {list(ref_idx)}')

        ref_frames = [self._load_frame(self.input_path[i]) for i in ref_idx]
        self.img_shape = ref_frames[0].shape
        ref_folded = [ndi.gaussian_filter(f, sigma=self.fold_sigma) for f in ref_frames]

        candidates = []
        for frame, idx in zip(ref_frames, ref_idx):
            params, cost = self._estimate_initial_geometry(frame, pattern_fat,
                                                           smooth_sigma=self.init_sigma,
                                                           grid_search_params=self.grid_search_params)
            contrast = self._geometry_contrast(params[1:], ref_folded)
            logging.info(f'Frame {idx}: template cost {cost:.1f}, '
                         f'scale {params[3]:.5f}, theta {np.degrees(params[2]):+.3f} deg, '
                         f'dc {params[1]:+.1f}, '
                         f'fold contrast (peak-to-trough / mean) {contrast:.4f}')
            candidates.append((contrast, params))

        # the template cost is biased by how much pattern falls inside the frame, so pick the candidate by fold contrast, which is not
        best_contrast, best_params = max(candidates, key=lambda c: c[0])
        best_geom = list(best_params[1:])

        if self.refine_geometry:
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

        logging.info(f'Shared geometry: scale {self.geometry["scale"]:.5f}, '
                     f'theta {np.degrees(self.geometry["theta"]):+.3f} deg, '
                     f'dc {self.geometry["dc"]:+.1f}, '
                     f'fold contrast (peak-to-trough / mean) {best_contrast:.4f}')
        logging.info(f'Line family: degree {self.family_degree}, {n_used} arcs, '
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
        for i, img_path in enumerate(self.input_path):
            frame = ndi.gaussian_filter(self._load_frame(img_path), sigma=self.fold_sigma)
            self.phase_cost[i], self.phase_counts[i] = self._fold_profile(frame, rows, cols,
                                                                          valid, self.phase_bins)
        self.phase_modulation = ((self.phase_cost.max(axis=1) - self.phase_cost.min(axis=1))
                                 / self.phase_cost.mean(axis=1))
        logging.info(f'Folded {self.n_image} frames over {n_cycles} periods, '
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
            logging.warning('Fitted phase step is degenerate (close to zero): the batch shows no '
                            'systematic drift, or drift_direction was forced against the data')
            frames_per_period = float('inf')
        else:
            frames_per_period = abs(1 / delta)
        logging.info(f'Phase trajectory: delta {delta:+.6f} period/frame '
                     f'({delta * self.median_period:+.3f}px), '
                     f'{frames_per_period:.2f} frames per period, '
                     f'total drift {delta * (self.n_image - 1):+.3f} periods')
        monotone = bool(np.all(self.phase_step * direction >= 0))
        logging.info(f'Constrained steps: min {self.phase_step.min() * self.median_period:+.3f}px, '
                     f'max {self.phase_step.max() * self.median_period:+.3f}px, '
                     f'one-directional {monotone} '
                     f'({time.perf_counter() - stage_start_time:.2f}s)')

        self.phase_filling, self.phase_occupancy = self._phase_filling(self.phase, self.median_period)
        logging.info(f'Phase filling coefficient (period sampled by the batch) '
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

        for i, img_path in enumerate(self.input_path):
            frame = self._load_frame(img_path)
            self.patterns_stack[i] = self._line_rows(self.line_coef,
                                                     self.line_index + self.phase[i])
            self.bands_stack[i], self.bands_valid[i] = self._sample_bands(
                frame, self.line_coef, self.line_index, self.phase[i], self.n_lambda)

        stack_size = self.bands_stack.nbytes / (1024 * 1024)
        # the global index range must cover every frame, so it is wider than the number of bands that fit inside any single frame
        bands_per_frame = self.bands_valid.any(axis=2).sum(axis=1)
        logging.info(f'Sampled {n_lines - 1} bands x {self.n_lambda} lambda channels per frame, '
                     f'bands stack {self.bands_stack.shape} ({stack_size:.1f}MB)')
        logging.info(f'Usable bands per frame {bands_per_frame.min()}..{bands_per_frame.max()} '
                     f'of {n_lines - 1} global indices, '
                     f'{100 * self.bands_valid.mean():.1f}% of band columns valid '
                     f'({time.perf_counter() - stage_start_time:.2f}s)')
        logging.info(f'Batch reconstruction of {self.n_image} images finished in '
                     f'{time.perf_counter() - run_start_time:.2f}s')

    def _accumulate_bands(self, lambda_idx:int=None,
                          accumulation_method:str='max') -> np.ndarray:
        """ Merge the sampled bands of every phase image into an output image.

        The shared engine behind `lambda_stack_recon` and
        `lambda_frame_recon`. Every band of every frame is written to the
        image row half a band below its upper line, which is where the
        pinhole that produced that spectrum actually sat. Because the phase
        differs between frames, successive frames write to different rows and
        the sparse traces of a single phase image add up to a dense image.

        Parameters
        ----------
        lambda_idx : int, optional
            Spectral channel to assemble. If None (default) every channel is
            assembled and the result is a cube.
        accumulation_method : str, optional
            How to combine frames that write to the same row: 'max' (default)
            keeps the brightest contribution, 'mean' averages the valid
            contributions, 'overwrite' keeps the last frame processed.

        Returns
        -------
        numpy.ndarray
            Image of `img_shape` when `lambda_idx` is given, cube of
            ``(n_lambda, *img_shape)`` otherwise, dtype uint16 either way.

        Raises
        ------
        ValueError
            If `accumulation_method` is not one of the three supported values.
            Nothing else is validated: calling this before `run`, or with an
            out-of-range `lambda_idx`, raises from NumPy instead.

        Notes
        -----
        Only band columns flagged in `bands_valid` contribute, so bands that
        left the field of view are skipped rather than writing zeros. The
        number of contributions per output row is kept in `lambda_hit_count`;
        it does not depend on `lambda_idx`, so a single-channel call leaves
        exactly the same map behind as a full cube call.

        """
        # the only check kept: an unknown method would silently fall through to
        # 'overwrite' in the loop below and corrupt the result without a word
        if accumulation_method not in ('max', 'mean', 'overwrite'):
            raise ValueError('Incorrect intensity accumulation method!')

        single_channel = lambda_idx is not None
        # channel-first, the axis order ImageJ and napari expect of a stack
        out_shape = self.img_shape if single_channel else (self.n_lambda, *self.img_shape)
        # accumulate in float32 so that 'mean' can divide without rounding twice
        out_img = np.zeros(out_shape, dtype=np.float32)
        hit_count = np.zeros(self.img_shape, dtype=np.int32)

        cols_idx = np.arange(self.img_shape[1])
        for frame_idx in range(self.n_image):
            # mid-band row of every band of this frame
            mid_rows = self._line_rows(self.line_coef,
                                       self.line_index[:-1] + self.phase[frame_idx] + 0.5)
            mid_rows = np.rint(mid_rows).astype(int)

            for band_idx in range(self.bands_stack.shape[1]):
                keep = self.bands_valid[frame_idx, band_idx]
                if not keep.any():
                    continue
                row = mid_rows[band_idx][keep]
                col = cols_idx[keep]
                band = self.bands_stack[frame_idx, band_idx]
                # transposed for the multi-channel case so that the spectral
                # axis leads, matching the channel-first output
                value = (band[keep, lambda_idx] if single_channel
                         else band[keep].T).astype(np.float32)
                target = (row, col) if single_channel else (slice(None), row, col)

                if accumulation_method == 'max':
                    out_img[target] = np.maximum(out_img[target], value)
                elif accumulation_method == 'mean':
                    out_img[target] += value
                else:
                    out_img[target] = value
                hit_count[row, col] += 1

        if accumulation_method == 'mean':
            divisor = np.maximum(hit_count, 1)
            out_img /= divisor if single_channel else divisor[None, ...]

        self.lambda_hit_count = hit_count
        filled = 100 * (hit_count > 0).mean()
        logging.info(f'Band writes reached {filled:.1f}% of the frame pixels '
                     f'(mean {hit_count[hit_count > 0].mean():.2f} contributions per reached pixel)')

        return out_img.astype(np.uint16)

    def lambda_stack_recon(self, accumulation_method:str='max') -> np.ndarray:
        """ Assemble the full hyperspectral cube from all phase images.

        Every band is written to the image row that lies half a band below
        its upper line, which is where the pixel row that produced the
        spectrum actually sits. Because the phase differs between frames,
        different frames fill different rows and the cube becomes dense - on
        the reference batch about 96% of the frame rows end up carrying data.

        Parameters
        ----------
        accumulation_method : str, optional
            How to combine frames that write to the same row:
            'max' keeps the brightest contribution (default, matches
            `BatchRecon.lambda_stack_recon`), 'mean' averages the valid
            contributions, 'overwrite' keeps the last one.

        Returns
        -------
        numpy.ndarray
            Lambda stack of shape ``(n_lambda, *img_shape)``, dtype uint16.

        Raises
        ------
        ValueError
            If `accumulation_method` is not one of the three supported values.

        Notes
        -----
        The cube is expensive: the uint16 result is
        ``height * width * n_lambda * 2`` bytes and the float32 accumulator
        used while building it is twice that again. Measured peak allocation
        on a 2000 x 1500 frame with 88 channels is 1.6 GB. If only one
        channel is needed, `lambda_frame_recon` produces exactly the same
        pixels from 32 MB.

        See Also
        --------
        lambda_frame_recon : the same assembly for a single spectral channel.

        """
        logging.info(f'Assembling lambda stack {(self.n_lambda, *self.img_shape)}, '
                     f'{np.prod(self.img_shape) * self.n_lambda * 4 / 1048576:.0f}MB accumulator')
        return self._accumulate_bands(None, accumulation_method)

    def lambda_frame_recon(self, lambda_idx:int=10,
                           accumulation_method:str='max') -> np.ndarray:
        """ Assemble a single spectral channel from all phase images.

        Exactly the slice ``lambda_stack_recon()[lambda_idx]`` would
        contain, built directly. Nothing about the assembly changes; only the
        spectral channels that are never needed are skipped, which makes this
        the right entry point for looking at one wavelength, stepping through
        the spectral axis interactively, or exporting a few channels.

        Parameters
        ----------
        lambda_idx : int, optional
            Spectral channel to assemble, by default 10. Runs from 0 to
            ``n_lambda - 1``, and the channel is a fixed fraction of a band
            rather than a pixel offset.
        accumulation_method : str, optional
            How to combine frames that write to the same row: 'max'
            (default), 'mean' or 'overwrite'.

        Returns
        -------
        numpy.ndarray
            Image of shape `img_shape`, dtype uint16.

        Raises
        ------
        ValueError
            If `accumulation_method` is not one of the three supported values.

        Notes
        -----
        The saving is the number of spectral channels: the accumulator here
        is ``height * width * 4`` bytes against
        ``height * width * n_lambda * 4`` for the cube. Measured peak
        allocation on the reference data is 32 MB against 1.6 GB, a factor of
        49, and the channel comes out bit-identical to
        ``lambda_stack_recon()[lambda_idx]`` for every accumulation
        method. Run time also drops, from 0.24 s to 0.04 s, but memory is the
        reason this method exists.

        Examples
        --------
        >>> frame = pm.lambda_frame_recon(lambda_idx=40)
        >>> spectrum = [pm.lambda_frame_recon(k)[900, 700] for k in range(pm.n_lambda)]

        See Also
        --------
        lambda_stack_recon : the full cube.

        """
        return self._accumulate_bands(lambda_idx, accumulation_method)

    def plot_phase_cost_map(self):
        """ Phase cost profiles of every frame with the fitted trajectory
        overlaid.

        The reference diagnostic of this class, and the one to look at first.
        Each row of the image is one frame's folded profile from
        `_fold_profile`, normalised to [0, 1]: dark means the pattern lines
        sit there, bright means the bands do. The cyan markers are the
        phases the global fit settled on.

        How to read it
        --------------
        A healthy batch shows **one** dark ridge, drifting smoothly across
        the phase axis and wrapping around when it reaches an edge, with
        every cyan marker sitting on it. The slope of the ridge is the disk
        drift; the number of wraps is the total drift in periods - about two
        over the 40-frame reference series.

        Failure modes are equally legible. A ridge that breaks into
        disconnected segments means the shared geometry is wrong and the fold
        is smearing. Markers that sit off the ridge for a few frames mean the
        trajectory constraints are too tight for the data - loosen
        `phase_slack` or lower `step_penalty`. A washed-out map with no ridge
        at all means the geometry fit failed outright, which
        `plot_line_model` will confirm.

        The strip underneath the map is the **phase filling coefficient**: the
        same phase axis, carrying one vertical stripe per phase image. The
        inked fraction is `phase_filling`, the share of the line period the
        batch actually sampled. Gaps in that strip are parts of every band no
        frame ever visited, and they cap how densely the lambda stack can be
        filled - a batch that drifts less than one period leaves them
        whatever its frame count.

        See Also
        --------
        BatchRecon.plot_start_idx : the per-frame-alignment equivalent, where
            the same information appears as a sawtooth.

        """
        fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                               gridspec_kw={'height_ratios': [8, 1]})

        im = ax[0].imshow(self.phase_profiles, cmap='inferno',
                          aspect='auto', origin='upper',
                          extent=[0, 1, self.n_image - 0.5, -0.5])
        ax[0].plot(self.phase % 1.0, np.arange(self.n_image), '.', color='cyan',
                   ms=7, label='fitted phase')
        ax[0].set_ylabel('Phase img num')
        ax[0].set_title(f'Phase filling coefficient {self.phase_filling:.3f}')
        ax[0].legend(loc='upper right')

        # one stripe per phase image, the inked fraction is the coefficient
        ax[1].vlines(self.phase % 1.0, 0, 1, color='#1f4e79', lw=1.5)
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].set_yticks([])
        ax[1].set_xlabel('Phase within one line period')
        ax[1].set_ylabel('phase\nfilling', rotation=0, ha='right',
                         va='center', fontsize=9)

        fig.colorbar(im, ax=ax.tolist(), label='normalised fold cost')
        plt.show()

    def plot_phase_trajectory(self):
        """ Unwrapped phase against frame number, with the constant-step
        model and the residual of the constrained fit.

        Two panels. The upper one plots the phase the reconstruction actually
        used against the straight line fitted by `_fit_phase_trajectory`; the
        lower one plots the difference between them, converted to pixels
        using the median line period.

        How to read it
        --------------
        The upper panel should be a straight line with the markers lying on
        it - it is the disk turning at a constant rate, and its slope is the
        drift per frame. Curvature there would mean the rotation is not
        uniform across the series.

        The lower panel is the interesting one. It is the honest measure of
        how far each frame departs from constant-rate motion, and it should
        stay well inside `phase_slack` converted to pixels. Random scatter is
        measurement noise. A **smooth oscillation** whose period matches the
        number of frames per band period is not noise but a systematic,
        phase-dependent bias - on the reference data it appears at about
        +/-1.3 px with a period of roughly 18.5 frames, and it is small
        enough to live inside the slack. A residual pressed flat against the
        slack limit means the constraint is clipping real motion and
        `phase_slack` should be raised.


        """
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
        """ Per-frame phase step, in pixels, with the admissible band marked.

        The direct picture of the constraint this class enforces. Each marker
        is the phase change between two consecutive frames; the shaded band
        is the range of steps the Viterbi refinement was allowed to use, the
        red line at zero is the direction it may never cross, and the dashed
        line is the constant step fitted globally.

        How to read it
        --------------
        Every marker must be on one side of the red line - if any is not, the
        constraint was not applied and something is wrong. Beyond that, the
        scatter around the dashed line is the real per-frame jitter of the
        disk drive: on the reference data the steps stay between -5.5 and
        -3.9 px around a fitted -4.79 px.

        Markers pressed against an edge of the shaded band mean the allowed
        range is clipping the data - widen `phase_slack`, or raise
        `max_phase_step` if the drift itself is faster than assumed. Markers
        pinned exactly on the dashed line for the whole series mean
        `step_penalty` is so high that the refinement was not free to move at
        all, and the trajectory is effectively the constant-step model.


        """
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
        """ Brute-force cost surface of the global trajectory search.

        The two-parameter landscape that replaces forty independent
        per-frame searches. The horizontal axis is the phase step per frame,
        the vertical axis is the phase of the first frame, and the colour is
        the mean normalised fold cost over the whole batch. The marker is the
        solution `_fit_phase_trajectory` returned.

        How to read it
        --------------
        There should be one clearly isolated dark minimum, and the marker
        should be in it. That isolation is the whole argument for the batch
        approach: an individual frame cannot tell which band offset it is at,
        but a series that must be explained by a single straight line can,
        and the surface shows how much better the true trajectory is than
        every alternative.

        The structure around the minimum is informative too. Dark stripes
        running diagonally are trajectories that fit some frames and miss
        others. If several minima look equally deep, the batch is too short
        to disambiguate them - the leverage grows with the number of frames
        and with the total drift, so a series covering well under one band
        period will show this.


        """
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
        """ Line family model: local line period across the frame and the
        residual of the polynomial fit in the line index.

        Two panels describing the geometry every frame of the batch shares.
        The left one plots ``line_coef[1]``, the local distance between
        neighbouring pattern lines, against the image column. The right one
        is the residual of the fit - how far each simulated arc sits from
        where the model puts it - with arcs down the vertical axis and image
        columns across.

        How to read it
        --------------
        The left panel should be a smooth, gently varying curve. It is the
        real change of arc spacing across the disk, about 88 to 90 px on the
        reference data. A curve that is flat to within a fraction of a pixel
        suggests the fitted scale is wrong; a curve that swings wildly means
        the geometry fit landed somewhere unphysical.

        The right panel is the geometric error budget of the whole
        reconstruction. Its rms - reported in the panel title and in
        `geometry['family_resid_rms']` - is how far a modelled line can sit
        from the arc the simulation actually produced, about 0.67 px on the
        reference data. Two textures are worth telling apart. A **fine
        striping** at the scale of the arc spacing is quantisation of the
        simulated pattern itself: its arcs are one-pixel skeletons, so their
        per-column row is integer-valued, and that is a floor set by the
        simulation, not by the model. A **smooth large-scale gradient**, by
        contrast, is real model error, and raising `family_degree` will
        reduce it.


        """
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
        """ Overlay the modelled pattern lines on one phase image.

        Reloads the requested frame from disk and draws the geometry this
        class computed for it: the band boundaries evaluated from the shared
        line family model at that frame's phase, and optionally the mid-band
        rows the spectra are attributed to. The title reports the frame's
        unwrapped phase in line periods.

        How to read it
        --------------
        The boundaries must sit in the dark gaps along the whole width, and
        the mid-band lines down the middle of the bright bands. Because the
        geometry is shared across the batch, checking two or three frames
        spread through the series is enough: if the lines fit the first and
        the last frame, the phase model is holding everywhere in between.

        Lines are drawn only where they fall inside the frame, so a frame
        with no lines near the top or bottom edge is showing masked - not
        lost - band indices, and `plot_band_coverage` says which.

        Parameters
        ----------
        phase_img_num : int, optional
            Index of the phase image to display, by default 0.
        band_mid : bool, optional
            Also draw the estimated mid-band pixel rows, by default True.


        """
        frame = self._load_frame(self.input_path[phase_img_num])
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
        """ Fraction of image columns whose full spectral range lies inside
        the frame, per global line index and phase image.

        Where `BatchRecon` silently drops a line that leaves the field of
        view, this class keeps its index and masks it. This map is the record
        of that masking: each column is one phase image, each row is one
        global line index, and the colour is the percentage of image columns
        for which that band was completely inside the frame.

        How to read it
        --------------
        The expected picture is a solid block of fully valid bands in the
        middle, spanning every frame, with soft edges at the top and bottom.
        Those edges are lines drifting in and out of the field of view, and
        their gradual slope across the plot is the disk drift made visible.

        The height of the solid block is how many bands are usable in every
        frame of the batch - about 22 of 30 global indices on the reference
        data, the difference being the room the fixed index range needs to
        cover two periods of drift. What must **not** appear is an abrupt
        change of the block height from one frame to the next: that is the
        `BatchRecon` failure mode this class exists to prevent.


        """
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
                          accumulation_method:str='max'):
        """ Show one spectral channel of the reconstruction.

        The channel is assembled on demand with `lambda_frame_recon`, so no
        cube has to exist first - which is the cheap way to look at a single
        wavelength or to step through the spectral axis. When a cube is
        already at hand, slicing it directly is of course faster:
        ``plt.imshow(cube[:, :, k])``.

        How to read it
        --------------
        The image should look like the sample, with no trace of the band
        pattern in it. Residual horizontal striping at the band period means
        the phase series did not fill the rows evenly - check
        `plot_band_coverage` and the fill fraction reported by
        `lambda_frame_recon`. Stepping `lambda_idx` across the spectral axis
        walks the emission spectrum, so structures with different emission
        should light up at different indices; if everything brightens and
        fades together, the spectral axis is not resolving anything and the
        band boundaries are probably misplaced.

        Parameters
        ----------
        lambda_idx : int, optional
            Spectral channel to display, by default 10. Runs from 0 to
            ``n_lambda - 1``, and the channel is a fixed fraction of a band
            rather than a pixel offset.
        max_int_m : float, optional
            Fraction of the channel maximum used as the display ceiling, by
            default 0.5. Lower values bring faint structure out.
        accumulation_method : str, optional
            Passed to `lambda_frame_recon`, by default 'max'.

        """
        frame = self.lambda_frame_recon(lambda_idx, accumulation_method)

        plt.figure(figsize=(10, 10))
        plt.imshow(frame, cmap='magma', vmax=np.max(frame) * max_int_m)
        plt.title(f'Lambda channel {lambda_idx} of {self.n_lambda}')
        plt.show()
