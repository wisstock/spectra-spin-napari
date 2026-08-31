""" Qt widgets of the `spectra-spin-napari` plugin.

Three dock widgets are contributed to napari:

* `PhaseModelReconWidget` - collects the input batch and the parameters of
  `PhaseModelRecon`, runs the reconstruction against the bundled simulated
  disk pattern and adds the assembled lambda stack, optionally together with
  the input batch and the modelled arcs, as new layers.
* `PostProcessingWidget` - applies the post-processing helpers of `utils`
  (`interpolate_zero_gaps`, `max_pooling2d`) to an existing image layer.
* `SpectraWidget` - mean spectra of the regions of interest of a labels
  layer, the wavelength calibration fitted from their emission peaks, and
  the export of the spectra to CSV.

Every widget runs its work in a `QThread`, so the napari window stays
responsive, and mirrors the `logging` output of the processing modules into a
status label next to the progress bar. The bar is driven by the work itself:
the loops of this module count their own iterations, and the reconstruction
is followed through `_ProgressRecon`. The single step whose length cannot be
known in advance - the Nelder-Mead refinement of the shared geometry - is the
only one that puts the bar into busy mode.

"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from napari.layers import Image, Labels
from napari.utils.theme import get_theme

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QProgressBar, QPushButton, QSizePolicy,
                               QSpinBox, QVBoxLayout, QWidget)

logger = logging.getLogger(__name__)

# The processing modules and matplotlib are imported where they are first
# used, not here. Opening a widget is not a reason to pay for scipy,
# scikit-image and pyplot: the three of them cost about three quarters of a
# second, which is the whole of what opening a widget used to take, and a
# widget that is merely opened has no use for any of them.

PATTERN_PATH = Path(__file__).parent / 'simulated_disk_img.tiff'
SAMPLE_PATH = Path(__file__).parent / 'qd_mix_batch.tif'
SAMPLE_ROIS_PATH = Path(__file__).parent / 'qd_mix_rois.tiff'

# starting point of the peak detection, shared by the fit window and by the
# calibration, which fits the same way before the window is ever opened
FIT_DEFAULTS = {'window': 15, 'poly': 3, 'noise_thresh': 0.02}


def layer_name(*parts) -> str:
    """ Join name parts into a layer name that is also a safe file name.

    A layer name becomes a file name the moment the layer is saved, so it
    carries no spaces: parts are joined with underscores and any whitespace
    inside a part - a folder called ``QD mix``, a layer napari named
    ``stack - Labels`` - becomes an underscore too. A word that is only
    punctuation is dropped rather than turned into an underscore of its own,
    which is what keeps ``a - b`` from becoming ``a_-_b``.

    Parameters
    ----------
    *parts : str
        Name parts, in order. Empty parts are skipped.

    Returns
    -------
    str
        The joined name.

    Examples
    --------
    >>> layer_name('lambda_stack', 'QD mix')
    'lambda_stack_QD_mix'
    >>> layer_name('lambda stack - Labels', '2D')
    'lambda_stack_Labels_2D'

    """
    words = [word for part in parts for word in str(part).split()
             if any(character.isalnum() for character in word)]
    return '_'.join(words)


# suffixes this plugin appends to a layer name, longest first, so that the
# batch a layer came from can be recovered from the name of any of them
LAYER_SUFFIXES = ('_lambda_stack', '_input_batch', '_arcs',
                  '_interpolated', '_pooled')


def batch_prefix(name:str) -> str:
    """ The batch name a plugin layer carries at the front of its own name.

    Every layer the plugin produces starts with the name of the batch it came
    from - the input folder, or the layer of phase images - so that the origin
    of a result is visible in the layer list and in the file it is saved to.
    This strips the suffixes the plugin appends, which is how a widget working
    on one of those layers can put the same batch name in front of its own
    results.

    Parameters
    ----------
    name : str
        Layer name.

    Returns
    -------
    str
        The name without the plugin suffixes. A layer the plugin did not make
        is returned unchanged and simply acts as its own batch name.

    Examples
    --------
    >>> batch_prefix('QD_mix_lambda_stack_interpolated')
    'QD_mix'

    """
    changed = True
    while changed:
        changed = False
        for suffix in LAYER_SUFFIXES:
            if name.endswith(suffix) and len(name) > len(suffix):
                name, changed = name[:-len(suffix)], True
    return name


def qd_mix_sample() -> list:
    """ Sample batch contributed to *File - Open Sample*.

    The reference batch of the project: 18 phase images of a mix of quantum
    dots emitting at 525, 585 and 659 nm. The frames are the demo crop
    ``[500, 2500, 1000, 2500]`` of the raw 3000 x 4096 camera images, stored
    as one compressed stack - 60 MB against the 422 MB the raw series takes,
    and exactly the region the reference workflow reconstructs.

    The batch arrives as an ordinary image layer, which is all
    `PhaseModelReconWidget` needs: its input source can be a layer as well as
    a folder, so the sample runs through the plugin with no special case
    anywhere.

    Returns
    -------
    list
        One napari LayerData tuple, ``(data, meta, 'image')``.

    See Also
    --------
    qd_mix_rois_sample : regions of interest drawn on this batch.

    """
    from skimage import io

    stack = io.imread(SAMPLE_PATH)
    logger.info(f'Loaded the bundled sample batch {stack.shape} from {SAMPLE_PATH.name}')
    return [(stack, {'name': 'QD_mix_phase_images'}, 'image')]


def qd_mix_rois_sample() -> list:
    """ Regions of interest of the sample batch, contributed as sample data.

    Four regions drawn on the reconstruction of `qd_mix_sample`, in the frame
    geometry of that batch, so they can be handed straight to the ROI spectra
    widget without drawing anything by hand.

    Returns
    -------
    list
        One napari LayerData tuple, ``(data, meta, 'labels')``.

    """
    from skimage import io

    mask = io.imread(SAMPLE_ROIS_PATH)
    logger.info(f'Loaded the bundled sample regions {mask.shape} from '
                f'{SAMPLE_ROIS_PATH.name}, labels {np.unique(mask[mask > 0]).tolist()}')
    return [(mask, {'name': 'QD_mix_rois'}, 'labels')]


@lru_cache(maxsize=1)
def bundled_pattern() -> np.ndarray:
    """ Simulated disk pattern shipped with the plugin.

    The reference pattern belongs to the instrument rather than to a single
    batch, so it travels with the plugin instead of being loaded by hand.
    The stored file is the raw simulation, and two fixed steps turn it into
    what `PhaseModelRecon` expects: a rotation by 180 degrees, because the
    simulation is drawn in the opposite orientation to the camera frames,
    and a skeleton, because the line family model is fitted to one-pixel-wide
    arcs.

    Both steps are fixed rather than optional: on the reference batch the
    flipped pattern reaches a fold contrast of 1.90 against 1.10 for the
    unflipped one, which is the difference between a locked geometry and a
    meaningless fit.

    Returns
    -------
    numpy.ndarray
        Two-dimensional boolean skeleton of the pattern. The result is
        cached, so the file is read and skeletonised once per session.

    """
    from skimage import io, morphology

    raw = io.imread(PATTERN_PATH)
    pattern = morphology.skeletonize(np.flip(raw) > 0)
    logger.info(f'Loaded the bundled pattern {PATTERN_PATH.name} {pattern.shape}, '
                f'{int(pattern.sum())} skeleton pixels')
    return pattern


class _StatusLabel(QLabel):
    """ Text label that never imposes its own width on the widget.

    The status line reports whatever the processing modules log, and an error
    message is easily several hundred pixels of text. A plain label answers
    the layout with the width of its text, which is how a single error
    message could widen the dock and keep it wide: the minimum width of the
    whole widget grew with the message and the user could not drag it back.

    This label reports no width of its own, elides the text into whatever
    width it is given, and keeps the full message in the tooltip.

    """

    def __init__(self):
        super().__init__()
        self._message = ''
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def message(self) -> str:
        """ The full message, as it was set. """
        return self._message

    def setMessage(self, message:str):
        """ Show `message` elided, with the full text as the tooltip.

        Parameters
        ----------
        message : str
            Text to show. Line breaks are kept and every line is elided on
            its own.

        """
        self._message = message
        self.setToolTip(message)
        self._elide()

    def resizeEvent(self, event):
        """ Re-elide the message whenever the width changes. """
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        """ Fit every line of the message into the current width. """
        metrics = QFontMetrics(self.font())
        width = max(self.width() - 2, 32)
        super().setText('\n'.join(
            metrics.elidedText(line, Qt.TextElideMode.ElideRight, width)
            for line in (self._message.splitlines() or [''])))


class _LogRelay(logging.Handler):
    """ Logging handler that forwards every record to a Qt signal.

    The processing modules report their progress with the `logging` module
    and know nothing about the GUI. Attaching this handler to the root
    logger for the lifetime of a background task is what puts their
    checkpoints into the status label of the widget.

    Parameters
    ----------
    signal : PySide6.QtCore.SignalInstance
        Signal taking a single string, emitted once per log record.

    """

    def __init__(self, signal):
        super().__init__(level=logging.INFO)
        self._signal = signal

    def emit(self, record:logging.LogRecord):
        """ Forward one formatted record to the connected signal. """
        self._signal.emit(record.getMessage())


class _Worker(QObject):
    """ Runs one blocking callable in a worker thread.

    Parameters
    ----------
    func : callable
        Function of a single argument, a ``report(percent)`` callable the
        task uses to drive the progress bar. A percentage moves a determinate
        bar, a negative value puts it back into busy mode for a step of
        unknown length, and a task that never calls it leaves the bar busy
        throughout. The return value of `func` is delivered by `finished`.

    Attributes
    ----------
    finished : Signal(object)
        Emitted with the return value of `func` on success.
    failed : Signal(str)
        Emitted with the exception message if `func` raises. The full
        traceback goes to the log.
    message : Signal(str)
        One emission per log record produced while `func` runs.
    progress : Signal(int)
        Percentage reported by `func` itself.

    """

    finished = Signal(object)
    failed = Signal(str)
    message = Signal(str)
    progress = Signal(int)

    def __init__(self, func):
        super().__init__()
        self._func = func

    def run(self):
        """ Execute the task, relaying log records and exceptions as signals. """
        relay = _LogRelay(self.message)
        logging.getLogger().addHandler(relay)
        try:
            self.finished.emit(self._func(self.progress.emit))
        except Exception as exc:
            logger.exception('Background task failed')
            self.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            logging.getLogger().removeHandler(relay)


@lru_cache(maxsize=1)
def _progress_recon_class():
    """ Build the progress-reporting `PhaseModelRecon` subclass on first use.

    A subclass needs its base class at definition time, and importing
    `phase_model_recon` pulls in scipy, scikit-image and pyplot. Defining the
    subclass inside this function is what keeps that import out of the path
    of merely opening a widget; the result is cached, so the class is built
    once per session.

    Returns
    -------
    type
        The `_ProgressRecon` subclass described below.

    """
    from .phase_model_recon import PhaseModelRecon

    class _ProgressRecon(PhaseModelRecon):
        """ `PhaseModelRecon` that reports how far a run has got.

        The reconstruction knows nothing about the GUI and must not be edited, so
        its progress is counted from the outside, by extending the two methods
        that do the countable work: `_estimate_initial_geometry`, called once per
        reference frame, and `_load_frame`, called once per frame in each of the
        two passes over the batch.

        The stage weights below are the measured share of the run time on the
        reference batch, where the initial geometry fits take about half of it
        and the two batch passes the rest. The one step that cannot be counted is
        the Nelder-Mead refinement of the shared geometry: it converges when it
        converges, so the bar goes busy for its duration instead of pretending to
        know.

        The same class also serves the second way a batch can arrive. A
        reconstruction normally reads its frames from disk, one path at a time,
        but a batch that is already an image layer - the bundled sample, a stack
        the user opened themselves - has no paths. Passing `stack` makes
        `input_path` a list of plane indices and reads the frames from that array
        instead, which is the whole of what the layer input needs: every other
        stage of `PhaseModelRecon` works on what `_load_frame` returns and cannot
        tell the difference.

        Parameters
        ----------
        report : callable
            ``report(percent)`` of `_Worker`, called with a percentage, or with
            -1 to put the bar into busy mode.
        stack : numpy.ndarray, optional
            Batch as a ``(n_image, height, width)`` array. If None (default) the
            frames are read from `input_path` as usual.
        *args, **kwargs
            Passed to `PhaseModelRecon`.

        """

        GEOMETRY_END = 55    # percent reached when the reference frames are fitted
        FOLD_END = 80        # percent reached when every frame has been folded
        SAMPLING_END = 95    # percent reached when every band has been sampled

        def __init__(self, *args, report=None, stack=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._report = report if report is not None else lambda percent: None
            self._stack = stack
            self._geometry_done = 0
            self._reads = 0
            # mirrors the reference frame choice of PhaseModelRecon.run, so that
            # the reads of the geometry stage are not counted as batch passes
            self._n_ref = len(np.unique(np.linspace(0, self.n_image - 1,
                                                    min(self.geometry_frames,
                                                        self.n_image)).astype(int)))

        def _estimate_initial_geometry(self, *args, **kwargs):
            """ Fit one reference frame and report the geometry stage. """
            geometry = super()._estimate_initial_geometry(*args, **kwargs)
            self._geometry_done += 1
            self._report(int(self.GEOMETRY_END * self._geometry_done / self._n_ref))
            if self._geometry_done >= self._n_ref and self.refine_geometry:
                self._report(-1)  # the optimiser runs for as long as it needs
                logger.info('Refining the shared geometry, this takes a while')
            return geometry

        def _load_frame(self, img_path) -> np.ndarray:
            """ Read one frame and report which pass over the batch it belongs to.

            Parameters
            ----------
            img_path : str or int
                Path of the frame, or its plane index when the batch was given as
                an array.

            Returns
            -------
            numpy.ndarray
                Cropped frame as float32, as `PhaseModelRecon._load_frame` returns
                it.

            """
            if self._stack is None:
                frame = super()._load_frame(img_path)
            else:
                frame = np.asarray(self._stack[int(img_path)], dtype=np.float32)
                if self.input_crop is not None:  # the same slice the class applies
                    frame = frame[self.input_crop[0]:self.input_crop[1],
                                  self.input_crop[2]:self.input_crop[3]]
            self._reads += 1
            pass_read = self._reads - self._n_ref  # 1..n_image folding, then sampling
            if 0 < pass_read <= self.n_image:
                self._report(int(self.GEOMETRY_END + (self.FOLD_END - self.GEOMETRY_END)
                                 * pass_read / self.n_image))
            elif self.n_image < pass_read <= 2 * self.n_image:
                self._report(int(self.FOLD_END + (self.SAMPLING_END - self.FOLD_END)
                                 * (pass_read - self.n_image) / self.n_image))
            return frame

    return _ProgressRecon


def arc_label_stack(recon) -> np.ndarray:
    """ Rasterise the modelled arcs of every frame into a labels volume.

    The line family model is analytic, so the arcs exist as coefficients
    rather than as an image. This draws them back into image coordinates, one
    plane per phase image and the same height and width as the frames the
    reconstruction read, which is what makes them a napari labels layer that
    overlays the input batch pixel for pixel and can be saved as a TIFF.

    Every arc keeps its own label value, shifted so that the lowest global
    line index becomes 1. The value therefore identifies the same physical
    arc in every frame, which is what turns the stack into trajectories: step
    through the frames and one label is one arc moving.

    Parameters
    ----------
    recon : PhaseModelRecon
        A finished reconstruction, after `run`.

    Returns
    -------
    numpy.ndarray
        Labels volume of shape ``(n_image, *img_shape)``, dtype uint16. Arcs
        are drawn only where they fall inside the frame, exactly as
        `PhaseModelRecon.plot_pattern` draws them.

    """
    height, width = recon.img_shape
    cols = np.arange(width)
    labels = np.zeros((recon.n_image, height, width), dtype=np.uint16)
    for i in range(recon.n_image):
        rows = np.rint(recon.patterns_stack[i]).astype(int)
        for value, row in enumerate(rows, start=1):
            inside = (row >= 0) & (row < height)
            labels[i, row[inside], cols[inside]] = value
    logger.info(f'Rasterised {len(recon.line_index)} arcs over {recon.n_image} frames, '
                f'labels {labels.shape} ({labels.nbytes / 1048576:.0f}MB), '
                f'label 1 is line index {int(recon.line_index[0])}')
    return labels


def flatten_roi_mask(mask:np.ndarray) -> np.ndarray:
    """ Reduce a labels volume to the one plane the spectra are measured on.

    Regions painted at different spectral channels sit on different planes of
    the same labels layer, because that is the shape napari gives a labels
    layer created over a lambda stack. They all describe places in the frame,
    so the plane a region was drawn at carries no meaning and the planes are
    collapsed into one.

    Parameters
    ----------
    mask : numpy.ndarray
        Labels array, ``(height, width)`` or ``(planes, height, width)``.

    Returns
    -------
    numpy.ndarray
        Two-dimensional labels array. A 2D input is returned unchanged.

    Notes
    -----
    The collapse takes the largest label of every column of the volume. Two
    regions that never overlap in the frame - the normal case, since they are
    different places - keep their own label whichever plane they were drawn
    at. Where two regions do overlap, the higher label wins the shared
    pixels, and the flattened mask that is added to the viewer shows exactly
    which pixels each region ended up with.

    """
    if mask.ndim == 2:
        return mask
    flat = mask.max(axis=0)
    logger.info(f'Flattened a {mask.shape} labels volume onto one plane, '
                f'{len(np.unique(flat[flat > 0]))} regions, '
                f'{int((flat > 0).sum())} labelled pixels')
    return flat


class _TaskWidget(QWidget):
    """ Base class holding the progress bar, the status label and the
    background-task machinery shared by both dock widgets.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    """

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._thread = None
        self._worker = None
        self._result_callback = None

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = _StatusLabel()
        self.status.setVisible(False)

    def _busy(self) -> bool:
        """ Whether a background task is currently running. """
        return self._thread is not None

    def _start_task(self, func, on_result, description:str):
        """ Run `func` in a worker thread and hand its result to `on_result`.

        Parameters
        ----------
        func : callable
            Task callable, see `_Worker`.
        on_result : callable
            Called in the GUI thread with the return value of `func`.
        description : str
            Text shown in the status label until the first log record
            arrives.

        """
        if self._busy():
            logger.warning('Another task is still running')
            return

        self._result_callback = on_result
        self._thread = QThread(self)
        self._worker = _Worker(func)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.message.connect(self._on_message)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._deliver_result)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_task_finished)

        self.progress.setRange(0, 0)  # busy until the task reports a percentage
        self.progress.setVisible(True)
        self.status.setVisible(True)
        self._on_message(description)
        self._set_controls_enabled(False)
        self._thread.start()

    def _deliver_result(self, result):
        """ Hand the result of a finished task to its callback.

        The callback is always reached through this bound slot. A plain
        function or lambda connected to a signal has no thread affinity, so
        Qt would run it in the worker thread, and a napari layer must never
        be touched from there.

        Parameters
        ----------
        result : object
            Return value of the task callable.

        """
        self._result_callback(result)

    def _set_controls_enabled(self, enabled:bool):
        """ Enable or disable the controls that start a task.

        Overridden by subclasses; the base implementation does nothing.

        """

    def _on_message(self, text:str):
        """ Show the latest log record in the status label. """
        self.status.setMessage(text)

    def _on_progress(self, percent:int):
        """ Show `percent`, or return the bar to busy mode if it is negative.

        Parameters
        ----------
        percent : int
            Progress in percent, or a negative value for a step whose length
            is not known in advance.

        """
        if percent < 0:
            self.progress.setRange(0, 0)
            return
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 100)
        self.progress.setValue(percent)

    def _on_failed(self, message:str):
        """ Report a failed task in the status label. """
        self._on_message(f'Failed: {message}')

    def _on_task_finished(self):
        """ Release the worker thread and re-enable the controls. """
        self._thread.deleteLater()
        self._thread = None
        self._worker = None
        self.progress.setVisible(False)
        self._set_controls_enabled(True)

    def _image_layers(self) -> list:
        """ Image layers currently open in the viewer, oldest first. """
        return [layer for layer in self.viewer.layers if isinstance(layer, Image)]

    def _refill_layer_combo(self, combo:QComboBox, layers:list):
        """ Repopulate `combo` with the names of `layers`.

        Parameters
        ----------
        combo : PySide6.QtWidgets.QComboBox
            Combo box to refill; the current selection is kept if the layer
            is still there.
        layers : list of napari.layers.Layer
            Layers to offer, in the order they should appear.

        """
        names = [layer.name for layer in layers]
        if names == [combo.itemText(i) for i in range(combo.count())]:
            return
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if current in names:
            combo.setCurrentText(current)
        combo.blockSignals(False)


def _form(parent:QWidget=None) -> QFormLayout:
    """ Build a form layout that survives a narrow dock.

    The widgets are docked into a side panel the user resizes freely, so no
    row may impose a width of its own: long rows wrap the label above the
    field and the fields follow the width of the dock.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget, optional
        Widget the layout is installed on, by default None, which leaves the
        layout free to be nested into another one.

    Returns
    -------
    PySide6.QtWidgets.QFormLayout
        The configured layout.

    """
    layout = QFormLayout(parent) if parent is not None else QFormLayout()
    layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
    layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    return layout


def _spin(minimum, maximum, value, step=1, decimals:int=None) -> QWidget:
    """ Build a configured integer or floating point spin box.

    Parameters
    ----------
    minimum, maximum, value : int or float
        Range and initial value of the box.
    step : int or float, optional
        Single step of the box, by default 1.
    decimals : int, optional
        Number of decimals; None (default) builds a `QSpinBox`, any integer
        builds a `QDoubleSpinBox`.

    Returns
    -------
    PySide6.QtWidgets.QAbstractSpinBox
        The configured spin box.

    """
    box = QSpinBox() if decimals is None else QDoubleSpinBox()
    if decimals is not None:
        box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    return box


class PhaseModelReconWidget(_TaskWidget):
    """ Dock widget running `PhaseModelRecon` on a folder of phase images.

    The batch can arrive two ways: as a folder of phase images, or as an
    image layer already open in the viewer - a stack the user opened
    themselves, or the bundled sample of *File - Open Sample*. Both end up in
    `_ProgressRecon`, which reads frames from a path or from an array without
    the rest of the reconstruction knowing which.

    The widget collects the two things the reconstruction needs from the
    user - the ordered batch of phase images and the parameters of the class
    - runs `PhaseModelRecon.run` followed by
    `PhaseModelRecon.lambda_stack_recon` in a worker thread, and adds the
    assembled cube of shape ``(n_lambda, height, width)`` as a new image
    layer.

    Two more layers are optional and on by default, because they are what
    makes the result checkable: the input batch as it was read, crop
    included, and the modelled arcs of every frame as a labels layer, see
    `arc_label_stack`. Both have the shape of the frames the reconstruction
    read, so they overlay each other pixel for pixel.

    The third input, the simulated disk pattern, is not asked for: it is the
    one shipped with the plugin, see `bundled_pattern`.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    Attributes
    ----------
    recon : PhaseModelRecon or None
        The instance of the last successful run, kept so that its
        diagnostics stay reachable from the napari console.

    """

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.recon = None
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_stacks)
        self.viewer.layers.events.removed.connect(self._refresh_stacks)
        self._refresh_stacks()
        self._refresh_source()

    def _build_ui(self):
        """ Assemble the input, parameter and run sections of the widget. """
        layout = QVBoxLayout(self)

        # input batch
        input_box = QGroupBox('Input batch')
        input_form = _form(input_box)
        self.source_combo = QComboBox()
        self.source_combo.addItems(['folder of phase images', 'image layer'])
        self.source_combo.currentIndexChanged.connect(self._refresh_source)
        input_form.addRow('Source', self.source_combo)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText('folder with the phase images')
        folder_btn = QPushButton('Browse')
        folder_btn.clicked.connect(self._browse_folder)
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(folder_btn)
        self.folder_row = input_form.rowCount()
        input_form.addRow('Folder', folder_row)

        self.glob_edit = QLineEdit('*.tif*')
        self.glob_edit.editingFinished.connect(self._refresh_file_count)
        self.mask_row = input_form.rowCount()
        input_form.addRow('File mask', self.glob_edit)
        self.stack_combo = QComboBox()
        self.stack_combo.setToolTip('A 3D image layer whose planes are the phase images, '
                                    'in acquisition order')
        self.stack_combo.currentIndexChanged.connect(self._refresh_file_count)
        self.stack_row = input_form.rowCount()
        input_form.addRow('Layer', self.stack_combo)
        self.count_label = QLabel('no folder selected')
        input_form.addRow('Found', self.count_label)
        self.input_form = input_form
        pattern_label = QLabel(PATTERN_PATH.name)
        pattern_label.setToolTip('The simulated disk pattern bundled with the plugin, '
                                 'flipped and skeletonised on first use')
        input_form.addRow('Pattern', pattern_label)

        # crop
        self.crop_box = QGroupBox('Crop every frame')
        self.crop_box.setCheckable(True)
        self.crop_box.setChecked(False)
        crop_form = _form(self.crop_box)
        self.crop_spins = {name: _spin(0, 65535, default)
                           for name, default in (('row_min', 1000), ('row_max', 2000),
                                                 ('col_min', 1000), ('col_max', 2000))}
        for label, names in (('Rows', ('row_min', 'row_max')),
                             ('Columns', ('col_min', 'col_max'))):
            row = QHBoxLayout()
            for name in names:
                row.addWidget(self.crop_spins[name])
            crop_form.addRow(label, row)

        # main parameters
        main_box = QGroupBox('Reconstruction')
        main_form = _form(main_box)
        self.geometry_frames_spin = _spin(1, 64, 4)
        main_form.addRow('Geometry frames', self.geometry_frames_spin)
        self.n_lambda_spin = _spin(0, 4096, 0)
        self.n_lambda_spin.setSpecialValueText('auto')
        main_form.addRow('Lambda channels', self.n_lambda_spin)
        self.accumulation_combo = QComboBox()
        self.accumulation_combo.addItems(['max', 'mean', 'overwrite'])
        main_form.addRow('Accumulation', self.accumulation_combo)
        self.input_layer_check = QCheckBox('Also add the input batch as a stack')
        self.input_layer_check.setChecked(True)
        self.input_layer_check.setToolTip('The frames as the reconstruction read them, '
                                          'crop included, as one image layer')
        self.arcs_layer_check = QCheckBox('Also add the modelled arcs as labels')
        self.arcs_layer_check.setChecked(True)
        self.arcs_layer_check.setToolTip('One label per arc, the same value for the same arc '
                                         'in every frame, in the shape of the input frames')
        main_form.addRow(self.input_layer_check)
        main_form.addRow(self.arcs_layer_check)

        # advanced parameters live in a window of their own: sixteen more rows
        # would make the dock taller than most screens, and the values are set
        # once and then left alone
        self.advanced_dialog = QDialog(self)
        self.advanced_dialog.setWindowTitle('Advanced parameters')
        advanced_layout = QVBoxLayout(self.advanced_dialog)
        advanced_form = _form()
        advanced_layout.addLayout(advanced_form)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.advanced_dialog.accept)
        advanced_layout.addWidget(close_btn)
        advanced_btn = QPushButton('Advanced parameters...')
        advanced_btn.clicked.connect(self.advanced_dialog.exec)

        self.dilation_spin = _spin(0, 20, 1)
        self.init_sigma_spin = _spin(0.0, 50.0, 6.0, 0.5, decimals=1)
        self.fold_sigma_spin = _spin(0.0, 50.0, 4.0, 0.5, decimals=1)
        self.arc_tolerance_spin = _spin(0, 100, 6)
        self.family_degree_spin = _spin(1, 5, 2)
        self.phase_bins_spin = _spin(8, 1024, 128)
        self.refine_check = QCheckBox()
        self.refine_check.setChecked(True)
        self.scale_min_spin = _spin(0.1, 10.0, 1.0, 0.1, decimals=2)
        self.scale_max_spin = _spin(0.1, 10.0, 2.0, 0.1, decimals=2)
        self.shift_spin = _spin(0.0, 1.0, 0.2, 0.05, decimals=2)
        self.rotation_spin = _spin(0.0, 180.0, 30.0, 1.0, decimals=1)
        self.max_step_spin = _spin(0.0, 0.5, 0.25, 0.01, decimals=3)
        self.min_step_spin = _spin(0.0, 0.5, 0.0, 0.01, decimals=3)
        self.slack_spin = _spin(0.0, 0.5, 0.06, 0.01, decimals=3)
        self.penalty_spin = _spin(0.0, 10.0, 0.02, 0.01, decimals=3)
        self.drift_combo = QComboBox()
        self.drift_combo.addItems(['auto', '+1', '-1'])
        for label, widget in (('Pattern dilation', self.dilation_spin),
                              ('Init sigma', self.init_sigma_spin),
                              ('Fold sigma', self.fold_sigma_spin),
                              ('Arc edge tolerance', self.arc_tolerance_spin),
                              ('Family degree', self.family_degree_spin),
                              ('Phase bins', self.phase_bins_spin),
                              ('Refine geometry', self.refine_check),
                              ('Grid scale min', self.scale_min_spin),
                              ('Grid scale max', self.scale_max_spin),
                              ('Grid shift fraction', self.shift_spin),
                              ('Grid rotation, deg', self.rotation_spin),
                              ('Max phase step', self.max_step_spin),
                              ('Min phase step', self.min_step_spin),
                              ('Phase slack', self.slack_spin),
                              ('Step penalty', self.penalty_spin),
                              ('Drift direction', self.drift_combo)):
            advanced_form.addRow(label, widget)

        # run
        self.run_btn = QPushButton('Run reconstruction')
        self.run_btn.clicked.connect(self._run)

        for widget in (input_box, self.crop_box, main_box, advanced_btn,
                       self.run_btn, self.progress, self.status):
            layout.addWidget(widget)
        layout.addStretch(1)

    def _set_controls_enabled(self, enabled:bool):
        """ Enable or disable the run button while a task is running. """
        self.run_btn.setEnabled(enabled)

    def _from_layer(self) -> bool:
        """ Whether the batch is taken from a layer rather than from a folder. """
        return self.source_combo.currentIndex() == 1

    def _refresh_source(self):
        """ Show the rows of the selected input source only. """
        layer_mode = self._from_layer()
        for row, visible in ((self.folder_row, not layer_mode),
                             (self.mask_row, not layer_mode),
                             (self.stack_row, layer_mode)):
            self.input_form.setRowVisible(row, visible)
        # a batch that already is a layer has nothing to add back
        self.input_layer_check.setEnabled(not layer_mode)
        self._refresh_file_count()

    def _refresh_stacks(self, event=None):
        """ Keep the layer combo in sync with the open 3D image layers. """
        self._refill_layer_combo(self.stack_combo,
                                 [layer for layer in self._image_layers()
                                  if layer.data.ndim == 3])
        if self._from_layer():
            self._refresh_file_count()

    def _browse_folder(self):
        """ Ask for the folder holding the batch of phase images. """
        folder = QFileDialog.getExistingDirectory(self, 'Select the input folder')
        if folder:
            self.folder_edit.setText(folder)
            self._refresh_file_count()

    def _input_paths(self) -> list:
        """ Sorted paths of the input batch.

        Returns
        -------
        list of str
            Files of the selected folder matching the file mask, in name
            order, which for the reference data is the acquisition order the
            monotonicity constraint of `PhaseModelRecon` relies on.

        """
        folder = Path(self.folder_edit.text())
        if not folder.is_dir():
            return []
        return sorted(str(path) for path in folder.glob(self.glob_edit.text()))

    def _refresh_file_count(self):
        """ Report how many phase images the current source holds. """
        if self._from_layer():
            name = self.stack_combo.currentText()
            if name in self.viewer.layers:
                data = self.viewer.layers[name].data
                self.count_label.setText(f'{len(data)} planes of {data.shape[1:]}')
            else:
                self.count_label.setText('no 3D image layer')
            return
        paths = self._input_paths()
        self.count_label.setText(f'{len(paths)} images' if paths else 'no images found')

    def _recon_kwargs(self) -> dict:
        """ Collect every widget value into `PhaseModelRecon` keyword arguments. """
        crop = None
        if self.crop_box.isChecked():
            crop = [self.crop_spins[name].value()
                    for name in ('row_min', 'row_max', 'col_min', 'col_max')]
        drift = {'auto': None, '+1': 1, '-1': -1}[self.drift_combo.currentText()]
        return {'input_crop': crop,
                'input_pattern_dilation': self.dilation_spin.value(),
                'init_sigma': self.init_sigma_spin.value(),
                'fold_sigma': self.fold_sigma_spin.value(),
                'arc_edge_tolerance': self.arc_tolerance_spin.value(),
                'family_degree': self.family_degree_spin.value(),
                'phase_bins': self.phase_bins_spin.value(),
                'n_lambda': self.n_lambda_spin.value() or None,
                'geometry_frames': self.geometry_frames_spin.value(),
                'refine_geometry': self.refine_check.isChecked(),
                'grid_search_params': {'scale': [self.scale_min_spin.value(),
                                                 self.scale_max_spin.value()],
                                       'shift': self.shift_spin.value(),
                                       'rotation': self.rotation_spin.value()},
                'max_phase_step': self.max_step_spin.value(),
                'min_phase_step': self.min_step_spin.value(),
                'phase_slack': self.slack_spin.value(),
                'step_penalty': self.penalty_spin.value(),
                'drift_direction': drift}

    def _run(self):
        """ Validate the inputs and start the reconstruction task. """
        if self._from_layer():
            name = self.stack_combo.currentText()
            if name not in self.viewer.layers:
                self._on_message('Select a 3D image layer holding the phase images')
                self.status.setVisible(True)
                return
            stack = np.asarray(self.viewer.layers[name].data)
            # the frames are addressed by plane index instead of by path
            paths, batch = list(range(len(stack))), name
        else:
            stack = None
            paths = self._input_paths()
            batch = Path(self.folder_edit.text()).name
            if not paths:
                self._on_message('No input images: check the folder and the file mask')
                self.status.setVisible(True)
                return

        kwargs = self._recon_kwargs()
        method = self.accumulation_combo.currentText()
        # a batch that already is a layer has nothing to add back
        with_input = self.input_layer_check.isChecked() and stack is None
        with_arcs = self.arcs_layer_check.isChecked()
        logger.info(f'Starting reconstruction of {len(paths)} images of "{batch}" '
                    f'with {method} accumulation')

        def task(report):
            """ Reconstruct the batch and build every requested layer.

            The pattern is prepared here rather than in `_run`, because
            skeletonising it takes about a second and the GUI thread must
            not spend it.

            """
            recon_class = _progress_recon_class()
            recon = recon_class(input_path=paths, pattern_sim=bundled_pattern(),
                                report=report, stack=stack, **kwargs)
            recon.run()
            lambda_stack = recon.lambda_stack_recon(accumulation_method=method)
            report(recon_class.SAMPLING_END)
            # the frames are re-read rather than kept: the two passes of run
            # hold no frame in memory and this stack is optional
            frames = np.stack([recon._load_frame(path) for path in paths]) \
                if with_input else None
            report(98)
            arcs = arc_label_stack(recon) if with_arcs else None
            report(100)
            if frames is not None and arcs is not None and frames.shape != arcs.shape:
                raise ValueError(f'The arcs {arcs.shape} and the input batch '
                                 f'{frames.shape} came out of different frame shapes')
            return recon, lambda_stack, frames, arcs, batch

        self._start_task(task, self._add_result, f'Reconstructing {len(paths)} images...')

    def _add_result(self, result:tuple):
        """ Add the assembled layers to the viewer.

        The lambda stack goes in first, so that the optional input batch and
        the arcs end up above it and the arcs stay visible over the frames
        they were fitted to. Every name starts with the batch, so the layer
        list groups the results of one reconstruction together.

        Parameters
        ----------
        result : tuple
            The `PhaseModelRecon` instance, its lambda stack, the input batch,
            the arc labels and the name of the batch. The two middle entries
            are None when they were not asked for.

        """
        self.recon, lambda_stack, frames, arcs, batch = result
        added = [self.viewer.add_image(lambda_stack, name=layer_name(batch, 'lambda_stack'))]
        if frames is not None:
            added.append(self.viewer.add_image(frames, name=layer_name(batch, 'input_batch')))
        if arcs is not None:
            added.append(self.viewer.add_labels(arcs, name=layer_name(batch, 'arcs')))
        self._on_message(f'Done: lambda stack {lambda_stack.shape}')
        logger.info('Added ' + ', '.join(f'"{layer.name}" {layer.data.shape}'
                                         for layer in added))


class PostProcessingWidget(_TaskWidget):
    """ Dock widget applying the `utils` post-processing helpers to a layer.

    Both helpers are two-dimensional: `interpolate_zero_gaps` fills the row
    gaps a sparse reconstruction leaves, `max_pooling2d` trades vertical
    resolution for coverage. A 3D lambda stack is therefore processed one
    spectral channel at a time, which is also what keeps the memory of the
    interpolation bounded, and the per-channel loop drives the progress bar.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    """

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self._pending = None
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    def _build_ui(self):
        """ Assemble the layer selector, the parameter forms and the run button. """
        layout = QVBoxLayout(self)

        source_form = _form()
        self.layer_combo = QComboBox()
        source_form.addRow('Layer', self.layer_combo)
        self.method_combo = QComboBox()
        self.method_combo.addItems(['interpolate zero gaps', 'max pooling 2D'])
        self.method_combo.currentIndexChanged.connect(self._refresh_params)
        source_form.addRow('Method', self.method_combo)
        layout.addLayout(source_form)

        # interpolate_zero_gaps parameters
        self.interpolate_box = QGroupBox('Interpolation')
        interpolate_form = _form(self.interpolate_box)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(['0 (down the columns)', '1 (along the rows)'])
        interpolate_form.addRow('Axis', self.axis_combo)
        self.extrapolate_check = QCheckBox('Extrapolate beyond the measured range')
        interpolate_form.addRow(self.extrapolate_check)

        # max_pooling2d parameters
        self.pooling_box = QGroupBox('Pooling')
        pooling_form = _form(self.pooling_box)
        self.pool_h_spin = _spin(1, 128, 4)
        self.pool_w_spin = _spin(1, 128, 1)
        self.stride_h_spin = _spin(1, 128, 4)
        self.stride_w_spin = _spin(1, 128, 1)
        for label, widget in (('Pool height', self.pool_h_spin),
                              ('Pool width', self.pool_w_spin),
                              ('Stride rows', self.stride_h_spin),
                              ('Stride columns', self.stride_w_spin)):
            pooling_form.addRow(label, widget)

        self.apply_btn = QPushButton('Apply')
        self.apply_btn.clicked.connect(self._apply)

        for widget in (self.interpolate_box, self.pooling_box,
                       self.apply_btn, self.progress, self.status):
            layout.addWidget(widget)
        layout.addStretch(1)
        self._refresh_params()

    def _set_controls_enabled(self, enabled:bool):
        """ Enable or disable the apply button while a task is running. """
        self.apply_btn.setEnabled(enabled)

    def _refresh_layers(self, event=None):
        """ Keep the layer combo in sync with the open image layers. """
        self._refill_layer_combo(self.layer_combo, self._image_layers())

    def _refresh_params(self):
        """ Show the parameter group of the selected method only. """
        pooling = self.method_combo.currentIndex() == 1
        self.interpolate_box.setVisible(not pooling)
        self.pooling_box.setVisible(pooling)

    def _apply(self):
        """ Validate the selection and start the post-processing task. """
        name = self.layer_combo.currentText()
        if name not in self.viewer.layers:
            self._on_message('No image layer selected')
            self.status.setVisible(True)
            return
        image = np.asarray(self.viewer.layers[name].data)
        if image.ndim not in (2, 3):
            self._on_message(f'Only 2D images and 3D stacks are supported, got {image.shape}')
            self.status.setVisible(True)
            return

        from .utils import interpolate_zero_gaps, max_pooling2d

        if self.method_combo.currentIndex() == 0:
            axis = self.axis_combo.currentIndex()
            extrapolate = self.extrapolate_check.isChecked()
            plane = lambda img: interpolate_zero_gaps(img, axis=axis, extrapolate=extrapolate)
            suffix, scale = 'interpolated', (1, 1)
            logger.info(f'Interpolating zero gaps of "{name}" along axis {axis}')
        else:
            pool = (self.pool_h_spin.value(), self.pool_w_spin.value())
            stride = (self.stride_h_spin.value(), self.stride_w_spin.value())
            plane = lambda img: max_pooling2d(img, pool_size=pool, stride=stride)
            suffix, scale = 'pooled', stride
            logger.info(f'Max pooling "{name}" with window {pool} and stride {stride}')

        def task(report):
            """ Apply the plane operation to a 2D image or channel by channel. """
            if image.ndim == 2:
                return plane(image)
            planes = []
            for index, channel in enumerate(image):
                planes.append(plane(channel))
                report(int(100 * (index + 1) / len(image)))
            return np.stack(planes)

        self._pending = (name, suffix, scale)
        self._start_task(task, self._add_result, f'Processing {name}...')

    def _add_result(self, result:np.ndarray):
        """ Add the processed image to the viewer.

        The name of the source layer, the suffix of the new one and its
        pixel scale are taken from `_pending`, set when the task was
        started.

        Parameters
        ----------
        result : numpy.ndarray
            Processed image or stack.

        """
        source, suffix, scale = self._pending
        name = layer_name(source, suffix)
        self.viewer.add_image(result, name=name,
                              scale=scale if result.ndim == 2 else (1, *scale))
        self._on_message(f'Done: {name} {result.shape}')
        logger.info(f'Added layer "{name}" of shape {result.shape}')


class _PlotDialog(QDialog):
    """ Base class of the windows that carry a plot of their own.

    The dock is narrow and short, and a plot that is only looked at now and
    then has no claim on it. Every such plot lives in a window instead, built
    the first time it is asked for.

    Parameters
    ----------
    widget : SpectraWidget
        Owner of the data, consulted for spectra, colours and theme.
    title : str
        Window title.
    subplots : int, optional
        Number of side by side axes, by default 1.

    """

    def __init__(self, widget, title:str, subplots:int=1):
        from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
        from matplotlib.figure import Figure

        super().__init__(widget)
        self._widget = widget
        self.setWindowTitle(title)
        self.layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(5, 4), layout='tight')
        self.axes = self.figure.subplots(1, subplots, squeeze=False)[0]
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(160, 160)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.report = _StatusLabel()

    def finish_layout(self):
        """ Add the plot, the report and a close button, in that order. """
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.layout.setStretchFactor(self.canvas, 1)
        self.layout.addWidget(self.report)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        self.layout.addWidget(close_btn)

    def style(self):
        """ Paint every axes of this window in the theme colours. """
        background, text = self._widget.plot_colors()
        self.figure.set_facecolor(background)
        for axes in self.axes:
            axes.set_facecolor(background)
            for spine in axes.spines.values():
                spine.set_color(text)
            axes.tick_params(colors=text, labelsize='small')
        return background, text


class _PeakFitDialog(QDialog):
    """ Window showing the Gaussian decomposition of one ROI spectrum.

    Peak fitting answers a different question from the spectra themselves -
    not what a region emits, but which emitters it is made of - and it needs
    its own settings and its own axes. Drawing it over the spectra of every
    region hides exactly what has to be looked at: whether each component
    sits where an emitter is, and whether their sum follows the data.

    The dialog is also where the fit is configured, and `SpectraWidget`
    calibrates with the same settings, so the peaks used for the wavelength
    axis are the peaks seen here.

    Parameters
    ----------
    widget : SpectraWidget
        Owner of the spectra, consulted for the data and the ROI colours.

    """

    def __init__(self, widget):
        super().__init__(widget)
        from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self._widget = widget
        self.setWindowTitle('Peak fit with Gaussian')
        self.resize(560, 620)
        layout = QVBoxLayout(self)

        form = _form()
        self.roi_combo = QComboBox()
        self.roi_combo.currentIndexChanged.connect(self.refresh)
        form.addRow('Region', self.roi_combo)
        self.max_peaks_spin = _spin(0, 20, 0)
        self.max_peaks_spin.setSpecialValueText('reference peaks')
        self.max_peaks_spin.setToolTip('Number of components to fit. The default follows '
                                       'the reference peaks of the calibration, which is '
                                       'the single most effective setting for stability')
        form.addRow('Components', self.max_peaks_spin)
        self.window_spin = _spin(5, 201, FIT_DEFAULTS['window'], 2)
        self.window_spin.setToolTip('Savitzky-Golay window in channels: the smallest '
                                    'structure that survives the peak detection')
        form.addRow('Detection window', self.window_spin)
        self.poly_spin = _spin(1, 7, FIT_DEFAULTS['poly'])
        form.addRow('Detection polynomial', self.poly_spin)
        self.noise_spin = _spin(0.0, 1.0, FIT_DEFAULTS['noise_thresh'], 0.01, decimals=3)
        self.noise_spin.setToolTip('A candidate below this fraction of the maximum is '
                                   'rejected as baseline')
        form.addRow('Noise threshold', self.noise_spin)
        layout.addLayout(form)

        fit_btn = QPushButton('Fit')
        fit_btn.clicked.connect(self.refresh)
        layout.addWidget(fit_btn)

        self.figure = Figure(figsize=(5, 4), layout='tight')
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(160, 160)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        layout.setStretchFactor(self.canvas, 1)

        self.report = _StatusLabel()
        layout.addWidget(self.report)
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def fit_kwargs(self, reference_peaks:int) -> dict:
        """ Settings of this dialog as `fit_spectral_peaks` arguments.

        Parameters
        ----------
        reference_peaks : int
            Number of reference wavelengths, used when the component count is
            left at its default.

        Returns
        -------
        dict
            Keyword arguments for `utils.fit_spectral_peaks`.

        """
        return {'max_peaks': self.max_peaks_spin.value() or reference_peaks or None,
                'window': self.window_spin.value(),
                'poly': self.poly_spin.value(),
                'noise_thresh': self.noise_spin.value()}

    def open_on(self, index:int=0):
        """ Show the dialog, listing the regions that are currently extracted.

        Parameters
        ----------
        index : int, optional
            Region to select, by default 0.

        """
        labels = self._widget.roi_labels
        self.roi_combo.blockSignals(True)
        self.roi_combo.clear()
        self.roi_combo.addItems([f'ROI {label}' for label in labels])
        self.roi_combo.setCurrentIndex(min(index, len(labels) - 1))
        self.roi_combo.blockSignals(False)
        self.refresh()
        self.show()
        self.raise_()

    def refresh(self):
        """ Fit the selected region with the current settings and draw it. """
        index = self.roi_combo.currentIndex()
        if self._widget.spectra is None or index < 0:
            return
        from matplotlib import colors as mcolors
        from .utils import fit_spectral_peaks, gaussian_sum

        spectrum = self._widget.spectra[index]
        peaks_nm = self._widget.reference_peaks()
        fit = fit_spectral_peaks(spectrum, **self.fit_kwargs(len(peaks_nm) if
                                                             peaks_nm is not None else 0))

        background, text = self._widget.plot_colors()
        color = self._widget.roi_colors()[index]
        calibration = self._widget.calibration
        x = calibration['wavelength'] if calibration is not None \
            else np.arange(len(spectrum))
        # the fit runs on channel indices, so its centres are converted for
        # the annotation rather than the curve being resampled
        to_axis = calibration['to_nm'] if calibration is not None else (lambda v: v)

        self.figure.set_facecolor(background)
        self.axes.clear()
        self.axes.set_facecolor(background)
        for spine in self.axes.spines.values():
            spine.set_color(text)
        self.axes.tick_params(colors=text)

        self.axes.plot(x, spectrum, 'o', ms=2.5, color=color, label='data')
        if fit['n_peaks']:
            self.axes.plot(x, fit['fit'], '-', lw=1.4, color=color, label='sum of the fit')
            channels = np.arange(len(spectrum))
            components = fit['params'].reshape(-1, 3)
            styles = ('--', ':', '-.')
            for k, params in enumerate(components):
                # every component keeps the colour of its region and is told
                # apart by how far it is mixed towards the foreground
                mix = 0.15 + 0.5 * k / max(len(components) - 1, 1)
                shade = np.clip((1 - mix) * np.asarray(color)
                                + mix * np.asarray(mcolors.to_rgba(text)), 0.0, 1.0)
                self.axes.plot(x, gaussian_sum(channels, *params),
                               styles[k % len(styles)], lw=1.0, color=shade,
                               label=f'component {k + 1} at {to_axis(params[1]):.0f}')
        self.axes.set_xlabel('Wavelength, nm' if calibration is not None else 'Lambda index',
                             color=text)
        self.axes.set_ylabel('a.u.', color=text)
        legend = self.axes.legend(fontsize='small', facecolor=background, edgecolor=text)
        for entry in legend.get_texts():
            entry.set_color(text)
        self.canvas.draw_idle()

        unit = 'nm' if calibration is not None else 'ch'
        # a width is a difference, so it scales with the dispersion rather than
        # going through the polynomial that maps a position
        per_channel = calibration['dispersion'] if calibration is not None else 1.0
        centres = ', '.join(f'{to_axis(c):.1f}' for c in fit['center'])
        sigmas = ', '.join(f'{abs(s * per_channel):.1f}' for s in fit['sigma'])
        self.report.setMessage(
            f'{fit["n_peaks"]} components, converged {fit["success"]}\n'
            f'centres {centres} {unit}\n'
            f'widths {sigmas} {unit}\n'
            f'residual rms {fit["residual_rms"]:.1f} a.u.')


class _CalibrationFitDialog(_PlotDialog):
    """ Window showing the fit the wavelength axis rests on.

    Every point is one reference emitter, placed at the channel where it was
    actually found in a ROI spectrum against the wavelength it is known to
    emit at, in the colour of its region; the line is the fitted polynomial.
    Points on the line mean the peaks and the model agree, and a region whose
    points sit consistently to one side of it does not share the dispersion of
    the others - which is the first sign of the drift `_PeakDriftDialog`
    measures.

    Parameters
    ----------
    widget : SpectraWidget
        Owner of the calibration.

    """

    def __init__(self, widget):
        super().__init__(widget, 'Calibration fit')
        self.resize(560, 520)
        self.finish_layout()

    def refresh(self):
        """ Redraw the points and the fitted line. """
        widget = self._widget
        if widget.calibration is None or not widget._calibration_fits:
            return
        background, text = self.style()
        axes = self.axes[0]
        axes.clear()
        self.style()

        colors = widget.roi_colors()
        peaks_nm = widget._calibration_peaks_nm
        channels = np.arange(widget.spectra.shape[1])
        axes.plot(channels, widget.calibration['to_nm'](channels), '-', color=text,
                  lw=1.0, zorder=1, label=f'degree {widget.calibration["degree"]} fit')
        for index, fit in widget._calibration_fits:
            axes.scatter(fit['center'], peaks_nm, s=26, zorder=2, color=colors[index],
                         label=f'ROI {widget.roi_labels[index]}')
        axes.set_xlabel('Lambda index', color=text)
        axes.set_ylabel('Reference wavelength, nm', color=text)
        legend = axes.legend(fontsize='small', facecolor=background, edgecolor=text)
        for entry in legend.get_texts():
            entry.set_color(text)
        self.canvas.draw_idle()

        residual = widget.calibration['residual']
        self.report.setMessage(
            f'{widget.calibration["dispersion"]:.2f} nm per channel, '
            f'axis {widget.calibration["range"][0]:.0f}..'
            f'{widget.calibration["range"][1]:.0f} nm\n'
            f'residual rms {widget.calibration["rms"]:.1f} nm, '
            f'largest {np.abs(residual).max():.1f} nm, '
            f'from {len(widget._calibration_fits)} of {len(widget.fits)} ROIs')

    def open_now(self):
        """ Redraw and show the window. """
        self.refresh()
        self.show()
        self.raise_()


class _PeakDriftDialog(_PlotDialog):
    """ Window measuring where in the frame each emitter is found.

    A calibration fitted from a few regions is one line for the whole frame,
    and it holds only as long as a spectral channel means the same wavelength
    everywhere. This cuts the frame into tiles, **unmixes** the emitters in
    each of them with `utils.peak_drift_unmixed` and draws the position of
    every component along the two axes of the frame.

    Unmixing rather than peak-finding is the point. Following the highest
    point of a spectrum measures where the *mixture* peaks, not where any
    emitter sits, so a sample whose species are deposited unevenly reads as a
    drift of its own. On the reference batch that artefact is larger than the
    real effect and even reverses its sign: raw peak positions put two
    apparent modes on opposite slopes, while unmixing the same reconstruction
    puts all three components on the same one.

    All components are drawn together because the comparison between them is
    what settles the question. A real tilt of the spectral axis moves every
    component the **same way**; a mis-scaled spectral coordinate moves them in
    the ratio of their channel indices. Slopes that disagree in sign, or in
    that ratio, belong to the specimen and must not be corrected away.

    Nothing is written to the viewer. The measurement is a diagnostic on a
    coarse tile grid, and one layer per emitter carrying a few dozen tile
    values would sit in the layer list at the resolution of the reconstruction
    while saying nothing the plot does not say better.

    Parameters
    ----------
    widget : SpectraWidget
        Owner of the stack selection and of the calibration.

    """

    def __init__(self, widget):
        super().__init__(widget, 'Peak drift across the frame', subplots=2)
        self.resize(720, 560)
        self.unmixed = None
        self._labels_used = None

        form = _form()
        self.tile_spin = _spin(8, 512, 64, step=8)
        self.tile_spin.setToolTip('Side of the square binning tile. Larger tiles fit more '
                                  'reliably and resolve the drift more coarsely; the drift '
                                  'is a smooth gradient, so coarse is cheap')
        form.addRow('Tile size, px', self.tile_spin)
        self.min_snr_spin = _spin(0.0, 100.0, 4.0, step=0.5, decimals=1)
        self.min_snr_spin.setToolTip('Noise sigmas a tile spectrum must clear before it is '
                                     'fitted at all')
        form.addRow('Min SNR', self.min_snr_spin)
        self.layout.addLayout(form)

        self.measure_btn = QPushButton('Measure the drift')
        self.measure_btn.clicked.connect(self._measure)
        self.layout.addWidget(self.measure_btn)
        self.finish_layout()

    def open_now(self):
        """ Drop a measurement made for other emitters and show the window. """
        labels = self._labels()
        if labels != self._labels_used:
            self.unmixed = None
        self.show()
        self.raise_()

    def _labels(self) -> list:
        """ One name per emitter, in nanometres once the peaks are known. """
        peaks_nm = self._widget.reference_peaks()
        return ([f'{value:.0f} nm' for value in peaks_nm] if peaks_nm is not None
                else [f'peak {k + 1}' for k in range(3)])

    def _measure(self):
        """ Unmix every tile of the selected stack, in the worker thread. """
        widget = self._widget
        stack_name = widget.stack_combo.currentText()
        if stack_name not in widget.viewer.layers:
            self.report.setMessage('Select a lambda stack in the widget first')
            return
        cube = np.asarray(widget.viewer.layers[stack_name].data)
        peaks_nm = widget.reference_peaks()
        n_peaks = len(peaks_nm) if peaks_nm is not None else 3
        tile = self.tile_spin.value()
        min_snr = float(self.min_snr_spin.value())
        fit_kwargs = widget._fit_kwargs(n_peaks)
        fit_kwargs.pop('max_peaks', None)
        self._shape = cube.shape[1:]
        self._labels_used = self._labels()
        # a reconstruction is sparse: average a tile over the pixels it filled
        mask = cube.max(axis=0) > 0

        def task(report):
            """ Unmix the emitters tile by tile. """
            from .utils import peak_drift_unmixed
            return peak_drift_unmixed(cube, n_peaks=n_peaks, tile=tile, mask=mask,
                                      min_snr=min_snr, plot=False, **fit_kwargs)

        self.measure_btn.setEnabled(False)
        widget._start_task(task, self._show_result,
                           f'Unmixing {n_peaks} components over {tile}px tiles...')

    def _show_result(self, result:dict):
        """ Keep the tile fits and draw the profiles. """
        self.unmixed = result
        self.measure_btn.setEnabled(True)
        self._replot()

    def _positions(self) -> tuple:
        """ Component positions in the unit the axis is calibrated in.

        Returns
        -------
        numpy.ndarray
            ``(n_peaks, tiles_y, tiles_x)`` positions, in nanometres when the
            axis is calibrated and in channels otherwise.
        str
            Name of that unit.

        """
        centres = self.unmixed['center']
        if self._widget.calibration is not None:
            return self._widget.calibration['to_nm'](centres), 'nm'
        return centres, 'channels'

    def _replot(self):
        """ Draw every component along both frame axes, columns first.

        The column panel carries the fitted straight lines as well, because
        their slopes - and above all the ratios between them - are what say
        whether the drift belongs to the instrument or to the specimen.

        """
        if self.unmixed is None:
            return
        background, text = self.style()
        position, unit = self._positions()
        drift = position - np.nanmedian(position.reshape(len(position), -1),
                                        axis=1)[:, None, None]
        tile = self.unmixed['tile']
        height, width = self._shape
        labels = self._labels_used or self._labels()

        for axes in self.axes:
            axes.clear()
        self.style()
        col_centres = (np.arange(drift.shape[2]) + 0.5) * tile
        row_centres = (np.arange(drift.shape[1]) + 0.5) * tile

        for k in range(len(drift)):
            label = labels[k] if k < len(labels) else f'peak {k + 1}'
            line, = self.axes[0].plot(col_centres, np.nanmedian(drift[k], axis=0),
                                      'o-', lw=1.2, ms=3, label=label)
            good = np.isfinite(np.nanmedian(drift[k], axis=0))
            if good.sum() > 2:
                fit = np.polyfit(col_centres[good],
                                 np.nanmedian(drift[k], axis=0)[good], 1)
                self.axes[0].plot(col_centres, np.polyval(fit, col_centres),
                                  color=line.get_color(), lw=1.0, ls='--', alpha=.7)
            self.axes[1].plot(row_centres, np.nanmedian(drift[k], axis=1),
                              'o-', lw=1.2, ms=3, color=line.get_color(), label=label)

        for axes, label in zip(self.axes, ('Frame column, px', 'Frame row, px')):
            axes.axhline(0.0, color=text, lw=0.8, ls='--', alpha=.5)
            axes.set_xlabel(label, color=text)
        self.axes[0].set_ylabel(f'Component drift, {unit}', color=text)
        self.axes[0].legend(loc='best', fontsize=8)
        self.canvas.draw_idle()
        self.report.setMessage(self._verdict(unit))

    def _verdict(self, unit:str) -> str:
        """ The slope comparison, in words.

        A tilt of the spectral axis moves every component the same way; an
        error in the normalised spectral coordinate moves them in the ratio of
        their channel indices. The ratios are always taken in channels, where
        that prediction is defined, whatever unit the plot is drawn in.

        """
        slope = self.unmixed['slope'] * 1000.0            # channels / 1000 columns
        reference = self.unmixed['reference']
        measured = slope / slope[0] if slope[0] else np.full(len(slope), np.nan)
        expected = reference / reference[0] if reference[0] else np.full(len(slope), np.nan)
        same_way = bool(np.all(np.sign(slope) == np.sign(slope[0])))
        rigid = bool(np.nanmax(np.abs(measured - 1.0)) < 0.5) if same_way else False

        if not same_way:
            reading = ('components drift in opposite directions - this is the specimen artifact')
        elif rigid:
            reading = ('one direction, ratios near 1 - a rigid tilt of the spectral axis')
        else:
            reading = ('one direction but ratios follow the channel indices - a mis-scaled spectral coordinates')
        span = self.unmixed['span']
        return (f'{int(self.unmixed["success"].sum())} tiles fitted '
                f'({100 * self.unmixed["coverage"]:.0f}%), {self.unmixed["tile"]}px each\n'
                f'column slopes ' + ', '.join(f'{value:+.2f}' for value in slope) +
                ' channels per 1000 columns\n'
                f'spans ' + ', '.join(f'{value:+.2f}' for value in span) + ' channels\n'
                f'ratios ' + ', '.join(f'{value:.2f}' for value in measured) +
                ' against ' + ', '.join(f'{value:.2f}' for value in expected) +
                ' if the coordinate were mis-scaled\n' + reading)


class SpectraWidget(_TaskWidget):
    """ Dock widget for ROI spectra, wavelength calibration and CSV export.

    This is where a reconstructed cube becomes a measurement. Regions of
    interest are taken from a labels layer - drawn with the napari brush or
    loaded as a mask - and every region gives one mean spectrum, exactly the
    plain mean over the region that the reference workflow uses.

    The labels layer napari creates over a lambda stack has the shape of that
    stack, one plane per spectral channel, and regions drawn at different
    channels therefore land on different planes. A spectrum, however, is a
    property of a place in the frame and not of the channel the region
    happened to be drawn at, so a three-dimensional mask is flattened onto
    one plane before anything is measured, see `flatten_roi_mask`. The
    flattened mask is added to the viewer as a layer of its own: it is the
    mask the numbers actually came from, and it can be reused, edited and
    saved.

    The spectral axis of a reconstruction is uncalibrated: channel `k` is a
    fixed fraction of a band, not a wavelength. Given the known emission
    peaks of the sample, `fit_spectral_peaks` locates those peaks in every
    ROI spectrum and `spectral_calibration` fits channel index against known
    wavelength, which turns the axis into nanometres and gives the
    dispersion in nm per channel.

    Only the ROIs whose fit returns exactly as many components as there are
    reference wavelengths can enter the calibration: with a different number
    there is no way to say which fitted peak belongs to which emitter.
    Feeding all of them at once is deliberate - repeated measurements of the
    same emitters are what give the fit the degrees of freedom its residuals
    need to mean anything.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    Attributes
    ----------
    roi_labels : numpy.ndarray or None
        Label values of the extracted regions, in the order of the rows of
        `spectra`.
    spectra : numpy.ndarray or None
        Mean spectra of shape ``(n_roi, n_lambda)``.
    fits : list of dict or None
        Per-ROI output of `fit_spectral_peaks`, one entry per row of
        `spectra`.
    calibration : dict or None
        Output of `spectral_calibration`, or None while the axis is
        uncalibrated.

    """

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.roi_labels = None
        self.spectra = None
        self.fits = None
        self.calibration = None
        self._calibration_fits = []
        self._calibration_peaks_nm = None
        self._roi_layer_name = None
        self._batch = ''
        self._pending_mask_name = None
        # every plot window is built on first use: each carries a canvas
        self.fit_dialog = None
        self.calibration_dialog = None
        self.drift_dialog = None
        self._plot_colors = ('white', 'black')
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.events.theme.connect(self._apply_theme)
        self._refresh_layers()

    def _build_ui(self):
        """ Assemble the source, calibration, plot and export sections. """
        from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
        from matplotlib.figure import Figure

        layout = QVBoxLayout(self)
        # the plot takes every pixel the dock can spare, so the widget must
        # be allowed to grow: with the default policy napari would pad it
        # with a stretch instead
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # source layers
        source_box = QGroupBox('Source')
        source_form = _form(source_box)
        self.stack_combo = QComboBox()
        source_form.addRow('Lambda stack', self.stack_combo)
        self.roi_combo = QComboBox()
        self.roi_combo.setToolTip('Labels layer holding the regions of interest, '
                                  'one label value per region')
        source_form.addRow('ROI labels', self.roi_combo)
        self.extract_btn = QPushButton('Extract ROI spectra')
        self.extract_btn.clicked.connect(self._extract)
        source_form.addRow(self.extract_btn)

        # calibration
        calibration_box = QGroupBox('Wavelength calibration')
        calibration_form = _form(calibration_box)
        self.peaks_edit = QLineEdit('525, 585, 659')
        self.peaks_edit.setToolTip('Known emission peaks of the sample in nanometres, '
                                   'comma separated')
        calibration_form.addRow('Reference peaks, nm', self.peaks_edit)
        self.degree_spin = _spin(1, 3, 1)
        calibration_form.addRow('Polynomial degree', self.degree_spin)
        self.calibrate_btn = QPushButton('Calibrate from ROI peaks')
        self.calibrate_btn.clicked.connect(self._calibrate)
        self.calibrate_btn.setEnabled(False)
        calibration_form.addRow(self.calibrate_btn)
        self.calibration_label = _StatusLabel()
        self.calibration_label.setMessage('not calibrated')
        calibration_form.addRow(self.calibration_label)
        self.calibration_fit_btn = QPushButton('Calibration fit...')
        self.calibration_fit_btn.setToolTip('The reference peaks against the channels they '
                                            'were found at, and the fitted line')
        self.calibration_fit_btn.clicked.connect(self._open_calibration_dialog)
        self.calibration_fit_btn.setEnabled(False)
        self.drift_btn = QPushButton('Peak drift...')
        self.drift_btn.setToolTip('Where in the frame each emitter is found, tile by tile')
        self.drift_btn.clicked.connect(self._open_drift_dialog)
        self.drift_btn.setEnabled(False)
        plots_row = QHBoxLayout()
        plots_row.addWidget(self.calibration_fit_btn)
        plots_row.addWidget(self.drift_btn)
        calibration_form.addRow(plots_row)

        # plot
        plot_box = QGroupBox('Spectra')
        plot_layout = QVBoxLayout(plot_box)
        self.fit_btn = QPushButton('Peak fit with Gaussian...')
        self.fit_btn.setToolTip('Fit a sum of Gaussians to one region and show the '
                                'components in a window of its own')
        self.fit_btn.clicked.connect(self._open_fit_dialog)
        self.fit_btn.setEnabled(False)
        self.figure = Figure(figsize=(4, 3), layout='tight')
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(120, 120)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        plot_layout.addWidget(self.fit_btn)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)

        self.save_btn = QPushButton('Save spectra to CSV')
        self.save_btn.clicked.connect(self._save_csv)
        self.save_btn.setEnabled(False)

        for widget in (source_box, calibration_box, plot_box,
                       self.save_btn, self.progress, self.status):
            layout.addWidget(widget)
        layout.setStretchFactor(plot_box, 1)  # the forms keep their height, the plot grows
        self._apply_theme()

    def _apply_theme(self, event=None):
        """ Paint the figure in the colours of the current napari theme.

        A white plot in a dark dock is the one thing that gives an embedded
        matplotlib canvas away, and the viewer already knows which colours
        to use.

        """
        try:
            theme = get_theme(self.viewer.theme)
            self._plot_colors = (theme.background.as_hex(), theme.text.as_hex())
        except Exception:  # an unknown theme must not cost the whole widget
            logger.warning(f'Unknown napari theme "{self.viewer.theme}", '
                           f'the plot keeps its previous colours')
        self._style_axes(self.figure, self.axes)
        self._replot()
        for dialog in (self.calibration_dialog, self.drift_dialog):
            if dialog is not None and dialog.isVisible():
                dialog.refresh() if dialog is self.calibration_dialog else dialog._replot()

    def _style_axes(self, figure, axes):
        """ Paint one figure and its axes in the current theme colours.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to paint.
        axes : matplotlib.axes.Axes
            Its axes, repainted after every `clear`.

        """
        background, text = self._plot_colors
        figure.set_facecolor(background)
        axes.set_facecolor(background)
        for spine in axes.spines.values():
            spine.set_color(text)
        axes.tick_params(colors=text, labelsize='small')
        axes.xaxis.label.set_color(text)
        axes.yaxis.label.set_color(text)

    def _set_controls_enabled(self, enabled:bool):
        """ Enable or disable the extract button while a task is running. """
        self.extract_btn.setEnabled(enabled)

    def _roi_layers(self) -> list:
        """ Layers usable as a ROI mask.

        Returns
        -------
        list of napari.layers.Layer
            Labels layers of two or three dimensions - three is the shape
            napari gives a labels layer created over a lambda stack, and it
            is flattened when the spectra are extracted - plus two
            dimensional integer image layers, which is what a mask read from
            a TIFF looks like until it is converted. An integer image of
            three dimensions is not offered: that is a stack, not a mask.

        """
        return [layer for layer in self.viewer.layers
                if (isinstance(layer, Labels) and layer.data.ndim in (2, 3))
                or (isinstance(layer, Image) and layer.data.ndim == 2
                    and np.issubdtype(layer.data.dtype, np.integer))]

    def _refresh_layers(self, event=None):
        """ Keep both source combos in sync with the open layers. """
        self._refill_layer_combo(self.stack_combo,
                                 [layer for layer in self._image_layers()
                                  if layer.data.ndim == 3])
        self._refill_layer_combo(self.roi_combo, self._roi_layers())
        self.drift_btn.setEnabled(self.stack_combo.count() > 0)

    def _extract(self):
        """ Validate the selection and start the spectra extraction task. """
        stack_name, roi_name = self.stack_combo.currentText(), self.roi_combo.currentText()
        if stack_name not in self.viewer.layers or roi_name not in self.viewer.layers:
            self._on_message('Select a 3D lambda stack and a ROI labels layer')
            self.status.setVisible(True)
            return

        cube = np.asarray(self.viewer.layers[stack_name].data)
        mask = np.asarray(self.viewer.layers[roi_name].data)
        # only the frame of the mask has to match, a labels volume may carry
        # any number of planes
        if cube.shape[1:] != mask.shape[-2:]:
            self._on_message(f'Shape mismatch: stack frames are {cube.shape[1:]}, '
                             f'the ROI mask is {mask.shape[-2:]}')
            self.status.setVisible(True)
            return
        if not mask.any():
            self._on_message(f'Layer "{roi_name}" holds no labelled region')
            self.status.setVisible(True)
            return
        logger.info(f'Extracting ROI spectra of {len(cube)} channels from "{stack_name}" '
                    f'with the regions of "{roi_name}" {mask.shape}')
        self._roi_layer_name = roi_name
        self._batch = batch_prefix(stack_name)
        self._pending_mask_name = (layer_name(self._batch, roi_name, '2D')
                                   if mask.ndim == 3 else None)

        def task(report):
            """ Flatten the mask, then average every region channel by channel.

            The regions are addressed by the flat indices of their pixels
            rather than by a boolean mask over the whole frame: the mean is
            then over the pixels of the region instead of over the frame, so
            the cost follows the size of the regions and not the size of the
            cube. The result is the plain mean of the reference workflow.

            """
            flat_mask = flatten_roi_mask(mask)
            labels = np.unique(flat_mask[flat_mask > 0])
            flat_cube = cube.reshape(len(cube), -1)
            flat_index = flat_mask.ravel()
            spectra = np.zeros((len(labels), len(cube)))
            for i, label in enumerate(labels):
                index = np.flatnonzero(flat_index == label)
                spectra[i] = [flat_cube[k, index].mean() for k in range(len(cube))]
                report(int(100 * (i + 1) / len(labels)))
            return labels, spectra, flat_mask

        self._start_task(task, self._show_spectra, f'Extracting the spectra of "{roi_name}"...')

    def _show_spectra(self, result:tuple):
        """ Store the extracted spectra, draw them and publish the flat mask.

        Parameters
        ----------
        result : tuple
            Label values, the ``(n_roi, n_lambda)`` array of mean spectra and
            the two-dimensional mask they were measured on. The mask becomes
            a layer of its own whenever it had to be flattened, so that what
            the numbers came from is visible and reusable.

        """
        self.roi_labels, self.spectra, flat_mask = result
        if self._pending_mask_name is not None:
            # a labels layer inherits the scale and the offset of the layer it
            # was created over - a stack that came out of pooling carries a
            # scale of its own - and the flattened mask has to keep them, or
            # it lands somewhere else in the viewer than the regions it holds
            source = self.viewer.layers[self._roi_layer_name]
            layer = self.viewer.add_labels(flat_mask, name=self._pending_mask_name,
                                           scale=tuple(source.scale[-2:]),
                                           translate=tuple(source.translate[-2:]))
            logger.info(f'Added layer "{layer.name}" of shape {layer.data.shape}, '
                        f'scale {tuple(layer.scale)}, translate {tuple(layer.translate)}')
            self._roi_layer_name = layer.name
        self.fits, self.calibration = None, None
        self._calibration_fits = []
        self.calibration_label.setMessage('not calibrated')
        self.calibration_fit_btn.setEnabled(False)
        self.calibrate_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.fit_btn.setEnabled(True)
        self._replot()
        if self.fit_dialog is not None and self.fit_dialog.isVisible():
            self.fit_dialog.open_on(self.fit_dialog.roi_combo.currentIndex())
        self._on_message(f'Done: {len(self.spectra)} ROI spectra '
                         f'of {self.spectra.shape[1]} channels')
        logger.info(f'Extracted spectra of ROIs {self.roi_labels.tolist()}')

    def _reference_peaks(self) -> np.ndarray:
        """ Reference wavelengths parsed from the input field.

        Returns
        -------
        numpy.ndarray or None
            Wavelengths in nanometres, or None if the field cannot be parsed.

        """
        try:
            return np.array([float(value) for value in
                             self.peaks_edit.text().replace(';', ',').split(',') if value.strip()])
        except ValueError:
            return None

    def _calibrate(self):
        """ Fit the emission peaks of every ROI and calibrate the axis.

        The fit and the calibration are fast enough to run in the GUI thread:
        a few Gaussians per ROI over some tens of channels.

        """
        peaks_nm = self._reference_peaks()
        if peaks_nm is None or len(peaks_nm) < 2:
            self._on_message('Give at least two reference wavelengths, comma separated')
            self.status.setVisible(True)
            return
        if len(peaks_nm) < self.degree_spin.value() + 1:
            self._on_message(f'A degree {self.degree_spin.value()} fit needs at least '
                             f'{self.degree_spin.value() + 1} reference peaks')
            self.status.setVisible(True)
            return

        from .utils import fit_spectral_peaks, spectral_calibration

        fit_kwargs = self._fit_kwargs(len(peaks_nm))
        self.fits = [fit_spectral_peaks(spectrum, **fit_kwargs) for spectrum in self.spectra]
        # only a ROI with one fitted component per reference emitter can be
        # paired with the known wavelengths, the rest would calibrate noise
        usable = [(index, fit) for index, fit in enumerate(self.fits)
                  if fit['n_peaks'] == len(peaks_nm)]
        self._calibration_fits = usable
        if not usable:
            self.calibration = None
            self.calibration_label.setMessage(f'No ROI gave {len(peaks_nm)} peaks, '
                                              f'calibration skipped')
            self.calibration_fit_btn.setEnabled(False)
            self._replot()
            return

        centres = np.concatenate([fit['center'] for index, fit in usable])
        self._calibration_peaks_nm = peaks_nm
        self.calibration = spectral_calibration(peak_index=centres,
                                                peak_wavelength=np.tile(peaks_nm, len(usable)),
                                                n_lambda=self.spectra.shape[1],
                                                degree=self.degree_spin.value())
        low, high = self.calibration['range']
        self.calibration_label.setMessage(
            f'{self.calibration["dispersion"]:.2f} nm per channel\n'
            f'axis {low:.0f}..{high:.0f} nm\n'
            f'residual rms {self.calibration["rms"]:.1f} nm '
            f'from {len(usable)} of {len(self.fits)} ROIs')
        self.calibration_fit_btn.setEnabled(True)
        self._replot()
        if self.calibration_dialog is not None and self.calibration_dialog.isVisible():
            self.calibration_dialog.refresh()
        if self.drift_dialog is not None and self.drift_dialog.isVisible():
            self.drift_dialog._replot()
        if self.fit_dialog is not None and self.fit_dialog.isVisible():
            self.fit_dialog.refresh()

    def _open_fit_dialog(self):
        """ Build the fit window on first use and show it. """
        if self.fit_dialog is None:
            self.fit_dialog = _PeakFitDialog(self)
        self.fit_dialog.open_on(0)

    def _open_calibration_dialog(self):
        """ Build the calibration fit window on first use and show it. """
        if self.calibration_dialog is None:
            self.calibration_dialog = _CalibrationFitDialog(self)
        self.calibration_dialog.open_now()

    def _open_drift_dialog(self):
        """ Build the peak drift window on first use and show it. """
        if self.drift_dialog is None:
            self.drift_dialog = _PeakDriftDialog(self)
        self.drift_dialog.open_now()

    def _fit_kwargs(self, reference_peaks:int) -> dict:
        """ Peak fit settings, from the fit window if it was ever opened.

        Parameters
        ----------
        reference_peaks : int
            Number of reference wavelengths, the default component count.

        Returns
        -------
        dict
            Keyword arguments for `utils.fit_spectral_peaks`.

        """
        if self.fit_dialog is not None:
            return self.fit_dialog.fit_kwargs(reference_peaks)
        return {'max_peaks': reference_peaks or None, **FIT_DEFAULTS}

    def plot_colors(self) -> tuple:
        """ Background and foreground colours of the current napari theme. """
        return self._plot_colors

    def reference_peaks(self) -> np.ndarray:
        """ Reference wavelengths of the calibration field, see `_reference_peaks`. """
        return self._reference_peaks()

    def roi_colors(self) -> np.ndarray:
        """ Colour of every extracted region, as napari draws it.

        A spectrum and the region it came from are the same thing seen twice,
        so they carry the same colour: the curve of ROI 3 is the colour of
        label 3 in the viewer.

        Returns
        -------
        numpy.ndarray
            RGBA rows, one per entry of `roi_labels`. Falls back to the
            matplotlib cycle if the labels layer is gone or cannot be asked.

        """
        try:
            layer = self.viewer.layers[self._roi_layer_name]
            return np.asarray(layer.colormap.map(np.asarray(self.roi_labels)))
        except Exception:
            from matplotlib import colors as mcolors, rcParams

            cycle = rcParams['axes.prop_cycle'].by_key()['color']
            return np.asarray([mcolors.to_rgba(cycle[i % len(cycle)])
                               for i in range(len(self.roi_labels))])

    def _replot(self):
        """ Redraw the spectra against the channel index or the calibrated axis. """
        background, text = self._plot_colors
        self.axes.clear()
        self.axes.set_facecolor(background)
        for spine in self.axes.spines.values():
            spine.set_color(text)
        self.axes.tick_params(colors=text)
        if self.spectra is None:
            self.axes.set_xlabel('Lambda index', color=text)
            self.axes.set_ylabel('a.u.', color=text)
            self.canvas.draw_idle()
            return

        calibrated = self.calibration is not None
        x = self.calibration['wavelength'] if calibrated else np.arange(self.spectra.shape[1])
        for label, spectrum, color in zip(self.roi_labels, self.spectra, self.roi_colors()):
            self.axes.plot(x, spectrum, lw=1, color=color, label=f'ROI {label}')

        self.axes.set_xlabel('Wavelength, nm' if calibrated else 'Lambda index',
                             color=text)
        self.axes.set_ylabel('a.u.', color=text)
        if len(self.spectra) <= 10:
            legend = self.axes.legend(fontsize='small', facecolor=background,
                                      edgecolor=text)
            for entry in legend.get_texts():
                entry.set_color(text)
        self.canvas.draw_idle()

    def _save_csv(self):
        """ Write the ROI spectra to a CSV file.

        The file holds one row per spectral channel: the raw channel index,
        always, then the calibrated wavelength if the axis was calibrated,
        then one column of mean intensity per region of interest.

        The index column is written as an integer and the wavelength to one
        decimal - the calibration is good to about a nanometre, so further
        digits would claim a precision the fit does not have. The intensities
        keep their full precision.

        """
        default = layer_name(self._batch, self._roi_layer_name or 'roi',
                             'spectra') + '.csv'
        path, _ = QFileDialog.getSaveFileName(self, 'Save the ROI spectra',
                                              default, 'CSV files (*.csv)')
        if not path:
            return

        columns = [np.arange(self.spectra.shape[1], dtype=float)]
        header, fmt = ['lambda_index'], ['%d']
        if self.calibration is not None:
            columns.append(self.calibration['wavelength'])
            header.append('wavelength_nm')
            fmt.append('%.1f')
        columns.extend(self.spectra)
        header.extend(f'roi_{label}' for label in self.roi_labels)
        fmt.extend(['%.6g'] * len(self.spectra))

        np.savetxt(path, np.column_stack(columns), delimiter=',',
                   header=','.join(header), comments='', fmt=fmt)
        self._on_message(f'Saved {path}')
        self.status.setVisible(True)
        logger.info(f'Saved {len(self.spectra)} ROI spectra to {path}')
