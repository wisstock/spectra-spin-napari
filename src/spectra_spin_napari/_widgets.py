""" Qt widgets of the `spectra-spin-napari` plugin.

Six dock widgets are contributed to napari:

* `PhaseModelReconWidget` - collects the input batch and the parameters of
  `PhaseModelRecon`, runs the reconstruction against the bundled simulated
  disk pattern and adds the assembled lambda stack, optionally together with
  the input batch and the modelled arcs, as new layers.
* `PostProcessingWidget` - applies the post-processing helpers of `utils`
  (`interpolate_zero_gaps`, `max_pooling2d`) to an existing image layer.
* `SpectraWidget` - mean spectra of the regions of interest of a labels
  layer, the wavelength calibration fitted from their emission peaks, and
  the export of the spectra to CSV. A calibration is passed on to the
  widgets below through `calibration_hub`.
* `LambdaRGBWidget` - renders a spectral stack as true-colour sRGB.
* `VirtualChannelsWidget` - sums a spectral stack over up to four
  wavelength ranges into as many channels.
* `AdaptiveDyeSeparationWidget` - separates the dyes of a spectral stack
  without knowing their spectra.

Every dock opens as a tab beside the docks already on its side.

Every widget runs its work in a `QThread`, so the napari window stays
responsive, and mirrors the `logging` output of the processing modules into a
status label next to the progress bar. The bar is driven by the work itself:
the loops of this module count their own iterations, and the reconstruction
reports through its `progress` callback. The single step whose length cannot
be known in advance - the Nelder-Mead refinement of the shared geometry - is
the only one that puts the bar into busy mode.

"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
from napari.layers import Image, Labels, Shapes
from napari.utils.theme import get_theme

from PySide6.QtCore import QEvent, QObject, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QGroupBox, QHBoxLayout,
                               QLabel, QMainWindow, QScrollArea,
                               QLineEdit, QProgressBar, QPushButton, QSizePolicy,
                               QSlider, QSpinBox, QVBoxLayout, QWidget)

logger = logging.getLogger(__name__)

# the processing modules and matplotlib are imported where they are used, so
# that opening a widget does not load scipy, scikit-image and pyplot

SETUP_PATH = Path(__file__).parent / 'setup_params.yaml'
# every key `setup_params.yaml` must hold, by section
SETUP_KEYS = {'disk': ('pattern_file', 'pattern_rotate_180', 'pass_width', 'pass_pulse_us',
                       'drift_direction'),
              'optics': ('invert_spectral_axis', 'n_lambda', 'spectral_fwhm')}
# the smoothing and the edge tolerance of the reconstruction widget are lengths
# on the frame, so their defaults follow the line period ``optics.n_lambda``:
# these are the `PhaseModelRecon` defaults, tuned at REFERENCE_PERIOD rows. Only
# the widget scales them; the class keeps its own fixed defaults
REFERENCE_PERIOD = 89
PERIOD_SCALED = {'init_sigma': 6.0, 'fold_sigma': 4.0, 'arc_edge_tolerance': 6}
SAMPLE_PATH = Path(__file__).parent / 'qd_mix_batch.tif'
SAMPLE_ROIS_PATH = Path(__file__).parent / 'qd_mix_rois.tiff'


# starting point of the peak detection, shared by the fit window and by the
# calibration, which fits the same way before the window is ever opened
FIT_DEFAULTS = {'window': 15, 'poly': 3, 'noise_thresh': 0.02}


def layer_name(*parts) -> str:
    """ Join name parts into a layer name that is also a safe file name.

    Parts are joined by underscores, whitespace inside a part becomes an
    underscore and words of punctuation only are dropped.

    Parameters
    ----------
    *parts : str
        Name parts, in order; empty parts are skipped.

    Returns
    -------
    str
        The joined name.

    Examples
    --------
    >>> layer_name('lambda stack - Labels', '2D')
    'lambda_stack_Labels_2D'

    """
    words = [word for part in parts for word in str(part).split()
             if any(character.isalnum() for character in word)]
    return '_'.join(words)


# suffixes this plugin appends to a layer name, longest first, so that the
# batch a layer came from can be recovered from the name of any of them
LAYER_SUFFIXES = ('_lambda_stack', '_input_batch', '_arcs',
                  '_interpolated', '_pooled', '_rgb', '_virtual_channels',
                  '_first_phase', '_crop', '_ads_clusters', '_ads')


def batch_prefix(name:str) -> str:
    """ Batch name at the front of a plugin layer name: the name without
    the suffixes of `LAYER_SUFFIXES`.

    Parameters
    ----------
    name : str
        Layer name.

    Returns
    -------
    str
        The prefix; a foreign layer name is returned unchanged.

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

    18 phase images of a mix of quantum dots at 525, 585 and 659 nm, the crop
    ``[500, 2500, 1000, 2500]`` of the raw camera frames, as one image layer.

    Returns
    -------
    list
        One napari LayerData tuple, ``(data, meta, 'image')``.

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
def setup_params() -> dict:
    """ Hardware-specific defaults of the widgets, from `setup_params.yaml`.

    Only the widgets read the file; `PhaseModelRecon` takes every value as an
    argument.

    Returns
    -------
    dict
        Sections 'disk' and 'optics', see `SETUP_KEYS`; cached for the session.

    Raises
    ------
    ValueError
        If a key of `SETUP_KEYS` is missing.

    """
    import yaml

    with open(SETUP_PATH, encoding='utf-8') as file:
        params = yaml.safe_load(file) or {}
    missing = [f'{section}.{key}' for section, keys in SETUP_KEYS.items()
               for key in keys if key not in (params.get(section) or {})]
    if missing:
        raise ValueError(f'{SETUP_PATH.name} misses {", ".join(missing)}')
    logger.info(f'Setup parameters from {SETUP_PATH}')
    return params


@lru_cache(maxsize=1)
def bundled_pattern() -> np.ndarray:
    """ Simulated disk pattern of the setup, ready for `PhaseModelRecon`.

    The file ``disk.pattern_file`` of `setup_params.yaml`, rotated by 180
    degrees when ``disk.pattern_rotate_180`` is set and skeletonised, since the
    line family is fitted to one-pixel arcs. See ``docs/design_notes.md``.

    Returns
    -------
    numpy.ndarray
        2D boolean skeleton, cached for the session.

    """
    from skimage import io, morphology

    disk = setup_params()['disk']
    path = SETUP_PATH.parent / disk['pattern_file']
    raw = io.imread(path)
    pattern = morphology.skeletonize((np.flip(raw) if disk['pattern_rotate_180'] else raw) > 0)
    logger.info(f'Loaded the pattern {path.name} {pattern.shape}, '
                f'{int(pattern.sum())} skeleton pixels')
    return pattern


class _SliderSpin(QWidget):
    """ A horizontal slider bound to a spin box: drag roughly, type exactly.

    The two always show the same value. The slider spans `set_span` - which
    may be narrower than the range the box accepts, a value beyond it pins
    the slider to its end - and moves on a linear or a logarithmic scale.

    Parameters
    ----------
    spin : PySide6.QtWidgets.QDoubleSpinBox
        The box holding the value; it stays the source of truth, and its
        ``valueChanged`` signal is the one to listen to.
    log : bool, optional
        Logarithmic slider scale, by default False; for a positive quantity
        such as gamma, whose 0.5 and 2 lie the same distance from 1.

    """

    STEPS = 1000

    def __init__(self, spin, log:bool=False):
        super().__init__()
        self.spin, self.log = spin, log
        self.span = (spin.minimum(), spin.maximum())
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setMinimumWidth(60)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # fixed 3:2 shares rather than size hints, so that the sliders of
        # several rows line up whatever the range of their boxes
        spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        spin.setMinimumWidth(70)
        layout.addWidget(self.slider, 3)
        layout.addWidget(spin, 2)
        # a short dock must not squeeze the row below its box
        self.setMinimumHeight(max(spin.sizeHint().height(), self.slider.sizeHint().height()))
        self.slider.valueChanged.connect(self._to_spin)
        spin.valueChanged.connect(self.sync)
        self.sync()

    def set_span(self, low:float, high:float):
        """ Set the values the two ends of the slider stand for.

        Parameters
        ----------
        low, high : float
            Values at the left and the right end; positive for a log scale.

        """
        self.span = (low, max(high, low + 1e-12))
        self.sync()

    def _scale(self, value:float) -> float:
        """ Position of `value` on the slider scale, linear or logarithmic. """
        return np.log(value) if self.log else value

    def _to_spin(self, position:int):
        """ Put the value under the slider into the box. """
        low, high = (self._scale(v) for v in self.span)
        value = low + (high - low) * position / self.STEPS
        self.spin.setValue(float(np.exp(value)) if self.log else value)

    def sync(self):
        """ Move the slider to the value of the box without echoing back. """
        low, high = (self._scale(v) for v in self.span)
        fraction = (self._scale(max(self.spin.value(), 1e-12) if self.log
                                else self.spin.value()) - low) / (high - low)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(np.clip(fraction, 0, 1) * self.STEPS)))
        self.slider.blockSignals(False)


class _CalibrationHub(QObject):
    """ Carries a new wavelength calibration to every widget that uses one.

    `SpectraWidget` emits `changed` after every calibration; the widgets with
    a spectral axis put the values into their boxes, where they can still be
    edited by hand. The last calibration is kept for widgets opened later.

    Attributes
    ----------
    changed : Signal(float, float)
        Wavelength of channel 0 and step per channel, in nanometres.
    last : tuple of float or None
        The most recent ``(start, step)``, None before any calibration.

    """

    changed = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.last = None
        self.changed.connect(lambda start, step: setattr(self, 'last', (start, step)))


@lru_cache(maxsize=1)
def calibration_hub() -> _CalibrationHub:
    """ The one `_CalibrationHub` of the session. """
    return _CalibrationHub()


class _Scroll(QScrollArea):
    """ Scroll area that asks for the full height of its content.

    A plain `QScrollArea` asks for a fixed default size, so a dock either
    shows a scroll bar it does not need or leaves a gap. This one asks for
    exactly its content and can still shrink to a few rows, with a vertical
    scroll bar, when the dock is shorter than that - so a widget never pushes
    the napari window past the screen. Content never scrolls sideways: the
    area is as wide as its content needs.

    Parameters
    ----------
    content : PySide6.QtWidgets.QWidget
        The widget to scroll.

    """

    def __init__(self, content:QWidget):
        super().__init__()
        self.setWidget(content)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        """ Pass a new content size on, so the dock follows a section opening. """
        if watched is self.widget() and event.type() == QEvent.Type.LayoutRequest:
            self.updateGeometry()
        return super().eventFilter(watched, event)

    def sizeHint(self):
        """ The size of the content at the current width, plus frame and scroll bar.

        The frame is counted as it is drawn: the napari theme gives every
        scroll area a border whatever its frame shape.

        """
        content = self.widget()
        hint = content.sizeHint()
        frame = 2 * self.frameWidth()
        # forms wrap their rows when narrow, so the height depends on the width
        width = self.viewport().width() if self.isVisible() else hint.width()
        if content.hasHeightForWidth() and width > 0:
            hint.setHeight(content.heightForWidth(width))
        hint.setWidth(hint.width() + frame + self.verticalScrollBar().sizeHint().width())
        hint.setHeight(hint.height() + frame)
        return hint

    def resizeEvent(self, event):
        """ Ask again for a height when the width changed: rows may (un)wrap. """
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width():
            self.updateGeometry()

    def minimumSizeHint(self):
        """ The width of the content, and the height of a few rows. """
        hint = self.widget().minimumSizeHint()
        hint.setWidth(hint.width() + 2 * self.frameWidth()
                      + self.verticalScrollBar().sizeHint().width())
        hint.setHeight(min(hint.height(), 120))
        return hint


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


class _TaskWidget(QWidget):
    """ Base class holding the progress bar, the status label and the
    background-task machinery shared by both dock widgets.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    """

    # emitted from a worker thread, delivered in the GUI thread as a napari
    # notification: a signal of the widget is what crosses the thread boundary
    notice = Signal(str)

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._thread = None
        self._worker = None
        self._result_callback = None
        self._tabified = False
        self.notice.connect(self._show_notice)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.status = _StatusLabel()
        self.status.setVisible(False)

    def _busy(self) -> bool:
        """ Whether a background task is currently running. """
        return self._thread is not None

    def _assemble(self, body:list, footer:list, fill:QWidget=None):
        """ Lay the widget out as a scrolling body above a fixed footer.

        The widget takes the full height of the dock. The body holds the
        parameters at the height they need, the footer - the action buttons,
        the progress bar and the status line - sits right under it, and the
        spare height stays empty below. When the dock is shorter than that,
        the body gives way and scrolls: the footer stays visible.

        Parameters
        ----------
        body : list of PySide6.QtWidgets.QWidget
            Sections of parameters, top to bottom.
        footer : list of PySide6.QtWidgets.QWidget
            Action widgets under the body; the progress bar and the status
            line are appended.
        fill : PySide6.QtWidgets.QWidget, optional
            A widget between body and footer that takes the spare height, a
            plot; the footer then sits at the bottom of the dock.

        """
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        for widget in body:
            content_layout.addWidget(widget)
        content_layout.addStretch(1)
        self.scroll = _Scroll(content)
        # the body asks for its content height and no more
        self.scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.addWidget(self.scroll)
        if fill is not None:
            layout.addWidget(fill, 1)
        for widget in (*footer, self.progress, self.status):
            layout.addWidget(widget)
        if fill is None:
            layout.addStretch(1)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

    def _show_notice(self, text:str):
        """ Show `text` as a napari info notification. """
        from napari.utils.notifications import show_info

        show_info(text)

    def showEvent(self, event):
        """ On first show, join the other docks of the same side as a tab. """
        super().showEvent(event)
        if not self._tabified:
            self._tabified = True
            self._tabify()

    def _tabify(self):
        """ Put the dock of this widget into a tab with the docks beside it.

        napari stacks a new dock widget under the docks already open, so
        every widget shares the height of the side panel with the others.
        Tabbed, each one gets the full height and the newest is in front.
        A floating dock, or one alone on its side, is left as it is.

        """
        dock = self.parentWidget()
        while dock is not None and not isinstance(dock, QDockWidget):
            dock = dock.parentWidget()
        main = dock.parentWidget() if dock is not None else None
        if not isinstance(main, QMainWindow):
            return
        # napari docks a widget with a 'Maximum' height policy, which pins
        # the dock to the height the content had when it was laid out; taking
        # the full height instead lets the body follow its sections and the
        # window, see `_assemble`
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        if dock.isFloating():
            return
        area = main.dockWidgetArea(dock)
        others = [other for other in main.findChildren(QDockWidget)
                  if other is not dock and other.isVisible() and not other.isFloating()
                  and main.dockWidgetArea(other) == area]
        if others:
            main.tabifyDockWidget(others[-1], dock)
            dock.show()
            dock.raise_()

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

        Notes
        -----
        The refill itself is silent, but a selection that changed through it
        - the first layer appearing, the selected one removed - is announced
        with one ``currentIndexChanged``, so the widget follows it.

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
        if combo.currentText() != current:
            combo.currentIndexChanged.emit(combo.currentIndex())


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


def _collapsing_body(box:QGroupBox) -> QWidget:
    """ Give a checkable group a body that disappears when it is unchecked.

    An unchecked group is normally still there, greyed out, holding its full
    height against a dock that has none to spare. This hides the rows
    instead, so a section that is off costs one line.

    Parameters
    ----------
    box : PySide6.QtWidgets.QGroupBox
        Checkable group to equip.

    Returns
    -------
    PySide6.QtWidgets.QWidget
        The body the rows go into.

    """
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    body = QWidget()
    body.setVisible(box.isChecked())
    box.toggled.connect(body.setVisible)
    # an empty frame under the title would still take a line
    box.setFlat(not box.isChecked())
    box.toggled.connect(lambda checked: box.setFlat(not checked))
    layout.addWidget(body)
    return body


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
    """ Dock widget running `PhaseModelRecon` on a batch of phase images.

    The batch is a folder of images or an open image layer (the bundled sample
    included); the pattern is `bundled_pattern`. `PhaseModelRecon.run` and
    `PhaseModelRecon.lambda_stack_recon` run in a worker and the
    ``(n_lambda, height, width)`` stack becomes a layer; the input batch and
    the modelled arcs (`PhaseModelRecon.arc_labels`) are optional layers of the
    frame shape. Frames are cropped (typed or from a shapes rectangle, see
    `_crop_from_shape`) and optionally cleaned by `utils.despeckle` and
    `utils.subtract_dark_level` on every read. *Pattern passes* is
    ``line_passes``, kept in the bold *Pattern line* box because no fit finds
    it; *Single-row lines* is ``line_passes=None``; the read-only width and
    resolution come from `_show_line`.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    Attributes
    ----------
    recon : PhaseModelRecon or None
        Instance of the last successful run, for the napari console.

    """

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.recon = None
        self.cost_map_dialog = None
        self._preview_path = None
        self._crop_layer = None
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_stacks)
        self.viewer.layers.events.removed.connect(self._refresh_stacks)
        self._refresh_stacks()
        self._refresh_source()

    def _build_ui(self):
        """ Assemble the input, parameter and run sections of the widget. """

        # input batch
        input_box = QGroupBox('Input batch')
        input_form = _form(input_box)
        self.source_combo = QComboBox()
        self.source_combo.addItems(['folder of phase images', 'image layer'])
        self.source_combo.currentIndexChanged.connect(self._refresh_source)
        input_form.addRow('Source', self.source_combo)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText('folder with the phase images')
        self.folder_edit.editingFinished.connect(self._refresh_file_count)
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

        # crop
        self.crop_box = QGroupBox('Crop every frame')
        self.crop_box.setCheckable(True)
        self.crop_box.setChecked(False)
        crop_form = _form(_collapsing_body(self.crop_box))
        setup = setup_params()
        disk, optics = setup['disk'], setup['optics']
        self.crop_spins = {name: _spin(0, 65535, int(default))
                           for name, default in zip(('row_min', 'row_max', 'col_min', 'col_max'),
                                                    (1000, 2000, 1000, 2000))}
        for label, names in (('Rows', ('row_min', 'row_max')),
                             ('Columns', ('col_min', 'col_max'))):
            row = QHBoxLayout()
            for name in names:
                row.addWidget(self.crop_spins[name])
            crop_form.addRow(label, row)
        self.shape_combo = QComboBox()
        self.shape_combo.setToolTip('Shapes layer to draw the crop in: its last rectangle '
                                    'fills the fields above as it is drawn or moved')
        self.shape_combo.currentIndexChanged.connect(self._watch_crop_layer)
        crop_form.addRow('Draw in', self.shape_combo)

        # preprocessing
        self.preprocess_box = QGroupBox('Preprocess every frame')
        self.preprocess_box.setCheckable(True)
        self.preprocess_box.setChecked(False)
        preprocess_form = _form(_collapsing_body(self.preprocess_box))
        self.despeckle_check = QCheckBox('Replace hot and dead pixels')
        self.despeckle_check.setChecked(True)
        self.despeckle_check.setToolTip('Replace only the pixels that are outliers against '
                                        'their own neighbourhood, so the resolution of the '
                                        'frame is untouched')
        preprocess_form.addRow(self.despeckle_check)
        self.despeckle_threshold_spin = _spin(1.0, 50.0, 6.0,
                                              0.5, decimals=1)
        preprocess_form.addRow('Outlier threshold, sigma', self.despeckle_threshold_spin)
        self.despeckle_size_spin = _spin(3, 9, 3, 2)
        self.despeckle_size_spin.setToolTip('Median window: 3 catches single pixels, '
                                            '5 also catches small clusters')
        preprocess_form.addRow('Median window, px', self.despeckle_size_spin)
        self.dark_check = QCheckBox('Subtract the camera dark level')
        self.dark_check.setToolTip('A constant pedestal carries no signal and costs fold '
                                   'contrast, because it inflates the mean of a frame '
                                   'without touching its modulation')
        preprocess_form.addRow(self.dark_check)
        self.dark_percentile_spin = _spin(0.0, 50.0, 1.0, 0.5,
                                          decimals=1)
        preprocess_form.addRow('Dark level percentile', self.dark_percentile_spin)

        # pattern line: the passes belong to the acquisition and no fit can find
        # them in the frames, so they get a box of their own above the rest, in
        # bold
        pulse = disk['pass_pulse_us']
        line_box = QGroupBox('Pattern line')
        line_form = _form(line_box)
        passes_title = QLabel('Pattern passes')
        self.passes_spin = _spin(1, 6, 1)
        self.passes_spin.setToolTip(f'Passes of the pattern during one exposure, one per '
                                    f'{pulse:g} us of illumination pulse: 1 = {pulse:g} us, '
                                    f'2 = {2 * pulse:g} us, 4 = {4 * pulse:g} us. Every band is '
                                    f'written into the rows its line swept')
        self.pulse_label = QLabel()
        for widget in (passes_title, self.passes_spin, self.pulse_label):
            widget.setStyleSheet('font-weight: bold')
        passes_row = QHBoxLayout()
        passes_row.addWidget(self.passes_spin)
        passes_row.addWidget(self.pulse_label)
        line_form.addRow(passes_title, passes_row)
        self.single_row_check = QCheckBox('Single-row lines (original algorithm)')
        self.single_row_check.setToolTip('Write every band into one row 1 px wide whatever the '
                                         'passes, as PhaseModelRecon does with '
                                         'line_passes=None; the passes then only set the '
                                         'spectral resolution below')
        line_form.addRow(self.single_row_check)
        self.width_label = QLabel()
        self.width_label.setToolTip('Rows the line swept during one exposure: Pattern passes '
                                    'x Pass width x line period; every band is written into '
                                    'all of them')
        line_form.addRow('Line width', self.width_label)
        self.resolution_label = QLabel()
        self.resolution_label.setToolTip('Expected FWHM of a narrow emission line in the '
                                         'lambda stack: optics.spectral_fwhm of '
                                         'setup_params.yaml at one pass, widened by the sweep '
                                         'of Pattern passes x Pass width. The channels sample '
                                         'it several times over; this, not the channel count, '
                                         'is the spectral resolution')
        line_form.addRow('Spectral resolution', self.resolution_label)

        # main parameters
        main_box = QGroupBox('Reconstruction')
        main_form = _form(main_box)
        self.geometry_frames_spin = _spin(1, 64, 4)
        main_form.addRow('Geometry frames', self.geometry_frames_spin)
        n_lambda = optics['n_lambda']
        self.n_lambda_spin = _spin(0, 4096, 0 if str(n_lambda) == 'auto' else int(n_lambda))
        self.n_lambda_spin.setSpecialValueText('auto')
        self.n_lambda_spin.setToolTip('Channels per band; auto = the median line period of '
                                      'the batch in rows')
        main_form.addRow('Lambda channels', self.n_lambda_spin)
        self.accumulation_combo = QComboBox()
        self.accumulation_combo.addItems(['mean', 'max', 'overwrite'])
        main_form.addRow('Accumulation', self.accumulation_combo)
        self.input_layer_check = QCheckBox('Also add the input batch as a stack')
        self.input_layer_check.setChecked(True)
        self.input_layer_check.setToolTip('The frames as the reconstruction read them, '
                                          'crop included, as one image layer')
        self.arcs_layer_check = QCheckBox('Also add the modelled arcs as labels')
        self.arcs_layer_check.setChecked(True)
        self.arcs_layer_check.setToolTip('One label per arc, the same value for the same arc '
                                         'in every frame, in the shape of the input frames')
        self.invert_check = QCheckBox('Invert the spectral axis')
        self.invert_check.setChecked(bool(optics['invert_spectral_axis']))
        self.invert_check.setToolTip('Reverse the channel order of the lambda stack, for '
                                     'an instrument whose dispersion runs the other way. '
                                     'A lambda stack should run from short to long '
                                     'wavelengths')
        self.cost_map_check = QCheckBox('Also show the phase cost map')
        self.cost_map_check.setToolTip('Open the reference diagnostic of the run in a '
                                       'window of its own: the folded phase cost of every '
                                       'frame with the fitted phase on it')
        main_form.addRow(self.invert_check)
        main_form.addRow(self.input_layer_check)
        main_form.addRow(self.arcs_layer_check)
        main_form.addRow(self.cost_map_check)

        # advanced parameters live in a window of their own: seventeen more rows
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
        period = (REFERENCE_PERIOD if str(optics['n_lambda']) == 'auto'
                  else int(optics['n_lambda']))
        scale = period / REFERENCE_PERIOD
        self.init_sigma_spin = _spin(0.0, 50.0, round(PERIOD_SCALED['init_sigma'] * scale, 1),
                                     0.5, decimals=1)
        self.fold_sigma_spin = _spin(0.0, 50.0, round(PERIOD_SCALED['fold_sigma'] * scale, 1),
                                     0.5, decimals=1)
        self.arc_tolerance_spin = _spin(0, 100,
                                        max(1, round(PERIOD_SCALED['arc_edge_tolerance'] * scale)))
        for spin in (self.init_sigma_spin, self.fold_sigma_spin, self.arc_tolerance_spin):
            spin.setToolTip(f'Default scaled with the line period optics.n_lambda = {period} '
                            f'rows from its value at {REFERENCE_PERIOD} rows')
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
        self.drift_combo.setCurrentText({'1': '+1', '+1': '+1', '-1': '-1'}.get(
            str(disk['drift_direction']), 'auto'))
        self.pass_width_spin = _spin(0.001, 0.5, float(disk['pass_width']), 0.001, decimals=4)
        self.pass_width_spin.setToolTip('Distance a line moves in one pass, as a fraction of '
                                        'the line period')
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
                              ('Drift direction', self.drift_combo),
                              ('Pass width, period', self.pass_width_spin)):
            advanced_form.addRow(label, widget)
        for spin in (self.n_lambda_spin, self.passes_spin, self.pass_width_spin):
            spin.valueChanged.connect(self._show_line)
        self.single_row_check.toggled.connect(self._show_line)
        self._show_line()

        # run
        self.run_btn = QPushButton('Run reconstruction')
        self.run_btn.clicked.connect(self._run)

        self._assemble([input_box, line_box, self.crop_box, self.preprocess_box, main_box,
                        advanced_btn], [self.run_btn])

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
        """ Keep the layer combos in sync with the open layers. """
        self._refill_layer_combo(self.stack_combo,
                                 [layer for layer in self._image_layers()
                                  if layer.data.ndim == 3])
        self._refill_layer_combo(self.shape_combo,
                                 [layer for layer in self.viewer.layers
                                  if isinstance(layer, Shapes)])
        if self._from_layer():
            self._refresh_file_count()

    def _watch_crop_layer(self):
        """ Follow the rectangles of the selected shapes layer into the crop fields. """
        if self._crop_layer is not None:
            try:
                self._crop_layer.events.data.disconnect(self._crop_from_shape)
            except (TypeError, RuntimeError, ValueError):
                pass
        name = self.shape_combo.currentText()
        self._crop_layer = self.viewer.layers[name] if name in self.viewer.layers else None
        if self._crop_layer is not None:
            self._crop_layer.events.data.connect(self._crop_from_shape)
            if len(self._crop_layer.data):
                self._crop_from_shape()

    def _crop_from_shape(self, event=None):
        """ Fill the crop fields from the last rectangle of the watched shapes layer.

        Called whenever a shape of that layer is drawn, moved or resized. Only
        a rectangle aligned with the frame axes is taken, because that is the
        only shape an array slice can express - `PhaseModelRecon` crops with a
        plain slice. The vertices are taken through the scale and the offset
        of their layer, so a rectangle drawn over a scaled layer still lands
        on the right frame pixels.

        Parameters
        ----------
        event : napari.utils.events.Event, optional
            The data event of the layer; unused.

        """
        layer = self._crop_layer
        if layer is None or not len(layer.data):
            return
        # the last two axes are the frame, whatever dimensionality the layer has
        vertices = np.asarray(layer.data[-1], dtype=float)[:, -2:]
        vertices = vertices * np.asarray(layer.scale[-2:]) + np.asarray(layer.translate[-2:])
        rows, cols = vertices[:, 0], vertices[:, 1]
        aligned = (len(vertices) == 4
                   and np.ptp(np.sort(rows)[:2]) < 1 and np.ptp(np.sort(rows)[2:]) < 1
                   and np.ptp(np.sort(cols)[:2]) < 1 and np.ptp(np.sort(cols)[2:]) < 1)
        if not aligned:
            self._on_message(f'The last shape of "{layer.name}" is not a rectangle aligned '
                             f'with the frame axes; a crop is a plain slice')
            self.status.setVisible(True)
            return
        bounds = {'row_min': rows.min(), 'row_max': rows.max(),
                  'col_min': cols.min(), 'col_max': cols.max()}
        for key, value in bounds.items():
            self.crop_spins[key].setValue(int(round(max(value, 0))))
        self.crop_box.setChecked(True)
        crop = [self.crop_spins[key].value() for key in
                ('row_min', 'row_max', 'col_min', 'col_max')]
        self._on_message(f'Crop {crop} from "{layer.name}"')
        self.status.setVisible(True)

    def _load_preview(self, path:str):
        """ Show the first phase image of the folder, with a layer to draw the crop in.

        The frame is read in the worker thread. A shapes layer `<batch>_crop`
        is added beside it in rectangle mode and watched, so the crop is drawn
        on the data instead of typed blind.

        Parameters
        ----------
        path : str
            Path of the first phase image.

        """
        if self._busy():
            return
        self._preview_path = path
        batch = Path(path).parent.name

        def task(report):
            """ Read the frame. """
            from skimage import io

            return batch, io.imread(path)

        self._start_task(task, self._show_preview, f'Loading {Path(path).name}...')

    def _show_preview(self, result:tuple):
        """ Add or update the first-phase layer and the crop layer.

        Parameters
        ----------
        result : tuple
            Batch name and frame, from `_load_preview`.

        """
        batch, frame = result
        name, crop_name = layer_name(batch, 'first', 'phase'), layer_name(batch, 'crop')
        if name in self.viewer.layers:
            self.viewer.layers[name].data = frame
        else:
            self.viewer.add_image(frame, name=name)
        if crop_name not in self.viewer.layers:
            crop = self.viewer.add_shapes(name=crop_name, ndim=2, edge_color='yellow',
                                          face_color='transparent',
                                          edge_width=max(frame.shape) / 400)
            crop.mode = 'add_rectangle'
        self.shape_combo.setCurrentText(crop_name)
        if self._crop_layer is not self.viewer.layers[crop_name]:
            self._watch_crop_layer()
        self.viewer.layers.selection.active = self.viewer.layers[crop_name]
        self._on_message(f'First phase {frame.shape}: draw the crop in "{crop_name}"')
        logger.info(f'First phase image of "{batch}" {frame.shape} shown for drawing the crop')

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
            monotonicity constraint of `PhaseModelRecon` relies on. Hidden
            files are skipped: on exFAT and FAT drives macOS stores the
            extended attributes of every file in a hidden `._<name>`
            companion, which matches the same mask but is not an image.

        """
        folder = Path(self.folder_edit.text())
        if not folder.is_dir():
            return []
        return sorted(str(path) for path in folder.glob(self.glob_edit.text())
                      if not path.name.startswith('.'))

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
        if paths and paths[0] != self._preview_path:
            self._load_preview(paths[0])

    def _show_line(self, *_):
        """ Fill the read-only fields of *Pattern line*: pulse, line width and FWHM.

        ``optics.n_lambda`` is the line period of the setup in rows, one
        channel per row, and turns the pass width into rows. The spectral FWHM
        is `line_fwhm` of ``optics.spectral_fwhm``, which is given in channels
        of ``optics.n_lambda``: another channel count in the box rescales it,
        *auto* is taken as the count of the file, and with *auto* in both the
        count is unknown until a run. *Single-row lines* writes one row per
        band, but the acquisition still swept its passes, so the FWHM follows
        the passes either way.

        """
        from .utils import line_fwhm

        setup = setup_params()
        optics, passes = setup['optics'], self.passes_spin.value()
        self.pulse_label.setText(f'= {passes * setup["disk"]["pass_pulse_us"]:g} µs pulse')
        reference = None if str(optics['n_lambda']) == 'auto' else int(optics['n_lambda'])
        channels = self.n_lambda_spin.value() or reference
        width = passes * self.pass_width_spin.value()
        # rows come from the period of the setup, not from the box: a different
        # channel count resamples the band, it does not change how far a line moves
        if self.single_row_check.isChecked():
            self.width_label.setText('1 row')
        elif channels is None:
            self.width_label.setText(f'{width:.3f} of the line period')
        else:
            self.width_label.setText(f'~ {width * (reference or channels):.1f} rows')
        if channels is None:
            self.resolution_label.setText('- (auto channels)')
            return
        fwhm = line_fwhm(passes, float(optics['spectral_fwhm']) * channels / (reference or channels),
                         self.pass_width_spin.value() * channels)
        self.resolution_label.setText(f'FWHM ~ {fwhm:.0f} channels')

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
                'drift_direction': drift,
                'line_passes': None if self.single_row_check.isChecked() else self.passes_spin.value(),
                'pass_width': self.pass_width_spin.value()}

    def _preprocess_callable(self):
        """ Build the per-frame preprocessing of the current settings.

        Returns
        -------
        callable or None
            ``preprocess(frame) -> frame``, or None when nothing is enabled.
            What it did is logged once, for the first frame of the run: it
            runs on every read of every frame and would otherwise fill the
            log with the same line.

        """
        from .utils import despeckle, subtract_dark_level

        if not self.preprocess_box.isChecked():
            return None
        despeckling = self.despeckle_check.isChecked()
        darkening = self.dark_check.isChecked()
        if not (despeckling or darkening):
            return None
        threshold = self.despeckle_threshold_spin.value()
        size = self.despeckle_size_spin.value()
        percentile = self.dark_percentile_spin.value()
        first = True

        def preprocess(frame):
            """ Clean one frame, in the order the camera adds its faults. """
            nonlocal first
            replaced, level = 0, 0.0
            if despeckling:
                frame, replaced = despeckle(frame, threshold=threshold, size=size)
            if darkening:
                frame, level = subtract_dark_level(frame, percentile=percentile)
            if first:
                first = False
                logger.info(f'Preprocessing every frame: '
                            + (f'{replaced} outlier pixels replaced '
                               f'({100 * replaced / frame.size:.3f}% of the first frame, '
                               f'{threshold} sigma, {size}x{size} window)'
                               if despeckling else 'no despeckling')
                            + (f', dark level {level:.1f} subtracted '
                               f'(percentile {percentile})' if darkening else ''))
            return frame

        return preprocess

    def _run(self):
        """ Validate the inputs and start the reconstruction task. """
        if self._from_layer():
            name = self.stack_combo.currentText()
            if name not in self.viewer.layers:
                self._on_message('Select a 3D image layer holding the phase images')
                self.status.setVisible(True)
                return
            frames = np.asarray(self.viewer.layers[name].data)
            batch, from_layer = name, True
        else:
            frames = self._input_paths()
            batch, from_layer = Path(self.folder_edit.text()).name, False
            if not frames:
                self._on_message('No input images: check the folder and the file mask')
                self.status.setVisible(True)
                return

        kwargs = self._recon_kwargs()
        method = self.accumulation_combo.currentText()
        preprocess = self._preprocess_callable()
        invert = self.invert_check.isChecked()
        # a batch that already is a layer has nothing to add back
        with_input = self.input_layer_check.isChecked() and not from_layer
        with_arcs = self.arcs_layer_check.isChecked()
        logger.info(f'Starting reconstruction of {len(frames)} images of "{batch}" '
                    f'with {method} accumulation'
                    + (', preprocessing on' if preprocess is not None else '')
                    + (', spectral axis inverted' if invert else ''))

        def task(report):
            """ Reconstruct the batch and build every requested layer.

            The pattern is prepared here rather than in `_run`, because
            skeletonising it takes about a second and the GUI thread must
            not spend it.

            """
            from .phase_model_recon import PhaseModelRecon

            recon = PhaseModelRecon(frames, bundled_pattern(), preprocess=preprocess,
                                    progress=report, **kwargs)
            self.notice.emit(f'Reconstruction runs on {recon.n_threads} threads')
            recon.run()
            lambda_stack = recon.lambda_stack_recon(accumulation_method=method)
            if invert:
                # the channel order is a property of the instrument, not of the
                # data: only the assembled stack is turned around, so that
                # anything asked of `recon` afterwards still answers in the
                # order it sampled
                lambda_stack = np.ascontiguousarray(lambda_stack[::-1])
                logger.info(f'Spectral axis inverted, channel 0 is now what the '
                            f'reconstruction sampled as channel {len(lambda_stack) - 1}')
            report(recon.PROGRESS['sampling'])
            batch_stack = recon.input_stack() if with_input else None
            report(98)
            arcs = recon.arc_labels() if with_arcs else None
            report(100)
            return recon, lambda_stack, batch_stack, arcs, batch

        self._start_task(task, self._add_result, f'Reconstructing {len(frames)} images...')

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
        if self.cost_map_check.isChecked():
            self._show_cost_map()

    def _show_cost_map(self):
        """ Open the phase cost map of the reconstruction just finished.

        A previous window is closed first: the figure belongs to one run, and
        leaving the old one open next to the new one invites reading the
        wrong batch.

        """
        if self.cost_map_dialog is not None:
            self.cost_map_dialog.close()
        try:
            self.cost_map_dialog = _PhaseCostDialog(self, self.recon)
        except Exception as err:      # a diagnostic must not sink the result
            logger.warning(f'Could not draw the phase cost map: {err}')
            return
        self.cost_map_dialog.show()
        self.cost_map_dialog.raise_()
        logger.info(f'Phase cost map shown, phase filling coefficient '
                    f'{self.recon.phase_filling:.3f}')


class _PhaseCostDialog(QDialog):
    """ Window showing `PhaseModelRecon.plot_phase_cost_map` of a finished run:
    whether the fitted phases sit on the cost ridge.

    Parameters
    ----------
    widget : PhaseModelReconWidget
        Owner, parent window and source of the viewer theme.
    recon : PhaseModelRecon
        A reconstruction whose `run` has completed.

    """

    def __init__(self, widget, recon):
        from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
        from matplotlib.figure import Figure

        super().__init__(widget)
        self._widget = widget
        self.setWindowTitle('Phase cost map')
        self.resize(760, 620)
        layout = QVBoxLayout(self)

        self.figure = recon.plot_phase_cost_map(Figure(figsize=(10, 6.5)))

        self._style()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(320, 240)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)

        report = _StatusLabel()
        report.setMessage(
            f'{recon.n_image} phase images, phase filling coefficient '
            f'{recon.phase_filling:.3f}\n'
            f'drift {recon.phase_delta:+.4f} period per frame, '
            f'{recon.phase_delta * (recon.n_image - 1):+.3f} periods in total')

        close_btn = QPushButton('Close')
        close_btn.clicked.connect(self.accept)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        layout.setStretchFactor(self.canvas, 1)
        layout.addWidget(report)
        layout.addWidget(close_btn)

    def _style(self):
        """ Repaint the captured figure in the colours of the napari theme.

        Every axes of the figure is covered, the colour bar included, because
        `plot_phase_cost_map` builds three of them and a white strip left
        among dark ones is more distracting than no styling at all.

        """
        background, text = 'white', 'black'
        try:
            theme = get_theme(self._widget.viewer.theme)
            background, text = theme.background.as_hex(), theme.text.as_hex()
        except Exception:  # an unknown theme must not cost the whole window
            logger.warning(f'Unknown napari theme "{self._widget.viewer.theme}", '
                           f'the phase cost map keeps its default colours')

        self.figure.set_facecolor(background)
        for axes in self.figure.axes:
            axes.set_facecolor(background)
            for spine in axes.spines.values():
                spine.set_color(text)
            axes.tick_params(colors=text, labelsize='small')
            axes.xaxis.label.set_color(text)
            axes.yaxis.label.set_color(text)
            axes.title.set_color(text)
            legend = axes.get_legend()
            if legend is not None:      # a default legend box is the last light patch
                legend.get_frame().set_facecolor(background)
                legend.get_frame().set_edgecolor(text)
                for label in legend.get_texts():
                    label.set_color(text)

class PostProcessingWidget(_TaskWidget):
    """ Dock widget applying `utils.interpolate_zero_gaps` or
    `utils.max_pooling2d` to a 2D image or, channel by channel, to a stack.

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
        source_widget = QWidget()
        source_form = _form(source_widget)
        self.layer_combo = QComboBox()
        source_form.addRow('Layer', self.layer_combo)
        self.method_combo = QComboBox()
        self.method_combo.addItems(['interpolate zero gaps', 'max pooling 2D'])
        self.method_combo.currentIndexChanged.connect(self._refresh_params)
        source_form.addRow('Method', self.method_combo)

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

        self._assemble([source_widget, self.interpolate_box, self.pooling_box], [self.apply_btn])
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

    It also holds the fit settings, which `SpectraWidget` calibrates with, so
    the peaks of the wavelength axis are the peaks shown here.

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
    """ Window showing the calibration fit: found channel against known
    wavelength per reference emitter, coloured by region, and the polynomial.

    A region whose points sit to one side of the line has its own dispersion,
    see `_PeakDriftDialog`; a region left out of the fit has a hollow legend
    marker.

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
        used = {index for index, fit in widget._calibration_fits}
        excluded = [(index, fit['n_peaks']) for index, fit in enumerate(widget.fits)
                    if index not in used]
        for index, n_peaks in excluded:
            # no point to draw: a legend entry only
            axes.scatter([], [], s=26, facecolors='none', edgecolors=colors[index],
                         label=f'ROI {widget.roi_labels[index]}: {n_peaks} of '
                               f'{len(peaks_nm)} peaks, excluded')
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
            f'from {len(widget._calibration_fits)} of {len(widget.fits)} ROIs'
            + (''.join(f'\nROI {widget.roi_labels[index]} excluded: {n_peaks} of '
                       f'{len(peaks_nm)} peaks found' for index, n_peaks in excluded)))

    def open_now(self):
        """ Redraw and show the window. """
        self.refresh()
        self.show()
        self.raise_()


class _PeakDriftDialog(_PlotDialog):
    """ Window drawing the drift of unmixed emitters across the frame.

    `utils.peak_drift_unmixed` fits the emitters on tiles; all components are
    drawn together along both frame axes, because a tilt of the spectral axis
    moves them the same way and the sample does not. Nothing is added to the
    viewer.

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

    Every region of a labels layer gives its plain mean spectrum; a 3D mask
    (one plane per channel, as napari creates over a stack) is flattened first
    by `utils.flatten_roi_mask` and added as a layer. `utils.fit_spectral_peaks`
    finds the known emission peaks in every ROI spectrum and
    `utils.spectral_calibration` fits channel against wavelength; only ROIs with
    exactly one component per reference peak enter the fit, all of them at once.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    Attributes
    ----------
    roi_labels : numpy.ndarray or None
        Label values, in the order of the rows of `spectra`.
    spectra : numpy.ndarray or None
        ``(n_roi, n_lambda)`` mean spectra.
    fits : list of dict or None
        `utils.fit_spectral_peaks` result per row of `spectra`.
    calibration : dict or None
        `utils.spectral_calibration` result, None while uncalibrated.

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

        # source layers
        source_box = QGroupBox('Source')
        source_form = _form(source_box)
        self.stack_combo = QComboBox()
        source_form.addRow('Stack', self.stack_combo)
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
        self.peaks_edit.setPlaceholderText('nm, comma separated')
        self.degree_spin = _spin(1, 3, 1)
        self.degree_spin.setToolTip('Polynomial degree of the fit from channel to wavelength')
        self.degree_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.degree_spin.setMaximumWidth(36)
        self.peaks_edit.setMinimumWidth(90)
        peaks_row = QHBoxLayout()
        peaks_row.addWidget(self.peaks_edit, 1)
        peaks_row.addWidget(QLabel('degree'))
        peaks_row.addWidget(self.degree_spin)
        calibration_form.addRow('Peaks', peaks_row)
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
        # the plot takes what the forms leave, not the height of its figure
        plot_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        self.fit_btn = QPushButton('Peak fit with Gaussian...')
        self.fit_btn.setToolTip('Fit a sum of Gaussians to one region and show the '
                                'components in a window of its own')
        self.fit_btn.clicked.connect(self._open_fit_dialog)
        self.fit_btn.setEnabled(False)
        self.figure = Figure(figsize=(4, 3), layout='tight')
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(120, 150)
        plot_box.setMinimumHeight(220)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Expanding)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)

        self.save_btn = QPushButton('Save CSV...')
        self.save_btn.setToolTip('Save the spectra, one row per channel, to a CSV file')
        self.save_btn.clicked.connect(self._save_csv)
        self.save_btn.setEnabled(False)

        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.fit_btn)
        actions_layout.addWidget(self.save_btn)
        # the plot takes every pixel the dock can spare
        self._assemble([source_box, calibration_box], [actions], fill=plot_box)
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
        from .utils import flatten_roi_mask, roi_mean_spectra

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
            """ Flatten the mask and average every region in the worker. """
            flat_mask = flatten_roi_mask(mask)
            return (*roi_mean_spectra(cube, flat_mask, report=report), flat_mask)

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
            Wavelengths in nanometres, ascending whatever the order typed, since
            they are paired with the fitted peaks in channel order; None if the
            field cannot be parsed.

        """
        try:
            return np.sort([float(value) for value in
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
            f'{self.calibration["dispersion"]:.2f} nm/ch, {low:.0f}..{high:.0f} nm, '
            f'rms {self.calibration["rms"]:.1f} nm, {len(usable)}/{len(self.fits)} ROIs')
        self.calibration_label.setToolTip(
            f'{self.calibration["dispersion"]:.3f} nm per channel, axis {low:.1f}..{high:.1f} '
            f'nm, residual rms {self.calibration["rms"]:.2f} nm, from {len(usable)} of '
            f'{len(self.fits)} ROIs'
            + ''.join(f'\nROI {self.roi_labels[index]} excluded: {fit["n_peaks"]} of '
                      f'{len(peaks_nm)} peaks found' for index, fit in enumerate(self.fits)
                      if fit['n_peaks'] != len(peaks_nm)))
        self.calibration_fit_btn.setEnabled(True)
        self._publish_axis()
        self._replot()
        if self.calibration_dialog is not None and self.calibration_dialog.isVisible():
            self.calibration_dialog.refresh()
        if self.drift_dialog is not None and self.drift_dialog.isVisible():
            self.drift_dialog._replot()
        if self.fit_dialog is not None and self.fit_dialog.isVisible():
            self.fit_dialog.refresh()

    def _publish_axis(self):
        """ Hand the calibrated axis to the widgets with a spectral axis.

        The other widgets describe a spectral axis by its start and its step,
        so the calibration is reduced to the line through its first and last
        channel; for a degree 1 calibration that is exact. The values go out
        through `calibration_hub` only. By convention the channels run from
        short to long wavelengths; a calibration that runs the other way is
        not passed on, the stack has to be reconstructed with *Invert the
        spectral axis*.

        """
        wavelength = np.asarray(self.calibration['wavelength'], dtype=float)
        start = float(wavelength[0])
        step = float((wavelength[-1] - wavelength[0]) / max(len(wavelength) - 1, 1))
        if step <= 0:
            self._on_message('The calibrated axis runs from long to short wavelengths: '
                             'reconstruct with "Invert the spectral axis"')
            self.status.setVisible(True)
            logger.warning(f'Calibrated step {step:.3f} nm per channel is not positive, '
                           f'the axis is not passed on')
            return
        linear = start + step * np.arange(len(wavelength))
        logger.info(f'Spectral axis {start:.2f} nm + {step:.3f} nm per channel published, '
                    f'largest deviation of the line from the calibration '
                    f'{np.abs(linear - wavelength).max():.2f} nm')
        calibration_hub().changed.emit(start, step)

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
        if calibrated:
            # the calibrated span, first to last channel, as the stack runs
            self.axes.set_title(f'{x[0]:.1f}..{x[-1]:.1f} nm', color=text, fontsize='medium')
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


class _AxisBox(QGroupBox):
    """ The spectral axis of a stack: wavelength of channel 0 and step per channel.

    Shared by the widgets that read a stack in nanometres. The values come
    from the last calibration of `SpectraWidget` (pushed through
    `calibration_hub`, also into a box that is already open) or from the
    defaults, the calibration of the bundled sample at the default 89
    channels (its 88-channel step of 3.55 nm times 88 / 89). Typed values
    hold until the next calibration. Nothing is read from the layers.

    Parameters
    ----------
    viewer : napari.Viewer
        Viewer holding the stack layers.
    stack_combo : PySide6.QtWidgets.QComboBox
        Combo of the widget selecting the stack the axis belongs to.
    visible : bool, optional
        Also count the channels inside the visible range, by default False.

    """

    def __init__(self, viewer, stack_combo, visible:bool=False):
        super().__init__('Spectral axis')
        self.viewer, self.stack_combo, self.visible = viewer, stack_combo, visible
        self.origin = 'default'
        form = _form(self)
        self.start_spin = _spin(100.0, 2000.0, 443.3, 1.0, decimals=2)
        self.start_spin.setToolTip('Wavelength of channel 0')
        form.addRow('Start wavelength, nm', self.start_spin)
        self.step_spin = _spin(0.0001, 100.0, 3.51, 0.05, decimals=4)
        self.step_spin.setToolTip('Wavelength step between consecutive channels; channels '
                                  'run from short to long wavelengths')
        form.addRow('Step, nm per slice', self.step_spin)
        self.label = _StatusLabel()
        form.addRow(self.label)
        for box in (self.start_spin, self.step_spin):
            box.valueChanged.connect(self._edited)
        stack_combo.currentIndexChanged.connect(self.refresh)
        hub = calibration_hub()
        hub.changed.connect(self.set_axis)
        if hub.last is not None:
            self.set_axis(*hub.last)
        self.refresh()

    def set_axis(self, start:float, step:float, origin:str='calibration'):
        """ Put a calibrated axis into the boxes.

        Parameters
        ----------
        start, step : float
            Wavelength of channel 0 and step per channel, in nanometres.
        origin : str, optional
            Where the values came from, shown under the boxes; by default
            'calibration'.

        """
        for box, value in ((self.start_spin, start), (self.step_spin, step)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self.origin = origin
        self.refresh()

    def _edited(self):
        """ Mark typed values as such. """
        self.origin = 'manual'
        self.refresh()

    def wavelengths(self, n_lambda:int) -> np.ndarray:
        """ Wavelength of every channel of an `n_lambda` stack, in nanometres. """
        return self.start_spin.value() + self.step_spin.value() * np.arange(n_lambda)

    def refresh(self):
        """ Report the range the current values describe, and their origin. """
        name = self.stack_combo.currentText()
        if name not in self.viewer.layers:
            self.label.setMessage('no 3D image layer')
            return
        wavelengths = self.wavelengths(len(self.viewer.layers[name].data))
        text = f'{len(wavelengths)} channels, {wavelengths[0]:.1f}..{wavelengths[-1]:.1f} nm'
        if self.visible:
            text += f', {int(((wavelengths >= 360) & (wavelengths <= 830)).sum())} visible'
        self.label.setMessage(f'{text} ({self.origin})')


class LambdaRGBWidget(_TaskWidget):
    """ Dock widget rendering a spectral stack as true-colour sRGB.

    `utils.lambda_stack_to_rgb` runs once in a worker (CIE 1931 2 degree
    observer, on which sRGB is defined); `utils.rgb_levels` then applies black,
    white, gamma and the intensity mode live. Any 3D layer with a spectral first
    axis works, the axis comes from `_AxisBox`.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    """

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self._pending = None
        self._linear = None             # linear sRGB of the last render
        self._target = None             # the layer that shows it
        self._levels_timer = QTimer(self, singleShot=True, interval=60)
        self._levels_timer.timeout.connect(self._apply_levels)
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    def _build_ui(self):
        """ Assemble the source, spectral axis, render and levels sections. """

        source_box = QGroupBox('Source')
        source_form = _form(source_box)
        self.stack_combo = QComboBox()
        self.stack_combo.setToolTip('Any 3D image layer whose first axis is spectral. An '
                                    'interpolated lambda stack renders best')
        source_form.addRow('Stack', self.stack_combo)

        axis_box = self.axis = _AxisBox(self.viewer, self.stack_combo, visible=True)
        self.start_spin, self.step_spin = axis_box.start_spin, axis_box.step_spin
        self.axis_label = axis_box.label

        self.render_btn = QPushButton('Render as RGB')
        self.render_btn.clicked.connect(self._render)

        levels_box = QGroupBox('Levels')
        levels_form = _form(levels_box)
        self.intensity_combo = QComboBox()
        self.intensity_combo.addItems(['absolute', 'per-pixel'])
        self.intensity_combo.setToolTip('absolute: brightness through the levels, relative '
                                        'brightness is kept. per-pixel: every pixel above '
                                        'the black level at the full brightness of its hue')
        levels_form.addRow('Mode', self.intensity_combo)
        self.black_spin = _spin(0.0, 1e12, 0.0, 1.0, decimals=2)
        self.black_spin.setToolTip('Brightness rendered black, in the units of the stack; '
                                   'a flat spectrum of intensity v has brightness v')
        self.black_slider = _SliderSpin(self.black_spin)
        levels_form.addRow('Black', self.black_slider)
        self.white_spin = _spin(0.0, 1e12, 1.0, 1.0, decimals=2)
        self.white_spin.setToolTip('Brightness rendered at full white; above it a pixel '
                                   'saturates in brightness, not in hue')
        self.white_row = levels_form.rowCount()
        self.white_slider = _SliderSpin(self.white_spin)
        levels_form.addRow('White', self.white_slider)
        self.gamma_spin = _spin(0.1, 5.0, 1.0, 0.05, decimals=2)
        self.gamma_spin.setToolTip('Exponent between the levels: 1 keeps relative brightness, '
                                   'below 1 lifts faint regions, above 1 presses them down')
        self.gamma_row = levels_form.rowCount()
        self.gamma_slider = _SliderSpin(self.gamma_spin, log=True)
        levels_form.addRow('Gamma', self.gamma_slider)
        self.auto_btn = QPushButton('Auto')
        self.auto_btn.setToolTip('Black at the median brightness, the background of sparse '
                                 'emitters; white at its 99.5th percentile')
        self.auto_btn.clicked.connect(self._auto_levels)
        levels_form.addRow(self.auto_btn)
        self.levels_form = levels_form
        for box in (self.black_spin, self.white_spin, self.gamma_spin):
            box.valueChanged.connect(self._levels_timer.start)
        self.intensity_combo.currentIndexChanged.connect(self._refresh_intensity)

        # render comes before the levels it unlocks, so it stays in the body
        self._assemble([source_box, axis_box, self.render_btn, levels_box], [])
        self._refresh_intensity()
        self._refresh_levels()

    def _set_controls_enabled(self, enabled:bool):
        """ Enable or disable the render button while a task is running. """
        self.render_btn.setEnabled(enabled)

    def _refresh_layers(self, event=None):
        """ Keep the stack combo in sync with the open 3D image layers. """
        self._refill_layer_combo(self.stack_combo,
                                 [layer for layer in self._image_layers()
                                  if layer.data.ndim == 3 and not layer.rgb])
        if hasattr(self, 'axis'):
            self.axis.refresh()
        self._refresh_levels()

    def _refresh_levels(self):
        """ Release the kept colour once its layer is gone; levels need a layer. """
        if self._target is not None and self._target not in self.viewer.layers:
            self._linear, self._target = None, None
        enabled = self._linear is not None
        for widget in (self.black_slider, self.white_slider, self.gamma_slider, self.auto_btn):
            widget.setEnabled(enabled)

    def _refresh_intensity(self):
        """ Show white and gamma only in the mode that uses them, then re-render. """
        absolute = self.intensity_combo.currentText() == 'absolute'
        self.levels_form.setRowVisible(self.white_row, absolute)
        self.levels_form.setRowVisible(self.gamma_row, absolute)
        self._levels_timer.start()

    def _render(self):
        """ Validate the selection and start the spectral conversion. """
        from .utils import lambda_stack_to_rgb, rgb_auto_levels

        name = self.stack_combo.currentText()
        if name not in self.viewer.layers:
            self._on_message('Select a 3D image layer whose first axis is spectral')
            self.status.setVisible(True)
            return
        layer = self.viewer.layers[name]
        stack = layer.data
        wavelengths = self.start_spin.value() + self.step_spin.value() * np.arange(len(stack))
        if not ((wavelengths >= 360) & (wavelengths <= 830)).any():
            self._on_message(f'The axis {wavelengths[0]:.0f}..{wavelengths[-1]:.0f} nm lies '
                             f'outside the visible range, 360..830 nm')
            self.status.setVisible(True)
            return
        kwargs = {'start_nm': self.start_spin.value(), 'step_nm': self.step_spin.value()}
        # the image keeps the frame geometry of its source, so it overlays it
        self._pending = (layer_name(batch_prefix(name), 'rgb'),
                         tuple(layer.scale[-2:]), tuple(layer.translate[-2:]))
        logger.info(f'Rendering "{name}" {stack.shape} as sRGB, {kwargs}')

        def task(report):
            """ Convert the stack to linear sRGB and find its levels in the worker. """
            rgb, info = lambda_stack_to_rgb(stack, report=report, **kwargs)
            return rgb, info, rgb_auto_levels(rgb), float(rgb.max())

        self._start_task(task, self._add_result, f'Rendering {name} as RGB...')

    def _set_levels(self, black:float, white:float):
        """ Put levels into the boxes, with steps and decimals to their scale.

        The sliders follow the boxes; their span is set by `_add_result`.

        Parameters
        ----------
        black, white : float
            Levels in the intensity units of the stack.

        """
        scale = max(abs(white), 1e-12)
        decimals = int(np.clip(3 - np.floor(np.log10(scale)), 0, 8))
        for box, value in ((self.black_spin, black), (self.white_spin, white)):
            box.blockSignals(True)
            box.setDecimals(decimals)
            box.setSingleStep(scale / 100)
            box.setValue(value)
            box.blockSignals(False)
        self.black_slider.sync()
        self.white_slider.sync()

    def _auto_levels(self):
        """ Set the levels from the brightness of the kept colour. """
        from .utils import rgb_auto_levels

        if self._linear is not None:
            self._set_levels(*rgb_auto_levels(self._linear))
            self._apply_levels()

    def _apply_levels(self):
        """ Re-map the kept colour through the current levels into its layer. """
        from .utils import rgb_levels

        self._refresh_levels()
        if self._linear is None:
            return
        black, white = self.black_spin.value(), self.white_spin.value()
        if self.intensity_combo.currentText() == 'absolute' and white <= black:
            self._on_message('The white level must lie above the black level')
            self.status.setVisible(True)
            return
        self._target.data = rgb_levels(self._linear, black, white, self.gamma_spin.value(),
                                       self.intensity_combo.currentText())
        self._on_message(f'Levels {black:.6g}..{white:.6g}, gamma '
                         f'{self.gamma_spin.value():.2f}, {self.intensity_combo.currentText()}')
        self.status.setVisible(True)

    def _add_result(self, result:tuple):
        """ Keep the linear colour, set its levels and show the image.

        Parameters
        ----------
        result : tuple
            Linear sRGB and the summary of `lambda_stack_to_rgb`, the
            levels of `rgb_auto_levels` and the highest brightness.

        """
        from .utils import rgb_levels

        rgb, info, (black, white), top = result
        name, scale, translate = self._pending
        self._linear = rgb
        # the sliders span the whole brightness range, as napari's contrast
        # limits span the data range; the boxes accept any value beyond it
        for slider in (self.black_slider, self.white_slider):
            slider.set_span(0.0, max(top, white))
        self._set_levels(black, white)
        # the levels as the boxes show them, so that the image matches them
        black, white = self.black_spin.value(), self.white_spin.value()
        image = rgb_levels(rgb, black, white, self.gamma_spin.value(),
                           self.intensity_combo.currentText())
        self._target = self.viewer.add_image(image, name=name, rgb=True, scale=scale,
                                             translate=translate)
        self._refresh_levels()
        self._on_message(f'Done: {self._target.name} {image.shape}, {info["visible"]} visible '
                         f'channels, {100 * info["out_of_gamut"]:.0f}% of pixels outside sRGB')
        logger.info(f'Added layer "{self._target.name}" of shape {image.shape}, '
                    f'levels {black:.6g}..{white:.6g}')


class VirtualChannelsWidget(_TaskWidget):
    """ Dock widget summing a lambda stack over wavelength ranges.

    Up to four ranges, each becomes one channel of the output, see
    `virtual_channels`. The spectral axis is read in nanometres through
    `_AxisBox`, so a calibration in the ROI widget reaches it directly.

    The output layer carries the ranges in nanometres and the number of input
    channels summed into each in its `metadata`, as ``'ranges_nm'`` and
    ``'channels_summed'``. napari does not show layer metadata and does not
    write it to a saved TIFF, so it is reachable from the napari console only;
    the channel counts also appear in the status line.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    """

    # pass bands of the three emitters of the bundled sample, and a blue one
    DEFAULT_RANGES = ((500.0, 550.0, True), (560.0, 610.0, True),
                      (630.0, 690.0, True), (450.0, 500.0, False))

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self._pending = None
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    def _build_ui(self):
        """ Assemble the source, spectral axis, ranges and run sections. """
        source_box = QGroupBox('Source')
        source_form = _form(source_box)
        self.stack_combo = QComboBox()
        self.stack_combo.setToolTip('Any 3D image layer whose first axis is spectral')
        source_form.addRow('Stack', self.stack_combo)
        self.axis = _AxisBox(self.viewer, self.stack_combo)

        ranges_box = QGroupBox('Ranges, nm')
        ranges_form = _form(ranges_box)
        self.range_rows = []
        for k, (low, high, on) in enumerate(self.DEFAULT_RANGES):
            check = QCheckBox(f'{k + 1}')
            check.setChecked(on)
            low_spin = _spin(100.0, 2000.0, low, 1.0, decimals=0)
            high_spin = _spin(100.0, 2000.0, high, 1.0, decimals=0)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(low_spin)
            row_layout.addWidget(QLabel('..'))
            row_layout.addWidget(high_spin)
            check.toggled.connect(row.setEnabled)
            row.setEnabled(on)
            ranges_form.addRow(check, row)
            self.range_rows.append((check, low_spin, high_spin))

        self.run_btn = QPushButton('Create virtual channels')
        self.run_btn.clicked.connect(self._run)
        self._assemble([source_box, self.axis, ranges_box], [self.run_btn])

    def _set_controls_enabled(self, enabled:bool):
        """ Enable or disable the run button while a task is running. """
        self.run_btn.setEnabled(enabled)

    def _refresh_layers(self, event=None):
        """ Keep the stack combo in sync with the open 3D image layers. """
        self._refill_layer_combo(self.stack_combo,
                                 [layer for layer in self._image_layers()
                                  if layer.data.ndim == 3 and not layer.rgb])
        self.axis.refresh()

    def ranges(self) -> list:
        """ ``(low, high)`` of every enabled range, in nanometres. """
        return [(low.value(), high.value()) for check, low, high in self.range_rows
                if check.isChecked()]

    def _run(self):
        """ Validate the selection and start the summation task. """
        from .utils import virtual_channels

        name = self.stack_combo.currentText()
        ranges = self.ranges()
        problem = None
        if name not in self.viewer.layers:
            problem = 'Select a 3D image layer whose first axis is spectral'
        elif not ranges:
            problem = 'Enable at least one range'
        else:
            wavelengths = self.axis.wavelengths(len(self.viewer.layers[name].data))
            empty = [r for r in ranges
                     if not ((wavelengths >= min(r)) & (wavelengths <= max(r))).any()]
            if empty:
                problem = (f'No channel of {wavelengths.min():.1f}..{wavelengths.max():.1f} nm '
                           f'in {min(empty[0]):g}..{max(empty[0]):g} nm')
        if problem is not None:
            self._on_message(problem)
            self.status.setVisible(True)
            return
        layer = self.viewer.layers[name]
        stack = layer.data
        start, step = self.axis.start_spin.value(), self.axis.step_spin.value()
        self._pending = (layer_name(batch_prefix(name), 'virtual', 'channels'), ranges,
                         tuple(layer.scale[-2:]), tuple(layer.translate[-2:]))
        logger.info(f'Virtual channels of "{name}" {stack.shape}: {ranges} nm')

        def task(report):
            """ Sum the ranges in the worker thread. """
            return virtual_channels(stack, start, step, ranges, report=report)

        self._start_task(task, self._add_result, f'Summing {len(ranges)} ranges of {name}...')

    def _add_result(self, result:tuple):
        """ Add the virtual channels as one image layer.

        Parameters
        ----------
        result : tuple
            Output and channel counts of `virtual_channels`.

        """
        image, counts = result
        name, ranges, scale, translate = self._pending
        self.viewer.add_image(image, name=name, scale=(1, *scale), translate=(0, *translate),
                              metadata={'ranges_nm': ranges, 'channels_summed': counts})
        self._on_message(f'Done: {name} {image.shape}, '
                         + ', '.join(f'{n} ch' for n in counts))
        logger.info(f'Added layer "{name}" of shape {image.shape}')


class _AdsResultDialog(_PlotDialog):
    """ Window showing what `utils.adaptive_dye_separation` found.

    Left, the reference spectra with their maxima (vertex of a parabola through
    the highest channel and its neighbours), listed under the plot. Right, the
    energy per principal component; the components above the noise floor
    estimate the number of dyes.

    Parameters
    ----------
    widget : AdaptiveDyeSeparationWidget
        Owner of the result.

    """

    def __init__(self, widget):
        super().__init__(widget, 'Adaptive dye separation', subplots=2)
        self.resize(760, 460)
        self.finish_layout()

    def refresh(self):
        """ Redraw the spectra and the component energies. """
        result = self._widget.result
        if result is None:
            return
        background, text = self.style()
        spectra_axes, energy_axes = self.axes
        spectra_axes.clear()
        energy_axes.clear()
        self.style()

        references = result['references']
        wavelengths = self._widget.axis.wavelengths(len(references))
        maxima = []
        for k in range(references.shape[1]):
            spectrum = references[:, k]
            spectra_axes.plot(wavelengths, spectrum, lw=1.2, label=f'Dye {k + 1}')
            top = int(np.clip(np.argmax(spectrum), 1, len(spectrum) - 2))
            left, centre, right = spectrum[top - 1:top + 2]
            curve = left - 2 * centre + right
            offset = 0.5 * (left - right) / curve if curve < 0 else 0.0
            peak_nm = np.interp(top + offset, np.arange(len(spectrum)), wavelengths)
            peak_value = centre - 0.25 * (left - right) * offset
            spectra_axes.plot(peak_nm, peak_value, 'o', ms=6, color=f'C{k}',
                              markeredgecolor=text, markeredgewidth=0.8)
            maxima.append(f'Dye {k + 1} {peak_nm:.1f} nm')
        spectra_axes.set_xlabel('Wavelength, nm', color=text)
        spectra_axes.set_ylabel('Reference spectrum, unit sum', color=text)
        legend = spectra_axes.legend(fontsize='small', facecolor=background, edgecolor=text)
        for entry in legend.get_texts():
            entry.set_color(text)

        explained = result['explained'][:12]
        rank = np.arange(1, len(explained) + 1)
        energy_axes.semilogy(rank, np.maximum(explained, 1e-12), 'o-', ms=3, lw=1.0, color=text)
        energy_axes.axvline(references.shape[1], color=text, lw=0.8, ls=':')
        energy_axes.set_xlabel('Principal component', color=text)
        energy_axes.set_ylabel('Fraction of the energy', color=text)
        self.canvas.draw_idle()
        self.report.setMessage('\n'.join(self._widget.summary()
                                         + ['Maxima: ' + ', '.join(maxima)]))

    def open_now(self):
        """ Redraw and show the window. """
        self.refresh()
        self.show()
        self.raise_()


class _AdsPcaDialog(_PlotDialog):
    """ Window showing the feature space `adaptive_dye_separation` clustered.

    Left, every shown pixel on the first two principal components, coloured
    by the dye it was clustered into, grey below the thresholds, with a ray
    from the origin along every reference: mixtures of dyes are rays through
    the origin, so a cluster should be a fan around its ray. Right, the
    directions alone - the second and third components divided by the first -
    which is the space the spherical k-means separates: a cluster is a blob
    there and a reference a point at its edge (weak mode) or centre (strong).

    Parameters
    ----------
    widget : AdaptiveDyeSeparationWidget
        Owner of the result.

    """

    def __init__(self, widget):
        super().__init__(widget, 'Adaptive dye separation: PCA and clusters', subplots=2)
        self.resize(820, 460)
        self.finish_layout()

    def refresh(self):
        """ Redraw both views of the feature space. """
        result = self._widget.result
        if result is None:
            return
        background, text = self.style()
        scores_axes, direction_axes = self.axes
        scores_axes.clear()
        direction_axes.clear()
        self.style()

        view = result['view']
        scores, dye, references = view['scores'], view['dye'], view['references']
        below = dye < 0
        scores_axes.scatter(scores[below, 0], scores[below, 1], s=3, color=text, alpha=0.25,
                            lw=0, label='below thresholds')
        reach = np.percentile(np.abs(scores[:, :2]), 99.5)
        for k in range(len(references)):
            color = f'C{k}'
            mine = dye == k
            scores_axes.scatter(scores[mine, 0], scores[mine, 1], s=4, color=color, lw=0,
                                label=f'Dye {k + 1}')
            ray = references[k, :2] / max(np.linalg.norm(references[k, :2]), 1e-12) * reach
            scores_axes.plot([0, ray[0]], [0, ray[1]], color=color, lw=1.6)
        scores_axes.set_xlabel(f'PC1 score ({100 * view["explained"][0]:.1f}% of the energy)',
                               color=text)
        scores_axes.set_ylabel(f'PC2 score ({100 * view["explained"][1]:.1f}%)', color=text)
        legend = scores_axes.legend(fontsize='small', facecolor=background, edgecolor=text,
                                    markerscale=3)
        for entry in legend.get_texts():
            entry.set_color(text)

        # directions: what the clustering on the unit sphere separates
        front = scores[:, 0] > 0
        ratio = scores[front, 1:3] / scores[front, :1]
        for k in range(len(references)):
            mine = dye[front] == k
            direction_axes.scatter(ratio[mine, 0], ratio[mine, 1], s=4, color=f'C{k}', lw=0)
            if references[k, 0] > 0:
                point = references[k, 1:3] / references[k, 0]
                direction_axes.scatter(*point, s=90, marker='X', color=f'C{k}',
                                       edgecolors=text, linewidths=0.8, zorder=3)
        clustered = ratio[dye[front] >= 0]
        if len(clustered):
            low, high = np.percentile(clustered, [0.5, 99.5], axis=0)
            pad = 0.1 * (high - low)
            direction_axes.set_xlim(low[0] - pad[0], high[0] + pad[0])
            direction_axes.set_ylim(low[1] - pad[1], high[1] + pad[1])
        direction_axes.set_xlabel('PC2 / PC1', color=text)
        direction_axes.set_ylabel('PC3 / PC1', color=text)
        self.canvas.draw_idle()
        self.report.setMessage(f'{len(scores)} of the sampled pixels shown, '
                               f'{100 * (~below).mean():.1f}% of them clustered; '
                               f'X marks the reference of each dye')

    def open_now(self):
        """ Redraw and show the window. """
        self.refresh()
        self.show()
        self.raise_()


class AdaptiveDyeSeparationWidget(_TaskWidget):
    """ Dock widget separating the dyes of a stack without their spectra.

    Only the stack and the parameters below enter the result, see
    `adaptive_dye_separation`. The spectral axis, filled by a calibration
    through `_AxisBox`, labels the spectra in the window and the CSV and does
    not touch the computation.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        Viewer instance napari passes to every widget contribution.

    """

    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self._pending = None
        self._dialog = None
        self._pca_dialog = None
        self.result = None
        self._build_ui()
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    def _build_ui(self):
        """ Assemble the source, spectral axis, separation and run sections. """
        source_box = QGroupBox('Source')
        source_form = _form(source_box)
        self.stack_combo = QComboBox()
        self.stack_combo.setToolTip('Any 3D image layer whose first axis is spectral')
        source_form.addRow('Stack', self.stack_combo)
        self.axis = _AxisBox(self.viewer, self.stack_combo)

        separation_box = QGroupBox('Separation')
        form = _form(separation_box)
        self.dyes_spin = _spin(2, 8, 3)
        self.dyes_spin.setToolTip('Number of dyes in the stack')
        self.dyes_spin.valueChanged.connect(self._dyes_changed)
        self._dyes = self.dyes_spin.value()
        form.addRow('Dyes', self.dyes_spin)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['weak', 'strong'])
        self.mode_combo.setToolTip('Reference point of a cluster: its side away from the '
                                   'other clusters (weak) or its centre (strong)')
        form.addRow('Mode', self.mode_combo)
        self.percentile_spin = _spin(0.0, 99.0, 50.0, 5.0, decimals=1)
        self.percentile_spin.setToolTip('Percentile of the pixels above the noise gate '
                                        'under which they are left out of the clustering')
        form.addRow('Signal threshold, %', self.percentile_spin)
        self.gate_spin = _spin(0.0, 20.0, 3.0, 0.5, decimals=1)
        self.gate_spin.setToolTip('Noise levels a pixel must exceed to enter the clustering')
        form.addRow('Noise gate', self.gate_spin)
        self.background_check = QCheckBox('Subtract background')
        self.background_check.setChecked(True)
        self.background_check.setToolTip('Subtract the median spectrum of the darkest 5% of '
                                         'the pixels; needs some empty pixels in the frame')
        form.addRow(self.background_check)
        self.pca_check = QCheckBox('Cluster in PCA components')
        self.pca_check.setChecked(True)
        form.addRow(self.pca_check)
        self.components_spin = _spin(self._dyes, 20, self._dyes)
        self.components_spin.setToolTip('Number of components, at least the number of dyes; '
                                        'used exactly when auto is off')
        self.auto_components_check = QCheckBox('auto')
        self.auto_components_check.setChecked(True)
        self.auto_components_check.setToolTip('Take the components standing above the noise '
                                              'floor, at least the number of dyes; the count '
                                              'chosen is put into the box after the run')
        self.auto_components_check.toggled.connect(self._components_mode)
        self.pca_check.toggled.connect(self._components_mode)
        components_row = QHBoxLayout()
        components_row.addWidget(self.components_spin)
        components_row.addWidget(self.auto_components_check)
        form.addRow('PCA components', components_row)
        self._components_mode()
        self.refine_check = QCheckBox('Refine spectra with NMF')
        self.refine_check.setChecked(True)
        form.addRow(self.refine_check)

        self.run_btn = QPushButton('Separate dyes')
        self.run_btn.clicked.connect(self._run)
        self.spectra_btn = QPushButton('Spectra...')
        self.spectra_btn.clicked.connect(self._open_dialog)
        self.pca_btn = QPushButton('Clusters...')
        self.pca_btn.setToolTip('The pixels on the leading principal components, '
                                'coloured by the dye they were clustered into')
        self.pca_btn.clicked.connect(self._open_pca_dialog)
        self.save_btn = QPushButton('Save CSV...')
        self.save_btn.clicked.connect(self._save_csv)
        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addWidget(self.spectra_btn)
        actions_layout.addWidget(self.pca_btn)
        actions_layout.addWidget(self.save_btn)
        self._assemble([source_box, self.axis, separation_box], [self.run_btn, actions])
        self._set_controls_enabled(True)

    def _dyes_changed(self, dyes:int):
        """ Keep the number of PCA components at or above the number of dyes.

        Parameters
        ----------
        dyes : int
            New number of dyes. A component count that was following the
            number of dyes goes on following it.

        """
        following = self.components_spin.value() == self._dyes
        self.components_spin.setMinimum(dyes)
        if following:
            self.components_spin.setValue(dyes)
        self._dyes = dyes

    def _set_controls_enabled(self, enabled:bool):
        """ Enable the run button, and the result buttons if there is a result. """
        self.run_btn.setEnabled(enabled)
        for button in (self.spectra_btn, self.pca_btn, self.save_btn):
            button.setEnabled(enabled and self.result is not None)

    def _refresh_layers(self, event=None):
        """ Keep the stack combo in sync with the open 3D image layers. """
        self._refill_layer_combo(self.stack_combo,
                                 [layer for layer in self._image_layers()
                                  if layer.data.ndim == 3 and not layer.rgb])
        self.axis.refresh()

    def plot_colors(self) -> tuple:
        """ Background and foreground colours of the current napari theme. """
        try:
            theme = get_theme(self.viewer.theme)
            return theme.background.as_hex(), theme.text.as_hex()
        except Exception:  # an unknown theme must not cost the whole window
            logger.warning(f'Unknown napari theme "{self.viewer.theme}"')
            return 'white', 'black'

    def _components_mode(self, *_):
        """ Enable the component box only when its value is used as it stands. """
        self.auto_components_check.setEnabled(self.pca_check.isChecked())
        self.components_spin.setEnabled(self.pca_check.isChecked()
                                        and not self.auto_components_check.isChecked())

    def summary(self) -> list:
        """ Lines describing the last result: what was used, and any warnings. """
        result = self.result
        components = (f'{result["components"]} PCA components'
                      if result['components'] is not None else 'clustered in the channels')
        return [f'{components}, {100 * result["used"]:.1f}% of the sampled pixels clustered, '
                f'{result["negative"]:.1f}% negative energy, '
                f'two halves differ by {result["stability"]:.1f} deg', *result['warnings']]

    def _run(self):
        """ Validate the selection and start the separation task. """
        from .utils import adaptive_dye_separation

        name = self.stack_combo.currentText()
        dyes = self.dyes_spin.value()
        problem = None
        if name not in self.viewer.layers:
            problem = 'Select a 3D image layer whose first axis is spectral'
        elif len(self.viewer.layers[name].data) <= dyes:
            problem = f'{dyes} dyes need more than {dyes} channels'
        if problem is not None:
            self._on_message(problem)
            self.status.setVisible(True)
            return
        layer = self.viewer.layers[name]
        stack = layer.data
        arguments = {'mode': self.mode_combo.currentText(),
                     'noise_gate': self.gate_spin.value(),
                     'signal_percentile': self.percentile_spin.value(),
                     'subtract_background': self.background_check.isChecked(),
                     'pca_components': (None if not self.pca_check.isChecked() else 'auto'
                                        if self.auto_components_check.isChecked()
                                        else self.components_spin.value()),
                     'refine': self.refine_check.isChecked()}
        self._pending = (batch_prefix(name), tuple(layer.scale[-2:]), tuple(layer.translate[-2:]))
        logger.info(f'Dye separation of "{name}" {stack.shape}: {dyes} dyes, {arguments}')

        def task(report):
            """ Separate the dyes in the worker thread. """
            return adaptive_dye_separation(stack, dyes, report=report, **arguments)

        self._start_task(task, self._add_result, f'Separating {dyes} dyes of {name}...')

    def _add_result(self, result:dict):
        """ Add the abundance and the cluster map as layers.

        Parameters
        ----------
        result : dict
            Return value of `adaptive_dye_separation`.

        """
        batch, scale, translate = self._pending
        self.result = result
        if result['components'] is not None and self.auto_components_check.isChecked():
            self.components_spin.setValue(result['components'])
        name = layer_name(batch, 'ads')
        self.viewer.add_labels(result['clusters'], name=layer_name(batch, 'ads', 'clusters'),
                               scale=scale, translate=translate, visible=False)
        self.viewer.add_image(result['abundance'], name=name, scale=(1, *scale),
                              translate=(0, *translate))
        self._on_message('\n'.join((f'Done: {name} {result["abundance"].shape}',
                                    *self.summary())))
        for warning in result['warnings']:
            logger.warning(f'ADS: {warning}')
        for dialog in (self._dialog, self._pca_dialog):
            if dialog is not None and dialog.isVisible():
                dialog.refresh()
        logger.info(f'Added layer "{name}" of shape {result["abundance"].shape}')

    def _open_dialog(self):
        """ Open the window with the spectra of the last result. """
        if self._dialog is None:
            self._dialog = _AdsResultDialog(self)
        self._dialog.open_now()

    def _open_pca_dialog(self):
        """ Open the window with the clustered feature space of the last result. """
        if self._pca_dialog is None:
            self._pca_dialog = _AdsPcaDialog(self)
        self._pca_dialog.open_now()

    def _save_csv(self):
        """ Write the reference spectra of the last result to a CSV file.

        One row per channel: index, wavelength on the current spectral axis to
        one decimal, and one column per dye in full precision.

        """
        batch = self._pending[0]
        path, _ = QFileDialog.getSaveFileName(self, 'Save the dye spectra',
                                              layer_name(batch, 'ads', 'spectra') + '.csv',
                                              'CSV files (*.csv)')
        if not path:
            return
        references = self.result['references']
        n_lambda, n_dyes = references.shape
        np.savetxt(path, np.column_stack((np.arange(n_lambda), self.axis.wavelengths(n_lambda),
                                          references)),
                   delimiter=',', comments='',
                   header=','.join(['lambda_index', 'wavelength_nm',
                                    *(f'dye_{k + 1}' for k in range(n_dyes))]),
                   fmt=['%d', '%.1f'] + ['%.6g'] * n_dyes)
        self._on_message(f'Saved {path}')
        self.status.setVisible(True)
        logger.info(f'Saved {n_dyes} dye spectra to {path}')
