""" Napari plugin for hyperspectral reconstruction of spinning-disk spectral
microscopy data.

The reconstruction lives in `phase_model_recon` (`PhaseModelRecon` and its
numba kernels), every other computation in `utils`; `_widgets` holds the
Qt dock widgets and the sample data the plugin contributes.

"""

from ._widgets import (AdaptiveDyeSeparationWidget, LambdaRGBWidget, PhaseModelReconWidget,
                       PostProcessingWidget, SpectraWidget, VirtualChannelsWidget,
                       qd_mix_rois_sample, qd_mix_sample)

__all__ = ['AdaptiveDyeSeparationWidget', 'LambdaRGBWidget', 'PhaseModelReconWidget',
           'PostProcessingWidget', 'SpectraWidget', 'VirtualChannelsWidget',
           'qd_mix_rois_sample', 'qd_mix_sample']
