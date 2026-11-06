""" Napari plugin for hyperspectral reconstruction of spinning-disk spectral
microscopy data.

The processing itself lives in `phase_model_recon` (the `PhaseModelRecon`
batch reconstruction) and in `utils` (post-processing and spectral analysis);
`_widgets` only wraps them into the dock widgets and the sample data the
plugin contributes.

"""

from ._widgets import (AdaptiveDyeSeparationWidget, LambdaRGBWidget, PhaseModelReconWidget,
                       PostProcessingWidget, SpectraWidget, VirtualChannelsWidget,
                       qd_mix_rois_sample, qd_mix_sample)

__all__ = ['AdaptiveDyeSeparationWidget', 'LambdaRGBWidget', 'PhaseModelReconWidget',
           'PostProcessingWidget', 'SpectraWidget', 'VirtualChannelsWidget',
           'qd_mix_rois_sample', 'qd_mix_sample']
