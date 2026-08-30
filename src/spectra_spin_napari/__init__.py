""" Napari plugin for hyperspectral reconstruction of spinning-disk spectral
microscopy data.

The processing itself lives in `phase_model_recon` (the `PhaseModelRecon`
batch reconstruction) and in `utils` (post-processing helpers); `_widgets`
only wraps them into the two dock widgets the plugin contributes.

"""

from ._widgets import (PhaseModelReconWidget, PostProcessingWidget, SpectraWidget,
                       qd_mix_rois_sample, qd_mix_sample)

__all__ = ['PhaseModelReconWidget', 'PostProcessingWidget', 'SpectraWidget',
           'qd_mix_rois_sample', 'qd_mix_sample']
