""" Reference run of `PhaseModelRecon` on the bundled sample, shared by the
golden-file generator and the regression test. """
import hashlib
from pathlib import Path

import numpy as np
import tifffile

from spectra_spin_napari import _widgets as w

DATA = Path(__file__).parent / 'data'


def sample_frames() -> np.ndarray:
    """ The bundled sample batch, as float32 frames like a file read gives. """
    return tifffile.imread(str(w.SAMPLE_PATH)).astype(np.float32)


def summary(recon) -> dict:
    """ Fit results and digests of the stacks of a finished run. """
    out = {'phase': recon.phase, 'line_coef': recon.line_coef}
    for method in ('mean', 'max'):
        stack = recon.lambda_stack_recon(method)
        out[f'{method}_sha256'] = np.array(hashlib.sha256(stack.tobytes()).hexdigest())
    out['hits_sha256'] = np.array(hashlib.sha256(recon.lambda_hit_count.tobytes()).hexdigest())
    return out


def run(line_passes) -> dict:
    """ Reconstruct the sample with the current module and summarise it.

    Parameters
    ----------
    line_passes : int or None
        `PhaseModelRecon` ``line_passes``.

    Returns
    -------
    dict
        See `summary`.

    """
    from spectra_spin_napari.phase_model_recon import PhaseModelRecon
    recon = PhaseModelRecon(sample_frames(), w.bundled_pattern(), line_passes=line_passes)
    recon.run()
    return summary(recon)


def save(path, runner):
    """ Write the golden file `path` from ``runner(line_passes)`` for both modes. """
    results = {f'{key}_{name}': value for key, passes in (('single', None), ('one', 1))
               for name, value in runner(passes).items()}
    np.savez_compressed(path, **results)
    print(f'saved {path}, {path.stat().st_size / 1024:.0f} kB')


if __name__ == '__main__':
    # regenerate only on purpose: the golden file is what a refactor is held to
    save(DATA / 'golden_sample.npz', run)
