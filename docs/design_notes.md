# Rationale for parameters and thresholds

The measurements behind the defaults and thresholds of `phase_model_recon.py`, `utils.py` and `_widgets.py`. Docstrings point here in one line. Reference data:

- the bundled sample `qd_mix_batch.tif`: 18 frames, a mixture of QD 525 / 585 / 659 nm;
- a 40-frame series of the same field (`demo_data/QD_mix`);
- wide pattern lines: `wide_lines_validation.md`;
- the spectral axis: `spectral_calibration.md`.

## 1. Reconstruction (`PhaseModelRecon`)

| What | Value on the reference data |
|---|---|
| Fold contrast with the pattern rotated by 180° (`disk.pattern_rotate_180`) | 1.90 against 1.10 without rotation |
| Change of the line period across the frame (`plot_line_model`) | 88–90 px |
| Rms residual of the line family model | 0.67 px; the fine stripes are the quantisation of the simulated skeleton |
| Phase step between frames (`plot_phase_steps`) | −5.5 to −3.9 px, fit −4.79 px |
| Systematic oscillation of the trajectory residual (`plot_phase_trajectory`) | ±1.3 px with a period of ≈ 18.5 frames, within `phase_slack` |
| Rows filled by 40 frames in single-row mode | ≈ 96% |
| `lambda_frame_recon` against the whole stack, bundled sample | 38 MB against 1.5 GB of memory; the channel is bit-identical for every accumulation mode |

## 2. Frame preprocessing

| What | Value |
|---|---|
| `despeckle`, test frame with 1-px lines and 500 bad pixels | the ring changes only the bad pixels; a 3 × 3 median changes ≈ 6000 |
| `despeckle`, time per 2000 × 1500 frame | 0.15 s (`size` 3), 0.21 s (`size` 5); median 0.20 / 0.70 s |

## 3. Gap filling (`max_pooling2d`)

Single-row mode, channel 40 of the stack:

| Series | Coverage | `(4, 1)` | `(12, 1)` |
|---|---|---|---|
| 40 frames | 41.7% | 95.6% | 96.4% |
| 6 frames | 6.4% | 25.4% | — |

- The longest run of empty rows in a column of the 40-frame series is 4, hence a window height of 4.
- The remaining holes are the top and bottom edges, which the series did not reach.

## 4. Console diagnostics

| Function | What was measured |
|---|---|
| `estimate_band_periodicity` | period 88.34 px, step −4.72 px/frame; the period within 0.1% of `PhaseModelRecon`, the drift within 1.5% |
| `residual_pattern_power` | raw frame ≈ 1.9; a good reconstruction 1–2 orders of magnitude lower |
| `peak_drift_map`, `mode_prominence` | the weaker emitter has a prominence of 0.043, a small margin above 0.03 |
| `detect_spectral_peaks` | three peaks on the sample ROIs, spaced 18–19 and 19–21 channels |
| `fit_spectral_peaks` | without bounds the fit failed on 3 of 5 ROIs (two peaks merged); with bounds it is stable on all |

### Peak drift across the field (88 channels, filled 40-frame stack)

- **Raw peaks, `peak_drift_map`:**
  - two modes, at channels 29.25 and 38.25; coverage 77.5%; rms 2.36 channels; range 7.7;
  - the median over columns runs from −1.5 to +2.0 channels, over rows a parabola of ≈ 1.2.
- **The raw slope is an artefact of overlapping emitters:** the two modes have slopes of opposite sign (ratio −4.31).
- **`peak_drift_unmixed`, three components:**
  - slopes +7.05 / +7.93 / +8.52 channels per 1000 columns;
  - stable to the third digit for tiles of 48 / 64 / 96;
  - the 18- and 40-frame series agree within 0.3%;
  - shifting the crop by 83 columns changes the slope by ≈ 7%, the raw one threefold;
  - range ≈ 11.7 channels over 1500 px.
- **Slope ratio** 1 : 1.12 : 1.21. A geometric error would give 1 : 1.62 : 2.38, so this is almost a rigid shift of the axis.
- **Angle:** the fitted `theta` is ≈ 0.001°, while a rigid shift corresponds to ≈ 0.34°. The reconstruction does not measure the dispersion direction of the prism.
- **`fit_column_drift` on a synthetic stack:** the slope is recovered within 0.14%, 99.7% of the range is removed.
- **Not checked:** a second sample with the same emitters.

## 5. Adaptive dye separation

| What | Value |
|---|---|
| Cluster fraction of the `weak` mode | 5%, chosen on synthetic spectra |
| Warning thresholds (`ADS_*_DEG`, `ADS_NEGATIVE_PERCENT`) | chosen on synthetic spectra; necessary conditions only: a mixture at 40 counts passed them with a spectrum off by 15° |
| Eigenvalues of the colour noise of the reconstruction | the leading ones are 5–10 times the median of the lower half, hence `ADS_AUTO_FACTOR` = 20 |
| `auto` components on QD stacks (3 dyes, 1–4 passes) | 7; with 3 the dyes merge, with ≥ 6 they separate |
| References on the bundled sample | mixtures, shifted from the known emitters; narrow frame bands and `correct_column_drift` do not help, the cause is not established |
