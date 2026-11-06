# Spectral calibration: sampling, resolution and uniformity of the axis

Report of 6 November 2026. Data:
- reconstructions of the plate `data/ChromaGreen` (p80, p160, p320) and of the QD mixture `data/QDmix2_aggregate_1000m-75u-25u-g20` from `docs/wide_lines_validation.md`;
- separate scripts (section 8).

The procedure for the nm axis is in the README, section "Spectral calibration". This report covers what sets the number of channels, what their real resolution is, and which calibration steps are worth adding.

## Conclusions

1. **89 channels is the rounded median period of the pattern lines on the camera (88.5 rows).** One channel ≈ one camera row: this is the sampling of the detector, not the spectral resolution.
2. **Real spectral resolution at K = 1:** σ ≈ 4.5–5 channels (16–18 nm), FWHM ≈ 10–12 channels (≈ 40 nm).
   - An order-of-magnitude estimate from the available data: upper bounds from the plate and from the QD peak at 528 nm, plus the spatial window transferred to the spectrum.
   - 89 channels sample such a spectrum 5–6 times more densely than the Nyquist limit requires (a step of FWHM/2 ≈ 5–6 channels).
3. **The plate alone gives only an upper bound** (FWHM ≤ 53 nm), because its spectrum is broad and smooth. But the plate series at different K proves that shifting the window by a row shifts the spectrum by exactly a channel, so the spatial window along the rows can be transferred to the spectrum.
4. **The wavelength axis is not uniform across the field.** On a uniform plate the band profile shifts as a whole by ±2.5 channels (≈ ±9 nm), and the map is the same in all three series. A single axis for the whole field is wrong at the edges by about half the FWHM.
5. **The available data contain no isolated QDs.** An exact measurement of the resolution needs a narrow-line source. Calibration steps: section 6.

## 1. Why 89 channels

| What | Value | Source |
|---|---|---|
| Default `n_lambda` | `round(median_period)` | `PhaseModelRecon.run`; docstring: "resamples the band without loss of resolution" |
| Median line period | 88.54 rows → 89 | fit of the plate and of the QD mixture |
| Local period across the field (1–99%) | 88.1–89.8 rows | line model `line_coef`, plate |
| Rows per channel | 0.99–1.01 | local period / 89 |
| Band | the gap between neighbouring lines, one period | `_sample_bands` |
| Samples | centres of 89 bins of the normalised coordinate `t = (i + 0.5) / 89`, bilinear interpolation | `_sample_bands` |
| Occupied part of the band | 60 of 89 channels above 5% of the maximum (plate) | `docs/wide_lines_validation.md`, section 4 |
| nm per channel | 3.65 | calibration of the QD mixture |

- **Why one period.** The spectrum of one row of pinholes must fit before the next line, so a band spans exactly one period. On the plate the rest of the band is dark, so the bands do not overlap.
- **What the channel count depends on:** the magnification between the disk and the camera and the pixel size, i.e. how many rows a period spans. On the bundled sample it is 84.
- **Why exactly one channel per row:**
  - denser sampling adds nothing, it only interpolates;
  - sparser sampling discards detector samples;
  - but at the real resolution 22 channels lose nothing for unmixing (`docs/wide_lines_validation.md`, section 7.2).
- **A channel is a fraction of the band, not a fixed row.** Where the local period differs by 1%, channel 60 is shifted by ±0.6 rows. This effect is smaller than the non-uniformity of the axis from section 4 and correlates with it weakly (r = +0.37).

## 2. What sets the spectral resolution

Model: a band pixel receives the light that passed the line window centred at c, at the coordinate `k = d(λ) + (x − c)`. One row of shift x equals one channel (section 3). So any window along the rows blurs the spectrum by the same number of channels.

| Contribution | Size at K = 1 | Dependence on K |
|---|---|---|
| Aperture: pinhole ≈ 17.5 rows along the rows | σ ≲ 4.9 channels | none |
| Line motion during the exposure: a box of W channels | σ = W/√12 = 1.4 (W = 4.72) | W = 4.72·K |
| Optics from the disk to the camera along the rows | no more than the edge blur along the columns (σ 4.7 px, including the intrinsic edge width) | none |
| Error of the phase and of the line model | a shift of c by δ rows shifts the spectrum by δ channels; part of σ_add of `docs/wide_lines_validation.md`, section 7.3 | — |
| Axis shift across the field (section 4) | σ 1.2 channels, only for spectra averaged over large areas | none |

**Point and extended objects.** If light passes the pinhole at the point where the source lies (a point spread function narrower than the pinhole), a point source gives a sharp spectrum in every row of the reconstruction, only shifted by `x − c`. The spectrum then shifts noticeably at object edges.
- **Prediction for the full shift:** window σ_g = 5, edge σ_e = 4.7, shift at half height `σ_g² · φ(0) / (0.5 · √(σ_g² + σ_e²))` ≈ 2.9 channels per edge, i.e. ≈ 6 channels between opposite edges.
- **Measured:** 1.9 channels between the top and bottom edges against 0.4–0.7 in the control (left and right edges), i.e. 1.2–1.5 channels (`docs/wide_lines_validation.md`, section 7.4).
- **Conclusion:** the shift is 20–25% of the full one. Light mostly fills the pinhole, so the aperture blur is the same for point and extended objects. An isolated point will not give a sharper spectrum.

## 3. Estimates from the available data

| Source | Method | Result |
|---|---|---|
| Plate, p80 | steepest slope of the band profile: `σ ≤ 1 / (√(2π) · max(abs(dy/dk)))` | σ ≤ 6.2 channels (23 nm), FWHM ≤ 14.6 channels (53 nm) |
| Plate, p160 / p320 | the same | FWHM ≤ 15.1 / 17.5 channels (55 / 64 nm) |
| Plate, K series | profile at K passes = sum of K copies of p80 shifted by 4.66–4.76 rows, residual ≤ 0.3% | a shift by a row = a shift by a channel; the motion contribution is known exactly |
| QD, peak at 528 nm | FWHM of the narrowest peak of the mean spectrum of pure pixels | ≤ 13.5 channels (49 nm), σ ≤ 5.64 |
| QD, aggregate edges | blur along the rows beyond the columns, single-row writing, K = 1 | σ_ap ≈ 4.9 rows → σ ≲ 4.9 channels |
| QD, spectrum shift at the edges | 1.2–1.5 channels against ≈ 6 for the full shift | the aperture blur applies to point objects too |
| **Together** | | **σ ≈ 4.5–5 channels (16–18 nm), FWHM ≈ 10–12 channels (≈ 40 nm)** |

- **The plate bound** holds because blur barely lowers the peak of a broad spectrum (profile FWHM 27 channels against σ ≈ 5).
- **Consistency:** at this σ the intrinsic FWHM of the QD 528 nm ensemble comes out at 22–29 nm, which is plausible.
- **nm scale:** taken from the linear QD calibration; for a prism the nm per channel may vary along the band.

**Resolution against K.** Model: the aperture Gaussian (σ_ap without motion 4.0–5.0 channels) convolved with a box of W = 4.72·K channels; FWHM computed numerically.

| K | W, channels | σ, channels | FWHM, channels | FWHM, nm |
|---|---|---|---|---|
| 1 | 4.7 | 4.2–5.2 | 10.0–12.2 | 36–45 |
| 2 | 9.4 | 4.8–5.7 | 11.8–13.6 | 43–50 |
| 3 | 14.2 | 5.7–6.5 | 14.9–16.1 | 54–59 |
| 4 | 18.9 | 6.8–7.4 | 19.0–19.6 | 70–72 |
| 5 | 23.6 | 7.9–8.5 | 23.6–23.8 | 86–87 |
| 6 | 28.3 | 9.1–9.6 | 28.3–28.4 | 103–104 |

- **Growth of the instrument blur:** by 40% already at K = 3; from K = 4 its FWHM equals W (the blur becomes a box).
- **Why the QD peaks grow more slowly** (+11% at 3 passes, `docs/wide_lines_validation.md`, section 7.1): the intrinsic peak width adds to the blur in quadrature and dilutes it.

## 4. Uniformity of the axis across the field

**Method.** Plate, blocks of 150 × 150 pixels, the mean spectrum of every block. For each block: the centroid, the half-height positions on both slopes and the apex (parabola over 9 channels).

| Series | Blocks | Centroid: sd / range, channels | Block FWHM, channels |
|---|---|---|---|
| p80 | 182 | 1.19 / 4.97 | 26–29 |
| p160 | 182 | 1.18 / 4.81 | 27–29 |
| p320 | 182 | 1.20 / 5.02 | 29–32 |

Centroid map of p80 in channels (every second block; full-frame coordinates, pixels):

| Row \ column | 1125 | 1425 | 1725 | 2025 | 2325 | 2625 | 2925 |
|---|---|---|---|---|---|---|---|
| 525 | 50.2 | 48.2 | 46.6 | 45.7 | 45.8 | 46.4 | 47.5 |
| 825 | 50.3 | 48.7 | 47.4 | 46.7 | 46.8 | 47.4 | 48.6 |
| 1125 | 50.6 | 49.1 | 48.1 | 47.5 | 47.8 | 48.4 | 49.5 |
| 1425 | 50.6 | 49.4 | 48.7 | 48.3 | 48.7 | 49.2 | 50.2 |
| 1725 | 50.1 | 49.3 | 48.8 | 48.6 | 48.9 | 49.5 | 50.5 |
| 2025 | 49.0 | 48.5 | 48.3 | 48.2 | 48.7 | 49.4 | 50.1 |
| 2475 | 46.1 | 46.1 | 46.2 | 46.4 | 47.0 | 47.5 | 48.1 |

- **The map is reproducible:** p160 and p320 give the same map within 0.3 channels, so it is not noise.
- **The shift is rigid:**
  - both slopes and the apex move together (r = 0.91–1.00);
  - sd separately: rise 1.34, fall 1.17, apex 1.25 channels;
  - the profile width barely changes (sd 0.57 channels).
  So the axis itself shifts, not the shape of the spectrum.
- **Check on QDs** (only blocks with aggregates):
  - the 590 nm peak follows the plate map (r = +0.81, slope 1.01, 9 blocks);
  - the 666 nm peak does not (r = −0.27, 15 blocks), but the map range in those blocks is only 1.8 channels.
  The result is inconclusive.
- **The bundled sample** confirms the rigid shift. The peaks of the three QDs in ROI 4 lie 9.3 / 10.0 / 10.9 channels further than in ROI 1, in ROI 2 by 2.6 / 3.1 / 2.6 channels; all three peaks shift almost equally. Hence a joint calibration over the four ROIs has a residual of 14.7 nm rms (script `analysis/yaml2/sample_fit.py`).
- **Possible causes, not established:**
  - the line model departs from the real start of the band depending on the position in the field;
  - curvature of the spectral line in the spectrograph optics;
  - a non-uniform plate.
- **Consequences:**
  - a single axis for the whole field is wrong at the edges by up to ±2.5 channels (≈ ±9 nm);
  - reference spectra from large areas get an extra blur of σ ≈ 1.2 channels.
  For QD peaks this is already checked by `Peak drift...` in the README.

## 5. Isolated quantum dots

**The available QD series contains no point sources.**
- **Search:** band-pass filter 1.5 / 6 pixels, threshold 8 σ of the noise, area at half height < 60 pixels, isolation.
- **Result:** 426 maxima, only 2 candidates among them. Both turned out to be horizontal artefacts: 14–22 pixels wide along the columns against 2–4 along the rows.
- **Aggregates** have an area at half height of about 2000 pixels.

What sparse point sources would give:

| Quantity | What it shows |
|---|---|
| Profile along the rows | the window of the aperture and the line directly, without the intrinsic edge width |
| Profile along the columns | the point spread function of the optics |
| Shift of the spectrum centroid against the row (0…1 channel per row) | the fraction of the full shift of section 2 |
| Spectrum of an isolated QD | narrower than the ensemble, so a tighter upper bound; not sharper than the aperture allows (section 2) |

- **QD blinking:** every row of the reconstruction comes from a different frame, so blinking turns into row stripes. Many points must be averaged, or particles smaller than the resolution that do not blink must be used.
- **Density:** the spacing of points along the rows must exceed the window, i.e. be more than ≈ 20 + W rows.

## 6. Proposed calibration steps

| Step | Sample | What to obtain | Idea for the plugin |
|---|---|---|---|
| 1. Sampling | any | period in rows and its spread across the field; nm per channel | exists: `n_lambda` = period; show the spread in the log |
| 2. Narrow line, extended source | a plate with a narrow band-pass filter (1–10 nm) in the emission path, laser reflection from a mirror, or a spectral lamp through a diffuser | instrument blur: σ, FWHM, shape; with ≥ 3 lines the non-linearity of the nm axis | a widget that fits a peak in every block of the reconstruction and saves the FWHM, the shift map and the axis to CSV |
| 3. Spectral edge (without a narrow filter) | a plate with a steep filter (long-pass, short-pass or notch) | blur as the derivative of the edge profile | the same widget, erf fit |
| 4. Spectral uniformity of the field | uniform plate | map of the axis shift δ(x, y) in channels | post-processing: shift the spectrum of every pixel by −δ |
| 5. Shift at several λ | source of step 2 at ≥ 2 wavelengths | whether the shift is the same for all λ (condition of step 4) | — |
| 6. Dependence on K | step 2 or a plate at K = 1, 2, 4 | check of `σ_K² = σ_1² + (W_K² − W_1²)/12` and of the box shape | blur reference spectra with the measured function |
| 7. Point sources | sparse particles smaller than the resolution | window along the rows, point spread along the columns, fraction of the shift | — |
| 8. Consistency | results of steps 2 and 7 | σ of the spectral window and σ of the spatial window along the rows must agree | — |

**What to store and where:**
- **The nm axis and the shift map**: per session, because they depend on the alignment and the sample. By the project rule they do not go into `setup_params.yaml`.
- **The blur at K = 1 and the channel count** are set by the hardware, so they are in `setup_params.yaml`:
  - `optics.spectral_fwhm` = 11 channels (estimate of section 3, to be replaced by a measurement of step 2);
  - `optics.n_lambda` = 89. The period of this instrument is 88.4–88.6 rows, so `auto` gives 88 or 89 by series (bundled sample 88.38 → 88, plate and QD 88.54–88.62 → 89), while a fixed count keeps the channels of different series comparable.
- **For the user:** the reconstruction widget shows `Line width` and `Spectral resolution` for the chosen `Pattern passes` (FWHM by the model of section 3, function `line_fwhm`). Likewise, a Leica λ-scan sets the detection band and the step separately.
- **Default axis step** in the widgets: 3.51 nm, the calibration of the bundled sample (3.55 nm at 88 channels) converted to 89 channels (× 88/89).

## 7. Limitations

- **An estimate, not a measurement:** the data contain no narrow-line source, so the resolution comes from bounds and from the spatial window (order of magnitude).
- **nm scale:** borrowed from the linear QD calibration (3.65 nm per channel); the non-linearity of the dispersion is not measured.
- **Uniformity of the plate** across the field is assumed, not checked.
- **The check of the shift map on QDs** covers only areas with aggregates.
- **The point spread of the optics along the rows** is not measured separately: the blur along the columns includes the intrinsic edge width of the aggregates.

## 8. Scripts

The scripts are in `analysis`; the run order is in `analysis/README.md`.

| Script | What it does |
|---|---|
| `analysis/plate/slope_bound.py` | upper bound from the steepest slope of the plate profile |
| `analysis/plate/field_axis.py` | centroid and FWHM of the plate by blocks, map, relation to the local period |
| `analysis/plate/rigid_shift.py` | slopes and apex by blocks: rigid shift or change of shape |
| `analysis/res/field_qd.py` | QD peaks against the plate map |
| `analysis/res/spots.py` | search for compact isolated spots |
| `analysis/res/psf_vs_k.py` | σ and FWHM of the blur against K |
| `analysis/res/edges.py`, `analysis/res/analyse4.py` | spatial window and spectrum shift at the edges (from `docs/wide_lines_validation.md`) |
| `analysis/yaml2/sample_axis.py`, `analysis/yaml2/sample_fit.py` | calibration of the bundled sample at 88 and 89 channels, ROI peaks |
