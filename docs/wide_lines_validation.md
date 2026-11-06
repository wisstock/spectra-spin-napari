# Validation of wide pattern lines: ChromaGreen plate and disk simulation

Report of 6 November 2026. Data: `data/ChromaGreen`, `data/simulation` and `data/QDmix2_aggregate_1000m-75u-25u-g20`, real long pulses. Scripts: section 9.

## Conclusions

1. **The simulation and the plate agree in shape but differ in scale by a factor of two.**
   - The band profile of the plate at K passes is exactly the sum of K copies of the one-pass profile, shifted by one pass each. Residual 0.2–0.3%. The simulation behaves the same way: the line width in `hist` grows linearly with K.
   - One pass (80 µs pulse) moves a line on the plate by 4.72 rows, i.e. 1/18.8 of the period. In the simulation it is 2.36 rows, half as much. The likely cause: the disk speed in the notebook (`rpm = 55.555/2*60`) is half the real one, because on the plate a pattern period passes in 1.50 ms and in the simulation in 3.0 ms.
2. **Reconstruction with a line width of K passes works** without elaborate unmixing, if the width is set in rows: `W = K × 4.72`, not K pixels.
3. **Limit of the spectral broadening:**
   - K ≤ 3: the broadening is negligible;
   - K = 4: the broadening is noticeable, but unmixing works with spectra taken at the same K;
   - K = 5–6: reference spectra must be blurred to the same K, and blind ADS on the QD mixture already finds wrong peaks.

   Neither deconvolution nor model-based reconstruction is needed: it is enough to match the reference spectra to K. These limits concern the spectrum only; the spatial cost of K is in point 5.
4. **The number of spectral channels does not depend on K and is not a resolution.** It is the sampling step: one channel per row of the period. 89 channels are redundant already at K = 1: a pixel spectrum has 14–16 independent values and ≈ 5 resolved bands. Line motion blurs the spectrum by W channels, not by K. The width of the QD peaks grows by 4–45% up to 5.3 passes, not K-fold (section 7).
5. **Along the rows the line window blurs both space and spectrum.**
   - **Why:** this camera coordinate is shared by position and wavelength, so the row of the window the light came from is not recorded. The window is the aperture of ≈ 17.5 rows plus the pass W.
   - **How much:** edges along the rows are blurred more than along the columns by σ = 5.9 rows at K = 1 (mostly the aperture) and by 6.6 / 7.7 / 8.9 / 11.8 rows at 1.8 / 3.0 / 4.1 / 5.3 passes. This is not K-fold.
   - **Writing into the width W** adds one more W contribution to the blur.
   - **Along the columns** detail is preserved (section 7.3).

## Data

| Series | Pulse, µs | K | Step label | Frames | Gain | Frame exposure | Phase step, rows | Periods covered |
|---|---|---|---|---|---|---|---|---|
| `G20_p80_ph25_exp72100` | 80 | 1 | 25 | 120 | G20 | 721 ms | 1.40 | 1.9 |
| `G30_p160_ph150_exp18025` | 160 | 2 | 150 | 60 | G30 | 180 ms | 8.83 | 5.9 |
| `G30_p320_ph300_exp18025` | 320 | 4 | 300 | 31 | G30 | 180 ms | 17.71 | 6.0 |

- **Acquisition:** pulsed illumination, the pulse repeats every 3 ms during the frame exposure, as in `disk_simulation.ipynb`.
- **Name labels:** `ph` is the shift of the pulse start between frames in µs; `exp` is the frame exposure in units of 0.01 ms.
- **Reconstruction:** crop rows 300..2700, columns 900..3200; 89 channels, period 88.54 rows in all three series.
- **nm scale:** taken from the calibration of the QD mixture (3.65 nm per channel), so the nm values of the plate are approximate.

## 1. Simulation against plate

| Quantity | Simulation | Plate |
|---|---|---|
| Pattern period | 46.0 px | 88.54 rows |
| Line shift in 80 µs | 1.23 px = 0.0267 period | 4.72 rows = 0.0533 period |
| Time of one period | 3.0 ms | 1.50 ms |
| 50 µm aperture | 9.1 px = 0.198 period (≈17.5 rows) | not measured directly |
| Line width in `hist`, K = 1 / 2 / 4 | 1 / 2 / 5 px | not applicable |
| Line σ in `img`, K = 1 / 2 / 4 | 2.40 / 2.47 / 2.75 px | not applicable |
| Band profile σ, K = 1 / 2 / 4 | not applicable | 12.51 / 12.61 / 13.30 rows |

The band profile on the plate includes the dye spectrum, so it grows relatively more slowly than the line in the simulation. Both sources show that up to K = 4 the line width is set mostly by the aperture: in the simulation σ grows by only 15%. The reconstruction of the QD mixture agrees: at K = 1 edges along the rows are blurred more than along the columns by σ ≈ 5 rows (section 7.3).

The shift per pass was measured in two independent ways:

| Way | Shift per 80 µs, rows |
|---|---|
| Phase step of p160 and p320 (8.83 / 150 µs, 17.71 / 300 µs) | 4.72 |
| Phase step of p80 (1.40 / 25 µs) | 4.49 |
| Fit of the p160 profile with two copies of p80 | 4.66 |
| Fit of the p320 profile with four copies of p80 | 4.76 |

The p80 step is 5% smaller than in the other two series. Whether this is an error of the 25 µs label or something else is not established.

## 2. Band profile: K shifted copies

| Series | Model | Residual (rms of the profile variance) |
|---|---|---|
| p160 | 2 copies of p80, shift 4.66 rows | 0.3% |
| p160 | no broadening | 1.9% |
| p320 | 4 copies of p80, shift 4.76 rows | 0.2% |
| p320 | no broadening | 9.1% |

The window centre found by the phase fit departs from the p80 centre by 0.11 rows (p160) and 0.80 rows (p320). The fit follows the centre of the wide line, so writing symmetrically around the phase is correct.

## 3. Coverage and writing into the line width

Single-row writing puts every band into one row. Writing into the line width puts it into `W = K × 4.72` rows around the centre and averages the overlaps. Area: 600 columns, the full frame height except 200 rows at each edge.

| Series | W, rows | `phase_filling`, single row | With the real W | Windows per phase | Rows written: 1 row → width W |
|---|---|---|---|---|---|
| p80 | 4.7 | 0.895 | 1.000 | 6.4 | 85.9% → 96.5% |
| p160 | 9.4 | 0.273 | 1.000 | 6.4 | 26.2% → 96.4% |
| p320 | 18.9 | 0.077 | 1.000 | 6.6 | 7.3% → 96.3% |

The remaining 3.5% of the rows are the frame edges that the bands do not reach.

Quality on a uniform plate (single row → width W; gaps filled with `interpolate_zero_gaps` in both):

| Series | Row stripes | Jitter of the spectrum centroid, channels | Angle of a pixel spectrum to the mean |
|---|---|---|---|
| p80 | 2.01% → 0.60% | 0.230 → 0.231 | 4.12° → 2.50° |
| p160 | 1.22% → 0.82% | 0.168 → 0.127 | 3.81° → 2.87° |
| p320 | 1.59% → 2.03% | 0.136 → 0.157 | 2.30° → 2.35° |

- **p80 is a wide line too:** 4.7 rows against the one written in single-row mode. Writing into the width cut the stripes threefold.
- **p320:** the stripes grew slightly. The cause (rounding of the window to rows, an inexact W, frame brightness fluctuations) is not established.

## 4. Spectral broadening up to K = 6

**Plate.** For K = 3, 5, 6 the profiles are predicted as sums of p80 copies; for K = 2 and 4 they are measured.

| K | W, rows | FWHM, channels (nm) | σ, channels | Angle to K = 1 | Measured angle |
|---|---|---|---|---|---|
| 1 | 4.7 | 27 (99) | 12.51 | 0° | 0° |
| 2 | 9.4 | 28 (102) | 12.73 | 0.98° | 0.94° |
| 3 | 14.2 | 29 (106) | 13.09 | 2.53° | not acquired |
| 4 | 18.9 | 31 (113) | 13.57 | 4.55° | 5.09° |
| 5 | 23.6 | 33 (120) | 14.17 | 6.93° | not acquired |
| 6 | 28.3 | 35 (128) | 14.86 | 9.57° | not acquired |

At K = 4 the measured broadening is slightly larger than predicted: the prediction is off by 2.5°. The mean spectrum of the reconstruction gives the same angles (1.0° at K = 2 and 5.1° at K = 4) whatever the writing mode. The p80 profile at the 5% level spans 60 of 89 channels, so at K = 6 the band will span ~84 channels and approach the neighbouring one.

**Calculation for the spectra of the QD mixture** (528 / 590 / 666 nm), blurred to K passes:

| K | Angles between dyes | Condition number | Noise gain | Share error, references at K = 1 |
|---|---|---|---|---|
| 1 | 64.5 / 58.4 / 86.7° | 2.32 | 1.00× | 0% |
| 2 | 63.0 / 57.3 / 86.5° | 2.42 | 1.04× | 1.0% |
| 3 | 60.7 / 55.4 / 86.1° | 2.59 | 1.12× | 2.7% |
| 4 | 57.9 / 52.9 / 85.3° | 2.88 | 1.23× | 4.8% |
| 5 | 54.8 / 49.8 / 84.0° | 3.29 | 1.40× | 7.4% |
| 6 | 51.7 / 46.5 / 82.0° | 3.84 | 1.62× | 10.1% |

With spectra of the same K the error is not systematic; only the noise gain remains.

**Real QD mixture.** The frames of this series have a 75 µs pulse (line ≈4.4 rows) and a step of 1.374 rows. A sum of K' consecutive frames corresponds to a longer pulse. Comparison with K = 1 of the same frames:

| K' | Line width, rows | Passes | References K = 1 | References blurred to W | Blind ADS | ADS peaks, nm |
|---|---|---|---|---|---|---|
| 4 | 8.5 | 1.8 | 2.7% | 2.9% | 3.0% | 528 / 590 / 666 |
| 8 | 14.0 | 3.0 | 3.5% | 3.5% | 5.1% | 528 / 586 / 666 |
| 12 | 19.5 | 4.1 | 5.1% | 4.1% | 5.8% | 531 / 615 / 666 |
| 16 | 25.0 | 5.3 | 8.0% | 5.8% | 10.3% | 539 / 626 / 670 |

For comparison: the dye shares of two independent sharp passes of this series differ by 5.6%.

## 5. Can elaborate unmixing be skipped

| K | Width, rows | Edge blur along the rows beyond the columns, σ in rows (section 7.3) | Recommendation |
|---|---|---|---|
| 1–3 | 4.7–14 | 5.9 → 7.7 | write into the width W; references may come from K = 1; blind ADS works |
| 4 | 19 | 8.9 | write into the width W; take references at the same K or blur them to it |
| 5–6 | 24–28 | 11.8 (5.3 passes) | write into the width W; references must be blurred to K; blind ADS unreliable; at the limit of the margin between bands |

No elaborate reconstruction is needed at any of these K. MLEM and model-based reconstruction made real data worse, while simple writing into the width W gives true coverage and no worse spectra. Components inside a line cannot be separated from one frame. Up to K ≈ 2 the blur along the rows is set mostly by the aperture (≈ 17.5 rows from the disk geometry); at K ≈ 3 the aperture and line contributions are equal. So this limit of spatial resolution exists already: at K = 1 edges along the rows are blurred by σ ≈ 5–6 rows more than along the columns (section 7.3).

## 6. Implementation

In `PhaseModelRecon`:
- parameters `line_passes` (1..6, default 1; `None` restores the single-row algorithm bit for bit) and `pass_width` (default 0.0533 period);
- the attribute `line_width` in rows, used by `_phase_filling`, `_accumulate_bands` and `plot_phase_cost_map`;
- in the reconstruction widget: the fields `Pattern passes` and `Single-row lines (original algorithm)` in the *Pattern line* box and `Pass width, period` in the advanced parameters.

Requirements the implementation follows:

- **Line width in rows:** `W = K × w₁`, where `w₁ = 4.72` rows for this instrument, i.e. 0.0533 period. It is set as a fraction of the period, so that it does not depend on the crop. Alternatively `w₁` can be estimated from the data by the phase step, if the pulse start shift in µs is known.
- **K = 1 is not one row:** even at 80 µs the line is 4.7 rows wide, so the width matters for ordinary series too.
- **Write centre:** symmetric around the found phase; the fit follows the window centre (offset up to 0.8 rows at K = 4).
- **`phase_filling`:** computed with the same `W`.
- **Window overlaps:** averaged. In these series every phase is covered by about 6 windows.
- **Reference spectra for unmixing:** at K ≥ 4, convolved with K shifted copies (or a box) of width `W − w₁`.
- **ADS on real QD data:** `PCA components` at least N + 3.

## 7. Spectral resolution, channel count and the spatial cost of K

Question: should the number of spectral channels (one per row of the period, here 89) be reduced in proportion to K, since a wide line mixes the spatial and spectral components?

**Model.** A band pixel collects the light from the points x swept by the line window centred at c during the exposure, and puts it at the band coordinate `k = d(λ) + (x − c)`. Hence:
- **the spectrum** is blurred by the window width, i.e. by W channels (channel ≈ row), not by K;
- **at object edges** along the rows the spectrum centroid shifts by about `σ²_window · ∂ ln I / ∂y`; along the columns it does not shift;
- **space:** the value is assigned to row c, and the position of the light inside the window is lost. So along the rows space is blurred by the window (aperture plus W), and writing into the width W, which copies one value into W rows, adds another W contribution. Along the columns the camera coordinate is purely spatial and detail is preserved;
- **measurement budget:** for the same pattern travel there are K times fewer frames. The tests show which axis takes this loss.

**Data.** QD mixture: 120 frames of 75 µs, step 1.374 rows, crop rows 205..2691, columns 860..3166.
- **K emulation:** sums of K' neighbouring frames, divided by K'.
- **Reconstruction:** `lambda_stack_recon('mean')`, `line_passes=1`, `pass_width = W / 88.54`; the single-row variant `line_passes=None`. 89 channels of 3.65 nm.
- **Noise:** estimated from the reconstructions of the even and odd frames of K1 (60 frames each).
- **Dye pixels** were selected once, by unmixing K1 with the ADS references (`PCA components` = 6), and used identically in all series.
  - Dye share > 0.85, brightness in the top 20%.
  - Pixel counts: 438 / 6650 / 30205 for 528 / 590 / 666 nm.

| Series | Frames summed | Frames | Passes | W, rows | Writing | `phase_filling` |
|---|---|---|---|---|---|---|
| K1 | 1 | 120 | 0.9 | 4.4 | width W | 1.000 |
| K1s | 1 | 120 | 0.9 | 4.4 | single row | 0.892 |
| K4 | 4 | 30 | 1.8 | 8.5 | width W | 1.000 |
| K8 | 8 | 15 | 3.0 | 14.0 | width W | 1.000 |
| K12 | 12 | 10 | 4.1 | 19.5 | width W | 1.000 |
| K12s | 12 | 10 | 4.1 | 19.5 | single row | 0.115 |
| K16 | 16 | 7 | 5.3 | 25.0 | width W | 1.000 |

### 7.1 Peak width

Mean spectrum of the pure pixels, background (20% darkest pixels) subtracted. FWHM from the half-height crossings, σ from a Gaussian fit. Model: `σ²_K = σ²_1 + (W²_K − W²_1) / 12`.

| Series | Passes | 528 nm: FWHM, channels (nm) | 590 nm: FWHM | 666 nm: FWHM | σ 590 nm: measured / model | σ 666 nm: measured / model |
|---|---|---|---|---|---|---|
| K1 | 0.9 | 13.5 (49) | 18.5 (67) | 16.9 (62) | 6.38 / — | 7.12 / — |
| K4 | 1.8 | 14.1 (51) | 19.3 (70) | 17.5 (64) | 6.55 / 6.72 | 7.34 / 7.42 |
| K8 | 3.0 | 15.3 (56) | 21.5 (78) | 18.7 (68) | 7.09 / 7.45 | 7.82 / 8.09 |
| K12 | 4.1 | 19.7 (72) | 27.2 (99) | 21.6 (79) | 9.67 / 8.42 | 9.11 / 8.99 |
| K16 | 5.3 | 19.2 (70) | 31.3 (114) | 24.5 (89) | 12.71 / 9.55 | 10.70 / 10.06 |
| K1s | 0.9 | 13.5 (49) | 18.4 (67) | 16.9 (62) | 6.38 / — | 7.12 / — |
| K12s | 4.1 | 15.7 (57) | 24.6 (90) | 21.7 (79) | 8.49 / 8.42 | 9.20 / 8.99 |

- **666 nm:** the FWHM grows by 4 / 11 / 28 / 45% at 1.8 / 3.0 / 4.1 / 5.3 passes, much more slowly than K. The model of a box blur of W matches the measurement within 0.6 channels.
- **590 nm:** at 4.1–5.3 passes the peak is wider than the model, and wider with writing into W than with a single row (27.2 against 24.6). Likely cause: the spatial blur along the rows mixes in the spectra of neighbouring aggregates of other dyes; not checked.
- **528 nm:** only 438 pixels, the values are noisy (at 5.3 passes the peak is narrower than at 4.1).
- **Writing mode:** does not affect the peak width: K1 = K1s, and for 666 nm K12 = K12s.
- **Upper bound of the resolution:** the width of the 528 nm peak at K = 1 (13.5 channels ≈ 49 nm) is an upper bound of the spectral resolution of the instrument. The intrinsic QD width cannot be separated from these data.

Ratio of the Fourier amplitudes of the mean spectra `|F_K| / |F_1|` (measured / model `|sinc((W_K − W_1) ν)|`):

| ν, cycles per channel | 0.011 | 0.022 | 0.034 | 0.045 |
|---|---|---|---|---|
| K4 | 0.99 / 1.00 | 0.94 / 0.99 | 0.98 / 0.97 | 0.83 / 0.94 |
| K8 | 0.97 / 0.98 | 0.84 / 0.92 | 0.92 / 0.84 | 0.54 / 0.72 |
| K12 | 0.94 / 0.95 | 0.65 / 0.82 | 0.70 / 0.62 | 0.20 / 0.40 |
| K16 | 0.91 / 0.91 | 0.54 / 0.68 | 0.66 / 0.38 | 0.15 / 0.08 |
| K12s | 0.94 / 0.95 | 0.72 / 0.82 | 0.68 / 0.62 | 0.38 / 0.40 |

The trend follows the model; single values deviate by up to 0.3: the mean spectrum of three dyes has few harmonics.

### 7.2 Channel redundancy

How it was computed:
- **Independent values:** the frequency below which the signal exceeds the noise × 2 × 89. The noise shape from the difference of the even and odd frames of K1, its level from the frequencies > 0.35 cycles per channel of each series. Sample: 30 000 bright pure pixels.
- **Resolved bands:** 89 / FWHM of the 666 nm peak.
- **Above Nyquist:** the fraction of the variance of the mean spectra above `1/(2b)` cycles per channel when binning b channels.

| Series | Passes | Independent values | Resolved bands | Noise, % of the pixel spectrum variance | Above Nyquist, b = 6 / 8 |
|---|---|---|---|---|---|
| K1 | 0.9 | 16 | 5.3 | 11.9 | 0.017 / 0.297% |
| K4 | 1.8 | 16 | 5.1 | 10.0 | 0.010 / 0.216% |
| K8 | 3.0 | 16 | 4.8 | 7.8 | 0.005 / 0.108% |
| K12 | 4.1 | 14 | 4.1 | 7.1 | 0.001 / 0.003% |
| K16 | 5.3 | 14 | 3.6 | 8.9 | 0.002 / 0.010% |
| K1s | 0.9 | 14 | 5.3 | 22.7 | 0.019 / 0.305% |
| K12s | 4.1 | 14 | 4.1 | 10.7 | 0.003 / 0.061% |

At b ≤ 4 no more than 0.002% of the variance lies above Nyquist in any series.

**Unmixing after binning b neighbouring channels.**
- **References:** mean spectra of the pure pixels of the same series, binned the same way.
- **Sample:** 60 000 pixels brighter than the median; the last 1–5 channels not divisible by b are dropped.
- **Metrics:** mean absolute change of the shares against 89 channels; noise gain is the row norm of the pseudo-inverse relative to 89 channels.

| Series | 44 channels | 22 | 14 | 11 | 8 |
|---|---|---|---|---|---|
| K1 | 0.03% | 0.11% | 0.21% | 0.33% | 0.75% |
| K4 | 0.03% | 0.10% | 0.18% | 0.30% | 0.73% |
| K8 | 0.02% | 0.08% | 0.16% | 0.29% | 0.71% |
| K12 | 0.02% | 0.08% | 0.16% | 0.26% | 0.56% |
| K16 | 0.03% | 0.10% | 0.19% | 0.35% | 0.81% |
| K1s | 0.06% | 0.20% | 0.35% | 0.53% | 1.01% |
| K12s | 0.03% | 0.10% | 0.19% | 0.31% | 0.81% |
| Noise gain, all series | ×1.00 | ×1.01 | ×1.02 | ×1.04 | ×1.07–1.09 |

- **The redundancy does not depend on K:** 89 channels are redundant already at K = 1. 14–16 independent values, 3.6–5.3 resolved bands at any K.
- **Binning** to 22 channels changes the shares by no more than 0.11% (0.20% for single-row writing), to 14 by no more than 0.21% (0.35%), at any K.
- **89/K channels:** reducing to 49 / 30 / 22 / 17 channels at 1.8 / 3.0 / 4.1 / 5.3 passes loses nothing either. But it does not reflect what actually changes, because the redundancy is the same at every K.
- **Noise:** writing into W at K = 1 halves the noise against single-row writing (11.9 against 22.7% of the spectrum variance).

### 7.3 Spatial resolution along the rows

**Why the rows are blurred.**
- **Along the columns** the camera sees the sample directly. Detail inside a line, diffraction-limited points included, is preserved.
- **Along the rows** the same camera coordinate encodes both position and wavelength (`k = d(λ) + x − c`). Camera row p sums the light of all rows of the window, each at its own wavelength. So one frame does not show which row of the window the light came from: a point inside the window gives only a spectrum shifted by `x − c` channels.
- **Writing into the width W** puts the same spectrum into all W rows. It fills the coverage but carries no detail from inside the line.
- **The position along the rows** is recovered only from different frames. Its precision is limited by the window (aperture plus W) and by one more W contribution from the writing.

**Method.**
- **Profiles:** 4000 aggregate edges across the rows and 4000 across the columns, ±30 pixels.
  - Edge selection: the 97th percentile of the gradient of the K1 map, `|∂x| < 0.3 |∂y|` for edges across the rows and vice versa for the columns.
  - Every profile is normalised to the levels on both sides of the edge and aligned on the K1 edge.
- **Edge blur:** σ of an erf fit to the mean profile of each series. Row stripes cancel in such a mean.
- **I** is the stack summed over the channels; empty rows of the single-row series are filled with `interpolate_zero_gaps`.
- **Blur beyond the columns** `σ_add = √(σ²_rows − σ²_columns)` removes the intrinsic edge width and the optics.
- **Model:** `σ²_add = σ²_ap + 2·W²/12` for writing into W (pass and writing), `σ²_ap + W²/12` for single rows. `σ²_ap = 23.6` rows² is taken from K1s.

| Series | Passes | σ across the rows | σ across the columns | σ_add | Model σ_add | Edge shift, rows |
|---|---|---|---|---|---|---|
| K1 | 0.9 | 7.54 | 4.71 | 5.89 | 5.18 | +0.15 |
| K4 | 1.8 | 8.32 | 5.04 | 6.62 | 5.98 | +0.10 |
| K8 | 3.0 | 9.39 | 5.34 | 7.72 | 7.52 | −0.26 |
| K12 | 4.1 | 10.63 | 5.87 | 8.86 | 9.35 | −1.12 |
| K16 | 5.3 | 13.39 | 6.30 | 11.81 | 11.32 | −1.32 |
| K1s | 0.9 | 6.81 | 4.60 | 5.02 | 5.02 (calibration) | +0.05 |
| K12s | 4.1 | 10.03 | 5.55 | 8.35 | 7.45 | −1.01 |

- **K = 1:** edges along the rows are already blurred by σ ≈ 5–6 rows more than along the columns. This agrees with an aperture of ≈ 17.5 rows (σ of a box of that width 5.1).
- **Growth with K:** σ_add grows by 12 / 31 / 50 / 100% at 1.8 / 3.0 / 4.1 / 5.3 passes. Not K-fold, but the quadrature sum of the aperture and two W contributions.
  - The model matches the measurement within 0.7 rows.
  - A model with one W contribution (`σ²_ap + W²/12`) gives 8.7 at 5.3 passes against the measured 11.8.
- **Single-row writing** at 4.1 passes blurs less (8.35 against 8.86), but leaves 88% of the rows empty, and interpolation fills them.
- **The columns** also blur with K (σ 4.7 → 6.3). The cause is not established; possibly the line arcs are tilted to the rows, and part of the blur along the motion falls into the columns.
- **The edge shift** of −1.1…−1.3 rows at 4.1–5.3 passes is the centre of the wide line found by the phase fit (0.8 rows on the plate at K = 4).
- **A point source** is narrower than an aggregate edge. σ_add does not depend on the object, so for a point the ratio of the width along the rows to the width along the columns will be larger than for aggregates.
- **The mean gradient is not a measure of sharpness:** row stripes inflate it.
  - Stripe size: at K = 1 the high-frequency power along the rows is 8 times that along the columns, although edges along the rows are more blurred; with single-row writing 14 times.
  - With growing K the stripes average out, and the gradient falls not only because of the blur.

### 7.4 Coupling of space and spectrum at the edges

**Method.** Centroid of the dye peak (±10 channels) in the edge pixels of K1, averaged over each edge. Compared are the differences "top − bottom" (along the line motion) and "left − right" (control), in channels.

| Series | 590 nm: top − bottom | 590 nm: left − right | 666 nm: top − bottom | 666 nm: left − right |
|---|---|---|---|---|
| K1 | +1.92 | +0.72 | +1.90 | +0.36 |
| K4 | +1.92 | +0.97 | +2.13 | +0.43 |
| K8 | +1.70 | +1.01 | +2.15 | +0.55 |
| K12 | +2.12 | +0.64 | +2.70 | +0.61 |
| K16 | +2.37 | −0.39 | +1.60 | +0.76 |
| K1s | +1.94 | +0.59 | +1.83 | +0.35 |
| K12s | +3.77 | +0.00 | +3.86 | +0.53 |

Pixels (top / bottom / left / right): 590 nm 550 / 275 / 375 / 248; 666 nm 1599 / 188 / 520 / 340.

- **Already at K = 1:** the edges along the line motion differ by 1.9 channels (≈ 7 nm), in the control by 0.4–0.7 channels.
- **Single-row writing** at 4.1 passes doubles the shift: 3.8 channels ≈ 14 nm.
- **Writing into W** partly averages the shift: 1.6–2.7 channels at 1.8–5.3 passes.
- **A linear regression** of the centroid on `∂ ln I` gave the same sign, but for writing into W the result is unstable and not monotonic in K. For single-row writing the coefficient along the rows grew from 11.7 to 56.9 at 4.1 passes (by 45), while σ² of the peak grew only by 30–34; along the columns the coefficient is 1.5–5.

### 7.5 Conclusions

- **Do not tie the channels to K.** The channel count is a sampling step, not a resolution:
  - 89 channels are redundant already at K = 1 (≈ 5 resolved bands, 14–16 independent values);
  - the spectral width grows by only 4–45% up to 5.3 passes.
- **The spectral blur of the motion** is a box of W channels (W = 4.72·K rows), not of K channels.
- **K blurs space along the rows more than the spectrum.**
  - The edge blur beyond the columns grows by 12 / 31 / 50 / 100% at 1.8 / 3.0 / 4.1 / 5.3 passes, the width of the 666 nm peak only by 4 / 11 / 28 / 45%.
  - Two reasons: the intrinsic peak width dilutes the W contribution to the spectrum, while space has no such admixture; and writing into W adds a second W contribution.
  - The limit "K ≤ 3" of sections 4–5 concerns the spectrum only.
- **The coupling of space and spectrum** at object edges exists already at K = 1. Writing into W does not create it but partly averages it.
- **How to describe the result honestly.** For the chosen K two quantities are needed:
  - the spectral blur ≈ W channels on top of the intrinsic resolution of the instrument (to be measured with a narrow line at K = 1);
  - the blur along the rows `σ² ≈ σ²_ap + 2·W²/12` with writing into W (σ_ap ≈ 4.9 rows); along the columns the blur barely depends on K.
- **Channel binning** (to ~22) can be offered as a separate user choice: it loses nothing for unmixing.

## 8. Limitations

- **The plate is uniform:** it checks the spectral part and the coverage, not spatial mixing.
- **The spatial part** is checked only by emulation on the QD mixture (a sum of frames, not a real long pulse).
- **K = 3, 5, 6 on the plate** were not acquired but extrapolated from p80. At K = 4 the extrapolation underestimated the broadening by 0.5°.
- **Different series conditions:** p80 was acquired with another gain (G20) and exposure (721 ms). Only the profile shapes are compared, so this does not affect the conclusions.
- **nm scale of the plate:** borrowed from the QD calibration.
- **The factor-of-two mismatch of the simulation:** the `rpm` explanation is an assumption; it must be checked against the instrument settings.
- **Aperture:** its size in rows (≈17.5) is taken from the disk geometry of the simulation and is not measured on the plate. The extra edge blur along the rows on the QD mixture at K = 1 (σ_ap ≈ 4.9 rows) agrees with this estimate (section 7.3).
- **Section 7:** K is emulated by sums of frames, not real long pulses; one QD mixture series is checked, and it has no compact isolated aggregates.
- **The intrinsic width of the QD peaks** is unknown, so the spectral resolution of the instrument cannot be obtained from section 7; it needs a narrow line (laser reflection or a spectral lamp).

## 9. Scripts

The scripts are in `analysis`; the run order is in `analysis/README.md`.

| Script | What it does |
|---|---|
| `analysis/plate/sim_profiles.py` | period and line width in the simulation |
| `analysis/plate/plate_recon.py` | reconstruction of the three plate series |
| `analysis/plate/profile_fit.py` | fit of the profiles with K copies, shift per pass |
| `analysis/plate/thick.py` | single-row writing against writing into the width W |
| `analysis/plate/extrap.py` | extrapolation to K = 6 and the calculation for the QD spectra |
| `analysis/qd2/qd_passes.py` | QD mixture at 1.8–5.3 passes |
| `analysis/res/recon_all.py` | reconstructions of the series of section 7 |
| `analysis/res/analyse.py` | peak width, σ model, power spectrum, Fourier ratio, fraction above Nyquist |
| `analysis/res/analyse2.py` | independent values with coloured noise, resolved bands |
| `analysis/res/analyse3.py` | linear regression of the centroid on the gradient |
| `analysis/res/analyse4.py` | centroid shift between opposite edges |
| `analysis/res/analyse5.py` | unmixing after channel binning; the mean gradient at the edges, an unfit measure of sharpness (section 7.3) |
| `analysis/res/isotropy.py` | row / column anisotropy, row stripes |
| `analysis/res/edges.py` | edge blur from averaged profiles |
