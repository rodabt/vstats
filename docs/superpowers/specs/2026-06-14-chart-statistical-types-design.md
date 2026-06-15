# Chart: Statistical Types & Gallery

**Date:** 2026-06-14
**Status:** approved

## Summary

Add 7 new series kinds to `chart/chart.v` and a new `examples/statistical-gallery/` that generates 12 SVGs covering the full statistical range of the module.

---

## New Series Kinds

Seven new `SeriesKind` enum values. All reuse existing `Series` struct fields where possible.

### `step` — Staircase line

```v
pub fn (c Chart) step(x []f64, y []f64, opts SeriesOpts) Chart
```

Same signature as `.line()`. Renders horizontal-then-vertical segments between points (post-step convention: step occurs after each x value). Used for survival curves, empirical CDFs.

Stored as: `x`, `y` arrays. No new fields needed.

### `box` — Box plot

```v
pub fn (c Chart) box(data []f64, opts SeriesOpts) Chart
```

Takes raw data. Computes internally: Q1, median, Q3, whisker bounds (1.5×IQR), outlier values. Multiple `.box()` calls stack side by side like `.bar()` (x-position = series index among box series).

Stored as:
- `y[0]` = Q1, `y[1]` = median, `y[2]` = Q3
- `lo[0]` = whisker_lo, `hi[0]` = whisker_hi
- `x` = outlier raw values (rendered as scatter dots at the box x-position)

Precondition: `assert data.len > 0`

### `dot` — Cleveland dot plot (horizontal)

```v
pub fn (c Chart) dot(values []f64, opts SeriesOpts) Chart
```

`opts.labels` provides category names. Renders a horizontal reference line from the y-axis to a filled dot at each value. Y-axis shows category labels instead of numeric ticks (special-cased in `draw_ticks`).

Stored as: values in `y`, categories in `labels`.  
Bounds: x from `[0, max_value]`, y from `[-0.5, n-0.5]`.

Precondition: if `opts.labels.len > 0`, `assert opts.labels.len == values.len`

### `violin` — Violin plot

```v
pub fn (c Chart) violin(data []f64, opts SeriesOpts) Chart
```

Takes raw data. Computes KDE internally (Gaussian kernel, Silverman bandwidth). Renders two mirrored area polygons (left and right half of the violin shape). Multiple `.violin()` calls stack side by side like `.box()`.

Stored as: `x` = KDE x-grid values, `y` = KDE density values (right half; left half mirrored in rendering). Position determined by count of violin series.

### `heatmap` — 2D colour grid

```v
@[params]
pub struct HeatmapOpts {
pub:
    row_labels []string
    col_labels []string
    color_lo   string = '#f7fbff'
    color_hi   string = '#08306b'
}

pub fn (c Chart) heatmap(data [][]f64, opts HeatmapOpts) Chart
```

Renders a grid of coloured rectangles. Colour is linearly interpolated between `color_lo` and `color_hi` based on the data range. Row and column labels drawn on the axes instead of numeric ticks.

Stored as: `x` = flattened data values (row-major), `labels` = flattened cell value strings for optional annotation. Row/col counts derived from `data.len` and `data[0].len`.

Precondition: `assert data.len > 0 && data[0].len > 0`

### `hbar` — Horizontal bar chart

```v
pub fn (c Chart) hbar(values []f64, opts SeriesOpts) Chart
```

Like `.bar()` but rotated: bars extend left→right, categories on the y-axis. `opts.labels` provides category names. Y-axis shows category labels; x-axis is the value scale.

Stored as: values in `y`, categories in `labels`.  
Bounds: x from `[min(0, min_value), max_value]`, y from `[-0.5, n-0.5]`.

### `stacked_bar` — Stacked bar chart

```v
pub fn (c Chart) stacked_bar(groups [][]f64, opts SeriesOpts) Chart
```

`groups[i]` is the slice of stacked segments for bar position `i`. Renders segments bottom-to-top with colours cycling through the theme palette. `opts.labels` provides category names for x-axis tick labels.

Stored as: `x` = flattened segment values, `y[i]` = number of segments per bar (all equal). Bounds account for total stack height.

Precondition: all inner slices must have the same length.

---

## Gallery

`examples/statistical-gallery/main.v` — single file, generates all SVGs into its own directory. Uses synthetic data generated inline (no external dataset dependencies).

| Output file | Series kinds used | Description |
|---|---|---|
| `line_ci.svg` | `.line()` + `.band()` | Time series with 95% confidence interval |
| `box_comparison.svg` | `.box()` | Distribution comparison across 4 groups |
| `violin_distribution.svg` | `.violin()` | Same 4 groups as violin for visual comparison |
| `step_cdf.svg` | `.step()` | Empirical CDF |
| `step_survival.svg` | `.step()` | Kaplan-Meier style survival curve (two arms) |
| `dot_ranking.svg` | `.dot()` | Feature importances (Cleveland dot plot) |
| `hbar_comparison.svg` | `.hbar()` | Group means, horizontal bar |
| `stacked_bar.svg` | `.stacked_bar()` | Revenue by segment over time |
| `heatmap_correlation.svg` | `.heatmap()` | Correlation matrix (5×5) |
| `forest_plot.svg` | `.scatter()` + `err` + `.axvline()` | Effect sizes with 95% CIs (composed) |
| `qq_normal.svg` | `.scatter()` + `.line()` | Q-Q plot for normality (composed) |
| `kde_density.svg` | `.histogram()` + `.line()` | KDE density overlaid on histogram (composed) |

---

## Implementation Notes

### Categorical axis (shared by `dot`, `hbar`, `heatmap`)

`draw_ticks` checks whether any series in the chart has kind `.dot`, `.hbar`, or `.heatmap`. If so:
- Skip numeric ticks on the categorical axis (y for dot/hbar, both axes for heatmap)
- Render category labels at integer positions instead

### KDE for `violin`

Gaussian kernel, Silverman bandwidth: `h = 1.06 × σ × n^(-1/5)`. Computed over 50 grid points spanning [min − h, max + h]. Implemented inline in `chart.v` (no dependency on `stats` module — keeps chart self-contained).

### Colour interpolation for `heatmap`

Linear interpolation between `color_lo` and `color_hi` in RGB space. Hex colours parsed and blended inline. Written as a private `lerp_color(lo string, hi string, t f64) string` function in `chart.v`.

### `stacked_bar` palette cycling

Segments within a bar cycle through `theme.palette`. This means segment 0 always gets `palette[0]`, segment 1 gets `palette[1]`, etc. — consistent across all bars.

---

## Error Handling

Follows existing chart conventions. `assert` for preconditions at the API boundary. No new error types.

## Testing

No unit tests for the chart module (consistent with current practice). The gallery itself is the integration test: `v run examples/statistical-gallery/` must produce all 12 SVGs without panicking.

## Out of Scope

Nothing previously deferred — all originally out-of-scope types are now included.
