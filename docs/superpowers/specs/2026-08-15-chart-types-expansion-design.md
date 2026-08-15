# Chart Types Expansion — Design

**Date:** 2026-08-15
**Module:** `chart/`
**Status:** Approved design, ready for implementation plan

## Goal

Extend the `chart` module with five capabilities needed for common analytics/reporting
use cases that aren't covered today:

1. **Waterfall** — cumulative build-up (e.g. revenue bridge: start → +gains → -losses → end).
2. **Grouped (clustered) bar** — side-by-side bars per category, complementing the
   existing `stacked_bar`.
3. **Combo / dual-axis** — mixing series (e.g. bar + line) with independent primary and
   secondary y-axes, for metrics on different scales.
4. **Categorical x-axis labels** — usable uniformly by line/scatter/bar charts, not just
   the bar-family kinds that special-case it today.
5. **Dumbbell (before/after) plot** — paired two-value-per-category comparison with a
   connecting line, e.g. metric before vs. after an intervention.

All five build on the existing `Chart` → `Series` → `Geom` → `Scene` → SVG pipeline
(`chart/chart.v`, `scales.v`, `scene.v`, `svg.v`, `theme.v`) without changing that
pipeline's shape.

## Background

`chart` is a dependency-free, purely server-side SVG generator with an immutable
builder API: `Chart.line(...)`, `.bar(...)`, `.stacked_bar(...)`, etc. each return a new
`Chart` with one more `Series` appended. `SeriesKind` currently covers: `line`, `scatter`,
`bar`, `histogram`, `band`, `area`, `step`, `box_plot`, `dot`, `violin`, `hbar`, `heatmap`,
`stacked_bar`.

Two relevant existing mechanisms this design reuses rather than replaces:
- **Per-series `labels []string`**, currently read ad hoc by `draw_ticks` for `dot`/`hbar`
  (y-axis categories) and `stacked_bar`/`heatmap` (x-axis categories).
- **`series_bounds()` / `data_bounds()`**, which compute the single shared x/y domain
  every series is plotted against via one `LinearScale` pair in `Geom`.

---

## Part 1 — Categorical x-axis (`Chart.xcategories`)

### Problem

Today, x-axis category labels are attached per-series and detected by scanning series
kinds in `draw_ticks` (`s.kind == .stacked_bar && s.labels.len > 0`, etc). `line` and
`scatter` have no categorical x-axis path at all — they always render numeric ticks.

### Design

Add a chart-level method:

```v
pub fn (c Chart) xcategories(labels []string) Chart
```

Stored as `Chart.xcategories_ []string`. When non-empty, every series in the chart is
expected to use **integer x-positions `0..labels.len-1`** as category indices (this is
already how `bar`, `stacked_bar`, `grouped_bar`, and `waterfall` position bars; `line` and
`scatter` callers now do the same — pass `x: [0.0, 1.0, 2.0, ...]` matching category
order).

`draw_ticks` changes from scanning series kinds to one check:

```v
if c.xcategories_.len > 0 {
    // render c.xcategories_[i] at xscale.map(f64(i)) for each i
} else {
    // existing numeric nice_ticks path
}
```

This subsumes the current `x_cat_labels` detection for `stacked_bar`/`heatmap` (their
`labels` field becomes redundant for x-axis purposes but is kept for backward
compatibility — a series-level `labels` still works if `xcategories_` is unset, so
existing chart-building code is not broken). The y-axis categorical path (`dot`/`hbar`/
`dumbbell` row labels) is untouched — this is specific to the x-axis.

### Testing

- `chart.new().xcategories(['Jan','Feb','Mar']).line([0,1,2], [10,20,15], ...)` renders
  month labels instead of numeric ticks.
- Existing bar-family tests (which rely on per-series `labels`) continue passing
  unchanged, since that fallback path stays in place.

---

## Part 2 — Secondary y-axis (combo charts)

### Problem

`Geom` has exactly one `yscale`. Mixing a bar series (e.g. revenue, thousands) with a
line series (e.g. conversion rate, 0–1) on the same axis makes one of them unreadable.

### Design

- `SeriesOpts` gains `secondary_axis bool` (default `false`); `Series` gains the same
  field, set from `opts.secondary_axis` in every series-constructor method.
- `Chart.data_bounds()` splits into `data_bounds_primary()` (series where
  `secondary_axis == false`) and `data_bounds_secondary()` (`== true`). If no series
  requests the secondary axis, secondary bounds are unused.
- `Geom` gains `yscale2 LinearScale` and `has_secondary bool`.
- `draw_series`, `draw_error_bars`, and `draw_value_labels` select `g.yscale` or
  `g.yscale2` per series based on `s.secondary_axis`.
- `draw_ticks` draws a mirrored right-side axis (ticks + `fmt_tick` labels, `anchor:
  .start`) when `g.has_secondary`, using `nice_ticks` against the secondary domain.
- `effective_margins()` grows `margin_right` to fit the secondary axis's widest tick
  label, the same way it already grows `margin_left` for the primary axis.
- Legend and tooltips are unaffected — a secondary-axis series still gets a normal
  legend swatch and hover `Meta`.

### Testing

- `chart.new().bar(revenue, {}).line(rate_x, rate_y, {secondary_axis: true})` produces
  two distinct y-scales; rendered SVG has both a left and right axis with correct,
  independently-scaled tick values.
- A chart with no secondary-axis series renders byte-identical output to before (no
  right axis drawn, `effective_margins` unchanged).

---

## Part 3 — Waterfall

```v
pub fn (c Chart) waterfall(values []f64, labels []string, opts SeriesOpts) Chart
```

- New `SeriesKind.waterfall`. Stored as one `Series{kind: .waterfall, y: values, labels: labels}`.
- **Convention:** `values[0]` and `values[values.len-1]` are absolute totals (bar drawn
  from 0 to that value). Every value in between is a signed delta from the running
  cumulative total.
- **Bounds** (`series_bounds`): x = `-0.5 .. n-0.5` (bar-style band layout, one bar per
  value); y = min/max of the running cumulative total across all bars (so intermediate
  dips/peaks aren't clipped).
- **Rendering** (`draw_series`): walk the values maintaining `cum`; for interior bars,
  draw from `cum` to `cum + values[i]` and advance `cum`; a thin connector `Line` (using
  `theme.grid_color`) bridges each bar's top to the next bar's starting point, Tufte-style.
- **Color:** `Theme` gains three new fields — `waterfall_increase` (default `#2ca02c`),
  `waterfall_decrease` (default `#d62728`), `waterfall_total` (default `#7f7f7f`).
  `SeriesOpts` gains matching optional overrides `color_increase`/`color_decrease`/
  `color_total`, falling back to theme when `''`.
- **Value labels** (`show_values`): show the delta for interior bars (with sign, e.g.
  `+120`/`-45`), the absolute value for total bars.
- **Tooltip:** new `waterfall_meta(label, delta_or_total, is_total, running_total)`
  helper, following the existing `*_meta()` pattern — includes running total in the
  tooltip text.

### Testing

- Values summing correctly: total-out bar equals `values[0] + sum(interior deltas)`.
- Mixed +/- deltas produce bars floating at the correct `cum` offsets.
- Single-total-only edge case (`values.len == 2`, no interior deltas) still renders two
  full bars with a connector.

---

## Part 4 — Grouped (clustered) bar

```v
pub fn (c Chart) grouped_bar(groups [][]f64, opts SeriesOpts) Chart
```

- New `SeriesKind.grouped_bar`. Input shape identical to `stacked_bar`:
  `groups[i][j]` = group `i`, segment `j`. Reuses the same flattening into `Series.x`
  (flat values) + `Series.nbins` (segment count) + `opts.colors`/`opts.labels`.
- **Bounds:** x = `-0.5 .. nbars-0.5`; y = `0 .. max(individual segment value)` (not the
  stacked sum — this is the key behavioral difference from `stacked_bar`).
- **Rendering:** within each group's band (width `bw`, same 0.8-of-band sizing as `bar`),
  segments are laid out side by side: `sub_bw := bw / nseg`, segment `j`'s rect starts at
  `cx - bw/2 + j*sub_bw`.
- Reuses `seg_meta()` for tooltips (same helper `stacked_bar` uses).
- `draw_ticks`'s categorical-x detection adds `.grouped_bar` alongside `.stacked_bar`
  (superseded by `xcategories_` per Part 1, but kept as fallback per Part 1's backward
  compatibility note).

### Testing

- 3 groups × 2 segments renders 6 bars in 3 clusters, each segment its own color.
- y bounds reflect max single segment, not stack sum (distinguishing test from
  `stacked_bar`).

---

## Part 5 — Dumbbell (before/after) plot

```v
pub fn (c Chart) dumbbell(before []f64, after []f64, labels []string, opts SeriesOpts) Chart
```

- New `SeriesKind.dumbbell`. Stored as one `Series{kind: .dumbbell, lo: before, hi: after,
  labels: labels}`, reusing the existing `lo`/`hi` fields (same reuse pattern as `.band`).
- **Orientation:** horizontal, matching existing `.dot`/`.hbar` — category index
  `0..n-1` maps to row y-positions (top-to-bottom in label order), values map to x.
- **Bounds:** x = combined extent of `before` and `after`; y = `-0.5 .. n-0.5`.
- **Rendering:** per row, a connector `Line` from `(xscale.map(before[i]), y)` to
  `(xscale.map(after[i]), y)` in `theme.grid_color`, then two `Circle` marks: "before" in
  `opts.color` (or default palette color), "after" in `opts.color_hi` (reusing the
  existing `Series.color_hi` field, `SeriesOpts` gains `color_hi` accordingly) — default
  `color_hi` falls back to a second palette slot if unset.
- Participates in the y-axis categorical label path in `draw_ticks` (add `.dumbbell`
  alongside `.dot`/`.hbar`).
- **Tooltip:** new `dumbbell_meta(label, before_v, after_v)` helper reporting both values
  and their delta.
- `show_values`: labels each dot with its formatted value, mirroring `.dot`'s style.

### Testing

- 5 categories, before/after pairs render 5 rows with 10 dots + 5 connectors.
- Row order matches label order (top-to-bottom), consistent with `.dot`/`.hbar`.

---

## Non-goals

- No zoom/pan, no client-side interactivity beyond what `chart-interactions.js` already
  provides (tooltips) — all five additions are static SVG, same as every existing kind.
- No general N-series (3+) dot plot — dumbbell specifically covers the 2-value
  before/after case; a general multi-dot variant is deferred until a concrete need arises.
- No mid-sequence waterfall subtotals (only first/last are absolute totals) — deferred
  until needed; would require a parallel `is_total []bool` if added later.
- Grouped bar and stacked bar remain separate methods/kinds — no unified "stacking mode"
  parameter.

## Testing Strategy

Each new `SeriesKind` gets a dedicated test file under `tests/`, following the existing
per-concern split (`chart_bins_test.v`, `chart_scales_test.v`, `chart_svg_test.v`, etc.):
- `chart_waterfall_test.v`
- `chart_grouped_bar_test.v`
- `chart_dumbbell_test.v`
- `chart_dualaxis_test.v` (secondary axis + `data_bounds_secondary`)
- `chart_xcategories_test.v`

Tests assert on `series_bounds()`/`data_bounds()` output and on presence/values of
specific primitives in `build_scene()` output (matching the existing test style of
inspecting `Scene.primitives`), not on raw SVG string matching.
