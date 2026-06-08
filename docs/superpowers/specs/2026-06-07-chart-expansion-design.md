# Design: chart module expansion

**Date:** 2026-06-07
**Status:** Approved (design phase)
**Scope:** Extend the existing `chart` module with confidence-interval bands, area fills, value labels, a left-aligned title + subtitle, per-series color override, gridlines, and error bars.

## Goal

Add the following to the `chart` module, reusing the existing `Chart → Scene → SVG backend` pipeline:

1. Confidence-interval **bands** on line charts (`.band`).
2. **Area fills** under a line (`.area`) — shares a primitive with bands.
3. **Value labels** on points and bars (`show_values`).
4. **Left-aligned title** plus a smaller, muted **subtitle**.
5. **Per-series color override**.
6. **Gridlines** (wire up the currently-dead `Theme.grid` field).
7. **Error bars** on points/bars.

## New shared primitive: `Polygon`

A filled closed shape is the missing building block; it powers both CI bands and area fills.

```v
// chart/scene.v
pub struct Polygon {
pub:
    points  []Point
    fill    string
    opacity f64    // fill-opacity in [0, 1]
    stroke  string
    width   f64
}
// extend: pub type Primitive = Line | Polyline | Rect | Circle | Text | Polygon
```

The SVG backend renders `<polygon points="..." fill="..." fill-opacity="..." stroke="..." stroke-width="..."/>`. To guard the zero-default gotcha (an unset `opacity` of `0.0` would be invisible), the backend treats `opacity <= 0` as `1.0`. Band/area builders set opacity explicitly from the theme.

## Features

### CI bands and area fills

- `(c Chart) band(x []f64, lower []f64, upper []f64, opts SeriesOpts) Chart` — shades the region between `lower` and `upper` over `x`. Builds one `Polygon` whose point list traces `upper` left-to-right then `lower` right-to-left (a closed ribbon).
- `(c Chart) area(x []f64, y []f64, opts SeriesOpts) Chart` — fills between the line `(x, y)` and the zero baseline.
- Both are new `SeriesKind` values: `.band`, `.area`.
- The `Series` struct gains `lo []f64` and `hi []f64` (used by `.band`).
- Fill opacity comes from a new `Theme.fill_opacity f64 = 0.2`.
- **Draw order:** `draw_series` runs in two passes — first `.band` and `.area` series (so fills sit behind), then `.line`/`.scatter`/`.bar`/`.histogram`. This makes `.band(...).line(...)` render correctly regardless of call order.

### Value labels on points and bars

- `SeriesOpts` gains `show_values bool` (default `false`) and `labels []string` (optional).
- When `show_values` is true, a small `Text` is drawn above each point/bar: the auto-formatted y-value via `fmt_tick`, or `labels[i]` when a non-empty `labels` array is supplied.
- Applies to `.line`, `.scatter`, and `.bar` series. Label text uses `Theme.font_size` and `Theme.axis_color`, anchored `middle`.
- Precondition: if `labels` is non-empty, `labels.len` must equal the series point count (`assert`).

### Left-aligned title + subtitle

- Title changes from centered (`anchor: .middle`, centered x) to **left-aligned** (`anchor: .start`, x at the plot's left edge `g.plot_x`).
- New subtitle: `ChartOpts.subtitle string` plus a fluent `(c Chart) subtitle(s string) Chart` method; `Chart` struct gains a `subtitle` field.
- The subtitle renders just below the title, left-aligned, in a smaller muted style. New `Theme` fields: `subtitle_size f64 = 12.0`, `subtitle_color string = '#666666'`.
- Layout: title baseline near `title_size`, subtitle baseline below it; both fit inside the existing default `margin_top` (40). No automatic margin growth in v1 — users with very large fonts adjust `margin_top` themselves.

### Per-series color override

- `SeriesOpts` gains `color string`.
- At series creation: `color: if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }`. Empty string preserves the existing palette-cycle behavior.

### Gridlines

- New `draw_grid` helper: when `theme.grid` is true, draw light lines (`theme.grid_color`) across the plot area at each `nice_ticks` position — vertical lines at x ticks, horizontal at y ticks.
- Rendered behind axes and series (first in `build_scene`, right after nothing / before `draw_axes`).
- Default remains `grid: false` (Tufte-clean).

### Error bars

- `SeriesOpts` gains `err []f64`; the `Series` struct stores it.
- When non-empty on a `.scatter` or `.bar` series, draw a vertical whisker from `y[i]-err[i]` to `y[i]+err[i]` with short horizontal end caps, using `theme.axis_color`.
- Bounds: `series_bounds` expands the y-extent to include `y ± err`.
- Precondition: if `err` is non-empty, `err.len` must equal the series point count (`assert`).

## Affected files

- `chart/scene.v` — add `Polygon`, extend `Primitive`.
- `chart/svg.v` — render `Polygon` (with opacity clamp).
- `chart/theme.v` — add `fill_opacity`, `subtitle_size`, `subtitle_color`.
- `chart/chart.v` — extend `ChartOpts` (subtitle), `SeriesOpts` (color, show_values, labels, err), `Series` (lo, hi, err, show_values, labels, plus new kinds); add `band`, `area`, `subtitle` methods; extend `geometry`/`series_bounds`; add `draw_grid`; rewrite `draw_series` (two-pass + value labels + error bars + band/area + per-series color); update `draw_labels` (left title + subtitle).

## Data flow

A `Chart` accumulates `Series` (now including band/area kinds and per-series err/labels/color). `geometry` unions all series bounds (bands via lo/hi, areas including baseline 0, points expanded by err). `build_scene` emits, in order: gridlines (if enabled) → axes → ticks → guides → fills (band/area) → data (line/scatter/bar/histogram) with value labels and error bars → title/subtitle/labels → legend. `render`/`save` are unchanged.

## Testing

New cases across `tests/chart_scene_test.v`, `tests/chart_svg_test.v`, and `tests/chart_test.v`:

- `Polygon` renders to `<polygon` with `fill-opacity`.
- `.band(x, lo, hi)` produces a polygon; point count is `2 * x.len`.
- `.area(x, y)` produces a polygon.
- `show_values: true` adds `Text` elements (count matches points); `labels` override appears verbatim.
- `err` on scatter adds `<line>` whiskers; bounds include `y ± err`.
- `grid: true` adds gridlines; `grid: false` adds none.
- `color: '#abcdef'` overrides the palette (the hex appears in output).
- Left title uses `text-anchor="start"`; subtitle text and muted color render.

## Housekeeping (in scope)

- Regenerate the committed `examples/chart-gallery/*.svg` (the title alignment change alters their output) by running the example.
- Update `docs/src/modules/chart.md` to document the new API, then rebuild HTML via `make docs`.

## Error handling

`assert` preconditions consistent with existing series methods: `lower.len == upper.len == x.len` for bands; `err.len == series point count` when provided; `labels.len == series point count` when provided. I/O still propagates via `!` in `save`.

## Out of scope

- Grouped/stacked bars, marker shapes, log scales, non-numeric axes, theme registry, raster output — future work.
- Automatic top-margin growth for subtitles with oversized fonts.
