# Design: `chart` module

**Date:** 2026-06-07
**Status:** Approved (design phase)
**Scope:** v1 of a dependency-free SVG charting module for VStats, architected to grow toward a full scientific-visualization library.

## Goal

A dependency-free charting module written in V that renders charts to SVG and saves them to disk. v1 ships a minimal, useful set of chart types with opinionated Tufte-style defaults. The architecture is deliberately layered so it can grow into a Matplotlib-class general plotting tool, and so a future raster (PNG/JPG) backend can slot in cleanly.

## Non-goals (v1)

- No PNG/JPG/raster output (designed for, not built — see Future).
- No theme registry / named themes.
- No subplots, faceting, or animation.
- No interactivity.
- No generic `[]T` series inputs (series data is `[]f64` in v1).

## Architecture

A three-stage pipeline; each stage is independently testable:

```
Chart  ──build──▶  Scene ([]Primitive)  ──render──▶  output
(config/builder)   (backend-agnostic)               SVGBackend → string → disk   (v1)
                                                     RasterBackend → []u8 → stbi  (future)
```

- **`Chart`** — user-facing config + fluent builder. Knows about series, axes, labels, theme. Knows nothing about SVG.
- **`Scene`** — a flat list of drawing primitives in pixel coordinates. This is the decoupling seam: it makes a future raster backend clean and makes tests robust (assert on primitives instead of string-matching SVG).
- **Backend** — a pure transform from `Scene` to an output format. Only the SVG backend exists in v1.

### Placement in the layer hierarchy

`chart` is a foundational/parallel output layer. It depends only on `linalg` and the V stdlib (`os`, `strings`, `math`, `arrays`). It does **not** depend on `stats` — histogram binning is self-contained. This keeps it low in the hierarchy and fully reusable.

Any vstats-specific plotting (plot a regression fit, plot a PDF/CDF, plot an A/B readout) belongs in a future **recipes** layer living inside the higher modules that already depend on everything, so they can import `chart` without creating dependency cycles.

## File layout

| File | Responsibility |
|------|----------------|
| `chart/chart.v` | `Chart` struct, `new()`, fluent methods, `render() string`, `save(path) !` |
| `chart/scene.v` | `Primitive` sum type, `Scene`, geometry types (`Point`) |
| `chart/scales.v` | Linear scale (data domain → pixel range) + "nice ticks" generation |
| `chart/theme.v` | `Theme` struct with explicit non-zero Tufte defaults + color cycle |
| `chart/svg.v` | SVG backend: `Scene → string` |
| `chart/bins.v` | Histogram binning (auto bin count via Sturges, or explicit) |

Tests live in `tests/` (per project convention), not alongside source:

- `tests/chart_scales_test.v` — tick generation and domain→range mapping.
- `tests/chart_scene_test.v` — `Chart` builds the expected primitive list.
- `tests/chart_svg_test.v` — SVG backend emits expected elements/structure.
- `tests/chart_test.v` — end-to-end: build → render → assert key SVG content; save to a temp path and confirm the file exists.

## Public API

```v
import vstats.chart

chart.new(title: 'Model fit', width: 720, height: 480)
    .scatter(x, y, label: 'data')
    .line(x, y_pred, label: 'fit')
    .xlabel('x')
    .ylabel('y')
    .save('fit.svg')!
```

- `new(cfg Chart) Chart` — `@[params]` config (title, width, height, theme).
- Series methods (each appends a series and returns `Chart` for chaining): `line`, `scatter`, `bar`, `histogram`.
- Decoration methods: `xlabel`, `ylabel`, `title`, `axhline(y f64)`, `axvline(x f64)`.
- Terminal methods: `render() string` (pure, testable) and `save(path string) !` (thin I/O wrapper over `render()`).

### Data input shape

Series data is `[]f64` in v1, consistent with the library's `[][]f64` matrix functions and avoiding V's generic-method friction. A generic `[]T` convenience layer can be added later.

### Series-level options

Series methods take an `@[params]` options struct, e.g. `label string`, allowing `chart.line(x, y, label: 'fit')`. Per-series color defaults to the next entry in the theme's color cycle; an explicit override can be added later.

### Multiple series

A `Chart` holds `[]Series`. All series share auto-scaled axes computed from the union of their data extents. Colors are drawn from the theme's color cycle. A minimal legend renders automatically when there are two or more labeled series.

## Scales & ticks (the core)

This is what separates "looks good by default" from "looks homemade":

- **Linear scale**: maps `[data_min, data_max]` → `[pixel_lo, pixel_hi]`, correctly handling SVG's downward-growing y-axis.
- **Nice ticks**: tick steps rounded to `1 / 2 / 5 × 10ⁿ` so axes read `0, 25, 50, 75, 100` rather than `0, 23.7, 47.4, …`.
- **Tufte layout**: range frame (axes span only the data extent), minimal short tick marks, no enclosing box, no or very-light gridlines, thin strokes, a restrained palette.

## Scene primitives

`Primitive` is a sum type. v1 set:

- `Line` — single segment (axes, tick marks, axhline/axvline).
- `Polyline` — connected points (line series).
- `Rect` — filled/stroked rectangle (bars, histogram bins, legend swatches).
- `Circle` — point markers (scatter).
- `Text` — labels, tick labels, title, legend text (with anchor/alignment).

All coordinates are in pixel space (post-scale). The SVG backend maps each primitive 1:1 to an SVG element with inline presentation attributes (`stroke`, `fill`, `stroke-width`) rather than CSS, for portability.

## Theme

`Theme` is an `@[params]` struct with **explicit non-zero defaults** (guarding V's zero-default-on-unpassed-params gotcha — an un-passed theme must not render a 0×0 blank). Fields include: margins, font family and sizes, stroke widths, axis color, color cycle, and grid on/off. Passing nothing yields the polished Tufte look; fields can be overridden per chart. No theme registry yet (a registry is just a `map[string]Theme` and is a trivial later add).

## Histogram binning

Self-contained in `chart/bins.v`: given data and an optional bin count, compute bin edges and counts. Default bin count uses Sturges' rule (`ceil(log2(n) + 1)`). No dependency on `stats`.

## Error handling

- `assert x.len == y.len` for series inputs (precondition / programmer error, per project conventions).
- `save()` propagates I/O errors with `!`.
- Empty series produce an error during the build stage.

## Testing strategy

- Assert on the `Scene` primitive list for layout/scale/binning logic (robust, no string fragility).
- Assert on key SVG structure (element presence/counts, substrings) for the backend.
- One end-to-end test renders and saves to a temp path and confirms the file exists.
- The bulk of tests have no filesystem dependence.

## SVG output specifics

- Self-contained `<svg>` with `viewBox` and `width`/`height` from the theme.
- Background fill from the theme (white or none).
- Inline presentation attributes (no external CSS) for portability.
- Text via `<text>` using the theme font family; y-axis flip handled in the scale layer.

## Future extension points (documented, not built in v1)

- **Raster backend**: `Scene → []u8` pixel buffer → `stbi.stbi_write_png` / `stbi_write_jpg`. Note: stbi only *encodes* a raw pixel buffer (`fn stbi_write_png(path string, w int, h int, nr_channels int, buf &u8, row_stride_in_bytes int) !`); it does **not** rasterize SVG. A raster backend therefore requires writing a rasterizer (line/polygon fill, text glyphs) — a substantial, separate effort. The `Scene` abstraction is what makes this addition non-invasive.
- More chart types: box plot, error bars, heatmap.
- Theme registry (named themes).
- A vstats **recipes** layer in higher modules (`regression.plot()`, plot a PDF/CDF, A/B readout charts).
- Subplots / faceting.
