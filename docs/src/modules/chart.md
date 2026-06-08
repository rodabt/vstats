# chart

`import vstats.chart`

Dependency-free SVG charting with sensible Tufte-style defaults. Build a chart with
a fluent API, render it to an SVG string, and save it to disk. Foundational output
layer — depends only on the V standard library.

> **vs Python:** replaces basic `matplotlib` plotting. A `Chart` builds a
> backend-agnostic scene of primitives that the SVG backend renders; the same scene
> can later feed a raster (PNG/JPG) backend.

## Building a chart

```v
new(opts ChartOpts) Chart            // ChartOpts{ title, width, height, theme }

// series — each appends a series and returns the Chart for chaining
(c Chart) line(x []f64, y []f64, opts SeriesOpts) Chart        // SeriesOpts{ label }
(c Chart) scatter(x []f64, y []f64, opts SeriesOpts) Chart
(c Chart) bar(values []f64, opts SeriesOpts) Chart
(c Chart) histogram(data []f64, opts HistogramOpts) Chart      // HistogramOpts{ label, nbins }

// decoration
(c Chart) title(s string) Chart
(c Chart) xlabel(s string) Chart
(c Chart) ylabel(s string) Chart
(c Chart) axhline(y f64) Chart       // horizontal reference line
(c Chart) axvline(x f64) Chart       // vertical reference line

// output
(c Chart) render() string            // pure: Chart -> SVG text
(c Chart) save(path string) !        // render() then write to disk
```

## Example

```v
import vstats.chart

chart.new(title: 'Fit', width: 640, height: 420)
    .scatter(x, y, label: 'observed')
    .line(xs, ys, label: 'fit')
    .xlabel('x')
    .ylabel('y')
    .save('fit.svg')!
```

Series data is `[]f64`. Multiple series share auto-scaled axes; with two or more
labeled series a legend is drawn automatically, and colors cycle through the theme
palette. `render()` is pure (handy for tests); `save()` is the only function that
touches the filesystem.

## Chart types

| Method | Renders |
|--------|---------|
| `line` | connected polyline through the `(x, y)` points |
| `scatter` | point markers at each `(x, y)` |
| `bar` | bars from a zero baseline, one per value |
| `histogram` | binned distribution (auto bin count via Sturges, or explicit `nbins`) |

## Theming

A single opinionated Tufte default (minimal ink, range-frame axes, no chartjunk,
no gridlines). Override any field via the `Theme` struct and pass it as
`new(theme: ...)`:

```v
Theme{
    background:    'white'
    axis_color:    '#333333'
    palette:       ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    font_family:   'sans-serif'
    font_size:     12.0
    title_size:    16.0
    series_width:  1.5
    marker_radius: 3.0
    // also: margin_left/right/top/bottom, axis_width, grid, grid_color
}

(t Theme) color(i int) string        // palette color i, cycled
```

(The `grid` field is reserved; gridline rendering is not yet implemented.)

## Under the hood

A `Chart` builds a `Scene` — a list of `Primitive`s (`Line`, `Polyline`, `Rect`,
`Circle`, `Text`) in pixel space — which a backend turns into output. This seam is
what keeps a future raster backend non-invasive.

```v
render_svg(scene Scene, width int, height int, theme Theme) string
histogram_bins(data []f64, nbins int) HistogramBins   // nbins <= 0 => Sturges' rule
nice_ticks(min f64, max f64, target int) []f64        // round 1/2/5×10ⁿ tick steps
```

See `examples/chart-gallery/` for a complete regression-diagnostics gallery
(scatter + fit line, residuals, histogram, coefficient bar).
