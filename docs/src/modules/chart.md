# chart

`import vstats.chart`

Dependency-free SVG charting with sensible Tufte-style defaults. Build a chart with
a fluent API, render it to an SVG string, and save it to disk. Foundational output
layer — depends only on the V standard library.

> **vs Python:** replaces `matplotlib` for the plot types below. A `Chart` builds a
> backend-agnostic scene of primitives that the SVG backend renders; the same scene
> can later feed a raster (PNG/JPG) backend. There is no log-scale axis yet, and
> raster (PNG/PDF) export isn't built — SVG is the only output format.

## Building a chart

```v
new(opts ChartOpts) Chart   // ChartOpts{ title, subtitle, width, height, theme }

// series — each appends a series and returns the Chart for chaining
(c Chart) line(x []f64, y []f64, opts SeriesOpts) Chart
(c Chart) scatter(x []f64, y []f64, opts SeriesOpts) Chart
(c Chart) bar(values []f64, opts SeriesOpts) Chart
(c Chart) hbar(values []f64, opts SeriesOpts) Chart
(c Chart) histogram(data []f64, opts HistogramOpts) Chart      // HistogramOpts{ label, nbins }
(c Chart) box(data []f64, opts SeriesOpts) Chart                // quartiles + 1.5xIQR whiskers + outliers
(c Chart) violin(data []f64, opts SeriesOpts) Chart              // Gaussian KDE (Silverman's rule)
(c Chart) dot(values []f64, opts SeriesOpts) Chart                // Cleveland dot plot
(c Chart) heatmap(data [][]f64, opts HeatmapOpts) Chart          // 2D color grid
(c Chart) stacked_bar(groups [][]f64, opts SeriesOpts) Chart
(c Chart) stacked_bar_pct(groups [][]f64, opts SeriesOpts) Chart // 100%-normalized stacked bars
(c Chart) grouped_bar(groups [][]f64, opts SeriesOpts) Chart
(c Chart) waterfall(values []f64, labels []string, opts SeriesOpts) Chart
(c Chart) dumbbell(before []f64, after []f64, labels []string, opts SeriesOpts) Chart

// fills (rendered behind data marks)
(c Chart) band(x []f64, lower []f64, upper []f64, opts SeriesOpts) Chart   // CI / shaded region
(c Chart) area(x []f64, y []f64, opts SeriesOpts) Chart                    // fill to zero baseline
(c Chart) step(x []f64, y []f64, opts SeriesOpts) Chart                    // staircase line

// decoration
(c Chart) title(s string) Chart
(c Chart) subtitle(s string) Chart   // smaller, muted, left-aligned
(c Chart) xlabel(s string) Chart
(c Chart) ylabel(s string) Chart
(c Chart) xcategories(labels []string) Chart  // categorical x-axis labels
(c Chart) xmin(v f64) Chart                   // override auto x-axis minimum
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

Series data is `[]f64` (or `[][]f64` for grouped/stacked/heatmap series). Multiple
series share auto-scaled axes; with two or more labeled series a legend is drawn
automatically, and colors cycle through the theme palette. `render()` is pure
(handy for tests); `save()` is the only function that touches the filesystem.

`SeriesOpts` also accepts `color` (override the palette), `show_values` (draw the
value above each point/bar — or pass `labels []string` for custom text), `err []f64`
(draw error-bar whiskers on points/bars), and `secondary_axis: true` (plot the
series against a second y-axis, drawn on the right).

## Chart types

| Method | Renders |
|--------|---------|
| `line` | connected polyline through the `(x, y)` points |
| `scatter` | point markers at each `(x, y)` |
| `bar` / `hbar` | vertical / horizontal bars from a zero baseline |
| `histogram` | binned distribution (auto bin count via Sturges, or explicit `nbins`) |
| `box` | quartile box, 1.5×IQR whiskers, outlier dots |
| `violin` | Gaussian KDE density shape (Silverman's rule bandwidth) |
| `dot` | Cleveland dot plot (categorical) |
| `heatmap` | 2D color grid from a `[][]f64` matrix |
| `stacked_bar` / `stacked_bar_pct` | stacked segments per group, raw or 100%-normalized |
| `grouped_bar` | clustered bars per group |
| `waterfall` | cumulative-delta bridge chart (increase/decrease/total coloring) |
| `dumbbell` | before/after paired-comparison plot |
| `area` | fill to zero baseline |
| `band` | filled region between two curves (confidence intervals) |
| `step` | staircase line (e.g. ECDFs) |

## Theming

A single opinionated Tufte default (minimal ink, range-frame axes, no chartjunk).
Override any field via the `Theme` struct and pass it as `new(theme: ...)`:

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
    grid:          false  // set true for light gridlines
    // also: margin_left/right/top/bottom, axis_width, grid_color,
    // subtitle_size/color, waterfall_increase/decrease/total colors
}

(t Theme) color(i int) string        // palette color i, cycled
```

There's no named theme registry (dark/print presets) or colorblind-safe palette
variant yet — only this one default, overridable field-by-field.

## Categorical axes and dual y-axis

`.xcategories(labels)` sets a chart-wide categorical x-axis; bar-family, dot,
hbar, dumbbell, and heatmap series also auto-infer category labels from their
own `labels []string`. Pass `secondary_axis: true` in `SeriesOpts` to plot a
series against an independent right-hand y-axis (useful for combo bar+line
charts with mismatched scales).

## Tooltips (opt-in interactivity)

Rect/circle primitives carry hover metadata (series, label, x, y) into the SVG's
`data-*` attributes and a `<title>` fallback. To get interactive hover tooltips
in an HTML page, embed the rendered SVG alongside `chart/chart-interactions.js`
and call `ChartTooltips.init()` — see `examples/chart-tooltips-demo/` for the
full pattern. This is bolt-on glue, not part of the `chart` module's own API;
there's no `render_html()` and no zoom/pan/animation.

## Composing multiple charts

`chart/grid.v` adds a `Grid` type that tiles several already-built `Chart`s into
one combined SVG — matplotlib's `plt.subplots()` equivalent:

```v
new_grid(charts []Chart, opts GridOpts) Grid   // GridOpts{ cols = 2, gap = 20, title }
(g Grid) render() string
(g Grid) save(path string) !

pair_plot(columns [][]f64, labels []string, opts GridOpts) Grid
```

`new_grid` lays panels out left-to-right, wrapping to a new row every `cols`
charts, sized by each panel's own `width`/`height`. `pair_plot` builds an N×N
scatter-matrix Grid from N `[]f64` columns — off-diagonal panels are pairwise
scatter plots, diagonal panels are that column's histogram. See
`examples/diagnostic-plots-gallery/` for a worked pair-plot over the Iris
dataset.

## Diagnostic plot recipes

These aren't chart types of their own — they're a data helper elsewhere in the
library plus one of the series builders above, following the project's
"charting stays the caller's job" convention (see `examples/chart-gallery/`):

```v
// ROC curve — utils.roc_curve() already computes fpr/tpr/auc
roc := utils.roc_curve(y_true, y_scores)
chart.new(title: 'ROC (AUC=${roc.auc:.3f})')
    .line(roc.fpr, roc.tpr, label: 'ROC')
    .line([0.0, 1.0], [0.0, 1.0], label: 'chance')

// ECDF — stats.ecdf() returns (sorted_x, cumulative_probability)
xs, probs := stats.ecdf(sample)
chart.new().step(xs, probs)

// Normal Q-Q plot — prob.qq_points() returns (theoretical, sample) quantiles
theoretical, sample := prob.qq_points(residuals)
chart.new().scatter(theoretical, sample)
```

See `examples/diagnostic-plots-gallery/` for all four (ROC, ECDF, Q-Q, pair plot)
together in one runnable example.

## Under the hood

A `Chart` builds a `Scene` — a list of `Primitive`s (`Line`, `Polyline`, `Rect`,
`Circle`, `Text`, `Polygon`) in pixel space — which a backend turns into output.
This seam is what keeps a future raster backend non-invasive.

```v
render_svg(scene Scene, width int, height int, theme Theme) string
histogram_bins(data []f64, nbins int) HistogramBins   // nbins <= 0 => Sturges' rule
nice_ticks(min f64, max f64, target int) []f64        // round 1/2/5×10ⁿ tick steps
```

See `examples/chart-gallery/` for a regression-diagnostics gallery (scatter + fit
line, residuals, histogram, coefficient bar), `examples/chart-types-expansion-demo/`
for combo/dual-axis/dumbbell/grouped-bar/stacked-pct/waterfall demos, and
`examples/diagnostic-plots-gallery/` for ROC/ECDF/Q-Q/pair-plot.
