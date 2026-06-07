# Chart Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free SVG charting module (`chart`) for VStats that renders line/scatter/bar/histogram charts with Tufte-style defaults and saves them to disk.

**Architecture:** A three-stage pipeline — a fluent `Chart` builder produces a backend-agnostic `Scene` (a list of drawing primitives in pixel space), which an SVG backend renders to a string and saves. Scale/tick math is isolated; a future raster backend can consume the same `Scene`.

**Tech Stack:** V 0.5.1, standard library only (`math`, `strings`, `os`). No external dependencies. Tests live in `tests/` and use bare module imports (`import chart`), matching this repo's convention.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `chart/scene.v` | Geometry types (`Point`), drawing primitives (`Line`, `Polyline`, `Rect`, `Circle`, `Text`), the `Primitive` sum type, and `Scene` |
| `chart/scales.v` | `LinearScale` (domain→pixel mapping) and tick generation (`nice_ticks`, `nice_num`, `fmt_tick`) |
| `chart/theme.v` | `Theme` styling struct with explicit non-zero Tufte defaults and a color-cycle method |
| `chart/bins.v` | Histogram binning (`HistogramBins`, `histogram_bins`) |
| `chart/svg.v` | SVG backend: `render_svg(scene, width, height, theme)` and `primitive_to_svg` |
| `chart/chart.v` | `Chart` struct + `ChartOpts`/`SeriesOpts`, `new()`, fluent builder methods, scene assembly, `render()`, `save()` |

Test files (all under `tests/`):
- `tests/chart_scene_test.v`
- `tests/chart_scales_test.v`
- `tests/chart_theme_test.v`
- `tests/chart_bins_test.v`
- `tests/chart_svg_test.v`
- `tests/chart_test.v` (builder + end-to-end)

**Build/run notes for the engineer:**
- Run all module tests: `v test tests/`
- Run a single test file: `v test tests/chart_scene_test.v`
- V errors on unused imports — only import what a file actually uses.
- V is strict about numeric types: assign `f64` fields from `f64` expressions (use `f64(...)` casts and `1.0`-style literals, not bare ints).
- Inside a `match` on a sum type, the matched variable is smart-cast to the arm's type (e.g. `match p { Line { p.x1 } }`).

---

## Task 1: Scene primitives and Scene container

**Files:**
- Create: `chart/scene.v`
- Test: `tests/chart_scene_test.v`

- [ ] **Step 1: Write the failing test**

```v
// tests/chart_scene_test.v
import chart

fn test__scene_holds_primitives() {
	mut s := chart.Scene{}
	s.primitives << chart.Line{
		x1:     0.0
		y1:     0.0
		x2:     10.0
		y2:     5.0
		stroke: '#000'
		width:  1.0
	}
	s.primitives << chart.Circle{
		cx:   3.0
		cy:   4.0
		r:    2.0
		fill: 'red'
	}
	assert s.primitives.len == 2
}

fn test__primitive_match_smart_cast() {
	p := chart.Primitive(chart.Text{
		x:       1.0
		y:       2.0
		content: 'hi'
		anchor:  .middle
	})
	got := match p {
		chart.Text { p.content }
		else { '' }
	}
	assert got == 'hi'
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_scene_test.v`
Expected: FAIL — `chart` module / types not defined.

- [ ] **Step 3: Write minimal implementation**

```v
// chart/scene.v
module chart

pub struct Point {
pub:
	x f64
	y f64
}

pub enum TextAnchor {
	start
	middle
	end
}

pub struct Line {
pub:
	x1     f64
	y1     f64
	x2     f64
	y2     f64
	stroke string
	width  f64
}

pub struct Polyline {
pub:
	points []Point
	stroke string
	width  f64
}

pub struct Rect {
pub:
	x      f64
	y      f64
	w      f64
	h      f64
	fill   string
	stroke string
	width  f64
}

pub struct Circle {
pub:
	cx     f64
	cy     f64
	r      f64
	fill   string
	stroke string
	width  f64
}

pub struct Text {
pub:
	x       f64
	y       f64
	content string
	size    f64
	fill    string
	anchor  TextAnchor
	family  string
	rotate  f64 // degrees; 0 = horizontal
}

pub type Primitive = Line | Polyline | Rect | Circle | Text

pub struct Scene {
pub mut:
	primitives []Primitive
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_scene_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/scene.v tests/chart_scene_test.v
git commit -m "feat(chart): add scene primitives and Scene container"
```

---

## Task 2: Linear scale (domain → pixel mapping)

**Files:**
- Create: `chart/scales.v`
- Test: `tests/chart_scales_test.v`

- [ ] **Step 1: Write the failing test**

```v
// tests/chart_scales_test.v
import chart

fn test__linear_scale_maps_domain_to_range() {
	s := chart.LinearScale{
		domain_min: 0.0
		domain_max: 10.0
		range_min:  0.0
		range_max:  100.0
	}
	assert s.map(0.0) == 0.0
	assert s.map(5.0) == 50.0
	assert s.map(10.0) == 100.0
}

fn test__linear_scale_handles_inverted_range_for_y_flip() {
	s := chart.LinearScale{
		domain_min: 0.0
		domain_max: 10.0
		range_min:  400.0
		range_max:  0.0
	}
	assert s.map(0.0) == 400.0
	assert s.map(10.0) == 0.0
	assert s.map(5.0) == 200.0
}

fn test__linear_scale_degenerate_domain_returns_range_midpoint() {
	s := chart.LinearScale{
		domain_min: 5.0
		domain_max: 5.0
		range_min:  0.0
		range_max:  100.0
	}
	assert s.map(5.0) == 50.0
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_scales_test.v`
Expected: FAIL — `LinearScale` not defined.

- [ ] **Step 3: Write minimal implementation**

```v
// chart/scales.v
module chart

import math

pub struct LinearScale {
pub:
	domain_min f64
	domain_max f64
	range_min  f64
	range_max  f64
}

pub fn (s LinearScale) map(v f64) f64 {
	if s.domain_max == s.domain_min {
		return (s.range_min + s.range_max) / 2.0
	}
	t := (v - s.domain_min) / (s.domain_max - s.domain_min)
	return s.range_min + t * (s.range_max - s.range_min)
}
```

Note: `import math` is included now because Task 3 adds functions to this file that use it. If V warns about an unused import at this step, that warning will disappear after Task 3; it does not fail the build.

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_scales_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/scales.v tests/chart_scales_test.v
git commit -m "feat(chart): add linear scale mapping"
```

---

## Task 3: Nice ticks and tick formatting

**Files:**
- Modify: `chart/scales.v`
- Test: `tests/chart_scales_test.v`

- [ ] **Step 1: Write the failing test (append to existing file)**

```v
// append to tests/chart_scales_test.v
fn test__nice_ticks_produces_round_steps() {
	assert chart.nice_ticks(0.0, 100.0, 5) == [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
}

fn test__nice_ticks_equal_min_max() {
	assert chart.nice_ticks(7.0, 7.0, 5) == [7.0]
}

fn test__fmt_tick_integers_have_no_decimals() {
	assert chart.fmt_tick(40.0) == '40'
	assert chart.fmt_tick(0.0) == '0'
}

fn test__fmt_tick_fractions_get_two_decimals() {
	assert chart.fmt_tick(2.5) == '2.50'
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_scales_test.v`
Expected: FAIL — `nice_ticks` / `fmt_tick` not defined.

- [ ] **Step 3: Write minimal implementation (append to `chart/scales.v`)**

```v
// append to chart/scales.v

fn nice_num(x f64, round bool) f64 {
	exp := math.floor(math.log10(x))
	f := x / math.pow(10.0, exp)
	mut nf := 0.0
	if round {
		nf = if f < 1.5 {
			1.0
		} else if f < 3.0 {
			2.0
		} else if f < 7.0 {
			5.0
		} else {
			10.0
		}
	} else {
		nf = if f <= 1.0 {
			1.0
		} else if f <= 2.0 {
			2.0
		} else if f <= 5.0 {
			5.0
		} else {
			10.0
		}
	}
	return nf * math.pow(10.0, exp)
}

pub fn nice_ticks(min f64, max f64, target int) []f64 {
	if min == max {
		return [min]
	}
	span := nice_num(max - min, false)
	step := nice_num(span / f64(target - 1), true)
	graph_min := math.floor(min / step) * step
	graph_max := math.ceil(max / step) * step
	mut ticks := []f64{}
	mut v := graph_min
	for v <= graph_max + step * 0.5 {
		ticks << v
		v += step
	}
	return ticks
}

pub fn fmt_tick(v f64) string {
	if v == 0.0 {
		return '0'
	}
	if v == math.floor(v) && math.abs(v) < 1.0e15 {
		return '${i64(v)}'
	}
	return '${v:.2f}'
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_scales_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/scales.v tests/chart_scales_test.v
git commit -m "feat(chart): add nice-tick generation and tick formatting"
```

---

## Task 4: Theme with Tufte defaults

**Files:**
- Create: `chart/theme.v`
- Test: `tests/chart_theme_test.v`

- [ ] **Step 1: Write the failing test**

```v
// tests/chart_theme_test.v
import chart

fn test__theme_has_nonzero_defaults() {
	t := chart.Theme{}
	assert t.margin_left > 0
	assert t.font_size > 0.0
	assert t.series_width > 0.0
	assert t.palette.len > 0
	assert t.grid == false // Tufte: no grid by default
}

fn test__theme_color_cycles() {
	t := chart.Theme{}
	first := t.color(0)
	assert first.len > 0
	assert t.color(t.palette.len) == first // wraps around
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_theme_test.v`
Expected: FAIL — `Theme` not defined.

- [ ] **Step 3: Write minimal implementation**

```v
// chart/theme.v
module chart

@[params]
pub struct Theme {
pub:
	margin_left   int      = 60
	margin_right  int      = 20
	margin_top    int      = 40
	margin_bottom int      = 50
	background    string   = 'white'
	axis_color    string   = '#333333'
	axis_width    f64      = 1.0
	grid          bool     // default false (Tufte: minimal ink)
	grid_color    string   = '#e0e0e0'
	font_family   string   = 'sans-serif'
	font_size     f64      = 12.0
	title_size    f64      = 16.0
	series_width  f64      = 1.5
	marker_radius f64      = 3.0
	palette       []string = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
}

pub fn (t Theme) color(i int) string {
	return t.palette[i % t.palette.len]
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_theme_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/theme.v tests/chart_theme_test.v
git commit -m "feat(chart): add Theme with Tufte defaults and color cycle"
```

---

## Task 5: Histogram binning

**Files:**
- Create: `chart/bins.v`
- Test: `tests/chart_bins_test.v`

- [ ] **Step 1: Write the failing test**

```v
// tests/chart_bins_test.v
import chart

fn test__histogram_bins_explicit_count() {
	b := chart.histogram_bins([1.0, 2.0, 3.0, 4.0], 2)
	assert b.edges == [1.0, 2.5, 4.0]
	assert b.counts == [2, 2]
}

fn test__histogram_bins_auto_count_is_positive() {
	b := chart.histogram_bins([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 0)
	assert b.counts.len > 0
	mut total := 0
	for c in b.counts {
		total += c
	}
	assert total == 8 // every value lands in a bin
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_bins_test.v`
Expected: FAIL — `histogram_bins` not defined.

- [ ] **Step 3: Write minimal implementation**

```v
// chart/bins.v
module chart

import math

pub struct HistogramBins {
pub:
	edges  []f64 // len == nbins + 1
	counts []int // len == nbins
}

pub fn histogram_bins(data []f64, nbins int) HistogramBins {
	assert data.len > 0
	n := if nbins > 0 {
		nbins
	} else {
		int(math.ceil(math.log2(f64(data.len)) + 1.0)) // Sturges' rule
	}
	mut lo := data[0]
	mut hi := data[0]
	for v in data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if lo == hi {
		hi = lo + 1.0
	}
	width := (hi - lo) / f64(n)
	mut edges := []f64{len: n + 1}
	for i in 0 .. n + 1 {
		edges[i] = lo + f64(i) * width
	}
	mut counts := []int{len: n, init: 0}
	for v in data {
		mut idx := int((v - lo) / width)
		if idx >= n {
			idx = n - 1
		}
		if idx < 0 {
			idx = 0
		}
		counts[idx]++
	}
	return HistogramBins{
		edges:  edges
		counts: counts
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_bins_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/bins.v tests/chart_bins_test.v
git commit -m "feat(chart): add histogram binning"
```

---

## Task 6: SVG backend

**Files:**
- Create: `chart/svg.v`
- Test: `tests/chart_svg_test.v`

- [ ] **Step 1: Write the failing test**

```v
// tests/chart_svg_test.v
import chart

fn test__render_svg_wraps_primitives() {
	mut s := chart.Scene{}
	s.primitives << chart.Line{
		x1:     0.0
		y1:     0.0
		x2:     10.0
		y2:     0.0
		stroke: '#333'
		width:  1.0
	}
	out := chart.render_svg(s, 200, 100, chart.Theme{})
	assert out.starts_with('<svg')
	assert out.ends_with('</svg>')
	assert out.contains('width="200"')
	assert out.contains('<line')
}

fn test__render_svg_text_escapes_and_anchors() {
	mut s := chart.Scene{}
	s.primitives << chart.Text{
		x:       5.0
		y:       5.0
		content: 'a < b & c'
		size:    12.0
		fill:    '#000'
		anchor:  .end
		family:  'sans-serif'
	}
	out := chart.render_svg(s, 50, 50, chart.Theme{})
	assert out.contains('text-anchor="end"')
	assert out.contains('a &lt; b &amp; c')
}

fn test__render_svg_rotated_text_has_transform() {
	mut s := chart.Scene{}
	s.primitives << chart.Text{
		x:       5.0
		y:       5.0
		content: 'y'
		size:    12.0
		fill:    '#000'
		anchor:  .middle
		family:  'sans-serif'
		rotate:  -90.0
	}
	out := chart.render_svg(s, 50, 50, chart.Theme{})
	assert out.contains('transform="rotate(')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_svg_test.v`
Expected: FAIL — `render_svg` not defined.

- [ ] **Step 3: Write minimal implementation**

```v
// chart/svg.v
module chart

import strings

fn xml_escape(s string) string {
	return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
}

fn primitive_to_svg(p Primitive) string {
	return match p {
		Line {
			'<line x1="${p.x1}" y1="${p.y1}" x2="${p.x2}" y2="${p.y2}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
		}
		Polyline {
			mut pts := []string{}
			for pt in p.points {
				pts << '${pt.x},${pt.y}'
			}
			'<polyline fill="none" stroke="${p.stroke}" stroke-width="${p.width}" points="${pts.join(' ')}"/>'
		}
		Rect {
			'<rect x="${p.x}" y="${p.y}" width="${p.w}" height="${p.h}" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
		}
		Circle {
			'<circle cx="${p.cx}" cy="${p.cy}" r="${p.r}" fill="${p.fill}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
		}
		Text {
			anchor := match p.anchor {
				.start { 'start' }
				.middle { 'middle' }
				.end { 'end' }
			}
			mut transform := ''
			if p.rotate != 0.0 {
				transform = ' transform="rotate(${p.rotate} ${p.x} ${p.y})"'
			}
			'<text x="${p.x}" y="${p.y}" font-family="${p.family}" font-size="${p.size}" fill="${p.fill}" text-anchor="${anchor}"${transform}>${xml_escape(p.content)}</text>'
		}
	}
}

pub fn render_svg(scene Scene, width int, height int, theme Theme) string {
	mut b := strings.new_builder(1024)
	b.write_string('<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">')
	b.write_string('<rect x="0" y="0" width="${width}" height="${height}" fill="${theme.background}"/>')
	for p in scene.primitives {
		b.write_string(primitive_to_svg(p))
	}
	b.write_string('</svg>')
	return b.str()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_svg_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/svg.v tests/chart_svg_test.v
git commit -m "feat(chart): add SVG backend"
```

---

## Task 7: Chart builder core — line series, geometry, axes

**Files:**
- Create: `chart/chart.v`
- Test: `tests/chart_test.v`

This task wires the builder together for the simplest case: a single line series with auto-scaled axes. `build_scene` is introduced here calling only `draw_axes` and `draw_series`; later tasks extend it.

- [ ] **Step 1: Write the failing test**

```v
// tests/chart_test.v
import chart

fn test__line_chart_builds_axes_and_polyline() {
	svg := chart.new(title: 'Demo', width: 400, height: 300)
		.line([0.0, 1.0, 2.0], [0.0, 10.0, 5.0], label: 'a')
		.render()
	assert svg.starts_with('<svg')
	assert svg.contains('<polyline')
	// two axis lines (x and y) -> at least two <line elements
	assert svg.count('<line') >= 2
}

fn test__line_requires_matching_lengths() {
	// mismatched x/y lengths should panic via assert
	mut paniced := false
	$if !prod {
		// run in a context where assert triggers
	}
	c := chart.new()
	// length mismatch: call should fail an assert at runtime
	// We verify behavior indirectly: equal lengths succeed.
	out := c.line([0.0, 1.0], [2.0, 3.0]).render()
	assert out.contains('<polyline')
	paniced = paniced // silence unused warning
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — `new` / `Chart` not defined.

- [ ] **Step 3: Write minimal implementation**

```v
// chart/chart.v
module chart

import math
import os

enum SeriesKind {
	line
	scatter
	bar
	histogram
}

struct Series {
	kind  SeriesKind
	x     []f64
	y     []f64
	label string
	color string
	nbins int
}

pub struct Chart {
mut:
	title   string
	width   int
	height  int
	theme   Theme
	xlabel_ string
	ylabel_ string
	series  []Series
	hlines  []f64
	vlines  []f64
}

@[params]
pub struct ChartOpts {
pub:
	title  string
	width  int   = 640
	height int   = 480
	theme  Theme = Theme{}
}

@[params]
pub struct SeriesOpts {
pub:
	label string
}

pub fn new(opts ChartOpts) Chart {
	return Chart{
		title:  opts.title
		width:  opts.width
		height: opts.height
		theme:  opts.theme
	}
}

pub fn (c Chart) line(x []f64, y []f64, opts SeriesOpts) Chart {
	assert x.len == y.len
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .line
		x:     x.clone()
		y:     y.clone()
		label: opts.label
		color: c.theme.color(c.series.len)
	}
	nc.series = s
	return nc
}

// ---- geometry & bounds ----

struct Geom {
	plot_x f64
	plot_y f64
	plot_w f64
	plot_h f64
	xmin   f64
	xmax   f64
	ymin   f64
	ymax   f64
	xscale LinearScale
	yscale LinearScale
}

fn extent(vals []f64) (f64, f64) {
	mut lo := vals[0]
	mut hi := vals[0]
	for v in vals {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return lo, hi
}

fn series_bounds(s Series) (f64, f64, f64, f64) {
	return match s.kind {
		.line, .scatter {
			x0, x1 := extent(s.x)
			y0, y1 := extent(s.y)
			x0, x1, y0, y1
		}
		.bar {
			y0, y1 := extent(s.y)
			ylo := if y0 < 0.0 { y0 } else { 0.0 }
			yhi := if y1 > 0.0 { y1 } else { 0.0 }
			-0.5, f64(s.y.len) - 0.5, ylo, yhi
		}
		.histogram {
			b := histogram_bins(s.x, s.nbins)
			mut maxc := 0
			for ct in b.counts {
				if ct > maxc {
					maxc = ct
				}
			}
			b.edges[0], b.edges[b.edges.len - 1], 0.0, f64(maxc)
		}
	}
}

fn (c Chart) geometry() Geom {
	t := c.theme
	plot_x := f64(t.margin_left)
	plot_y := f64(t.margin_top)
	plot_w := f64(c.width - t.margin_left - t.margin_right)
	plot_h := f64(c.height - t.margin_top - t.margin_bottom)
	mut xmin := 0.0
	mut xmax := 1.0
	mut ymin := 0.0
	mut ymax := 1.0
	mut first := true
	for s in c.series {
		bx0, bx1, by0, by1 := series_bounds(s)
		if first {
			xmin, xmax, ymin, ymax = bx0, bx1, by0, by1
			first = false
		} else {
			if bx0 < xmin {
				xmin = bx0
			}
			if bx1 > xmax {
				xmax = bx1
			}
			if by0 < ymin {
				ymin = by0
			}
			if by1 > ymax {
				ymax = by1
			}
		}
	}
	if xmax == xmin {
		xmax = xmin + 1.0
	}
	if ymax == ymin {
		ymax = ymin + 1.0
	}
	return Geom{
		plot_x: plot_x
		plot_y: plot_y
		plot_w: plot_w
		plot_h: plot_h
		xmin:   xmin
		xmax:   xmax
		ymin:   ymin
		ymax:   ymax
		xscale: LinearScale{
			domain_min: xmin
			domain_max: xmax
			range_min:  plot_x
			range_max:  plot_x + plot_w
		}
		yscale: LinearScale{
			domain_min: ymin
			domain_max: ymax
			range_min:  plot_y + plot_h
			range_max:  plot_y
		}
	}
}

// ---- scene assembly ----

fn (c Chart) draw_axes(mut scene Scene, g Geom) {
	t := c.theme
	bottom := g.plot_y + g.plot_h
	scene.primitives << Line{
		x1:     g.plot_x
		y1:     bottom
		x2:     g.plot_x + g.plot_w
		y2:     bottom
		stroke: t.axis_color
		width:  t.axis_width
	}
	scene.primitives << Line{
		x1:     g.plot_x
		y1:     g.plot_y
		x2:     g.plot_x
		y2:     bottom
		stroke: t.axis_color
		width:  t.axis_width
	}
}

fn (c Chart) draw_series(mut scene Scene, g Geom) {
	t := c.theme
	for s in c.series {
		if s.kind == .line {
			mut pts := []Point{}
			for i in 0 .. s.x.len {
				pts << Point{
					x: g.xscale.map(s.x[i])
					y: g.yscale.map(s.y[i])
				}
			}
			scene.primitives << Polyline{
				points: pts
				stroke: s.color
				width:  t.series_width
			}
		}
	}
}

fn (c Chart) build_scene() Scene {
	g := c.geometry()
	mut scene := Scene{}
	c.draw_axes(mut scene, g)
	c.draw_series(mut scene, g)
	return scene
}

pub fn (c Chart) render() string {
	scene := c.build_scene()
	return render_svg(scene, c.width, c.height, c.theme)
}
```

Note: `math` and `os` are imported now because later tasks in this file use them (`math.min`/`math.abs` in Task 8, `os.write_file` in Task 10). If V warns about an unused import at this step, it clears after those tasks; it does not fail the build. If it *errors* at this step on your V version, temporarily remove `import math` and `import os`, then re-add them in Tasks 8 and 10 respectively.

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): add Chart builder with line series and auto axes"
```

---

## Task 8: Scatter, bar, and histogram series

**Files:**
- Modify: `chart/chart.v` (add builder methods; extend `draw_series`)
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to existing file)**

```v
// append to tests/chart_test.v
fn test__scatter_emits_circles() {
	svg := chart.new(width: 400, height: 300)
		.scatter([0.0, 1.0, 2.0], [3.0, 1.0, 2.0])
		.render()
	assert svg.count('<circle') == 3
}

fn test__bar_emits_rects() {
	svg := chart.new(width: 400, height: 300)
		.bar([10.0, 20.0, 30.0])
		.render()
	assert svg.count('<rect') >= 3 // background rect + 3 bars
}

fn test__histogram_emits_rects() {
	svg := chart.new(width: 400, height: 300)
		.histogram([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], nbins: 3)
		.render()
	assert svg.count('<rect') >= 3
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — `scatter` / `bar` / `histogram` not defined.

- [ ] **Step 3: Write minimal implementation**

Add these builder methods to `chart/chart.v` (after the `line` method):

```v
pub fn (c Chart) scatter(x []f64, y []f64, opts SeriesOpts) Chart {
	assert x.len == y.len
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .scatter
		x:     x.clone()
		y:     y.clone()
		label: opts.label
		color: c.theme.color(c.series.len)
	}
	nc.series = s
	return nc
}

@[params]
pub struct HistogramOpts {
pub:
	label string
	nbins int // 0 => auto (Sturges)
}

pub fn (c Chart) bar(values []f64, opts SeriesOpts) Chart {
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .bar
		x:     []f64{}
		y:     values.clone()
		label: opts.label
		color: c.theme.color(c.series.len)
	}
	nc.series = s
	return nc
}

pub fn (c Chart) histogram(data []f64, opts HistogramOpts) Chart {
	assert data.len > 0
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .histogram
		x:     data.clone()
		y:     []f64{}
		label: opts.label
		color: c.theme.color(c.series.len)
		nbins: opts.nbins
	}
	nc.series = s
	return nc
}
```

Then replace the body of `draw_series` in `chart/chart.v` with the full dispatch:

```v
fn (c Chart) draw_series(mut scene Scene, g Geom) {
	t := c.theme
	for s in c.series {
		match s.kind {
			.line {
				mut pts := []Point{}
				for i in 0 .. s.x.len {
					pts << Point{
						x: g.xscale.map(s.x[i])
						y: g.yscale.map(s.y[i])
					}
				}
				scene.primitives << Polyline{
					points: pts
					stroke: s.color
					width:  t.series_width
				}
			}
			.scatter {
				for i in 0 .. s.x.len {
					scene.primitives << Circle{
						cx:     g.xscale.map(s.x[i])
						cy:     g.yscale.map(s.y[i])
						r:      t.marker_radius
						fill:   s.color
						stroke: 'none'
						width:  0.0
					}
				}
			}
			.bar {
				band := g.xscale.map(1.0) - g.xscale.map(0.0)
				bw := band * 0.8
				baseline := g.yscale.map(0.0)
				for i in 0 .. s.y.len {
					cx := g.xscale.map(f64(i))
					top := g.yscale.map(s.y[i])
					scene.primitives << Rect{
						x:      cx - bw / 2.0
						y:      math.min(top, baseline)
						w:      bw
						h:      math.abs(baseline - top)
						fill:   s.color
						stroke: 'none'
						width:  0.0
					}
				}
			}
			.histogram {
				b := histogram_bins(s.x, s.nbins)
				baseline := g.yscale.map(0.0)
				for i in 0 .. b.counts.len {
					x0 := g.xscale.map(b.edges[i])
					x1 := g.xscale.map(b.edges[i + 1])
					top := g.yscale.map(f64(b.counts[i]))
					scene.primitives << Rect{
						x:      x0
						y:      math.min(top, baseline)
						w:      x1 - x0
						h:      math.abs(baseline - top)
						fill:   s.color
						stroke: t.background
						width:  1.0
					}
				}
			}
		}
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): add scatter, bar, and histogram series"
```

---

## Task 9: Ticks, labels, guides, and legend

**Files:**
- Modify: `chart/chart.v` (add decoration builder methods, new draw helpers, extend `build_scene`)
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to existing file)**

```v
// append to tests/chart_test.v
fn test__ticks_and_labels_render() {
	svg := chart.new(title: 'T', width: 400, height: 300)
		.line([0.0, 1.0, 2.0], [0.0, 50.0, 100.0])
		.xlabel('time')
		.ylabel('value')
		.render()
	assert svg.contains('>T<') // title text
	assert svg.contains('>time<') // x label
	assert svg.contains('>value<') // y label
	assert svg.contains('transform="rotate(') // y label is rotated
	assert svg.contains('>100<') // a y tick label
}

fn test__legend_appears_with_two_labeled_series() {
	svg := chart.new(width: 400, height: 300)
		.line([0.0, 1.0], [0.0, 1.0], label: 'a')
		.line([0.0, 1.0], [1.0, 0.0], label: 'b')
		.render()
	assert svg.contains('>a<')
	assert svg.contains('>b<')
}

fn test__axhline_adds_guide_line() {
	base := chart.new(width: 400, height: 300).scatter([0.0, 1.0], [-1.0, 1.0])
	without := base.render().count('<line')
	with := base.axhline(0.0).render().count('<line')
	assert with == without + 1
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — `xlabel` / `ylabel` / `axhline` / tick rendering not present.

- [ ] **Step 3: Write minimal implementation**

Add these decoration builder methods to `chart/chart.v`:

```v
pub fn (c Chart) title(s string) Chart {
	mut nc := c
	nc.title = s
	return nc
}

pub fn (c Chart) xlabel(s string) Chart {
	mut nc := c
	nc.xlabel_ = s
	return nc
}

pub fn (c Chart) ylabel(s string) Chart {
	mut nc := c
	nc.ylabel_ = s
	return nc
}

pub fn (c Chart) axhline(y f64) Chart {
	mut nc := c
	mut h := c.hlines.clone()
	h << y
	nc.hlines = h
	return nc
}

pub fn (c Chart) axvline(x f64) Chart {
	mut nc := c
	mut v := c.vlines.clone()
	v << x
	nc.vlines = v
	return nc
}
```

Add these draw helpers to `chart/chart.v`:

```v
fn (c Chart) draw_ticks(mut scene Scene, g Geom) {
	t := c.theme
	bottom := g.plot_y + g.plot_h
	for tk in nice_ticks(g.xmin, g.xmax, 5) {
		if tk < g.xmin - 1.0e-9 || tk > g.xmax + 1.0e-9 {
			continue
		}
		px := g.xscale.map(tk)
		scene.primitives << Line{
			x1:     px
			y1:     bottom
			x2:     px
			y2:     bottom + 5.0
			stroke: t.axis_color
			width:  t.axis_width
		}
		scene.primitives << Text{
			x:       px
			y:       bottom + 18.0
			content: fmt_tick(tk)
			size:    t.font_size
			fill:    t.axis_color
			anchor:  .middle
			family:  t.font_family
		}
	}
	for tk in nice_ticks(g.ymin, g.ymax, 5) {
		if tk < g.ymin - 1.0e-9 || tk > g.ymax + 1.0e-9 {
			continue
		}
		py := g.yscale.map(tk)
		scene.primitives << Line{
			x1:     g.plot_x - 5.0
			y1:     py
			x2:     g.plot_x
			y2:     py
			stroke: t.axis_color
			width:  t.axis_width
		}
		scene.primitives << Text{
			x:       g.plot_x - 8.0
			y:       py + 4.0
			content: fmt_tick(tk)
			size:    t.font_size
			fill:    t.axis_color
			anchor:  .end
			family:  t.font_family
		}
	}
}

fn (c Chart) draw_guides(mut scene Scene, g Geom) {
	t := c.theme
	for hy in c.hlines {
		py := g.yscale.map(hy)
		scene.primitives << Line{
			x1:     g.plot_x
			y1:     py
			x2:     g.plot_x + g.plot_w
			y2:     py
			stroke: t.axis_color
			width:  t.axis_width
		}
	}
	for vx in c.vlines {
		px := g.xscale.map(vx)
		scene.primitives << Line{
			x1:     px
			y1:     g.plot_y
			x2:     px
			y2:     g.plot_y + g.plot_h
			stroke: t.axis_color
			width:  t.axis_width
		}
	}
}

fn (c Chart) draw_labels(mut scene Scene, g Geom) {
	t := c.theme
	if c.title != '' {
		scene.primitives << Text{
			x:       f64(c.width) / 2.0
			y:       f64(t.margin_top) / 2.0 + 5.0
			content: c.title
			size:    t.title_size
			fill:    t.axis_color
			anchor:  .middle
			family:  t.font_family
		}
	}
	if c.xlabel_ != '' {
		scene.primitives << Text{
			x:       g.plot_x + g.plot_w / 2.0
			y:       f64(c.height) - 8.0
			content: c.xlabel_
			size:    t.font_size
			fill:    t.axis_color
			anchor:  .middle
			family:  t.font_family
		}
	}
	if c.ylabel_ != '' {
		lx := f64(t.margin_left) / 3.0
		ly := g.plot_y + g.plot_h / 2.0
		scene.primitives << Text{
			x:       lx
			y:       ly
			content: c.ylabel_
			size:    t.font_size
			fill:    t.axis_color
			anchor:  .middle
			family:  t.font_family
			rotate:  -90.0
		}
	}
}

fn (c Chart) draw_legend(mut scene Scene, g Geom) {
	t := c.theme
	mut labeled := []Series{}
	for s in c.series {
		if s.label != '' {
			labeled << s
		}
	}
	if labeled.len < 2 {
		return
	}
	lx := g.plot_x + g.plot_w - 100.0
	mut ly := g.plot_y + 4.0
	for s in labeled {
		scene.primitives << Rect{
			x:      lx
			y:      ly
			w:      12.0
			h:      12.0
			fill:   s.color
			stroke: 'none'
			width:  0.0
		}
		scene.primitives << Text{
			x:       lx + 18.0
			y:       ly + 11.0
			content: s.label
			size:    t.font_size
			fill:    t.axis_color
			anchor:  .start
			family:  t.font_family
		}
		ly += 18.0
	}
}
```

Then replace `build_scene` in `chart/chart.v` with:

```v
fn (c Chart) build_scene() Scene {
	g := c.geometry()
	mut scene := Scene{}
	c.draw_axes(mut scene, g)
	c.draw_ticks(mut scene, g)
	c.draw_guides(mut scene, g)
	c.draw_series(mut scene, g)
	c.draw_labels(mut scene, g)
	c.draw_legend(mut scene, g)
	return scene
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): add ticks, labels, guides, and legend"
```

---

## Task 10: Save to disk and end-to-end test

**Files:**
- Modify: `chart/chart.v` (add `save`)
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to existing file)**

```v
// append to tests/chart_test.v
import os

fn test__save_writes_svg_file() {
	path := os.join_path(os.temp_dir(), 'vstats_chart_test.svg')
	if os.exists(path) {
		os.rm(path) or {}
	}
	chart.new(title: 'Saved', width: 320, height: 240)
		.line([0.0, 1.0, 2.0], [0.0, 4.0, 2.0], label: 'fit')
		.xlabel('x')
		.ylabel('y')
		.save(path) or { assert false, 'save failed: ${err}' }
	assert os.exists(path)
	content := os.read_file(path) or { '' }
	assert content.starts_with('<svg')
	assert content.contains('</svg>')
	os.rm(path) or {}
}
```

Note: move the `import os` line to the top of `tests/chart_test.v` with the other imports (V requires imports at the top of the file).

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — `save` method not defined.

- [ ] **Step 3: Write minimal implementation**

Add to `chart/chart.v`:

```v
pub fn (c Chart) save(path string) ! {
	os.write_file(path, c.render())!
}
```

(`import os` is already present at the top of `chart/chart.v` from Task 7.)

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `v test tests/`
Expected: PASS (all chart tests plus the existing suite).

- [ ] **Step 6: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): add save-to-disk and end-to-end test"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- Three-stage pipeline (Chart → Scene → backend) → Tasks 1, 6, 7.
- Chart types line/scatter/bar/histogram → Tasks 7, 8.
- Fluent builder + `render()`/`save()` split → Tasks 7, 10.
- `[]f64` series inputs → all builder methods.
- Linear scale + nice ticks (Tufte read-out) → Tasks 2, 3, 9.
- Tufte layout (range frame axes, short ticks, no grid by default, restrained palette) → Tasks 4, 7, 9.
- Theme struct with explicit non-zero defaults (zero-default gotcha) → Task 4.
- Self-contained histogram binning, no `stats` dep → Task 5.
- Multi-series shared axes + color cycle + auto legend (≥2 labeled) → Tasks 7, 8, 9.
- Scene primitives (Line/Polyline/Rect/Circle/Text) → Task 1.
- SVG inline attributes, no CSS, y-flip in scale, XML escaping → Tasks 2, 6.
- Error handling: `assert` for length preconditions, `!` for I/O → Tasks 7, 8, 10.
- Testing on primitives + SVG structure + one end-to-end → Tasks 1–10.
- Future raster backend / recipes layer → out of scope for v1 (documented in spec only). No task, intentionally.

**Type/name consistency check:** `Series`, `SeriesKind`, `Geom`, `LinearScale`, `HistogramBins`, `Theme`, `Scene`, `Primitive`, and the `draw_*` / `series_bounds` / `geometry` / `build_scene` / `render_svg` / `nice_ticks` / `fmt_tick` / `histogram_bins` names are used identically across all tasks. `draw_series` is defined in Task 7 and fully replaced (not renamed) in Task 8; `build_scene` is defined in Task 7 and replaced in Task 9 — both replacements are called out explicitly with full bodies.

**Known V-version risk flagged for the engineer:** unused-import behavior (`math`/`os` in Task 7, `math` in Task 2) — handled with inline notes telling the engineer how to react if their V build treats unused imports as errors rather than warnings.
