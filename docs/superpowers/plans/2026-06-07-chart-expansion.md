# Chart Module Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `chart` module with CI bands, area fills, value labels, a left-aligned title + subtitle, per-series color override, gridlines, and error bars.

**Architecture:** Reuse the existing `Chart → Scene → SVG backend` pipeline. Add one new filled `Polygon` primitive (powers bands and areas); thread new options through `SeriesOpts`/`Series`; add focused `draw_*` helpers wired into `build_scene`. Each feature is independently testable against the rendered SVG / scene.

**Tech Stack:** V 0.5.1, stdlib only. Tests live in `tests/` and use `import chart`. Run a single test file with `v test tests/chart_test.v`; run all with `v test tests/`.

---

## Conventions the engineer must know

- V errors on unused **imports** and unused **variables** (struct fields may be unused).
- Inside `match p { Polygon { ... } }`, the matched value is smart-cast.
- A `match s.kind` over an enum must be **exhaustive**; non-handled arms need `else {}`.
- f64 fields must be assigned f64 expressions (use `f64(...)`, `1.0`-style literals).
- The whole SVG is one line; in shell checks use `grep -o ... | wc -l`, not `grep -c`.
- Builder methods take a value receiver and return a new `Chart` (clone arrays before appending), matching the existing methods.

---

## Task 1: Add the `Polygon` primitive

**Files:**
- Modify: `chart/scene.v`
- Modify: `chart/svg.v`
- Test: `tests/chart_svg_test.v`

- [ ] **Step 1: Write the failing test (append to `tests/chart_svg_test.v`)**

```v
fn test__render_svg_polygon_has_fill_opacity() {
	mut s := chart.Scene{}
	s.primitives << chart.Polygon{
		points:  [chart.Point{x: 0.0, y: 0.0}, chart.Point{x: 10.0, y: 0.0},
			chart.Point{x: 10.0, y: 10.0}]
		fill:    '#abcdef'
		opacity: 0.2
		stroke:  'none'
		width:   0.0
	}
	out := chart.render_svg(s, 50, 50, chart.Theme{})
	assert out.contains('<polygon')
	assert out.contains('fill-opacity="0.2"')
	assert out.contains('#abcdef')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_svg_test.v`
Expected: FAIL — `Polygon` not defined.

- [ ] **Step 3: Add the `Polygon` struct and extend `Primitive` in `chart/scene.v`**

Find:
```v
pub type Primitive = Line | Polyline | Rect | Circle | Text
```
Replace with:
```v
pub struct Polygon {
pub:
	points  []Point
	fill    string
	opacity f64
	stroke  string
	width   f64
}

pub type Primitive = Line | Polyline | Rect | Circle | Text | Polygon
```

- [ ] **Step 4: Render `Polygon` in `chart/svg.v`**

Find the `Text` arm's closing inside `primitive_to_svg` (the last arm before the match closes):
```v
			'<text x="${p.x}" y="${p.y}" font-family="${p.family}" font-size="${p.size}" fill="${p.fill}" text-anchor="${anchor}"${transform}>${xml_escape(p.content)}</text>'
		}
	}
}
```
Replace with:
```v
			'<text x="${p.x}" y="${p.y}" font-family="${p.family}" font-size="${p.size}" fill="${p.fill}" text-anchor="${anchor}"${transform}>${xml_escape(p.content)}</text>'
		}
		Polygon {
			mut pts := []string{}
			for pt in p.points {
				pts << '${pt.x},${pt.y}'
			}
			op := if p.opacity <= 0.0 { 1.0 } else { p.opacity }
			'<polygon points="${pts.join(' ')}" fill="${p.fill}" fill-opacity="${op}" stroke="${p.stroke}" stroke-width="${p.width}"/>'
		}
	}
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `v test tests/chart_svg_test.v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add chart/scene.v chart/svg.v tests/chart_svg_test.v
git commit -m "feat(chart): add filled Polygon primitive"
```

---

## Task 2: Add theme fields (fill opacity, subtitle styling)

**Files:**
- Modify: `chart/theme.v`
- Test: `tests/chart_theme_test.v`

- [ ] **Step 1: Write the failing test (append to `tests/chart_theme_test.v`)**

```v
fn test__theme_expansion_defaults() {
	t := chart.Theme{}
	assert t.fill_opacity > 0.0
	assert t.subtitle_size > 0.0
	assert t.subtitle_size < t.title_size
	assert t.subtitle_color.len > 0
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_theme_test.v`
Expected: FAIL — unknown field `fill_opacity`.

- [ ] **Step 3: Add fields in `chart/theme.v`**

Find:
```v
	marker_radius f64      = 3.0
	palette       []string = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```
Replace with:
```v
	marker_radius  f64      = 3.0
	fill_opacity   f64      = 0.2
	subtitle_size  f64      = 12.0
	subtitle_color string   = '#666666'
	palette        []string = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

- [ ] **Step 4: Run test to verify it passes**

Run: `v test tests/chart_theme_test.v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add chart/theme.v tests/chart_theme_test.v
git commit -m "feat(chart): add fill_opacity and subtitle theme fields"
```

---

## Task 3: Extend the series data model + per-series color override

**Files:**
- Modify: `chart/chart.v`
- Test: `tests/chart_test.v`

This task extends `Series`, `SeriesOpts`, `HistogramOpts`, and threads the new options (color override, error array, value-label fields) through the existing `line`/`scatter`/`bar`/`histogram` methods. Rendering of the new fields comes in later tasks; here we ship the **color override** feature.

- [ ] **Step 1: Write the failing test (append to `tests/chart_test.v`)**

```v
fn test__series_color_override() {
	svg := chart.new(width: 300, height: 200)
		.line([0.0, 1.0], [0.0, 1.0], color: '#123456')
		.render()
	assert svg.contains('#123456')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — unknown field `color` in `SeriesOpts`.

- [ ] **Step 3: Extend the structs in `chart/chart.v`**

Find:
```v
struct Series {
	kind  SeriesKind
	x     []f64
	y     []f64
	label string
	color string
	nbins int
}
```
Replace with:
```v
struct Series {
	kind        SeriesKind
	x           []f64
	y           []f64
	lo          []f64
	hi          []f64
	err         []f64
	label       string
	color       string
	nbins       int
	show_values bool
	labels      []string
}
```

Find:
```v
pub struct SeriesOpts {
pub:
	label string
}
```
Replace with:
```v
pub struct SeriesOpts {
pub:
	label       string
	color       string
	show_values bool
	labels      []string
	err         []f64
}
```

Find:
```v
pub struct HistogramOpts {
pub:
	label string
	nbins int // 0 => auto (Sturges)
}
```
Replace with:
```v
pub struct HistogramOpts {
pub:
	label string
	nbins int // 0 => auto (Sturges)
	color string
}
```

- [ ] **Step 4: Thread the new options through the four series methods**

Find the `line` method:
```v
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
```
Replace with:
```v
pub fn (c Chart) line(x []f64, y []f64, opts SeriesOpts) Chart {
	assert x.len == y.len
	if opts.labels.len > 0 {
		assert opts.labels.len == x.len
	}
	if opts.err.len > 0 {
		assert opts.err.len == x.len
	}
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:        .line
		x:           x.clone()
		y:           y.clone()
		err:         opts.err.clone()
		label:       opts.label
		color:       if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
		show_values: opts.show_values
		labels:      opts.labels.clone()
	}
	nc.series = s
	return nc
}
```

Find the `scatter` method:
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
```
Replace with:
```v
pub fn (c Chart) scatter(x []f64, y []f64, opts SeriesOpts) Chart {
	assert x.len == y.len
	if opts.labels.len > 0 {
		assert opts.labels.len == x.len
	}
	if opts.err.len > 0 {
		assert opts.err.len == x.len
	}
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:        .scatter
		x:           x.clone()
		y:           y.clone()
		err:         opts.err.clone()
		label:       opts.label
		color:       if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
		show_values: opts.show_values
		labels:      opts.labels.clone()
	}
	nc.series = s
	return nc
}
```

Find the `bar` method:
```v
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
```
Replace with:
```v
pub fn (c Chart) bar(values []f64, opts SeriesOpts) Chart {
	if opts.labels.len > 0 {
		assert opts.labels.len == values.len
	}
	if opts.err.len > 0 {
		assert opts.err.len == values.len
	}
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:        .bar
		x:           []f64{}
		y:           values.clone()
		err:         opts.err.clone()
		label:       opts.label
		color:       if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
		show_values: opts.show_values
		labels:      opts.labels.clone()
	}
	nc.series = s
	return nc
}
```

Find the `histogram` method:
```v
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
Replace with:
```v
pub fn (c Chart) histogram(data []f64, opts HistogramOpts) Chart {
	assert data.len > 0
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .histogram
		x:     data.clone()
		y:     []f64{}
		label: opts.label
		color: if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
		nbins: opts.nbins
	}
	nc.series = s
	return nc
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `v test tests/chart_test.v`
Expected: PASS (new color-override test plus all existing chart tests).

- [ ] **Step 6: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): extend series data model and add per-series color override"
```

---

## Task 4: Left-aligned title + subtitle

**Files:**
- Modify: `chart/chart.v`
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to `tests/chart_test.v`)**

```v
fn test__left_title_and_subtitle() {
	svg := chart.new(title: 'Main', subtitle: 'desc here', width: 400, height: 300)
		.line([0.0, 1.0], [0.0, 1.0])
		.render()
	assert svg.contains('text-anchor="start"') // title/subtitle are left-aligned
	assert svg.contains('>Main<')
	assert svg.contains('>desc here<')
	assert svg.contains('#666666') // subtitle color
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — unknown field `subtitle` in `ChartOpts`.

- [ ] **Step 3: Add the subtitle field/option/method in `chart/chart.v`**

Find:
```v
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
```
Replace with:
```v
pub struct Chart {
mut:
	title    string
	subtitle string
	width    int
	height   int
	theme    Theme
	xlabel_  string
	ylabel_  string
	series   []Series
	hlines   []f64
	vlines   []f64
}
```

Find:
```v
pub struct ChartOpts {
pub:
	title  string
	width  int   = 640
	height int   = 480
	theme  Theme = Theme{}
}
```
Replace with:
```v
pub struct ChartOpts {
pub:
	title    string
	subtitle string
	width    int   = 640
	height   int   = 480
	theme    Theme = Theme{}
}
```

Find:
```v
pub fn new(opts ChartOpts) Chart {
	return Chart{
		title:  opts.title
		width:  opts.width
		height: opts.height
		theme:  opts.theme
	}
}
```
Replace with:
```v
pub fn new(opts ChartOpts) Chart {
	return Chart{
		title:    opts.title
		subtitle: opts.subtitle
		width:    opts.width
		height:   opts.height
		theme:    opts.theme
	}
}
```

Find the `title` method:
```v
pub fn (c Chart) title(s string) Chart {
	mut nc := c
	nc.title = s
	return nc
}
```
Replace with:
```v
pub fn (c Chart) title(s string) Chart {
	mut nc := c
	nc.title = s
	return nc
}

pub fn (c Chart) subtitle(s string) Chart {
	mut nc := c
	nc.subtitle = s
	return nc
}
```

- [ ] **Step 4: Rewrite the title block in `draw_labels`**

Find:
```v
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
```
Replace with:
```v
	if c.title != '' {
		scene.primitives << Text{
			x:       g.plot_x
			y:       t.title_size
			content: c.title
			size:    t.title_size
			fill:    t.axis_color
			anchor:  .start
			family:  t.font_family
		}
	}
	if c.subtitle != '' {
		scene.primitives << Text{
			x:       g.plot_x
			y:       t.title_size + t.subtitle_size + 4.0
			content: c.subtitle
			size:    t.subtitle_size
			fill:    t.subtitle_color
			anchor:  .start
			family:  t.font_family
		}
	}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): left-align title and add subtitle"
```

---

## Task 5: Confidence-interval bands and area fills

**Files:**
- Modify: `chart/chart.v`
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to `tests/chart_test.v`)**

```v
fn test__band_produces_polygon() {
	svg := chart.new(width: 400, height: 300)
		.band([0.0, 1.0, 2.0], [0.0, 1.0, 0.5], [2.0, 3.0, 2.5])
		.line([0.0, 1.0, 2.0], [1.0, 2.0, 1.5])
		.render()
	assert svg.contains('<polygon')
	assert svg.contains('<polyline')
}

fn test__area_produces_polygon() {
	svg := chart.new(width: 400, height: 300)
		.area([0.0, 1.0, 2.0], [1.0, 3.0, 2.0])
		.render()
	assert svg.contains('<polygon')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — unknown method `band`.

- [ ] **Step 3: Add `band`/`area` enum values and methods**

Find:
```v
enum SeriesKind {
	line
	scatter
	bar
	histogram
}
```
Replace with:
```v
enum SeriesKind {
	line
	scatter
	bar
	histogram
	band
	area
}
```

Find the `histogram` method (it ends just before the `// ---- geometry & bounds ----` comment) and insert these two methods immediately after it. Find:
```v
		nbins: opts.nbins
	}
	nc.series = s
	return nc
}

// ---- geometry & bounds ----
```
Replace with:
```v
		nbins: opts.nbins
	}
	nc.series = s
	return nc
}

pub fn (c Chart) band(x []f64, lower []f64, upper []f64, opts SeriesOpts) Chart {
	assert x.len == lower.len
	assert x.len == upper.len
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .band
		x:     x.clone()
		lo:    lower.clone()
		hi:    upper.clone()
		label: opts.label
		color: if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
	}
	nc.series = s
	return nc
}

pub fn (c Chart) area(x []f64, y []f64, opts SeriesOpts) Chart {
	assert x.len == y.len
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .area
		x:     x.clone()
		y:     y.clone()
		label: opts.label
		color: if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
	}
	nc.series = s
	return nc
}

// ---- geometry & bounds ----
```

- [ ] **Step 4: Add `band`/`area` arms to `series_bounds`**

Find:
```v
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
```
Replace with:
```v
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
		.band {
			x0, x1 := extent(s.x)
			lo0, _ := extent(s.lo)
			_, hi1 := extent(s.hi)
			x0, x1, lo0, hi1
		}
		.area {
			x0, x1 := extent(s.x)
			y0, y1 := extent(s.y)
			ylo := if y0 < 0.0 { y0 } else { 0.0 }
			yhi := if y1 > 0.0 { y1 } else { 0.0 }
			x0, x1, ylo, yhi
		}
	}
}
```

- [ ] **Step 5: Rewrite `draw_series` into a fills pass + a data pass**

Find the entire current `draw_series` function:
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
Replace with:
```v
fn (c Chart) draw_series(mut scene Scene, g Geom) {
	t := c.theme
	// pass 1: filled regions (bands, areas) render behind everything else
	for s in c.series {
		match s.kind {
			.band {
				mut pts := []Point{}
				for i in 0 .. s.x.len {
					pts << Point{
						x: g.xscale.map(s.x[i])
						y: g.yscale.map(s.hi[i])
					}
				}
				for i := s.x.len - 1; i >= 0; i-- {
					pts << Point{
						x: g.xscale.map(s.x[i])
						y: g.yscale.map(s.lo[i])
					}
				}
				scene.primitives << Polygon{
					points:  pts
					fill:    s.color
					opacity: t.fill_opacity
					stroke:  'none'
					width:   0.0
				}
			}
			.area {
				baseline := g.yscale.map(0.0)
				mut pts := []Point{}
				pts << Point{
					x: g.xscale.map(s.x[0])
					y: baseline
				}
				for i in 0 .. s.x.len {
					pts << Point{
						x: g.xscale.map(s.x[i])
						y: g.yscale.map(s.y[i])
					}
				}
				pts << Point{
					x: g.xscale.map(s.x[s.x.len - 1])
					y: baseline
				}
				scene.primitives << Polygon{
					points:  pts
					fill:    s.color
					opacity: t.fill_opacity
					stroke:  'none'
					width:   0.0
				}
			}
			else {}
		}
	}
	// pass 2: data marks
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
			else {}
		}
	}
}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): add CI bands and area fills"
```

---

## Task 6: Value labels on points and bars

**Files:**
- Modify: `chart/chart.v`
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to `tests/chart_test.v`)**

```v
fn test__show_values_adds_labels() {
	svg := chart.new(width: 500, height: 300)
		.bar([11.0, 23.0, 37.0], show_values: true)
		.render()
	assert svg.contains('>11<')
	assert svg.contains('>23<')
	assert svg.contains('>37<')
}

fn test__custom_point_labels() {
	svg := chart.new(width: 400, height: 300)
		.scatter([0.0, 1.0], [1.0, 2.0], show_values: true, labels: ['A', 'B'])
		.render()
	assert svg.contains('>A<')
	assert svg.contains('>B<')
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — the value labels are not rendered yet (assert fails).

- [ ] **Step 3: Add `draw_value_labels` and a `value_text` helper**

Find the `draw_legend` function (it is the last `draw_*` helper). Insert the new helpers immediately before it. Find:
```v
fn (c Chart) draw_legend(mut scene Scene, g Geom) {
```
Replace with:
```v
fn (c Chart) value_text(s Series, i int, v f64) string {
	if s.labels.len > i {
		return s.labels[i]
	}
	return fmt_tick(v)
}

fn (c Chart) draw_value_labels(mut scene Scene, g Geom) {
	t := c.theme
	for s in c.series {
		if !s.show_values {
			continue
		}
		match s.kind {
			.line, .scatter {
				for i in 0 .. s.x.len {
					scene.primitives << Text{
						x:       g.xscale.map(s.x[i])
						y:       g.yscale.map(s.y[i]) - t.marker_radius - 4.0
						content: c.value_text(s, i, s.y[i])
						size:    t.font_size
						fill:    t.axis_color
						anchor:  .middle
						family:  t.font_family
					}
				}
			}
			.bar {
				baseline := g.yscale.map(0.0)
				for i in 0 .. s.y.len {
					top := g.yscale.map(s.y[i])
					scene.primitives << Text{
						x:       g.xscale.map(f64(i))
						y:       math.min(top, baseline) - 4.0
						content: c.value_text(s, i, s.y[i])
						size:    t.font_size
						fill:    t.axis_color
						anchor:  .middle
						family:  t.font_family
					}
				}
			}
			else {}
		}
	}
}

fn (c Chart) draw_legend(mut scene Scene, g Geom) {
```

- [ ] **Step 4: Wire `draw_value_labels` into `build_scene`**

Find:
```v
	c.draw_series(mut scene, g)
	c.draw_labels(mut scene, g)
```
Replace with:
```v
	c.draw_series(mut scene, g)
	c.draw_value_labels(mut scene, g)
	c.draw_labels(mut scene, g)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): add value labels on points and bars"
```

---

## Task 7: Error bars

**Files:**
- Modify: `chart/chart.v`
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to `tests/chart_test.v`)**

```v
fn test__error_bars_add_lines() {
	xs := [0.0, 1.0, 2.0, 3.0, 4.0]
	ys := [1.0, 2.0, 1.5, 2.5, 2.0]
	errs := [0.2, 0.3, 0.2, 0.25, 0.3]
	no_err := chart.new(width: 400, height: 300).scatter(xs, ys).render().count('<line')
	with_err := chart.new(width: 400, height: 300).scatter(xs, ys, err: errs).render().count('<line')
	assert with_err > no_err // each point adds a stem + 2 caps
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — error bars are not drawn yet, so the counts are equal.

- [ ] **Step 3: Replace `series_bounds` to expand y-extent by error bars**

Find:
```v
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
```
Replace with:
```v
fn series_bounds(s Series) (f64, f64, f64, f64) {
	return match s.kind {
		.line, .scatter {
			x0, x1 := extent(s.x)
			ext_lo, ext_hi := extent(s.y)
			mut y0 := ext_lo
			mut y1 := ext_hi
			if s.err.len == s.y.len && s.err.len > 0 {
				for i in 0 .. s.y.len {
					if s.y[i] - s.err[i] < y0 {
						y0 = s.y[i] - s.err[i]
					}
					if s.y[i] + s.err[i] > y1 {
						y1 = s.y[i] + s.err[i]
					}
				}
			}
			x0, x1, y0, y1
		}
		.bar {
			ext_lo, ext_hi := extent(s.y)
			mut y0 := if ext_lo < 0.0 { ext_lo } else { 0.0 }
			mut y1 := if ext_hi > 0.0 { ext_hi } else { 0.0 }
			if s.err.len == s.y.len && s.err.len > 0 {
				for i in 0 .. s.y.len {
					if s.y[i] - s.err[i] < y0 {
						y0 = s.y[i] - s.err[i]
					}
					if s.y[i] + s.err[i] > y1 {
						y1 = s.y[i] + s.err[i]
					}
				}
			}
			-0.5, f64(s.y.len) - 0.5, y0, y1
		}
```

- [ ] **Step 4: Add `draw_error_bars` before `draw_legend`**

Find:
```v
fn (c Chart) draw_value_labels(mut scene Scene, g Geom) {
```
Replace with:
```v
fn (c Chart) draw_error_bars(mut scene Scene, g Geom) {
	t := c.theme
	capw := 4.0
	for s in c.series {
		if s.err.len == 0 {
			continue
		}
		for i in 0 .. s.y.len {
			px := match s.kind {
				.bar { g.xscale.map(f64(i)) }
				else { g.xscale.map(s.x[i]) }
			}
			y_hi := g.yscale.map(s.y[i] + s.err[i])
			y_lo := g.yscale.map(s.y[i] - s.err[i])
			scene.primitives << Line{
				x1:     px
				y1:     y_lo
				x2:     px
				y2:     y_hi
				stroke: t.axis_color
				width:  t.axis_width
			}
			scene.primitives << Line{
				x1:     px - capw
				y1:     y_hi
				x2:     px + capw
				y2:     y_hi
				stroke: t.axis_color
				width:  t.axis_width
			}
			scene.primitives << Line{
				x1:     px - capw
				y1:     y_lo
				x2:     px + capw
				y2:     y_lo
				stroke: t.axis_color
				width:  t.axis_width
			}
		}
	}
}

fn (c Chart) draw_value_labels(mut scene Scene, g Geom) {
```

Note: only `.scatter`, `.line`, and `.bar` series carry `err` (set via their methods); other kinds always have `err.len == 0`, so the `continue` guard skips them.

- [ ] **Step 5: Wire `draw_error_bars` into `build_scene` (before value labels)**

Find:
```v
	c.draw_series(mut scene, g)
	c.draw_value_labels(mut scene, g)
```
Replace with:
```v
	c.draw_series(mut scene, g)
	c.draw_error_bars(mut scene, g)
	c.draw_value_labels(mut scene, g)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): add error bars on points and bars"
```

---

## Task 8: Gridlines

**Files:**
- Modify: `chart/chart.v`
- Test: `tests/chart_test.v`

- [ ] **Step 1: Write the failing test (append to `tests/chart_test.v`)**

```v
fn test__grid_lines_when_enabled() {
	gridless := chart.new(width: 400, height: 300)
		.line([0.0, 1.0], [0.0, 1.0])
		.render()
		.count('<line')
	gridded := chart.new(width: 400, height: 300, theme: chart.Theme{ grid: true })
		.line([0.0, 1.0], [0.0, 1.0])
		.render()
		.count('<line')
	assert gridded > gridless
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `v test tests/chart_test.v`
Expected: FAIL — grid is not drawn, counts are equal.

- [ ] **Step 3: Add `draw_grid` before `draw_axes`**

Find:
```v
fn (c Chart) draw_axes(mut scene Scene, g Geom) {
```
Replace with:
```v
fn (c Chart) draw_grid(mut scene Scene, g Geom) {
	t := c.theme
	if !t.grid {
		return
	}
	for tk in nice_ticks(g.xmin, g.xmax, 5) {
		if tk < g.xmin - 1.0e-9 || tk > g.xmax + 1.0e-9 {
			continue
		}
		px := g.xscale.map(tk)
		scene.primitives << Line{
			x1:     px
			y1:     g.plot_y
			x2:     px
			y2:     g.plot_y + g.plot_h
			stroke: t.grid_color
			width:  t.axis_width
		}
	}
	for tk in nice_ticks(g.ymin, g.ymax, 5) {
		if tk < g.ymin - 1.0e-9 || tk > g.ymax + 1.0e-9 {
			continue
		}
		py := g.yscale.map(tk)
		scene.primitives << Line{
			x1:     g.plot_x
			y1:     py
			x2:     g.plot_x + g.plot_w
			y2:     py
			stroke: t.grid_color
			width:  t.axis_width
		}
	}
}

fn (c Chart) draw_axes(mut scene Scene, g Geom) {
```

- [ ] **Step 4: Wire `draw_grid` first in `build_scene`**

Find:
```v
	mut scene := Scene{}
	c.draw_axes(mut scene, g)
```
Replace with:
```v
	mut scene := Scene{}
	c.draw_grid(mut scene, g)
	c.draw_axes(mut scene, g)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `v test tests/chart_test.v`
Expected: PASS

- [ ] **Step 6: Run the full suite to confirm no regressions**

Run: `v test tests/`
Expected: PASS (all test files).

- [ ] **Step 7: Commit**

```bash
git add chart/chart.v tests/chart_test.v
git commit -m "feat(chart): implement gridlines via theme.grid"
```

---

## Task 9: Housekeeping — regenerate example SVGs and update docs

**Files:**
- Modify (generated): `examples/chart-gallery/*.svg`
- Modify: `docs/src/modules/chart.md`
- Modify (generated): `docs/modules/chart.html`

- [ ] **Step 1: Regenerate the gallery SVGs (the left-aligned title changes them)**

Run: `v run examples/chart-gallery/main.v`
Expected: prints `wrote 4 charts to ...`. Confirm the title is now left-aligned:

```bash
grep -o 'text-anchor="start"[^>]*>Coefficient Magnitude<' examples/chart-gallery/coefficients.svg | head -1
```
Expected: a match (title uses `text-anchor="start"`).

- [ ] **Step 2: Document the new API in `docs/src/modules/chart.md`**

Find:
```v
// decoration
(c Chart) title(s string) Chart
(c Chart) xlabel(s string) Chart
(c Chart) ylabel(s string) Chart
(c Chart) axhline(y f64) Chart       // horizontal reference line
(c Chart) axvline(x f64) Chart       // vertical reference line
```
Replace with:
```v
// fills (rendered behind data marks)
(c Chart) band(x []f64, lower []f64, upper []f64, opts SeriesOpts) Chart   // CI / shaded region
(c Chart) area(x []f64, y []f64, opts SeriesOpts) Chart                    // fill to zero baseline

// decoration
(c Chart) title(s string) Chart
(c Chart) subtitle(s string) Chart   // smaller, muted, left-aligned
(c Chart) xlabel(s string) Chart
(c Chart) ylabel(s string) Chart
(c Chart) axhline(y f64) Chart       // horizontal reference line
(c Chart) axvline(x f64) Chart       // vertical reference line
```

Find:
```v
Series data is `[]f64`. Multiple series share auto-scaled axes; with two or more
labeled series a legend is drawn automatically, and colors cycle through the theme
palette. `render()` is pure (handy for tests); `save()` is the only function that
touches the filesystem.
```
Replace with:
```v
Series data is `[]f64`. Multiple series share auto-scaled axes; with two or more
labeled series a legend is drawn automatically, and colors cycle through the theme
palette. `render()` is pure (handy for tests); `save()` is the only function that
touches the filesystem.

`SeriesOpts` also accepts `color` (override the palette), `show_values` (draw the
value above each point/bar — or pass `labels []string` for custom text), and
`err []f64` (draw error-bar whiskers on points/bars). `ChartOpts` accepts a
`subtitle`, and `Theme{ grid: true }` turns on light gridlines.
```

- [ ] **Step 3: Rebuild the HTML docs**

Run: `make docs`
Expected: `Done. 15 pages built.`

- [ ] **Step 4: Commit**

```bash
git add examples/chart-gallery docs/src/modules/chart.md docs/modules/chart.html
git commit -m "docs(chart): document expansion API and regenerate example SVGs"
```

---

## Self-Review (completed by plan author)

**Spec coverage:**
- `Polygon` primitive → Task 1. Theme fields (`fill_opacity`, `subtitle_size`, `subtitle_color`) → Task 2.
- CI bands (`.band`) + area fills (`.area`) + two-pass draw order + `series_bounds` arms → Task 5.
- Value labels (`show_values`, `labels` override) → Task 6 (fields added Task 3).
- Left-aligned title + subtitle (`ChartOpts.subtitle`, `.subtitle()`, theme styling) → Task 4.
- Per-series color override → Task 3.
- Gridlines (wire up `Theme.grid`) → Task 8.
- Error bars (`err`, whiskers, bounds expansion) → Task 7.
- Tests for every feature → Tasks 1–8. Housekeeping (regenerate gallery SVGs, update `chart.md`, rebuild HTML) → Task 9.
- Error handling (`assert` length preconditions) → Tasks 3 (labels/err), 5 (band lower/upper).

**Placeholder scan:** none — all steps contain concrete code and commands.

**Type/name consistency:** `Series` fields (`lo`, `hi`, `err`, `show_values`, `labels`, `color`), `SeriesKind` values (`band`, `area`), helper names (`draw_grid`, `draw_error_bars`, `draw_value_labels`, `value_text`), and `Theme`/`ChartOpts`/`SeriesOpts` fields are used identically across tasks. `build_scene` is edited additively in Tasks 6, 7, 8 with distinct anchors (after `draw_series`, after `draw_series`, after `Scene{}`), producing the final order: grid → axes → ticks → guides → series → error_bars → value_labels → labels → legend.

**Risk note flagged:** the error-bar and gridline tests use directional assertions (`>`), not exact counts, because enabling them changes data bounds and therefore tick counts — exact arithmetic would be fragile.
