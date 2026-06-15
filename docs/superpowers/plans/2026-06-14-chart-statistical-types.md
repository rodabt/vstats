# Chart Statistical Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 new series kinds (`step`, `box`, `dot`, `violin`, `hbar`, `heatmap`, `stacked_bar`) to `chart/chart.v` and build `examples/statistical-gallery/main.v` generating 12 SVGs.

**Architecture:** All new series kinds live in `chart/chart.v` (consistent with current structure). The gallery file `examples/statistical-gallery/main.v` is built incrementally — each task adds its SVG section. Private helpers (`box_stats`, `percentile`, `silverman_kde`, `lerp_color`, `hex_to_rgb`) also live in `chart.v`. Categorical axis support is added to `draw_ticks`.

**Tech Stack:** V language, `vstats.chart` module, `import math`, `import strconv` (new — for hex parsing), `import os`, `v run examples/statistical-gallery/` as integration test.

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `chart/chart.v` | Modify | New enum values, Series fields, public methods, bounds cases, draw cases, helpers |
| `examples/statistical-gallery/main.v` | Create | Gallery entrypoint, 12 SVG outputs |

---

## Task 1: `step` series kind + gallery scaffold

**Files:**
- Modify: `chart/chart.v`
- Create: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Create gallery scaffold with step examples**

Create `examples/statistical-gallery/main.v`:

```v
module main

import os
import math
import vstats.chart

fn main() {
	out := os.dir(@FILE)

	// ── step: empirical CDF ──────────────────────────────────────────────
	raw := [22.0, 25, 27, 28, 30, 31, 31, 33, 35, 36, 38, 39, 40, 42, 45,
		48, 50, 53, 57, 65]
	n := f64(raw.len)
	mut xs_cdf := raw.clone()
	xs_cdf.sort()
	mut ys_cdf := []f64{len: raw.len}
	for i in 0 .. raw.len {
		ys_cdf[i] = f64(i + 1) / n
	}
	chart.new(title: 'Empirical CDF', width: 640, height: 420,
		theme: chart.Theme{ grid: true })
		.step(xs_cdf, ys_cdf, label: 'CDF')
		.xlabel('Age')
		.ylabel('Cumulative probability')
		.save(os.join_path(out, 'step_cdf.svg'))!

	// ── step: survival curves ────────────────────────────────────────────
	ts := [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	control := [1.0, 0.95, 0.88, 0.79, 0.71, 0.63, 0.55, 0.48, 0.41, 0.35, 0.30]
	treatment := [1.0, 0.98, 0.94, 0.89, 0.83, 0.77, 0.71, 0.65, 0.59, 0.53, 0.47]
	chart.new(title: 'Survival Curves', subtitle: 'Kaplan-Meier estimate', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.step(ts, control, label: 'Control')
		.step(ts, treatment, label: 'Treatment')
		.axhline(0.5)
		.xlabel('Time (months)')
		.ylabel('Survival probability')
		.save(os.join_path(out, 'step_survival.svg'))!

	_ = math.pi // suppress unused import until later tasks use it
	println('done — wrote SVGs to ${out}')
}
```

- [ ] **Step 2: Run gallery to confirm compile error**

```bash
v run examples/statistical-gallery/
```

Expected: compile error — `unknown method or field: step` (or similar).

- [ ] **Step 3: Add `step` to `chart/chart.v`**

**3a.** Add `.step` to the `SeriesKind` enum (after `.area`):

```v
enum SeriesKind {
	line
	scatter
	bar
	histogram
	band
	area
	step
}
```

**3b.** Add the `.step` case to `series_bounds` (inside the `match s.kind` block, after `.area`):

```v
.step {
    x0, x1 := extent(s.x)
    y0, y1 := extent(s.y)
    x0, x1, y0, y1
}
```

**3c.** Add the public `.step()` method (after the `.area()` method):

```v
pub fn (c Chart) step(x []f64, y []f64, opts SeriesOpts) Chart {
	assert x.len == y.len
	mut nc := c
	mut s := c.series.clone()
	s << Series{
		kind:  .step
		x:     x.clone()
		y:     y.clone()
		label: opts.label
		color: if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
	}
	nc.series = s
	return nc
}
```

**3d.** Add `.step` rendering in `draw_series` pass 2 (inside the second `for s in c.series` loop, in the `match s.kind` block). Add it after the `.scatter` case:

```v
.step {
	if s.x.len < 2 {
		continue
	}
	for i in 0 .. s.x.len - 1 {
		px1 := g.xscale.map(s.x[i])
		px2 := g.xscale.map(s.x[i + 1])
		py1 := g.yscale.map(s.y[i])
		py2 := g.yscale.map(s.y[i + 1])
		scene.primitives << Line{
			x1:     px1
			y1:     py1
			x2:     px2
			y2:     py1
			stroke: s.color
			width:  t.series_width
		}
		scene.primitives << Line{
			x1:     px2
			y1:     py1
			x2:     px2
			y2:     py2
			stroke: s.color
			width:  t.series_width
		}
	}
	// final horizontal tail to last point
	scene.primitives << Line{
		x1:     g.xscale.map(s.x[s.x.len - 1])
		y1:     g.yscale.map(s.y[s.y.len - 1])
		x2:     g.xscale.map(s.x[s.x.len - 1])
		y2:     g.yscale.map(s.y[s.y.len - 1])
		stroke: s.color
		width:  t.series_width
	}
}
```

- [ ] **Step 4: Run gallery to confirm two SVGs generated**

```bash
v run examples/statistical-gallery/
```

Expected: `done — wrote SVGs to .../statistical-gallery`  
Check: `ls examples/statistical-gallery/*.svg` shows `step_cdf.svg` and `step_survival.svg`.

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v examples/statistical-gallery/main.v
git commit -m "feat(chart): add step series kind + gallery scaffold"
```

---

## Task 2: `box` series kind

**Files:**
- Modify: `chart/chart.v`
- Modify: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Add box gallery section to `main.v`**

Add after the survival curves block (before the `_ = math.pi` line):

```v
	// ── box: distribution comparison ────────────────────────────────────
	a_data := [45.0, 47, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 60]
	b_data := [20.0, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80]
	c_data := [30.0, 32, 33, 34, 35, 35, 36, 37, 38, 42, 55, 68, 70]
	d_data := [40.0, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64]
	chart.new(title: 'Distribution Comparison', subtitle: 'Box plots — Q1/median/Q3, 1.5×IQR whiskers',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.box(a_data, label: 'A')
		.box(b_data, label: 'B')
		.box(c_data, label: 'C')
		.box(d_data, label: 'D')
		.ylabel('Value')
		.save(os.join_path(out, 'box_comparison.svg'))!
```

- [ ] **Step 2: Run gallery to confirm compile error**

```bash
v run examples/statistical-gallery/
```

Expected: error — `unknown method or field: box`.

- [ ] **Step 3: Implement `box` in `chart/chart.v`**

**3a.** Add `.box_plot` to `SeriesKind` enum (use `box_plot` because `box` is a V keyword):

```v
enum SeriesKind {
	line
	scatter
	bar
	histogram
	band
	area
	step
	box_plot
}
```

> Note: The public method is named `.box()` but the enum value is `.box_plot` to avoid the V keyword conflict.

**3b.** Add two new private fields to the `Series` struct (after `color string`):

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
	color_lo    string
	color_hi    string
	nbins       int
	show_values bool
	labels      []string
}
```

**3c.** Add private helpers after the `extent` function:

```v
fn percentile(sorted []f64, p f64) f64 {
	n := sorted.len
	if n == 0 {
		return 0.0
	}
	idx := p * f64(n - 1)
	lo := int(math.floor(idx))
	hi := int(math.ceil(idx))
	if lo == hi {
		return sorted[lo]
	}
	frac := idx - f64(lo)
	return sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn box_stats(data []f64) (f64, f64, f64, f64, f64, []f64) {
	mut s := data.clone()
	s.sort()
	q1 := percentile(s, 0.25)
	med := percentile(s, 0.50)
	q3 := percentile(s, 0.75)
	iqr := q3 - q1
	fence_lo := q1 - 1.5 * iqr
	fence_hi := q3 + 1.5 * iqr
	mut outliers := []f64{}
	mut wlo := q1
	mut whi := q3
	for v in s {
		if v < fence_lo {
			outliers << v
		} else if wlo == q1 && v >= fence_lo {
			wlo = v
		}
	}
	for i := s.len - 1; i >= 0; i-- {
		if s[i] > fence_hi {
			outliers << s[i]
		} else if whi == q3 && s[i] <= fence_hi {
			whi = s[i]
			break
		}
	}
	return q1, med, q3, wlo, whi, outliers
}
```

**3d.** Add `.box_plot` case to `series_bounds`:

```v
.box_plot {
	cx := s.x[0]
	cx - 0.5, cx + 0.5, s.lo[0], s.hi[0]
}
```

**3e.** Add public `.box()` method (after `.step()`):

```v
pub fn (c Chart) box(data []f64, opts SeriesOpts) Chart {
	assert data.len > 0
	mut box_idx := 0
	for s in c.series {
		if s.kind == .box_plot {
			box_idx++
		}
	}
	q1, med, q3, wlo, whi, outliers := box_stats(data)
	mut nc := c
	mut sv := c.series.clone()
	sv << Series{
		kind:  .box_plot
		x:     [f64(box_idx)]
		y:     [q1, med, q3]
		lo:    [wlo]
		hi:    [whi]
		err:   outliers
		label: opts.label
		color: if opts.color != '' { opts.color } else { c.theme.color(box_idx) }
	}
	nc.series = sv
	return nc
}
```

**3f.** Add `.box_plot` rendering in `draw_series` pass 2 (after `.step` case):

```v
.box_plot {
	band := g.xscale.map(1.0) - g.xscale.map(0.0)
	bw := band * 0.6
	cx := g.xscale.map(s.x[0])
	q1_px := g.yscale.map(s.y[0])
	med_px := g.yscale.map(s.y[1])
	q3_px := g.yscale.map(s.y[2])
	wlo_px := g.yscale.map(s.lo[0])
	whi_px := g.yscale.map(s.hi[0])
	capw := bw * 0.3
	// box body
	scene.primitives << Rect{
		x:      cx - bw / 2.0
		y:      q3_px
		w:      bw
		h:      q1_px - q3_px
		fill:   s.color
		stroke: t.axis_color
		width:  t.axis_width
		opacity: 0.7
	}
	// median line
	scene.primitives << Line{
		x1:     cx - bw / 2.0
		y1:     med_px
		x2:     cx + bw / 2.0
		y2:     med_px
		stroke: t.axis_color
		width:  t.axis_width * 2.0
	}
	// whiskers
	scene.primitives << Line{ x1: cx, y1: q3_px, x2: cx, y2: whi_px, stroke: t.axis_color, width: t.axis_width }
	scene.primitives << Line{ x1: cx, y1: q1_px, x2: cx, y2: wlo_px, stroke: t.axis_color, width: t.axis_width }
	scene.primitives << Line{ x1: cx - capw, y1: whi_px, x2: cx + capw, y2: whi_px, stroke: t.axis_color, width: t.axis_width }
	scene.primitives << Line{ x1: cx - capw, y1: wlo_px, x2: cx + capw, y2: wlo_px, stroke: t.axis_color, width: t.axis_width }
	// outliers
	for ov in s.err {
		scene.primitives << Circle{
			cx:     cx
			cy:     g.yscale.map(ov)
			r:      t.marker_radius
			fill:   'none'
			stroke: s.color
			width:  t.axis_width
		}
	}
}
```

> Note: `Rect` does not currently have an `opacity` field. Either omit it and use the fill color directly, or skip opacity for box fills. Use the fill color at full opacity — box fill looks fine without transparency.

Remove the `opacity: 0.7` line from the Rect above (Rect struct has no opacity field):

```v
scene.primitives << Rect{
	x:      cx - bw / 2.0
	y:      q3_px
	w:      bw
	h:      q1_px - q3_px
	fill:   s.color
	stroke: t.axis_color
	width:  t.axis_width
}
```

- [ ] **Step 4: Run gallery**

```bash
v run examples/statistical-gallery/
```

Expected: `done — wrote SVGs to ...`  
Check: `box_comparison.svg` exists.

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v examples/statistical-gallery/main.v
git commit -m "feat(chart): add box series kind"
```

---

## Task 3: `dot` series kind + categorical y-axis

**Files:**
- Modify: `chart/chart.v`
- Modify: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Add dot gallery section to `main.v`**

Add after the box block:

```v
	// ── dot: Cleveland dot plot ──────────────────────────────────────────
	feat_labels := ['Age', 'Income', 'Education', 'Distance', 'Tenure', 'Score', 'Visits',
		'Days']
	feat_vals := [0.82, 0.74, 0.68, 0.61, 0.55, 0.49, 0.37, 0.22]
	chart.new(title: 'Feature Importance', subtitle: 'Cleveland dot plot', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.dot(feat_vals, labels: feat_labels)
		.xlabel('Importance score')
		.save(os.join_path(out, 'dot_ranking.svg'))!
```

- [ ] **Step 2: Run to confirm compile error**

```bash
v run examples/statistical-gallery/
```

Expected: error — `unknown method or field: dot`.

- [ ] **Step 3: Implement `dot` in `chart/chart.v`**

**3a.** Add `.dot` to `SeriesKind` enum (after `.box_plot`):

```v
	box_plot
	dot
```

**3b.** Add `.dot` case to `series_bounds`:

```v
.dot {
	_, xmax := extent(s.y)
	0.0, xmax, -0.5, f64(s.y.len) - 0.5
}
```

**3c.** Add public `.dot()` method (after `.box()`):

```v
pub fn (c Chart) dot(values []f64, opts SeriesOpts) Chart {
	if opts.labels.len > 0 {
		assert opts.labels.len == values.len
	}
	mut nc := c
	mut sv := c.series.clone()
	sv << Series{
		kind:   .dot
		y:      values.clone()
		label:  opts.label
		color:  if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
		labels: opts.labels.clone()
	}
	nc.series = sv
	return nc
}
```

**3d.** Add `.dot` rendering in `draw_series` pass 2 (after `.box_plot`):

```v
.dot {
	x0 := g.xscale.map(0.0)
	for i in 0 .. s.y.len {
		py := g.yscale.map(f64(i))
		px := g.xscale.map(s.y[i])
		scene.primitives << Line{
			x1:     x0
			y1:     py
			x2:     px
			y2:     py
			stroke: t.grid_color
			width:  t.axis_width
		}
		scene.primitives << Circle{
			cx:     px
			cy:     py
			r:      t.marker_radius + 1.0
			fill:   s.color
			stroke: 'none'
			width:  0.0
		}
	}
}
```

**3e.** Modify `draw_ticks` in `chart/chart.v` to support categorical y-axis.

Replace the entire `draw_ticks` function with:

```v
fn (c Chart) draw_ticks(mut scene Scene, g Geom) {
	t := c.theme
	bottom := g.plot_y + g.plot_h

	// detect categorical axes
	mut y_cat_labels := []string{}
	mut x_cat_labels := []string{}
	for s in c.series {
		if s.kind in [.dot, .hbar] && s.labels.len > 0 && y_cat_labels.len == 0 {
			y_cat_labels = s.labels
		}
		if s.kind == .stacked_bar && s.labels.len > 0 && x_cat_labels.len == 0 {
			x_cat_labels = s.labels
		}
		if s.kind == .heatmap && s.labels.len > 0 {
			ncols := s.nbins
			if ncols > 0 && s.labels.len > ncols {
				x_cat_labels = s.labels[0..ncols]
				y_cat_labels = s.labels[ncols..]
			}
		}
	}

	// x-axis ticks
	if x_cat_labels.len > 0 {
		for i, lbl in x_cat_labels {
			px := g.xscale.map(f64(i))
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
				content: lbl
				size:    t.font_size
				fill:    t.axis_color
				anchor:  .middle
				family:  t.font_family
			}
		}
	} else {
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
	}

	// y-axis ticks
	if y_cat_labels.len > 0 {
		for i, lbl in y_cat_labels {
			py := g.yscale.map(f64(i))
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
				content: lbl
				size:    t.font_size
				fill:    t.axis_color
				anchor:  .end
				family:  t.font_family
			}
		}
	} else {
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
}
```

> Note: `.stacked_bar` and `.heatmap` are referenced here but not yet implemented. The compiler is fine with enum values in match arms as long as they're defined in the enum — so add `.stacked_bar` and `.heatmap` to the `SeriesKind` enum now (as stubs, without bounds/draw cases), to let this compile. Add them after `.dot`:

```v
enum SeriesKind {
	line
	scatter
	bar
	histogram
	band
	area
	step
	box_plot
	dot
	violin
	hbar
	heatmap
	stacked_bar
}
```

> Add all remaining enum values now so `draw_ticks` compiles. Bounds and draw cases for these kinds will be added in later tasks. Add stub cases to `series_bounds` and `draw_series` to satisfy the exhaustive match:

In `series_bounds`, add at the end of the match:

```v
.violin {
	cx := s.x[0]
	cx - 0.5, cx + 0.5, s.lo[0], s.hi[0]
}
.hbar {
	_, xmax := extent(s.y)
	0.0, xmax, -0.5, f64(s.y.len) - 0.5
}
.heatmap {
	ncols := s.nbins
	nrows := if ncols > 0 { s.x.len / ncols } else { 1 }
	-0.5, f64(ncols) - 0.5, -0.5, f64(nrows) - 0.5
}
.stacked_bar {
	nseg := s.nbins
	nbars := if nseg > 0 { s.x.len / nseg } else { 1 }
	-0.5, f64(nbars) - 0.5, 0.0, 1.0
}
```

In `draw_series` pass 2 match, add at the end:

```v
.violin, .hbar, .heatmap, .stacked_bar {}
```

- [ ] **Step 4: Run gallery**

```bash
v run examples/statistical-gallery/
```

Expected: success. Check `dot_ranking.svg` — dots should appear with labels on y-axis.

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v examples/statistical-gallery/main.v
git commit -m "feat(chart): add dot series kind + categorical y-axis"
```

---

## Task 4: `violin` series kind + KDE helper

**Files:**
- Modify: `chart/chart.v`
- Modify: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Add violin gallery section to `main.v`**

Add after the dot block:

```v
	// ── violin: distribution shape ───────────────────────────────────────
	chart.new(title: 'Distribution Shape', subtitle: 'Violin plots — same groups as box',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.violin(a_data, label: 'A')
		.violin(b_data, label: 'B')
		.violin(c_data, label: 'C')
		.violin(d_data, label: 'D')
		.ylabel('Value')
		.save(os.join_path(out, 'violin_distribution.svg'))!
```

- [ ] **Step 2: Run to confirm compile error**

```bash
v run examples/statistical-gallery/
```

Expected: error — `unknown method or field: violin`.

- [ ] **Step 3: Implement `violin` in `chart/chart.v`**

**3a.** Add KDE private helper (after `box_stats`):

```v
fn silverman_kde(data []f64, n_grid int) ([]f64, []f64) {
	n := f64(data.len)
	mut sum := 0.0
	for v in data {
		sum += v
	}
	mean := sum / n
	mut sq_sum := 0.0
	for v in data {
		sq_sum += (v - mean) * (v - mean)
	}
	sigma := math.sqrt(sq_sum / (n - 1.0))
	h := 1.06 * sigma * math.pow(n, -0.2)
	lo, hi := extent(data)
	range_ := hi - lo + 4.0 * h
	mut grid := []f64{len: n_grid}
	mut density := []f64{len: n_grid}
	for i in 0 .. n_grid {
		gv := lo - 2.0 * h + f64(i) * range_ / f64(n_grid - 1)
		grid[i] = gv
		mut d := 0.0
		for x in data {
			u := (gv - x) / h
			d += math.exp(-0.5 * u * u)
		}
		density[i] = d / (n * h * math.sqrt(2.0 * math.pi))
	}
	return grid, density
}
```

**3b.** Add `.violin()` public method (after `.dot()`):

```v
pub fn (c Chart) violin(data []f64, opts SeriesOpts) Chart {
	assert data.len > 1
	mut violin_idx := 0
	for s in c.series {
		if s.kind == .violin {
			violin_idx++
		}
	}
	grid, density := silverman_kde(data, 50)
	lo, hi := extent(data)
	mut nc := c
	mut sv := c.series.clone()
	sv << Series{
		kind:  .violin
		x:     [f64(violin_idx)]
		y:     grid
		err:   density
		lo:    [lo]
		hi:    [hi]
		label: opts.label
		color: if opts.color != '' { opts.color } else { c.theme.color(violin_idx) }
	}
	nc.series = sv
	return nc
}
```

**3c.** Replace the stub `.violin` case in `series_bounds` — already correct from Task 3. No change needed.

**3d.** Add `.violin` rendering in `draw_series` **pass 1** (the filled regions loop), after the `.area` case:

```v
.violin {
	cx := g.xscale.map(s.x[0])
	mut max_d := 0.0
	for d in s.err {
		if d > max_d {
			max_d = d
		}
	}
	if max_d <= 0.0 || s.y.len == 0 {
		continue
	}
	half_w := (g.xscale.map(1.0) - g.xscale.map(0.0)) * 0.4
	mut left_pts := []Point{}
	mut right_pts := []Point{}
	for i in 0 .. s.y.len {
		py := g.yscale.map(s.y[i])
		d_px := s.err[i] / max_d * half_w
		left_pts << Point{ x: cx - d_px, y: py }
		right_pts << Point{ x: cx + d_px, y: py }
	}
	mut pts := left_pts.clone()
	for i := right_pts.len - 1; i >= 0; i-- {
		pts << right_pts[i]
	}
	scene.primitives << Polygon{
		points:  pts
		fill:    s.color
		opacity: t.fill_opacity * 2.5
		stroke:  s.color
		width:   t.axis_width
	}
}
```

**3e.** Remove `.violin` from the pass 2 stub. Update the stub at the end of pass 2 match:

```v
.hbar, .heatmap, .stacked_bar {}
```

- [ ] **Step 4: Run gallery**

```bash
v run examples/statistical-gallery/
```

Expected: success. Check `violin_distribution.svg`.

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v examples/statistical-gallery/main.v
git commit -m "feat(chart): add violin series kind with Silverman KDE"
```

---

## Task 5: `hbar` series kind

**Files:**
- Modify: `chart/chart.v`
- Modify: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Add hbar gallery section to `main.v`**

Add after the violin block:

```v
	// ── hbar: horizontal bar ─────────────────────────────────────────────
	hbar_labels := ['Control', 'Variant A', 'Variant B', 'Variant C', 'Variant D', 'Variant E']
	hbar_vals := [12.4, 14.1, 13.8, 16.2, 11.9, 15.5]
	chart.new(title: 'Conversion Rate by Variant', subtitle: 'Horizontal bar chart', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.hbar(hbar_vals, labels: hbar_labels)
		.xlabel('Conversion rate (%)')
		.save(os.join_path(out, 'hbar_comparison.svg'))!
```

- [ ] **Step 2: Run to confirm compile error**

```bash
v run examples/statistical-gallery/
```

Expected: error — `unknown method or field: hbar`.

- [ ] **Step 3: Implement `hbar` in `chart/chart.v`**

**3a.** Update `.hbar` case in `series_bounds` (replace stub with full version):

```v
.hbar {
	lo_val, hi_val := extent(s.y)
	xlo := if lo_val < 0.0 { lo_val } else { 0.0 }
	xlo, hi_val, -0.5, f64(s.y.len) - 0.5
}
```

**3b.** Add public `.hbar()` method (after `.violin()`):

```v
pub fn (c Chart) hbar(values []f64, opts SeriesOpts) Chart {
	if opts.labels.len > 0 {
		assert opts.labels.len == values.len
	}
	mut nc := c
	mut sv := c.series.clone()
	sv << Series{
		kind:   .hbar
		y:      values.clone()
		label:  opts.label
		color:  if opts.color != '' { opts.color } else { c.theme.color(c.series.len) }
		labels: opts.labels.clone()
	}
	nc.series = sv
	return nc
}
```

**3c.** Add `.hbar` rendering in `draw_series` pass 2. Replace the stub:

```v
.hbar {
	band := g.yscale.map(0.0) - g.yscale.map(1.0)
	bh := math.abs(band) * 0.8
	baseline := g.xscale.map(0.0)
	for i in 0 .. s.y.len {
		cy := g.yscale.map(f64(i))
		right_edge := g.xscale.map(s.y[i])
		scene.primitives << Rect{
			x:      math.min(baseline, right_edge)
			y:      cy - bh / 2.0
			w:      math.abs(right_edge - baseline)
			h:      bh
			fill:   s.color
			stroke: 'none'
			width:  0.0
		}
	}
}
```

Update the pass 2 stub to remove `.hbar`:

```v
.heatmap, .stacked_bar {}
```

- [ ] **Step 4: Run gallery**

```bash
v run examples/statistical-gallery/
```

Expected: success. Check `hbar_comparison.svg` — horizontal bars with category labels on y-axis.

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v examples/statistical-gallery/main.v
git commit -m "feat(chart): add hbar series kind"
```

---

## Task 6: `heatmap` series kind + colour interpolation

**Files:**
- Modify: `chart/chart.v`
- Modify: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Add heatmap gallery section to `main.v`**

Add after the hbar block:

```v
	// ── heatmap: correlation matrix ──────────────────────────────────────
	heat_labels := ['A', 'B', 'C', 'D', 'E']
	corr := [
		[1.0, 0.8, 0.3, -0.2, 0.1],
		[0.8, 1.0, 0.4, -0.1, 0.2],
		[0.3, 0.4, 1.0, 0.5, -0.3],
		[-0.2, -0.1, 0.5, 1.0, 0.6],
		[0.1, 0.2, -0.3, 0.6, 1.0],
	]
	chart.new(title: 'Correlation Matrix', width: 500, height: 500)
		.heatmap(corr, row_labels: heat_labels, col_labels: heat_labels,
		color_lo: '#d73027', color_hi: '#4575b4')
		.save(os.join_path(out, 'heatmap_correlation.svg'))!
```

- [ ] **Step 2: Run to confirm compile error**

```bash
v run examples/statistical-gallery/
```

Expected: error — `unknown method or field: heatmap`.

- [ ] **Step 3: Implement `heatmap` in `chart/chart.v`**

**3a.** Add `import strconv` at the top of `chart.v` (alongside existing imports):

```v
import math
import os
import strconv
```

**3b.** Add `HeatmapOpts` struct (near the other opts structs):

```v
@[params]
pub struct HeatmapOpts {
pub:
	row_labels []string
	col_labels []string
	color_lo   string = '#f7fbff'
	color_hi   string = '#08306b'
}
```

**3c.** Add colour helpers (after `silverman_kde`):

```v
fn hex_to_rgb(hex string) (f64, f64, f64) {
	h := if hex.starts_with('#') { hex[1..] } else { hex }
	r := f64(strconv.parse_uint(h[0..2], 16, 8) or { 0 })
	g := f64(strconv.parse_uint(h[2..4], 16, 8) or { 0 })
	b := f64(strconv.parse_uint(h[4..6], 16, 8) or { 0 })
	return r, g, b
}

fn lerp_color(lo string, hi string, t f64) string {
	r0, g0, b0 := hex_to_rgb(lo)
	r1, g1, b1 := hex_to_rgb(hi)
	r := int(r0 + (r1 - r0) * t)
	g := int(g0 + (g1 - g0) * t)
	b := int(b0 + (b1 - b0) * t)
	return '#${r:02x}${g:02x}${b:02x}'
}
```

**3d.** Update `.heatmap` case in `series_bounds` (already correct from Task 3 stub — no change needed).

**3e.** Add public `.heatmap()` method (after `.hbar()`):

```v
pub fn (c Chart) heatmap(data [][]f64, opts HeatmapOpts) Chart {
	assert data.len > 0
	assert data[0].len > 0
	nrows := data.len
	ncols := data[0].len
	mut flat := []f64{cap: nrows * ncols}
	for row in data {
		for v in row {
			flat << v
		}
	}
	// labels: col_labels first, then row_labels (nbins = ncols as split)
	mut lbs := opts.col_labels.clone()
	lbs << opts.row_labels
	mut nc := c
	mut sv := c.series.clone()
	sv << Series{
		kind:     .heatmap
		x:        flat
		nbins:    ncols
		labels:   lbs
		color_lo: opts.color_lo
		color_hi: opts.color_hi
		label:    ''
		color:    ''
	}
	nc.series = sv
	return nc
}
```

**3f.** Add `.heatmap` rendering in `draw_series` pass 2. Replace the stub (update it to remove `.heatmap`, keep `.stacked_bar`):

```v
.heatmap {
	ncols := s.nbins
	if ncols == 0 {
		continue
	}
	nrows := s.x.len / ncols
	lo_val, hi_val := extent(s.x)
	cell_w := g.plot_w / f64(ncols)
	cell_h := g.plot_h / f64(nrows)
	for i in 0 .. nrows {
		for j in 0 .. ncols {
			val := s.x[i * ncols + j]
			t_val := if hi_val > lo_val {
				(val - lo_val) / (hi_val - lo_val)
			} else {
				0.5
			}
			col := lerp_color(s.color_lo, s.color_hi, t_val)
			px := g.xscale.map(f64(j))
			py := g.yscale.map(f64(nrows - 1 - i))
			scene.primitives << Rect{
				x:      px - cell_w / 2.0
				y:      py - cell_h / 2.0
				w:      cell_w
				h:      cell_h
				fill:   col
				stroke: 'none'
				width:  0.0
			}
		}
	}
}
```

Update pass 2 stub to only keep `.stacked_bar`:

```v
.stacked_bar {}
```

**3g.** Update `draw_ticks` to handle heatmap y-axis label ordering. In the `y_cat_labels` branch, heatmap rows are stored top→bottom but rendered with row 0 at top (using `nrows-1-i` flip). The y_cat_labels are already in row 0..n-1 order. Update the y categorical section:

In `draw_ticks`, in the `y_cat_labels.len > 0` branch, the current code plots label `i` at `g.yscale.map(f64(i))`. For heatmap this maps row 0 near the bottom. We need to check if we're in heatmap mode and flip.

Add a flag before the loop:

```v
mut is_heatmap_y := false
for s in c.series {
    if s.kind == .heatmap {
        is_heatmap_y = true
        break
    }
}
```

Then in the y cat label loop:

```v
for i, lbl in y_cat_labels {
    y_pos := if is_heatmap_y {
        f64(y_cat_labels.len - 1 - i)
    } else {
        f64(i)
    }
    py := g.yscale.map(y_pos)
    // ... tick and text as before
}
```

- [ ] **Step 4: Run gallery**

```bash
v run examples/statistical-gallery/
```

Expected: success. Check `heatmap_correlation.svg` — 5×5 grid with blue-red diverging colours.

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v examples/statistical-gallery/main.v
git commit -m "feat(chart): add heatmap series kind with colour interpolation"
```

---

## Task 7: `stacked_bar` series kind

**Files:**
- Modify: `chart/chart.v`
- Modify: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Add stacked_bar gallery section to `main.v`**

Add after the heatmap block:

```v
	// ── stacked_bar: revenue by segment ──────────────────────────────────
	quarters := ['Q1', 'Q2', 'Q3', 'Q4']
	stack_groups := [
		[120.0, 85.0, 60.0],
		[145.0, 90.0, 75.0],
		[110.0, 95.0, 80.0],
		[160.0, 100.0, 90.0],
	]
	chart.new(title: 'Revenue by Segment', subtitle: 'Stacked bar — Products A, B, C',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.stacked_bar(stack_groups, labels: quarters)
		.ylabel('Revenue ($k)')
		.save(os.join_path(out, 'stacked_bar.svg'))!
```

- [ ] **Step 2: Run to confirm compile error**

```bash
v run examples/statistical-gallery/
```

Expected: error — `unknown method or field: stacked_bar`.

- [ ] **Step 3: Implement `stacked_bar` in `chart/chart.v`**

**3a.** Update `.stacked_bar` case in `series_bounds` (replace stub):

```v
.stacked_bar {
	nseg := s.nbins
	if nseg == 0 {
		return -0.5, 0.5, 0.0, 1.0
	}
	nbars := s.x.len / nseg
	mut max_stack := 0.0
	for i in 0 .. nbars {
		mut sum := 0.0
		for j in 0 .. nseg {
			sum += s.x[i * nseg + j]
		}
		if sum > max_stack {
			max_stack = sum
		}
	}
	-0.5, f64(nbars) - 0.5, 0.0, max_stack
}
```

**3b.** Add public `.stacked_bar()` method (after `.heatmap()`):

```v
pub fn (c Chart) stacked_bar(groups [][]f64, opts SeriesOpts) Chart {
	assert groups.len > 0
	nseg := groups[0].len
	assert nseg > 0
	mut flat := []f64{cap: groups.len * nseg}
	for g in groups {
		assert g.len == nseg
		for v in g {
			flat << v
		}
	}
	mut nc := c
	mut sv := c.series.clone()
	sv << Series{
		kind:   .stacked_bar
		x:      flat
		nbins:  nseg
		label:  opts.label
		color:  ''
		labels: opts.labels.clone()
	}
	nc.series = sv
	return nc
}
```

**3c.** Add `.stacked_bar` rendering in `draw_series` pass 2. Replace the stub:

```v
.stacked_bar {
	nseg := s.nbins
	if nseg == 0 {
		continue
	}
	nbars := s.x.len / nseg
	band := g.xscale.map(1.0) - g.xscale.map(0.0)
	bw := band * 0.8
	for i in 0 .. nbars {
		cx := g.xscale.map(f64(i))
		mut cum := 0.0
		for j in 0 .. nseg {
			seg_val := s.x[i * nseg + j]
			bottom_px := g.yscale.map(cum)
			top_px := g.yscale.map(cum + seg_val)
			col := c.theme.color(j)
			scene.primitives << Rect{
				x:      cx - bw / 2.0
				y:      math.min(top_px, bottom_px)
				w:      bw
				h:      math.abs(bottom_px - top_px)
				fill:   col
				stroke: 'none'
				width:  0.0
			}
			cum += seg_val
		}
	}
}
```

Remove the pass 2 stub entirely (all series kinds now handled).

- [ ] **Step 4: Run gallery**

```bash
v run examples/statistical-gallery/
```

Expected: success. Check `stacked_bar.svg` — 4 bars, 3 segments each, quarter labels on x-axis.

- [ ] **Step 5: Commit**

```bash
git add chart/chart.v examples/statistical-gallery/main.v
git commit -m "feat(chart): add stacked_bar series kind"
```

---

## Task 8: Complete gallery with composed examples

**Files:**
- Modify: `examples/statistical-gallery/main.v`

- [ ] **Step 1: Add all four composed examples to `main.v`**

Add after the stacked_bar block, replacing `_ = math.pi` with full usage:

```v
	// ── line + band: CI ribbon ───────────────────────────────────────────
	xs_ci := [0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	ys_ci := [2.1, 2.8, 3.9, 4.5, 5.2, 5.8, 6.1, 6.8, 7.2, 7.9, 8.3, 8.9]
	lo_ci := [1.6, 2.1, 3.0, 3.5, 4.0, 4.6, 4.8, 5.4, 5.7, 6.3, 6.6, 7.1]
	hi_ci := [2.6, 3.5, 4.8, 5.5, 6.4, 7.0, 7.4, 8.2, 8.7, 9.5, 10.0, 10.7]
	chart.new(title: 'Monthly Trend with 95% CI', width: 640, height: 420,
		theme: chart.Theme{ grid: true })
		.band(xs_ci, lo_ci, hi_ci, label: '95% CI')
		.line(xs_ci, ys_ci, label: 'mean')
		.xlabel('Month')
		.ylabel('Value')
		.save(os.join_path(out, 'line_ci.svg'))!

	// ── forest plot (composed: scatter + err + vline) ────────────────────
	study_y := [5.0, 4, 3, 2, 1, 0]
	effects := [0.32, 0.18, 0.45, -0.08, 0.28, 0.27]
	errs := [0.12, 0.18, 0.22, 0.15, 0.10, 0.06]
	study_labels := ['Study 1', 'Study 2', 'Study 3', 'Study 4', 'Study 5', 'Meta-analysis']
	chart.new(title: 'Forest Plot', subtitle: 'Effect sizes with 95% CI', width: 640,
		height: 420, theme: chart.Theme{ grid: true })
		.scatter(effects, study_y, err: errs, labels: study_labels, show_values: false)
		.axvline(0.0)
		.xlabel('Effect size')
		.save(os.join_path(out, 'forest_plot.svg'))!

	// ── Q-Q plot (composed: scatter + reference line) ────────────────────
	// 20 near-normal values, sorted
	sample_sorted := [-2.1, -1.7, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.1,
		0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.6, 1.9, 2.3]
	// theoretical normal quantiles for n=20
	theory := [-1.87, -1.40, -1.13, -0.93, -0.76, -0.60, -0.45, -0.32, -0.18, -0.06,
		0.06, 0.18, 0.32, 0.45, 0.60, 0.76, 0.93, 1.13, 1.40, 1.87]
	qq_ref_x := [-2.0, 2.0]
	qq_ref_y := [-2.0, 2.0]
	chart.new(title: 'Normal Q-Q Plot', width: 480, height: 480,
		theme: chart.Theme{ grid: true })
		.line(qq_ref_x, qq_ref_y, color: '#cccccc', label: 'reference')
		.scatter(theory, sample_sorted, label: 'sample')
		.xlabel('Theoretical quantiles')
		.ylabel('Sample quantiles')
		.save(os.join_path(out, 'qq_normal.svg'))!

	// ── KDE density overlay on histogram (composed) ──────────────────────
	kde_data := [1.2, 1.5, 1.8, 2.0, 2.1, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
		3.1, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 4.0, 4.1, 4.3, 4.5, 4.8, 5.0,
		5.3, 5.8]
	// inline KDE (Gaussian, Silverman bandwidth)
	kde_n := f64(kde_data.len)
	mut kde_sum := 0.0
	for v in kde_data {
		kde_sum += v
	}
	kde_mean := kde_sum / kde_n
	mut kde_sq := 0.0
	for v in kde_data {
		kde_sq += (v - kde_mean) * (v - kde_mean)
	}
	kde_sigma := math.sqrt(kde_sq / (kde_n - 1.0))
	kde_h := 1.06 * kde_sigma * math.pow(kde_n, -0.2)
	kde_lo := kde_data[0] - 2.0 * kde_h
	kde_hi := kde_data[kde_data.len - 1] + 2.0 * kde_h
	n_kde := 60
	mut kde_xs := []f64{len: n_kde}
	mut kde_ys := []f64{len: n_kde}
	for i in 0 .. n_kde {
		gv := kde_lo + f64(i) * (kde_hi - kde_lo) / f64(n_kde - 1)
		kde_xs[i] = gv
		mut d := 0.0
		for x in kde_data {
			u := (gv - x) / kde_h
			d += math.exp(-0.5 * u * u)
		}
		kde_ys[i] = d / (kde_n * kde_h * math.sqrt(2.0 * math.pi))
	}
	// scale KDE to match histogram counts: multiply by n * bin_width
	bins := chart.histogram_bins(kde_data, 0)
	bin_w := bins.edges[1] - bins.edges[0]
	for i in 0 .. n_kde {
		kde_ys[i] *= kde_n * bin_w
	}
	chart.new(title: 'KDE Density Overlay', subtitle: 'Histogram with Gaussian kernel density',
		width: 640, height: 420, theme: chart.Theme{ grid: true })
		.histogram(kde_data, label: 'data')
		.line(kde_xs, kde_ys, label: 'KDE')
		.xlabel('Value')
		.ylabel('Count')
		.save(os.join_path(out, 'kde_density.svg'))!

	println('done — wrote 12 SVGs to ${out}')
```

Also remove the `_ = math.pi` placeholder line.

- [ ] **Step 2: Run full gallery**

```bash
v run examples/statistical-gallery/
```

Expected output:
```
done — wrote 12 SVGs to .../examples/statistical-gallery
```

```bash
ls examples/statistical-gallery/*.svg | wc -l
```

Expected: `12`

- [ ] **Step 3: Verify all 12 files exist**

```bash
ls examples/statistical-gallery/*.svg
```

Expected: `box_comparison.svg`, `dot_ranking.svg`, `forest_plot.svg`, `hbar_comparison.svg`, `heatmap_correlation.svg`, `kde_density.svg`, `line_ci.svg`, `qq_normal.svg`, `stacked_bar.svg`, `step_cdf.svg`, `step_survival.svg`, `violin_distribution.svg`

- [ ] **Step 4: Commit**

```bash
git add examples/statistical-gallery/main.v
git commit -m "feat(examples): complete statistical gallery with 12 SVGs"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `step` series kind | Task 1 |
| `box` series kind | Task 2 |
| `dot` series kind | Task 3 |
| `violin` series kind + Silverman KDE | Task 4 |
| `hbar` series kind | Task 5 |
| `heatmap` series kind + colour interpolation | Task 6 |
| `stacked_bar` series kind | Task 7 |
| `line_ci.svg` — line + 95% CI band | Task 8 |
| `box_comparison.svg` | Task 2 |
| `violin_distribution.svg` | Task 4 |
| `step_cdf.svg` | Task 1 |
| `step_survival.svg` | Task 1 |
| `dot_ranking.svg` | Task 3 |
| `hbar_comparison.svg` | Task 5 |
| `stacked_bar.svg` | Task 7 |
| `heatmap_correlation.svg` | Task 6 |
| `forest_plot.svg` | Task 8 |
| `qq_normal.svg` | Task 8 |
| `kde_density.svg` | Task 8 |
| Categorical y-axis (dot, hbar, heatmap) | Task 3 |
| Categorical x-axis (stacked_bar, heatmap) | Task 3 |
| `lerp_color` + `hex_to_rgb` | Task 6 |
| `HeatmapOpts` struct | Task 6 |
| Synthetic inline data — no external deps | All tasks |

All spec requirements covered. No TBDs or placeholders.

**Type consistency:**
- `box_plot` enum value used consistently (`box_plot` not `box` — avoids V keyword)
- `.box()` public method name is correct (V allows methods named `box`)
- `silverman_kde` signature `(data []f64, n_grid int) ([]f64, []f64)` used consistently
- `lerp_color(lo string, hi string, t f64) string` used consistently
- `Series.nbins` reused for: histogram nbins, heatmap ncols, stacked_bar nsegments
- `Series.color_lo` and `Series.color_hi` added in Task 2 (Step 3b) for heatmap use in Task 6
