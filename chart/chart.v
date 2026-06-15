module chart

import math
import os

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

@[params]
pub struct ChartOpts {
pub:
	title    string
	subtitle string
	width    int   = 640
	height   int   = 480
	theme    Theme = Theme{}
}

@[params]
pub struct SeriesOpts {
pub:
	label       string
	color       string
	show_values bool
	labels      []string
	err         []f64
}

pub fn new(opts ChartOpts) Chart {
	return Chart{
		title:    opts.title
		subtitle: opts.subtitle
		width:    opts.width
		height:   opts.height
		theme:    opts.theme
	}
}

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

@[params]
pub struct HistogramOpts {
pub:
	label string
	nbins int // 0 => auto (Sturges)
	color string
}

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
	s.sort(a < b)
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
		} else {
			wlo = v
			break
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
		.step {
			x0, x1 := extent(s.x)
			y0, y1 := extent(s.y)
			x0, x1, y0, y1
		}
		.box_plot {
			cx := s.x[0]
			cx - 0.5, cx + 0.5, s.lo[0], s.hi[0]
		}
		.dot {
			_, xmax := extent(s.y)
			0.0, xmax, -0.5, f64(s.y.len) - 0.5
		}
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
					opacity: math.min(t.fill_opacity * 2.5, 1.0)
					stroke:  s.color
					width:   t.axis_width
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
				scene.primitives << Rect{
					x:      cx - bw / 2.0
					y:      q3_px
					w:      bw
					h:      q1_px - q3_px
					fill:   s.color
					stroke: t.axis_color
					width:  t.axis_width
				}
				scene.primitives << Line{
					x1:     cx - bw / 2.0
					y1:     med_px
					x2:     cx + bw / 2.0
					y2:     med_px
					stroke: t.axis_color
					width:  t.axis_width * 2.0
				}
				scene.primitives << Line{ x1: cx, y1: q3_px, x2: cx, y2: whi_px, stroke: t.axis_color, width: t.axis_width }
				scene.primitives << Line{ x1: cx, y1: q1_px, x2: cx, y2: wlo_px, stroke: t.axis_color, width: t.axis_width }
				scene.primitives << Line{ x1: cx - capw, y1: whi_px, x2: cx + capw, y2: whi_px, stroke: t.axis_color, width: t.axis_width }
				scene.primitives << Line{ x1: cx - capw, y1: wlo_px, x2: cx + capw, y2: wlo_px, stroke: t.axis_color, width: t.axis_width }
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
			.dot {
				x0 := g.xscale.map(0.0)
				for i in 0 .. s.y.len {
					py := g.yscale.map(f64(s.y.len - 1 - i))
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
			.band, .area, .violin {}
			.hbar, .heatmap, .stacked_bar {}
		}
	}
}

fn (c Chart) draw_ticks(mut scene Scene, g Geom) {
	t := c.theme
	bottom := g.plot_y + g.plot_h

	// detect categorical axes
	mut y_cat_labels := []string{}
	mut x_cat_labels := []string{}
	for s in c.series {
		if s.kind in [.dot, .hbar] && s.labels.len > 0 && y_cat_labels.len == 0 {
			y_cat_labels = s.labels.clone()
		}
		if s.kind == .stacked_bar && s.labels.len > 0 && x_cat_labels.len == 0 {
			x_cat_labels = s.labels.clone()
		}
		if s.kind == .heatmap && s.labels.len > 0 {
			ncols := s.nbins
			if ncols > 0 && s.labels.len > ncols {
				x_cat_labels = s.labels[0..ncols].clone()
				y_cat_labels = s.labels[ncols..].clone()
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
			// row 0 = top: reverse so label[0] appears at the top of the plot
			y_pos := f64(y_cat_labels.len - 1 - i)
			py := g.yscale.map(y_pos)
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

fn (c Chart) value_text(s Series, i int, v f64) string {
	if s.labels.len > i {
		return s.labels[i]
	}
	return fmt_tick(v)
}

fn (c Chart) draw_error_bars(mut scene Scene, g Geom) {
	t := c.theme
	capw := 4.0
	for s in c.series {
		if s.err.len == 0 {
			continue
		}
		if s.kind in [.box_plot, .dot, .violin] {
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

fn (c Chart) build_scene() Scene {
	g := c.geometry()
	mut scene := Scene{}
	c.draw_grid(mut scene, g)
	c.draw_axes(mut scene, g)
	c.draw_ticks(mut scene, g)
	c.draw_guides(mut scene, g)
	c.draw_series(mut scene, g)
	c.draw_error_bars(mut scene, g)
	c.draw_value_labels(mut scene, g)
	c.draw_labels(mut scene, g)
	c.draw_legend(mut scene, g)
	return scene
}

pub fn (c Chart) render() string {
	scene := c.build_scene()
	return render_svg(scene, c.width, c.height, c.theme)
}

pub fn (c Chart) save(path string) ! {
	os.write_file(path, c.render())!
}
