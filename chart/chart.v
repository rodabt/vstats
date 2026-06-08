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
	c.draw_axes(mut scene, g)
	c.draw_ticks(mut scene, g)
	c.draw_guides(mut scene, g)
	c.draw_series(mut scene, g)
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
