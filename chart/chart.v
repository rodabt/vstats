module chart

import math

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
