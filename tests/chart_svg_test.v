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
