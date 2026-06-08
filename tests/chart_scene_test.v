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
