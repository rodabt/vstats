import chart

fn test__rect_with_meta_emits_title_and_data() {
	scene := chart.Scene{
		primitives: [
			chart.Primitive(chart.Rect{
				x:      1.0
				y:      2.0
				w:      3.0
				h:      4.0
				fill:   'red'
				stroke: 'none'
				width:  0.0
				meta:   chart.Meta{
					tooltip: 'Revenue\nQ3: 42'
					series:  'Revenue'
					label:   'Q3'
					y:       '42'
				}
			}),
		]
	}
	svg := chart.render_svg(scene, 100, 100, chart.Theme{})
	assert svg.contains('<title>Revenue')
	assert svg.contains('data-series="Revenue"')
	assert svg.contains('data-label="Q3"')
	assert svg.contains('data-y="42"')
	assert svg.contains('data-tooltip="Revenue&#10;Q3: 42"')
	assert !svg.contains('data-x=') // empty field omitted
}

fn test__rect_without_meta_is_self_closing() {
	scene := chart.Scene{
		primitives: [
			chart.Primitive(chart.Rect{
				x:      0.0
				y:      0.0
				w:      5.0
				h:      5.0
				fill:   'blue'
				stroke: 'none'
				width:  0.0
			}),
		]
	}
	svg := chart.render_svg(scene, 100, 100, chart.Theme{})
	assert svg.contains('<rect x="0.0" y="0.0" width="5.0" height="5.0" fill="blue" stroke="none" stroke-width="0.0"/>')
	assert !svg.contains('<title>')
}

fn test__xml_escape_handles_quote_and_angle() {
	scene := chart.Scene{
		primitives: [
			chart.Primitive(chart.Circle{
				cx:     1.0
				cy:     1.0
				r:      2.0
				fill:   'green'
				stroke: 'none'
				width:  0.0
				meta:   chart.Meta{ tooltip: 'a<b & "c"' }
			}),
		]
	}
	svg := chart.render_svg(scene, 50, 50, chart.Theme{})
	assert svg.contains('a&lt;b &amp; &quot;c&quot;')
}
