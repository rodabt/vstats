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

fn test__bar_tooltip_has_series_and_label() {
	svg := chart.new(width: 300, height: 200)
		.bar([10.0, 20.0], label: 'Revenue', labels: ['Q1', 'Q2'])
		.render()
	assert svg.contains('<title>Revenue')
	assert svg.contains('data-series="Revenue"')
	assert svg.contains('Q1: 10')
	assert svg.contains('Q2: 20')
}

fn test__scatter_tooltip_has_xy() {
	svg := chart.new(width: 300, height: 200)
		.scatter([1.0], [8.0], label: 'A')
		.render()
	assert svg.contains('x: 1, y: 8')
	assert svg.contains('data-x="1"')
	assert svg.contains('data-y="8"')
}

fn test__histogram_tooltip_has_range_and_count() {
	svg := chart.new(width: 300, height: 200)
		.histogram([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], nbins: 3)
		.render()
	assert svg.contains('data-tooltip="[')
}

fn test__heatmap_tooltip_has_value() {
	svg := chart.new(width: 300, height: 300)
		.heatmap([[1.0, 2.0], [3.0, 4.0]], row_labels: ['r0', 'r1'], col_labels: ['c0', 'c1'])
		.render()
	assert svg.contains('r0 × c0')
}

fn test__line_emits_transparent_hover_targets() {
	svg := chart.new(width: 300, height: 200)
		.line([0.0, 1.0, 2.0], [3.0, 4.0, 5.0], label: 'L')
		.render()
	assert svg.contains('fill="transparent"')
	// one hover target per vertex, each carrying a tooltip
	assert svg.count('data-tooltip') >= 3
}

fn test__area_emits_hover_targets() {
	svg := chart.new(width: 300, height: 200)
		.area([0.0, 1.0], [2.0, 3.0], label: 'A')
		.render()
	assert svg.contains('fill="transparent"')
	assert svg.count('data-tooltip') >= 2
}
