import chart

fn test__line_chart_builds_axes_and_polyline() {
	svg := chart.new(title: 'Demo', width: 400, height: 300)
		.line([0.0, 1.0, 2.0], [0.0, 10.0, 5.0], label: 'a')
		.render()
	assert svg.starts_with('<svg')
	assert svg.contains('<polyline')
	// two axis lines (x and y) -> at least two <line> elements
	assert svg.count('<line') >= 2
}

fn test__line_equal_lengths_render() {
	out := chart.new().line([0.0, 1.0], [2.0, 3.0]).render()
	assert out.contains('<polyline')
}

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
