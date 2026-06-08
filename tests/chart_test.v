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

fn test__ticks_and_labels_render() {
	svg := chart.new(title: 'T', width: 400, height: 300)
		.line([0.0, 1.0, 2.0], [0.0, 50.0, 100.0])
		.xlabel('time')
		.ylabel('value')
		.render()
	assert svg.contains('>T<') // title text
	assert svg.contains('>time<') // x label
	assert svg.contains('>value<') // y label
	assert svg.contains('transform="rotate(') // y label is rotated
	assert svg.contains('>100<') // a y tick label
}

fn test__legend_appears_with_two_labeled_series() {
	svg := chart.new(width: 400, height: 300)
		.line([0.0, 1.0], [0.0, 1.0], label: 'a')
		.line([0.0, 1.0], [1.0, 0.0], label: 'b')
		.render()
	assert svg.contains('>a<')
	assert svg.contains('>b<')
}

fn test__axhline_adds_guide_line() {
	base := chart.new(width: 400, height: 300).scatter([0.0, 1.0], [-1.0, 1.0])
	without := base.render().count('<line')
	with := base.axhline(0.0).render().count('<line')
	assert with == without + 1
}
